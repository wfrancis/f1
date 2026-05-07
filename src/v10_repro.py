"""
Reproduce v10: multi-resolution orig-aggregate features + teacher with imputed NTL.

Per agent description:
- 5 join-keys at increasing granularity:
  1. (R, Y, L) — Race, Year, LapNumber
  2. (R, Y, L, Stint)
  3. (R, Y, L, Stint, Compound)
  4. (R, Y, L, Stint, Compound, TyreLife)
  5. (R, Y, L, Stint, Compound, TyreLife, Position)
- For each, compute orig PNL/NTL/LapsLeft mean and count
- Coalesce from finest to coarsest (use finer if available, else coarser)
- Train orig LGBM teacher (with Normalized_TyreLife, LapsLeft, IsLastStint)
- Apply teacher to imputed NTL/LapsLeft features
- Final student: LGBM (no Driver) + orig aggregates + teacher_pred
"""
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3407
OUT_NAME = f"v10_repro_s{SEED}"
t0 = time.time()
print(f"=== {OUT_NAME} ===", flush=True)

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

# Compute LapsLeft and IsLastStint in orig
orig = orig.with_columns([
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Race_LapMax"),
    pl.col("Stint").max().over(["Driver", "Race", "Year"]).alias("Race_StintMax"),
])
orig = orig.with_columns([
    (pl.col("Race_LapMax") - pl.col("LapNumber")).alias("LapsLeft"),
    (pl.col("Stint") == pl.col("Race_StintMax")).cast(pl.Int8).alias("IsLastStint"),
])

# === Multi-resolution orig aggregates ===
# Compute mean PNL, mean NTL, count at each granularity
print(f"computing multi-resolution aggregates ({time.time()-t0:.0f}s)", flush=True)
agg_cols = ["Race", "Year", "LapNumber", "Stint", "Compound", "TyreLife", "Position"]
granularities = []
for k in range(3, 8):
    keys = agg_cols[:k]
    name = "_".join([c[0] for c in keys])  # short label
    a = orig.group_by(keys).agg([
        pl.col("PitNextLap").mean().alias(f"PNL_{name}"),
        pl.col("Normalized_TyreLife").mean().alias(f"NTL_{name}"),
        pl.col("LapsLeft").mean().alias(f"LL_{name}"),
        pl.col("IsLastStint").mean().alias(f"ILS_{name}"),
        pl.len().alias(f"cnt_{name}"),
    ])
    granularities.append((keys, a))
    print(f"  granularity {keys[-1]}: {a.shape[0]} groups", flush=True)

# Join all to train and test
def add_agg_features(df):
    out = df
    for keys, a in granularities:
        out = out.join(a, on=keys, how="left")
    return out

train_fe = add_agg_features(train)
test_fe = add_agg_features(test)

# Coalesce from finest to coarsest (PNL, NTL, LL, ILS)
def coalesce_features(df):
    for prefix in ["PNL", "NTL", "LL", "ILS"]:
        cols = [f"{prefix}_{'_'.join([c[0] for c in keys])}" for keys, _ in granularities]
        # finest is last (most cols in keys), coarsest is first
        cols_reversed = list(reversed(cols))  # finest first
        df = df.with_columns(pl.coalesce(cols_reversed).alias(f"{prefix}_coal"))
    return df

train_fe = coalesce_features(train_fe)
test_fe = coalesce_features(test_fe)

# Match to orig for direct NTL (where exact match)
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_match_cols = orig.select(key_cols + ["Normalized_TyreLife", "LapsLeft", "IsLastStint"]).unique(subset=key_cols)
train_fe = train_fe.join(orig_match_cols, on=key_cols, how="left", suffix="_match")
test_fe = test_fe.join(orig_match_cols, on=key_cols, how="left", suffix="_match")

# Reconstructed NTL (for unmatched)
combined = pl.concat([
    train.drop("PitNextLap").with_columns(pl.lit(1).alias("_is_train")),
    test.with_columns(pl.lit(0).alias("_is_train")),
], how="vertical").sort(["Race", "Year", "Driver", "LapNumber"])
combined = combined.with_columns([
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
])
combined = combined.with_columns([(pl.col("TyreLife") / pl.col("Stint_MaxTL")).alias("NTL_reconstructed")])
ntl_recon = combined.select(["id", "NTL_reconstructed", "Stint_MaxTL", "Sess_LapMax"])
train_fe = train_fe.join(ntl_recon, on="id", how="left")
test_fe = test_fe.join(ntl_recon, on="id", how="left")

# Final NTL = matched > coalesced > reconstructed
train_fe = train_fe.with_columns(
    pl.coalesce(["Normalized_TyreLife", "NTL_coal", "NTL_reconstructed"]).alias("NTL_final")
)
test_fe = test_fe.with_columns(
    pl.coalesce(["Normalized_TyreLife", "NTL_coal", "NTL_reconstructed"]).alias("NTL_final")
)

# Convert to pandas
train_pd = train_fe.sort("id").to_pandas()
test_pd = test_fe.sort("id").to_pandas()
y = train_pd["PitNextLap"].astype(int).values
ids_test = test_pd["id"].values

# === Teacher trained on orig with NTL + LapsLeft + IsLastStint ===
print(f"training teacher ({time.time()-t0:.0f}s)", flush=True)
orig_pd = orig.to_pandas()
teach_cat = ["Compound", "Race"]
teach_num = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
             "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
             "RaceProgress", "Position_Change",
             "Normalized_TyreLife", "LapsLeft", "IsLastStint"]
teach_features = teach_cat + teach_num
orig_X = orig_pd[teach_features].copy()
orig_y = orig_pd["PitNextLap"].astype(int).values
for c in teach_cat:
    orig_X[c] = orig_X[c].astype("category")

teacher_train = np.zeros(len(train_pd))
teacher_test = np.zeros(len(test_pd))
for tseed in [42, 123]:
    dall = lgb.Dataset(orig_X, orig_y, categorical_feature=teach_cat)
    teacher = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 63, "min_child_samples": 30, "verbose": -1, "n_jobs": 3, "seed": tseed},
        dall, num_boost_round=600,
    )
    def teacher_input(df):
        return pd.DataFrame({
            "Compound": df["Compound"].values, "Race": df["Race"].values,
            "Year": df["Year"].values, "PitStop": df["PitStop"].values,
            "LapNumber": df["LapNumber"].values, "Stint": df["Stint"].values,
            "TyreLife": df["TyreLife"].values, "Position": df["Position"].values,
            "LapTime (s)": df["LapTime (s)"].values, "LapTime_Delta": df["LapTime_Delta"].values,
            "Cumulative_Degradation": df["Cumulative_Degradation"].values,
            "RaceProgress": df["RaceProgress"].values, "Position_Change": df["Position_Change"].values,
            "Normalized_TyreLife": df["NTL_final"].values,
            "LapsLeft": df.get("LL_coal", df["Sess_LapMax"] - df["LapNumber"]).values,
            "IsLastStint": df.get("ILS_coal", pd.Series([0.0] * len(df))).fillna(0).values,
        })
    tr_in = teacher_input(train_pd)
    te_in = teacher_input(test_pd)
    for c in teach_cat:
        tr_in[c] = tr_in[c].astype("category").cat.set_categories(orig_X[c].cat.categories)
        te_in[c] = te_in[c].astype("category").cat.set_categories(orig_X[c].cat.categories)
    teacher_train += teacher.predict(tr_in) / 2
    teacher_test += teacher.predict(te_in) / 2
train_pd["teacher_pred"] = teacher_train
test_pd["teacher_pred"] = teacher_test

# === Final student (NO Driver) ===
print(f"training student ({time.time()-t0:.0f}s)", flush=True)
cat_cols = ["Compound", "Race"]
num_features = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
    "NTL_final", "NTL_reconstructed", "Stint_MaxTL", "Sess_LapMax",
    "teacher_pred",
    # multi-res orig aggregates (coalesced and per-granularity)
    "PNL_coal", "NTL_coal", "LL_coal", "ILS_coal",
]
# also add per-granularity counts
for keys, _ in granularities:
    name = "_".join([c[0] for c in keys])
    if f"cnt_{name}" in train_pd.columns:
        num_features.append(f"cnt_{name}")
    if f"PNL_{name}" in train_pd.columns:
        num_features.append(f"PNL_{name}")

features = cat_cols + num_features
print(f"feature count: {len(features)}", flush=True)

# Drop columns that don't exist
features = [f for f in features if f in train_pd.columns]
X = train_pd[features].copy()
X_test = test_pd[features].copy()
for c in cat_cols:
    X[c] = X[c].astype(str).astype("category")
    X_test[c] = X_test[c].astype(str).astype("category")
    X_test[c] = X_test[c].cat.set_categories(X[c].cat.categories)

strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.04, "num_leaves": 127,
    "min_child_samples": 50, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": 3, "seed": SEED,
}
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key)):
    print(f"  fold {fold+1}/{N_SPLITS} ({time.time()-t0:.0f}s)", flush=True)
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS

cv_auc = roc_auc_score(y, oof)
print(f"\n=== {OUT_NAME} OOF AUC: {cv_auc:.6f} === ({time.time()-t0:.0f}s)", flush=True)
sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv(f"submissions/{OUT_NAME}.csv", index=False)
np.save(f"submissions/{OUT_NAME}_oof.npy", oof)
np.save(f"submissions/{OUT_NAME}_test.npy", test_pred)
print(f"saved submissions/{OUT_NAME}.*")
