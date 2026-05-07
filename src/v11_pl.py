"""
v11 with pseudo-labels: use current best blend predictions on test as additional training data.
Test rows with high-confidence predictions (>0.95 or <0.05) are added to train with the predicted label.
"""
import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

t0 = time.time()
SEED = 3407
OUT_NAME = f"v11_pl_s{SEED}"
print(f"=== {OUT_NAME} ===", flush=True)

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

# Best blend test preds for pseudo-labels
best_blend = pd.read_csv("submissions/best_v11ens_v10ens.csv").sort_values("id").reset_index(drop=True)
test_probs = best_blend["PitNextLap"].values

# Pseudo-label: keep only high-confidence
HIGH = 0.92
LOW = 0.05
mask_pos = test_probs > HIGH
mask_neg = test_probs < LOW
print(f"pseudo-label: {mask_pos.sum()} positive (prob>{HIGH}), {mask_neg.sum()} negative (prob<{LOW})", flush=True)

# Match orig for NTL
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_subset = orig.select(key_cols + ["Normalized_TyreLife"]).unique(subset=key_cols)
train_match = train.join(orig_subset, on=key_cols, how="left")
test_match = test.join(orig_subset, on=key_cols, how="left")

combined = pl.concat([
    train.drop("PitNextLap").with_columns(pl.lit(1).alias("_is_train")),
    test.with_columns(pl.lit(0).alias("_is_train")),
], how="vertical").sort(["Race", "Year", "Driver", "LapNumber"])
combined = combined.with_columns([
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
])
combined = combined.with_columns([(pl.col("TyreLife") / pl.col("Stint_MaxTL")).alias("NTL_reconstructed")])
ntl = combined.select(["id", "NTL_reconstructed", "Stint_MaxTL", "Sess_LapMax"])
train_match = train_match.join(ntl, on="id", how="left").with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])
test_match = test_match.join(ntl, on="id", how="left").with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])

train_pd = train_match.sort("id").to_pandas()
test_pd = test_match.sort("id").to_pandas()

def fe(df):
    out = df.copy().rename(columns={"LapTime (s)": "LapTime_s"})
    eps = 1e-3
    out["EstimatedTotalLaps"] = out["LapNumber"] / out["RaceProgress"].clip(lower=eps)
    out["LapsRemaining"] = out["EstimatedTotalLaps"] - out["LapNumber"]
    out["TyreAgeRatio"] = out["TyreLife"] / out["LapNumber"].clip(lower=1)
    out["TyreAgeVsRace"] = out["TyreLife"] / out["EstimatedTotalLaps"].clip(lower=1)
    out["DegPerTyreLap"] = out["Cumulative_Degradation"] / out["TyreLife"].clip(lower=1)
    out["DegPerRaceLap"] = out["Cumulative_Degradation"] / out["LapNumber"].clip(lower=1)
    out["DeltaPerTyreLap"] = out["LapTime_Delta"] / out["TyreLife"].clip(lower=1)
    out["DeltaAbs"] = out["LapTime_Delta"].abs()
    out["PositionPressure"] = out["Position"] * out["RaceProgress"]
    out["StintPressure"] = out["Stint"] * out["TyreLife"]
    out["PitWindowPressure"] = out["TyreLife"] * out["RaceProgress"]
    out["LapMinusTyreLife"] = out["LapNumber"] - out["TyreLife"]
    return out

train_pd = fe(train_pd)
test_pd = fe(test_pd)
y = train_pd["PitNextLap"].astype(int).values
ids_test = test_pd["id"].values

# Pseudo-label rows from test
pl_mask = mask_pos | mask_neg
pl_y = np.where(test_probs > 0.5, 1, 0)
pl_pseudo = test_pd[pl_mask].copy()
pl_y_sub = pl_y[pl_mask]
print(f"adding {len(pl_pseudo)} pseudo-labeled rows from test (y mean: {pl_y_sub.mean():.3f})", flush=True)

cat_cols = ["Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime_s", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
    "EstimatedTotalLaps", "LapsRemaining", "TyreAgeRatio", "TyreAgeVsRace",
    "DegPerTyreLap", "DegPerRaceLap", "DeltaPerTyreLap", "DeltaAbs",
    "PositionPressure", "StintPressure", "PitWindowPressure", "LapMinusTyreLife",
    "NTL_combined", "NTL_matched", "NTL_reconstructed",
    "Stint_MaxTL", "Sess_LapMax",
]
features = cat_cols + num_cols

X_full = train_pd[features].copy()
X_pl = pl_pseudo[features].copy()
X_test = test_pd[features].copy()
for c in cat_cols:
    X_full[c] = X_full[c].astype(str).astype("category")
    cats = X_full[c].cat.categories
    X_pl[c] = X_pl[c].astype(str).astype("category").cat.set_categories(cats)
    X_test[c] = X_test[c].astype(str).astype("category").cat.set_categories(cats)

# CV: only on REAL train rows. Pseudo rows added to TRAIN side of every fold (with lower weight).
strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X_full))
test_pred = np.zeros(len(X_test))
PL_WEIGHT = 0.3  # lower weight for pseudo-labels

params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.04, "num_leaves": 127,
    "min_child_samples": 50, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": 3, "seed": SEED,
}
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, strat_key)):
    print(f"  fold {fold+1}/{N_SPLITS}", flush=True)
    X_tr_real = X_full.iloc[tr_idx]
    y_tr_real = y[tr_idx]
    # add pseudo
    X_tr = pd.concat([X_tr_real, X_pl], ignore_index=True)
    y_tr = np.concatenate([y_tr_real, pl_y_sub])
    weights = np.concatenate([np.ones(len(y_tr_real)), np.full(len(pl_y_sub), PL_WEIGHT)])

    dtr = lgb.Dataset(X_tr, y_tr, weight=weights, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_full.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    oof[va_idx] = m.predict(X_full.iloc[va_idx], num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS

cv_auc = roc_auc_score(y, oof)
print(f"\n=== {OUT_NAME} OOF AUC: {cv_auc:.6f} === ({time.time()-t0:.0f}s)", flush=True)
sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv(f"submissions/{OUT_NAME}.csv", index=False)
np.save(f"submissions/{OUT_NAME}_oof.npy", oof)
np.save(f"submissions/{OUT_NAME}_test.npy", test_pred)
print(f"saved")
