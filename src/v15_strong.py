"""
v15: train on (comp_train + orig 101k clean rows) combined.
CV folds on comp rows only (correct OOF on competition labels).
This is the 'ADD_EXTERN_DATA=True' trick.
"""
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
OUT_NAME = f"v15_strong_s{SEED}"
t0 = time.time()
print(f"=== {OUT_NAME} (seed={SEED}) ===", flush=True)

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

# Match orig for NTL on comp data
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_subset = orig.select(key_cols + ["Normalized_TyreLife"]).unique(subset=key_cols)
train_match = train.join(orig_subset, on=key_cols, how="left")
test_match = test.join(orig_subset, on=key_cols, how="left")

# Reconstruct NTL via combined train+test
combined = pl.concat([
    train.drop("PitNextLap").with_columns(pl.lit(1).alias("_is_train")),
    test.with_columns(pl.lit(0).alias("_is_train")),
], how="vertical").sort(["Race", "Year", "Driver", "LapNumber"])
combined = combined.with_columns([
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
])
combined = combined.with_columns([
    (pl.col("TyreLife") / pl.col("Stint_MaxTL")).alias("NTL_reconstructed"),
])
ntl = combined.select(["id", "NTL_reconstructed", "Stint_MaxTL", "Sess_LapMax"])
train_match = train_match.join(ntl, on="id", how="left").with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])
test_match = test_match.join(ntl, on="id", how="left").with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])

# Prepare orig as additional training data (with PNL labels and true NTL)
# For orig, recompute NTL via TyreLife/max(TyreLife) per (Driver, Race, Year, Stint) using DRIVER (real)
orig_with_ntl = orig.with_columns([
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
])
orig_with_ntl = orig_with_ntl.with_columns([
    pl.col("Normalized_TyreLife").alias("NTL_combined"),
    pl.lit(1).cast(pl.Int8).alias("NTL_matched"),
    pl.col("Normalized_TyreLife").alias("NTL_reconstructed"),
])

# Convert all to pandas
train_pd = train_match.sort("id").to_pandas()
test_pd = test_match.sort("id").to_pandas()
orig_pd = orig_with_ntl.to_pandas()
y = train_pd["PitNextLap"].astype(int).values
ids_test = test_pd["id"].values

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
orig_pd = fe(orig_pd)

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

X = train_pd[features].copy()
X_test = test_pd[features].copy()
X_orig = orig_pd[features].copy()
y_orig = orig_pd["PitNextLap"].astype(int).values

# Align categories using union of train + orig
for c in cat_cols:
    cats = pd.concat([X[c].astype(str), X_orig[c].astype(str), X_test[c].astype(str)], ignore_index=True).astype(str).unique()
    X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    X_orig[c] = pd.Categorical(X_orig[c].astype(str), categories=cats)
    X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats)

print(f"comp train: {X.shape}, orig: {X_orig.shape}, test: {X_test.shape}", flush=True)

# CV: stratify on comp rows only. Each fold trains on (comp_tr + orig) and validates on (comp_va).
strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.025, "num_leaves": 191,
    "min_child_samples": 50, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": 3, "seed": SEED,
}
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key)):
    print(f"  fold {fold+1}/{N_SPLITS} ({time.time()-t0:.0f}s)", flush=True)
    # Combine comp_tr with orig
    X_tr = pd.concat([X.iloc[tr_idx], X_orig], ignore_index=True)
    y_tr = np.concatenate([y[tr_idx], y_orig])
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=8000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
    print(f"    fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}", flush=True)

cv_auc = roc_auc_score(y, oof)
print(f"\n=== {OUT_NAME} OOF AUC: {cv_auc:.6f} === ({time.time()-t0:.0f}s)", flush=True)

sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv(f"submissions/{OUT_NAME}.csv", index=False)
np.save(f"submissions/{OUT_NAME}_oof.npy", oof)
np.save(f"submissions/{OUT_NAME}_test.npy", test_pred)
print(f"saved")
