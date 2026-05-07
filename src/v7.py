"""
v7: lessons learned —
  - Public-notebook NUMERIC FE helps: keep PitWindowPressure, EstimatedTotalLaps, etc.
  - High-cardinality cat INTERACTIONS hurt OOF calibration: drop them.
  - Teacher_pred dominates and miscalibrates: use it as ENSEMBLE component, not feature.
Strategy: train two models —
  (a) student LGBM with basic cats + numeric FE
  (b) teacher trained on orig
  Then blend.
"""
import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

t0 = time.time()

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

# match to orig
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_subset = orig.select(key_cols + ["Normalized_TyreLife"]).unique(subset=key_cols)
train_match = train.join(orig_subset, on=key_cols, how="left")
test_match = test.join(orig_subset, on=key_cols, how="left")

# reconstructed NTL
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

train_pd = train_match.to_pandas()
test_pd = test_match.to_pandas()
y = train_pd["PitNextLap"].astype(int).values
ids_test = test_pd["id"].values

# numeric FE only (no cat interactions)
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

# Features: basic cats + numeric FE + NTL features (no teacher_pred, no high-card interactions)
cat_cols = ["Driver", "Compound", "Race"]
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
for c in cat_cols:
    X[c] = X[c].astype(str).astype("category")
    X_test[c] = X_test[c].astype(str).astype("category")
    X_test[c] = X_test[c].cat.set_categories(X[c].cat.categories)
print(f"features: {len(features)}, cat: {len(cat_cols)}")

strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=3407)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.04, "num_leaves": 127,
    "min_child_samples": 50, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1, "seed": 3407,
}
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key)):
    print(f"\n--- fold {fold+1}/{N_SPLITS} ---")
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)])
    oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
    print(f"fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\n=== v7 student OOF AUC: {cv_auc:.6f} ===")
print(f"baseline 0.943912 -> v4 0.945071 -> v7 {cv_auc:.6f} (delta {cv_auc - 0.943912:+.6f})")
print(f"elapsed: {time.time()-t0:.1f}s")

sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv("submissions/v7.csv", index=False)
np.save("submissions/v7_oof.npy", oof)
np.save("submissions/v7_test.npy", test_pred)
print(f"submission: submissions/v7.csv")

# importance
imp = sorted(zip(features, m.feature_importance(importance_type="gain")), key=lambda x: -x[1])
print("\n=== top 20 features ===")
for f, g in imp[:20]:
    print(f"  {f:35s}  {g:>15.0f}")
