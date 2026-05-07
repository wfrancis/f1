"""
v6: combine public-notebook feature engineering with our orig-teacher edge.
Pure-row features (PitWindowPressure, etc.) + categorical interactions
+ orig-data teacher_pred + NTL_combined.
LightGBM only first; ensemble in v7.
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
print(f"train: {train.shape}, test: {test.shape}, orig: {orig.shape}")

# --- match to orig for NTL ---
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_subset = orig.select(key_cols + ["Normalized_TyreLife"]).unique(subset=key_cols)
train_match = train.join(orig_subset, on=key_cols, how="left")
test_match = test.join(orig_subset, on=key_cols, how="left")

# --- reconstruct NTL ---
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
ntl_recon = combined.select(["id", "NTL_reconstructed", "Stint_MaxTL", "Sess_LapMax"])
train_match = train_match.join(ntl_recon, on="id", how="left")
test_match = test_match.join(ntl_recon, on="id", how="left")
train_match = train_match.with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])
test_match = test_match.with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])

# Convert to pandas for the pipeline
train_pd = train_match.to_pandas()
test_pd = test_match.to_pandas()
y = train_pd["PitNextLap"].astype(int).values
ids_test = test_pd["id"].values

# --- feature engineering (from public notebook + ours) ---
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"LapTime (s)": "LapTime_s"})
    eps = 1e-3
    # public notebook features
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
    out["PitWindowPressure"] = out["TyreLife"] * out["RaceProgress"]  # NTL-like!
    out["LapMinusTyreLife"] = out["LapNumber"] - out["TyreLife"]

    # bins
    out["TyreLifeBin"] = pd.cut(out["TyreLife"], bins=[-np.inf, 3, 6, 10, 15, 20, 30, 40, np.inf],
                                  labels=False).fillna(-1).astype(int).astype(str)
    out["RaceProgressBin"] = pd.cut(out["RaceProgress"], bins=np.linspace(0.0, 1.0, 11),
                                       labels=False, include_lowest=True).fillna(-1).astype(int).astype(str)

    # categorical interactions
    out["Year_str"] = out["Year"].astype(str)
    out["Driver_Compound"] = out["Driver"].astype(str) + "__" + out["Compound"].astype(str)
    out["Race_Compound"] = out["Race"].astype(str) + "__" + out["Compound"].astype(str)
    out["Race_Year"] = out["Race"].astype(str) + "__" + out["Year_str"]
    out["Driver_Race"] = out["Driver"].astype(str) + "__" + out["Race"].astype(str)
    out["Driver_Year"] = out["Driver"].astype(str) + "__" + out["Year_str"]
    out["Compound_TyreLifeBin"] = out["Compound"].astype(str) + "__" + out["TyreLifeBin"]
    out["Compound_RaceProgressBin"] = out["Compound"].astype(str) + "__" + out["RaceProgressBin"]
    out["Stint_Compound"] = out["Stint"].astype(str) + "__" + out["Compound"].astype(str)
    return out

train_pd = build_features(train_pd)
test_pd = build_features(test_pd)

# --- teacher trained on orig ---
print("\n=== teacher ===")
teacher_features_cat = ["Compound", "Race"]
teacher_features_num = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
                        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                        "RaceProgress", "Position_Change", "Normalized_TyreLife"]
teacher_features = teacher_features_cat + teacher_features_num
orig_pd = orig.to_pandas()
orig_X = orig_pd[teacher_features].copy()
orig_y = orig_pd["PitNextLap"].astype(int).values
for c in teacher_features_cat:
    orig_X[c] = orig_X[c].astype("category")

# Train teacher with 2 seeds, average predictions
def teacher_inputs(df):
    return pd.DataFrame({
        "Compound": df["Compound"].values, "Race": df["Race"].values,
        "Year": df["Year"].values, "PitStop": df["PitStop"].values,
        "LapNumber": df["LapNumber"].values, "Stint": df["Stint"].values,
        "TyreLife": df["TyreLife"].values, "Position": df["Position"].values,
        "LapTime (s)": df["LapTime_s"].values, "LapTime_Delta": df["LapTime_Delta"].values,
        "Cumulative_Degradation": df["Cumulative_Degradation"].values,
        "RaceProgress": df["RaceProgress"].values, "Position_Change": df["Position_Change"].values,
        "Normalized_TyreLife": df["NTL_combined"].values,
    })

teacher_train = np.zeros(len(train_pd))
teacher_test = np.zeros(len(test_pd))
for seed in [42, 123]:
    dall = lgb.Dataset(orig_X, orig_y, categorical_feature=teacher_features_cat)
    teacher = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 63, "min_child_samples": 30, "verbose": -1, "n_jobs": -1, "seed": seed},
        dall, num_boost_round=500,
    )
    tr_in = teacher_inputs(train_pd)
    te_in = teacher_inputs(test_pd)
    for c in teacher_features_cat:
        tr_in[c] = tr_in[c].astype("category").cat.set_categories(orig_X[c].cat.categories)
        te_in[c] = te_in[c].astype("category").cat.set_categories(orig_X[c].cat.categories)
    teacher_train += teacher.predict(tr_in) / 2
    teacher_test += teacher.predict(te_in) / 2
train_pd["teacher_pred"] = teacher_train
test_pd["teacher_pred"] = teacher_test

# --- final feature set ---
exclude = {"id", "PitNextLap", "Normalized_TyreLife", "_is_train", "_in_train"}
feature_cols = [c for c in train_pd.columns if c not in exclude]
# explicit cat list (pandas 3 changed string dtype detection)
cat_cols = [
    "Driver", "Compound", "Race",
    "TyreLifeBin", "RaceProgressBin", "Year_str",
    "Driver_Compound", "Race_Compound", "Race_Year",
    "Driver_Race", "Driver_Year",
    "Compound_TyreLifeBin", "Compound_RaceProgressBin", "Stint_Compound",
]
cat_cols = [c for c in cat_cols if c in feature_cols]
num_cols = [c for c in feature_cols if c not in cat_cols]
print(f"feature count: {len(feature_cols)}, categorical: {len(cat_cols)}, numeric: {len(num_cols)}")

# Cast cat cols to category, align test
for c in cat_cols:
    train_pd[c] = train_pd[c].astype(str)
    test_pd[c] = test_pd[c].astype(str)
X = train_pd[feature_cols].copy()
X_test = test_pd[feature_cols].copy()
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")
    X_test[c] = X_test[c].cat.set_categories(X[c].cat.categories)

# --- stratified target × Year CV (per public notebook) ---
strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=3407)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.04, "num_leaves": 255,
    "min_child_samples": 30, "feature_fraction": 0.8,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1, "seed": 3407,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key)):
    print(f"\n--- fold {fold+1}/{N_SPLITS} ---")
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(
        params, dtr, num_boost_round=6000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(250), lgb.log_evaluation(500)],
    )
    oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
    print(f"fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\n=== v6 OOF AUC: {cv_auc:.6f} ===")
print(f"baseline 0.943912 -> v4 0.945071 -> v6 {cv_auc:.6f} (delta vs baseline {cv_auc - 0.943912:+.6f})")
print(f"elapsed: {time.time() - t0:.1f}s")

# importance
imp = sorted(zip(feature_cols, m.feature_importance(importance_type="gain")), key=lambda x: -x[1])
print("\n=== top 30 features ===")
for f, g in imp[:30]:
    print(f"  {f:35s}  {g:>15.0f}")

sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv("submissions/v6.csv", index=False)
print(f"\nsubmission: submissions/v6.csv ({sub.shape})")

# Save OOF and test preds for ensembling
np.save("submissions/v6_oof.npy", oof)
np.save("submissions/v6_test.npy", test_pred)
