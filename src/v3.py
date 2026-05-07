"""
v3: Add the killer feature — reconstructed Normalized_TyreLife.
Definition: for each (Driver, Race, Year, Stint), NTL = TyreLife / max(TyreLife in stint).
This is the feature the organizers removed because it makes prediction trivial.
We reconstruct it from train+test combined.
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

t0 = time.time()

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
print(f"train: {train.shape}, test: {test.shape}")

# Combine for stint-level max(TyreLife)
combined = pl.concat([
    train.drop("PitNextLap").with_columns(pl.lit(1).alias("_is_train")),
    test.with_columns(pl.lit(0).alias("_is_train")),
], how="vertical").sort(["Race", "Year", "Driver", "LapNumber"])

# === KILLER FEATURE: reconstruct Normalized_TyreLife ===
combined = combined.with_columns([
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("TyreLife").min().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MinTL"),
    pl.col("LapNumber").count().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_RowCount"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
    pl.col("Stint").max().over(["Driver", "Race", "Year"]).alias("Sess_MaxStint"),
])

combined = combined.with_columns([
    (pl.col("TyreLife") / pl.col("Stint_MaxTL")).alias("NTL_reconstructed"),
    # Are we at the visible max? (proxy for end-of-stint)
    (pl.col("TyreLife") == pl.col("Stint_MaxTL")).cast(pl.Int8).alias("IsStintMaxTL"),
    # Distance from max
    (pl.col("Stint_MaxTL") - pl.col("TyreLife")).alias("LapsToStintEnd"),
    # Stint length (visible)
    (pl.col("Stint_MaxTL") - pl.col("Stint_MinTL") + 1).alias("Stint_LengthVisible"),
])

# Race-level features
combined = combined.with_columns([
    (pl.col("Sess_LapMax") - pl.col("LapNumber")).alias("LapsRemaining"),
    (pl.col("LapNumber") / pl.col("Sess_LapMax")).alias("RaceProgressV2"),
])

# Compound-relative TyreLife (typical max across all stints of this compound)
cmp_max = combined.group_by("Compound").agg(
    pl.col("Stint_MaxTL").mean().alias("Cmp_AvgStintMax"),
    pl.col("Stint_MaxTL").quantile(0.75).alias("Cmp_StintMaxP75"),
    pl.col("Stint_MaxTL").quantile(0.90).alias("Cmp_StintMaxP90"),
)
combined = combined.join(cmp_max, on="Compound", how="left")
combined = combined.with_columns([
    (pl.col("TyreLife") / pl.col("Cmp_AvgStintMax")).alias("TyreLife_RelCmpAvg"),
])

print(f"combined: {combined.shape}")

# Re-split
train_id_set = set(train["id"].to_list())
combined = combined.with_columns(pl.col("id").is_in(train_id_set).alias("_in_train"))
train_fe = combined.filter(pl.col("_in_train")).join(
    train.select(["id", "PitNextLap"]), on="id", how="left"
).sort("id")
test_fe = combined.filter(~pl.col("_in_train")).sort("id")
print(f"train_fe: {train_fe.shape}, test_fe: {test_fe.shape}")

# Features
cat_cols = ["Driver", "Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
    # killer feature
    "NTL_reconstructed", "IsStintMaxTL", "LapsToStintEnd", "Stint_LengthVisible",
    "Stint_MaxTL", "Stint_MinTL", "Stint_RowCount",
    # session
    "Sess_LapMax", "Sess_MaxStint", "LapsRemaining", "RaceProgressV2",
    # compound aggregates
    "Cmp_AvgStintMax", "Cmp_StintMaxP75", "Cmp_StintMaxP90", "TyreLife_RelCmpAvg",
]
features = cat_cols + num_cols
print(f"feature count: {len(features)}")

X = train_fe.select(features).to_pandas()
y = train_fe["PitNextLap"].to_numpy().astype(int)
X_test = test_fe.select(features).to_pandas()
ids_test = test_fe["id"].to_numpy()

for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")
    X_test[c] = X_test[c].cat.set_categories(X[c].cat.categories)

print(f"X: {X.shape}, y mean: {y.mean():.4f}")

# 5-fold
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- fold {fold+1}/{N_SPLITS} ---")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
    model = lgb.train(
        params, dtr, num_boost_round=4000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)],
    )
    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS
    print(f"fold {fold+1} AUC: {roc_auc_score(y_va, oof[va_idx]):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\n=== v3 OOF AUC: {cv_auc:.6f} ===")
print(f"baseline 0.943912  ->  v3 {cv_auc:.6f}  (delta {cv_auc - 0.943912:+.6f})")
print(f"elapsed: {time.time() - t0:.1f}s")

# importance
imp = sorted(zip(features, model.feature_importance(importance_type="gain")),
             key=lambda x: -x[1])
print("\n=== top 25 features by gain ===")
for f, g in imp[:25]:
    print(f"  {f:35s}  {g:>15.0f}")

sub = pl.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.write_csv("submissions/v3.csv")
print(f"\nsubmission: submissions/v3.csv ({sub.shape})")
