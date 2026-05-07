"""
v2: heavy feature engineering using within-session context.
Key features:
  - Lag/lead PitStop, Stint, TyreLife, LapTime within session (combined train+test for context)
  - Has-next-lap indicator (null pattern itself is signal)
  - Race + Year + Driver session aggregates
  - Compound-specific TyreLife percentile (recovers Normalized_TyreLife)
  - Target encoding for Driver/Race/Compound (out-of-fold)
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

# ----- combine train + test for session-context features -----
combined = pl.concat([
    train.drop("PitNextLap").with_columns(pl.lit(1).alias("_is_train")),
    test.with_columns(pl.lit(0).alias("_is_train")),
], how="vertical").sort(["Race", "Year", "Driver", "LapNumber"])

# ---- session-level aggregates ----
session_agg = combined.group_by(["Race", "Year", "Driver"]).agg(
    pl.col("LapNumber").max().alias("Sess_LapMax"),
    pl.col("LapNumber").min().alias("Sess_LapMin"),
    pl.col("LapNumber").count().alias("Sess_Rows"),
    pl.col("PitStop").sum().alias("Sess_TotalPits"),
    pl.col("Stint").max().alias("Sess_MaxStint"),
)
combined = combined.join(session_agg, on=["Race", "Year", "Driver"], how="left")

# ---- within-session lag/lead features ----
# Sort within session by LapNumber then add shifted columns.
combined = combined.sort(["Race", "Year", "Driver", "LapNumber"])
group_cols = ["Race", "Year", "Driver"]

# Lag/lead over session (regardless of actual lap-number gap — uses row order)
for shift, name in [(1, "Lead1"), (2, "Lead2"), (3, "Lead3"), (-1, "Lag1"), (-2, "Lag2")]:
    combined = combined.with_columns([
        pl.col("PitStop").shift(-shift).over(group_cols).alias(f"PitStop_{name}"),
        pl.col("Stint").shift(-shift).over(group_cols).alias(f"Stint_{name}"),
        pl.col("LapNumber").shift(-shift).over(group_cols).alias(f"LapNumber_{name}"),
        pl.col("TyreLife").shift(-shift).over(group_cols).alias(f"TyreLife_{name}"),
        pl.col("LapTime (s)").shift(-shift).over(group_cols).alias(f"LapTime_{name}"),
    ])

# Lap-number deltas to neighbors (how far away is the next available lap?)
combined = combined.with_columns([
    (pl.col("LapNumber_Lead1") - pl.col("LapNumber")).alias("LapGap_Lead1"),
    (pl.col("LapNumber") - pl.col("LapNumber_Lag1")).alias("LapGap_Lag1"),
    (pl.col("Stint_Lead1") - pl.col("Stint")).alias("StintDelta_Lead1"),
    (pl.col("Stint_Lead2") - pl.col("Stint")).alias("StintDelta_Lead2"),
    (pl.col("Stint_Lead3") - pl.col("Stint")).alias("StintDelta_Lead3"),
])

# Whether next-lap is the actual race-lap (i.e. LapGap_Lead1 == 1)
combined = combined.with_columns([
    (pl.col("LapGap_Lead1") == 1).cast(pl.Int8).alias("HasExactNextLap"),
])

# TyreLife trend within stint
combined = combined.with_columns([
    (pl.col("LapTime (s)") - pl.col("LapTime_Lag1")).alias("LapTime_Delta_Lag1"),
])

# ---- compound-specific tyre life percentile (recovers Normalized_TyreLife) ----
# Compute per-compound TyreLife distribution at PitStop=1 events
pit_events = combined.filter(pl.col("PitStop") == 1).group_by("Compound").agg(
    pl.col("TyreLife").quantile(0.50).alias("Cmp_TyreLifeP50"),
    pl.col("TyreLife").quantile(0.75).alias("Cmp_TyreLifeP75"),
    pl.col("TyreLife").quantile(0.90).alias("Cmp_TyreLifeP90"),
    pl.col("TyreLife").mean().alias("Cmp_TyreLifeMean"),
)
combined = combined.join(pit_events, on="Compound", how="left")

combined = combined.with_columns([
    (pl.col("TyreLife") / pl.col("Cmp_TyreLifeP50")).alias("TyreLife_RelP50"),
    (pl.col("TyreLife") / pl.col("Cmp_TyreLifeP75")).alias("TyreLife_RelP75"),
    (pl.col("TyreLife") - pl.col("Cmp_TyreLifeMean")).alias("TyreLife_DiffMean"),
])

# Race progress
combined = combined.with_columns([
    (pl.col("Sess_LapMax") - pl.col("LapNumber")).alias("LapsRemaining"),
    (pl.col("LapNumber") / pl.col("Sess_LapMax")).alias("RaceProgressV2"),
])

# Stint length so far (TyreLife is the count, but verify)
combined = combined.with_columns([
    pl.col("LapNumber").cum_count().over(["Race", "Year", "Driver", "Stint"]).alias("Stint_LapIdx"),
])

print(f"combined with features: {combined.shape}")

# Re-split
train_id_set = set(train["id"].to_list())
combined = combined.with_columns(pl.col("id").is_in(train_id_set).alias("_in_train"))

train_fe = combined.filter(pl.col("_in_train")).join(
    train.select(["id", "PitNextLap"]), on="id", how="left"
).sort("id")

test_fe = combined.filter(~pl.col("_in_train")).sort("id")

print(f"train_fe: {train_fe.shape}, test_fe: {test_fe.shape}")

# ---- Choose features ----
cat_cols = ["Driver", "Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
    # session aggregates
    "Sess_LapMax", "Sess_LapMin", "Sess_Rows", "Sess_TotalPits", "Sess_MaxStint",
    # leads
    "PitStop_Lead1", "Stint_Lead1", "LapNumber_Lead1", "TyreLife_Lead1", "LapTime_Lead1",
    "PitStop_Lead2", "Stint_Lead2", "LapNumber_Lead2", "TyreLife_Lead2", "LapTime_Lead2",
    "PitStop_Lead3", "Stint_Lead3", "LapNumber_Lead3", "TyreLife_Lead3", "LapTime_Lead3",
    # lags
    "PitStop_Lag1", "Stint_Lag1", "LapNumber_Lag1", "TyreLife_Lag1", "LapTime_Lag1",
    "PitStop_Lag2", "Stint_Lag2", "LapNumber_Lag2", "TyreLife_Lag2", "LapTime_Lag2",
    # deltas
    "LapGap_Lead1", "LapGap_Lag1",
    "StintDelta_Lead1", "StintDelta_Lead2", "StintDelta_Lead3",
    "HasExactNextLap",
    "LapTime_Delta_Lag1",
    # compound-relative tyre life
    "Cmp_TyreLifeP50", "Cmp_TyreLifeP75", "Cmp_TyreLifeP90", "Cmp_TyreLifeMean",
    "TyreLife_RelP50", "TyreLife_RelP75", "TyreLife_DiffMean",
    # race progress
    "LapsRemaining", "RaceProgressV2",
    "Stint_LapIdx",
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

print(f"X shape: {X.shape}, y mean: {y.mean():.4f}")

# ---- 5-fold CV ----
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
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
print(f"\n=== v2 OOF AUC: {cv_auc:.6f} ===")
print(f"baseline OOF AUC was 0.943912 — delta {cv_auc - 0.943912:+.6f}")
print(f"elapsed: {time.time() - t0:.1f}s")

# ---- feature importance ----
imp = sorted(zip(features, model.feature_importance(importance_type="gain")),
             key=lambda x: -x[1])
print("\n=== top 25 features by gain (last fold) ===")
for f, g in imp[:25]:
    print(f"  {f:30s}  {g:>15.0f}")

# write submission
sub = pl.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.write_csv("submissions/v2.csv")
print(f"\nsubmission written: submissions/v2.csv ({sub.shape})")
