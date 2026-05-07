"""
Baseline: LightGBM with stratified k-fold, minimal feature engineering.
Goal: get a submittable model and reference AUC fast.
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

# --- session-level feature: race length (max LapNumber per Race+Year)
all_df = pl.concat([train.drop("PitNextLap"), test], how="vertical")
race_len = all_df.group_by(["Race", "Year"]).agg(
    pl.col("LapNumber").max().alias("RaceLapMax")
)

def add_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.join(race_len, on=["Race", "Year"], how="left")
    df = df.with_columns([
        (pl.col("RaceLapMax") - pl.col("LapNumber")).alias("LapsRemaining"),
        (pl.col("LapNumber") / pl.col("RaceLapMax")).alias("RaceProgressRecomputed"),
        # interaction: tyre life relative to typical for compound
        pl.col("TyreLife").alias("TyreLife_raw"),
    ])
    return df

train = add_features(train)
test = add_features(test)

# --- categorical encoding: cast to pandas categorical for LightGBM
cat_cols = ["Driver", "Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change", "RaceLapMax", "LapsRemaining", "RaceProgressRecomputed",
]
features = cat_cols + num_cols

X = train.select(features).to_pandas()
y = train["PitNextLap"].to_numpy().astype(int)
X_test = test.select(features).to_pandas()
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")
    # align categories so test sees same set
    X_test[c] = X_test[c].cat.set_categories(X[c].cat.categories)

print(f"features: {len(features)}")
print(f"target balance: {y.mean():.4f}")

# --- 5-fold stratified CV
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
    print(f"\n--- fold {fold + 1}/{N_SPLITS} ---")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_SPLITS

    fold_auc = roc_auc_score(y_va, oof[va_idx])
    print(f"fold {fold + 1} AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\n=== OOF AUC: {cv_auc:.6f} ===")
print(f"elapsed: {time.time() - t0:.1f}s")

# --- write submission
sub = pl.DataFrame({"id": test["id"], "PitNextLap": test_pred})
sub.write_csv("submissions/baseline_v1.csv")
print(f"submission written: submissions/baseline_v1.csv ({sub.shape})")
print(sub.head(5))
