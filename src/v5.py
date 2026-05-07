"""
v5: same as v4 but also train final model on competition_train + orig_data combined.
Orig data has clean Normalized_TyreLife and clean labels — should help.
Also: 2 seeds for ensemble robustness.
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
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")
print(f"train: {train.shape}, test: {test.shape}, orig: {orig.shape}")

# === Match competition rows to original ===
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_subset = orig.select(key_cols + ["Normalized_TyreLife"]).unique(subset=key_cols)
train_match = train.join(orig_subset, on=key_cols, how="left")
test_match = test.join(orig_subset, on=key_cols, how="left")

# Reconstruct NTL
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

# === Teacher trained on orig (no Driver) ===
print("\n=== teacher on orig ===")
teacher_features_cat = ["Compound", "Race"]
teacher_features_num = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
                        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                        "RaceProgress", "Position_Change", "Normalized_TyreLife"]
teacher_features = teacher_features_cat + teacher_features_num

orig_X = orig.select(teacher_features).to_pandas()
orig_y = orig["PitNextLap"].to_numpy().astype(int)
for c in teacher_features_cat:
    orig_X[c] = orig_X[c].astype("category")

# Train teacher on all orig with 2 seeds, average
teacher_preds_train = np.zeros(len(train_match))
teacher_preds_test = np.zeros(len(test_match))
for seed in [42, 123]:
    dall = lgb.Dataset(orig_X, orig_y, categorical_feature=teacher_features_cat)
    teacher = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 63, "min_child_samples": 30, "verbose": -1, "n_jobs": -1, "seed": seed},
        dall, num_boost_round=500,
    )
    def teacher_inputs(df):
        return df.select([
            pl.col("Compound"), pl.col("Race"),
            pl.col("Year"), pl.col("PitStop"), pl.col("LapNumber"), pl.col("Stint"),
            pl.col("TyreLife"), pl.col("Position"), pl.col("LapTime (s)"),
            pl.col("LapTime_Delta"), pl.col("Cumulative_Degradation"),
            pl.col("RaceProgress"), pl.col("Position_Change"),
            pl.col("NTL_combined").alias("Normalized_TyreLife"),
        ])
    tr_in = teacher_inputs(train_match).to_pandas()
    te_in = teacher_inputs(test_match).to_pandas()
    for c in teacher_features_cat:
        tr_in[c] = tr_in[c].astype("category").cat.set_categories(orig_X[c].cat.categories)
        te_in[c] = te_in[c].astype("category").cat.set_categories(orig_X[c].cat.categories)
    teacher_preds_train += teacher.predict(tr_in) / 2
    teacher_preds_test += teacher.predict(te_in) / 2

train_match = train_match.with_columns(pl.Series("teacher_pred", teacher_preds_train))
test_match = test_match.with_columns(pl.Series("teacher_pred", teacher_preds_test))

# === Final student ===
print("\n=== final student on competition data ===")
cat_cols = ["Driver", "Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
    "NTL_combined", "NTL_matched", "NTL_reconstructed",
    "Stint_MaxTL", "Sess_LapMax",
    "teacher_pred",
]
features = cat_cols + num_cols
print(f"feature count: {len(features)}")

X = train_match.select(features).to_pandas()
y = train_match["PitNextLap"].to_numpy().astype(int)
X_test = test_match.select(features).to_pandas()
ids_test = test_match["id"].to_numpy()
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")
    X_test[c] = X_test[c].cat.set_categories(X[c].cat.categories)

# Multi-seed CV
N_SPLITS = 5
SEEDS = [42, 7, 99]
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

base_params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.04, "num_leaves": 127,
    "min_child_samples": 50, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
}

for seed in SEEDS:
    print(f"\n========== seed {seed} ==========")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(X))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        params = {**base_params, "seed": seed}
        dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=cat_cols, reference=dtr)
        m = lgb.train(
            params, dtr, num_boost_round=5000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)],
        )
        seed_oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
        test_pred += m.predict(X_test, num_iteration=m.best_iteration) / (N_SPLITS * len(SEEDS))
    print(f"seed {seed} OOF AUC: {roc_auc_score(y, seed_oof):.6f}")
    oof += seed_oof / len(SEEDS)

cv_auc = roc_auc_score(y, oof)
print(f"\n=== v5 OOF AUC (3-seed avg): {cv_auc:.6f} ===")
print(f"baseline 0.943912  ->  v5 {cv_auc:.6f}  (delta {cv_auc - 0.943912:+.6f})")
print(f"elapsed: {time.time() - t0:.1f}s")

sub = pl.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.write_csv("submissions/v5.csv")
print(f"\nsubmission: submissions/v5.csv ({sub.shape})")
