"""
v4: add TRUE Normalized_TyreLife from original dataset (where exact key-match exists),
plus reconstructed NTL fallback. Plus a teacher model trained on the original.
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

# === Match competition rows to original on key_cols (no Driver, since codes differ) ===
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]
orig_subset = orig.select(key_cols + ["Normalized_TyreLife"]).unique(subset=key_cols)
print(f"orig unique on key_cols: {orig_subset.shape}")

train_match = train.join(orig_subset, on=key_cols, how="left")
test_match = test.join(orig_subset, on=key_cols, how="left")
print(f"train matched (NTL not null): {train_match['Normalized_TyreLife'].is_not_null().sum()} / {len(train)}")
print(f"test  matched (NTL not null): {test_match['Normalized_TyreLife'].is_not_null().sum()} / {len(test)}")

# Reconstruct NTL from train+test combined
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

# Bring NTL_reconstructed to train/test
ntl_recon = combined.select(["id", "NTL_reconstructed", "Stint_MaxTL", "Sess_LapMax"])
train_match = train_match.join(ntl_recon, on="id", how="left")
test_match = test_match.join(ntl_recon, on="id", how="left")

# Final: NTL_combined = NTL_true if matched, else NTL_reconstructed
train_match = train_match.with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])
test_match = test_match.with_columns([
    pl.coalesce(["Normalized_TyreLife", "NTL_reconstructed"]).alias("NTL_combined"),
    pl.col("Normalized_TyreLife").is_not_null().cast(pl.Int8).alias("NTL_matched"),
])

# === Train teacher on original dataset ===
print("\n=== training teacher on original 101k rows ===")
orig_features_cat = ["Driver", "Compound", "Race"]
orig_features_num = ["Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
                     "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
                     "RaceProgress", "Position_Change", "Normalized_TyreLife"]
orig_features = orig_features_cat + orig_features_num

# orig has different driver codes than competition; teacher won't apply directly
# Instead, train teacher WITHOUT Driver feature, then predict on competition
teacher_features_cat = ["Compound", "Race"]
teacher_features_num = orig_features_num
teacher_features = teacher_features_cat + teacher_features_num

orig_X = orig.select(teacher_features).to_pandas()
orig_y = orig["PitNextLap"].to_numpy().astype(int)
for c in teacher_features_cat:
    orig_X[c] = orig_X[c].astype("category")

# Quick teacher: 5-fold OOF on orig
N_TEACH_FOLDS = 5
skf_t = StratifiedKFold(n_splits=N_TEACH_FOLDS, shuffle=True, random_state=42)
teacher_oof = np.zeros(len(orig_X))

for fold, (tr_idx, va_idx) in enumerate(skf_t.split(orig_X, orig_y)):
    dtr = lgb.Dataset(orig_X.iloc[tr_idx], orig_y[tr_idx], categorical_feature=teacher_features_cat)
    dva = lgb.Dataset(orig_X.iloc[va_idx], orig_y[va_idx], categorical_feature=teacher_features_cat, reference=dtr)
    m = lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 63, "min_child_samples": 30, "verbose": -1, "n_jobs": -1},
        dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    teacher_oof[va_idx] = m.predict(orig_X.iloc[va_idx], num_iteration=m.best_iteration)
print(f"teacher OOF AUC on orig: {roc_auc_score(orig_y, teacher_oof):.6f}")

# Train final teacher on all original data
dall = lgb.Dataset(orig_X, orig_y, categorical_feature=teacher_features_cat)
teacher = lgb.train(
    {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
     "num_leaves": 63, "min_child_samples": 30, "verbose": -1, "n_jobs": -1},
    dall, num_boost_round=500,
)

# Apply teacher to competition data — needs NTL_combined (which we have)
def make_teacher_input(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([
        pl.col("Compound"),
        pl.col("Race"),
        pl.col("Year"),
        pl.col("PitStop"),
        pl.col("LapNumber"),
        pl.col("Stint"),
        pl.col("TyreLife"),
        pl.col("Position"),
        pl.col("LapTime (s)"),
        pl.col("LapTime_Delta"),
        pl.col("Cumulative_Degradation"),
        pl.col("RaceProgress"),
        pl.col("Position_Change"),
        pl.col("NTL_combined").alias("Normalized_TyreLife"),
    ])

train_teach_in = make_teacher_input(train_match).to_pandas()
test_teach_in = make_teacher_input(test_match).to_pandas()
for c in teacher_features_cat:
    train_teach_in[c] = train_teach_in[c].astype("category")
    test_teach_in[c] = test_teach_in[c].astype("category")
    train_teach_in[c] = train_teach_in[c].cat.set_categories(orig_X[c].cat.categories)
    test_teach_in[c] = test_teach_in[c].cat.set_categories(orig_X[c].cat.categories)

train_teacher_pred = teacher.predict(train_teach_in)
test_teacher_pred = teacher.predict(test_teach_in)
train_match = train_match.with_columns(pl.Series("teacher_pred", train_teacher_pred))
test_match = test_match.with_columns(pl.Series("teacher_pred", test_teacher_pred))
print(f"teacher_pred train mean: {train_teacher_pred.mean():.4f}, test mean: {test_teacher_pred.mean():.4f}")

# === Final student model ===
print("\n=== training final student on competition data ===")
cat_cols = ["Driver", "Compound", "Race"]
num_cols = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Position_Change",
    # killer features
    "NTL_combined", "NTL_matched",
    "Stint_MaxTL", "Sess_LapMax",
    # teacher
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

print(f"X shape: {X.shape}, y mean: {y.mean():.4f}")

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 127,
    "min_child_samples": 50, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5,
    "lambda_l2": 1.0, "verbose": -1, "n_jobs": -1,
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- fold {fold+1}/{N_SPLITS} ---")
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cat_cols, reference=dtr)
    m = lgb.train(
        params, dtr, num_boost_round=4000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(300)],
    )
    oof[va_idx] = m.predict(X_va, num_iteration=m.best_iteration)
    test_pred += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
    print(f"fold {fold+1} AUC: {roc_auc_score(y_va, oof[va_idx]):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"\n=== v4 OOF AUC: {cv_auc:.6f} ===")
print(f"baseline 0.943912  ->  v4 {cv_auc:.6f}  (delta {cv_auc - 0.943912:+.6f})")
print(f"elapsed: {time.time() - t0:.1f}s")

imp = sorted(zip(features, m.feature_importance(importance_type="gain")), key=lambda x: -x[1])
print("\n=== top features by gain ===")
for f, g in imp[:20]:
    print(f"  {f:30s}  {g:>15.0f}")

sub = pl.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.write_csv("submissions/v4.csv")
print(f"\nsubmission: submissions/v4.csv ({sub.shape})")
