"""Simple MLP on (comp+orig) — different model class for ensemble diversity."""
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
OUT_NAME = f"v15_mlp_s{SEED}"
t0 = time.time()
print(f"=== {OUT_NAME} ===", flush=True)

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

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

orig_with_ntl = orig.with_columns([
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
    pl.col("Normalized_TyreLife").alias("NTL_combined"),
    pl.lit(1).cast(pl.Int8).alias("NTL_matched"),
    pl.col("Normalized_TyreLife").alias("NTL_reconstructed"),
])

train_pd = train_match.sort("id").to_pandas()
test_pd = test_match.sort("id").to_pandas()
orig_pd = orig_with_ntl.to_pandas()
y = train_pd["PitNextLap"].astype(int).values
y_orig = orig_pd["PitNextLap"].astype(int).values
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

# One-hot encode categoricals (small)
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

# Build feature matrices
def to_features(df):
    parts = [df[num_cols].values.astype(np.float32)]
    for c in cat_cols:
        dummies = pd.get_dummies(df[c], prefix=c).astype(np.float32)
        parts.append(dummies.values)
    return np.hstack(parts), [c for col in cat_cols for c in pd.get_dummies(df[col], prefix=col).columns]

# Combine all dfs to align dummy columns
all_df = pd.concat([train_pd[num_cols + cat_cols], test_pd[num_cols + cat_cols], orig_pd[num_cols + cat_cols]], ignore_index=True)
parts = [all_df[num_cols].values.astype(np.float32)]
for c in cat_cols:
    dummies = pd.get_dummies(all_df[c], prefix=c).astype(np.float32)
    parts.append(dummies.values)
X_all = np.hstack(parts)
n_train, n_test, n_orig = len(train_pd), len(test_pd), len(orig_pd)
X = X_all[:n_train]
X_test = X_all[n_train:n_train+n_test]
X_orig = X_all[n_train+n_test:]
print(f"feature dim: {X.shape[1]}, train: {X.shape}, test: {X_test.shape}, orig: {X_orig.shape}", flush=True)

# Replace inf/nan
X = np.nan_to_num(X, nan=0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0, posinf=1e6, neginf=-1e6)
X_orig = np.nan_to_num(X_orig, nan=0, posinf=1e6, neginf=-1e6)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)
X_orig = scaler.transform(X_orig)

# 5-fold CV
strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, strat_key)):
    print(f"  fold {fold+1}/{N_SPLITS} ({time.time()-t0:.0f}s)", flush=True)
    X_tr = np.vstack([X[tr_idx], X_orig])
    y_tr = np.concatenate([y[tr_idx], y_orig])
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=20,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
        verbose=False,
        batch_size=2048,
    )
    model.fit(X_tr, y_tr)
    oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS
    print(f"    fold {fold+1} AUC: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}", flush=True)

cv_auc = roc_auc_score(y, oof)
print(f"\n=== {OUT_NAME} OOF AUC: {cv_auc:.6f} === ({time.time()-t0:.0f}s)", flush=True)
sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv(f"submissions/{OUT_NAME}.csv", index=False)
np.save(f"submissions/{OUT_NAME}_oof.npy", oof)
np.save(f"submissions/{OUT_NAME}_test.npy", test_pred)
print(f"saved")
