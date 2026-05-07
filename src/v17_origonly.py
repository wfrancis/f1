"""v17: train ONLY on (orig + fastf1) clean data, predict on comp test.
Uses NO comp data for training. Tests if pure clean signal beats noisy comp."""
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
OUT_NAME = f"v17_origonly_s{SEED}"
t0 = time.time()
print(f"=== {OUT_NAME} ===", flush=True)

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")
fastf1 = pl.read_parquet("data/fastf1_orig_format.parquet")

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

def add_ntl_cols(df):
    return df.with_columns([
        pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
        pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Sess_LapMax"),
        pl.col("Normalized_TyreLife").alias("NTL_combined"),
        pl.lit(1).cast(pl.Int8).alias("NTL_matched"),
        pl.col("Normalized_TyreLife").alias("NTL_reconstructed"),
    ])

orig_with_ntl = add_ntl_cols(orig)
ff_with_ntl = add_ntl_cols(fastf1)

train_pd = train_match.sort("id").to_pandas()
test_pd = test_match.sort("id").to_pandas()
orig_pd = orig_with_ntl.to_pandas()
ff_pd = ff_with_ntl.to_pandas()

y_train = train_pd["PitNextLap"].astype(int).values
y_orig = orig_pd["PitNextLap"].astype(int).values
y_ff = ff_pd["PitNextLap"].astype(int).values
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
ff_pd = fe(ff_pd)

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

X_train = train_pd[features].copy()
X_test = test_pd[features].copy()
X_orig = orig_pd[features].copy()
X_ff = ff_pd[features].copy()
for c in cat_cols:
    cats = pd.concat([X_train[c].astype(str), X_orig[c].astype(str), X_test[c].astype(str), X_ff[c].astype(str)], ignore_index=True).astype(str).unique()
    X_train[c] = pd.Categorical(X_train[c].astype(str), categories=cats)
    X_orig[c] = pd.Categorical(X_orig[c].astype(str), categories=cats)
    X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats)
    X_ff[c] = pd.Categorical(X_ff[c].astype(str), categories=cats)

# Train ONLY on orig + fastf1, evaluate via 5-fold CV on COMP train
X_pure = pd.concat([X_orig, X_ff], ignore_index=True)
y_pure = np.concatenate([y_orig, y_ff])
print(f"pure train (orig+ff): {X_pure.shape}, comp val: {X_train.shape}", flush=True)

# OOF on comp by training on (orig + ff + fold-train-comp): wait, that's v15. Want pure orig+ff only.
# Actually: train on (orig+ff) ALONE, predict on comp validation folds
# This gives a single model; OOF doesn't really apply to single model
# Use 5 random subsets of (orig+ff) to make 5 models, average preds on comp test

# Simpler: train on orig+ff with 5 different seeds, average test preds
# But to give an OOF on COMP train, just predict comp train directly with single model

strat_key = (train_pd["PitNextLap"].astype(int).astype(str) + "_" + train_pd["Year"].astype(str)).values
N_SPLITS = 5
oof = np.zeros(len(X_train))
test_pred = np.zeros(len(X_test))

# Bag 5 models, each with different seed
for fold in range(N_SPLITS):
    fold_seed = SEED + fold * 100
    print(f"  bag {fold+1}/{N_SPLITS} (seed={fold_seed}, {time.time()-t0:.0f}s)", flush=True)
    # Subsample 80% of pure data per bag
    rng = np.random.RandomState(fold_seed)
    sub_idx = rng.choice(len(X_pure), size=int(0.8 * len(X_pure)), replace=False)
    X_sub = X_pure.iloc[sub_idx]
    y_sub = y_pure[sub_idx]
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.04, "num_leaves": 127,
        "min_child_samples": 30, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "lambda_l2": 1.0, "verbose": -1, "n_jobs": 2, "seed": fold_seed,
    }
    dtr = lgb.Dataset(X_sub, y_sub, categorical_feature=cat_cols)
    m = lgb.train(params, dtr, num_boost_round=1500)
    oof += m.predict(X_train) / N_SPLITS
    test_pred += m.predict(X_test) / N_SPLITS

cv_auc = roc_auc_score(y_train, oof)
print(f"\n=== {OUT_NAME} OOF AUC on comp train: {cv_auc:.6f} === ({time.time()-t0:.0f}s)", flush=True)
sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv(f"submissions/{OUT_NAME}.csv", index=False)
np.save(f"submissions/{OUT_NAME}_oof.npy", oof)
np.save(f"submissions/{OUT_NAME}_test.npy", test_pred)
print(f"saved")
