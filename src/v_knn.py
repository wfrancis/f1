"""KNN on orig+fastf1: for each row, find K nearest neighbors and use their PNL mean.
Gives a fundamentally different signal vs gradient boosting."""
import sys, time
import numpy as np
import pandas as pd
import polars as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
K = 30
OUT_NAME = f"v_knn_k{K}"
t0 = time.time()
print(f"=== {OUT_NAME} ===", flush=True)

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

train_pd = train.sort("id").to_pandas()
test_pd = test.sort("id").to_pandas()
orig_pd = orig.to_pandas()
y = train_pd["PitNextLap"].astype(int).values
y_orig = orig_pd["PitNextLap"].astype(int).values
ids_test = test_pd["id"].values

# Numeric features only for distance
num_features = ["LapNumber", "Stint", "TyreLife", "Position", "Cumulative_Degradation", "RaceProgress", "Position_Change", "Year", "PitStop", "LapTime_Delta"]
def get_features(df):
    X = df[num_features + ["LapTime (s)"]].rename(columns={"LapTime (s)": "LapTime_s"})
    # Add Compound and Race as integer encoded
    cmp_map = {"SOFT":0, "MEDIUM":1, "HARD":2, "INTERMEDIATE":3, "WET":4}
    X["Compound_int"] = df["Compound"].map(cmp_map).fillna(5).astype(int)
    return X.fillna(0).values.astype(float)

X_train = get_features(train_pd)
X_test = get_features(test_pd)
X_orig = get_features(orig_pd)

scaler = StandardScaler()
X_orig_s = scaler.fit_transform(X_orig)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# For each comp row, find K nearest orig rows, take mean PNL
print(f"orig: {X_orig_s.shape}, train: {X_train_s.shape}, test: {X_test_s.shape}", flush=True)

print(f"Building KNN index ({time.time()-t0:.0f}s)...", flush=True)
nbrs = NearestNeighbors(n_neighbors=K, n_jobs=2).fit(X_orig_s)

print(f"Querying train ({time.time()-t0:.0f}s)...", flush=True)
_, ind_tr = nbrs.kneighbors(X_train_s)
oof = y_orig[ind_tr].mean(axis=1)
print(f"Querying test ({time.time()-t0:.0f}s)...", flush=True)
_, ind_te = nbrs.kneighbors(X_test_s)
test_pred = y_orig[ind_te].mean(axis=1)

cv_auc = roc_auc_score(y, oof)
print(f"\n=== {OUT_NAME} OOF AUC: {cv_auc:.6f} === ({time.time()-t0:.0f}s)", flush=True)
sub = pd.DataFrame({"id": ids_test, "PitNextLap": test_pred})
sub.to_csv(f"submissions/{OUT_NAME}.csv", index=False)
np.save(f"submissions/{OUT_NAME}_oof.npy", oof)
np.save(f"submissions/{OUT_NAME}_test.npy", test_pred)
print(f"saved")
