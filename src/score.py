"""
Local blend scorer. Takes OOF .npy files, computes blended OOF AUC.
Use this to verify blends LOCALLY before burning Kaggle submissions.

Usage:
    python src/score.py                          # report all single-model OOF AUCs
    python src/score.py v10 v11 v7_s7            # equal-weight logit blend
    python src/score.py --search v10 v11 v7_s7   # grid search blend weights
"""
import sys
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OOF_DIR = "submissions"

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def load_y():
    train = pd.read_csv("data/train.csv").sort_values("id").reset_index(drop=True)
    return train["PitNextLap"].astype(int).values

def load_oof(name):
    path = f"{OOF_DIR}/{name}_oof.npy"
    return np.load(path)

def list_available():
    files = sorted(glob.glob(f"{OOF_DIR}/*_oof.npy"))
    return [f.split("/")[-1].replace("_oof.npy", "") for f in files]

def report_singles(y):
    print(f"=== single-model OOF AUCs (n={len(y)}, target rate {y.mean():.4f}) ===")
    rows = []
    for name in list_available():
        try:
            oof = load_oof(name)
            if len(oof) != len(y):
                continue
            auc = roc_auc_score(y, oof)
            rows.append((name, auc))
        except Exception as e:
            print(f"  {name}: ERROR {e}")
    rows.sort(key=lambda r: -r[1])
    for name, auc in rows:
        print(f"  {name:20s}  AUC {auc:.6f}")
    return rows

def blend_logit(oofs, weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    z = np.zeros_like(oofs[0])
    for w, o in zip(weights, oofs):
        z = z + w * logit(o)
    return sigmoid(z)

def grid_search(names, y, top_k=10):
    oofs = [load_oof(n) for n in names]
    print(f"=== grid search over {len(names)} models: {names} ===")
    candidates = list(itertools.product([0, 0.5, 1, 2, 3], repeat=len(names)))
    candidates = [c for c in candidates if sum(c) > 0]
    results = []
    for w in candidates:
        b = blend_logit(oofs, w)
        auc = roc_auc_score(y, b)
        results.append((w, auc))
    results.sort(key=lambda r: -r[1])
    print(f"  Top {top_k} blends:")
    for w, auc in results[:top_k]:
        print(f"    weights {w}  AUC {auc:.6f}")
    return results

def main():
    args = sys.argv[1:]
    y = load_y()
    if not args:
        report_singles(y)
        return
    if args[0] == "--search":
        names = args[1:]
        if not names:
            print("usage: score.py --search NAME1 NAME2 ...")
            sys.exit(1)
        grid_search(names, y)
        return
    if args[0] == "--auto":
        # auto: load top-k singles by AUC, grid search them
        rows = report_singles(y)
        names = [n for n, _ in rows[:5]]
        print()
        grid_search(names, y)
        return
    # Equal-weight logit blend of args
    names = args
    oofs = [load_oof(n) for n in names]
    auc = roc_auc_score(y, blend_logit(oofs, [1] * len(names)))
    print(f"Equal-weight logit blend of {names}: OOF AUC {auc:.6f}")
    # also try uniform vs each individually
    for n, o in zip(names, oofs):
        print(f"  {n}: {roc_auc_score(y, o):.6f}")

if __name__ == "__main__":
    main()
