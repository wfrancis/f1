"""Build higher-level stack submissions from saved OOF/test predictions.

The handover stack fits one L1 logistic model on all 107 base OOFs and then
fits a final meta-model on all rows.  This script adds a fold-bagged variant:
for several meta-CV split seeds, each row is predicted only by stackers that
did not train on that row, and test predictions are averaged over the same
fold stackers.

Current best local variant:
    python src/stack_meta.py --name best_meta_bag_c008_s6
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions"


def clipped_logit(a: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    return logit(np.clip(a, eps, 1.0 - eps)).astype("float32")


def load_y() -> np.ndarray:
    train = pd.read_csv(ROOT / "data/train.csv").sort_values("id").reset_index(drop=True)
    return train["PitNextLap"].astype(int).to_numpy()


def load_test_ids() -> np.ndarray:
    test = pd.read_csv(ROOT / "data/test.csv").sort_values("id").reset_index(drop=True)
    return test["id"].to_numpy()


def load_stack_inputs(y: np.ndarray, min_auc: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    oof_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []

    for path in sorted(glob.glob(str(SUB_DIR / "*_oof.npy"))):
        name = Path(path).name.replace("_oof.npy", "")
        test_path = SUB_DIR / f"{name}_test.npy"
        if not test_path.exists():
            continue

        oof = np.load(path)
        test = np.load(test_path)
        if len(oof) != len(y) or len(test) == 0:
            continue
        if not np.isfinite(oof).all() or not np.isfinite(test).all():
            continue

        auc = roc_auc_score(y, oof)
        if auc <= min_auc:
            continue

        names.append(name)
        oof_cols.append(clipped_logit(oof))
        test_cols.append(clipped_logit(test))

    if not names:
        raise RuntimeError("No eligible OOF/test pairs found.")

    return names, np.stack(oof_cols, axis=1), np.stack(test_cols, axis=1)


def fit_meta(C: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        l1_ratio=1.0,
        solver="liblinear",
        max_iter=500,
        random_state=seed,
    )


def fold_bag_stack(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    *,
    C: float,
    seeds: list[int],
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof_sum = np.zeros(len(y), dtype="float64")
    test_sum = np.zeros(X_test.shape[0], dtype="float64")

    for seed in seeds:
        seed_oof = np.zeros(len(y), dtype="float64")
        seed_test = np.zeros(X_test.shape[0], dtype="float64")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        nonzero = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            model = fit_meta(C, seed)
            model.fit(X[tr_idx], y[tr_idx])
            seed_oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
            seed_test += model.predict_proba(X_test)[:, 1] / n_splits
            nonzero.append(int((np.abs(model.coef_[0]) > 1e-10).sum()))

        seed_auc = roc_auc_score(y, seed_oof)
        oof_sum += seed_oof / len(seeds)
        test_sum += seed_test / len(seeds)
        bag_auc = roc_auc_score(y, oof_sum * len(seeds) / (seeds.index(seed) + 1))
        print(
            f"seed={seed:5d} seed_auc={seed_auc:.9f} "
            f"bag_auc={bag_auc:.9f} nz={np.mean(nonzero):.1f}",
            flush=True,
        )

    return oof_sum, test_sum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="best_meta_bag_c008_s6")
    parser.add_argument("--C", type=float, default=0.08)
    parser.add_argument("--seeds", default="42,7,99,3407,1234,2024")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-auc", type=float, default=0.85)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    y = load_y()
    ids = load_test_ids()
    names, X, X_test = load_stack_inputs(y, args.min_auc)
    print(f"loaded {len(names)} models; X={X.shape}; X_test={X_test.shape}")
    print(f"C={args.C}, seeds={seeds}, folds={args.n_splits}")

    oof, test_pred = fold_bag_stack(
        X,
        y,
        X_test,
        C=args.C,
        seeds=seeds,
        n_splits=args.n_splits,
    )

    auc = roc_auc_score(y, oof)
    print(f"\n{args.name} OOF AUC: {auc:.9f}")
    print(
        "test stats: "
        f"mean={test_pred.mean():.8f} std={test_pred.std():.8f} "
        f"min={test_pred.min():.8f} max={test_pred.max():.8f}"
    )

    np.save(SUB_DIR / f"{args.name}_oof.npy", oof)
    np.save(SUB_DIR / f"{args.name}_test.npy", test_pred)
    pd.DataFrame({"id": ids, "PitNextLap": test_pred}).to_csv(
        SUB_DIR / f"{args.name}.csv",
        index=False,
    )
    print(f"wrote submissions/{args.name}.csv")


if __name__ == "__main__":
    main()
