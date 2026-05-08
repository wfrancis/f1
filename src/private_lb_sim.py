"""Simulate Kaggle's public/private split on train OOF predictions.

The public leaderboard uses about 20% of test. This script repeatedly carves
the train OOF rows into fake 20% "public" and 80% "private" splits, including
grouped splits by race-year and year. Candidates are ranked by simulated
private AUC, especially median and lower-tail private AUC.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SUB_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"


@dataclass
class Candidate:
    name: str
    oof: np.ndarray
    full_auc: float
    order: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="*", default=None, help="Candidate names without _oof.npy.")
    p.add_argument("--top", type=int, default=0, help="Also include top-N single OOF files by full AUC.")
    p.add_argument("--ref", default="best_meta_bag_c008_s6", help="Reference candidate for deltas.")
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--public-frac", type=float, default=0.20)
    p.add_argument("--n-strat", type=int, default=300)
    p.add_argument("--n-race-year", type=int, default=300)
    p.add_argument("--n-year", type=int, default=120)
    p.add_argument("--min-auc", type=float, default=0.85)
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def load_train() -> pd.DataFrame:
    train = pd.read_csv(ROOT / "data/train.csv").sort_values("id").reset_index(drop=True)
    train["target"] = train["PitNextLap"].astype("int8")
    train["race_year"] = train["Race"].astype(str) + "_" + train["Year"].astype(str)
    return train


def fast_auc(y: np.ndarray, pred: np.ndarray) -> float:
    return float(roc_auc_score(y, pred))


def auc_from_sorted_order(y: np.ndarray, order: np.ndarray, mask: np.ndarray) -> float:
    """AUC for a subset, reusing the candidate's global prediction order.

    AUC depends only on pairwise ordering inside the subset. Filtering a global
    prediction order gives the same relative order as sorting the subset, but
    avoids repeated O(n log n) sorts across hundreds of simulated splits.
    """
    yy = y[order[mask[order]]]
    pos = int(yy.sum())
    neg = len(yy) - pos
    if pos <= 0 or neg <= 0:
        return float("nan")
    neg_before = np.cumsum(1 - yy, dtype=np.int64)
    return float(neg_before[yy == 1].sum() / (pos * neg))


def available_oofs(y: np.ndarray, min_auc: float) -> list[Candidate]:
    out: list[Candidate] = []
    for path in sorted(SUB_DIR.glob("*_oof.npy")):
        name = path.name.removesuffix("_oof.npy")
        try:
            oof = np.load(path).astype("float64", copy=False)
        except Exception:
            continue
        if len(oof) != len(y) or not np.isfinite(oof).all():
            continue
        auc = fast_auc(y, oof)
        if auc >= min_auc:
            out.append(Candidate(name, oof, auc, np.argsort(oof, kind="mergesort")))
    out.sort(key=lambda c: c.full_auc, reverse=True)
    return out


def load_candidates(args: argparse.Namespace, y: np.ndarray) -> list[Candidate]:
    all_cands = {c.name: c for c in available_oofs(y, args.min_auc)}
    selected: dict[str, Candidate] = {}

    if args.top:
        for cand in sorted(all_cands.values(), key=lambda c: c.full_auc, reverse=True)[: args.top]:
            selected[cand.name] = cand

    for name in args.names or []:
        if name not in all_cands:
            path = SUB_DIR / f"{name}_oof.npy"
            raise FileNotFoundError(f"missing or invalid OOF for {name}: {path}")
        selected[name] = all_cands[name]

    if args.ref not in selected:
        if args.ref not in all_cands:
            raise FileNotFoundError(f"reference OOF not found: {args.ref}")
        selected[args.ref] = all_cands[args.ref]

    return sorted(selected.values(), key=lambda c: c.full_auc, reverse=True)


def valid_split(y: np.ndarray, public_idx: np.ndarray, private_idx: np.ndarray) -> bool:
    if len(public_idx) == 0 or len(private_idx) == 0:
        return False
    pub_pos = int(y[public_idx].sum())
    pri_pos = int(y[private_idx].sum())
    return pub_pos > 10 and len(public_idx) - pub_pos > 10 and pri_pos > 10 and len(private_idx) - pri_pos > 10


def grouped_public_indices(
    groups: np.ndarray,
    y: np.ndarray,
    public_frac: float,
    rng: np.random.Generator,
    n_splits: int,
    family: str,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    labels, inv = np.unique(groups, return_inverse=True)
    group_indices = [np.flatnonzero(inv == i) for i in range(len(labels))]
    sizes = np.array([len(idx) for idx in group_indices])
    target = int(round(len(groups) * public_frac))
    all_idx = np.arange(len(groups))

    splits: list[tuple[str, np.ndarray, np.ndarray]] = []
    tries = 0
    max_tries = max(n_splits * 50, 1000)
    while len(splits) < n_splits and tries < max_tries:
        tries += 1
        order = rng.permutation(len(labels))
        chosen: list[int] = []
        total = 0
        for g in order:
            if not chosen or total < target:
                chosen.append(int(g))
                total += int(sizes[g])
            if total >= target:
                break
        public_idx = np.concatenate([group_indices[g] for g in chosen])
        public_idx.sort()
        private_idx = np.setdiff1d(all_idx, public_idx, assume_unique=True)
        if not valid_split(y, public_idx, private_idx):
            continue
        split_name = "+".join(str(labels[g]) for g in sorted(chosen))
        splits.append((family, public_idx, private_idx))
    if len(splits) < n_splits:
        print(f"warning: requested {n_splits} {family} splits, built {len(splits)}")
    return splits


def build_splits(train: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y = train["target"].to_numpy()
    rng = np.random.default_rng(args.seed)
    all_splits: list[tuple[str, np.ndarray, np.ndarray]] = []

    if args.n_strat:
        sss = StratifiedShuffleSplit(
            n_splits=args.n_strat,
            test_size=args.public_frac,
            random_state=args.seed,
        )
        for _, public_idx in sss.split(np.zeros(len(y)), y):
            public_idx = np.asarray(public_idx)
            private_mask = np.ones(len(y), dtype=bool)
            private_mask[public_idx] = False
            private_idx = np.flatnonzero(private_mask)
            if valid_split(y, public_idx, private_idx):
                all_splits.append(("stratified", public_idx, private_idx))

    if args.n_race_year:
        all_splits.extend(
            grouped_public_indices(
                train["race_year"].to_numpy(),
                y,
                args.public_frac,
                rng,
                args.n_race_year,
                "race_year",
            )
        )

    if args.n_year:
        all_splits.extend(
            grouped_public_indices(
                train["Year"].astype(str).to_numpy(),
                y,
                args.public_frac,
                rng,
                args.n_year,
                "year",
            )
        )

    return all_splits


def summarize(split_df: pd.DataFrame, cands: list[Candidate], ref: str) -> pd.DataFrame:
    full_auc = {c.name: c.full_auc for c in cands}
    rows = []
    for (family, name), g in split_df.groupby(["family", "name"], sort=False):
        rows.append(
            {
                "family": family,
                "name": name,
                "n_splits": len(g),
                "full_auc": full_auc[name],
                "private_median": g["private_auc"].median(),
                "private_p10": g["private_auc"].quantile(0.10),
                "private_p05": g["private_auc"].quantile(0.05),
                "private_mean": g["private_auc"].mean(),
                "private_std": g["private_auc"].std(ddof=0),
                "public_median": g["public_auc"].median(),
                "gap_median": (g["public_auc"] - g["private_auc"]).median(),
                "delta_ref_median": g["private_delta_ref"].median(),
                "delta_ref_p10": g["private_delta_ref"].quantile(0.10),
                "private_win_rate": (g["private_rank"] == 1).mean(),
                "beats_ref_rate": (g["private_delta_ref"] > 0).mean() if name != ref else 0.0,
            }
        )
    summary = pd.DataFrame(rows)

    overall = []
    for name, g in split_df.groupby("name", sort=False):
        overall.append(
            {
                "family": "ALL",
                "name": name,
                "n_splits": len(g),
                "full_auc": full_auc[name],
                "private_median": g["private_auc"].median(),
                "private_p10": g["private_auc"].quantile(0.10),
                "private_p05": g["private_auc"].quantile(0.05),
                "private_mean": g["private_auc"].mean(),
                "private_std": g["private_auc"].std(ddof=0),
                "public_median": g["public_auc"].median(),
                "gap_median": (g["public_auc"] - g["private_auc"]).median(),
                "delta_ref_median": g["private_delta_ref"].median(),
                "delta_ref_p10": g["private_delta_ref"].quantile(0.10),
                "private_win_rate": (g["private_rank"] == 1).mean(),
                "beats_ref_rate": (g["private_delta_ref"] > 0).mean() if name != ref else 0.0,
            }
        )
    summary = pd.concat([pd.DataFrame(overall), summary], ignore_index=True)
    return summary.sort_values(["family", "private_p10", "private_median"], ascending=[True, False, False])


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(exist_ok=True)
    train = load_train()
    y = train["target"].to_numpy()
    cands = load_candidates(args, y)
    splits = build_splits(train, args)

    print(f"loaded {len(cands)} candidates, {len(splits)} simulated splits")
    for cand in cands:
        print(f"  {cand.name:55s} full_auc={cand.full_auc:.9f}")

    records = []
    for split_id, (family, public_idx, private_idx) in enumerate(splits):
        private_scores: dict[str, float] = {}
        public_scores: dict[str, float] = {}
        public_mask = np.zeros(len(y), dtype=bool)
        public_mask[public_idx] = True
        private_mask = ~public_mask
        for cand in cands:
            public_scores[cand.name] = auc_from_sorted_order(y, cand.order, public_mask)
            private_scores[cand.name] = auc_from_sorted_order(y, cand.order, private_mask)
        ref_private = private_scores[args.ref]
        ranks = {
            name: rank + 1
            for rank, (name, _) in enumerate(
                sorted(private_scores.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        for cand in cands:
            records.append(
                {
                    "split_id": split_id,
                    "family": family,
                    "name": cand.name,
                    "public_n": len(public_idx),
                    "private_n": len(private_idx),
                    "public_pos": int(y[public_idx].sum()),
                    "private_pos": int(y[private_idx].sum()),
                    "public_auc": public_scores[cand.name],
                    "private_auc": private_scores[cand.name],
                    "private_delta_ref": private_scores[cand.name] - ref_private,
                    "private_rank": ranks[cand.name],
                }
            )

    split_df = pd.DataFrame(records)
    summary = summarize(split_df, cands, args.ref)

    prefix = args.out_prefix or f"private_lb_sim_seed{args.seed}"
    split_path = REPORT_DIR / f"{prefix}_splits.csv"
    summary_path = REPORT_DIR / f"{prefix}_summary.csv"
    meta_path = REPORT_DIR / f"{prefix}_meta.json"
    split_df.to_csv(split_path, index=False)
    summary.to_csv(summary_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "n_rows": len(train),
                "n_splits": len(splits),
                "candidates": [c.name for c in cands],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Overall simulated-private ranking ===")
    overall = summary[summary["family"] == "ALL"].sort_values(
        ["private_p10", "private_median"], ascending=False
    )
    cols = [
        "name",
        "full_auc",
        "private_median",
        "private_p10",
        "delta_ref_median",
        "delta_ref_p10",
        "private_win_rate",
        "beats_ref_rate",
        "gap_median",
    ]
    print(overall[cols].to_string(index=False, float_format=lambda x: f"{x:.9f}"))

    print("\n=== By split family ===")
    for family in ["stratified", "race_year", "year"]:
        fam = summary[summary["family"] == family].sort_values(
            ["private_p10", "private_median"], ascending=False
        )
        if fam.empty:
            continue
        print(f"\n[{family}]")
        print(fam[cols].to_string(index=False, float_format=lambda x: f"{x:.9f}"))

    print(f"\nwrote {summary_path.relative_to(ROOT)}")
    print(f"wrote {split_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
