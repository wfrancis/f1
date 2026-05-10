# Predicting F1 Pit Stops — Kaggle Playground S6E5

Solution pipeline for [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5).

**Current best**: public LB **0.95422**, rank **#13/1123 (top 1.16%)**. Best submission: `submissions/vault_blend_90.csv` (90% vault + 10% our 137-OOF stack). The "vault" is `anthonytherrien/predicting-f1-pit-stops-vault` — see HANDOVER.md.

See **[HANDOVER.md](HANDOVER.md)** for the full project context: what's been tried, what worked, what didn't, the file map, and the prioritized list of moves to try next.

## Quick start

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install polars pandas numpy lightgbm xgboost catboost scikit-learn pyarrow pytabkit torch fastf1

kaggle competitions download -c playground-series-s6e5 -p data/
unzip data/playground-series-s6e5.zip -d data/

python src/score.py --auto    # see all OOF AUCs + auto-stack
```

## Layout

- `src/` — training scripts (one per model variant)
- `submissions/` — OOF + test predictions (`*_oof.npy`, `*_test.npy`) and the best Kaggle submission CSVs
- `data/orig/` — the original F1 dataset that the comp was synthesized from (CC BY-SA 4.0)
- `data/fastf1_*.parquet` — additional sessions pulled via the `fastf1` library
