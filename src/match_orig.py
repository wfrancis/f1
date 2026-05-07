"""Try to match competition rows to original dataset rows.
If we can find an exact match, we can read the true Normalized_TyreLife.
"""
import polars as pl

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

print(f"orig: {orig.shape}, train: {train.shape}, test: {test.shape}")
print(f"sum: {orig.shape[0]} vs combined train+test: {train.shape[0] + test.shape[0]}")
print()

# Method 1: try matching on (Race, Year, LapNumber, Stint, TyreLife, Position, Compound)
# These should be identical if data is just driver-anonymized
key_cols = ["Race", "Year", "LapNumber", "Stint", "TyreLife", "Position", "Compound", "PitStop"]

print("=== check for exact matches on key_cols (no Driver) ===")
orig_keys = orig.select(key_cols + ["Normalized_TyreLife", "Driver"])
print(f"orig unique on key_cols: {orig_keys.unique(subset=key_cols).shape[0]} / {len(orig)}")

# Try matching train to orig
match = train.select(["id"] + key_cols).join(
    orig_keys, on=key_cols, how="left"
)
print(f"train rows: {len(train)}")
print(f"  with at least one orig match: {match['Normalized_TyreLife'].is_not_null().sum()}")
print(f"  unique mapping (1:1): {(match.group_by('id').len()['len'] == 1).sum()}")

# How many train rows match to multiple orig rows?
group_sizes = match.group_by("id").agg(pl.len().alias("matches")).group_by("matches").agg(pl.len().alias("count")).sort("matches")
print(f"\n=== match-count distribution (per-id) ===")
print(group_sizes)

# For uniquely-matched rows, do values agree on PitNextLap?
unique_match = match.filter(
    pl.col("id").is_in(
        match.group_by("id").len().filter(pl.col("len") == 1)["id"]
    )
).join(train.select(["id", "PitNextLap"]), on="id", how="left")
print(f"\nuniquely matched train rows: {len(unique_match)}")
print(f"sample comparison:")
print(unique_match.head(10).select(["id", "PitNextLap", "Normalized_TyreLife"]))
