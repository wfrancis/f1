"""Verify: when next lap is exactly LapNumber+1, does PitStop[N+1] == PitNextLap[N]?"""
import polars as pl

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")

combined = pl.concat([
    train.drop("PitNextLap"), test
], how="vertical").sort(["Race", "Year", "Driver", "LapNumber"])

# Build a lookup: (race, year, driver, lap) -> PitStop
nxt = combined.select([
    pl.col("Race"),
    pl.col("Year"),
    pl.col("Driver"),
    (pl.col("LapNumber") - 1).alias("LapNumber"),
    pl.col("PitStop").alias("PitStop_at_lap_plus_1"),
    pl.col("Stint").alias("Stint_at_lap_plus_1"),
])

merged = train.join(nxt, on=["Race", "Year", "Driver", "LapNumber"], how="left")
print(f"train rows: {len(train)}")
print(f"with exact lap+1 match: {merged['PitStop_at_lap_plus_1'].is_not_null().sum()}")

# Compare PitNextLap to PitStop_at_lap_plus_1 where available
sub = merged.filter(pl.col("PitStop_at_lap_plus_1").is_not_null()).select([
    pl.col("PitNextLap").cast(pl.Int64),
    pl.col("PitStop_at_lap_plus_1").cast(pl.Int64),
])

agreement = (sub["PitNextLap"] == sub["PitStop_at_lap_plus_1"]).mean()
print(f"PitNextLap == PitStop[N+1] agreement: {agreement:.6f}")
print()

# Cross-tabulate
ct = sub.group_by(["PitNextLap", "PitStop_at_lap_plus_1"]).len().sort(["PitNextLap", "PitStop_at_lap_plus_1"])
print("=== contingency ===")
print(ct)
print()

# Coverage: what fraction of TEST rows have an exact lap+1 match?
test_merged = test.join(nxt, on=["Race", "Year", "Driver", "LapNumber"], how="left")
coverage = test_merged["PitStop_at_lap_plus_1"].is_not_null().mean()
print(f"test coverage with exact lap+1 match: {coverage:.4f}")
print()

# What about lap+2, lap+3?
for k in [2, 3, 4, 5]:
    nxt_k = combined.select([
        pl.col("Race"), pl.col("Year"), pl.col("Driver"),
        (pl.col("LapNumber") - k).alias("LapNumber"),
        pl.col("PitStop").alias(f"PitStop_at_lap_plus_{k}"),
    ])
    test_merged = test_merged.join(nxt_k, on=["Race", "Year", "Driver", "LapNumber"], how="left")
    cov = test_merged[f"PitStop_at_lap_plus_{k}"].is_not_null().mean()
    print(f"test coverage with lap+{k} match: {cov:.4f}")

# How many test rows have ANY future-lap match (within +1 to +5)?
any_future = pl.any_horizontal([
    test_merged[f"PitStop_at_lap_plus_{k}"].is_not_null() for k in [1, 2, 3, 4, 5]
])
print(f"\ntest rows with at least one future lap visible (lap+1..lap+5): {any_future.mean():.4f}")
