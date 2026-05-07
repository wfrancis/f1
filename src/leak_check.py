"""Check whether the next lap's Stint can recover PitNextLap perfectly in train,
and whether next-lap data is available for test rows."""
import polars as pl

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")

# Combine
combined = pl.concat(
    [train.drop("PitNextLap").with_columns(pl.lit("train").alias("split")),
     test.with_columns(pl.lit("test").alias("split"))],
    how="vertical",
)
print(f"combined: {combined.shape}")

# For each (Race, Year, Driver, LapNumber), find the Stint at LapNumber+1
combined_sorted = combined.sort(["Race", "Year", "Driver", "LapNumber"])

# Self-join: lap N+1 stint
next_lap = combined.select([
    "Race", "Year", "Driver",
    (pl.col("LapNumber") - 1).alias("LapNumber"),
    pl.col("Stint").alias("NextStint"),
    pl.col("split").alias("next_split"),
])
print(f"next_lap rows: {next_lap.shape}")

merged = combined.join(next_lap, on=["Race", "Year", "Driver", "LapNumber"], how="left")
print(f"merged rows: {merged.shape}")
print()

# Check coverage: for test rows, what fraction have NextStint available?
test_coverage = merged.filter(pl.col("split") == "test").select(
    pl.col("NextStint").is_not_null().mean().alias("coverage"),
    pl.col("NextStint").is_not_null().sum().alias("with_next"),
    pl.len().alias("total")
)
print("=== test rows: next-lap coverage ===")
print(test_coverage)
print()

# In train, verify the leak: PitNextLap == (NextStint > Stint)
train_with_next = merged.filter(pl.col("split") == "train").join(
    train.select(["id", "PitNextLap"]), on="id", how="left"
)
verify = train_with_next.filter(pl.col("NextStint").is_not_null()).select([
    pl.col("PitNextLap"),
    (pl.col("NextStint") > pl.col("Stint")).cast(pl.Int64).alias("predicted"),
])
agree = (verify["PitNextLap"].cast(pl.Int64) == verify["predicted"]).mean()
print(f"=== train: next-stint == PitNextLap agreement = {agree:.6f} ===")
print(verify.head(10))
print()

# What about train rows where next-lap is missing? What does the target look like?
train_no_next = merged.filter(pl.col("split") == "train").join(
    train.select(["id", "PitNextLap"]), on="id", how="left"
).filter(pl.col("NextStint").is_null())
print(f"=== train rows without next-lap: {train_no_next.shape[0]} ===")
print(f"target rate among them: {train_no_next['PitNextLap'].mean():.4f}")
print(f"are these last laps of session?")
print(train_no_next.group_by("Stint").agg(pl.len(), pl.col("PitNextLap").mean()).sort("Stint"))
