"""
Convert fastf1 data to orig-compatible format with derived features.
Computes: LapTime_s, LapTime_Delta, Cumulative_Degradation, PitStop, PitNextLap,
RaceProgress, Normalized_TyreLife, Position_Change.
"""
import pandas as pd
import numpy as np
import polars as pl

ff = pd.read_parquet("data/fastf1_missing_sessions.parquet")
print(f"input: {ff.shape}")

# Convert LapTime to seconds
ff["LapTime_s"] = ff["LapTime"].dt.total_seconds()
ff["Race"] = ff["__race"]
ff["Year"] = ff["__year"].astype(int)
ff["LapNumber"] = ff["LapNumber"].astype(int)
ff["Stint"] = ff["Stint"].fillna(1).astype(int)
ff["TyreLife"] = ff["TyreLife"].fillna(1).astype(float)
ff["Position"] = ff["Position"].fillna(20).astype(int)

# PitStop = pitted IN time present (driver came in to pit on this lap)
ff["PitStop"] = ff["PitInTime"].notna().astype(int)

# Drop bad rows (no LapTime_s, no Compound)
ff = ff.dropna(subset=["LapTime_s", "Compound"])
ff["Compound"] = ff["Compound"].astype(str)
print(f"after dropna: {ff.shape}")

# Sort within (Driver, Race, Year)
ff = ff.sort_values(["Driver", "Race", "Year", "LapNumber"]).reset_index(drop=True)

# Per-session derived features
ff_pl = pl.from_pandas(ff[[
    "Driver", "Race", "Year", "LapNumber", "Stint", "TyreLife", "Position",
    "Compound", "LapTime_s", "PitStop"
]])

ff_pl = ff_pl.with_columns([
    pl.col("LapTime_s").shift(1).over(["Driver", "Race", "Year"]).alias("LapTime_prev"),
    pl.col("Position").shift(1).over(["Driver", "Race", "Year"]).alias("Position_prev"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Race_LapMax"),
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("PitStop").shift(-1).over(["Driver", "Race", "Year"]).alias("PitStop_next"),
])

ff_pl = ff_pl.with_columns([
    (pl.col("LapTime_s") - pl.col("LapTime_prev")).alias("LapTime_Delta"),
    (pl.col("Position_prev") - pl.col("Position")).alias("Position_Change"),
    (pl.col("LapNumber") / pl.col("Race_LapMax")).alias("RaceProgress"),
    (pl.col("TyreLife") / pl.col("Stint_MaxTL")).alias("Normalized_TyreLife"),
    pl.col("PitStop_next").fill_null(0).alias("PitNextLap"),
])

# Cumulative_Degradation: cumulative sum of LapTime_Delta within stint
ff_pl = ff_pl.with_columns([
    pl.col("LapTime_Delta").fill_null(0).cum_sum().over(["Driver", "Race", "Year", "Stint"]).alias("Cumulative_Degradation")
])

# Fill defaults for null values (first lap of session etc.)
ff_pl = ff_pl.with_columns([
    pl.col("LapTime_Delta").fill_null(0),
    pl.col("Position_Change").fill_null(0),
])

# Select final columns matching orig schema
final = ff_pl.select([
    "Driver", "Race", "Year", "LapNumber", "Stint", "TyreLife", "Position",
    "Compound", "LapTime_s", "PitStop", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Normalized_TyreLife", "Position_Change", "PitNextLap"
]).rename({"LapTime_s": "LapTime (s)"})

final = final.with_columns([
    pl.col("PitStop").cast(pl.Int64),
    pl.col("PitNextLap").cast(pl.Int64),
    pl.col("Position_Change").cast(pl.Float64),
])

# Some sanity
print(f"\nfinal: {final.shape}")
print(final.head(5))
print()
print(f"PitStop: {final['PitStop'].mean():.4f} pit rate")
print(f"PitNextLap: {final['PitNextLap'].mean():.4f}")
print(f"unique races: {sorted(final['Race'].unique().to_list())}")
print(f"unique years: {sorted(final['Year'].unique().to_list())}")

final.write_parquet("data/fastf1_orig_format.parquet")
print(f"\nsaved data/fastf1_orig_format.parquet ({final.shape})")
