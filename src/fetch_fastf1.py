"""
Try to use fastf1 to fetch race lap data for the 11 sessions missing from orig.
Add as additional clean training data.
"""
import time
import polars as pl
import pandas as pd
import numpy as np

t0 = time.time()
print("=== fastf1 fetch ===", flush=True)

import fastf1

# Set up cache
import os
os.makedirs("data/fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("data/fastf1_cache")

# What sessions does orig have?
orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")
orig_sessions = set((orig["Race"] + "|" + orig["Year"].cast(pl.String)).unique().to_list())
print(f"orig has {len(orig_sessions)} sessions")

# What sessions does comp have?
train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
comp = pl.concat([train.select("Race", "Year"), test.select("Race", "Year")], how="vertical")
comp_sessions = set((comp["Race"] + "|" + comp["Year"].cast(pl.String)).unique().to_list())
print(f"comp has {len(comp_sessions)} sessions")

missing = comp_sessions - orig_sessions
print(f"missing in orig: {len(missing)}")
for m in sorted(missing):
    print(f"  {m}")

# Try to fetch each missing session via fastf1
results = []
for s in sorted(missing):
    parts = s.split("|")
    race_name, year = parts[0], int(parts[1])
    print(f"\nFetching {race_name} {year}...", flush=True)
    try:
        # Race name needs to match fastf1 convention
        # Try direct name first
        session = fastf1.get_session(year, race_name, "R")  # R = Race
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        print(f"  got {len(laps)} laps", flush=True)
        # convert to dataframe and save
        df = laps.copy()
        df["__race"] = race_name
        df["__year"] = year
        results.append(df)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

if results:
    combined = pd.concat(results, ignore_index=True)
    combined.to_parquet("data/fastf1_missing_sessions.parquet", index=False)
    print(f"\nSaved {len(combined)} rows to data/fastf1_missing_sessions.parquet")
else:
    print("No results.")
print(f"elapsed: {time.time()-t0:.0f}s")
