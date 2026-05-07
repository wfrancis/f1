"""Fetch fastf1 data for ALL races/years (not just missing) to get more clean training data.
This is a different angle than just adding missing sessions."""
import time
import polars as pl
import pandas as pd
import numpy as np
import os

t0 = time.time()
print("=== fastf1 fetch ALL ===", flush=True)

import fastf1
os.makedirs("data/fastf1_cache", exist_ok=True)
fastf1.Cache.enable_cache("data/fastf1_cache")

# All sessions in comp data
train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
comp = pl.concat([train.select("Race", "Year"), test.select("Race", "Year")], how="vertical")
comp_sessions = sorted(set((comp["Race"] + "|" + comp["Year"].cast(pl.String)).unique().to_list()))
print(f"comp has {len(comp_sessions)} sessions")

# Try to fetch ALL of them
results = []
for s in comp_sessions:
    parts = s.split("|")
    race_name, year = parts[0], int(parts[1])
    print(f"\n{race_name} {year}...", flush=True)
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        df = laps.copy()
        df["__race"] = race_name
        df["__year"] = year
        results.append(df)
        print(f"  {len(laps)} laps", flush=True)
    except Exception as e:
        print(f"  ERROR: {str(e)[:100]}", flush=True)

if results:
    combined = pd.concat(results, ignore_index=True)
    combined.to_parquet("data/fastf1_all_sessions.parquet", index=False)
    print(f"\nSaved {len(combined)} rows to data/fastf1_all_sessions.parquet")
print(f"elapsed: {time.time()-t0:.0f}s")
