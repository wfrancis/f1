import polars as pl

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")

print("=== Year distribution ===")
print("train:")
print(train.group_by("Year").len().sort("Year"))
print("test:")
print(test.group_by("Year").len().sort("Year"))
print()

print("=== driver overlap ===")
tr_drivers = set(train["Driver"].unique().to_list())
te_drivers = set(test["Driver"].unique().to_list())
print(f"train unique drivers: {len(tr_drivers)}")
print(f"test  unique drivers: {len(te_drivers)}")
print(f"intersection:         {len(tr_drivers & te_drivers)}")
print(f"only in train:        {len(tr_drivers - te_drivers)}")
print(f"only in test:         {len(te_drivers - tr_drivers)}")
print()

print("=== race overlap ===")
tr_races = set(train["Race"].unique().to_list())
te_races = set(test["Race"].unique().to_list())
print(f"train: {len(tr_races)}, test: {len(te_races)}, both: {len(tr_races & te_races)}")
print()

# Race session = (Race, Year, Driver) — does this group split between train/test?
print("=== session overlap (Race+Year+Driver) ===")
tr_sess = set(train.select(pl.concat_str(["Race", "Year", "Driver"], separator="|"))[
    pl.col("Race").alias("s") if False else train.select(pl.concat_str(["Race", "Year", "Driver"], separator="|")).columns[0]
].to_list()) if False else None

tr_sess = set((train["Race"] + "|" + train["Year"].cast(pl.String) + "|" + train["Driver"]).to_list())
te_sess = set((test["Race"] + "|" + test["Year"].cast(pl.String) + "|" + test["Driver"]).to_list())
print(f"train sessions: {len(tr_sess)}")
print(f"test  sessions: {len(te_sess)}")
print(f"overlap:        {len(tr_sess & te_sess)}")
print()

# Are id values contiguous within a session in train, or shuffled?
print("=== first 20 rows: same session? ===")
print(train.head(20).select(["id", "Race", "Year", "Driver", "LapNumber", "Stint", "PitNextLap"]))
print()

# id range
print(f"train id range: {train['id'].min()} -- {train['id'].max()}")
print(f"test  id range: {test['id'].min()}  -- {test['id'].max()}")
print()

# Compound distribution
print("=== Compound distribution ===")
print(train.group_by("Compound").len().sort("len", descending=True))
print()

# RaceProgress range
print("=== numeric ranges ===")
for c in ["LapNumber", "Stint", "TyreLife", "Position", "LapTime (s)", "LapTime_Delta",
          "Cumulative_Degradation", "RaceProgress", "Position_Change", "PitStop"]:
    s = train[c]
    print(f"  {c:30s}  min={s.min():>12.4f}  max={s.max():>12.4f}  mean={s.mean():>10.4f}")
