"""Reverse-engineer Normalized_TyreLife from raw features."""
import polars as pl
import numpy as np

orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")

# Look at relationship between TyreLife and Normalized_TyreLife per Compound
print("=== Per-compound relationship ===")
for cmp in sorted(orig["Compound"].unique().to_list()):
    sub = orig.filter(pl.col("Compound") == cmp)
    print(f"\nCompound: {cmp} (n={len(sub)})")
    # NTL ≈ TyreLife / max_life_for_compound
    sub_with_ratio = sub.with_columns(
        (pl.col("TyreLife") / pl.col("Normalized_TyreLife")).alias("ratio")
    ).filter(pl.col("Normalized_TyreLife") > 0)
    if len(sub_with_ratio) > 0:
        ratios = sub_with_ratio["ratio"]
        print(f"  TyreLife / NTL: mean={ratios.mean():.3f}, std={ratios.std():.3f}, "
              f"min={ratios.min():.3f}, max={ratios.max():.3f}")

print()
print("=== Try: NTL == TyreLife / (some compound constant)? ===")
# Try fitting: NTL = TyreLife / k_compound, find best k
for cmp in sorted(orig["Compound"].unique().to_list()):
    sub = orig.filter(pl.col("Compound") == cmp).filter(pl.col("Normalized_TyreLife") > 0)
    if len(sub) == 0:
        continue
    tl = sub["TyreLife"].to_numpy()
    ntl = sub["Normalized_TyreLife"].to_numpy()
    # ntl = tl / k -> k = tl / ntl
    k = tl / ntl
    print(f"  {cmp}: k={k.mean():.3f} ± {k.std():.3f}")

print()
# Check: maybe NTL uses TyreLife_at_pit_for_session as denominator?
# Group by (Driver, Race, Year, Stint) - find max tyrelife in stint, divide
orig_keys = orig.with_columns(
    pl.col("TyreLife").max().over(["Driver", "Race", "Year", "Stint"]).alias("Stint_MaxTL"),
    pl.col("LapNumber").max().over(["Driver", "Race", "Year"]).alias("Race_MaxLap"),
)
ratio = orig_keys.select(
    (pl.col("Normalized_TyreLife") * pl.col("Stint_MaxTL") / pl.col("TyreLife")).alias("r")
).filter(pl.col("r").is_not_null() & pl.col("r").is_finite())
print(f"NTL * StintMaxTL / TyreLife: mean={ratio['r'].mean():.3f}, std={ratio['r'].std():.3f}")

# Maybe: NTL = TyreLife / some race-progression normalizer?
# Or: NTL = (TyreLife - 1) / (typical_pit_lap - 1)?
# Just try: NTL vs TyreLife scatterplot, check linearity
sample = orig.filter(pl.col("Compound") == "MEDIUM").head(50)
print()
print("=== sample MEDIUM rows: TyreLife vs Normalized_TyreLife ===")
for r in sample.iter_rows(named=True):
    print(f"  TL={r['TyreLife']:5.1f}  NTL={r['Normalized_TyreLife']:6.4f}  "
          f"ratio={r['Normalized_TyreLife']/r['TyreLife']:.4f}  "
          f"Driver={r['Driver']}  Race={r['Race']}  Stint={r['Stint']}")
