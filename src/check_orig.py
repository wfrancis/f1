"""Check structure of the original F1 strategy dataset and how to merge it."""
import polars as pl

orig = pl.read_csv("data/orig/f1_strategy_dataset_v4.csv")
print(f"orig shape: {orig.shape}")
print()
print("=== orig columns ===")
for c, dt in zip(orig.columns, orig.dtypes):
    print(f"  {c:35s}  {str(dt)}")
print()
print("=== orig head ===")
print(orig.head(5))
print()

# Compare with our train
train = pl.read_csv("data/train.csv")
print("=== train columns (for reference) ===")
print(train.columns)
print()

# Check key overlap
common_cols = set(orig.columns) & set(train.columns)
print(f"=== common columns between orig and our train: {len(common_cols)} ===")
print(sorted(common_cols))
print()
extra_in_orig = set(orig.columns) - set(train.columns)
print(f"=== columns in orig but NOT in our train: {len(extra_in_orig)} ===")
print(sorted(extra_in_orig))
