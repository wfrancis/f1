import polars as pl

train = pl.read_csv("data/train.csv")
test = pl.read_csv("data/test.csv")
sub = pl.read_csv("data/sample_submission.csv")

print(f"train shape: {train.shape}")
print(f"test  shape: {test.shape}")
print(f"sub   shape: {sub.shape}")
print()
print("=== train columns ===")
for c, dt in zip(train.columns, train.dtypes):
    print(f"  {c:30s} {str(dt):15s}")
print()
print("=== test columns ===")
for c, dt in zip(test.columns, test.dtypes):
    print(f"  {c:30s} {str(dt):15s}")
print()
print("=== train head ===")
print(train.head(5))
print()
print("=== target balance (PitNextLap) ===")
print(train.group_by("PitNextLap").len().sort("PitNextLap"))
print()
print("=== nulls in train ===")
nulls = train.null_count()
for c in train.columns:
    n = nulls[c][0]
    if n > 0:
        print(f"  {c}: {n}")
print()
print("=== unique counts of likely-categorical cols ===")
for c in train.columns:
    if train[c].dtype == pl.String:
        print(f"  {c}: {train[c].n_unique()} unique")
