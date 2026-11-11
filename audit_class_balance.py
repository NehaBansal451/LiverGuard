import pandas as pd
from pathlib import Path

manifest = Path(
    "demo_workspace/dataset/"
    "patient_level_manifest_seed42_f9452f1f5b23.csv"
)

df = pd.read_csv(manifest)

print("CLASS BALANCE AUDIT")
print("=" * 45)

print("\n1. IMAGE COUNTS BY CLASS AND SPLIT")
print(
    pd.crosstab(
        df["class_label"],
        df["split"],
        margins=True
    ).to_string()
)

print("\n2. CLASS PERCENTAGES WITHIN EACH SPLIT")
for split in ["train", "val", "test"]:
    subset = df[df["split"] == split]
    print(f"\n{split.upper()} ({len(subset)} images)")
    print(
        (subset["class_label"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )

print("\n3. IMAGE DISTRIBUTION ACROSS SPLITS")
print(
    (df["split"].value_counts(normalize=True) * 100)
    .round(2)
    .to_string()
)

print("\nAUDIT COMPLETE")