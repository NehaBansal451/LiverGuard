import pandas as pd
from pathlib import Path

manifest = Path(
    "demo_workspace/dataset/"
    "patient_level_manifest_seed42_f9452f1f5b23.csv"
)

df = pd.read_csv(manifest)

print("PATIENT-LEVEL LEAKAGE AUDIT")
print("=" * 45)

# 1. Patient counts by split
print("\n1. PATIENTS BY SPLIT")
print(
    df.groupby("split")["patient_key"]
    .nunique()
    .to_string()
)

# 2. Variant counts by split
print("\n2. IMAGES BY VARIANT AND SPLIT")
print(
    pd.crosstab(df["variant"], df["split"])
    .to_string()
)

# 3. Check whether each patient's variants stay in one split
patient_splits = df.groupby("patient_key")["split"].nunique()
split_leakage = patient_splits[patient_splits > 1]

print("\n3. PATIENTS IN MULTIPLE SPLITS:", len(split_leakage))

# 4. Check patients with both Clean and Caliper variants
variants = df.groupby("patient_key")["variant"].agg(
    lambda x: set(x.dropna())
)

both_variants = variants[
    variants.apply(lambda x: {"clean", "caliper"}.issubset(x))
]

print("4. PATIENTS WITH BOTH VARIANTS:", len(both_variants))

# 5. Check variant split consistency
variant_split_errors = []

for patient, group in df.groupby("patient_key"):
    if group["variant"].nunique() > 1:
        if group["split"].nunique() > 1:
            variant_split_errors.append(patient)

print(
    "5. PATIENTS WHOSE VARIANTS CROSS SPLITS:",
    len(variant_split_errors)
)

print("\nAUDIT COMPLETE")
