import pandas as pd
from pathlib import Path
import os

ROOT = Path(os.environ.get("LIVERGUARD_DATA_ROOT", "data/original_data"))

tables = {
    "HCC": ROOT / "HCC" / "HCC_patient_table.csv",
    "Hemangioma": ROOT / "Hemangioma" / "Hem_patient_table.csv",
}

for class_name, csv_path in tables.items():
    print(f"\n{'=' * 50}")
    print(class_name.upper())
    print("=" * 50)

    if not csv_path.exists():
        print("ERROR: File not found:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    metadata_ids = {
        str(value).strip()
        for value in df["Subject_id"].dropna()
    }

    class_folder = ROOT / class_name

    folder_ids = {
        folder.name
        for folder in class_folder.rglob("*")
        if folder.is_dir()
        and folder.name not in {
            "HCC_clean (wo caliper)",
            "HCC_caliper",
            "Hem_clean (wo caliper)",
            "Hem_caliper",
        }
    }

    if class_name == "HCC":
        metadata_ids = {f"HCC_{value}" for value in metadata_ids}

    missing_folders = sorted(metadata_ids - folder_ids)
    missing_metadata = sorted(folder_ids - metadata_ids)

    print("CSV rows:", len(df))
    print("Unique metadata IDs:", len(metadata_ids))
    print("Actual patient folders:", len(folder_ids))
    print("Metadata IDs without folders:", len(missing_folders))
    print("Folders without metadata:", len(missing_metadata))

    if missing_folders:
        print("Metadata IDs without folders:", missing_folders)

    if missing_metadata:
        print("Folders without metadata:", missing_metadata)

print("\nMETADATA RECONCILIATION COMPLETE")
