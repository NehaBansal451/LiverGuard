import pandas as pd
from pathlib import Path

import os

ROOT = Path(
    os.environ.get(
        "LIVERGUARD_DATA_ROOT",
        "data/original_data"
    )
)

tables = {
    "HCC": ROOT / "HCC_patient_table.csv",
    "Hemangioma": ROOT / "Hem_patient_table.csv",
}

for class_name, csv_path in tables.items():

    print(f"\n{'=' * 50}")
    print(class_name.upper())
    print("=" * 50)

    df = pd.read_csv(csv_path)

    print("CSV rows:", len(df))
    print("Columns:", df.columns.tolist())

    id_column = "Subject_id"
    metadata_ids = set(df[id_column].dropna().astype(str).str.strip())

    class_folder = ROOT / class_name
    folder_ids = {
        folder.name
        for folder in class_folder.iterdir()
        if folder.is_dir()
    }

    missing_folders = sorted(metadata_ids - folder_ids)
    missing_metadata = sorted(folder_ids - metadata_ids)

    print("Unique metadata IDs:", len(metadata_ids))
    print("Actual patient folders:", len(folder_ids))
    print("Metadata IDs without folders:", len(missing_folders))
    print("Folders without metadata:", len(missing_metadata))

    if missing_folders:
        print("\nMetadata IDs without folders:")
        print(missing_folders)

    if missing_metadata:
        print("\nFolders without metadata:")
        print(missing_metadata)

print("\nMETADATA RECONCILIATION COMPLETE")