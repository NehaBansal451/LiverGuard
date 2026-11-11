# Metadata Reconciliation Report

## Purpose
Compare patient IDs in the supplied metadata tables with patient
folders in the local original_data dataset.

## Results

### HCC
- Metadata records: 600
- Unique metadata IDs: 600
- Patient folders: 600
- Metadata IDs without folders: 0
- Folders without metadata: 0
- Result: Complete ID match after normalizing the HCC_ folder prefix.

### Hemangioma
- Metadata records: 421
- Unique metadata IDs: 421
- Patient folders: 413
- Metadata IDs without folders: 9
- Folders without metadata: 1

### Unmatched Hemangioma records
Nine metadata IDs have no corresponding local patient folder.
One local patient folder has no corresponding metadata record.

These discrepancies are documented rather than silently corrected.
No source images or metadata records were modified.

## Limitations
This audit checks patient ID correspondence only. It does not establish
clinical validity, diagnostic accuracy, or completeness of the original
published dataset.

The source metadata tables and medical images are not included in this
repository.
