# LiverGuard — Block 1 dataset finalization

**Status: READY WITH CAVEATS**

This is an aggregate summary of a dataset audit run by the project owner in Google Colab. It is not a clinical validation and does not establish diagnostic performance.

## Dataset inventory

| Measure | Observed |
|---|---:|
| Indexed images | 5,385 |
| Unique patients in manifest | 1,013 |
| HCC images | 2,716 |
| Hemangioma images | 2,669 |
| Clean images | 2,656 |
| Caliper images | 2,729 |

### Image split

| Split | Images | Patients |
|---|---:|---:|
| Train | 3,759 | 709 |
| Validation | 854 | 152 |
| Test | 772 | 152 |

The existing patient split was preserved. It was not rebuilt.

### Data integrity checks

- Image readability check: completed previously for all 5,385 indexed images.
- Image hashes computed: 5,385; hash errors: 0.
- Patient split leakage: 0 patients reported across partitions.
- Near-duplicate cross-split pairs at dHash Hamming distance <= 2: 0.
- Exact duplicate count: not included in the console summary captured for this publication; consult the private audit report.

## Patient-table reconciliation

| Class | Patient table IDs | Manifest IDs | Image-folder IDs |
|---|---:|---:|---:|
| HCC | 600 | 600 | 600 |
| Hemangioma | 421 | 413 | 413 |

HCC IDs matched across the table, manifest and folders.

For Hemangioma, nine patient-table IDs had no corresponding image folders, while one image-folder ID was absent from the patient table. This accounts for the net difference of eight between the expected table count and observed manifest/folder count. The discrepancy remains documented, not silently corrected.

## Split caveat

The observed Clean image counts by split were:

| Split | HCC | Hemangioma |
|---|---:|---:|
| Train | 1,003 | 913 |
| Validation | 185 | 201 |
| Test | 201 | 153 |

These differ from the repository's configured published-reference Clean counts. The existing split is retained as the project's current split and must not be represented as an exact reproduction of the published split without further provenance verification.

## Release decision

**READY WITH CAVEATS for baseline training.**

The project can proceed using the preserved split, provided the above discrepancies are disclosed in experiment manifests and the final report. The test split must remain untouched until the final evaluation protocol is frozen.

## Data handling

This repository entry contains aggregate counts only. No ultrasound images, patient IDs, image paths, per-image hashes, per-image predictions, checkpoints or detailed duplicate-pair files are included. Those artifacts remain in the project owner's private Google Drive.

LiverGuard is a research and model-auditing platform, not a clinical diagnostic system.
