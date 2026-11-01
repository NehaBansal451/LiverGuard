# ARCHITECTURE (M1 state)

```
configs/dataset.yaml ──► data/config.py (pydantic, aliases, documented reference)
data_root/ + metadata CSV ─► data/indexer.py ─► IndexResult.frame  ◄── contract for all later modules
                                   │
          ┌────────────────────────┼─────────────────────────┐
   data/validation.py        data/splits.py            data/duplicates.py
   (labels, CSV, files,      (leakage, split stats,    (SHA-256 + dHash,
    published counts)         split_id, rebuild)        cross-split / label conflicts)
          └──────────── data/audit.py (run_audit / write_audit) ◄── scripts/verify_dataset.py
```

## `image_index.csv` contract (one row per image)
`image_id` (posix path relative to data_root, unique) · `path` · `image_filename` · `patient_id` · `patient_key`
(= patient_id, or `class::id` when `patient_scope: per_class`; **always group by this**) · `class_label` (`HCC`/`Hemangioma`) ·
`label` (Hemangioma=0, HCC=1, Int64) · `split` (`train`/`val`/`test`, may be empty) · `variant` (`clean`/`caliper`, may be empty) ·
`has_metadata` · plus any extra CSV columns (e.g. tumour size, UICC/AJCC stage; HCC only).

## Rules enforced
- Path tokens give class/split/variant; patient = parent directory; CSV overrides and disagreements are reported, never hidden.
- Published counts are comparison references (warnings), never ground truth.
- Official split is preserved; rebuilt splits are separate files, grouped by patient first.
- Errors: leakage within a variant, missing class/patient, unreadable images, exact cross-split or label-conflicting duplicates.

## M2 training flow (`liverguard/ml/`)
`image_index.csv` -> `data.load_split_frame` (variant filter, optional rebuilt split, **LeakageError** on any patient in >1 split)
-> `UltrasoundDataset` (resize to square, cache) -> `trainer.run_experiment`: train on *train*, select best epoch on *val*
(AUROC, ties -> val loss) -> restore checkpoint -> *only then* decode and evaluate *test* once -> `manifest.json`, `predictions.csv`,
`best.pt` (state + meta), curves / confusion matrix / ROC PNGs. `models.load_checkpoint` rebuilds a model for the later audit modules.
Non-result runs (smoke, null control) are watermarked and marked `valid_as_result: false`.

## Planned (not built): `robustness/`, `explain/`, `reliability/`, `improve/`, `db/`, `app/`, `reports/`. See PROJECT_PLAN.md.
