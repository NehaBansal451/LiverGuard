# ARCHITECTURE

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
(AUROC, ties -> val loss) -> restore checkpoint -> *only if `evaluate_test: true`* decode and evaluate *test* once (the Model Laboratory defaults this OFF) -> `manifest.json`, `predictions.csv`,
`best.pt` (state + meta; `results.test` is absent when test was not evaluated), curves / confusion matrix / ROC PNGs. `models.load_checkpoint` rebuilds a model for the later audit modules.
Non-result runs (smoke, null control) are watermarked and marked `valid_as_result: false`.

## M3 audit flow (`liverguard/robustness/`)
checkpoint (`load_checkpoint`) + `image_index.csv` -> same split guards as M2 (default **val**; test only with `--final-report`; split_id must match)
-> clean probabilities -> for each condition x level: `distortions.apply_distortion` (RNG keyed by seed+image+condition) -> predictions
-> `measures.summarize` (PFR, confidence drift, dAUROC/dAcc/dF1, MFS, worst case, ranking, paired patient-bootstrap CIs) + `fidelity` proxies
-> `plots` (degradation curves, heatmaps, ranking, first-flip CDF, gallery) -> `audit_manifest.json`, CSV tables, `review/` expert sheet.
Config: `configs/robustness.yaml`. CLI: `scripts/audit_robustness.py`.

## M4 reliability flow (`liverguard/reliability/`, CLI `scripts/reliability.py`)
`session.open_session` (guards) -> p(HCC) -> entropy, MC-Dropout (head), TTA (5 fixed views) -> `calibration` (ECE width/mass, MCE, Brier, NLL, temperature
scaling fitted on val, cross-fitted by patient on val) -> `selective` (risk-coverage, AURC vs oracle/random) -> `failure` (TP/TN/FP/FN, confident errors,
error-by-uncertainty, patient-level aggregation) -> CSVs, figures, `reliability_manifest.json` (limitations embedded).

## M5 explainability flow (`liverguard/explain/`, CLI `scripts/explain.py`)
`session.open_session` -> Grad-CAM (all images, predicted class; class-specific on selected cases) -> region masking (ROI-only / background-only with
matched-area controls; ROI = annotation masks or centre proxy) -> CAM deletion/insertion vs random -> occlusion on selected cases -> randomisation sanity
check -> CSVs, figures, `explain_manifest.json` (limitations embedded).

## M6 flow (`liverguard/improve/`, CLI `scripts/robust_train.py`)
`diagnose` (train baselines, val audit -> weaknesses -> plan) -> `train` (interventions x seeds; `RobustAugment` on top of basic augmentation) -> `validate`
(audit all on val, paired patient-bootstrap vs same-seed baseline, pre-declared selection frozen with checkpoint hashes) -> `final --final-report`
(same code on the test split; selection cannot change). Outputs: weakness_report.md, report.md, CSVs, trade-off / forest / ablation / degradation plots.

## M7 platform layer (`liverguard/db`, `liverguard/app`, `liverguard/reports`)
```
 scripts/*.py  (the only code that runs experiments)  ──writes──►  manifests + CSV + PNG on disk  ◄── SOURCE OF TRUTH
        ▲ subprocess (job runner: log file + exit-code file, cancel)                │ Registry.scan (idempotent, prunes deleted runs)
        │                                                                           ▼
 app/pages/*.render()  ◄── app/state.py (caching, selection widgets) ◄── db/registry.py (SQLAlchemy, SQLite: runs, jobs, reports)
        │                         ▲
        ├── app/ops.py (pure: model loading, distortion, Grad-CAM, occlusion, ROI, argv builders)   app/charts.py (pure DataFrame -> Plotly)
        └── reports/pdf.py (reportlab; first page and every section state when a run is not a valid result)
```
- **Pages** (`app/pages/`): `dashboard`, `dataset`, `model_lab`, `image_audit` (+ `audit_results` = dataset-level audit tab), `explain`, `reliability`, `robust_lab`, `compare`, `reports`. Entry: `app/app.py` (`st.navigation`).
- **The app never computes a reported metric.** Headline numbers are copied from manifests into the registry; per-image tools (single-image distortion, Grad-CAM, occlusion, ROI) compute live on one image and are labelled as illustrations.
- **Protocol guards in the UI** mirror the CLIs: test-split audit / reliability / explain / final are disabled until the final-report box is ticked; the robust-training final stage is disabled until `validation/selection.json` exists; run names are validated (no path traversal) and never overwrite a finished run.
- **Database**: three tables (`runs`, `jobs`, `reports`) in `<workspace>/liverguard.db`. Large artefacts (images, checkpoints, CSVs) stay on disk. Not implemented from the original plan: separate dataset/image/metric tables, MLflow, PostgreSQL.
- **Config/env**: `LIVERGUARD_WORKSPACE`, `LIVERGUARD_INDEX_CSV`, `LIVERGUARD_ROBUSTNESS_CONFIG`, `LIVERGUARD_DATA_ROOT`, `LIVERGUARD_ARTIFACTS_DIR`.

## Not built
Clean-vs-Caliper domain-shift experiment, caliper-removal control, Integrated Gradients / Captum, ROI-aware training, TTA/calibration as intervention arms in M6, out-of-distribution detection, cross-dataset generalisation, MLflow, DVC, FastAPI, PostgreSQL. See PROJECT_PLAN.md.
