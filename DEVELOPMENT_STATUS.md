# DEVELOPMENT_STATUS

Updated: 2026-11-05 · **M1-M4, M6, M7 (app) code complete, M5 mostly complete; 262 tests pass; NO real results exist**

## What exists
M1 data, M2 baselines, M3 robustness audit, M4 reliability, M5 explainability (see ARCHITECTURE).
- **M6** (`liverguard/improve/`, `scripts/robust_train.py`): weakness diagnosis from validation audits, 9 named interventions (brightness, contrast, gamma, photometric, noise, blur, jpeg,
  geometric, combined) as plausible-tier training augmentation, multi-seed training, validation ablation with paired patient-bootstrap CIs and a pre-declared selection rule,
  checkpoint-hash-frozen selection, one test re-audit; compares clean performance, plausible/stress robustness, flip rate (stability), calibration (ECE/Brier), adaptation vs transfer.
- **M7** (`liverguard/app`, `liverguard/db`, `liverguard/reports`, `scripts/run_app.py`): 9-page Streamlit app (Dashboard, Dataset Explorer, Model Lab, Image Audit Studio, Explainability Studio, Reliability Center, Robust Training Lab, Model Comparison, Research Reports), SQLAlchemy registry indexing manifests, background job runner (logs, cancel, exit codes), PDF reports (reportlab), design system, Plotly charts. Test-split actions locked behind the final-report checkbox like the CLIs.
- Tests: **262 passing** (229 + 33 app tests: registry/jobs, ops, charts, PDF, AppTest workflows for all pages incl. empty workspace, upload-free image audit, Grad-CAM/occlusion/ROI, report generation, guard locks). App workspace for tests is produced by the real CLIs through the job runner on synthetic data.

## Executed experiments
Only watermarked synthetic smoke runs (EXPERIMENT_LOG: M2-M6). The M6 smoke run validated the protocol and exposed a confound (gain in plausible AUROC = better clean model, not less degradation).

## Bugs / issues found (M6)
No "real weakness" can be reported: the toy baselines are weak, so 11/12 conditions were flagged. Selection rule can reward a better clean model (D42, open decision).

## Not done / not verified
- Real data and ImageNet weights unavailable -> no real weakness diagnosis, no real improvement claim, test set only exercised synthetically (30 images).
- ROI-aware training (no annotations), TTA and temperature scaling as intervention arms, Clean-vs-Caliper analysis, caliper-removal control, Integrated Gradients.
- Not built: MLflow, DVC, Docker/Compose, FastAPI (not needed), PostgreSQL option, Clean-vs-Caliper page analysis beyond the Dataset Explorer tab.
- App verified only on the synthetic smoke workspace (headless server + Playwright screenshots of all 9 pages: no exceptions; visually checked dashboard, reliability, image audit, comparison, explainability top). Not checked: narrow/mobile layouts, long-running real training from the UI, very large datasets (image browsing/stat sampling is capped).
- Known smells: `ops.severity_sweep` unused tuple element; `charts.tradeoff_scatter` watermark arg; datetime.utcnow deprecation warnings in the registry.

## Independent audit (2026-11-06)
See AUDIT_REPORT.md: 12 defects fixed (train.py crash without test eval, test-peeking default in UI, missing dataset-level audit UI, registry ghosts/cancel race, run-name traversal, UI flags), full suite 262 passed, synthetic end-to-end pipeline run to completion, Docker untested. New docs: SETUP.md, LIMITATIONS.md, DEMO_GUIDE.md, AUDIT_REPORT.md; new scripts: demo_pipeline.sh, make_synthetic_demo.py, configs/demo_tiny.yaml. Open: D42, evaluate_test default in YAMLs, Docker.
