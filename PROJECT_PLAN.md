# PROJECT_PLAN: LiverGuard roadmap

Scope is fixed: HCC vs Hemangioma, robustness audit, explainability, reliability, targeted improvement and re-audit. Python-only, free/open-source.

| # | Milestone | Key deliverables | Status |
|---|-----------|------------------|--------|
| M1 | Data foundation | config, indexer, metadata validation, patient-level split verification, duplicate/leakage checks, CLI, tests | **Done** (synthetic-tested; real data not yet seen) |
| M2 | Baselines (**code done + tested; real-data runs pending**) | torch/timm env, EfficientNet-B0 + ResNet-50 training on the official Clean split, metrics (AUROC, sens/spec, F1), seeds, checkpoints, JSON run manifests, patient-level cluster-bootstrap CIs, multi-seed runs | Code done; blocked on data/weights |
| M3 | Robustness audit engine | 9 distortions x severities, flip rate, confidence drift, min-flip severity, degradation curves, label-preservation check, plausible-vs-stress-test tiers | **Code done + tested (165 tests); only synthetic-checkpoint runs; real-data audit pending** |
| M4 | Reliability | ECE, Brier, reliability diagrams, entropy, MC-Dropout, TTA uncertainty, risk-coverage, temperature scaling (fit on val only), failure cases | **Code done + tested; only synthetic-checkpoint runs** |
| M5 | Explainability and shortcuts | Grad-CAM, occlusion, background-only / ROI masking, Clean-vs-Caliper analysis, caliper-removal control | **Partly done**: Grad-CAM, occlusion, ROI/background masking, CAM faithfulness, randomisation check (tested; synthetic runs only). **Not done**: Clean-vs-Caliper analysis, caliper-removal control, Integrated Gradients |
| M6 | Improvement and re-audit | targeted augmentations from M3 findings, ablations, calibration, TTA; same untouched test set; clean-vs-robust trade-off | **Code done + tested; run only on synthetic smoke models.** Not done: ROI-aware training (no masks), TTA/calibration as intervention arms |
| M7 | Platform | SQLAlchemy/SQLite registry (ingests M2+ manifests), Streamlit 9-page app + design system, PDF reports (MLflow, Docker deferred) | done (verified on synthetic workspace only) |
| M8 | Reports and release | PDF reports, leaderboard, Docker, docs, final QA | |

Optional (only after M6): OOD detection, failure-case explorer, image-quality assessment, cross-dataset tests (needs approval: new data).
