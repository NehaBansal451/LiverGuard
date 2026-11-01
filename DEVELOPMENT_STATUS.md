# DEVELOPMENT_STATUS

Updated: 2026-11-01 · **M1 complete · M2 pipeline complete and tested; NO real results exist**

## What exists
- **M1 (data)**: config, indexer, validation, patient-level split verification, duplicate/leakage checks, `scripts/verify_dataset.py`.
- **M2 (baselines)**: `liverguard/ml/` - `config`, `data` (patient-safe loading + leakage guard + null-control shuffle), `models`
  (EfficientNet-B0, ResNet-50, TinyCNN for smoke), `metrics` (+ patient-level bootstrap), `trainer`, `evaluate`, `plots`;
  `scripts/train.py`; configs `baseline_efficientnet_b0.yaml`, `baseline_resnet50.yaml`, `smoke_tiny.yaml`.
- Tests: **62 passing** (33 M1 + 29 M2), ~1 min on 2 CPUs. Includes: hand-computed metrics, leakage refusal (within-variant and pooled-variant),
  "test images are not decoded before training ends", seed reproducibility, shuffled-label null control, both real architectures end-to-end, CLI exit codes.

## Executed experiments
Only **labelled smoke runs on synthetic data** (EXPERIMENT_LOG). They prove the pipeline runs and the null control collapses to chance.
They are not findings about liver ultrasound.

## Bugs / design issues found and fixed in M2
1. EfficientNet smoke run: AUROC 1.0 but sensitivity 0 at threshold 0.5, with the checkpoint chosen at the first AUROC=1.0 epoch → added validation-loss tie-break and a validation-fitted second operating point.
2. Run watermark showed only one reason when a run was both smoke and null-control → now lists both.
3. Plot title overlapped a panel title → fixed.

## Process incident (disclosed)
An earlier M2 draft was already in the workspace. Before noticing, I overwrote its `config.py`, `models.py`, `metrics.py` and `__init__.py`
(no VCS existed). They were reconstructed from the interfaces the surviving files require (DECISIONS D11). **Initialise git before M3.**

## Blocked / not verified
- **Real SMC-LUD data not available** here → real-data audit, real baselines and CIs not done.
- **ImageNet weights not downloadable here** (403) → transfer learning is implemented and unit-tested via local weights, but has **never been run with real pretrained weights**.
- Not done by design: robustness engine, explainability, calibration/uncertainty, database, frontend.
- Known limitations: BatchNorm statistics keep updating while the backbone is frozen (train mode); no resume-from-checkpoint; single seed per run (multi-seed driver not yet written);
  images are resized to a square, which distorts aspect ratio; patient-level (rather than image-level) aggregation of predictions is not yet reported.
