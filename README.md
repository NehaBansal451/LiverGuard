# LiverGuard: Auditing and Improving the Reliability of Liver Ultrasound AI

> **LiverGuard is a research and model-auditing platform. It is not a clinical diagnostic system and makes no clinical claims.**

Cycle: **BUILD → ATTACK → DIAGNOSE → MEASURE → FIX → RE-TEST** on the SMC-LUD liver ultrasound dataset (HCC vs Hemangioma).
Current status: **Milestones 1-2 (data foundation, baseline training pipeline) implemented and tested**; see `DEVELOPMENT_STATUS.md`. **No result on real data exists yet**: the only executed runs are watermarked synthetic smoke tests.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -r requirements.txt -r requirements-ml.txt
python -m pytest                                      # 62 tests, no real data needed (~1 min, CPU)
```

## Dataset
Download SMC-LUD yourself (Figshare 31112716 v3; paper: Tak et al., *Sci Data* 13:649, 2026) and unpack to `data/raw/SMC-LUD`
(or set `LIVERGUARD_DATA_ROOT`). **Licence: non-commercial research use only** (stricter than the CC-BY label on Figshare).
Never commit the images or bake them into a Docker image or public host. `data/` is git-ignored.

## Verify a download (first thing to run on real data)
```bash
python scripts/verify_dataset.py --data-root data/raw/SMC-LUD --check-images --hash
# outputs artifacts/dataset/{dataset_report.md,.json, image_index.csv, image_stats.csv}
```
Exit code 0 = no errors, 1 = errors (e.g. patient leakage), 2 = could not run. `--strict` also fails on warnings,
`--rebuild-split` additionally writes a patient-grouped split (never replaces the official one).

## Train a baseline (after the audit above has produced `image_index.csv`)
```bash
python scripts/train.py --config configs/baseline_efficientnet_b0.yaml \
    --index-csv artifacts/dataset/image_index.csv --out-dir artifacts/runs/effb0_s42
# offline / blocked download host:  add  --pretrained-weights-path <torchvision .pth>
# null control (must score ~0.5 AUROC; watermarked):  add  --label-control patient_shuffle
```
Trains on the official patient-level Clean split, selects the epoch on validation, evaluates the test split once, and writes
`manifest.json`, `predictions.csv`, `best.pt` and plots. Pretrained weights that cannot be loaded are an error, never a silent random start.

## Layout
`configs/` (dataset + training) · `liverguard/data/` (indexing, validation, splits, duplicates, audit) · `liverguard/ml/` (data, models, metrics, trainer, plots) ·
`scripts/` · `tests/` · docs: `PROJECT_PLAN`, `ARCHITECTURE`, `DEVELOPMENT_STATUS`, `RESEARCH_NOTES`, `EXPERIMENT_LOG`, `DECISIONS`, `HANDOFF`.
