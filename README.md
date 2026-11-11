# LiverGuard: Auditing and Improving the Reliability of Liver Ultrasound AI

> **LiverGuard is a research and model-auditing platform. It is not a clinical diagnostic system and makes no clinical claims.**

Cycle: **BUILD → ATTACK → DIAGNOSE → MEASURE → FIX → RE-TEST** on the SMC-LUD liver ultrasound dataset (HCC vs Hemangioma).
Status (see `AUDIT_REPORT.md` for exactly what was verified): data tooling, baseline training, robustness audit, reliability, explainability, robust-training ablation and a nine-page Streamlit app are implemented and tested **on synthetic data only**. **No result on real SMC-LUD data exists**: the dataset and ImageNet weights were not reachable from the build environment, so every number the software has produced so far is a watermarked pipeline check, not a finding.

Documents: [SETUP.md](SETUP.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [LIMITATIONS.md](LIMITATIONS.md) · [DEMO_GUIDE.md](DEMO_GUIDE.md) · [AUDIT_REPORT.md](AUDIT_REPORT.md) · [DECISIONS.md](DECISIONS.md) · [RESEARCH_NOTES.md](RESEARCH_NOTES.md)

## Quick start (no dataset needed)
```bash
python -m venv .venv && source .venv/bin/activate         # Python 3.13 was used for all testing; >=3.10 expected to work (untested)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest tests -q                                  # 262 tests; ~10 min on 2 CPU cores
bash scripts/demo_pipeline.sh demo_workspace               # synthetic end-to-end run, ~3 min (watermarked NOT A RESULT)
python scripts/run_app.py --workspace demo_workspace --index-csv demo_workspace/dataset/image_index.csv
```
Full instructions, real-data workflow and troubleshooting: [SETUP.md](SETUP.md). Guided tour of the app: [DEMO_GUIDE.md](DEMO_GUIDE.md).

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
Trains on the official patient-level Clean split, selects the epoch on validation, and (only when `evaluate_test: true`, which the baseline YAMLs set and the app's Model Laboratory leaves OFF) evaluates the test split once. Writes
`manifest.json`, `predictions.csv`, `best.pt` and plots. Pretrained weights that cannot be loaded are an error, never a silent random start.

## Audit a trained model's robustness (M3)
```bash
python scripts/audit_robustness.py --checkpoint artifacts/runs/effb0_s42/best.pt \
    --index-csv artifacts/dataset/image_index.csv --out-dir artifacts/audits/effb0_s42_val   # validation split by default
# options: --threshold val_youden  --conditions brightness_dark,jpeg  --n-boot 500  --seed 7 ; test split needs --final-report
```
12 conditions (brightness, contrast, Gaussian/speckle noise, blur, rotation, crop, gamma, JPEG) x 5 levels; writes CSV tables, figures,
`audit_manifest.json` and an expert label-preservation review sheet. Config: `configs/robustness.yaml`.

## Reliability and explainability (M4/M5)
```bash
python scripts/reliability.py --checkpoint artifacts/runs/effb0_s42/best.pt --index-csv artifacts/dataset/image_index.csv \
    --out-dir artifacts/reliability/effb0_s42_val --threshold val_youden   # ECE, Brier, entropy, MC-Dropout, TTA, risk-coverage, failure cases
python scripts/explain.py --checkpoint artifacts/runs/effb0_s42/best.pt --index-csv artifacts/dataset/image_index.csv \
    --out-dir artifacts/explain/effb0_s42_val [--mask-dir <annotation masks>]  # Grad-CAM, occlusion, ROI/background masking
```
Both default to the validation split (test needs `--final-report`). Without `--mask-dir` the ROI is a centred geometric proxy, **not** the liver. Limitations of each method: `RESEARCH_NOTES.md`.

## Targeted robust training and re-audit (M6)
```bash
python scripts/robust_train.py all --workdir artifacts/m6 --config configs/baseline_efficientnet_b0.yaml \
    --index-csv artifacts/dataset/image_index.csv --seeds 42,43,44      # diagnose -> train interventions -> validation ablation + frozen selection
python scripts/robust_train.py final --workdir artifacts/m6 ... --final-report   # ONE test re-audit, only after the selection is frozen
```

## Run the app (M7)
```bash
pip install -r requirements-ml.txt
python scripts/run_app.py --workspace artifacts --index-csv artifacts/dataset/image_index.csv   # or: streamlit run liverguard/app/app.py
```
Env vars: `LIVERGUARD_WORKSPACE` (default `./artifacts`; runs, jobs, reports and `liverguard.db` live here), `LIVERGUARD_INDEX_CSV` (default `<workspace>/dataset/image_index.csv`), `LIVERGUARD_ROBUSTNESS_CONFIG`.
Nine pages: Dashboard, Dataset Explorer, Model Laboratory, Image Audit Studio, Explainability Studio, Reliability Center, Robust Training Lab, Model Comparison, Research Reports. Every number comes from a manifest/CSV written by the ML modules; runs that are smoke tests or null controls are marked NOT A RESULT everywhere. Without data the pages show empty states, not placeholders.

## Layout
`configs/` (dataset + training) · `liverguard/data/` (indexing, validation, splits, duplicates, audit) · `liverguard/ml/` (data, models, metrics, trainer, plots) · `liverguard/robustness/` (distortions, measures, engine, plots) · `liverguard/reliability/` · `liverguard/explain/` ·
`liverguard/{db,app,reports}/` (registry, Streamlit app, PDF reports) · `scripts/` · `tests/` · docs: `PROJECT_PLAN`, `ARCHITECTURE`, `DEVELOPMENT_STATUS`, `RESEARCH_NOTES`, `EXPERIMENT_LOG`, `DECISIONS`, `HANDOFF`.
