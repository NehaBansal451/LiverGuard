# SETUP

LiverGuard is a **research and model-auditing tool, not a clinical system**. All instructions below were run on Linux, Python 3.13, CPU only, unless marked *untested*.

## 1. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu   # CPU torch; drop the extra index for CUDA wheels (untested)
python -m pytest tests -q          # expect 262 passed
```
`requirements.txt` lists what the code uses. timm, albumentations, OpenCV, Captum, MLflow, DVC and FastAPI from the original plan are **not** used and not installed.

## 2. Try it without data (synthetic)
```bash
bash scripts/demo_pipeline.sh demo_workspace
python scripts/run_app.py --workspace demo_workspace --index-csv demo_workspace/dataset/image_index.csv   # http://localhost:8501
```
The script writes a synthetic patient-level dataset, verifies it, trains a tiny CNN, audits robustness, computes reliability and explainability on the validation split, and runs robust-training diagnose/train/validate. It deliberately stops before the one-shot test stage; run `scripts/robust_train.py final ... --final-report` yourself if you want to see that stage (synthetic only). Everything is stamped NOT A RESULT.

## 3. Real data (SMC-LUD)
1. Download SMC-LUD (Figshare 31112716 v3) yourself. **Licence: non-commercial research use only.** Never commit it, copy it into an image, or host it publicly. Unpack to `data/raw/SMC-LUD` (or set `LIVERGUARD_DATA_ROOT`).
2. `python scripts/verify_dataset.py --data-root data/raw/SMC-LUD --check-images --hash` and read `artifacts/dataset/dataset_report.md`. Resolve every error (patient leakage, unreadable files, exact cross-split duplicates) before training. The layout assumptions (folder tokens, aliases in `configs/dataset.yaml`) were only checked against a synthetic tree because the real data was unreachable: **expect to adjust aliases**.
3. Pretrained weights: torchvision downloads ImageNet weights on first use. If the host is blocked, download the `.pth` elsewhere and pass `--pretrained-weights-path` (or set it in the Model Laboratory). A failed weight load is an error, never a silent random start. Real EfficientNet-B0 / ResNet-50 training has **not** been run here.
4. Suggested order: baseline train (`configs/baseline_*.yaml`, ideally 3 seeds) -> `audit_robustness.py` on **val** -> `reliability.py` / `explain.py` on val -> `robust_train.py all` -> decide the open selection-rule question (DECISIONS D42) -> `robust_train.py final --final-report` once. The test split must stay untouched until then: set `evaluate_test: false` in training configs while experimenting.

## 4. App configuration
| variable | default | meaning |
|---|---|---|
| `LIVERGUARD_WORKSPACE` | `./artifacts` | runs, jobs, reports, `liverguard.db` |
| `LIVERGUARD_INDEX_CSV` | `<workspace>/dataset/image_index.csv` | image index produced by `verify_dataset.py` |
| `LIVERGUARD_ROBUSTNESS_CONFIG` | `configs/robustness.yaml` | distortion conditions / levels / seed |
| `LIVERGUARD_DATA_ROOT` | `data/raw/SMC-LUD` | dataset root for the CLIs |

The database is created automatically (SQLite). It only indexes manifests: **Rescan workspace** in the sidebar rebuilds/prunes it; deleting `liverguard.db` loses nothing but job history and report history.

## 5. Docker (*untested*)
`Dockerfile`, `.dockerignore` and `docker-compose.yml` exist but **were never built**: no Docker daemon was available. Review them before use. The dataset is mounted read-only as a volume, never copied into the image.

## 6. Troubleshooting
- *App says no runs*: set `LIVERGUARD_WORKSPACE` to the folder containing `runs/`, then press Rescan.
- *Job shows failed*: open the job log in the page (tail shown) or `<workspace>/jobs/job_N.log`.
- *"split mismatch" / LeakageError*: the checkpoint was trained on a different split than the current index; use the index it was trained with.
- *Slow pages*: Grad-CAM/occlusion/audits run on CPU; use the CLI for whole-split analyses.
