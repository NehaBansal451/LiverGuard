# HANDOFF (end of Milestone 2)

## Read first
`DEVELOPMENT_STATUS.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md` (D11-D19), `RESEARCH_NOTES.md`, `ARCHITECTURE.md`.

## Status
M1 + M2 code complete, 62 tests pass. **No real data and no pretrained weights were available, so no real result exists.**
The only executed runs are watermarked synthetic smoke tests.

## Next exact task (needs you, on a machine with internet access)
1. Download SMC-LUD (non-commercial research use) and run
   `python scripts/verify_dataset.py --data-root <path> --check-images --hash` → fix any layout/CSV mismatch; keep `dataset_report.md`.
2. Pretrained weights: on a machine with normal internet torchvision downloads them automatically on first use. If the run is offline, download the two torchvision ImageNet files once elsewhere (the ResNet-50 V2 file is named `resnet50-11ad3fa6.pth`; use the file name torchvision prints for EfficientNet-B0) and pass them with `--pretrained-weights-path`.
3. Train: `python scripts/train.py --config configs/baseline_efficientnet_b0.yaml --index-csv artifacts/dataset/image_index.csv --out-dir artifacts/runs/effb0_s42`
   (add `--pretrained-weights-path <file>` if offline; repeat for resnet50 and seeds 42/43/44). CPU is slow for 224 px; a GPU is strongly preferred.
4. Record E-001... in EXPERIMENT_LOG from each `manifest.json`; run one `--label-control patient_shuffle` null control per architecture on the real data.
Then start **M3 (robustness audit engine)** - it consumes `liverguard.ml.models.load_checkpoint` and `liverguard.ml.data.preprocess_pil`.

## Before more coding
`git init` and commit (no history exists; see process incident in DEVELOPMENT_STATUS).

## Resume
```bash
cd liverguard && python -m pytest -q      # expect 62 passed
python scripts/train.py --help
```
