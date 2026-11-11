#!/usr/bin/env bash
# End-to-end pipeline on SYNTHETIC data (CPU, a few minutes). Every output is watermarked NOT A RESULT.
# Usage: bash scripts/demo_pipeline.sh [WORKDIR]     then:  python scripts/run_app.py --workspace WORKDIR --index-csv WORKDIR/dataset/image_index.csv
set -euo pipefail
W=${1:-demo_workspace}; cd "$(dirname "$0")/.."; PY=${PYTHON:-python}
$PY scripts/make_synthetic_demo.py "$W/raw" --n-per-class 40
$PY scripts/verify_dataset.py --data-root "$W/raw" --out-dir "$W/dataset" --check-images --hash
IDX="$W/dataset/image_index.csv"
$PY scripts/train.py --config configs/demo_tiny.yaml --index-csv "$IDX" --out-dir "$W/runs/demo_tiny"
CK="$W/runs/demo_tiny/best.pt"
$PY scripts/audit_robustness.py --checkpoint "$CK" --index-csv "$W/dataset/image_index.csv" --out-dir "$W/audit/smoke_val" --split val --n-boot 50
$PY scripts/reliability.py --checkpoint "$CK" --index-csv "$IDX" --out-dir "$W/reliability/smoke_val" --split val --n-boot 50 --mc-samples 10
$PY scripts/explain.py --checkpoint "$CK" --index-csv "$IDX" --out-dir "$W/explain/smoke_val" --split val --n-boot 50
$PY scripts/robust_train.py all --workdir "$W/m6" --config configs/demo_tiny.yaml --index-csv "$IDX" --seeds 42,43,44 --n-boot 50
echo "pipeline finished (through validation selection). Final test stage is deliberately NOT run here."
