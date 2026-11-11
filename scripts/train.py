#!/usr/bin/env python
"""Train a baseline classifier from an M1 image_index.csv and evaluate it on the held-out test split.

Example:
  python scripts/train.py --config configs/baseline_efficientnet_b0.yaml \\
      --index-csv artifacts/dataset/image_index.csv --out-dir artifacts/runs/effb0_seed42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liverguard.ml.config import load_train_config  # noqa: E402
from liverguard.ml.data import LeakageError  # noqa: E402
from liverguard.ml.trainer import run_experiment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-file", help="Alternative split (image_id, split); default = split in index")
    ap.add_argument("--data-root", help="Re-root image paths if the data moved since indexing")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device")
    ap.add_argument("--label-control", choices=["none", "patient_shuffle"])
    ap.add_argument("--pretrained-weights-path")
    args = ap.parse_args(argv)
    try:
        cfg = load_train_config(args.config, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                                device=args.device, label_control=args.label_control,
                                pretrained_weights_path=args.pretrained_weights_path)
        m = run_experiment(cfg, args.index_csv, args.out_dir, split_file=args.split_file, data_root=args.data_root)
    except (FileNotFoundError, ValueError, LeakageError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if m["watermark"]:
        print(f"*** {m['watermark']} ***")
    split = "test" if "test" in m["results"] else "val"
    t = m["results"][split]
    ci = t["ci95_patient_bootstrap"]["auroc"]
    print(f"{m['model']['arch']} {split}: AUROC {t['auroc']:.3f} (95% CI {ci[0]:.3f}-{ci[1]:.3f})  "
          f"sens {t['sensitivity']:.3f}  spec {t['specificity']:.3f}  F1 {t['f1']:.3f}  n={t['n']}")
    if split == "test":
        v = m["results"]["test_at_val_youden"]
        print(f"  at validation-fitted threshold {m['results']['val_youden_threshold']:.3f}: "
              f"sens {v['sensitivity']:.3f}  spec {v['specificity']:.3f}  F1 {v['f1']:.3f}")
    else:
        print("  test split NOT evaluated (evaluate_test=false); it is reserved for the final re-audit.")
    print(f"Run dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
