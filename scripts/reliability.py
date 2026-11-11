#!/usr/bin/env python
"""Uncertainty, calibration (ECE, Brier, temperature scaling), risk-coverage and failure-case analysis for a checkpoint.

Default = VALIDATION split (diagnostic; calibrated numbers are cross-fitted). TEST needs --final-report (T is then fitted on val).
Example:
  python scripts/reliability.py --checkpoint artifacts/runs/effb0_s42/best.pt --index-csv artifacts/dataset/image_index.csv \\
      --out-dir artifacts/reliability/effb0_s42_val --threshold val_youden
Exit codes: 0 ok, 2 could not run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liverguard.ml.data import LeakageError  # noqa: E402
from liverguard.reliability.engine import run_reliability_analysis  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-report", action="store_true")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--split-file")
    ap.add_argument("--data-root")
    ap.add_argument("--threshold", default="0.5", help="number or 'val_youden'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--confident-at", type=float, default=0.9)
    ap.add_argument("--max-patients", type=int)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    try:
        r = run_reliability_analysis(a.checkpoint, a.index_csv, a.out_dir, split=a.split, variant=a.variant,
                                     threshold=a.threshold, device=a.device, final_report=a.final_report,
                                     split_file=a.split_file, data_root=a.data_root, max_patients=a.max_patients, seed=a.seed,
                                     mc_samples=a.mc_samples, n_bins=a.n_bins, n_boot=a.n_boot, confident_at=a.confident_at,
                                     batch_size=a.batch_size, make_plots=not a.no_plots)
    except (FileNotFoundError, ValueError, LeakageError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    m = r.manifest
    if m["watermark"]:
        print(f"*** {m['watermark']} ***")
    print(m["scope"], f"| n={m['dataset']['n_images']} images / {m['dataset']['n_patients']} patients")
    for c in m["calibration"]:
        print(f"  {c['probabilities']:<48} ECE {c['ece_width']:.3f}  Brier {c['brier']:.3f}  NLL {c['nll']:.3f}")
    print(f"outcomes: {m['outcomes']}  confident errors: {m['n_confident_errors']}")
    print(f"Outputs: {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
