#!/usr/bin/env python
"""Robustness audit of a trained checkpoint: distort, predict, compare with clean, plot.

By default audits the VALIDATION split (diagnosis; use it to choose M6 interventions). The test split needs --final-report.

Example:
  python scripts/audit_robustness.py --checkpoint artifacts/runs/effb0_s42/best.pt \\
      --index-csv artifacts/dataset/image_index.csv --out-dir artifacts/audits/effb0_s42_val
Exit codes: 0 ok, 2 could not run (bad input, split guard, missing file).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liverguard.data.report import json_default  # noqa: E402,F401
from liverguard.ml.data import LeakageError  # noqa: E402
from liverguard.robustness.config import load_robustness_config  # noqa: E402
from liverguard.robustness.engine import run_robustness_audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", help="Robustness YAML (default configs/robustness.yaml)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-report", action="store_true", help="Required to audit the TEST split")
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--split-file")
    ap.add_argument("--data-root")
    ap.add_argument("--threshold", default="0.5", help="number, or 'val_youden' (from the training manifest)")
    ap.add_argument("--conditions", help="comma-separated subset of condition names")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n-boot", type=int, help="bootstrap replicates (0 disables CIs)")
    ap.add_argument("--max-patients", type=int, help="seeded patient subsample for quick runs (not for final reporting)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--review-n", type=int, default=3, help="images in the gallery / expert review sheet")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = load_robustness_config(a.config, seed=a.seed, n_boot=a.n_boot)
        if a.conditions:
            cfg = cfg.select([c.strip() for c in a.conditions.split(",")])
        res = run_robustness_audit(a.checkpoint, a.index_csv, a.out_dir, cfg=cfg, split=a.split, variant=a.variant,
                                   threshold=a.threshold, batch_size=a.batch_size, device=a.device,
                                   final_report=a.final_report, split_file=a.split_file, data_root=a.data_root,
                                   max_patients=a.max_patients, review_n=a.review_n, make_plots=not a.no_plots)
    except (FileNotFoundError, ValueError, LeakageError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    m = res.manifest
    if m["watermark"]:
        print(f"*** {m['watermark']} ***")
    print(m["audit_scope"])
    c = m["clean"]
    print(f"clean: AUROC {c['auroc']:.3f}  acc {c['accuracy']:.3f}  F1 {c['f1']:.3f}  "
          f"(n={m['dataset']['n_images']} images / {m['dataset']['n_patients']} patients, threshold {m['threshold']['value']:.3f})")
    w = m["worst_case"]
    print(f"worst case, plausible tier: AUROC {w['plausible_only']['auroc']['value']:.3f}"
          f" ({w['plausible_only']['auroc']['condition']} L{w['plausible_only']['auroc']['level']}); "
          f"all levels: {w['all_levels']['auroc']['value']:.3f}")
    print("most disruptive (by mean AUROC change):")
    for r in m["top_disruptive_conditions"][:3]:
        print(f"  {r['rank']}. {r['condition']:<18} mean dAUROC {r['mean_d_auroc_all']:+.3f}  mean flip rate "
              f"{r['mean_pfr_all']:.3f}  median MFS {r['mfs_median_label']}")
    print(f"Outputs: {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
