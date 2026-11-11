#!/usr/bin/env python
"""Grad-CAM, occlusion, ROI/background masking and Grad-CAM faithfulness experiments for a checkpoint.

Default = VALIDATION split; TEST needs --final-report. ROI = --mask-dir of user-supplied annotation masks (named after the image's relative path), else a
centred geometric PROXY (not the liver). Example:
  python scripts/explain.py --checkpoint artifacts/runs/effb0_s42/best.pt --index-csv artifacts/dataset/image_index.csv \\
      --out-dir artifacts/explain/effb0_s42_val --threshold val_youden
Exit codes: 0 ok, 2 could not run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liverguard.explain.engine import run_explainability  # noqa: E402
from liverguard.ml.data import LeakageError  # noqa: E402


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
    ap.add_argument("--threshold", default="0.5")
    ap.add_argument("--mask-dir", help="directory of binary ROI masks: mirrored tree <image_id>, <image_id>.png-suffixed, or flat with '/'->'__' (non-zero = ROI)")
    ap.add_argument("--proxy-area", type=float, default=0.4)
    ap.add_argument("--cam-fraction", type=float, default=0.2)
    ap.add_argument("--n-control", type=int, default=3)
    ap.add_argument("--fill", choices=["mean", "black"], default="mean")
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-patients", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    try:
        r = run_explainability(a.checkpoint, a.index_csv, a.out_dir, split=a.split, variant=a.variant, threshold=a.threshold,
                               device=a.device, final_report=a.final_report, split_file=a.split_file, data_root=a.data_root,
                               max_patients=a.max_patients, seed=a.seed, mask_dir=a.mask_dir, proxy_area=a.proxy_area,
                               cam_fraction=a.cam_fraction, n_control=a.n_control, fill=a.fill, n_cases=a.n_cases,
                               n_boot=a.n_boot, make_plots=not a.no_plots)
    except (FileNotFoundError, ValueError, LeakageError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    m = r.manifest
    if m["watermark"]:
        print(f"*** {m['watermark']} ***")
    print(m["scope"], "| ROI:", m["roi_source"])
    print(r.masking_summary[["variant", "n", "auroc", "accuracy", "flip_rate_vs_full"]].round(3).to_string(index=False))
    print(r.cam_faithfulness[["variant", "mean_evidence_drop", "lo", "hi"]].round(3).to_string(index=False))
    g = m["gradcam_randomization"]
    print(f"Grad-CAM vs randomised model: median Spearman {g['median_spearman']}, share > 0.5: {g['share_above_0.5']}")
    print(f"Outputs: {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
