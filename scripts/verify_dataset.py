#!/usr/bin/env python
"""Audit an SMC-LUD download: index, validate metadata, verify patient-level splits.

Exit code: 0 = no errors, 1 = errors found (or warnings with --strict), 2 = could not run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liverguard.data.audit import run_audit, write_audit  # noqa: E402
from liverguard.data.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Path to dataset.yaml (default: configs/dataset.yaml)")
    ap.add_argument("--data-root", help="Dataset root (overrides $LIVERGUARD_DATA_ROOT and YAML)")
    ap.add_argument("--metadata-csv", help="Metadata CSV (default: auto-detect)")
    ap.add_argument("--patient-scope", choices=["global", "per_class"])
    ap.add_argument("--out-dir", help="Output dir (default: $LIVERGUARD_ARTIFACTS_DIR/dataset)")
    ap.add_argument("--check-images", action="store_true", help="Open every image (corruption, sizes, modes)")
    ap.add_argument("--hash", action="store_true", help="Exact + perceptual duplicate analysis")
    ap.add_argument("--max-hamming", type=int, default=2, help="dHash distance for near-duplicates")
    ap.add_argument("--rebuild-split", action="store_true",
                    help="Also write a patient-grouped split (never replaces the official split)")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = ap.parse_args(argv)

    try:
        cfg = load_config(args.config, data_root=args.data_root, metadata_csv=args.metadata_csv,
                          patient_scope=args.patient_scope)
        result = run_audit(cfg, check_images=args.check_images, hash_images=args.hash,
                           rebuild_split=args.rebuild_split, max_hamming=args.max_hamming)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir or Path(os.environ.get("LIVERGUARD_ARTIFACTS_DIR", "artifacts")) / "dataset")
    paths = write_audit(result, out_dir)
    for rep in result.reports:
        c = rep.counts()
        print(f"[{'PASS' if rep.ok else 'FAIL'}] {rep.title}: {c['error']} error(s), {c['warning']} warning(s)")
        for issue in rep.issues:
            if issue.severity != "info":
                print(f"   {issue.severity.upper():7} {issue.code}: {issue.message}"
                      + (f" (n={issue.count})" if issue.count is not None else ""))
    print(f"\nReport: {paths['markdown']}\nIndex:  {paths['index']}")
    return 0 if result.ok and not (args.strict and result.n_warnings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
