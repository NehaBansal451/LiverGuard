#!/usr/bin/env python
"""Write a SYNTHETIC patient-level dataset (blob images with a planted class signal) to exercise the whole pipeline without SMC-LUD.

Everything trained on it is a pipeline check, never a result. Usage: python scripts/make_synthetic_demo.py OUT_DIR [--n-per-class 40]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.synthetic import build_tree  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--n-per-class", type=int, default=40)
a = ap.parse_args()
build_tree(Path(a.out_dir), n_per_class=a.n_per_class, signal=True)
print(f"synthetic dataset written to {a.out_dir} (patients H*/M*, 70/20/10 patient-level split, Clean + Caliper variants). NOT medical data.")
