#!/usr/bin/env python
"""Launch the LiverGuard app:  python scripts/run_app.py [--workspace DIR] [--index-csv CSV] [--port 8501]"""
import argparse, os, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--workspace"); ap.add_argument("--index-csv"); ap.add_argument("--port", default="8501")
a = ap.parse_args()
env = dict(os.environ)
if a.workspace: env["LIVERGUARD_WORKSPACE"] = a.workspace
if a.index_csv: env["LIVERGUARD_INDEX_CSV"] = a.index_csv
app = Path(__file__).resolve().parents[1] / "liverguard" / "app" / "app.py"
sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", a.port, "--server.headless", "true"], env=env))
