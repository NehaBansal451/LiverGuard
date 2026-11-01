import json
import subprocess
import sys
from pathlib import Path

from liverguard.data.audit import run_audit, write_audit
from tests.synthetic import build_tree

REPO = Path(__file__).resolve().parents[1]


def _cli(*args):
    return subprocess.run([sys.executable, "scripts/verify_dataset.py", *map(str, args)],
                          cwd=REPO, capture_output=True, text=True)


def test_run_and_write_audit(tmp_path, make_cfg):
    root = tmp_path / "data"
    build_tree(root)
    result = run_audit(make_cfg(root), check_images=True, hash_images=True, rebuild_split=True)
    assert result.ok and len(result.reports) == 3
    paths = write_audit(result, tmp_path / "out")
    doc = json.loads(paths["json"].read_text())
    assert doc["ok"] and doc["split_id"] and "not a clinical" in doc["disclaimer"]
    assert len(paths["index"].read_text().splitlines()) == len(result.index.frame) + 1
    assert paths["rebuilt_split"].exists() and paths["image_stats"].exists()
    assert "dataset audit" in paths["markdown"].read_text()


def test_cli_exit_codes(tmp_path):
    clean, leaky = tmp_path / "clean", tmp_path / "leaky"
    build_tree(clean)
    build_tree(leaky, leak=True)
    ok = _cli("--data-root", clean, "--out-dir", tmp_path / "o1", "--check-images", "--hash")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert (tmp_path / "o1" / "dataset_report.md").exists()
    bad = _cli("--data-root", leaky, "--out-dir", tmp_path / "o2")
    assert bad.returncode == 1 and "patient_leakage_clean" in bad.stdout
    assert _cli("--data-root", clean, "--out-dir", tmp_path / "o3", "--strict").returncode == 1  # warnings fail
    assert _cli("--data-root", tmp_path / "missing", "--out-dir", tmp_path / "o4").returncode == 2
