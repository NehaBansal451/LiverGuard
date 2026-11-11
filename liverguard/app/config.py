"""Workspace configuration from environment variables (no hard-coded paths)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Workspace:
    root: Path
    index_csv: Path
    robustness_config: Path
    repo: Path = REPO_ROOT

    @property
    def db_path(self) -> Path:
        return self.root / "liverguard.db"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def dataset_dir(self) -> Path:
        return self.index_csv.parent

    def script(self, name: str) -> str:
        return str(self.repo / "scripts" / name)


def get_workspace() -> Workspace:
    """LIVERGUARD_WORKSPACE (default ./artifacts) holds runs, audits, reports, the registry DB and background-job logs.
    LIVERGUARD_INDEX_CSV (default <workspace>/dataset/image_index.csv) is the output of scripts/verify_dataset.py."""
    root = Path(os.environ.get("LIVERGUARD_WORKSPACE", "artifacts")).expanduser().resolve()
    idx = Path(os.environ.get("LIVERGUARD_INDEX_CSV", root / "dataset" / "image_index.csv")).expanduser().resolve()
    rc = Path(os.environ.get("LIVERGUARD_ROBUSTNESS_CONFIG", REPO_ROOT / "configs" / "robustness.yaml"))
    return Workspace(root, idx, rc)
