"""Orchestrate dataset indexing, validation, patient split checks, and duplicate analysis."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from liverguard.data.duplicates import compute_hashes, duplicate_report
from liverguard.data.indexer import IndexResult, build_index
from liverguard.data.report import json_default
from liverguard.data.splits import build_patient_splits, split_manifest_id, verify_patient_splits
from liverguard.data.validation import AuditReport, inspect_images, validate_index


DISCLAIMER = "LiverGuard is a research and model-auditing platform, not a clinical diagnostic system."


@dataclass
class DatasetAuditResult:
    index: IndexResult
    reports: list[AuditReport]
    image_stats: pd.DataFrame
    rebuilt_split: pd.Series | None = None

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.reports)

    @property
    def n_warnings(self) -> int:
        return sum(r.counts()["warning"] for r in self.reports)

    @property
    def split_id(self) -> str:
        return split_manifest_id(self.index.frame)


def run_audit(cfg, *, check_images: bool = False, hash_images: bool = False,
              rebuild_split: bool = False, max_hamming: int = 2) -> DatasetAuditResult:
    """Run the three required dataset reports. Never changes or replaces official split assignments."""
    indexed = build_index(cfg)
    stats = inspect_images(indexed.frame) if check_images else pd.DataFrame(
        columns=["image_id", "path", "bytes", "width", "height", "mode", "format", "mean_intensity", "std_intensity", "error"]
    )
    validation = validate_index(indexed, cfg, stats if check_images else None)
    splits = verify_patient_splits(indexed.frame, cfg)
    if hash_images:
        hashes = compute_hashes(indexed.frame)
        duplicates = duplicate_report(indexed.frame, hashes, max_hamming=max_hamming)
    else:
        from liverguard.data.validation import AuditIssue
        duplicates = AuditReport(title="Duplicate and leakage analysis", stats={"n_hashed": 0})
        duplicates.issues.append(AuditIssue("info", "hash_skipped",
                                            "Hash-based duplicate analysis was not requested. Re-run with --hash."))
    rebuilt = build_patient_splits(indexed.frame, cfg) if rebuild_split else None
    return DatasetAuditResult(index=indexed, reports=[validation, splits, duplicates],
                              image_stats=stats, rebuilt_split=rebuilt)


def write_audit(result: DatasetAuditResult, out_dir: str | Path) -> dict[str, Path]:
    """Write the canonical index, human-readable/JSON report, image stats and optional rebuilt split."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "dataset_report.md",
        "json": out / "dataset_report.json",
        "index": out / "image_index.csv",
        "image_stats": out / "image_stats.csv",
        "rebuilt_split": out / "patient_grouped_split.csv",
    }
    result.index.frame.to_csv(paths["index"], index=False)
    result.image_stats.to_csv(paths["image_stats"], index=False)
    if result.rebuilt_split is not None:
        pd.DataFrame({"image_id": result.index.frame["image_id"], "split": result.rebuilt_split}).to_csv(
            paths["rebuilt_split"], index=False
        )
    else:
        # A stable artifact path is returned for callers, but do not create a misleading empty split file.
        paths.pop("rebuilt_split")

    report_obj: dict[str, Any] = {
        "ok": result.ok,
        "data_root": str(result.index.data_root) if result.index.data_root else "",
        "split_id": result.split_id,
        "disclaimer": DISCLAIMER,
        "reports": [r.to_dict() for r in result.reports],
        "known_documentation_discrepancies": [],
        "metadata_csv": str(result.index.metadata_csv) if result.index.metadata_csv else None,
        "index_conflicts": result.index.conflicts,
        "unknown_metadata_values": result.index.unknown_values,
        "unmatched_metadata_rows": int(len(result.index.csv_unmatched)),
    }
    # Recover the configured source notes where available without requiring them to be copied into every report.
    report_obj["known_documentation_discrepancies"] = result.index.documented_discrepancies
    paths["json"].write_text(json.dumps(report_obj, indent=2, default=json_default), encoding="utf-8")

    lines = ["# LiverGuard dataset audit", "", f"**Overall: {'PASS' if result.ok else 'FAIL'}**", "",
             DISCLAIMER, "", f"Split manifest ID: `{result.split_id}`", ""]
    for rep in result.reports:
        lines.extend([rep.to_markdown(), ""])
    if result.index.conflicts:
        lines.extend(["## Metadata/path conflicts", ""])
        for key, values in result.index.conflicts.items():
            lines.append(f"- **{key}:** {', '.join(values[:20])}")
    if result.index.unknown_values:
        lines.extend(["", "## Unmapped metadata values", ""])
        for key, values in result.index.unknown_values.items():
            lines.append(f"- **{key}:** {', '.join(values)}")
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
