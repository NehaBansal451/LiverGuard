"""Dataset index validation, image inspection, and structured audit reports."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    count: int | None = None
    examples: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    title: str
    stats: dict[str, Any] = field(default_factory=dict)
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(x.severity == "error" for x in self.issues)

    def get(self, code: str) -> AuditIssue | None:
        return next((x for x in self.issues if x.code == code), None)

    def codes(self, severity: str | None = None) -> set[str]:
        return {x.code for x in self.issues if severity is None or x.severity == severity}

    def counts(self) -> dict[str, int]:
        return {level: sum(x.severity == level for x in self.issues) for level in ("error", "warning", "info")}

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "ok": self.ok, "stats": self.stats,
                "issues": [asdict(x) for x in self.issues]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [f"## {self.title}: {status}", "", "### Statistics", ""]
        if self.stats:
            lines.extend([f"- **{k}:** {v}" for k, v in self.stats.items()])
        else:
            lines.append("- No statistics available.")
        lines.extend(["", "### Findings", ""])
        if not self.issues:
            lines.append("- No findings.")
        for issue in self.issues:
            lines.append(f"- **{issue.severity.upper()} · {issue.code}:** {issue.message}"
                         + (f" (n={issue.count})" if issue.count is not None else ""))
            for example in issue.examples[:10]:
                lines.append(f"  - {example}")
        return "\n".join(lines) + "\n"


def inspect_images(frame: pd.DataFrame) -> pd.DataFrame:
    """Open all indexed images and collect dimensions/intensity summaries; unreadable files are recorded."""
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        path = Path(row.path)
        base = {"image_id": row.image_id, "path": str(path), "bytes": path.stat().st_size if path.exists() else 0}
        try:
            with Image.open(path) as im:
                im.load()
                arr = np.asarray(im.convert("L"), dtype=np.float32)
                rows.append({**base, "width": im.width, "height": im.height, "mode": im.mode, "format": im.format,
                             "mean_intensity": float(arr.mean()), "std_intensity": float(arr.std()), "error": ""})
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            rows.append({**base, "width": None, "height": None, "mode": None, "format": None,
                         "mean_intensity": None, "std_intensity": None, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def validate_index(index, cfg, image_stats: pd.DataFrame | None = None) -> AuditReport:
    """Check index completeness, metadata reconciliation, patient identity and documented dataset counts."""
    frame = index.frame.copy()
    rep = AuditReport(title="Dataset index validation", stats={})
    required = {"image_id", "path", "image_filename", "patient_id", "patient_key", "class_label",
                "label", "split", "variant", "has_metadata"}
    missing = sorted(required - set(frame.columns))
    if missing:
        rep.issues.append(AuditIssue("error", "missing_index_columns", f"Index lacks columns: {missing}", len(missing)))
    if frame["image_id"].duplicated().any():
        n = int(frame["image_id"].duplicated().sum())
        rep.issues.append(AuditIssue("error", "duplicate_image_ids", "image_id must be unique.", n,
                                     frame.loc[frame["image_id"].duplicated(), "image_id"].head(10).tolist()))
    for col, code in (("patient_id", "missing_patient_id"), ("class_label", "missing_class_label")):
        if col in frame:
            mask = frame[col].isna() | frame[col].astype(str).str.strip().eq("")
            if mask.any():
                rep.issues.append(AuditIssue("error", code, f"{int(mask.sum())} image(s) have no {col}.",
                                             int(mask.sum()), frame.loc[mask, "image_id"].head(10).tolist()))
    if index.metadata_csv is None:
        rep.issues.append(AuditIssue("warning", "no_metadata_csv",
                                     "No metadata CSV was found; values were inferred from directory names where possible."))
    if getattr(index, "csv_missing_columns", []):
        missing_csv = index.csv_missing_columns
        rep.issues.append(AuditIssue("error", "csv_missing_columns",
                                     f"Metadata CSV lacks required columns: {missing_csv}", len(missing_csv), missing_csv))
    if index.conflicts:
        n = sum(len(v) for v in index.conflicts.values())
        rep.issues.append(AuditIssue("warning", "metadata_path_conflicts",
                                     "Metadata values override path-derived values; inspect these disagreements.", n,
                                     [f"{k}: {v[0]}" for k, v in index.conflicts.items() if v]))
    if index.unknown_values:
        vals = [f"{k}: {v}" for k, v in index.unknown_values.items()]
        rep.issues.append(AuditIssue("warning", "unknown_metadata_values",
                                     "Metadata contains values that could not be mapped to canonical labels/splits/variants.",
                                     sum(len(v) for v in index.unknown_values.values()), vals[:10]))
    unmatched = getattr(index, "csv_unmatched", pd.DataFrame())
    if unmatched is not None and len(unmatched):
        rep.issues.append(AuditIssue("warning", "csv_unmatched_rows",
                                     "Metadata rows did not match an image file.", len(unmatched),
                                     unmatched.astype(str).head(10).to_dict(orient="records")))
    if "has_metadata" in frame:
        without = ~frame["has_metadata"].fillna(False).astype(bool)
        if without.any() and index.metadata_csv is not None:
            rep.issues.append(AuditIssue("warning", "files_without_metadata",
                                         "Image files exist but have no matching metadata row.", int(without.sum()),
                                         frame.loc[without, "image_id"].head(10).tolist()))
    if {"patient_id", "class_label"} <= set(frame.columns):
        valid = frame.dropna(subset=["patient_id", "class_label"])
        multi = valid.groupby("patient_id")["class_label"].nunique()
        multi = multi[multi > 1]
        if len(multi):
            rep.issues.append(AuditIssue("warning", "patient_multiple_classes",
                                         "Some patient IDs occur under multiple classes; confirm whether IDs are global or per-class.",
                                         len(multi), multi.index.astype(str).tolist()[:10]))

    if image_stats is not None and "error" in image_stats:
        bad = image_stats[image_stats["error"].fillna("").astype(str).ne("")]
        if len(bad):
            rep.issues.append(AuditIssue("error", "unreadable_images",
                                         "Some image files cannot be decoded.", len(bad), bad["image_id"].head(10).tolist()))
        if "width" in image_stats and "height" in image_stats:
            readable = image_stats.dropna(subset=["width", "height"])
            rep.stats["image_dimensions"] = readable.groupby(["width", "height"]).size().to_dict()
            rep.stats["mean_image_width"] = float(readable["width"].mean()) if len(readable) else None
            rep.stats["mean_image_height"] = float(readable["height"].mean()) if len(readable) else None

    n_images = int(len(frame))
    n_patients = int(frame["patient_key"].nunique()) if "patient_key" in frame else 0
    by_class = frame["class_label"].value_counts(dropna=False).to_dict() if "class_label" in frame else {}
    by_variant = frame["variant"].value_counts(dropna=False).to_dict() if "variant" in frame else {}
    rep.stats.update({"n_images": n_images, "n_patients": n_patients,
                      "patients_by_class": (frame.dropna(subset=["patient_key"]).drop_duplicates("patient_key")
                                            .groupby("class_label")["patient_key"].nunique().to_dict()
                                            if {"patient_key", "class_label"} <= set(frame.columns) else {}),
                      "images_by_class": by_class, "images_by_variant": by_variant})

    ref = cfg.documented_reference
    mismatches: list[str] = []
    if ref.total_images is not None and n_images != ref.total_images:
        mismatches.append(f"images: expected {ref.total_images}, observed {n_images}")
    for cls, expected in ref.images_by_class.items():
        got = int((frame["class_label"] == cls).sum())
        if got != expected:
            mismatches.append(f"images[{cls}]: expected {expected}, observed {got}")
    if ref.total_patients is not None and n_patients != ref.total_patients:
        mismatches.append(f"patients: expected {ref.total_patients}, observed {n_patients}")
    if mismatches:
        rep.issues.append(AuditIssue("warning", "reference_mismatch",
                                     "Observed counts differ from the documented reference (references are checks, not labels): "
                                     + "; ".join(mismatches), len(mismatches), mismatches))
    elif ref.total_images is not None or ref.images_by_class:
        rep.issues.append(AuditIssue("info", "reference_match", "Observed counts match the configured documented reference."))
    if not any(x.severity in {"warning", "error"} for x in rep.issues):
        rep.issues.append(AuditIssue("info", "index_valid", "No validation problems were detected."))
    return rep
