"""Patient-level split verification and deterministic, patient-grouped split rebuilding."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from liverguard.data.report import json_default
from liverguard.data.validation import AuditIssue, AuditReport


def _stable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"image_id", "patient_key", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot create split manifest: missing columns {sorted(missing)}")
    cols = [c for c in ("image_id", "patient_key", "split", "variant", "label", "class_label") if c in frame]
    out = frame[cols].copy()
    for c in out.columns:
        out[c] = out[c].astype("string").fillna("")
    return out.sort_values(["patient_key", "image_id"], kind="mergesort").to_dict(orient="records")


def split_manifest_id(frame: pd.DataFrame) -> str:
    """Stable SHA-256 identity for the image-to-patient/split assignment."""
    import json
    payload = json.dumps(_stable_records(frame), sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _push(rep: AuditReport, severity: str, code: str, message: str, count: int | None = None,
          examples: list[str] | None = None) -> None:
    rep.issues.append(AuditIssue(severity=severity, code=code, message=message, count=count, examples=examples or []))


def verify_patient_splits(frame: pd.DataFrame, cfg: Any) -> AuditReport:
    """Check patient separation independently within each image variant and compare reference image counts."""
    rep = AuditReport(title="Patient-level split verification", stats={})
    required = {"patient_key", "split", "class_label", "variant"}
    missing = required - set(frame.columns)
    if missing:
        _push(rep, "error", "missing_split_columns", f"Missing columns: {sorted(missing)}", len(missing))
        return rep

    valid = frame[frame["split"].notna()].copy()
    if valid.empty:
        _push(rep, "info", "no_split_information", "No official split assignments were found; patient-level splits must be supplied or rebuilt separately.")
    known_splits = {"train", "val", "test"}
    bad_names = sorted(set(valid["split"].astype(str)) - known_splits)
    if bad_names:
        _push(rep, "error", "invalid_split_names", f"Unknown split names: {bad_names}", len(bad_names))

    leakage_examples: list[str] = []
    leakage_count = 0
    for variant, group in valid.groupby("variant", dropna=False):
        for patient, pg in group.groupby("patient_key", dropna=False):
            splits = sorted(set(pg["split"].astype(str)))
            if len(splits) > 1:
                leakage_count += 1
                if len(leakage_examples) < 10:
                    leakage_examples.append(f"{patient} [{'+'.join(splits)}]")
    if leakage_count:
        _push(rep, "error", "patient_leakage_clean" if "clean" in set(valid["variant"]) else "patient_leakage",
              f"{leakage_count} patient/variant assignments span multiple splits", leakage_count, leakage_examples)

    # Cross-variant overlap is distinct from leakage within a single variant.
    cross: list[str] = []
    for patient, group in valid.groupby("patient_key", dropna=False):
        if group["variant"].nunique(dropna=False) > 1 and group["split"].nunique(dropna=False) > 1:
            if len(cross) < 10:
                cross.append(str(patient))
    if cross:
        _push(rep, "warning", "cross_variant_split_overlap",
              "A patient has Clean and Caliper images assigned to different splits; analyse variants separately or use a patient-complete split.",
              len(cross), cross)
    elif leakage_count == 0:
        _push(rep, "info", "no_patient_leakage", "No patient appears in multiple splits within or across variants.")

