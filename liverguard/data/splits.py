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
                    leakage_examples.append(f"{patient} [{'+'.join(splits)}; variant={variant}]")
    if leakage_count:
        _push(rep, "error", "patient_leakage_clean" if "clean" in set(valid["variant"]) else "patient_leakage",
              f"{leakage_count} patient/variant assignments span multiple splits", leakage_count, leakage_examples)
    else:
        _push(rep, "info", "no_patient_leakage", "No patient appears in multiple splits within the same variant.")

    # Clean and Caliper images from the same patient must share a patient-level partition.
    patient_variant = valid.groupby(["patient_key", "variant"])["split"].agg(lambda s: set(s.astype(str)))
    cross: list[str] = []
    for patient in valid["patient_key"].dropna().unique():
        rows = patient_variant.loc[patient] if patient in patient_variant.index.get_level_values(0) else None
        if rows is not None:
            all_splits = set().union(*rows.tolist())
            if len(all_splits) > 1 and len(cross) < 10:
                cross.append(str(patient))
    if cross:
        _push(rep, "warning", "cross_variant_split_overlap",
              "A patient has Clean and Caliper images assigned to different splits; analyse variants separately or use a patient-complete split.",
              len(cross), cross)

    per_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        part = valid[valid["split"] == split]
        per_split[split] = {
            "images": int(len(part)),
            "patients": int(part["patient_key"].nunique()),
        }
    clean = valid[valid["variant"].astype(str).str.lower() == "clean"]
    for split in ("train", "val", "test"):
        part = clean[clean["split"] == split]
        per_split[split]["clean_images"] = int(len(part))
        for cls in ("HCC", "Hemangioma"):
            per_split[split][cls] = int((part["class_label"] == cls).sum())

    stats = {
        "has_official_split": bool(valid["split"].notna().all() and len(valid)),
        "per_split": per_split,
        "split_id": split_manifest_id(frame),
        "patients_with_both_variants": int(frame.groupby("patient_key")["variant"].nunique().ge(2).sum())
            if "variant" in frame else 0,
    }
    rep.stats.update(stats)

    reference = getattr(cfg, "documented_reference", None)
    expected = getattr(reference, "clean_split_images", {}) if reference is not None else {}
    mismatch: list[str] = []
    if expected:
        for split, classes in expected.items():
            for cls, count in classes.items():
                observed = per_split.get(split, {}).get(cls)
                if observed is not None and observed != count:
                    mismatch.append(f"{split}/{cls}: expected {count}, observed {observed}")
    if mismatch:
        _push(rep, "warning", "split_reference_mismatch",
              "Clean split image counts differ from the documented reference: " + "; ".join(mismatch),
              len(mismatch), mismatch[:10])
    elif expected:
        _push(rep, "info", "split_reference_match", "Clean split counts match the documented reference.")

    return rep


def build_patient_splits(frame: pd.DataFrame, cfg: Any, seed: int | None = None) -> pd.Series:
    """Build a deterministic stratified patient-group split. Never mutates the input or writes over official splits."""
    required = {"patient_key", "class_label", "image_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot rebuild patient splits: missing columns {sorted(missing)}")

    patients = frame[["patient_key", "class_label"]].drop_duplicates()
    nlabels = patients.groupby("patient_key")["class_label"].nunique()
    ambiguous = nlabels[nlabels > 1]
    if len(ambiguous):
        raise ValueError(f"Cannot rebuild: {len(ambiguous)} patient(s) have several class labels, e.g. {ambiguous.index[:5].tolist()}")
    patients = patients.drop_duplicates("patient_key").reset_index(drop=True)
    if patients["patient_key"].isna().any():
        raise ValueError("Cannot rebuild splits: missing patient_key values")
    counts = patients["class_label"].value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise ValueError("Patient-grouped stratification requires at least 10 patients in each of two classes.")

    seed = int(seed if seed is not None else getattr(cfg, "seed", 42))
    frac = getattr(cfg, "split_fractions", {"train": .7, "val": .2, "test": .1})
    # SGKF provides group-safe stratification; a 10-fold partition gives a close 70/20/10 allocation.
    n_splits = 10
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = patients["patient_key"].astype(str).to_numpy()
    y = patients["class_label"].astype(str).to_numpy()
    fold_ids = np.full(len(patients), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(patients)), y, groups)):
        fold_ids[test_idx] = fold
    # Pick 7 folds train, 2 val, 1 test; deterministic fold mapping.
    order = np.random.default_rng(seed).permutation(n_splits)
    mapping = {int(f): ("train" if rank < 7 else "val" if rank < 9 else "test")
               for rank, f in enumerate(order)}
    patient_split = {p: mapping[int(f)] for p, f in zip(groups, fold_ids)}
    return frame["patient_key"].astype(str).map(patient_split).rename("split")


def json_split(frame: pd.DataFrame) -> str:
    """Canonical compact representation useful for audit artifacts."""
    return split_manifest_id(frame)
