"""Patient-level split verification and deterministic, patient-grouped split rebuilding."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from liverguard.data.validation import AuditIssue, AuditReport


def _stable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"image_id", "patient_key", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot create split manifest: missing columns {sorted(missing)}")
    cols = [c for c in ("image_id", "patient_key", "split", "variant", "label", "class_label") if c in frame]
    out = frame[cols].copy()
    for col in out.columns:
        out[col] = out[col].astype("string").fillna("")
    return out.sort_values(["patient_key", "image_id"], kind="mergesort").to_dict(orient="records")


def split_manifest_id(frame: pd.DataFrame) -> str:
    """Stable SHA-256 identity for the image-to-patient/split assignment."""
    payload = json.dumps(_stable_records(frame), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _push(rep: AuditReport, severity: str, code: str, message: str,
          count: int | None = None, examples: list[str] | None = None) -> None:
    rep.issues.append(AuditIssue(severity, code, message, count, examples or []))


def verify_patient_splits(frame: pd.DataFrame, cfg: Any) -> AuditReport:
    """Check within-variant leakage, cross-variant patient overlap, and documented Clean counts."""
    rep = AuditReport(title="Patient-level split verification", stats={})
    required = {"patient_key", "split", "class_label", "variant"}
    missing = required - set(frame.columns)
    if missing:
        _push(rep, "error", "missing_split_columns", f"Missing columns: {sorted(missing)}", len(missing))
        return rep

    valid = frame[frame["split"].notna()].copy()
    if valid.empty:
        _push(rep, "info", "no_split_information",
              "No official split assignments were found; patient-level splits must be supplied or rebuilt separately.")

    known_splits = {"train", "val", "test"}
    bad_names = sorted(set(valid["split"].astype(str)) - known_splits)
    if bad_names:
        _push(rep, "error", "invalid_split_names", f"Unknown split names: {bad_names}", len(bad_names))

    # First test: a patient may not straddle partitions within the same image variant.
    leakage_examples: list[str] = []
    leakage_count = 0
    for _, group in valid.groupby(["variant", "patient_key"], dropna=False):
        splits = sorted(set(group["split"].astype(str)))
        if len(splits) > 1:
            leakage_count += 1
            if len(leakage_examples) < 10:
                leakage_examples.append(f"{group['patient_key'].iloc[0]} [{'+'.join(splits)}]")
    if leakage_count:
        code = "patient_leakage_clean" if "clean" in set(valid["variant"].astype(str).str.lower()) else "patient_leakage"
        _push(rep, "error", code,
              f"{leakage_count} patient/variant assignments span multiple splits.",
              leakage_count, leakage_examples)

    # Second test: when Clean and Caliper both exist for a patient, they must use the same partition.
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
        _push(rep, "info", "no_patient_leakage",
              "No patient appears in multiple splits within or across variants.")

    per_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        part = valid[valid["split"] == split]
        per_split[split] = {"images": int(len(part)), "patients": int(part["patient_key"].nunique())}
    clean = valid[valid["variant"].astype(str).str.lower() == "clean"]
    for split in ("train", "val", "test"):
        part = clean[clean["split"] == split]
        per_split[split]["clean_images"] = int(len(part))
        for cls in ("HCC", "Hemangioma"):
            per_split[split][cls] = int((part["class_label"] == cls).sum())

    rep.stats.update({
        "has_official_split": bool(len(frame) and frame["split"].notna().all()),
        "per_split": per_split,
        "split_id": split_manifest_id(frame),
        "patients_with_both_variants": int(
            frame.groupby("patient_key")["variant"].nunique().ge(2).sum()
        ) if "variant" in frame else 0,
    })

    reference = getattr(cfg, "documented_reference", None)
    expected = getattr(reference, "clean_split_images", {}) if reference is not None else {}
    mismatch: list[str] = []
    for split, classes in (expected or {}).items():
        for cls, count in classes.items():
            observed = per_split.get(split, {}).get(cls)
            if observed is not None and observed != count:
                mismatch.append(f"{split}/{cls}: expected {count}, observed {observed}")
    if mismatch:
        _push(rep, "warning", "split_reference_mismatch",
              "Clean split image counts differ from the documented reference: " + "; ".join(mismatch),
              len(mismatch), mismatch[:10])
    elif expected:
        _push(rep, "info", "split_reference_match",
              "Clean split counts match the configured documented reference.")
    return rep


def build_patient_splits(frame: pd.DataFrame, cfg: Any, seed: int | None = None) -> pd.Series:
    """Build deterministic stratified patient-level splits; never mutate or replace official assignments."""
    required = {"patient_key", "class_label", "image_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot rebuild patient splits: missing columns {sorted(missing)}")

    label_col = "label" if "label" in frame.columns else "class_label"
    label_counts = frame.groupby("patient_key", dropna=False)[label_col].nunique(dropna=False)
    class_counts = frame.groupby("patient_key", dropna=False)["class_label"].nunique(dropna=False)
    ambiguous = label_counts[label_counts > 1].index.union(class_counts[class_counts > 1].index)
    if len(ambiguous):
        raise ValueError(
            f"Cannot rebuild: {len(ambiguous)} patient(s) have several class labels, "
            f"e.g. {list(ambiguous[:5])}"
        )

    patients = frame.drop_duplicates("patient_key")[["patient_key", "class_label"]].reset_index(drop=True)
    if patients["patient_key"].isna().any():
        raise ValueError("Cannot rebuild splits: missing patient_key values")
    counts = patients["class_label"].value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise ValueError("Patient-grouped stratification requires at least 10 patients in each of two classes.")

    seed = int(seed if seed is not None else getattr(cfg, "seed", 42))
    fractions = getattr(cfg, "split_fractions", {"train": .7, "val": .2, "test": .1})
    # Ten stratified folds give a reproducible 70/20/10 allocation for the documented SMC-LUD protocol.
    n_folds = 10
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    groups = patients["patient_key"].astype(str).to_numpy()
    labels = patients["class_label"].astype(str).to_numpy()
    fold_ids = np.full(len(patients), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(patients)), labels, groups)):
        fold_ids[test_idx] = fold

    n_train = max(1, int(round(float(fractions["train"]) * n_folds)))
    n_val = max(1, int(round(float(fractions["val"]) * n_folds)))
    if n_train + n_val >= n_folds:
        n_train = max(1, n_folds - n_val - 1)
    order = np.random.default_rng(seed).permutation(n_folds)
    fold_to_split = {
        int(fold): ("train" if rank < n_train else "val" if rank < n_train + n_val else "test")
        for rank, fold in enumerate(order)
    }
    patient_split = {p: fold_to_split[int(fold)] for p, fold in zip(groups, fold_ids)}
    return frame["patient_key"].astype(str).map(patient_split).rename("split")


def json_split(frame: pd.DataFrame) -> str:
    """Canonical compact representation useful for audit artifacts."""
    return split_manifest_id(frame)
