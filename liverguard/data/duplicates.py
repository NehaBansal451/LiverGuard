"""Exact and perceptual duplicate detection for dataset integrity audits."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from liverguard.data.validation import AuditIssue, AuditReport


@dataclass
class DuplicateReport(AuditReport):
    pass


def dhash(path: str | Path) -> str:
    """64-bit difference hash encoded as 16 lowercase hex characters."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def popcount(values: np.ndarray) -> np.ndarray:
    """Population count for uint64 arrays, implemented portably across NumPy versions."""
    arr = np.asarray(values, dtype=np.uint64)
    return np.fromiter((int(v).bit_count() for v in arr.flat), dtype=np.uint8, count=arr.size).reshape(arr.shape)


def near_duplicate_pairs(hashes: np.ndarray, max_hamming: int = 2, *, max_pairs: int = 100_000) -> tuple[list[tuple[int, int, int]], bool]:
    """Return (left_index, right_index, Hamming distance) pairs within threshold and a truncation flag."""
    values = np.asarray(hashes, dtype=np.uint64).reshape(-1)
    found: list[tuple[int, int, int]] = []
    truncated = False
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            dist = (int(values[i]) ^ int(values[j])).bit_count()
            if dist <= max_hamming:
                if len(found) >= max_pairs:
                    truncated = True
                    return found, truncated
                found.append((i, j, dist))
    return found, truncated


def compute_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute SHA-256 and dHash for each path; unreadable files are represented with an error."""
    rows = []
    for r in frame.itertuples(index=False):
        p = Path(r.path)
        try:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            perceptual = int(dhash(p), 16)
            error = ""
        except (OSError, ValueError) as exc:
            sha, perceptual, error = "", 0, f"{type(exc).__name__}: {exc}"
        rows.append({"image_id": r.image_id, "sha256": sha, "dhash": np.uint64(perceptual), "error": error})
    return pd.DataFrame(rows)


def duplicate_report(frame: pd.DataFrame, hashes: pd.DataFrame, max_hamming: int = 2) -> DuplicateReport:
    """Flag exact cross-split/label conflicts as errors and near-duplicate cross-split pairs as warnings."""
    rep = DuplicateReport(title="Duplicate and leakage analysis", stats={"n_hashed": int(len(hashes))})
    merged = frame.merge(hashes, on="image_id", how="left")
    exact = merged[merged["sha256"].fillna("").ne("")]
    exact_cross_split: list[str] = []
    exact_cross_patient: list[str] = []
    label_conflicts: list[str] = []
    for sha, group in exact.groupby("sha256"):
        if len(group) < 2:
            continue
        splits = sorted(set(group["split"].dropna().astype(str)))
        patients = sorted(set(group["patient_key"].dropna().astype(str)))
        labels = sorted(set(group["class_label"].dropna().astype(str)))
        example = ", ".join(group["image_id"].astype(str).head(4))
        if len(splits) > 1:
            exact_cross_split.append(example)
        if len(patients) > 1:
            exact_cross_patient.append(example)
        if len(labels) > 1:
            label_conflicts.append(example)
    if exact_cross_split:
        rep.issues.append(AuditIssue("error", "duplicate_cross_split",
                                     "Byte-identical images occur in different splits.", len(exact_cross_split), exact_cross_split[:10]))
    if exact_cross_patient:
        rep.issues.append(AuditIssue("error", "duplicate_cross_patient",
                                     "Byte-identical images are assigned to different patients.", len(exact_cross_patient), exact_cross_patient[:10]))
    if label_conflicts:
        rep.issues.append(AuditIssue("error", "duplicate_label_conflict",
                                     "Byte-identical images have conflicting class labels.", len(label_conflicts), label_conflicts[:10]))

    # Near-duplicate search is pairwise; cap it to avoid unbounded memory on very large datasets.
    valid = merged[merged["error"].fillna("").eq("")].reset_index(drop=True)
    raw_hashes = np.array([int(x, 16) for x in valid["dhash"].map(lambda x: f"{int(x):016x}")], dtype=np.uint64)
    pairs, truncated = near_duplicate_pairs(raw_hashes, max_hamming=max_hamming)
    near_examples: list[str] = []
    for i, j, dist in pairs:
        a, b = valid.iloc[i], valid.iloc[j]
        if pd.notna(a["split"]) and pd.notna(b["split"]) and a["split"] != b["split"]:
            near_examples.append(f"{a['image_id']} <> {b['image_id']} (d={dist})")
    if near_examples:
        rep.issues.append(AuditIssue("warning", "near_duplicate_cross_split",
                                     "Perceptually similar images occur in different splits; inspect manually.",
                                     len(near_examples), near_examples[:10]))
    if truncated:
        rep.issues.append(AuditIssue("warning", "near_duplicate_search_truncated",
                                     "Near-duplicate search reached its pair cap; results may be incomplete."))
    if not any(i.severity in {"warning", "error"} for i in rep.issues):
        rep.issues.append(AuditIssue("info", "no_duplicates", "No exact cross-split or label-conflicting duplicates were found."))
    return rep
