"""Layout-tolerant SMC-LUD image indexing with metadata reconciliation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from liverguard.data.config import DatasetConfig


@dataclass
class IndexResult:
    frame: pd.DataFrame
    metadata_csv: Path | None
    conflicts: dict[str, list[str]] = field(default_factory=dict)
    unknown_values: dict[str, list[str]] = field(default_factory=dict)
    csv_unmatched: pd.DataFrame = field(default_factory=pd.DataFrame)
    csv_missing_columns: list[str] = field(default_factory=list)
    data_root: Path | None = None
    documented_discrepancies: list[str] = field(default_factory=list)


def _read_metadata(cfg: DatasetConfig) -> tuple[Path | None, pd.DataFrame | None]:
    candidate = cfg.metadata_csv
    if candidate is not None:
        p = candidate if candidate.is_absolute() else cfg.data_root / candidate
        if not p.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {p}")
        return p, pd.read_csv(p, dtype=str, keep_default_na=False)
    if not cfg.data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {cfg.data_root}")
    matches: list[tuple[Path, pd.DataFrame]] = []
    for p in sorted(cfg.data_root.glob("*.csv")):
        df = pd.read_csv(p, nrows=5, dtype=str, keep_default_na=False)
        if any(cfg.map_column(c) == "image_filename" for c in df.columns):
            matches.append((p, pd.read_csv(p, dtype=str, keep_default_na=False)))
    if len(matches) > 1:
        raise ValueError(f"Multiple metadata CSV files found; set metadata_csv explicitly: {[str(x[0]) for x in matches]}")
    return matches[0] if matches else (None, None)


def _canonical_metadata(df: pd.DataFrame | None, cfg: DatasetConfig) -> tuple[pd.DataFrame | None, list[str]]:
    if df is None:
        return None, []
    renames = {c: cfg.map_column(c) for c in df.columns if cfg.map_column(c)}
    out = df.rename(columns=renames).copy()
    required = ["patient_id", "image_filename", "class_label"]
    missing = [c for c in required if c not in out.columns]
    return out, missing


def _tokens(path: Path, root: Path) -> tuple[str | None, str | None, str | None, str | None]:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    cls = split = variant = None
    for p in parts[:-1]:
        cls = cls or p if p in {"HCC", "Hemangioma"} else cls
    # Canonical mapping is applied by caller for aliases; token detection is done there.
    return rel.as_posix(), cls, split, variant


def build_index(cfg: DatasetConfig) -> IndexResult:
    """Create one index row per image. CSV values override path inference; disagreements are retained."""
    root = cfg.data_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    metadata_path, raw_meta = _read_metadata(cfg)
    meta, missing_cols = _canonical_metadata(raw_meta, cfg)
    records: list[dict[str, Any]] = []
    conflicts: dict[str, list[str]] = {}
    unknown: dict[str, set[str]] = {}
    used_meta: set[int] = set()

    meta_lookup: dict[tuple[str, str], list[int]] = {}
    basename_lookup: dict[str, list[int]] = {}
    if meta is not None:
        for i, row in meta.iterrows():
            fn = str(row.get("image_filename", "")).replace("\\", "/").split("/")[-1]
            pid = str(row.get("patient_id", "")).strip()
            meta_lookup.setdefault((pid, fn), []).append(i)
            basename_lookup.setdefault(fn, []).append(i)

    for image in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in cfg.image_extensions):
        rel = image.relative_to(root).as_posix()
        parts = image.relative_to(root).parts[:-1]
        path_class = next((cfg.map_class(p) for p in parts if cfg.map_class(p)), None)
        path_split = next((cfg.map_split(p) for p in parts if cfg.map_split(p)), None)
        path_variant = next((cfg.map_variant(p) for p in parts if cfg.map_variant(p)), None)
        # In both supported layouts, patient ID is the image's parent directory when a class token exists.
        path_patient = image.parent.name if path_class and image.parent != root else None
        candidates: list[int] = []
        if meta is not None:
            candidates = meta_lookup.get((str(path_patient or ""), image.name), [])
            if not candidates and len(basename_lookup.get(image.name, [])) == 1:
                candidates = basename_lookup[image.name]
        mi = next((i for i in candidates if i not in used_meta), None)
        mrow = meta.loc[mi] if mi is not None else None
        if mi is not None:
            used_meta.add(mi)

        def resolved(field: str, path_value: Any, mapper):
            csv_value = mapper(mrow.get(field)) if mrow is not None and field in mrow else None
            if csv_value and path_value and csv_value != path_value:
                conflicts.setdefault(field, []).append(rel)
            value = csv_value or path_value
            raw = mrow.get(field) if mrow is not None and field in mrow else None
            if raw is not None and str(raw).strip() and mapper(raw) is None:
                unknown.setdefault(field, set()).add(str(raw))
            return value

        cls = resolved("class_label", path_class, cfg.map_class)
        split = resolved("split", path_split, cfg.map_split)
        variant = resolved("variant", path_variant, cfg.map_variant)
        patient = str(mrow.get("patient_id")).strip() if mrow is not None and "patient_id" in mrow and str(mrow.get("patient_id")).strip() else path_patient
        patient_key = (f"{cls}::{patient}" if cfg.patient_scope == "per_class" and cls and patient else patient)
        row: dict[str, Any] = {
            "image_id": rel, "path": str(image), "image_filename": image.name,
            "patient_id": patient, "patient_key": patient_key, "class_label": cls,
            "label": (cfg.classes.index(cls) if cls in cfg.classes else pd.NA),
            "split": split, "variant": variant, "has_metadata": mi is not None,
        }
        if mrow is not None:
            for col, val in mrow.items():
                if col not in row and col not in {"image_filename", "patient_id"}:
                    row[col] = val
        records.append(row)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No images with extensions {cfg.image_extensions} found under {root}")
    frame["label"] = pd.array(frame["label"], dtype="Int64")
    unmatched = pd.DataFrame()
    if meta is not None:
        unmatched = meta.loc[[i for i in meta.index if i not in used_meta]].copy()
    return IndexResult(frame=frame, metadata_csv=metadata_path, conflicts=conflicts,
                       unknown_values={k: sorted(v) for k, v in unknown.items()},
                       csv_unmatched=unmatched, csv_missing_columns=missing_cols, data_root=root,
                       documented_discrepancies=list(cfg.documented_reference.known_discrepancies))
