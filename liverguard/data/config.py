"""Typed dataset configuration and metadata aliases for LiverGuard."""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "na", "n/a"}:
        return None
    return re.sub(r"[-\\s]+", "_", text)


_DEFAULT_DISCREPANCIES = [
    "Figshare abstract text says 5,466 images; Figshare Table 1 and the paper say 5,385 (81 images unexplained).",
    "Figshare Table 1 lists 422 hemangioma patients; Table 2 and the 1,021 total imply 421.",
    "HCC mean height differs between the Figshare table and the published paper.",
    "Figshare license metadata and the paper's non-commercial research restriction need conservative handling.",
    "The reported representative-image method and HCC image counts per patient require reconciliation.",
    "The cited Figshare record identifiers differ across the source documents.",
    "A per-image Caliper split is not documented in the same detail as the Clean split.",
    "Patients with both Clean and Caliper variants must remain in one partition for cross-variant experiments.",
]


@dataclass
class DocumentedReference:
    source: str = ""
    total_images: int | None = None
    total_patients: int | None = None
    images_by_class: dict[str, int] = field(default_factory=dict)
    patients_by_class: dict[str, int] = field(default_factory=dict)
    images_by_variant_class: dict[str, dict[str, int]] = field(default_factory=dict)
    clean_split_images: dict[str, dict[str, int]] = field(default_factory=dict)
    known_discrepancies: list[str] = field(default_factory=lambda: list(_DEFAULT_DISCREPANCIES))


@dataclass
class DatasetConfig:
    name: str = "SMC-LUD"
    data_root: Path = Path("data")
    metadata_csv: Path | None = None
    seed: int = 42
    patient_scope: str = "global"
    image_extensions: list[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])
    classes: list[str] = field(default_factory=lambda: ["Hemangioma", "HCC"])
    positive_class: str = "HCC"
    split_fractions: dict[str, float] = field(default_factory=lambda: {"train": .7, "val": .2, "test": .1})
    class_aliases: dict[str, list[str]] = field(default_factory=lambda: {
        "HCC": ["hcc", "hepatocellular_carcinoma", "hepatocellular carcinoma"],
        "Hemangioma": ["hemangioma", "haemangioma"],
    })
    split_aliases: dict[str, list[str]] = field(default_factory=lambda: {
        "train": ["train_set", "training"], "val": ["val_set", "valid", "validation"],
        "test": ["test_set"],
    })
    variant_aliases: dict[str, list[str]] = field(default_factory=lambda: {
        "clean": ["clean", "hcc_clean_(wo_caliper)", "hem_clean_(wo_caliper)"],
        "caliper": ["caliper", "hcc_caliper", "hem_caliper"],
    })
    column_aliases: dict[str, list[str]] = field(default_factory=lambda: {
        "image_filename": ["filename", "file", "file_name", "image", "image_name", "image_path", "path"],
        "patient_id": ["patient", "patient_no", "pid", "subject_id", "case_id"],
        "class_label": ["label", "class", "diagnosis", "category"],
        "split": ["subset", "partition", "set"],
        "variant": ["version", "image_version", "type"],
    })
    documented_reference: DocumentedReference = field(default_factory=DocumentedReference)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser()
        self.metadata_csv = Path(self.metadata_csv).expanduser() if self.metadata_csv else None
        self.image_extensions = [str(x).lower() if str(x).startswith(".") else "." + str(x).lower()
                                 for x in self.image_extensions]
        if self.patient_scope not in {"global", "per_class"}:
            raise ValueError("patient_scope must be 'global' or 'per_class'")
        if self.positive_class not in self.classes:
            raise ValueError("positive_class must be one of classes")
        if set(self.split_fractions) != {"train", "val", "test"}:
            raise ValueError("split_fractions must define train, val and test")
        if any(float(v) <= 0 for v in self.split_fractions.values()) or abs(sum(self.split_fractions.values()) - 1) > 1e-6:
            raise ValueError("split_fractions must be positive and sum to 1.0")
        if len(self.classes) != 2:
            raise ValueError("LiverGuard currently supports exactly two classes")

    def _map(self, value: Any, aliases: dict[str, list[str]], canonical: list[str]) -> str | None:
        n = _norm(value)
        if n is None:
            return None
        for key in canonical:
            if n == _norm(key) or n in {_norm(a) for a in aliases.get(key, [])}:
                return key
        return None

    def map_class(self, value: Any) -> str | None:
        return self._map(value, self.class_aliases, self.classes)

    def map_split(self, value: Any) -> str | None:
        return self._map(value, self.split_aliases, ["train", "val", "test"])

    def map_variant(self, value: Any) -> str | None:
        return self._map(value, self.variant_aliases, ["clean", "caliper"])

    def map_column(self, value: Any) -> str | None:
        n = _norm(value)
        if n is None:
            return None
        for canonical, aliases in self.column_aliases.items():
            if n == _norm(canonical) or n in {_norm(a) for a in aliases}:
                return canonical
        return None

    def model_dump(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["metadata_csv"] = str(self.metadata_csv) if self.metadata_csv else None
        return result


def load_config(path: str | Path | None = None, *, data_root: str | Path | None = None,
                metadata_csv: str | Path | None = None, patient_scope: str | None = None) -> DatasetConfig:
    """Load configs/dataset.yaml and apply CLI > environment > YAML precedence."""
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "dataset.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        raw = dict(loaded)
    known = {k: raw[k] for k in DatasetConfig.__dataclass_fields__ if k in raw}
    ref = raw.get("documented_reference")
    if isinstance(ref, dict):
        known["documented_reference"] = DocumentedReference(**ref)
    root = data_root if data_root is not None else os.getenv("LIVERGUARD_DATA_ROOT", known.get("data_root", "data"))
    csv = metadata_csv if metadata_csv is not None else os.getenv("LIVERGUARD_METADATA_CSV", known.get("metadata_csv"))
    scope = patient_scope if patient_scope is not None else known.get("patient_scope", "global")
    known.update(data_root=root, metadata_csv=csv, patient_scope=scope)
    return DatasetConfig(**known)
