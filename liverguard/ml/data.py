"""Patient-safe data loading from the M1 ``image_index.csv``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from liverguard.utils import get_logger

log = get_logger(__name__)
SPLITS = ("train", "val", "test")
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class LeakageError(RuntimeError):
    """A patient appears in more than one split."""


def assert_patient_separation(frame: pd.DataFrame) -> None:
    n = frame.groupby("patient_key")["split"].nunique()
    bad = n[n > 1]
    if len(bad):
        raise LeakageError(f"{len(bad)} patient(s) occur in more than one split, e.g. {bad.index[:5].tolist()}")


def load_split_frame(index_csv: str | Path, *, split_file: str | Path | None = None,
                     variant: str | None = "clean", data_root: str | Path | None = None) -> pd.DataFrame:
    """Load image_index.csv, optionally override the split, filter to one variant, and verify.

    Raises ValueError for missing labels/patients/splits and LeakageError for patient overlap.
    """
    df = pd.read_csv(index_csv, dtype={"patient_key": str, "patient_id": str, "split": str})
    missing = {"image_id", "path", "patient_key", "label", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"{index_csv} lacks columns {sorted(missing)}; regenerate it with verify_dataset.py")
    if split_file is not None:
        s = pd.read_csv(split_file, dtype=str)
        if not {"image_id", "split"} <= set(s.columns):
            raise ValueError("split file needs image_id and split columns")
        df = df.drop(columns="split").merge(s[["image_id", "split"]], on="image_id", how="left")
    if variant is not None:
        if df["variant"].notna().any():
            df = df[df["variant"] == variant]
        else:
            log.warning("Index has no variant information; using all images")
        if df.empty:
            raise ValueError(f"No images with variant '{variant}'")
    if data_root is not None:
        df["path"] = [str(Path(data_root) / i) for i in df["image_id"]]
    if df["split"].isna().any():
        raise ValueError(f"{int(df['split'].isna().sum())} images have no split. Use the official split or "
                         "pass --split-file (e.g. the patient-grouped split from verify_dataset.py --rebuild-split).")
    if df[["label", "patient_key"]].isna().any().any():
        raise ValueError("Every image needs a label and a patient_key; run verify_dataset.py and fix errors first.")
    if not set(df["split"]) <= set(SPLITS):
        raise ValueError(f"Unexpected split names: {sorted(set(df['split']) - set(SPLITS))}")
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    assert_patient_separation(df)
    for s in SPLITS:
        sub = df[df["split"] == s]
        if sub.empty or sub["label"].nunique() < 2:
            raise ValueError(f"Split '{s}' is empty or contains a single class")
    return df


def apply_patient_label_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Null control: permute labels across patients (class proportions preserved)."""
    pat = frame.drop_duplicates("patient_key")[["patient_key", "label"]].reset_index(drop=True)
    perm = np.random.default_rng(seed).permutation(pat["label"].to_numpy())
    mapping = dict(zip(pat["patient_key"], perm))
    out = frame.copy()
    out["label"] = out["patient_key"].map(mapping).astype(int)
    return out


def compute_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    return counts.sum() / (n_classes * np.maximum(counts, 1))


def load_resized(path: str | Path, size: int) -> Image.Image:
    """The single place where images are decoded and resized (stretch to size x size, RGB)."""
    with Image.open(path) as im:
        return im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def build_transform(augment: bool) -> T.Compose:
    ops: list = [T.RandomHorizontalFlip(0.5), T.RandomRotation(10, fill=0)] if augment else []
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def preprocess_pil(img: Image.Image, image_size: int) -> torch.Tensor:
    """Evaluation-time preprocessing for an already-decoded image (reused by later audit modules)."""
    img = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    return build_transform(False)(img)


class UltrasoundDataset(Dataset):
    """Yields (tensor, label, row_index). Images are decoded once and cached resized when cache=True."""

    def __init__(self, frame: pd.DataFrame, image_size: int, augment: bool = False, cache: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.paths = self.frame["path"].tolist()
        self.labels = self.frame["label"].to_numpy(dtype=int)
        self.size = image_size
        self.tf = build_transform(augment)
        self.cache = [load_resized(p, image_size) for p in self.paths] if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = self.cache[i] if self.cache is not None else load_resized(self.paths[i], self.size)
        return self.tf(img), int(self.labels[i]), i


def make_loader(ds: UltrasoundDataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    drop_last = shuffle and len(ds) > batch_size and len(ds) % batch_size == 1  # BatchNorm needs >1 sample
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=g, drop_last=drop_last, persistent_workers=num_workers > 0)
