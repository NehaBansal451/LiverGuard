"""Shared setup for post-training analyses (M3 audit already has its own copy; M4/M5 use this one).

Enforces the same discipline everywhere: patient-level split from the index, checkpoint/index split_id must match, the TEST split
needs an explicit ``final_report=True``, and runs from smoke tests / null controls are watermarked as non-results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image

from liverguard.data.splits import split_manifest_id
from liverguard.ml.data import build_transform, load_resized, load_split_frame
from liverguard.ml.models import load_checkpoint
from liverguard.ml.trainer import resolve_device
from liverguard.robustness.engine import resolve_threshold, subsample_patients
from liverguard.utils import get_logger

log = get_logger(__name__)


@dataclass
class Session:
    model: nn.Module
    meta: dict
    device: torch.device
    frame: pd.DataFrame          # ALL splits of the variant (patient-checked)
    part: pd.DataFrame           # the analysed split
    split: str
    split_id: str
    threshold: float
    threshold_source: str
    size: int
    watermark: str | None
    checkpoint: Path

    @property
    def y(self) -> np.ndarray:
        return self.part["label"].to_numpy(dtype=int)

    @property
    def ids(self) -> list[str]:
        return self.part["image_id"].astype(str).tolist()

    @property
    def groups(self) -> np.ndarray:
        return self.part["patient_key"].to_numpy()

    def load_images(self, frame: pd.DataFrame | None = None) -> list[Image.Image]:
        return [load_resized(p, self.size) for p in (self.part if frame is None else frame)["path"]]


def watermark_for(meta: dict) -> str | None:
    if bool(meta.get("valid_as_result", True)):
        return None
    reasons = []
    if meta.get("experiment_kind") == "smoke":
        reasons.append("model from a SMOKE TEST on synthetic data")
    if meta.get("label_control", "none") != "none":
        reasons.append("model from a NULL CONTROL (shuffled labels)")
    return "NOT A RESULT: " + (" + ".join(reasons) or "invalid run")


def open_session(checkpoint: str | Path, index_csv: str | Path, *, split: str = "val", variant: str | None = "clean",
                 threshold: float | str = 0.5, device: str = "auto", final_report: bool = False,
                 split_file: str | Path | None = None, data_root: str | Path | None = None,
                 max_patients: int | None = None, seed: int = 0) -> Session:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    if split == "test" and not final_report:
        raise ValueError("Analysing the TEST split requires final_report=True (--final-report). Use the validation split while "
                         "developing; the test split is for the final report only.")
    ckpt = Path(checkpoint)
    dev = resolve_device(device)
    model, meta = load_checkpoint(ckpt, dev)
    thr, src = resolve_threshold(threshold, ckpt)
    frame = load_split_frame(index_csv, split_file=split_file, variant=variant, data_root=data_root)
    sid = split_manifest_id(frame)
    if meta.get("split_id") and meta["split_id"] != sid:
        raise ValueError(f"Checkpoint was trained on split_id {meta['split_id']} but this index/variant gives {sid}; refusing.")
    part = frame[frame["split"] == split].reset_index(drop=True)
    if max_patients:
        part = subsample_patients(part, max_patients, seed)
    if part.empty:
        raise ValueError(f"No images in split {split!r}")
    return Session(model, meta, dev, frame, part, split, sid, thr, src, int(meta["image_size"]), watermark_for(meta), ckpt)


@torch.inference_mode()
def predict_proba(model: nn.Module, images: list[Image.Image], device, batch_size: int = 64) -> np.ndarray:
    """p(HCC) for PIL images (model-input resolution), eval mode."""
    model.eval()
    tf, out = build_transform(False), []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(im) for im in images[i:i + batch_size]]).to(device)
        out.append(torch.softmax(model(x).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
