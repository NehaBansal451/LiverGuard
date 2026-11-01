"""Synthetic SMC-LUD-like trees for tests (random noise images; no real patient data)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LAYOUT_A = "split_variant_class_patient"  # train_set/Clean/HCC/<patient>/<n>.png  (no CSV needed)
LAYOUT_B = "class_patient"                # HCC/<patient>/<n>.png + metadata.csv (split & variant in CSV)


def write_png(path: Path, seed: int, size: int = 48) -> None:
    arr = np.random.default_rng(seed).integers(0, 256, size=(size, size), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def write_signal_png(path: Path, seed: int, cls: str, size: int = 48) -> None:
    """Noise image; HCC gets a bright blob at a random position (a learnable toy signal, NOT ultrasound)."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(90, 25, (size, size))
    if cls == "HCC":
        cy, cx = rng.integers(12, size - 12, 2)
        yy, xx = np.mgrid[:size, :size]
        arr += 110 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0**2)))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def _split_for(i: int) -> str:
    r = i % 10
    return "train" if r < 7 else ("val" if r < 9 else "test")


def build_tree(root: Path, *, layout: str = LAYOUT_A, n_per_class: int = 30,
               leak: bool = False, cross_variant_leak: bool = False, signal: bool = False) -> dict:
    """Create a patient-level 70/20/10 dataset. Returns the ground-truth records."""
    recs: list[dict] = []
    for cls, prefix in (("HCC", "H"), ("Hemangioma", "M")):
        for i in range(n_per_class):
            pid, split = f"{prefix}{i:03d}", _split_for(i)
            for k in range(1 + i % 3):
                recs.append(dict(patient=pid, cls=cls, split=split, variant="Clean", name=f"{k}.png"))
            if i % 2 == 0:
                recs.append(dict(patient=pid, cls=cls, split=split, variant="Caliper", name="c0.png"))
    if leak:  # patient H000 (train) gets an extra Clean image in the test split
        recs.append(dict(patient="H000", cls="HCC", split="test", variant="Clean", name="9.png"))
    if cross_variant_leak:  # H001 has Clean images in train only; give it a Caliper image in test
        recs.append(dict(patient="H001", cls="HCC", split="test", variant="Caliper", name="c9.png"))

    for n, r in enumerate(recs):
        if layout == LAYOUT_A:
            rel = Path(f"{r['split']}_set") / r["variant"] / r["cls"] / r["patient"] / r["name"]
        else:
            rel = Path(r["cls"]) / r["patient"] / r["name"]
        r["rel"] = rel.as_posix()
        if signal:
            write_signal_png(root / rel, n, r["cls"])
        else:
            write_png(root / rel, seed=n)

    csv = None
    if layout == LAYOUT_B:
        csv = root / "metadata.csv"
        pd.DataFrame({
            "patient_id": [r["patient"] for r in recs], "image_filename": [r["name"] for r in recs],
            "class_label": [r["cls"] for r in recs], "split": [r["split"] for r in recs],
            "variant": [r["variant"].lower() for r in recs],
            "tumor_size_cm": [f"{1 + (n % 5) * 0.5}" for n in range(len(recs))],
        }).to_csv(csv, index=False)
    return {"records": recs, "csv": csv}
