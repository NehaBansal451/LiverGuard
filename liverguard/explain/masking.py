"""Counterfactual region-masking experiments: is the model using the region a clinician would use, or something else?

Regions
- 'annotation': a binary mask supplied by the user (named after the image relative path, see mask_candidates; non-zero = region of interest). Its clinical validity
  is whatever the supplier guarantees; LiverGuard does NOT generate liver or lesion segmentations.
- 'centre_proxy': a centred ellipse covering a fixed share of the image. It is a GEOMETRIC PROXY, NOT the liver and NOT the lesion.
  Results with it only say whether the model depends on the image centre versus the periphery.
- 'cam_top': the top-k share of Grad-CAM pixels (model-attended region) for a deletion / insertion faithfulness test.

Every experiment has a matched-area random control (the same mask randomly translated with wrap-around, so area and shape are
identical). A drop that the control also produces is attributable to 'a hole in the image', not to the region's content.

LIMITATIONS (also in the manifest): filled regions create artificial edges and out-of-distribution inputs; AUROC on a region-only
image above 0.5 shows information is present in that region but not that clinicians use it; a lesion near the border can be split
by the proxy; controls average n_control random placements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def proxy_roi(size: int, area_fraction: float = 0.4) -> np.ndarray:
    """Centred circle covering `area_fraction` of the (square) image."""
    r = size * np.sqrt(area_fraction / np.pi)
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    return ((yy - c) ** 2 + (xx - c) ** 2) <= r**2


def mask_candidates(mask_dir: str | Path, image_id: str) -> list[Path]:
    """Where a mask for `image_id` may live. image_ids are relative paths such as 'val_set/Clean/HCC/H007/0.png' and the bare file
    name ('0.png') repeats across patients, so it is NEVER used on its own (it would attach one patient's mask to another's image).

    Accepted (first hit wins): <mask_dir>/<image_id> (mirrored tree) | <mask_dir>/<image_id without suffix>.png |
    <mask_dir>/<image_id with '/' replaced by '__'> (flat directory, suffix kept or replaced by .png).
    """
    d, rel = Path(mask_dir), Path(image_id)
    flat = str(rel).replace("/", "__").replace("\\", "__")
    return [d / rel, d / rel.with_suffix(".png"), d / flat, d / (Path(flat).with_suffix(".png").name)]


def load_annotation_mask(mask_dir: str | Path | None, image_id: str, size: int) -> np.ndarray | None:
    """Binary mask resized (nearest) to the model input; None if no file. Empty / full masks are rejected as uninformative."""
    if mask_dir is None:
        return None
    for cand in mask_candidates(mask_dir, image_id):
        if cand.is_file():
            with Image.open(cand) as im:
                m = np.asarray(im.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 0
            if m.sum() == 0 or m.all():
                raise ValueError(f"Mask {cand} is empty or covers the whole image")
            return m
    return None


def random_translate(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same shape and exact same area, placed at a random location (wrap-around)."""
    h, w = mask.shape
    return np.roll(mask, (int(rng.integers(0, h)), int(rng.integers(0, w))), axis=(0, 1))


def cam_region(cam: np.ndarray, top_fraction: float) -> np.ndarray:
    """Boolean mask of the highest `top_fraction` of CAM pixels (ties broken by position -> exact area)."""
    k = max(1, int(round(top_fraction * cam.size)))
    idx = np.argsort(-cam.ravel(), kind="stable")[:k]
    m = np.zeros(cam.size, dtype=bool)
    m[idx] = True
    return m.reshape(cam.shape)


def fill_outside(img: Image.Image, keep: np.ndarray, fill: str = "mean") -> Image.Image:
    """Keep pixels where `keep` is True; replace the rest with the image's mean grey ('mean') or black."""
    a = np.asarray(img.convert("RGB")).copy()
    v = a.reshape(-1, 3).mean(0) if fill == "mean" else np.zeros(3)
    a[~keep] = np.round(v).astype(np.uint8)
    return Image.fromarray(a)


def region_variants(img: Image.Image, roi: np.ndarray, rng: np.random.Generator, n_control: int = 3,
                    fill: str = "mean") -> dict[str, list[Image.Image]]:
    """Named image variants for one image and region. Lists hold n_control draws for controls, one image otherwise."""
    ctl = [random_translate(roi, rng) for _ in range(n_control)]
    return {"roi_only": [fill_outside(img, roi, fill)], "background_only": [fill_outside(img, ~roi, fill)],
            "control_roi_only": [fill_outside(img, c, fill) for c in ctl],
            "control_background_only": [fill_outside(img, ~c, fill) for c in ctl]}


def cam_variants(img: Image.Image, cam: np.ndarray, top_fraction: float, rng: np.random.Generator, n_control: int = 3,
                 fill: str = "mean") -> dict[str, list[Image.Image]]:
    """Deletion (remove the top-CAM region) and insertion (keep only it), each with matched-area random controls."""
    reg = cam_region(cam, top_fraction)
    ctl = [random_translate(reg, rng) for _ in range(n_control)]
    return {"cam_delete": [fill_outside(img, ~reg, fill)], "cam_keep": [fill_outside(img, reg, fill)],
            "control_cam_delete": [fill_outside(img, ~c, fill) for c in ctl],
            "control_cam_keep": [fill_outside(img, c, fill) for c in ctl]}
