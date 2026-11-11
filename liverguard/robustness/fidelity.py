"""Label-preservation PROXIES: how much of the original image survives a distortion.

These are quantitative heuristics. They can flag distortions that very likely destroy information, but they CANNOT
confirm that a diagnosis is still possible; that needs expert review (see the distortion gallery plot).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from liverguard.robustness.config import FidelityCfg


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def fidelity_metrics(original: Image.Image, distorted: Image.Image) -> dict[str, float]:
    a, b = _gray(original), _gray(distorted)
    mse = float(np.mean((a - b) ** 2))
    psnr = 99.0 if mse == 0 else float(10 * np.log10(255.0**2 / mse))
    sa, sb = a.std(), b.std()
    corr = 1.0 if mse == 0 else (float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if sa > 0 and sb > 0 else 0.0)
    sat_a = (a <= 0) | (a >= 255)
    sat_b = (b <= 0) | (b >= 255)
    return {"psnr_db": psnr, "correlation": corr, "std_ratio": float(sb / sa) if sa > 0 else 1.0,
            "new_saturated_fraction": float(np.mean(sat_b & ~sat_a))}


# Distortions that MOVE content. PSNR / pixel correlation compare unaligned pixels, so for these a harmless 5% zoom or 5 degree
# rotation looks like total destruction (observed on the first audit run: centre-crop at 95% was flagged). Whether a lesion was
# cropped out or rotated out of view cannot be judged by pixel metrics at all and needs expert review.
GEOMETRIC_FAMILIES = frozenset({"rotation", "crop"})


def at_risk(m: dict[str, float], cfg: FidelityCfg, *, geometric: bool = False) -> bool:
    """True when any heuristic suggests diagnostic information may have been destroyed.

    geometric=True skips the alignment-dependent checks (PSNR, correlation) and keeps contrast-collapse and saturation.
    """
    if geometric:
        return bool(m["std_ratio"] < cfg.min_std_ratio or m["new_saturated_fraction"] > cfg.max_new_saturated_fraction)
    return bool(m["psnr_db"] < cfg.min_psnr_db or m["correlation"] < cfg.min_correlation
                or m["std_ratio"] < cfg.min_std_ratio or m["new_saturated_fraction"] > cfg.max_new_saturated_fraction)
