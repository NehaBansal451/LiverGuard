"""Image distortions for the robustness audit.

Every function maps (PIL RGB image, severity parameter, numpy Generator) -> NEW PIL RGB image; the input is never modified.

Design rules
- Ultrasound is greyscale, so stochastic noise is drawn once per pixel and shared by the 3 channels (no colour noise).
- Randomness is keyed by (seed, image_id, condition) and NOT by level ("common random numbers"): the level-3 noise field is
  the level-2 field scaled, so severity levels are nested and degradation curves are comparable across levels.
- Results are independent of batching, worker count and processing order.
- Parameters are applied at model-input resolution (documented limitation: JPEG/blur at 224 px differ from native resolution).
"""
from __future__ import annotations

import hashlib
import io
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


def rng_for(seed: int, *keys: object) -> np.random.Generator:
    """Deterministic generator from (seed, keys...)."""
    digest = hashlib.sha256("|".join(map(str, (seed, *keys))).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _to_array(img: Image.Image) -> np.ndarray:
    # float64 on purpose: in float32, products such as 90 * 0.35 land on rounding ties that float32 resolves differently
    # from the exact value, making results depend on arithmetic precision (off by one grey level on some pixels).
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def brightness(img: Image.Image, factor: float, rng: np.random.Generator) -> Image.Image:
    """Multiplicative gain (factor < 1 darker, > 1 brighter), clipped to [0, 255]."""
    return _to_image(_to_array(img) * factor)


def contrast(img: Image.Image, factor: float, rng: np.random.Generator) -> Image.Image:
    """Scale deviations from the image mean (mean preserved before clipping)."""
    a = _to_array(img)
    m = a.mean()
    return _to_image((a - m) * factor + m)


def gaussian_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Additive zero-mean Gaussian noise, std `sigma` grey levels, identical across channels."""
    a = _to_array(img)
    n = rng.standard_normal((a.shape[0], a.shape[1], 1))
    return _to_image(a + sigma * n)


def speckle_noise(img: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    """Multiplicative noise out = x * (1 + sigma * n), n ~ N(0, 1).

    A simplified model: real ultrasound speckle is signal-dependent with a Rayleigh/gamma-like envelope
    and spatial correlation, which this does not reproduce.
    """
    a = _to_array(img)
    n = rng.standard_normal((a.shape[0], a.shape[1], 1))
    return _to_image(a * (1.0 + sigma * n))


def gaussian_blur(img: Image.Image, frac: float, rng: np.random.Generator) -> Image.Image:
    """Gaussian blur with sigma = frac * min(width, height) pixels (resolution independent)."""
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=frac * min(img.size)))


def rotation(img: Image.Image, degrees: float, rng: np.random.Generator) -> Image.Image:
    """Rotate about the centre by +/- degrees (sign random per image, fixed across levels); no expansion, black fill."""
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return img.convert("RGB").rotate(sign * degrees, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def crop(img: Image.Image, keep: float, rng: np.random.Generator) -> Image.Image:
    """Centre crop keeping `keep` of each side, resized back to the original size (a zoom).

    The crop removes peripheral anatomy; if a lesion sits near the edge it may be lost (label preservation is not guaranteed).
    """
    img = img.convert("RGB")
    w, h = img.size
    cw, ch = max(1, round(w * keep)), max(1, round(h * keep))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.BICUBIC)


def gamma(img: Image.Image, g: float, rng: np.random.Generator) -> Image.Image:
    """out = 255 * (x / 255) ** g. g < 1 brightens midtones, g > 1 darkens; 0 and 255 are fixed points."""
    return _to_image(255.0 * np.power(_to_array(img) / 255.0, g))


def jpeg(img: Image.Image, quality: float, rng: np.random.Generator) -> Image.Image:
    """In-memory JPEG round trip at the given quality (1-100)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


DISTORTION_FUNCS: dict[str, Callable[[Image.Image, float, np.random.Generator], Image.Image]] = {
    "brightness": brightness, "contrast": contrast, "gaussian_noise": gaussian_noise,
    "speckle_noise": speckle_noise, "gaussian_blur": gaussian_blur, "rotation": rotation,
    "crop": crop, "gamma": gamma, "jpeg": jpeg,
}


def apply_distortion(img: Image.Image, func: str, param: float, *, seed: int, image_id: str,
                     condition: str) -> Image.Image:
    """Apply one distortion deterministically. The original image object is left untouched."""
    if func not in DISTORTION_FUNCS:
        raise ValueError(f"Unknown distortion function {func!r}; known: {sorted(DISTORTION_FUNCS)}")
    return DISTORTION_FUNCS[func](img, param, rng_for(seed, image_id, condition))
