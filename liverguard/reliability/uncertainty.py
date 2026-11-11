"""Per-image uncertainty signals for a binary HCC-vs-hemangioma classifier.

LIMITATIONS (also written into every output manifest):
- Single-pass confidence max(p, 1-p) and predictive entropy are RANK-EQUIVALENT for a binary model: they order images identically,
  so they give identical risk-coverage curves. Entropy adds no ranking information over confidence here (it is reported for
  completeness and because it is the form MC-Dropout decomposes).
- MC-Dropout here samples only the dropout in the classification head (the architecture has no other dropout), so it measures
  sensitivity of the head to feature dropout, NOT full Bayesian model uncertainty. It can underestimate epistemic uncertainty.
- TTA spread measures sensitivity to a fixed set of small geometric changes (flip, +-5 deg, 5% zoom). Horizontal flip alters apparent
  organ side, so TTA is a stability probe, not an unbiased uncertainty estimate.
- None of these signals is clinical certainty; a model can be confidently wrong, and all are uncalibrated unless stated otherwise.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn

from liverguard.ml.data import build_transform
from liverguard.ml.models import head_module

LN2 = float(np.log(2.0))


def binary_entropy(p, base2: bool = True) -> np.ndarray:
    """Predictive entropy of Bernoulli(p); in bits (max 1.0 at p = 0.5) when base2."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return h / LN2 if base2 else h


def confidence(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1 - p)


# ------------------------------------------------------------------ MC dropout
def _non_head_dropouts(model: nn.Module) -> int:
    head_ids = {id(m) for m in head_module(model).modules()}
    return sum(isinstance(m, nn.Dropout) and id(m) not in head_ids for m in model.modules())


@torch.inference_mode()
def mc_dropout_probs(model: nn.Module, x: torch.Tensor, n_samples: int = 20, seed: int = 0, fast: bool = True) -> np.ndarray:
    """(n_samples, N) array of p(HCC) with dropout ACTIVE and BatchNorm/others in eval mode.

    fast=True runs the backbone once and re-applies only the head n_samples times; this is exact when all dropout lives in the
    head (true for the project's models) and otherwise falls back to full passes. Seeded -> reproducible.
    """
    model.eval()
    head = head_module(model)
    if fast and _non_head_dropouts(model) == 0:
        feats: list[torch.Tensor] = []
        h = head.register_forward_pre_hook(lambda m, a: feats.append(a[0].detach()))
        model(x)
        h.remove()
        f = feats[0]
        head.train()                                           # head dropout on
        torch.manual_seed(seed)
        out = [torch.softmax(head(f).float(), 1)[:, 1] for _ in range(n_samples)]
        head.eval()
    else:
        drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        for m in drops:
            m.train()
        torch.manual_seed(seed)
        out = [torch.softmax(model(x).float(), 1)[:, 1] for _ in range(n_samples)]
        for m in drops:
            m.eval()
    return torch.stack(out).cpu().numpy()


def mc_summary(samples: np.ndarray) -> dict[str, np.ndarray]:
    """Mean prob, std across samples, and mutual information (BALD) = H(mean) - mean(H) in bits (epistemic part)."""
    mean = samples.mean(0)
    mi = binary_entropy(mean) - binary_entropy(samples).mean(0)
    return {"mc_mean": mean, "mc_std": samples.std(0), "mc_mi": np.maximum(mi, 0.0)}


# ------------------------------------------------------------------ TTA
def _tta_views(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    z = int(round(0.05 * min(w, h)))
    crop = img.crop((z, z, w - z, h - z)).resize((w, h), Image.Resampling.BICUBIC)
    return [img, img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            img.rotate(5, Image.Resampling.BICUBIC, fillcolor=(0, 0, 0)),
            img.rotate(-5, Image.Resampling.BICUBIC, fillcolor=(0, 0, 0)), crop]


TTA_VIEW_NAMES = ["identity", "hflip", "rot+5", "rot-5", "zoom5"]


@torch.inference_mode()
def tta_probs(model: nn.Module, images: list[Image.Image], device, batch_size: int = 64) -> np.ndarray:
    """(n_views, N) array of p(HCC) over a fixed, deterministic set of views (see TTA_VIEW_NAMES)."""
    model.eval()
    tf = build_transform(False)
    views = [_tta_views(im) for im in images]
    out = np.zeros((len(TTA_VIEW_NAMES), len(images)))
    for v in range(len(TTA_VIEW_NAMES)):
        for i in range(0, len(images), batch_size):
            x = torch.stack([tf(views[j][v]) for j in range(i, min(i + batch_size, len(images)))]).to(device)
            out[v, i:i + x.shape[0]] = torch.softmax(model(x).float(), 1)[:, 1].cpu().numpy()
    return out


def tta_summary(P: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """TTA mean, std, and whether the thresholded decision changes in any view vs the identity view."""
    base = P[0] >= threshold
    return {"tta_mean": P.mean(0), "tta_std": P.std(0), "tta_flip": ((P >= threshold) != base).any(0)}
