"""Training-time robust augmentation built on the SAME distortion functions the audit uses (so attack == defence family)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from liverguard.improve.interventions import conditions_of
from liverguard.robustness.config import RobustnessConfig, load_robustness_config
from liverguard.robustness.distortions import apply_distortion


class RobustAugment:
    """Applies, per image and per selected condition, the distortion with probability `p` and a random plausible-tier severity.

    Severity: u ~ U(0, 1); param = p1 + u * (p_max - p1), where p1 / p_max are the audit config's level-1 and
    level-`plausible_max_level` parameters. For `jpeg` (quality) the interpolation runs from the milder to the stronger level too.
    Conditions are applied in config order, each independently (so several can stack, like a real acquisition chain).
    """

    def __init__(self, intervention: str, p: float = 0.5, rob_cfg: RobustnessConfig | str | Path | None = None,
                 max_level: int | None = None) -> None:
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        cfg = rob_cfg if isinstance(rob_cfg, RobustnessConfig) else load_robustness_config(rob_cfg)
        self.intervention, self.p = intervention, p
        names = conditions_of(intervention)
        by_name = {c.name: c for c in cfg.conditions}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise ValueError(f"Audit config lacks conditions {missing} needed by intervention {intervention!r}")
        top = min(max_level or cfg.plausible_max_level, 10**6)
        self.specs = []
        for n in names:
            c = by_name[n]
            k = min(top, len(c.params))
            if k < 1:
                raise ValueError("max_level must be >= 1")
            self.specs.append((c.name, c.func, float(c.params[0]), float(c.params[k - 1])))
        self._rng: np.random.Generator | None = None

    @property
    def conditions(self) -> list[str]:
        return [s[0] for s in self.specs]

    def _generator(self) -> np.random.Generator:
        # One generator per process, seeded from torch's per-process seed: deterministic given the run seed and, with
        # DataLoader workers, distinct per worker. Not related to the audit's keyed RNG.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**63))
        return self._rng

    def __call__(self, img: Image.Image, key: str) -> Image.Image:
        if not self.specs:
            return img
        rng = self._generator()
        out = img
        for name, func, p1, pmax in self.specs:
            if rng.random() < self.p:
                param = p1 + float(rng.random()) * (pmax - p1)
                out = apply_distortion(out, func, param, seed=int(rng.integers(0, 2**31 - 1)), image_id=f"train/{key}",
                                       condition=name)
        return out
