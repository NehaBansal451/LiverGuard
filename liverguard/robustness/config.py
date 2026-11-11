"""Robustness audit configuration."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from liverguard.robustness.distortions import DISTORTION_FUNCS

DEFAULT_ROBUSTNESS_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "robustness.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Condition(_Strict):
    name: str
    family: str
    func: str
    params: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _known_func(self) -> "Condition":
        if self.func not in DISTORTION_FUNCS:
            raise ValueError(f"condition {self.name!r}: unknown func {self.func!r}")
        return self


class FidelityCfg(_Strict):
    min_psnr_db: float = 15.0
    min_correlation: float = 0.5
    min_std_ratio: float = 0.3
    max_new_saturated_fraction: float = 0.30


class RobustnessConfig(_Strict):
    seed: int = 0
    plausible_max_level: int = Field(3, ge=0)
    n_boot: int = Field(500, ge=0)
    ci: float = Field(0.95, gt=0, lt=1)
    flip_alert: float = Field(0.05, gt=0, le=1)
    fidelity: FidelityCfg = FidelityCfg()
    conditions: list[Condition]

    @model_validator(mode="after")
    def _unique_names(self) -> "RobustnessConfig":
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names) or "clean" in names:
            raise ValueError("condition names must be unique and must not be 'clean'")
        return self

    def tier(self, level: int) -> str:
        return "plausible" if level <= self.plausible_max_level else "stress"

    def select(self, names: list[str] | None) -> "RobustnessConfig":
        """Copy restricted to the named conditions (error on unknown names)."""
        if not names:
            return self
        known = {c.name for c in self.conditions}
        bad = sorted(set(names) - known)
        if bad:
            raise ValueError(f"Unknown condition(s) {bad}; known: {sorted(known)}")
        return self.model_copy(update={"conditions": [c for c in self.conditions if c.name in names]})


def load_robustness_config(path: str | Path | None = None, **overrides) -> RobustnessConfig:
    with open(path or DEFAULT_ROBUSTNESS_CONFIG, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return RobustnessConfig(**raw)
