"""Training configuration (validated; loaded from configs/*.yaml with optional CLI overrides)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str = "baseline"
    # 'smoke' = pipeline check on synthetic data; 'intervention' is reserved for M6 robust-training runs.
    experiment_kind: Literal["baseline", "smoke", "intervention"] = "baseline"
    # Null control: permute labels across PATIENTS. A pipeline free of leakage must score ~0.5 AUROC.
    label_control: Literal["none", "patient_shuffle"] = "none"

    arch: Literal["efficientnet_b0", "resnet50", "tiny_cnn"] = "efficientnet_b0"
    pretrained: bool = True
    pretrained_weights_path: Path | None = None   # local torchvision state dict (offline / blocked download host)
    dropout: float = Field(0.2, ge=0.0, lt=1.0)

    image_size: int = Field(224, ge=16)
    variant: str | None = "clean"                 # official split is documented for Clean only
    augment: Literal["basic", "none"] = "basic"   # basic = h-flip + small rotation; no photometric/noise/blur
    batch_size: int = Field(32, ge=2)
    num_workers: int = Field(2, ge=0)
    cache_images: bool = True

    epochs: int = Field(30, ge=1)
    freeze_epochs: int = Field(2, ge=0)           # head-only warm-up
    lr: float = Field(3e-4, gt=0)
    backbone_lr_mult: float = Field(0.1, ge=0)
    weight_decay: float = Field(1e-4, ge=0)
    grad_clip: float | None = 1.0
    amp: bool = True                              # CUDA only
    class_weighting: Literal["auto", "none"] = "auto"   # inverse frequency from the TRAIN split only

    monitor: Literal["val_auroc", "val_loss"] = "val_auroc"
    patience: int = Field(8, ge=1)
    min_delta: float = Field(0.0, ge=0)

    threshold: float = Field(0.5, gt=0, lt=1)     # fixed in config: never tuned on test
    bootstrap_samples: int = Field(1000, ge=0)
    seed: int = 42
    device: str = "auto"

    @property
    def valid_as_result(self) -> bool:
        """False for smoke tests and null controls: their numbers must never be quoted as findings."""
        return self.experiment_kind != "smoke" and self.label_control == "none"

    @property
    def watermark(self) -> str | None:
        reasons = []
        if self.experiment_kind == "smoke":
            reasons.append("SMOKE TEST: synthetic data")
        if self.label_control != "none":
            reasons.append("NULL CONTROL (shuffled labels)")
        return " + ".join(reasons) + ": NOT A RESULT" if reasons else None


def load_train_config(path: str | Path, **overrides) -> TrainConfig:
    """Load YAML; any override whose value is not None replaces the YAML value."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**raw)
