"""EfficientNet-B0 / ResNet-50 (torchvision) with a 2-class head (index 1 = HCC), plus a tiny CNN for smoke tests."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models as tvm

from liverguard.utils import get_logger

log = get_logger(__name__)
NUM_CLASSES = 2
_WEIGHTS = {"efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}


class TinyCNN(nn.Module):
    """~6k-parameter CNN used ONLY to smoke-test the pipeline quickly on CPU."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        blocks, c_in = [], 3
        for c_out in (8, 16, 32):
            blocks += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(32, NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def head_module(model: nn.Module) -> nn.Module:
    return model.classifier if hasattr(model, "classifier") else model.fc


def _head_prefix(model: nn.Module) -> str:
    return "classifier." if hasattr(model, "classifier") else "fc."


def _load_pretrained(model: nn.Module, arch: str, weights_path: Path | None) -> str:
    """Load ImageNet weights into the backbone only. Never falls back silently to random weights."""
    prefix = _head_prefix(model)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith(prefix)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith(prefix)] + list(unexpected)
        if bad:
            raise RuntimeError(f"{weights_path} does not match {arch}; offending keys: {bad[:5]}")
        return f"local:{weights_path}"
    try:
        ref = (tvm.efficientnet_b0 if arch == "efficientnet_b0" else tvm.resnet50)(weights=_WEIGHTS[arch])
    except Exception as exc:  # noqa: BLE001 - download/cache errors differ by platform
        raise RuntimeError(
            f"Could not load pretrained {arch} weights ({type(exc).__name__}: {exc}). Refusing to fall back to "
            "random initialisation. Download the torchvision weights on a machine with access and pass "
            "--pretrained-weights-path, or set pretrained: false to train from scratch on purpose.") from exc
    model.load_state_dict({k: v for k, v in ref.state_dict().items() if not k.startswith(prefix)}, strict=False)
    return f"torchvision:{_WEIGHTS[arch]}"


def build_model(arch: str, *, pretrained: bool, weights_path: Path | None = None,
                dropout: float = 0.2) -> tuple[nn.Module, str]:
    """Return (model, weights_source). Logits have shape (N, 2); p(HCC) = softmax(logits)[:, 1]."""
    if arch == "tiny_cnn":
        if pretrained:
            raise ValueError("tiny_cnn has no pretrained weights; set pretrained: false")
        return TinyCNN(dropout), "random_init"
    if arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, NUM_CLASSES))
    elif arch == "resnet50":
        model = tvm.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, NUM_CLASSES))
    else:
        raise ValueError(f"Unknown architecture {arch!r}")
    source = _load_pretrained(model, arch, weights_path) if pretrained else "random_init"
    log.info("Built %s (weights: %s)", arch, source)
    return model, source


def set_backbone_requires_grad(model: nn.Module, flag: bool) -> None:
    """Freeze (False) / unfreeze (True) everything except the classification head."""
    head_ids = {id(p) for p in head_module(model).parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad_(flag)


def param_groups(model: nn.Module, lr: float, backbone_lr_mult: float) -> list[dict]:
    """[head group, backbone group]; backbone LR = lr * backbone_lr_mult (discriminative fine-tuning)."""
    head_ids = {id(p) for p in head_module(model).parameters()}
    return [{"params": [p for p in model.parameters() if id(p) in head_ids], "lr": lr},
            {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": lr * backbone_lr_mult}]


def save_checkpoint(path: str | Path, model: nn.Module, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint (weights_only load). Used by the later audit modules."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    meta = ckpt["meta"]
    model, _ = build_model(meta["arch"], pretrained=False, dropout=meta.get("dropout", 0.2))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), meta
