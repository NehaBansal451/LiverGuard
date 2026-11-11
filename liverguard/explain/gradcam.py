"""Grad-CAM (Selvaraju et al. 2017) implemented directly with hooks, plus a model-randomisation sanity check.

LIMITATIONS (also written into every output manifest):
- Grad-CAM shows where the gradient of ONE class logit is large in the last convolutional feature map. It is low resolution
  (7x7 for 224 px input, upsampled bilinearly) and is NOT evidence of medical reasoning or of causal importance.
- Maps are min-max normalised per image, which hides absolute magnitude: a faint map looks as 'confident' as a strong one.
- Class-specific maps for the two classes of a 2-class softmax model are often near-complements; a map for the non-predicted class is
  not 'what the model would look at if it were right'.
- A map can look plausible even for a model with random weights (Adebayo et al. 2018). We therefore report a randomisation check;
  a high correlation with a randomised model means the map is mostly image structure, not learned evidence.
- Heatmaps must be read together with the counterfactual masking and occlusion experiments, never alone.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn


def default_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional feature stage: ResNet layer4[-1], EfficientNet features[-1], TinyCNN last ReLU, else last Conv2d."""
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "classifier") and hasattr(model, "features") and isinstance(model.features, nn.Sequential) \
            and type(model).__name__ == "EfficientNet":
        return model.features[-1]
    if type(model).__name__ == "TinyCNN":
        return model.features[10]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No convolutional layer found for Grad-CAM")
    return convs[-1]


class GradCAM:
    """cam, probs = GradCAM(model)(x, target). target: 'predicted' | class index (0 = hemangioma, 1 = HCC)."""

    def __init__(self, model: nn.Module, layer: nn.Module | None = None) -> None:
        self.model = model.eval()
        self.layer = layer if layer is not None else default_target_layer(model)
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self._h = self.layer.register_forward_hook(self._fwd)

    def _fwd(self, module, args, out):
        # The hook stays installed while the same model is used elsewhere (e.g. occlusion under inference_mode); only passes that
        # carry gradients are Grad-CAM passes.
        if not out.requires_grad:
            return
        self._act = out
        out.register_hook(lambda g: setattr(self, "_grad", g))

    def close(self) -> None:
        self._h.remove()

    def __call__(self, x: torch.Tensor, target: str | int = "predicted") -> tuple[np.ndarray, np.ndarray]:
        """Returns (cam, probs): cam (N, H, W) in [0, 1] at input resolution; probs (N, 2) softmax."""
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            cls = logits.argmax(1) if target == "predicted" else torch.full((x.shape[0],), int(target), device=x.device)
            logits.gather(1, cls[:, None]).sum().backward()
        a, g = self._act.detach(), self._grad.detach()
        w = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mx = cam.amax(dim=(1, 2), keepdim=True)
        mn = cam.amin(dim=(1, 2), keepdim=True)
        cam = torch.where(mx > mn, (cam - mn) / (mx - mn + 1e-12), torch.zeros_like(cam))
        return cam.cpu().numpy(), torch.softmax(logits.detach(), 1).cpu().numpy()


def batched_gradcam(model: nn.Module, X: torch.Tensor, target: str | int = "predicted", batch_size: int = 32):
    gc = GradCAM(model)
    try:
        cams, probs = zip(*[gc(X[i:i + batch_size], target) for i in range(0, len(X), batch_size)])
    finally:
        gc.close()
    return np.concatenate(cams), np.concatenate(probs)


def randomized_copy(model: nn.Module, seed: int = 0) -> nn.Module:
    """Copy of the model with ALL parameters re-initialised (and BN statistics reset): the 'untrained model' baseline."""
    m = copy.deepcopy(model)
    torch.manual_seed(seed)
    for mod in m.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    return m.eval()


def randomization_check(model: nn.Module, X: torch.Tensor, targets: np.ndarray, seed: int = 0) -> np.ndarray:
    """Per-image Spearman correlation between the trained model's CAM (for `targets`) and a fully randomised model's CAM.

    Values near 1 mean the explanation barely depends on what the model learned (sanity check FAILED for that image);
    values near 0 mean it does. Returns NaN where a map is constant.
    """
    rnd = randomized_copy(model, seed)
    out = np.full(len(X), np.nan)
    gc_t, gc_r = GradCAM(model), GradCAM(rnd)
    try:
        for i in range(len(X)):
            a, _ = gc_t(X[i:i + 1], int(targets[i]))
            b, _ = gc_r(X[i:i + 1], int(targets[i]))
            if a.std() > 0 and b.std() > 0:
                out[i] = float(spearmanr(a.ravel(), b.ravel())[0])
    finally:
        gc_t.close(); gc_r.close()
    return out
