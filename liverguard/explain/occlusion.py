"""Occlusion sensitivity (Zeiler & Fergus 2014): slide a neutral patch over the image and record the drop in class probability.

LIMITATIONS (also written into every output manifest):
- Resolution is the patch size; a lesion smaller than the patch is blurred into its surroundings.
- A grey patch is itself an out-of-distribution input: part of any drop can be 'the model dislikes holes', not 'information removed'.
- Only single patches are removed, so redundant evidence (two regions that each suffice) is invisible; interactions are not modelled.
- Cost is (H/stride)^2 forward passes per image; use it on selected cases, not the whole split.
- The map is a statement about THIS model's input-output behaviour, not about anatomy or pathology.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


@torch.inference_mode()
def occlusion_map(model: nn.Module, x: torch.Tensor, target: int, patch: int = 32, stride: int = 16,
                  batch_size: int = 64) -> tuple[np.ndarray, float]:
    """x: (1, 3, H, W) normalised. Returns (map (H, W), base_prob).

    map = mean over covering patches of [p_target(clean) - p_target(patch occluded)]. Positive = removing the area LOWERS
    the target-class probability. The fill is the image's own per-channel mean (neutral grey for the image).
    """
    model.eval()
    _, _, H, W = x.shape
    base = float(torch.softmax(model(x).float(), 1)[0, target])
    fill = x.mean(dim=(2, 3), keepdim=True)
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    if ys[-1] != H - patch:
        ys.append(max(H - patch, 0))
    if xs[-1] != W - patch:
        xs.append(max(W - patch, 0))
    pos = [(i, j) for i in ys for j in xs]
    acc, cnt = np.zeros((H, W)), np.zeros((H, W))
    for s in range(0, len(pos), batch_size):
        chunk = pos[s:s + batch_size]
        xb = x.repeat(len(chunk), 1, 1, 1)
        for k, (i, j) in enumerate(chunk):
            xb[k, :, i:i + patch, j:j + patch] = fill[0]
        p = torch.softmax(model(xb).float(), 1)[:, target].cpu().numpy()
        for (i, j), pk in zip(chunk, p):
            acc[i:i + patch, j:j + patch] += base - pk
            cnt[i:i + patch, j:j + patch] += 1
    return acc / np.maximum(cnt, 1), base
