"""Inference and split-level evaluation (shared by training and later audit modules)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from liverguard.ml.metrics import cluster_bootstrap_ci, compute_metrics


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (p_hcc, labels, row_index) in dataset order; model is put in eval mode."""
    model.eval()
    probs, labels, rows = [], [], []
    for x, y, i in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
        rows.append(i.numpy())
    p, y, r = np.concatenate(probs), np.concatenate(labels), np.concatenate(rows)
    order = np.argsort(r)
    return p[order], y[order], r[order]


def evaluate_frame(frame: pd.DataFrame, p: np.ndarray, *, threshold: float, n_boot: int, seed: int) -> dict:
    """Metrics (+ patient-level bootstrap CIs) for one split; also returns the per-image prediction table."""
    y = frame["label"].to_numpy(dtype=int)
    metrics = compute_metrics(y, p, threshold)
    metrics["ci95_patient_bootstrap"] = cluster_bootstrap_ci(y, p, frame["patient_key"].to_numpy(),
                                                             threshold=threshold, n_boot=n_boot, seed=seed)
    metrics["n_patients"] = int(frame["patient_key"].nunique())
    return metrics


def prediction_table(frame: pd.DataFrame, p: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame({"image_id": frame["image_id"].to_numpy(), "patient_key": frame["patient_key"].to_numpy(),
                         "split": frame["split"].to_numpy(), "label": frame["label"].to_numpy(),
                         "p_hcc": p, "pred": (p >= threshold).astype(int)})
