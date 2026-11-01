"""Binary metrics (HCC = positive class) and patient-level cluster-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CI_KEYS = ("auroc", "accuracy", "sensitivity", "specificity", "precision", "f1")


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def compute_metrics(y, p, threshold: float = 0.5) -> dict:
    """Metrics for labels y (0/1) and P(HCC) p at a fixed decision threshold.

    confusion_matrix is [[tn, fp], [fn, tp]] (rows = true class, columns = predicted).
    AUROC/AUPRC are NaN when only one class is present.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    two = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "threshold": float(threshold),
        "accuracy": _div(tp + tn, len(y)), "sensitivity": sens, "specificity": spec,
        "precision": _div(tp, tp + fp), "f1": _div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "auroc": float(roc_auc_score(y, p)) if two else float("nan"),
        "auprc": float(average_precision_score(y, p)) if two else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (ties -> the higher threshold).

    Must be fitted on VALIDATION predictions only and then applied unchanged to the test split.
    Returns 0.5 when only one class is present.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    cands = np.unique(p)
    j = np.array([(((p >= t) & (y == 1)).sum() / (y == 1).sum()) - (((p >= t) & (y == 0)).sum() / (y == 0).sum())
                  for t in cands])
    best = cands[np.flatnonzero(j == j.max())[-1]]
    # midpoint to the next-lower distinct score so the cut is not placed exactly on a validation sample
    lower = cands[cands < best]
    return float((best + lower.max()) / 2) if len(lower) else float(best)


def cluster_bootstrap_ci(y, p, groups, *, threshold: float = 0.5, n_boot: int = 1000,
                         ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile CIs from resampling PATIENTS with replacement.

    Images of one patient are correlated, so an image-level bootstrap would give intervals that are too narrow.
    Returns {metric: [lo, hi]} for CI_KEYS plus 'n_valid_replicates'. Replicates with a single class are skipped.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(int(inv.max()) + 1)]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in CI_KEYS}
    valid = 0
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) < 2:
            continue
        valid += 1
        m = compute_metrics(y[idx], p[idx], threshold)
        for k in CI_KEYS:
            draws[k].append(m[k])
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    out: dict = {}
    for k in CI_KEYS:
        v = np.asarray(draws[k], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))] if len(v) else [float("nan")] * 2
    out["n_valid_replicates"] = valid
    return out
