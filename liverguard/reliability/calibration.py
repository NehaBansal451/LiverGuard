"""Calibration metrics and post-hoc temperature scaling for a binary classifier (positive class = HCC).

LIMITATIONS (also written into every output manifest):
- ECE depends on the number/placement of bins and is biased for small n (each bin holds few images). Report n and the bin counts;
  never compare ECE across datasets of very different size. Equal-width and equal-mass ECE are both given.
- Brier score mixes calibration and discrimination; NLL punishes confident errors heavily.
- Temperature scaling is a single global rescale of the logit. It cannot repair class- or region-specific miscalibration and
  does not change AUROC or ranking. It must be fitted on VALIDATION data only.
- Images of one patient are correlated: CIs resample patients, not images.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def brier_score(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def nll(y, p) -> float:
    p, y = _clip(p), np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _bin_edges(conf: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "width":
        return np.linspace(0.5, 1.0, n_bins + 1)          # top-label confidence of a binary model lives in [0.5, 1]
    q = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = 0.5, 1.0
    return q


def reliability_table(y, p, n_bins: int = 10, strategy: str = "width") -> pd.DataFrame:
    """Top-label reliability: confidence = max(p, 1-p) vs accuracy of the predicted class, per bin.

    strategy: 'width' (equal-width bins on [0.5, 1]) or 'mass' (equal-count bins).
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    edges = _bin_edges(conf, n_bins, strategy)
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "confidence": float(conf[m].mean()) if n else np.nan,
                     "accuracy": float(correct[m].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "width") -> float:
    """Top-label ECE = sum_b (n_b / n) * |accuracy_b - confidence_b|. Uses the 0.5 argmax decision, independent of any threshold."""
    t = reliability_table(y, p, n_bins, strategy)
    t = t[t["n"] > 0]
    return float(np.sum(t["n"] / t["n"].sum() * np.abs(t["accuracy"] - t["confidence"])))


def maximum_calibration_error(y, p, n_bins: int = 10) -> float:
    t = reliability_table(y, p, n_bins, "width")
    t = t[t["n"] > 0]
    return float(np.max(np.abs(t["accuracy"] - t["confidence"])))


def calibration_summary(y, p, n_bins: int = 10) -> dict:
    return {"n": int(len(y)), "ece_width": expected_calibration_error(y, p, n_bins, "width"),
            "ece_mass": expected_calibration_error(y, p, n_bins, "mass"), "mce": maximum_calibration_error(y, p, n_bins),
            "brier": brier_score(y, p), "nll": nll(y, p), "mean_confidence": float(np.mean(np.maximum(p, 1 - np.asarray(p)))),
            "accuracy_at_0.5": float(np.mean((np.asarray(p) >= 0.5).astype(int) == np.asarray(y).astype(int))),
            "n_bins": n_bins}


# ------------------------------------------------------------------ temperature scaling
def apply_temperature(p, T: float) -> np.ndarray:
    """p' = sigmoid(logit(p) / T). For a 2-class softmax this equals temperature scaling of the two logits."""
    return sigmoid(logit(p) / T)


def fit_temperature(y, p, t_min: float = 0.05, t_max: float = 20.0) -> float:
    """T minimising validation NLL (log-space grid + golden-section refinement; deterministic, no optimiser dependency).

    Returns 1.0 (no change) if only one class is present, since NLL is then minimised by an unbounded T.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 1.0
    z = logit(p)
    f = lambda t: nll(y, sigmoid(z / t))                       # noqa: E731
    grid = np.exp(np.linspace(np.log(t_min), np.log(t_max), 200))
    k = int(np.argmin([f(t) for t in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(60):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if f(c) < f(d) else (c, b)
    return float((a + b) / 2)


def crossfit_temperature(y, p, groups, k: int = 5, seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """Out-of-sample calibrated probabilities from validation data alone: fit T on k-1 patient folds, apply to the held-out fold.

    Gives an honest estimate of post-calibration error WITHOUT touching the test split (fitting and scoring T on the same
    validation images would flatter it). Patients never span folds.
    """
    y, p, groups = np.asarray(y).astype(int), np.asarray(p, dtype=float), np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    k = max(2, min(k, len(uniq)))
    fold_of = {g: i % k for i, g in zip(range(len(uniq)), uniq[perm])}
    f = np.array([fold_of[g] for g in groups])
    out, Ts = p.copy(), []
    for i in range(k):
        te = f == i
        T = fit_temperature(y[~te], p[~te])
        Ts.append(T)
        out[te] = apply_temperature(p[te], T)
    return out, Ts


def cluster_bootstrap(y, p, groups, fn, *, n_boot: int = 500, ci: float = 0.95, seed: int = 0) -> list[float]:
    """Percentile CI of fn(y, p) when resampling PATIENTS with replacement. Returns [lo, hi] (NaN if n_boot == 0)."""
    if n_boot <= 0:
        return [float("nan"), float("nan")]
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        vals.append(fn(y[idx], p[idx]))
    a = (1 - ci) / 2
    return [float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))]
