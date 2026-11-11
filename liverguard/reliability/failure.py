"""Failure-case analysis built from real per-image predictions.

LIMITATIONS: categories are defined by the chosen threshold and a confidence cut-off; a 'confident error' list is a screening
tool for human review, not a causal explanation of why the model failed. Small splits give very few failures, so per-category
rates are noisy. Patient-level aggregation (mean p over a patient's images) is a simple rule, not a validated clinical decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def label_failures(df: pd.DataFrame, threshold: float, confident_at: float = 0.9) -> pd.DataFrame:
    """Add pred, correct, outcome (TP/TN/FP/FN), confident_error, and error rank (most confident error = 1).

    Needs columns: label (1 = HCC), p. `confident_at` is applied to max(p, 1-p) of the 0.5-argmax, set on validation data.
    """
    out = df.copy()
    out["pred"] = (out["p"] >= threshold).astype(int)
    out["correct"] = out["pred"] == out["label"]
    out["outcome"] = np.select([(out.pred == 1) & (out.label == 1), (out.pred == 0) & (out.label == 0),
                                (out.pred == 1) & (out.label == 0)], ["TP", "TN", "FP"], default="FN")
    out["confident_error"] = (~out["correct"]) & (np.maximum(out["p"], 1 - out["p"]) >= confident_at)
    # distance of p from the threshold on the wrong side = how confidently wrong
    wrong_margin = np.where(out["label"] == 1, threshold - out["p"], out["p"] - threshold)
    out["error_margin"] = np.where(out["correct"], np.nan, wrong_margin)
    out["error_rank"] = out["error_margin"].rank(ascending=False, method="first")
    return out


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("outcome").agg(n=("p", "size"), mean_p=("p", "mean"), mean_entropy=("entropy", "mean"),
                                  **({"mean_tta_std": ("tta_std", "mean")} if "tta_std" in df else {}),
                                  **({"mean_mc_std": ("mc_std", "mean")} if "mc_std" in df else {})).reset_index()
    return g


def error_by_uncertainty(df: pd.DataFrame, score: str, n_groups: int = 4) -> pd.DataFrame:
    """Error rate in equal-count groups of increasing uncertainty (is the uncertainty score informative about errors?)."""
    d = df.copy()
    d["group"] = pd.qcut(d[score].rank(method="first"), n_groups, labels=False)
    g = d.groupby("group").agg(n=("correct", "size"), error_rate=("correct", lambda s: 1 - s.mean()),
                               score_lo=(score, "min"), score_hi=(score, "max")).reset_index()
    g["score"] = score
    return g


def patient_level(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate by patient: mean p over the patient's images, patient-level prediction, and image-level disagreement."""
    g = df.groupby("patient_key").agg(label=("label", "first"), n_images=("p", "size"), mean_p=("p", "mean"),
                                      min_p=("p", "min"), max_p=("p", "max"), n_img_errors=("correct", lambda s: int((~s).sum())))
    g["pred"] = (g["mean_p"] >= threshold).astype(int)
    g["patient_correct"] = g["pred"] == g["label"]
    g["images_disagree"] = (g["min_p"] < threshold) & (g["max_p"] >= threshold)
    return g.reset_index()
