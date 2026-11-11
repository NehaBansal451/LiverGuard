"""Compare models on exactly the same images: clean performance, robustness, calibration, prediction stability.

Everything is computed from the per-image audit tables of M3 (clean probability + one probability per condition and level), so
the comparison uses the same distortions, the same images and the same threshold logic for every model.

Uncertainty: PAIRED patient-level cluster bootstrap. One resample of patients is shared by the reference (baseline) and the
compared model, so differences are not inflated by sampling noise common to both.

Verdicts per metric and intervention (several seeds): 'improved' / 'worse' only if the paired CI excludes 0 in the same direction
for EVERY seed; otherwise 'inconclusive'. One seed gives a 1-seed caveat: seed-to-seed training variance is not measured then.

LIMITATIONS: metrics on distorted images inherit the audit's assumptions (a-priori tiers, label preservation unverified, simplified
speckle). Training on a family and auditing the same family measures adaptation to a known corruption, so 'trained' and
'not trained' conditions are reported separately. ECE is binned and noisy for small n. Many metrics are compared, so a single
'improved' at 95% is not multiplicity-corrected.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_BINS = 10
# metric name -> higher is better?
METRICS: dict[str, bool] = {
    "clean_auroc": True, "clean_accuracy": True, "clean_sensitivity": True, "clean_specificity": True,
    "clean_brier": False, "clean_ece": False,
    "plaus_auroc": True, "plaus_minus_clean_auroc": True, "plaus_worst_auroc": True, "plaus_auroc_trained": True, "plaus_auroc_untrained": True,
    "stress_auroc": True,
    "plaus_flip_rate": False, "stress_flip_rate": False, "plaus_mean_abs_dp": False,
    "plaus_brier": False, "plaus_ece": False,
}
# plaus_minus_clean_auroc = plausible-tier AUROC minus clean AUROC (<= 0 normally): separates "less degradation" from "better clean model".
# Descriptive only: it is NOT part of the pre-declared selection rule.
PRIMARY = ["clean_auroc", "plaus_auroc", "plaus_minus_clean_auroc", "plaus_worst_auroc", "plaus_flip_rate", "clean_ece", "plaus_ece"]


# ------------------------------------------------------------------ fast vectorised building blocks
def auroc_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of each row of S (K, n) against binary y (n,), average ranks for ties. NaN if a class is missing."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[0], np.nan)
    R = rankdata(S, axis=1)
    return (R[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def ece_rows(y: np.ndarray, P: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Top-label ECE (equal-width bins on [0.5, 1]) for each row of P (K, n) of p(HCC)."""
    out = np.empty(P.shape[0])
    for k, p in enumerate(P):
        conf = np.maximum(p, 1 - p)
        correct = ((p >= 0.5).astype(int) == y).astype(float)
        b = np.clip(((conf - 0.5) / 0.5 * n_bins).astype(int), 0, n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        sc = np.bincount(b, weights=conf, minlength=n_bins)
        sa = np.bincount(b, weights=correct, minlength=n_bins)
        nz = cnt > 0
        out[k] = float(np.sum(np.abs(sa[nz] - sc[nz])) / len(y))
    return out


@dataclass
class ModelEval:
    name: str
    intervention: str
    seed: int
    trained: frozenset
    threshold: float
    ids: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    p_clean: np.ndarray
    keys: list[tuple[str, int]]          # (condition, level) in a fixed order
    tier: np.ndarray                     # 'plausible' | 'stress' per key
    cond: np.ndarray                     # condition per key
    P: np.ndarray                        # (K, n) distorted probabilities

    @classmethod
    def from_audit(cls, name: str, intervention: str, seed: int, trained, audit_dir: str | Path) -> "ModelEval":
        d = Path(audit_dir)
        man = json.loads((d / "audit_manifest.json").read_text())
        df = pd.read_csv(d / "per_image.csv").sort_values(["condition", "level", "image_id"], kind="stable")
        ids = np.sort(df["image_id"].unique())
        first = df[(df["condition"] == df["condition"].iloc[0]) & (df["level"] == df["level"].iloc[0])].set_index("image_id").loc[ids]
        keys, rows, tier = [], [], []
        for (c, l), g in df.groupby(["condition", "level"], sort=False):
            g = g.set_index("image_id").loc[ids]
            keys.append((c, int(l)))
            rows.append(g["p_dist"].to_numpy(dtype=float))
            tier.append(g["tier"].iloc[0])
        return cls(name, intervention, seed, frozenset(trained), float(man["threshold"]["value"]), ids,
                   first["label"].to_numpy(dtype=int), first["patient_key"].astype(str).to_numpy(),
                   first["p_clean"].to_numpy(dtype=float), keys, np.array(tier), np.array([k[0] for k in keys]), np.vstack(rows))


def check_same_images(models: list[ModelEval]) -> None:
    ref = models[0]
    for m in models[1:]:
        if not (np.array_equal(m.ids, ref.ids) and np.array_equal(m.y, ref.y) and np.array_equal(m.groups, ref.groups)):
            raise ValueError(f"{m.name} and {ref.name} were not audited on identical images/labels/patients; refusing to compare")
        if m.keys != ref.keys:
            raise ValueError(f"{m.name} and {ref.name} were audited with different condition/level sets")


def metrics_at(m: ModelEval, idx: np.ndarray | None = None, trained: frozenset | None = None) -> dict[str, float]:
    """All METRICS for the images `idx` (default all). `trained` decides which conditions count as 'trained' (default m's own)."""
    idx = np.arange(len(m.y)) if idx is None else idx
    y, pc, P = m.y[idx], m.p_clean[idx], m.P[:, idx]
    thr, trained = m.threshold, (m.trained if trained is None else trained)
    pred = pc >= thr
    pos, neg = y == 1, y == 0
    out = {"clean_auroc": float(auroc_rows(y, pc[None])[0]), "clean_accuracy": float((pred == (y == 1)).mean()),
           "clean_sensitivity": float(pred[pos].mean()) if pos.any() else np.nan,
           "clean_specificity": float((~pred[neg]).mean()) if neg.any() else np.nan,
           "clean_brier": float(np.mean((pc - y) ** 2)), "clean_ece": float(ece_rows(y, pc[None])[0])}
    au = auroc_rows(y, P)
    flip = ((P >= thr) != pred[None]).mean(1)
    dp = np.abs(P - pc[None]).mean(1)
    brier = ((P - y[None]) ** 2).mean(1)
    pl, st = m.tier == "plausible", m.tier == "stress"
    tr = np.array([c in trained for c in m.cond])
    ece_pl = ece_rows(y, P[pl]) if pl.any() else np.array([np.nan])
    out.update({"plaus_auroc": float(np.nanmean(au[pl])), "plaus_minus_clean_auroc": float(np.nanmean(au[pl])) - out["clean_auroc"], "plaus_worst_auroc": float(np.nanmin(au[pl])),
                "plaus_auroc_trained": float(np.nanmean(au[pl & tr])) if (pl & tr).any() else np.nan,
                "plaus_auroc_untrained": float(np.nanmean(au[pl & ~tr])) if (pl & ~tr).any() else np.nan,
                "stress_auroc": float(np.nanmean(au[st])) if st.any() else np.nan,
                "plaus_flip_rate": float(flip[pl].mean()), "stress_flip_rate": float(flip[st].mean()) if st.any() else np.nan,
                "plaus_mean_abs_dp": float(dp[pl].mean()), "plaus_brier": float(brier[pl].mean()),
                "plaus_ece": float(np.nanmean(ece_pl))})
    return out


def paired_bootstrap(ref: ModelEval, other: ModelEval, n_boot: int, seed: int, ci: float = 0.95) -> pd.DataFrame:
    """Difference (other - ref) per metric with a paired patient-level cluster bootstrap CI. 'trained' = other's conditions for both."""
    check_same_images([ref, other])
    tr = other.trained
    d0 = {k: v - w for (k, v), w in zip(metrics_at(other, trained=tr).items(), metrics_at(ref, trained=tr).values())}
    rows = {k: {"metric": k, "diff": d0[k], "lo": np.nan, "hi": np.nan} for k in d0}
    if n_boot > 0:
        groups = ref.groups
        uniq = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in uniq}
        rng = np.random.default_rng(seed)
        draws = {k: [] for k in d0}
        for _ in range(n_boot):
            idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
            if len(np.unique(ref.y[idx])) < 2:        # AUROC undefined on a one-class resample: skip it (rare unless few patients)
                continue
            a, b = metrics_at(other, idx, tr), metrics_at(ref, idx, tr)
            for k in d0:
                draws[k].append(a[k] - b[k])
        q = (1 - ci) / 2
        for k, v in draws.items():
            v = [x for x in v if np.isfinite(x)]
            if len(v) >= 20:                           # too few valid resamples -> leave the CI undefined rather than invent one
                rows[k]["lo"], rows[k]["hi"] = float(np.quantile(v, q)), float(np.quantile(v, 1 - q))
    return pd.DataFrame(list(rows.values()))


def point_table(models: list[ModelEval]) -> pd.DataFrame:
    return pd.DataFrame([{"model": m.name, "intervention": m.intervention, "seed": m.seed, "threshold": m.threshold,
                          "n_images": len(m.y), "n_patients": len(np.unique(m.groups)), **metrics_at(m)} for m in models])


def per_condition_auroc(m: ModelEval) -> pd.DataFrame:
    au = auroc_rows(m.y, m.P)
    clean = float(auroc_rows(m.y, m.p_clean[None])[0])
    return pd.DataFrame({"model": m.name, "intervention": m.intervention, "seed": m.seed, "condition": m.cond,
                         "level": [k[1] for k in m.keys], "tier": m.tier, "auroc": au, "clean_auroc": clean})


def verdict(diffs: pd.DataFrame, metric: str) -> str:
    """diffs: one row per seed for one (intervention, metric) with columns diff, lo, hi."""
    hib = METRICS[metric]
    d = diffs
    if d["lo"].isna().any():
        return "no_ci"
    good = (d["lo"] > 0) if hib else (d["hi"] < 0)
    bad = (d["hi"] < 0) if hib else (d["lo"] > 0)
    return "improved" if good.all() else "worse" if bad.all() else "inconclusive"


def compare_to_baseline(models: list[ModelEval], *, n_boot: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-run paired differences, per-intervention summary with seed mean/sd and verdicts). Baseline runs are the reference
    for the non-baseline run with the SAME seed (same data order, initialisation seed)."""
    check_same_images(models)
    base = {m.seed: m for m in models if m.intervention == "baseline"}
    if not base:
        raise ValueError("no baseline model among the models to compare")
    runs = []
    for m in models:
        if m.intervention == "baseline":
            continue
        ref = base.get(m.seed) or next(iter(base.values()))
        d = paired_bootstrap(ref, m, n_boot, seed + m.seed)
        d.insert(0, "model", m.name); d.insert(1, "intervention", m.intervention); d.insert(2, "seed", m.seed)
        d.insert(3, "reference", ref.name)
        runs.append(d)
    per_run = pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()
    rows = []
    for (iv, metric), g in per_run.groupby(["intervention", "metric"], sort=False):
        rows.append({"intervention": iv, "metric": metric, "n_seeds": len(g), "mean_diff": g["diff"].mean(),
                     "sd_diff": g["diff"].std(ddof=1) if len(g) > 1 else np.nan, "min_diff": g["diff"].min(), "max_diff": g["diff"].max(),
                     "ci_lo_min_seed": g["lo"].min(), "ci_hi_max_seed": g["hi"].max(),
                     "higher_is_better": METRICS[metric], "verdict": verdict(g, metric)})
    return per_run, pd.DataFrame(rows)


def select_final(summary: pd.DataFrame, n_conditions: dict[str, int], *, max_clean_drop: float = 0.02) -> dict:
    """PRE-DECLARED selection rule (applied to VALIDATION comparisons only; written to disk before any test evaluation):
    among non-baseline interventions with (a) mean clean-AUROC change >= -max_clean_drop and clean AUROC verdict != 'worse' and
    (b) plausible-tier AUROC verdict == 'improved', choose the largest mean plausible-AUROC gain; ties -> fewer trained conditions.
    If none qualifies the answer is 'baseline': robust training is NOT adopted."""
    cands, table = [], []
    for iv, g in summary.groupby("intervention", sort=False):
        s = g.set_index("metric")
        clean_ok = s.loc["clean_auroc", "mean_diff"] >= -max_clean_drop and s.loc["clean_auroc", "verdict"] != "worse"
        rob_ok = s.loc["plaus_auroc", "verdict"] == "improved"
        table.append({"intervention": iv, "clean_auroc_diff": float(s.loc["clean_auroc", "mean_diff"]), "clean_ok": bool(clean_ok),
                      "plaus_auroc_diff": float(s.loc["plaus_auroc", "mean_diff"]), "robust_improved": bool(rob_ok)})
        if clean_ok and rob_ok:
            cands.append((-float(s.loc["plaus_auroc", "mean_diff"]), n_conditions.get(iv, 99), iv))
    chosen = sorted(cands)[0][2] if cands else "baseline"
    return {"rule": {"max_clean_auroc_drop": max_clean_drop, "needs": "plaus_auroc verdict improved in every seed AND clean AUROC "
                     "not worse; ties -> fewer trained conditions; none qualifies -> baseline"},
            "candidates": table, "chosen": chosen}
