"""Robustness measures and tables. Pure numpy/pandas: no model, no images (unit-testable on hand-built arrays).

Definitions (HCC = positive class; `thr` is a fixed decision threshold on p(HCC)):
- Prediction Flip Rate (PFR): share of images whose predicted class under distortion differs from the clean prediction.
  Label-free (a consistency measure). Split into flip-to-wrong (clean right -> distorted wrong) and flip-to-right.
- Confidence: probability assigned to the predicted class, max(p, 1-p). Confidence drift = mean(conf_distorted - conf_clean),
  signed (negative = the model becomes less sure). NOT clinical certainty. mean_abs_dp = mean |p_dist - p_clean| (threshold-free).
- Metric degradation: metric(distorted) - metric(clean) on the same images (AUROC is threshold-free; accuracy/F1 use `thr`).
- Minimum Flip Severity (MFS): per image, the lowest severity level at which its prediction flips; images that never flip
  within the tested levels are censored at (n_levels + 1). Reported as a median (">L" when most images never flip) and as
  the cumulative share of images flipped by level k.
- Worst case: the minimum of a metric over conditions/levels (overall, per family, and restricted to the plausible tier).
- Paired patient-level cluster bootstrap: patients are resampled, and the clean and distorted results are recomputed on the
  same resample, so intervals for degradations respect both within-patient correlation and the pairing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from liverguard.robustness.config import Condition, RobustnessConfig

AT_RISK_MAJORITY = 0.5   # a level is "fidelity-flagged" when at least this share of images trips a fidelity heuristic
STATS = ("d_auroc", "d_accuracy", "d_f1", "pfr", "conf_drift")
NAN = float("nan")


# ------------------------------------------------------------------ elementary measures
def predicted_confidence(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1.0 - p)


def prediction_flips(p_clean, p, thr: float) -> np.ndarray:
    return (np.asarray(p_clean) >= thr) != (np.asarray(p) >= thr)


def confidence_drift(p_clean, p) -> np.ndarray:
    return predicted_confidence(p) - predicted_confidence(p_clean)


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    return NAN if y.min() == y.max() else float(roc_auc_score(y, p))


def level_metrics(y, p, thr: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else NAN  # noqa: E731
    return {"auroc": _auroc(y, p), "accuracy": div(tp + tn, len(y)), "f1": div(2 * tp, 2 * tp + fp + fn),
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp)}


def flip_breakdown(y, p_clean, p, thr: float) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    c, d = np.asarray(p_clean) >= thr, np.asarray(p) >= thr
    right_c, right_d = c == (y == 1), d == (y == 1)
    return {"pfr": float(np.mean(c != d)), "flip_to_wrong": float(np.mean(right_c & ~right_d)),
            "flip_to_right": float(np.mean(~right_c & right_d)),
            "conf_drift": float(np.mean(confidence_drift(p_clean, p))),
            "mean_abs_dp": float(np.mean(np.abs(np.asarray(p) - np.asarray(p_clean))))}


def first_flip_level(flip_matrix: np.ndarray) -> np.ndarray:
    """(n_images, n_levels) bool -> per-image first flipping level (1-based); never flipping -> n_levels + 1 (censored)."""
    fm = np.asarray(flip_matrix, dtype=bool)
    return np.where(fm.any(axis=1), fm.argmax(axis=1) + 1, fm.shape[1] + 1)


def mfs_summary(first: np.ndarray, n_levels: int) -> dict:
    med = float(np.median(first))
    censored = med > n_levels
    return {"frac_ever_flipped": float(np.mean(first <= n_levels)),
            "mfs_median": NAN if censored else med,
            "mfs_median_label": f">{n_levels}" if censored else f"{med:g}",
            "cdf": [float(np.mean(first <= k)) for k in range(1, n_levels + 1)]}


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else NAN


# ------------------------------------------------------------------ paired patient-level bootstrap
def paired_bootstrap(y, groups, p_clean, P: dict[str, np.ndarray], thr: float, plausible_max: int,
                     n_boot: int, ci: float, seed: int):
    """Returns (level_ci[(cond, level)][stat] -> [lo, hi], cond_ci[cond][stat] -> [lo, hi], clean_ci[metric] -> [lo, hi])."""
    if n_boot <= 0:
        return {}, {}, {}
    y = np.asarray(y).astype(int)
    p_clean = np.asarray(p_clean, dtype=float)
    _, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    n_groups = int(inv.max()) + 1
    members = [np.flatnonzero(inv == g) for g in range(n_groups)]
    rng = np.random.default_rng(seed)
    lvl = {(c, k + 1): {s: [] for s in STATS} for c, M in P.items() for k in range(M.shape[0])}
    cnd = {c: {"mean_d_auroc_all": [], "mean_d_auroc_plausible": [], "mean_pfr_all": []} for c in P}
    cln = {m: [] for m in ("auroc", "accuracy", "f1")}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, n_groups, n_groups)])
        yb, pc = y[idx], p_clean[idx]
        if yb.min() == yb.max():
            continue
        c0 = level_metrics(yb, pc, thr)
        for m in cln:
            cln[m].append(c0[m])
        for c, M in P.items():
            d_auc, pfrs = [], []
            for k in range(M.shape[0]):
                pb = M[k, idx]
                m, fb = level_metrics(yb, pb, thr), flip_breakdown(yb, pc, pb, thr)
                vals = {"d_auroc": m["auroc"] - c0["auroc"], "d_accuracy": m["accuracy"] - c0["accuracy"],
                        "d_f1": m["f1"] - c0["f1"], "pfr": fb["pfr"], "conf_drift": fb["conf_drift"]}
                for s, v in vals.items():
                    lvl[(c, k + 1)][s].append(v)
                d_auc.append(vals["d_auroc"]); pfrs.append(fb["pfr"])
            cnd[c]["mean_d_auroc_all"].append(_nanmean(d_auc))
            cnd[c]["mean_d_auroc_plausible"].append(_nanmean(d_auc[:plausible_max]))
            cnd[c]["mean_pfr_all"].append(float(np.mean(pfrs)))
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    def interval(v) -> list[float]:
        a = np.asarray(v, dtype=float)
        a = a[~np.isnan(a)]
        return [float(np.percentile(a, lo_q)), float(np.percentile(a, hi_q))] if len(a) else [NAN, NAN]
    return ({k: {s: interval(v) for s, v in d.items()} for k, d in lvl.items()},
            {c: {s: interval(v) for s, v in d.items()} for c, d in cnd.items()},
            {m: interval(v) for m, v in cln.items()})


# ------------------------------------------------------------------ tables
@dataclass
class AuditTables:
    clean: dict
    summary: pd.DataFrame            # one row per (condition, level)
    conditions: pd.DataFrame         # one row per condition, ranked
    families: pd.DataFrame           # worst case per distortion family
    worst_case: dict                 # overall and plausible-only worst AUROC / accuracy / F1
    any_flip: dict                   # share of images flipping under >= 1 condition/level
    first_flip_cdf: dict = field(default_factory=dict)


def _worst(df: pd.DataFrame, metric: str) -> dict:
    d = df.dropna(subset=[metric])
    if d.empty:
        return {"value": NAN, "condition": None, "level": None, "tier": None}
    r = d.loc[d[metric].idxmin()]
    return {"value": float(r[metric]), "condition": r["condition"], "level": int(r["level"]), "tier": r["tier"]}


def summarize(y, groups, p_clean, conds: list[Condition], P: dict[str, np.ndarray], FID: dict[str, dict],
              cfg: RobustnessConfig, thr: float) -> AuditTables:
    y = np.asarray(y).astype(int)
    p_clean = np.asarray(p_clean, dtype=float)
    clean = level_metrics(y, p_clean, thr)
    level_ci, cond_ci, clean_ci = paired_bootstrap(y, groups, p_clean, P, thr, cfg.plausible_max_level,
                                                   cfg.n_boot, cfg.ci, cfg.seed + 1)
    clean["ci"] = clean_ci
    rows = []
    for c in conds:
        M, f = P[c.name], FID[c.name]
        for k in range(1, M.shape[0] + 1):
            m, fb = level_metrics(y, M[k - 1], thr), flip_breakdown(y, p_clean, M[k - 1], thr)
            row = {"condition": c.name, "family": c.family, "level": k, "tier": cfg.tier(k), "param": c.params[k - 1],
                   "n": len(y), "auroc": m["auroc"], "accuracy": m["accuracy"], "f1": m["f1"],
                   "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                   "d_auroc": m["auroc"] - clean["auroc"], "d_accuracy": m["accuracy"] - clean["accuracy"],
                   "d_f1": m["f1"] - clean["f1"], **fb}
            ci = level_ci.get((c.name, k), {})
            for s in STATS:
                row[f"{s}_lo"], row[f"{s}_hi"] = ci.get(s, [NAN, NAN])
            row.update({key: float(np.mean(f[key][k - 1])) for key in
                        ("psnr_db", "correlation", "std_ratio", "new_saturated_fraction")})
            row["frac_at_risk"] = float(np.mean(f["at_risk"][k - 1]))
            row["fidelity_flag"] = bool(row["frac_at_risk"] >= AT_RISK_MAJORITY)
            row["breaks_flip_alert"] = bool(row["pfr"] >= cfg.flip_alert)
            rows.append(row)
    summary = pd.DataFrame(rows)

    crow, cdfs = [], {}
    any_all = np.zeros(len(y), bool)
    any_pl = np.zeros(len(y), bool)
    for c in conds:
        M, s = P[c.name], summary[summary["condition"] == c.name].sort_values("level")
        L = M.shape[0]
        flipmat = np.stack([prediction_flips(p_clean, M[k], thr) for k in range(L)], axis=1)
        any_all |= flipmat.any(axis=1)
        if cfg.plausible_max_level > 0:
            any_pl |= flipmat[:, :cfg.plausible_max_level].any(axis=1)
        mfs = mfs_summary(first_flip_level(flipmat), L)
        cdfs[c.name] = mfs["cdf"]
        pl, hit, cc = s[s["tier"] == "plausible"], s[s["breaks_flip_alert"]], cond_ci.get(c.name, {})
        r = {"condition": c.name, "family": c.family, "n_levels": L,
             "mean_d_auroc_all": _nanmean(s["d_auroc"]), "mean_d_auroc_plausible": _nanmean(pl["d_auroc"]),
             "mean_d_auroc_all_lo": cc.get("mean_d_auroc_all", [NAN, NAN])[0],
             "mean_d_auroc_all_hi": cc.get("mean_d_auroc_all", [NAN, NAN])[1],
             "mean_pfr_all": float(s["pfr"].mean()), "mean_pfr_plausible": float(pl["pfr"].mean()) if len(pl) else NAN,
             "mean_pfr_all_lo": cc.get("mean_pfr_all", [NAN, NAN])[0], "mean_pfr_all_hi": cc.get("mean_pfr_all", [NAN, NAN])[1],
             "pfr_at_plausible_max": float(pl["pfr"].iloc[-1]) if len(pl) else NAN, "pfr_at_max_level": float(s["pfr"].iloc[-1]),
             "min_level_flip_alert": float(hit["level"].min()) if len(hit) else NAN,
             "frac_images_ever_flipped": mfs["frac_ever_flipped"], "mfs_median": mfs["mfs_median"],
             "mfs_median_label": mfs["mfs_median_label"]}
        for k, v in enumerate(mfs["cdf"], 1):
            r[f"first_flip_cdf_L{k}"] = v
        for metric in ("auroc", "accuracy", "f1"):
            w = _worst(s, metric)
            r[f"worst_{metric}"], r[f"worst_{metric}_level"] = w["value"], w["level"]
            wp = _worst(pl, metric) if len(pl) else {"value": NAN}
            r[f"worst_{metric}_plausible"] = wp["value"]
        crow.append(r)
    cdf = pd.DataFrame(crow)
    # Primary rank: most negative mean AUROC change (threshold-free); ties / undefined AUROC -> highest mean flip rate.
    order = cdf.assign(_a=cdf["mean_d_auroc_all"].fillna(np.inf), _f=-cdf["mean_pfr_all"]).sort_values(["_a", "_f"]).index
    cdf["rank"] = pd.Series(range(1, len(cdf) + 1), index=order)
    cdf["rank_by_flip_rate"] = cdf["mean_pfr_all"].rank(ascending=False, method="min").astype(int)
    cdf = cdf.sort_values("rank").reset_index(drop=True)

    fam = []
    for name, g in summary.groupby("family", sort=False):
        r = {"family": name, "conditions": ",".join(sorted(g["condition"].unique()))}
        for metric in ("auroc", "accuracy", "f1"):
            w = _worst(g, metric)
            r.update({f"worst_{metric}": w["value"], f"worst_{metric}_condition": w["condition"],
                      f"worst_{metric}_level": w["level"], f"worst_{metric}_tier": w["tier"]})
        r["max_pfr"] = float(g["pfr"].max())
        fam.append(r)
    pl_all = summary[summary["tier"] == "plausible"]
    worst = {"all_levels": {m: _worst(summary, m) for m in ("auroc", "accuracy", "f1")},
             "plausible_only": {m: _worst(pl_all, m) for m in ("auroc", "accuracy", "f1")},
             "max_pfr_all": float(summary["pfr"].max()),
             "max_pfr_plausible": float(pl_all["pfr"].max()) if len(pl_all) else NAN}
    return AuditTables(clean, summary, cdf, pd.DataFrame(fam), worst,
                       {"any_condition_all_levels": float(any_all.mean()),
                        "any_condition_plausible_levels": float(any_pl.mean())}, cdfs)
