"""M6 figures. Every figure carries the watermark of non-result runs; differences are always model minus baseline."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from liverguard.improve.compare import METRICS, PRIMARY
from liverguard.improve.interventions import conditions_of
from liverguard.plotstyle import BLUE, DIV, INK2, ORANGE, RC, STRESS, save_fig

LABEL = {"clean_auroc": "clean AUROC", "clean_accuracy": "clean accuracy", "plaus_auroc": "AUROC, plausible tier",
         "plaus_worst_auroc": "worst-case AUROC, plausible", "plaus_flip_rate": "flip rate, plausible",
         "clean_ece": "clean ECE", "plaus_ece": "ECE, plausible tier", "plaus_brier": "Brier, plausible"}


def plot_tradeoff(summary: pd.DataFrame, path: Path, title: str, wm=None) -> Path:
    """x = change in clean AUROC, y = change in plausible-tier AUROC (mean over seeds; bars = range over seeds)."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        offsets = [(6, 5), (6, -12), (-46, 5), (-46, -12), (6, 17)]
        for k, (iv, g) in enumerate(summary.groupby("intervention", sort=False)):
            s = g.set_index("metric")
            x, y = s.loc["clean_auroc"], s.loc["plaus_auroc"]
            ax.errorbar(x["mean_diff"], y["mean_diff"], xerr=[[x["mean_diff"] - x["min_diff"]], [x["max_diff"] - x["mean_diff"]]],
                        yerr=[[y["mean_diff"] - y["min_diff"]], [y["max_diff"] - y["mean_diff"]]], fmt="o", color=BLUE, ms=7, capsize=3, lw=1.2)
            ax.annotate(iv, (x["mean_diff"], y["mean_diff"]), xytext=offsets[k % len(offsets)], textcoords="offset points", fontsize=8.5, color=INK2)
        ax.axhline(0, color=INK2, lw=0.9); ax.axvline(0, color=INK2, lw=0.9)
        ax.set_xlabel("change in clean AUROC (model - baseline)"); ax.set_ylabel("change in AUROC under plausible distortions")
        ax.set_title(f"Trade-off: clean vs robust performance · {title}", loc="left", fontsize=10, weight="bold")
        ax.text(0.99, 0.97, "better on both: upper right", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        ax.text(0.01, -0.14, "Points = mean over seeds; bars = min-max over seeds (not a CI).", transform=ax.transAxes, fontsize=7.5, color=INK2)
        return save_fig(fig, path, wm)


def plot_forest(summary: pd.DataFrame, path: Path, title: str, wm=None) -> Path:
    """One panel per primary metric; right = better for every panel (sign flipped for lower-is-better metrics).
    Dot = mean difference over seeds; whisker = widest paired-bootstrap 95% CI over seeds."""
    mets = [m for m in PRIMARY if m in set(summary["metric"])]
    ivs = list(dict.fromkeys(summary["intervention"]))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(mets), figsize=(2.9 * len(mets), 0.42 * len(ivs) + 1.9), sharey=True, squeeze=False)
        for ax, met in zip(axes[0], mets):
            sgn = 1 if METRICS[met] else -1
            for i, iv in enumerate(ivs[::-1]):
                r = summary[(summary["intervention"] == iv) & (summary["metric"] == met)].iloc[0]
                x = sgn * r["mean_diff"]
                lo, hi = sorted([sgn * r["ci_lo_min_seed"], sgn * r["ci_hi_max_seed"]])
                col = {"improved": BLUE, "worse": ORANGE}.get(r["verdict"], "#8a8985")
                mk = {"improved": "o", "worse": "s"}.get(r["verdict"], "D")
                if np.isfinite(lo):
                    ax.plot([lo, hi], [i, i], color=col, lw=1.6)
                ax.plot([x], [i], mk, color=col, ms=6.5)
            ax.axvline(0, color=INK2, lw=0.9)
            ax.set_title(LABEL.get(met, met) + ("" if METRICS[met] else " (sign flipped)"), fontsize=8.5, loc="left")
            ax.set_yticks(range(len(ivs))); ax.set_yticklabels(ivs[::-1]); ax.grid(axis="y", visible=False)
        axes[0, len(mets) // 2].set_xlabel("improvement vs baseline (right = better)")
        fig.suptitle(f"Paired differences vs baseline · {title}", x=0.01, ha="left", fontsize=10, weight="bold", y=1.02)
        fig.text(0.01, -0.04, "● improved (CI excludes 0 in every seed)  ■ worse  ◆ inconclusive. Whisker = widest 95% paired-bootstrap CI over seeds.",
                 fontsize=7.5, color=INK2)
        return save_fig(fig, path, wm)


def plot_ablation_matrix(per_cond: pd.DataFrame, path: Path, title: str, wm=None) -> Path:
    """Rows = interventions, columns = audit conditions; cell = mean plausible-tier AUROC change vs baseline (mean over seeds).
    Outlined cells = conditions the intervention trained on (adaptation); other cells show transfer."""
    pl = per_cond[per_cond["tier"] == "plausible"]
    g = pl.groupby(["intervention", "seed", "condition"], sort=False)["auroc"].mean().reset_index()
    base = g[g["intervention"] == "baseline"].groupby("condition", sort=False)["auroc"].mean()
    ivs = [i for i in dict.fromkeys(g["intervention"]) if i != "baseline"]
    conds = list(base.index)
    M = np.array([[g[(g.intervention == iv) & (g.condition == c)]["auroc"].mean() - base[c] for c in conds] for iv in ivs])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(0.75 * len(conds) + 2.4, 0.55 * len(ivs) + 2.2))
        lim = max(float(np.nanmax(np.abs(M))) if M.size else 0.05, 0.05)
        im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i, iv in enumerate(ivs):
            for j, c in enumerate(conds):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color="#0b0b0b")
                if c in conditions_of(iv):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#0b0b0b", lw=1.6))
        ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=40, ha="right"); ax.set_yticks(range(len(ivs))); ax.set_yticklabels(ivs)
        ax.grid(False); fig.colorbar(im, ax=ax, label="AUROC change vs baseline (blue = better)", shrink=0.8)
        ax.set_title(f"Ablation: plausible-tier AUROC change per condition · {title}", loc="left", fontsize=10, weight="bold")
        ax.set_xlabel("Black outline = condition the intervention trained on (adaptation); no outline = transfer to unseen corruptions.",
                      fontsize=7.5, labelpad=70)
        return save_fig(fig, path, wm)


def plot_degradation(per_cond: pd.DataFrame, path: Path, title: str, wm=None, highlight: str | None = None,
                     selected: str | None = None) -> Path:
    """AUROC vs severity level per condition (mean over seeds): baseline vs one intervention (default: best plausible AUROC)."""
    g = per_cond.groupby(["intervention", "condition", "level", "tier"], sort=False)["auroc"].mean().reset_index()
    clean = per_cond.groupby("intervention")["clean_auroc"].mean()
    cand = [i for i in dict.fromkeys(g["intervention"]) if i != "baseline"]
    if highlight in (None, "baseline"):
        highlight = max(cand, key=lambda i: g[(g.intervention == i) & (g.tier == "plausible")]["auroc"].mean()) if cand else None
    conds = list(dict.fromkeys(g["condition"]))
    ncol = 4
    nrow = int(np.ceil(len(conds) / ncol))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.6 * nrow + 0.9), sharey=True, squeeze=False,
                                 gridspec_kw={"hspace": 0.6})
        for ax, c in zip(axes.ravel(), conds):
            for iv, col, mk, ls in (("baseline", BLUE, "o", "-"), (highlight, ORANGE, "s", "--")):
                if iv is None:
                    continue
                d = g[(g.intervention == iv) & (g.condition == c)].sort_values("level")
                ax.plot([0] + d["level"].tolist(), [clean[iv]] + d["auroc"].tolist(), color=col, marker=mk, ls=ls, ms=4.5, lw=1.5,
                        label=iv)
            st = g[(g.condition == c) & (g.tier == "stress")]["level"]
            if len(st):
                ax.axvspan(st.min() - .5, st.max() + .5, color=STRESS, zorder=0)
            ax.set_title(c, fontsize=8.5, loc="left"); ax.set_xticks(range(0, int(g["level"].max()) + 1)); ax.set_ylim(min(0.3, float(np.nanmin(g['auroc'])) - 0.05), 1.02)
        for ax in axes.ravel()[len(conds):]:
            ax.axis("off")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", frameon=False, ncol=2, fontsize=8.5)
        fig.supxlabel("severity level (0 = clean; shaded = stress tier)", fontsize=8.5); fig.supylabel("AUROC", fontsize=8.5)
        shown = f"{highlight} (selected)" if highlight == selected else f"{highlight} (NOT selected; best plausible AUROC, shown for reference)"
        fig.suptitle(f"AUROC under distortion: baseline vs {shown} · {title}", x=0.01, ha="left", fontsize=10, weight="bold", y=1.0)
        return save_fig(fig, path, wm)
