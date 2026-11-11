"""Scientific figures for the robustness audit (matplotlib, Agg).

Colour (validated with the dataviz palette script): categorical slots 1-3 = blue / orange / aqua, always paired with a marker and a
line style because aqua is below 3:1 contrast on the surface. Magnitudes use a single-hue blue ramp; signed confidence drift uses
orange <-> neutral <-> blue. Hatched cells / hollow markers = fidelity heuristics flag most images (information likely destroyed).
Every figure from a non-result model carries a diagonal "NOT A RESULT" watermark. Tables with the same numbers are written as CSV.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from liverguard.robustness.config import RobustnessConfig  # noqa: E402
from liverguard.robustness.distortions import apply_distortion  # noqa: E402
from liverguard.robustness.fidelity import GEOMETRIC_FAMILIES, at_risk, fidelity_metrics  # noqa: E402

from liverguard.plotstyle import AQUA, BLUE, DIV, INK, INK2, ORANGE, RC, SEQ, STRESS, SURFACE, save_fig  # noqa: E402

SERIES = [("auroc", "AUROC", BLUE, "o", "-"), ("accuracy", "Accuracy", ORANGE, "s", "--"), ("f1", "F1", AQUA, "^", ":")]
_save = save_fig


def _subtitle(split: str, thr: float, cfg: RobustnessConfig) -> str:
    return (f"{split} split · threshold {thr:.2f} · levels 1-{cfg.plausible_max_level} = plausible, higher = stress test "
            f"(a-priori, not expert-validated)")


# ------------------------------------------------------------------ 1. degradation curves
def plot_degradation_curves(tables, cfg: RobustnessConfig, thr: float, split: str, path: Path, watermark=None) -> Path:
    s, clean = tables.summary, tables.clean
    names = list(dict.fromkeys(s["condition"]))
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    lmax = int(s["level"].max())
    with plt.rc_context(RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.4 * nrow + 0.9), sharey=True, squeeze=False)
        for ax, name in zip(axes.ravel(), names):
            g = s[s["condition"] == name].sort_values("level")
            ax.axvspan(cfg.plausible_max_level + 0.5, lmax + 0.5, color=STRESS, zorder=0)
            for key, _, col, mk, ls in SERIES:
                x = np.r_[0, g["level"].to_numpy()]
                yv = np.r_[clean[key], g[key].to_numpy()]
                ax.plot(x, yv, color=col, ls=ls, lw=1.6, zorder=2)
                ax.plot([0], [clean[key]], marker=mk, color=col, ms=5, zorder=3)
                hollow = g["fidelity_flag"].to_numpy()
                for lv, v, h in zip(g["level"], g[key], hollow):
                    ax.plot([lv], [v], marker=mk, ms=5.5, color=col, mfc=SURFACE if h else col, mew=1.3, zorder=3)
            ax.set_title(name, fontsize=9, color=INK, loc="left")
            ax.set_xticks(range(0, lmax + 1), ["clean"] + [str(i) for i in range(1, lmax + 1)])
            ax.set_xlim(-0.4, lmax + 0.5)
            ax.set_ylim(min(0.4, np.nanmin(s[[k for k, *_ in SERIES]].to_numpy()) - 0.03), 1.03)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("metric")
        for ax in axes[-1]:
            ax.set_xlabel("severity level")
        handles = [Line2D([0], [0], color=c, marker=m, ls=ls, lw=1.6, ms=5.5, label=lab) for _, lab, c, m, ls in SERIES]
        handles += [Line2D([0], [0], marker="o", color=INK2, mfc=SURFACE, ls="", ms=5.5, label="hollow = fidelity flag"),
                    Patch(color=STRESS, label="stress tier")]
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle("Metric degradation under distortion", y=1.07, fontsize=12, weight="bold", color=INK)
        fig.text(0.5, 1.035, _subtitle(split, thr, cfg), ha="center", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return _save(fig, path, watermark)


# ------------------------------------------------------------------ heatmap helper
def _heat(ax, M, rows, cols, cmap, norm, fmt, flags, pm, title, text=None):
    """M colours the cells; `text` (default M) is what is printed. Text colour is chosen from the CELL colour."""
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"L{c}" for c in cols])
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    T = M if text is None else text
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]) or np.isnan(T[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=INK2)
                continue
            rgb = np.asarray(cmap(norm(M[i, j])))[:3]
            dark = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.5
            ax.text(j, i, fmt.format(T[i, j]), ha="center", va="center", fontsize=6.5, color="white" if dark else INK)
            if flags[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////", edgecolor=INK2, lw=0, alpha=.55))
    if 0 < pm < len(cols):
        ax.axvline(pm - 0.5, color=INK, ls="--", lw=1)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK, pad=18)
    return im


def plot_heatmaps(tables, cfg: RobustnessConfig, thr: float, split: str, path: Path, watermark=None) -> Path:
    s, order = tables.summary, tables.conditions["condition"].tolist()
    levels = sorted(s["level"].unique())

    def mat(col):
        return np.array([[s[(s.condition == c) & (s.level == k)][col].mean() for k in levels] for c in order], dtype=float)
    flags = np.array([[bool(s[(s.condition == c) & (s.level == k)]["fidelity_flag"].any()) for k in levels] for c in order])
    d_auc = mat("d_auroc")
    drop = np.clip(-d_auc, 0, None)                 # colour = size of the AUROC drop; printed text = the signed change
    vmax = max(0.05, float(np.nanmax(np.abs(mat("conf_drift")))))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.42 * len(order) + 2.4), sharey=True)
        ims = [
            _heat(axes[0], mat("pfr"), order, levels, SEQ, plt.Normalize(0, 1), "{:.2f}", flags, cfg.plausible_max_level,
                  "Prediction flip rate"),
            _heat(axes[1], drop, order, levels, SEQ, plt.Normalize(0, max(0.05, float(np.nanmax(drop)))), "{:+.2f}", flags,
                  cfg.plausible_max_level, "AUROC change vs clean (darker = larger drop)", text=d_auc),
            _heat(axes[2], mat("conf_drift"), order, levels, DIV, TwoSlopeNorm(0, -vmax, vmax), "{:+.2f}", flags,
                  cfg.plausible_max_level, "Confidence drift (orange = less sure)"),
        ]
        for ax, im in zip(axes, ims):
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for ax in axes:
            ax.text(cfg.plausible_max_level / 2 - 0.5, -0.62, "plausible", ha="center", fontsize=7.5, color=INK2)
            ax.text((len(levels) + cfg.plausible_max_level) / 2 - 0.5, -0.62, "stress", ha="center", fontsize=7.5, color=INK2)
        fig.suptitle("Where the model flips, loses discrimination, and changes its confidence", y=1.03, fontsize=12,
                     weight="bold", color=INK)
        fig.text(0.5, 0.985, _subtitle(split, thr, cfg) + " · hatched = fidelity flag; rows ranked most→least disruptive",
                 ha="center", fontsize=8, color=INK2)
        fig.tight_layout()
        return _save(fig, path, watermark)


# ------------------------------------------------------------------ ranking
def plot_ranking(tables, cfg: RobustnessConfig, thr: float, split: str, path: Path, watermark=None) -> Path:
    c = tables.conditions.iloc[::-1].reset_index(drop=True)          # most disruptive at the top
    y = np.arange(len(c))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.38 * len(c) + 2.2), sharey=True)
        specs = [("mean_d_auroc_all", "mean_d_auroc_all_lo", "mean_d_auroc_all_hi", "mean_d_auroc_plausible",
                  "Mean AUROC change over all levels", "{:+.3f}"),
                 ("mean_pfr_all", "mean_pfr_all_lo", "mean_pfr_all_hi", "mean_pfr_plausible",
                  "Mean flip rate over all levels", "{:.3f}")]
        for ax, (v, lo, hi, pl, title, fmt) in zip(axes, specs):
            val = c[v].to_numpy(float)
            ax.barh(y, val, color=BLUE, height=0.55, zorder=2)
            err = np.vstack([np.nan_to_num(val - c[lo].to_numpy(float)), np.nan_to_num(c[hi].to_numpy(float) - val)])
            ax.errorbar(val, y, xerr=np.abs(err), fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
            ax.scatter(c[pl], y, marker="D", s=22, facecolor=SURFACE, edgecolor=INK, linewidth=1.1, zorder=4)
            for yi, x in zip(y, val):
                if not np.isnan(x):
                    ax.text(x + (0.004 if x >= 0 else -0.004), yi + 0.34, fmt.format(x), fontsize=6.5, color=INK2,
                            ha="left" if x >= 0 else "right", va="center")
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_title(title, loc="left", fontsize=9.5)
            ax.set_yticks(y, c["condition"])
        handles = [Patch(color=BLUE, label="all levels (95% patient-bootstrap CI)"),
                   Line2D([0], [0], marker="D", color=INK, mfc=SURFACE, ls="", ms=5, label="plausible levels only")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Distortion sensitivity ranking (most disruptive at top)", y=1.02, fontsize=12, weight="bold")
        fig.text(0.5, 0.965, _subtitle(split, thr, cfg), ha="center", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        return _save(fig, path, watermark)


# ------------------------------------------------------------------ minimum flip severity
def plot_first_flip(tables, cfg: RobustnessConfig, thr: float, split: str, path: Path, watermark=None) -> Path:
    c = tables.conditions
    L = int(c["n_levels"].max())
    M = np.array([[r.get(f"first_flip_cdf_L{k}", np.nan) for k in range(1, L + 1)] for _, r in c.iterrows()], dtype=float)
    flags = np.zeros_like(M, dtype=bool)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(c) + 2.2))
        im = _heat(ax, M, c["condition"].tolist(), list(range(1, L + 1)), SEQ, plt.Normalize(0, 1), "{:.2f}", flags,
                   cfg.plausible_max_level, "Share of images flipped at or before level k")
        for i, lab in enumerate(c["mfs_median_label"]):
            ax.text(L - 0.35, i, f"median MFS {lab}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(-0.5, L + 1.4)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        fig.suptitle("Minimum flip severity (MFS)", y=1.02, fontsize=12, weight="bold")
        fig.text(0.5, 0.975, _subtitle(split, thr, cfg) + " · '>5' = fewer than half of images ever flip", ha="center",
                 fontsize=8, color=INK2)
        fig.tight_layout()
        return _save(fig, path, watermark)


# ------------------------------------------------------------------ gallery
def plot_gallery(image, image_id: str, cfg: RobustnessConfig, seed: int, path: Path, watermark=None) -> Path:
    """One image under every condition x level, for expert review of label preservation. '!' + dashed frame = fidelity flag."""
    conds = cfg.conditions
    L = max(len(c.params) for c in conds)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(conds), L + 1, figsize=(1.25 * (L + 1) + 1.6, 1.2 * len(conds) + 0.8), squeeze=False)
        for r, c in enumerate(conds):
            for k in range(0, L + 1):
                ax = axes[r, k]
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if k > len(c.params):
                    ax.axis("off")
                    continue
                if k == 0:
                    ax.imshow(image)
                    ax.set_ylabel(c.name, rotation=0, ha="right", va="center", fontsize=8, labelpad=4)
                else:
                    d = apply_distortion(image, c.func, c.params[k - 1], seed=seed, image_id=image_id, condition=c.name)
                    ax.imshow(d)
                    if at_risk(fidelity_metrics(image, d), cfg.fidelity, geometric=c.family in GEOMETRIC_FAMILIES):
                        ax.text(0.06, 0.94, "!", transform=ax.transAxes, color="white", fontsize=11, weight="bold", va="top",
                                bbox=dict(boxstyle="round,pad=0.1", fc=INK, ec="none"))
                        for sp in ax.spines.values():
                            sp.set_linestyle("--"); sp.set_edgecolor(INK); sp.set_linewidth(1.5)
                if r == 0:
                    ax.set_title("clean" if k == 0 else f"L{k}" + (" *" if k > cfg.plausible_max_level else ""), fontsize=8)
        fig.suptitle(f"Distortion gallery: {image_id}", y=1.015, fontsize=11, weight="bold")
        fig.text(0.5, 0.982, "* stress tier · '!' / dashed frame = fidelity heuristics suggest information may be destroyed "
                 "(proxy only; expert review needed)", ha="center", fontsize=7.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return _save(fig, path, watermark)


def make_all(out: Path, tables, cfg: RobustnessConfig, thr: float, split: str, watermark, sample_images, sample_ids, seed) -> dict:
    fig_dir = out / "figures"
    paths = {
        "fig_degradation": plot_degradation_curves(tables, cfg, thr, split, fig_dir / "degradation_curves.png", watermark),
        "fig_heatmaps": plot_heatmaps(tables, cfg, thr, split, fig_dir / "flip_auroc_confidence_heatmaps.png", watermark),
        "fig_ranking": plot_ranking(tables, cfg, thr, split, fig_dir / "sensitivity_ranking.png", watermark),
        "fig_first_flip": plot_first_flip(tables, cfg, thr, split, fig_dir / "minimum_flip_severity.png", watermark)}
    for n, (img, iid) in enumerate(zip(sample_images, sample_ids), 1):
        paths[f"fig_gallery_{n}"] = plot_gallery(img, iid, cfg, seed, fig_dir / f"distortion_gallery_{n}.png", watermark)
    return paths
