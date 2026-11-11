"""Explainability figures. Each panel prints the model output next to the image so maps are never read without the prediction."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from liverguard.plotstyle import AQUA, BLUE, DIV, INK2, ORANGE, RC, save_fig


def _overlay(ax, img, heat, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.5):
    ax.imshow(img); ax.imshow(heat, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)


OCC_MIN_RANGE = 0.1


def plot_case_panels(cases: list[dict], path: Path, title: str, watermark=None) -> Path:
    """One row per case: original | Grad-CAM (HCC) | Grad-CAM (hemangioma) | occlusion (predicted class) | ROI-only | background-only."""
    cols = ["original", "Grad-CAM: HCC", "Grad-CAM: hemangioma", "occlusion Δp\n(predicted class)", "ROI only", "background only"]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(cases), 6, figsize=(15, 3.3 * len(cases) + 0.6), squeeze=False,
                                 gridspec_kw={'hspace': 0.45})
        for r, c in enumerate(cases):
            for ax in axes[r]:
                ax.axis("off"); ax.grid(False)
            axes[r, 0].imshow(c["image"])
            axes[r, 0].set_title(f"{c['group']}\n{c['image_id']} · true {c['true']}\np(HCC)={c['p']:.2f} → {c['pred']}", fontsize=7.5, loc="left")
            _overlay(axes[r, 1], c["image"], c["cam_hcc"]); _overlay(axes[r, 2], c["image"], c["cam_hem"])
            raw = float(np.abs(c["occ"]).max())
            lim = max(raw, OCC_MIN_RANGE)       # floor: do not stretch pure noise (|dp| < 0.1) to full colour range
            axes[r, 3].imshow(c["image"]); im = axes[r, 3].imshow(c["occ"], cmap=DIV.reversed(), alpha=0.6, vmin=-lim, vmax=lim)
            axes[r, 4].imshow(c["roi_only"]); axes[r, 5].imshow(c["background_only"])
            axes[r, 4].set_title(f"p(HCC)={c['p_roi_only']:.2f}", fontsize=8, loc="left")
            axes[r, 5].set_title(f"p(HCC)={c['p_background_only']:.2f}", fontsize=8, loc="left")
            axes[r, 3].set_title(f"max |Δp| {raw:.2f} (colour ±{lim:.2f})", fontsize=8, loc="left")
        for j, t in enumerate(cols):
            axes[0, j].annotate(t, (0, 1.0), xycoords="axes fraction", xytext=(0, 62), textcoords="offset points",
                                fontsize=8.5, weight="bold", color=INK2)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, weight="bold", y=1.0)
        fig.text(0.01, 0.0, "Heatmaps show model behaviour, not medical reasoning. ROI = " + title.split("ROI: ")[-1], fontsize=7.5, color=INK2)
        return save_fig(fig, path, watermark)


def plot_masking_summary(tab: pd.DataFrame, path: Path, title: str, watermark=None) -> Path:
    """AUROC (patient-bootstrap CI) per variant, with matched-area controls drawn in a different colour and marker."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(tab) + 1.6))
        y = np.arange(len(tab))[::-1]
        for yi, (_, r) in zip(y, tab.iterrows()):
            ctl = r["variant"].startswith("control")
            col, mk = (ORANGE, "s") if ctl else (BLUE, "o")
            ax.errorbar(r["auroc"], yi, xerr=[[r["auroc"] - r["auroc_lo"]], [r["auroc_hi"] - r["auroc"]]] if np.isfinite(r["auroc_lo"]) else None,
                        fmt=mk, color=col, ms=6, capsize=3, lw=1.4)
            ax.text(1.02, yi, f"{r['auroc']:.2f}", va="center", fontsize=8, color=INK2)
        ax.axvline(0.5, color=INK2, lw=1, ls="--"); ax.text(0.5, len(tab) - 0.35, "chance", fontsize=7.5, color=INK2, ha="center")
        ax.set_yticks(y); ax.set_yticklabels(tab["variant"]); ax.set_xlim(0.3, 1.08); ax.set_xlabel("AUROC on the masked images")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.scatter([], [], marker="o", color=BLUE, label="experiment"); ax.scatter([], [], marker="s", color=ORANGE, label="matched-area random control")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        return save_fig(fig, path, watermark)


def plot_cam_faithfulness(tab: pd.DataFrame, path: Path, title: str, watermark=None) -> Path:
    """Evidence drop for the predicted class when the top-CAM region is deleted vs a matched random region (and kept-only)."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        for i, (_, r) in enumerate(tab.iterrows()):
            ctl = r["variant"].startswith("control")
            ax.errorbar(r["mean_evidence_drop"], len(tab) - 1 - i,
                        xerr=[[r["mean_evidence_drop"] - r["lo"]], [r["hi"] - r["mean_evidence_drop"]]] if np.isfinite(r["lo"]) else None,
                        fmt="s" if ctl else "o", color=ORANGE if ctl else BLUE, capsize=3, ms=6)
        ax.axvline(0, color=INK2, lw=1, ls="--")
        ax.set_yticks(range(len(tab))); ax.set_yticklabels(list(tab["variant"])[::-1])
        ax.set_xlabel("drop in probability of the originally predicted class (mean, patient-bootstrap CI)")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        return save_fig(fig, path, watermark)


def plot_sanity(corr: np.ndarray, path: Path, title: str, watermark=None) -> Path:
    c = corr[np.isfinite(corr)]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.hist(c, bins=np.linspace(-1, 1, 21), color=AQUA)
        ax.axvline(0.5, color=ORANGE, ls="--", lw=1.4); ax.text(0.52, ax.get_ylim()[1] * 0.9, "high: map mostly\nimage structure", fontsize=7.5, color=ORANGE)
        ax.set_xlabel("Spearman correlation: trained vs randomised model Grad-CAM"); ax.set_ylabel("images")
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        return save_fig(fig, path, watermark)
