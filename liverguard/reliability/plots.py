"""Reliability figures (matplotlib). Every figure states split, n and the method limitation that matters for reading it."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from liverguard.plotstyle import AQUA, BLUE, GRID, INK2, ORANGE, RC, save_fig


def plot_reliability(tables: dict[str, pd.DataFrame], stats: dict[str, dict], path: Path, title: str, watermark=None) -> Path:
    """Top-label reliability diagram per probability set (e.g. uncalibrated vs calibrated) with a bin-count strip."""
    with plt.rc_context(RC):
        k = len(tables)
        fig = plt.figure(figsize=(4.6 * k, 5.4))
        gs = fig.add_gridspec(2, k, height_ratios=[4, 1], hspace=0.08, wspace=0.28)
        for i, (name, t) in enumerate(tables.items()):
            ax, ax2 = fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])
            ax.plot([0.5, 1], [0.5, 1], "--", color=INK2, lw=1, label="perfect calibration")
            tt = t[t["n"] > 0]
            ax.plot(tt["confidence"], tt["accuracy"], "o-", color=BLUE, lw=1.6, ms=5, label="observed")
            ax.set_xlim(0.5, 1.0); ax.set_ylim(-0.02, 1.02)
            ax.set_xticklabels([]); ax.set_ylabel("accuracy of predicted class")
            s = stats[name]
            ax.set_title(f"{name}\nECE {s['ece_width']:.3f} · Brier {s['brier']:.3f} · n={s['n']}", fontsize=9, loc="left")
            if i == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=8)
            ax2.bar((t["lo"] + t["hi"]) / 2, t["n"], width=(t["hi"] - t["lo"]) * 0.9, color=INK2, alpha=0.6)
            ax2.set_xlim(0.5, 1.0); ax2.set_xlabel("confidence = max(p, 1-p)"); ax2.set_ylabel("images")
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, weight="bold")
        return save_fig(fig, path, watermark)


def plot_confidence_distribution(df: pd.DataFrame, path: Path, title: str, watermark=None) -> Path:
    """Confidence histograms for correct vs incorrect predictions (different line styles + colours)."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"wspace": 0.25})
        bins = np.linspace(0.5, 1.0, 11)
        conf = np.maximum(df["p"], 1 - df["p"])
        for ax, (col, lab) in zip(axes, [("conf", "confidence max(p, 1-p)"), ("entropy", "predictive entropy (bits)")]):
            v = conf if col == "conf" else df["entropy"]
            b = bins if col == "conf" else np.linspace(0, 1, 11)
            ax.hist(v[df["correct"]], bins=b, color=BLUE, alpha=0.75, label=f"correct (n={int(df['correct'].sum())})")
            ax.hist(v[~df["correct"]], bins=b, histtype="step", color=ORANGE, lw=2, ls="--",
                    label=f"incorrect (n={int((~df['correct']).sum())})")
            ax.set_xlabel(lab); ax.set_ylabel("images"); ax.legend(frameon=False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, weight="bold")
        return save_fig(fig, path, watermark)


def plot_risk_coverage(curves: dict[str, pd.DataFrame], oracle: pd.DataFrame, base_risk: float, summary: pd.DataFrame,
                       path: Path, title: str, watermark=None) -> Path:
    styles = [(BLUE, "-", "o"), (ORANGE, "--", "s"), (AQUA, ":", "^"), ("#8a6bd1", "-.", "D")]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for (name, rc), (c, ls, m) in zip(curves.items(), styles):
            ax.plot(rc["coverage"], rc["risk"], color=c, ls=ls, lw=1.8, label=f"{name} (AURC {summary.set_index('score').loc[name, 'aurc']:.3f})")
        ax.plot(oracle["coverage"], oracle["risk"], color=INK2, lw=1, ls=(0, (1, 3)), label="oracle ranking")
        ax.axhline(base_risk, color=GRID, lw=1.5, label=f"no abstention / random ({base_risk:.3f})")
        ax.set_xlabel("coverage (fraction of cases kept, most certain first)"); ax.set_ylabel("error rate among kept cases")
        ax.set_xlim(0, 1.0); ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        return save_fig(fig, path, watermark)


def plot_error_by_uncertainty(tab: pd.DataFrame, path: Path, title: str, watermark=None) -> Path:
    scores = list(dict.fromkeys(tab["score"]))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(scores), figsize=(3.6 * len(scores), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, scores):
            t = tab[tab["score"] == s]
            ax.bar(t["group"] + 1, t["error_rate"], color=BLUE)
            for g, r, n in zip(t["group"] + 1, t["error_rate"], t["n"]):
                ax.text(g, r + 0.01, f"n={n}", ha="center", fontsize=7, color=INK2)
            ax.set_title(s, fontsize=9, loc="left"); ax.set_xlabel("uncertainty group (1 = most certain)")
            ax.set_xticks(t["group"] + 1)
        axes[0].set_ylabel("error rate"); axes[0].set_ylim(0, 1.05)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, weight="bold")
        return save_fig(fig, path, watermark)


def plot_failure_gallery(items: list[dict], path: Path, title: str, watermark=None) -> Path:
    """Thumbnails of the most confident errors. items: dict(image, caption)."""
    n = max(len(items), 1)
    cols = min(n, 4); rows = int(np.ceil(n / cols))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, it in zip(axes.ravel(), items):
            ax.imshow(it["image"]); ax.set_title(it["caption"], fontsize=7.5, loc="left")
        if not items:
            axes[0, 0].text(0.5, 0.5, "no errors in this split", ha="center", transform=axes[0, 0].transAxes)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, weight="bold", y=1.0)
        return save_fig(fig, path, watermark)
