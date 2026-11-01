"""Scientific plots (matplotlib, Agg). Runs that are not valid results get a visible watermark."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

NAVY, TEAL, LAVENDER, GREY, BG = "#1F2A44", "#3E8E8B", "#9A8FC2", "#8A8F98", "#FAF8F5"
plt.rcParams.update({"axes.facecolor": BG, "figure.facecolor": BG, "axes.edgecolor": GREY,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _finish(fig, path: Path, watermark: str | None) -> Path:
    if watermark:
        fig.text(0.5, 0.5, watermark, rotation=25, ha="center", va="center", fontsize=20,
                 color="#C0392B", alpha=0.25, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def plot_training_curves(history, path: Path, title: str, watermark: str | None = None,
                         best_epoch: int | None = None) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("loss", "Loss"), ("acc", "Accuracy"), ("auroc", "AUROC")]
    for ax, (key, label) in zip(axes, panels):
        if f"train_{key}" in history:
            ax.plot(history["epoch"], history[f"train_{key}"], color=NAVY, label="train")
        if f"val_{key}" in history:
            ax.plot(history["epoch"], history[f"val_{key}"], color=TEAL, label="validation")
        frozen = history["epoch"][history["phase"] == "frozen"]
        if len(frozen):
            ax.axvspan(frozen.min() - .5, frozen.max() + .5, color=LAVENDER, alpha=.18, label="head-only")
        if best_epoch is not None:
            ax.axvline(best_epoch, color=GREY, ls="--", lw=1, label="selected")
        ax.set_xlabel("epoch")
        ax.set_title(label)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=NAVY, weight="bold", y=1.05)
    return _finish(fig, path, watermark)


def plot_confusion_matrix(cm, class_names: list[str], path: Path, title: str,
                          watermark: str | None = None) -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="GnBu")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color=NAVY if v < cm.max() * .6 else "white", weight="bold")
    ax.set_title(title, color=NAVY, fontsize=10)
    return _finish(fig, path, watermark)


def plot_roc(y, p, path: Path, title: str, auroc: float, ci: list[float] | None = None,
             watermark: str | None = None) -> Path:
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(fpr, tpr, color=TEAL, lw=2)
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    txt = f"AUROC {auroc:.3f}" + (f"  (95% CI {ci[0]:.3f}-{ci[1]:.3f})" if ci else "")
    ax.text(0.97, 0.05, txt, ha="right", fontsize=9, color=NAVY)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity (HCC)")
    ax.set_title(title, color=NAVY, fontsize=10)
    return _finish(fig, path, watermark)
