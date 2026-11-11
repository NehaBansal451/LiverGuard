"""Shared chart style for M3+ figures (validated colour-blind-safe categorical colours, secondary encodings carried by callers)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID, STRESS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#efeeea"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#f0efec"
SEQ = LinearSegmentedColormap.from_list("lg_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("lg_div", [ORANGE, NEUTRAL, BLUE])
RC = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
      "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "font.size": 9}


def save_fig(fig, path: Path, watermark: str | None) -> Path:
    """Save with a visible watermark when the underlying model/run is not a valid result."""
    if watermark:
        fig.text(0.5, 0.5, watermark, rotation=25, ha="center", va="center", fontsize=22, color="#C0392B", alpha=0.22,
                 weight="bold")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return Path(path)
