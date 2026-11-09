"""shared figure settings so every chart in the project looks the same"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

# one fixed colour per defect class used in every figure
# the eight defect classes take the eight validated categorical slots in order
# and none is a neutral grey because it is the clean baseline not a defect
CLASS_COLORS = {
    "Center": "#2a78d6",
    "Donut": "#eb6834",
    "Edge-Loc": "#1baf7a",
    "Edge-Ring": "#eda100",
    "Loc": "#e87ba4",
    "Near-full": "#008300",
    "Random": "#6250d6",
    "Scratch": "#e34948",
    "none": "#8a8984",
}

# models are not defect classes so they get ink and blue instead of class colours
# and every model line is labelled directly in the figure
# the accent is a royal slate blue that sits with the portfolio site palette
ACCENT = "#3e4f8a"
MODEL_COLORS = {"rf": "#52514e", "cnn": ACCENT}
MODEL_NAMES = {"rf": "random forest", "cnn": "small cnn"}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def setup() -> None:
    """applies the house style to matplotlib"""
    # light grid and no top or right frame so the data marks carry the eye
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "font.size": 10,
        }
    )


def confusion_figure(conf: pd.DataFrame, title: str, path: Path) -> None:
    """confusion matrix shown as the share of each true class so rare classes
    are as readable as none"""
    share = conf.div(conf.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = LinearSegmentedColormap.from_list("accent", [SURFACE, ACCENT])
    ax.imshow(share.to_numpy(), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(conf)), conf.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(conf)), conf.index)
    for i in range(len(conf)):
        for j in range(len(conf)):
            v = share.iat[i, j]
            if v >= 0.01:
                color = SURFACE if v > 0.55 else TEXT
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("predicted pattern")
    ax.set_ylabel("true pattern")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
