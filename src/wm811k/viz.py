"""shared figure settings so every chart in the project looks the same"""

import matplotlib.pyplot as plt

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
