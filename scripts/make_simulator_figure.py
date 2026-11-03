"""draws example simulated wafer maps for every class and saves them to figures
run from the repo root  python scripts/make_simulator_figure.py
one column per class and one row per seed so you can see what the seed changes
and what the physics keeps fixed  the last row uses the most common wm811k map size"""

from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from wm811k.simulator import GENERATORS, WaferGrid, generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "simulator_examples.png"

# column order follows the root cause table so the figure and the doc read side by side
ORDER = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]

# d34 no die blends into the page so the wafer outline comes from the pass dies
# pass is a quiet neutral so the eye goes to the fails
# fail is the one strong color in the figure
SURFACE = "#fcfcfb"
PASS_COLOR = "#d3d1ca"
FAIL_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"


def _draw(ax: plt.Axes, wafer_map, aspect: float, caption: str, cmap: ListedColormap) -> None:
    # vmin and vmax pin the three codes to the three colors even when a map has no fails
    # aspect is die height over die width so a grid of non square dies still draws a round wafer
    ax.imshow(wafer_map, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect=aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(SURFACE)
    ax.text(
        0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=MUTED
    )


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["simulator"]
    assert set(ORDER) == set(GENERATORS)
    seeds = cfg["figure_seeds"]
    grid = WaferGrid(rows=cfg["rows"], cols=cfg["cols"])
    wm_grid = WaferGrid(rows=cfg["wm811k_rows"], cols=cfg["wm811k_cols"])
    cmap = ListedColormap([SURFACE, PASS_COLOR, FAIL_COLOR])

    # one row per seed on the tuning grid then one row on the wm811k grid with the first seed
    layouts = [(grid, s) for s in seeds] + [(wm_grid, seeds[0])]
    n_rows, n_cols = len(layouts), len(ORDER)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.7 * n_cols, 1.95 * n_rows + 0.9),
        facecolor=SURFACE,
        squeeze=False,
    )
    for c, pattern in enumerate(ORDER):
        for r, (g, seed) in enumerate(layouts):
            w = generate(pattern, g, seed=seed)
            caption = f"{g.rows}x{g.cols}  seed {seed}  yield {w.die_yield:.2f}"
            _draw(axes[r][c], w.wafer_map, g.pitch_y_mm / g.pitch_x_mm, caption, cmap)
            if r == 0:
                axes[r][c].set_title(pattern, fontsize=10, color=TEXT, pad=4)

    fig.legend(
        handles=[Patch(color=PASS_COLOR, label="pass"), Patch(color=FAIL_COLOR, label="fail")],
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
    )
    fig.suptitle(
        f"Simulated wafer maps for each failure pattern, from one process fault each, "
        f"on a {grid.wafer_diameter_mm:.0f} mm wafer with default fault settings. "
        f"Bottom row uses the most common WM-811K map size ({wm_grid.rows} by {wm_grid.cols} dies).",
        x=0.01,
        ha="left",
        fontsize=10,
        color=TEXT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
