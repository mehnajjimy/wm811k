"""lot level view  clustering inside lots  runs across slots  excursion lots
run from the repo root  python scripts/run_lots.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter, NullFormatter

from wm811k import viz
from wm811k.data import CLASSES, load
from wm811k.lots import analysis_pool, excursion_lots, lot_clustering, slot_order

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"


def ratio_panel(ax, table: pd.DataFrame, title: str, xlabel: str, log: bool) -> None:
    """one dot per class for the real ratio with a grey band for what chance gives"""
    order = table.sort_values("ratio").index.tolist()
    y = np.arange(len(order))
    t = table.loc[order]
    # the grey band is the middle 95 percent of the shuffled runs
    ax.hlines(y, t["null_ratio_lo"], t["null_ratio_hi"], color=viz.GRID, linewidth=7)
    ax.scatter(
        t["ratio"],
        y,
        s=60,
        zorder=3,
        color=[viz.CLASS_COLORS[c] for c in order],
        edgecolor=viz.SURFACE,
        linewidth=1.5,
    )
    # labels sit to the right of each dot so they never land on the 1x line
    for yi, v in zip(y, t["ratio"]):
        ax.annotate(
            f"{v:.1f}x",
            (v, yi),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=viz.TEXT_SECONDARY,
        )
    ax.axvline(1, color=viz.TEXT_SECONDARY, linewidth=0.8)
    # clustering spans two orders of magnitude so it needs a log axis  slot
    # order stays close to one so a plain axis reads better there
    if log:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}x"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y, order)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    viz.setup()
    pool = analysis_pool(load(ROOT / cfg["data"]["pickle"]))

    clustering = lot_clustering(pool, CLASSES, seed=cfg["seed"])
    order = slot_order(pool, CLASSES, seed=cfg["seed"])
    excursions = excursion_lots(pool)
    clustering.to_csv(OUT / "lots_clustering.csv")
    order.to_csv(OUT / "lots_slot_order.csv")
    excursions.to_csv(OUT / "lots_excursions.csv", index=False)

    lots_in_pool = pool["lot"].nunique()
    facts = {
        "lots_in_pool": lots_in_pool,
        "wafers_in_pool": len(pool),
        "excursion_lots": excursions["lot"].nunique(),
        "excursion_lot_share": excursions["lot"].nunique() / lots_in_pool,
    }
    pd.Series(facts, name="value").rename_axis("fact").to_csv(OUT / "lots_facts.csv")
    per_class = excursions.groupby("label").size().rename("excursion_lots")
    per_class.to_csv(OUT / "lots_excursions_by_class.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ratio_panel(
        axes[0],
        clustering,
        "piling up inside lots",
        "same pattern wafers per lot vs chance",
        log=True,
    )
    ratio_panel(
        axes[1],
        order,
        "running in neighbouring slots",
        "next slot same pattern vs chance",
        log=False,
    )
    fig.text(
        0.01,
        0.01,
        "grey band is what 95% of shuffled runs give  1x means no different from chance",
        fontsize=8,
        color=viz.TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "lots_clustering.png", dpi=160)
    plt.close(fig)

    print(pd.Series(facts).to_string())
    print(clustering.round(3).to_string())
    print(order.round(3).to_string())
    print(per_class.to_string())


if __name__ == "__main__":
    main()
