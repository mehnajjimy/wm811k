"""yield loss per defect pattern  table and pareto figures
run from the repo root  python scripts/run_yield.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wm811k import viz
from wm811k.data import load
from wm811k.yield_stats import loss_by_class, per_wafer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"


def pareto_figure(table: pd.DataFrame, baseline: float, path: Path) -> None:
    """bar per defect class of extra dies lost per 1000 wafers biggest first
    with the running share of total loss written on each bar"""
    defects = table.drop(index="none")
    values = defects["excess_dies_per_1000_wafers"]
    running = defects["share_of_excess_loss"].cumsum()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = viz.ACCENT
    ax.bar(defects.index, values, color=colors, width=0.7, edgecolor=viz.SURFACE, linewidth=2)
    # the running share sits as text above each bar instead of a second axis
    # so there is only one scale to read
    for i, (v, r) in enumerate(zip(values, running)):
        # a pattern that loses fewer dies than a clean wafer has a negative bar
        # so its label goes just above zero where it does not hit the class name
        ax.text(
            i,
            max(v, 0),
            f"{v:,.0f}\n{r:.0%} cum.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=viz.TEXT_SECONDARY,
        )
    ax.axhline(0, color=viz.TEXT_SECONDARY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("extra dies lost per 1,000 labeled wafers")
    ax.set_title("which defect patterns cost the most dies")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(min(values.min(), 0) * 1.3, values.max() * 1.22)
    ax.tick_params(axis="x", length=0)
    fig.text(
        0.01,
        0.01,
        f"extra loss is measured against the median yield of clean wafers of the same product (overall clean median {baseline:.1%})",
        fontsize=8,
        color=viz.TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def yield_spread_figure(wafers: pd.DataFrame, order: list[str], path: Path) -> None:
    """box per class showing how yield spreads across wafers"""
    labeled = wafers[wafers["label"].notna()]
    data = [labeled.loc[labeled["label"] == c, "yield"].to_numpy() * 100 for c in order]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    box = ax.boxplot(
        data,
        orientation="horizontal",
        tick_labels=order,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops={"color": viz.TEXT, "linewidth": 1.5},
    )
    for patch, c in zip(box["boxes"], order):
        patch.set_facecolor(viz.CLASS_COLORS[c])
        patch.set_alpha(0.85)
        patch.set_edgecolor(viz.SURFACE)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.set_xlabel("die yield per wafer (%)")
    ax.set_title("yield spread by defect pattern")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    fig.text(
        0.01,
        0.01,
        "box is the middle half of wafers  line is the median  whiskers 1.5 iqr",
        fontsize=8,
        color=viz.TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    viz.setup()

    wafers = per_wafer(load(ROOT / "LSWMD.pkl"))
    table, baseline = loss_by_class(wafers)
    table.to_csv(OUT / "yield_by_class.csv")

    # labeled and unlabeled wafers compared so the readme can say whether the
    # labeled subset looks like the whole fab population or not
    wafers["labeled"] = np.where(wafers["label"].notna(), "labeled", "unlabeled")
    pop = wafers.groupby("labeled")["yield"].describe(percentiles=[0.25, 0.5, 0.75])
    pop.to_csv(OUT / "yield_labeled_vs_unlabeled.csv")

    pd.Series({"clean_baseline_yield_median": baseline}, name="value").rename_axis("fact").to_csv(
        OUT / "yield_facts.csv"
    )

    pareto_figure(table, baseline, FIG / "yield_pareto.png")
    # clean wafers go first as the reference then the rest from best to worst yield
    order = ["none"] + table.drop(index="none").sort_values(
        "yield_median", ascending=False
    ).index.tolist()
    yield_spread_figure(wafers, order, FIG / "yield_spread.png")

    print(f"clean baseline median yield {baseline:.4f}")
    print(table.round(3).to_string())
    print(pop.round(4).to_string())


if __name__ == "__main__":
    main()
