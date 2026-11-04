"""item g  how many missed defects a review gate saves and what it costs in review
run from the repo root after run_rf.py and run_cnn.py  python scripts/run_abstain.py
the gate threshold is picked on validation and only then applied to test"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from wm811k import abstain, evaluate, viz
from wm811k.data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"


def default_point(table: pd.DataFrame) -> dict:
    """the same two numbers for the plain most likely class with no gate"""
    p = table[[f"p_{c}" for c in CLASSES]].to_numpy()
    pred = pd.Series(np.asarray(CLASSES)[p.argmax(1)], index=table.index)
    y = table["label"]
    defect = y != "none"
    return {
        "missed_defect_share": float((pred[defect] == "none").mean()),
        "clean_flagged_per_1000": float(1000 * (pred[~defect] != "none").mean()),
        "macro_f1": evaluate.macro_f1(y, pred, CLASSES),
    }


def figure(curves: dict, points: pd.DataFrame, coverage: dict, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, curve in curves.items():
        color = viz.MODEL_COLORS[name]
        a.plot(curve["clean_flagged_per_1000"], 100 * curve["missed_defect_share"], color=color,
               lw=2, label=viz.MODEL_NAMES[name])  # fmt: skip
        row = points.loc[name]
        # open circle is the model as it is and the filled one is with the gate
        a.plot(row["default_clean_flagged_per_1000"], 100 * row["default_missed_defect_share"],
               "o", ms=8, mfc=viz.SURFACE, mec=color, mew=2)  # fmt: skip
        a.plot(row["test_clean_flagged_per_1000"], 100 * row["test_missed_defect_share"],
               "o", ms=8, color=color, mec=viz.SURFACE, mew=2)  # fmt: skip
        cov = coverage[name]
        b.plot(100 * cov["coverage"], cov["macro_f1_kept"], color=color, lw=2,
               label=viz.MODEL_NAMES[name])  # fmt: skip
    a.set_xlabel("clean wafers sent to review per 1000 clean wafers")
    a.set_ylabel("defect wafers passed as clean (%)")
    a.set_title("a stricter clean call trades review load for missed defects")
    a.set_xlim(left=0)
    a.set_ylim(bottom=0)
    a.legend(frameon=False, title="open dot model as is  filled dot with gate", title_fontsize=8)
    b.set_xlabel("wafers decided without an engineer (%)")
    b.set_ylabel("macro f1 on those wafers")
    b.set_title("sending the least sure wafers to review")
    b.invert_xaxis()
    b.legend(frameon=False)
    for ax in (a, b):
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["abstain"]
    viz.setup()
    thresholds = np.round(np.arange(cfg["gate_min"], cfg["gate_max"] + 1e-9, cfg["gate_step"]), 2)
    coverages = np.round(np.arange(0.5, 1.0001, 0.01), 2)
    # a model is included once both its validation and test scores exist
    models = [m for m in ("rf", "cnn") if (OUT / f"{m}_val_predictions.csv").exists()]

    rows, curves, coverage = {}, {}, {}
    for m in models:
        val = pd.read_csv(OUT / f"{m}_val_predictions.csv", index_col=0)
        test = pd.read_csv(OUT / f"{m}_test_predictions.csv", index_col=0)
        val_curve = abstain.gate_curve(val, CLASSES, thresholds)
        val_curve.to_csv(OUT / f"abstain_gate_val_{m}.csv", index=False)
        t = abstain.pick_gate(val_curve, cfg["max_missed_defect_share"])

        # the whole test curve is shown in the figure but nothing is picked from it
        test_curve = abstain.gate_curve(test, CLASSES, thresholds)
        test_curve.to_csv(OUT / f"abstain_gate_test_{m}.csv", index=False)
        cov = abstain.coverage_curve(test, CLASSES, coverages)
        cov.to_csv(OUT / f"abstain_coverage_test_{m}.csv", index=False)
        curves[m], coverage[m] = test_curve, cov

        at_t = test_curve.loc[test_curve["threshold"] == t].iloc[0]
        base = default_point(test)
        rows[m] = {
            "threshold_from_val": t,
            "val_missed_defect_share": float(
                val_curve.loc[val_curve["threshold"] == t, "missed_defect_share"].iloc[0]
            ),
            **{f"default_{k}": v for k, v in base.items()},
            "test_missed_defect_share": at_t["missed_defect_share"],
            "test_clean_flagged_per_1000": at_t["clean_flagged_per_1000"],
            "test_macro_f1": at_t["macro_f1"],
            # how many defect wafers the gate catches and how many clean
            # wafers it adds to review on this test set
            "test_defects": int((test["label"] != "none").sum()),
            "test_clean": int((test["label"] == "none").sum()),
        }
    points = pd.DataFrame(rows).T.rename_axis("model")
    points["defects_caught_by_gate"] = (
        (points["default_missed_defect_share"] - points["test_missed_defect_share"])
        * points["test_defects"]
    ).round()
    points["extra_clean_reviews"] = (
        (points["test_clean_flagged_per_1000"] - points["default_clean_flagged_per_1000"])
        / 1000
        * points["test_clean"]
    ).round()
    points.to_csv(OUT / "abstain_operating_points.csv")
    figure(curves, points, coverage, FIG / "abstain_tradeoff.png")
    print(points.T.to_string())


if __name__ == "__main__":
    main()
