"""item h  test wafers whose label looks wrong
run from the repo root after run_rf.py and run_cnn.py  python scripts/run_label_audit.py
a wafer is a candidate when the forest and the cnn both call it the same other
class and both are sure  two models built in very different ways agreeing
against the label is a hint the label may be the problem
the verdict column is left empty for a person to fill in by looking at the maps"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap

from wm811k import viz
from wm811k.data import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures" / "label_audit"
PASS_COLOR = "#d3d1ca"
FAIL_COLOR = "#2a78d6"


def candidates(rf: pd.DataFrame, cn: pd.DataFrame, min_p: float, n: int) -> pd.DataFrame:
    """maps where both models agree on one class that is not the label"""
    agree = (rf["pred"] == cn["pred"]) & (rf["pred"] != rf["label"])
    # each model's probability for the class they agreed on
    p_rf = np.array([rf.at[i, f"p_{c}"] for i, c in zip(rf.index, rf["pred"], strict=True)])
    p_cn = np.array([cn.at[i, f"p_{c}"] for i, c in zip(cn.index, cn["pred"], strict=True)])
    out = pd.DataFrame(
        {
            "lot": rf["lot"],
            "label": rf["label"],
            "both_say": rf["pred"],
            "p_rf": p_rf,
            "p_cnn": p_cn,
        }
    )
    out = out[agree & (p_rf >= min_p) & (p_cn >= min_p)]
    # the surest pairs first since those are the strongest hints
    out = out.assign(p_min=out[["p_rf", "p_cnn"]].min(axis=1)).sort_values("p_min", ascending=False)
    return out.head(n).drop(columns="p_min")


def sheets(cands: pd.DataFrame, maps: pd.Series, per_sheet: int = 20) -> list[Path]:
    """contact sheets of the raw maps at their own size  numbered to match the csv"""
    FIG.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap([viz.SURFACE, PASS_COLOR, FAIL_COLOR])
    paths = []
    for s in range(0, len(cands), per_sheet):
        part = cands.iloc[s : s + per_sheet]
        fig, axes = plt.subplots(4, 5, figsize=(11, 10))
        for ax in axes.ravel():
            ax.axis("off")
        for k, (idx, row) in enumerate(part.iterrows()):
            ax = axes.ravel()[k]
            ax.imshow(maps[idx], cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            ax.set_title(
                f"#{s + k + 1}  label {row['label']}\nboth say {row['both_say']}", fontsize=8
            )
        fig.tight_layout()
        path = FIG / f"sheet_{s // per_sheet + 1:02d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    a = cfg["label_audit"]
    viz.setup()
    rf = pd.read_csv(OUT / "rf_test_predictions.csv", index_col=0)
    cn = pd.read_csv(OUT / "cnn_test_predictions.csv", index_col=0)
    assert rf.index.equals(cn.index), "rf and cnn test maps differ"

    cands = candidates(rf, cn, a["min_p"], a["max_candidates"])
    cands.insert(0, "number", np.arange(1, len(cands) + 1))
    cands["verdict"] = ""
    cands["note"] = ""
    cands.to_csv(OUT / "label_audit_candidates.csv")

    # how often each label to prediction swap shows up among candidates
    swaps = cands.groupby(["label", "both_say"]).size().rename("maps").sort_values(ascending=False)
    swaps.to_csv(OUT / "label_audit_swaps.csv")
    agree = (rf["pred"] == cn["pred"]) & (rf["pred"] != rf["label"])
    facts = {
        "test_maps": len(rf),
        "both_models_wrong_same_way": int(agree.sum()),
        "candidates_listed": len(cands),
        "min_p_each_model": a["min_p"],
    }
    pd.Series(facts, name="value").rename_axis("fact").to_csv(OUT / "label_audit_facts.csv")

    df = load(ROOT / cfg["data"]["pickle"])
    sheets(cands, df.loc[cands.index, "wafer_map"])
    print(pd.Series(facts).to_string())
    print(swaps.to_string())


if __name__ == "__main__":
    main()
