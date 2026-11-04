"""item f prime  train on simulated wafers only and test on the real test set
run from the repo root after run_rf.py  python scripts/run_synthetic.py
no real label is used for training  the only real facts used are map sizes
and the yield spread of maps that have no label"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from wm811k import evaluate, synthetic, viz
from wm811k.data import CLASSES, load
from wm811k.features import extract_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"
CACHE = ROOT / "cache"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    s, seed = cfg["synthetic"], cfg["seed"]
    viz.setup()

    df = load(ROOT / cfg["data"]["pickle"])
    lots = pd.read_csv(OUT / "split_lots.csv", index_col="lot")
    df["split_ours"] = df["lot"].map(lots["split"])
    # map sizes come from lots outside validation and test so the test set
    # shapes nothing about the synthetic data
    pool = df[~df["split_ours"].isin(["val", "test"])]
    sizes = pool.groupby(["rows", "cols"]).size().rename("maps").reset_index()
    sizes = sizes[(sizes["rows"] >= s["min_side"]) & (sizes["cols"] >= s["min_side"])]

    syn = synthetic.make_set(s["per_class"], sizes, s, seed)
    pd.to_pickle(syn, CACHE / "synthetic_train.pkl")
    x_syn = extract_all(syn["wafer_map"])

    # d52 same forest settings as the real baseline with the leaf size fixed at
    # the value picked on real validation in d38 so nothing is tuned on test
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    ).fit(x_syn, syn["label"])

    real = pd.read_pickle(CACHE / "features_labeled.pkl")
    lab = df[df["label"].notna()]
    test = lab["split_ours"] == "test"
    y = lab.loc[test, "label"]
    pred = pd.Series(model.predict(real.loc[y.index]), index=y.index)

    table = evaluate.per_class(y, pred, CLASSES)
    table.to_csv(OUT / "synthetic_rf_test_per_class.csv")
    ci = evaluate.lot_bootstrap(y, pred, lab.loc[test, "lot"], CLASSES, seed=seed)
    ci.to_csv(OUT / "synthetic_rf_test_bootstrap_ci.csv")
    conf = evaluate.confusion(y, pred, CLASSES)
    conf.to_csv(OUT / "synthetic_rf_test_confusion.csv")
    viz.confusion_figure(
        conf, "trained on simulated wafers only  share of each true pattern",
        FIG / "synthetic_rf_confusion.png",
    )  # fmt: skip
    pd.DataFrame({"lot": lab.loc[test, "lot"], "label": y, "pred": pred}).to_csv(
        OUT / "synthetic_rf_test_predictions.csv"
    )

    rf_real = pd.read_csv(OUT / "rf_test_per_class.csv", index_col=0)
    side = pd.DataFrame(
        {
            "recall_trained_on_synthetic": table["recall"],
            "recall_trained_on_real": rf_real["recall"],
        }
    )
    side.to_csv(OUT / "synthetic_vs_real_recall.csv")
    summary = {
        "synthetic_maps": len(syn),
        "test_maps": len(y),
        "macro_f1_9class": evaluate.macro_f1(y, pred, CLASSES),
        "macro_f1_9class_lo": ci.loc["macro_f1", "lo"],
        "macro_f1_9class_hi": ci.loc["macro_f1", "hi"],
        "defects_called_none": float((pred[y != "none"] == "none").mean()),
        "clean_called_defect": float((pred[y == "none"] != "none").mean()),
    }
    pd.Series(summary, name="value").rename_axis("fact").to_csv(OUT / "synthetic_rf_summary.csv")
    print(pd.Series(summary).to_string())
    print(side.round(3).to_string())


if __name__ == "__main__":
    main()
