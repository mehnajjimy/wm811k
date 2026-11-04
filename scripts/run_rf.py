"""random forest baseline on hand built features under the lot split
run from the repo root  python scripts/run_rf.py"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

from wm811k import evaluate, viz
from wm811k.data import CLASSES, load
from wm811k.features import extract_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIG = ROOT / "figures"
CACHE = ROOT / "cache"
DEFECTS = [c for c in CLASSES if c != "none"]


def labeled_with_split(cfg: dict) -> pd.DataFrame:
    """labeled maps with our own split attached from the saved lot table"""
    df = load(ROOT / cfg["data"]["pickle"])
    lots = pd.read_csv(OUT / "split_lots.csv", index_col="lot")
    df["split_ours"] = df["lot"].map(lots["split"])
    return df[df["label"].notna()].copy()


def features(df: pd.DataFrame) -> pd.DataFrame:
    """features for every labeled map cached on disk because they never change"""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "features_labeled.pkl"
    if path.exists():
        cached = pd.read_pickle(path)
        if cached.index.equals(df.index):
            return cached
    feats = extract_all(df["wafer_map"])
    feats.to_pickle(path)
    return feats


def fit(x: pd.DataFrame, y: pd.Series, leaf: int, seed: int) -> RandomForestClassifier:
    # d38 balanced subsample weights make every class count the same inside each
    # tree so the forest does not learn to call everything none
    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    return model.fit(x, y)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed = cfg["seed"]
    viz.setup()
    df = labeled_with_split(cfg)
    x = features(df)
    y = df["label"]
    train, val, test = (df["split_ours"] == s for s in ("train", "val", "test"))

    # pick the leaf size on validation only  the test set is touched once at the end
    val_scores, val_proba = {}, {}
    for leaf in cfg["rf"]["leaf_sizes"]:
        m = fit(x[train], y[train], leaf, seed)
        val_proba[leaf] = pd.DataFrame(
            m.predict_proba(x[val]), index=y[val].index, columns=[f"p_{c}" for c in m.classes_]
        )
        val_scores[leaf] = evaluate.macro_f1(y[val], pd.Series(m.predict(x[val])), CLASSES)
    best_leaf = max(val_scores, key=val_scores.get)
    pd.Series(val_scores, name="val_macro_f1").rename_axis("min_samples_leaf").to_csv(
        OUT / "rf_val_tuning.csv"
    )
    # validation scores from the train only model are kept because thresholds
    # for item g have to be picked on maps the model never trained on
    vp = val_proba[best_leaf]
    vp.insert(0, "label", y[val])
    vp.insert(0, "lot", df.loc[val, "lot"])
    vp.to_csv(OUT / "rf_val_predictions.csv")

    # final model on train plus validation then one look at test
    fit_mask = train | val
    model = fit(x[fit_mask], y[fit_mask], best_leaf, seed)
    pred = pd.Series(model.predict(x[test]), index=y[test].index)
    proba = model.predict_proba(x[test])

    table = evaluate.per_class(y[test], pred, CLASSES)
    table.to_csv(OUT / "rf_test_per_class.csv")
    ci = evaluate.lot_bootstrap(y[test], pred, df.loc[test, "lot"], CLASSES, seed=seed)
    ci.to_csv(OUT / "rf_test_bootstrap_ci.csv")
    conf = evaluate.confusion(y[test], pred, CLASSES)
    conf.to_csv(OUT / "rf_test_confusion.csv")
    viz.confusion_figure(
        conf, "random forest  share of each true pattern", FIG / "rf_confusion.png"
    )

    # predictions are saved so the cnn and vlm can be compared on the exact same maps
    out = pd.DataFrame(proba, index=pred.index, columns=[f"p_{c}" for c in model.classes_])
    out.insert(0, "pred", pred)
    out.insert(0, "label", y[test])
    out.insert(0, "lot", df.loc[test, "lot"])
    out.to_csv(OUT / "rf_test_predictions.csv")

    pd.Series(model.feature_importances_, index=x.columns, name="importance").sort_values(
        ascending=False
    ).to_csv(OUT / "rf_feature_importance.csv")

    # d05 side view  defect maps only with an 8 class model  for comparing with
    # papers that drop none
    d_fit = fit_mask & (y != "none")
    d_test = test & (y != "none")
    m8 = fit(x[d_fit], y[d_fit], best_leaf, seed)
    pred8 = pd.Series(m8.predict(x[d_test]), index=y[d_test].index)
    table8 = evaluate.per_class(y[d_test], pred8, DEFECTS)
    table8.to_csv(OUT / "rf8_test_per_class.csv")

    summary = {
        "best_min_samples_leaf": best_leaf,
        "test_maps": int(test.sum()),
        "macro_f1_9class": evaluate.macro_f1(y[test], pred, CLASSES),
        "macro_f1_9class_lo": ci.loc["macro_f1", "lo"],
        "macro_f1_9class_hi": ci.loc["macro_f1", "hi"],
        "macro_f1_8class": evaluate.macro_f1(y[d_test], pred8, DEFECTS),
        # accuracy is kept only to show why it is misleading  never as the headline
        "accuracy_9class_not_headline": float((pred == y[test]).mean()),
        "accuracy_if_always_none": float((y[test] == "none").mean()),
    }
    pd.Series(summary, name="value").rename_axis("fact").to_csv(OUT / "rf_summary.csv")
    print(pd.Series(summary).to_string())
    print(table.round(3).to_string())
    print(ci.round(3).to_string())
    print(table8.round(3).to_string())


if __name__ == "__main__":
    main()
