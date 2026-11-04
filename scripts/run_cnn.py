"""small cnn on the same lot split and the same test maps as the random forest
run from the repo root  python scripts/run_cnn.py
trains on the apple gpu when there is one and falls back to the cpu"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from wm811k import cnn, evaluate, viz
from wm811k.data import CLASSES, load

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


def images(df: pd.DataFrame, size: int) -> np.ndarray:
    """resized images for every labeled map cached on disk"""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"cnn_images_{size}.npz"
    if path.exists():
        cached = np.load(path)
        if np.array_equal(cached["index"], df.index.to_numpy()):
            return cached["images"]
    imgs = cnn.to_images(df["wafer_map"], size)
    np.savez(path, images=imgs, index=df.index.to_numpy())
    return imgs


def scored(model, x: np.ndarray, index: pd.Index, device: str) -> tuple[pd.Series, np.ndarray]:
    proba = cnn.predict_proba(model, x, device)
    return pd.Series(np.asarray(CLASSES)[proba.argmax(1)], index=index), proba


def save_val_predictions(model, df: pd.DataFrame, x: np.ndarray, val: np.ndarray, device: str):
    """validation scores from the chosen model for picking thresholds in item g"""
    _, proba = scored(model, x[val], df.index[val], device)
    vp = pd.DataFrame(proba, index=df.index[val], columns=[f"p_{k}" for k in CLASSES])
    vp.insert(0, "label", df.loc[val, "label"])
    vp.insert(0, "lot", df.loc[val, "lot"])
    vp.to_csv(OUT / "cnn_val_predictions.csv")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    c = cfg["cnn"]
    seed = cfg["seed"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    viz.setup()

    df = labeled_with_split(cfg)
    x = images(df, c["size"])
    codes = df["label"].map({k: i for i, k in enumerate(CLASSES)}).to_numpy()
    train, val, test = ((df["split_ours"] == s).to_numpy() for s in ("train", "val", "test"))

    def run(weights: str, s: int, tag: str):
        t0 = time.time()
        model, hist = cnn.fit(
            x[train], codes[train], x[val], codes[val], CLASSES, s, device,
            weights=weights, max_epochs=c["max_epochs"], patience=c["patience"],
            log=lambda m: print(f"[{tag}] {m}", flush=True),
        )  # fmt: skip
        hist["minutes"] = (time.time() - t0) / 60
        hist.to_csv(OUT / f"cnn_history_{tag}.csv", index=False)
        return model, hist

    # d42 the loss weighting is picked on validation only like the forest leaf size
    tuning, models = [], {}
    for w in c["weight_schemes"]:
        model, hist = run(w, seed, f"{w}_seed{seed}")
        models[w] = model
        tuning.append({"weights": w, "best_val_macro_f1": hist["val_macro_f1"].max(),
                       "best_epoch": int(hist["val_macro_f1"].idxmax()) + 1,
                       "minutes": hist["minutes"].iloc[0]})  # fmt: skip
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(OUT / "cnn_val_tuning.csv", index=False)
    best_w = tuning.loc[tuning["best_val_macro_f1"].idxmax(), "weights"]
    model = models[best_w]
    torch.save(model.state_dict(), CACHE / "cnn_main.pt")
    save_val_predictions(model, df, x, val, device)

    # one look at test with the main model
    test_idx = df.index[test]
    y = df.loc[test, "label"]
    lots = df.loc[test, "lot"]
    pred, proba = scored(model, x[test], test_idx, device)
    table = evaluate.per_class(y, pred, CLASSES)
    table.to_csv(OUT / "cnn_test_per_class.csv")
    ci = evaluate.lot_bootstrap(y, pred, lots, CLASSES, seed=seed)
    ci.to_csv(OUT / "cnn_test_bootstrap_ci.csv")
    conf = evaluate.confusion(y, pred, CLASSES)
    conf.to_csv(OUT / "cnn_test_confusion.csv")
    viz.confusion_figure(conf, "small cnn  share of each true pattern", FIG / "cnn_confusion.png")

    out = pd.DataFrame(proba, index=test_idx, columns=[f"p_{k}" for k in CLASSES])
    out.insert(0, "pred", pred)
    out.insert(0, "label", y)
    out.insert(0, "lot", lots)
    out.to_csv(OUT / "cnn_test_predictions.csv")

    # same maps as the forest so the two can be compared map by map
    rf = pd.read_csv(OUT / "rf_test_predictions.csv", index_col=0)
    assert rf.index.equals(test_idx), "rf and cnn test maps differ"
    gap = evaluate.paired_lot_bootstrap(y, pred, rf["pred"], lots, CLASSES, seed=seed)
    gap.to_csv(OUT / "cnn_minus_rf.csv")
    missed = pd.DataFrame(
        {"rf": evaluate.missed_as_none(y, rf["pred"], DEFECTS),
         "cnn": evaluate.missed_as_none(y, pred, DEFECTS)}
    )  # fmt: skip
    missed.to_csv(OUT / "missed_as_none_rf_vs_cnn.csv")

    # d43 other seeds show how much of the score is luck of the starting weights
    seeds = [{"seed": seed, "test_macro_f1": evaluate.macro_f1(y, pred, CLASSES)}]
    for s in c["extra_seeds"]:
        m, _ = run(best_w, s, f"{best_w}_seed{s}")
        p, _ = scored(m, x[test], test_idx, device)
        seeds.append({"seed": s, "test_macro_f1": evaluate.macro_f1(y, p, CLASSES)})
    seeds = pd.DataFrame(seeds)
    seeds.to_csv(OUT / "cnn_seed_spread.csv", index=False)

    summary = {
        "device": device,
        "weights": best_w,
        "test_maps": int(test.sum()),
        "macro_f1_9class": evaluate.macro_f1(y, pred, CLASSES),
        "macro_f1_9class_lo": ci.loc["macro_f1", "lo"],
        "macro_f1_9class_hi": ci.loc["macro_f1", "hi"],
        # defect maps only scored on the defect classes for comparing with papers
        # this is the 9 class model scored on defect maps not a separate 8 class
        # model so a defect called none still counts against it
        "macro_f1_on_defect_maps": evaluate.macro_f1(y[y != "none"], pred[y != "none"], DEFECTS),
        "rf_macro_f1_on_defect_maps": evaluate.macro_f1(
            y[y != "none"], rf["pred"][y != "none"], DEFECTS
        ),
        "seed_macro_f1_min": seeds["test_macro_f1"].min(),
        "seed_macro_f1_max": seeds["test_macro_f1"].max(),
        "cnn_minus_rf_macro_f1": gap.loc["macro_f1", "diff"],
        "cnn_minus_rf_lo": gap.loc["macro_f1", "lo"],
        "cnn_minus_rf_hi": gap.loc["macro_f1", "hi"],
        "accuracy_9class_not_headline": float((pred == y).mean()),
    }
    pd.Series(summary, name="value").rename_axis("fact").to_csv(OUT / "cnn_summary.csv")
    print(pd.Series(summary).to_string())
    print(table.round(3).to_string())
    print(gap.round(3).to_string())
    print(missed.round(3).to_string())


if __name__ == "__main__":
    main()
