"""item e  zero shot vision language model on a class balanced sample of the test set
run from the repo root  python scripts/run_vlm.py
calls are saved one row at a time so a stopped run picks up where it left off
the forest and the cnn are scored on the exact same sampled maps"""

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from wm811k import evaluate, vlm
from wm811k.data import CLASSES, load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
IMG = ROOT / "cache" / "vlm_images"
RAW = OUT / "vlm_raw.csv"


def sample(test: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """the same number of maps from every class or all of a class if it has fewer
    d47 a balanced sample is used because a random one would be about 85 percent
    none and leave a handful of maps for the rare classes"""
    rng = np.random.default_rng(seed)
    picks = []
    for c in CLASSES:
        idx = test.index[test["label"] == c].to_numpy()
        picks.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    out = test.loc[np.concatenate(picks), ["lot", "label"]].copy()
    # file names are random so nothing in the path hints at the label
    names = [uuid.UUID(int=int(rng.integers(0, 2**63)) << 64).hex[:16] for _ in range(len(out))]
    out["image"] = names
    return out


def collect(chosen: pd.DataFrame, maps: pd.Series, cfg: dict) -> pd.DataFrame:
    IMG.mkdir(parents=True, exist_ok=True)
    for idx, row in chosen.iterrows():
        path = IMG / f"{row['image']}.png"
        if not path.exists():
            vlm.render(maps[idx], path)

    done = pd.read_csv(RAW, index_col=0) if RAW.exists() else pd.DataFrame()
    have = set(zip(done.index, done["prompt"], strict=True)) if len(done) else set()
    todo = [(i, p) for p in cfg["prompts"] for i in chosen.index if (i, p) not in have]
    print(f"{len(have)} calls saved  {len(todo)} to go", flush=True)

    def one(i, p):
        r = vlm.ask(IMG / f"{chosen.at[i, 'image']}.png", vlm.PROMPTS[p], cfg["model"])
        return i, p, r

    with ThreadPoolExecutor(cfg["workers"]) as pool:
        futures = [pool.submit(one, i, p) for i, p in todo]
        for n, f in enumerate(as_completed(futures), 1):
            i, p, r = f.result()
            row = pd.DataFrame([{"prompt": p, "model": cfg["model"], **r}], index=[i])
            row.to_csv(RAW, mode="a", header=not RAW.exists())
            if n % 20 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    return pd.read_csv(RAW, index_col=0)


def score(chosen: pd.DataFrame, raw: pd.DataFrame, cfg: dict, seed: int) -> None:
    y, lots = chosen["label"], chosen["lot"]
    preds = {}
    for p in cfg["prompts"]:
        r = raw[raw["prompt"] == p]
        preds[f"luna_{p}"] = r["pred"].reindex(chosen.index)
    # the other models are scored on the same sampled maps
    for m in ("rf", "cnn"):
        path = OUT / f"{m}_test_predictions.csv"
        if path.exists():
            preds[m] = pd.read_csv(path, index_col=0)["pred"].reindex(chosen.index)

    price = cfg["price_per_million_usd"]
    rows, recall = [], {}
    for name, pred in preds.items():
        ci = evaluate.lot_bootstrap(y, pred, lots, CLASSES, seed=seed)
        recall[name] = evaluate.per_class(y, pred, CLASSES)["recall"]
        row = {
            "model": name,
            "maps": int(pred.notna().sum()),
            "macro_f1": evaluate.macro_f1(y, pred, CLASSES),
            "macro_f1_lo": ci.loc["macro_f1", "lo"],
            "macro_f1_hi": ci.loc["macro_f1", "hi"],
            "defects_called_none": float((pred[y != "none"] == "none").mean()),
            "clean_called_defect": float((pred[y == "none"] != "none").mean()),
        }
        if name.startswith("luna_"):
            r = raw[raw["prompt"] == name.removeprefix("luna_")]
            usd = r.apply(lambda s: vlm.cost_usd(s.to_dict(), price), axis=1)
            row |= {
                "invalid_replies": int((r["pred"] == "invalid").sum()),
                "calls_that_ran_something": int(r["ran_something"].sum()),
                "median_seconds_per_map": float(r["seconds"].median()),
                "mean_input_tokens": float(r["input_tokens"].mean()),
                "mean_output_tokens": float(r["output_tokens"].mean()),
                # d48 cost as run through codex  the codex system prompt is most of
                # the input tokens so a direct api call would cost less
                "usd_per_1000_maps": float(1000 * usd.mean()),
            }
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(OUT / "vlm_summary.csv")
    pd.DataFrame(recall).to_csv(OUT / "vlm_recall_by_class.csv")
    for name, pred in preds.items():
        evaluate.confusion(y, pred.fillna("invalid"), CLASSES).to_csv(
            OUT / f"vlm_confusion_{name}.csv"
        )
    print(summary.round(3).T.to_string())
    print(pd.DataFrame(recall).round(2).to_string())


def main() -> None:
    full = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg, seed = full["vlm"], full["seed"]
    df = load(ROOT / full["data"]["pickle"])
    lots = pd.read_csv(OUT / "split_lots.csv", index_col="lot")
    df["split_ours"] = df["lot"].map(lots["split"])
    test = df[(df["split_ours"] == "test") & df["label"].notna()]

    chosen = sample(test, cfg["per_class"], seed)
    chosen.to_csv(OUT / "vlm_sample.csv")
    raw = collect(chosen, df["wafer_map"], cfg)
    score(chosen, raw, cfg, seed)


if __name__ == "__main__":
    main()
