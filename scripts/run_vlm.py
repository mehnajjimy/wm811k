"""item e  zero shot vision language model on a class balanced sample of the test set
run from the repo root  python scripts/run_vlm.py
calls are saved one row at a time so a stopped run picks up where it left off
the forest and the cnn are scored on the exact same sampled maps"""

import argparse
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


# d65 each model has its own raw file and its own name in the summary
# d68 fable 5.2 runs through the same claude code path as opus
BACKENDS = {"luna": "", "opus": "_opus", "fable": "_fable"}


def ask_one(backend: str, image: Path, prompt: str, cfg: dict) -> dict:
    """one call to whichever model is being run"""
    if backend != "luna":
        o = cfg[backend]
        return vlm.ask_claude(image, prompt, o["model"], o["effort"])
    return vlm.ask(image, prompt, cfg["model"])


def collect(
    chosen: pd.DataFrame,
    maps: pd.Series,
    cfg: dict,
    prompts: list[str],
    raw_path: Path,
    backend: str = "luna",
) -> pd.DataFrame:
    IMG.mkdir(parents=True, exist_ok=True)
    for idx, row in chosen.iterrows():
        path = IMG / f"{row['image']}.png"
        if not path.exists():
            vlm.render(maps[idx], path)

    done = pd.read_csv(raw_path, index_col=0) if raw_path.exists() else pd.DataFrame()
    have = set(zip(done.index, done["prompt"], strict=True)) if len(done) else set()
    todo = [(i, p) for p in prompts for i in chosen.index if (i, p) not in have]
    print(f"{len(have)} calls saved  {len(todo)} to go", flush=True)

    model = cfg[backend]["model"] if backend != "luna" else cfg["model"]
    workers = cfg[backend]["workers"] if backend != "luna" else cfg["workers"]

    def one(i, p):
        r = ask_one(backend, IMG / f"{chosen.at[i, 'image']}.png", vlm.PROMPTS[p], cfg)
        return i, p, r

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, i, p) for i, p in todo]
        failed = 0
        for n, f in enumerate(as_completed(futures), 1):
            i, p, r = f.result()
            # d65 a call that failed such as one past a usage limit is left unsaved
            # so the next run asks it again instead of scoring it as a wrong answer
            if r.pop("failed", False):
                failed += 1
                continue
            row = pd.DataFrame([{"prompt": p, "model": model, **r}], index=[i])
            # the whole file is rewritten so a column added later such as actions
            # lines up with the older rows instead of shifting them
            done = pd.concat([done, row])
            done.to_csv(raw_path)
            if n % 20 == 0:
                print(f"{n}/{len(todo)}  {failed} failed", flush=True)
    if failed:
        print(f"{failed} calls failed and were not saved  run again to retry them", flush=True)
    return done


# d66 cao and tao 2026 arxiv 2610.35080 score six classes on wm811k  they drop
# none and near full and merge the two edge classes  these are their per class
# counts in the 9000 map random sample  used to reweight overall accuracy
PAPER_MERGE = {"Edge-Loc": "Edge", "Edge-Ring": "Edge", "Loc": "Local"}
PAPER_CLASSES = ["Center", "Donut", "Edge", "Local", "Random", "Scratch"]
PAPER_MIX = {"Center": 1747, "Donut": 502, "Edge": 3430, "Local": 1830, "Random": 555,
             "Scratch": 936}  # fmt: skip


def paper_style(y: pd.Series, preds: dict) -> pd.DataFrame:
    """every model scored the way the paper scores  on the six class subset only
    our models could also answer none or near full so those count as wrong here
    overall is reported two ways  on our balanced sample and reweighted to the
    paper class mix since a plain overall depends on how common each class is"""
    keep = y.isin(PAPER_MERGE.keys() | set(PAPER_CLASSES))
    yt = y[keep].replace(PAPER_MERGE)
    rows = []
    for name, pred in preds.items():
        pt = pred[keep].replace(PAPER_MERGE)
        acc = {c: float((pt[yt == c] == c).mean()) for c in PAPER_CLASSES}
        w = sum(PAPER_MIX.values())
        rows.append({
            "model": name, "maps": int(keep.sum()), **acc,
            "macro_accuracy": sum(acc.values()) / len(acc),
            "overall_our_sample": float((pt == yt).mean()),
            "overall_paper_mix": sum(acc[c] * PAPER_MIX[c] for c in PAPER_CLASSES) / w,
        })  # fmt: skip
    return pd.DataFrame(rows).set_index("model")


def score(chosen: pd.DataFrame, raws: dict, cfg: dict, seed: int, split: str) -> None:
    """raws maps a backend name to its raw calls  every model is scored on the same maps"""
    y, lots = chosen["label"], chosen["lot"]
    tag = "" if split == "test" else f"_{split}"
    preds, calls = {}, {}
    for backend, raw in raws.items():
        for p in raw["prompt"].unique():
            r = raw[raw["prompt"] == p]
            preds[f"{backend}_{p}"] = r["pred"].reindex(chosen.index)
            calls[f"{backend}_{p}"] = (backend, r)
    # the other models are scored on the same sampled maps
    for m in ("rf", "cnn"):
        path = OUT / f"{m}_{split}_predictions.csv"
        if path.exists():
            t = pd.read_csv(path, index_col=0)
            if "pred" not in t:
                # validation files hold only probabilities so take the most likely class
                cols = [f"p_{c}" for c in CLASSES]
                t["pred"] = np.asarray(CLASSES)[t[cols].to_numpy().argmax(1)]
            preds[m] = t["pred"].reindex(chosen.index)

    prices = {"luna": cfg["price_per_million_usd"]}
    prices |= {b: cfg[b]["price_per_million_usd"] for b in BACKENDS if b != "luna" and b in cfg}
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
        if name in calls:
            backend, r = calls[name]
            price = prices[backend]
            usd = pd.Series([vlm.cost_usd(s.fillna(0).to_dict(), price) for _, s in r.iterrows()])
            row |= {
                "invalid_replies": int((r["pred"] == "invalid").sum()),
                "calls_that_ran_something": int(r["ran_something"].sum()),
                "median_seconds_per_map": float(r["seconds"].median()),
                "mean_input_tokens": float(r["input_tokens"].mean()),
                "mean_output_tokens": float(r["output_tokens"].mean()),
                # d48 cost as run through codex  the codex system prompt is most of
                # the input tokens so a direct api call would cost less
                # d65 opus runs with a one line system prompt so its overhead is smaller
                "usd_per_1000_maps": float(1000 * usd.mean()),
                # d65 the same tokens billed as a plain api call with no caching at all
                # so the two harnesses can be compared on what the model itself used
                "usd_per_1000_maps_no_cache": float(
                    1000
                    * (
                        r["input_tokens"] * price["input"] + r["output_tokens"] * price["output"]
                    ).mean()
                    / 1e6
                ),
            }
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("model")
    summary.to_csv(OUT / f"vlm_summary{tag}.csv")
    pd.DataFrame(recall).to_csv(OUT / f"vlm_recall_by_class{tag}.csv")
    for name, pred in preds.items():
        evaluate.confusion(y, pred.fillna("invalid"), CLASSES).to_csv(
            OUT / f"vlm_confusion{tag}_{name}.csv"
        )
    # d65 paired gaps on the same maps  each prompt opus minus luna and the
    # trained models minus the best zero shot run  only rows whose models exist
    pairs = [(f"opus_{p}", f"luna_{p}") for p in vlm.PROMPTS]
    pairs += [("cnn", "opus_rules_v2"), ("rf", "opus_rules_v2")]
    pairs += [(f"fable_{p}", f"opus_{p}") for p in vlm.PROMPTS]
    pairs += [("cnn", "fable_rules_v2"), ("rf", "fable_rules_v2")]
    gaps = []
    for a, b in pairs:
        if a in preds and b in preds:
            g = evaluate.paired_lot_bootstrap(y, preds[a], preds[b], lots, CLASSES, seed=seed)
            gaps.append({"a": a, "b": b, **g.loc["macro_f1"].to_dict()})
    if gaps:
        pd.DataFrame(gaps).to_csv(OUT / f"vlm_pairs{tag}.csv", index=False)
        print(pd.DataFrame(gaps).round(3).to_string(index=False))
    paper = paper_style(y, preds)
    paper.to_csv(OUT / f"vlm_paper_style{tag}.csv")
    print(paper.round(3).to_string())
    print(summary.round(3).T.to_string())
    print(pd.DataFrame(recall).round(2).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--prompts", nargs="+", default=None)
    ap.add_argument("--backend", default="luna", choices=list(BACKENDS))
    args = ap.parse_args()
    full = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg, seed = full["vlm"], full["seed"]
    prompts = args.prompts or cfg["prompts"]
    df = load(ROOT / full["data"]["pickle"])
    lots = pd.read_csv(OUT / "split_lots.csv", index_col="lot")
    df["split_ours"] = df["lot"].map(lots["split"])
    part = df[(df["split_ours"] == args.split) & df["label"].notna()]

    # d55 validation gets its own sample and raw file so a new prompt can be
    # checked there before it ever sees the test sample
    tag = "" if args.split == "test" else f"_{args.split}"
    chosen = sample(part, cfg["per_class"], seed)
    # the split goes in front of the file name so a validation picture can never
    # reuse a test picture that happened to draw the same random name
    chosen["image"] = args.split + "_" + chosen["image"]
    chosen.to_csv(OUT / f"vlm_sample{tag}.csv")
    raw_path = OUT / f"vlm_raw{tag}{BACKENDS[args.backend]}.csv"
    collect(chosen, df["wafer_map"], cfg, prompts, raw_path, args.backend)
    # every model that has calls saved is scored again so the summary holds them all
    raws = {}
    for b, suffix in BACKENDS.items():
        path = OUT / f"vlm_raw{tag}{suffix}.csv"
        if path.exists():
            raws[b] = pd.read_csv(path, index_col=0)
    score(chosen, raws, cfg, seed, args.split)


if __name__ == "__main__":
    main()
