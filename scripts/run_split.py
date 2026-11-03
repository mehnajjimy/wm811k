"""builds our own lot grouped split and writes it to results
run from the repo root  python scripts/run_split.py"""

from pathlib import Path

import pandas as pd
import yaml

from wm811k.data import load
from wm811k.split import make_split, split_groups

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    df = load(ROOT / cfg["data"]["pickle"])
    s = cfg["split"]
    df["group"] = split_groups(df)
    df["split_ours"] = make_split(
        df,
        seed=cfg["seed"],
        n_folds=s["n_folds"],
        val_folds=s["val_folds"],
        test_folds=s["test_folds"],
    )

    # the split is saved per lot not per map  every map in a lot shares one
    # split so this small file is enough to rebuild the full assignment
    lots = df.groupby("lot").agg(group=("group", "first"), split=("split_ours", "first"))
    lots.to_csv(OUT / "split_lots.csv")

    # class counts per split to check the stratification actually worked
    labeled = df[df["label"].notna()]
    counts = pd.crosstab(labeled["label"], labeled["split_ours"])
    counts.to_csv(OUT / "split_class_counts.csv")
    print(counts.to_string())
    print((counts / counts.sum()).round(4).to_string())
    print("merged groups", int((df.groupby("group")["lot"].nunique() > 1).sum()))


if __name__ == "__main__":
    main()
