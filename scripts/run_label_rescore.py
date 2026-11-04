"""item h  scores again with the labels mehnaj corrected by hand
run from the repo root after run_label_audit.py and the verdicts are filled in
python scripts/run_label_rescore.py
the original scores stay the headline  this is a side table that shows how
much the reviewed labels move the numbers"""

from pathlib import Path

import pandas as pd

from wm811k import evaluate
from wm811k.data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def corrected(labels: pd.Series, audit: pd.DataFrame) -> tuple[pd.Series, pd.Index]:
    """labels with every wrong label verdict swapped for the class it names
    gives back the new labels and the maps to leave out because they are mixed
    d64 a mixed wafer has no single right class so it is dropped instead of forced"""
    out = labels.copy()
    drop = []
    for i, v in audit["verdict"].items():
        if v.startswith("wrong label:"):
            out[i] = v.split(":", 1)[1].strip()
        elif v.startswith("mixed"):
            drop.append(i)
    return out, pd.Index(drop)


def main() -> None:
    audit = pd.read_csv(OUT / "label_audit_candidates.csv", index_col=0)
    assert audit["verdict"].notna().all(), "every candidate needs a verdict first"
    rows, recall = [], {}
    for m in ("rf", "cnn"):
        p = pd.read_csv(OUT / f"{m}_test_predictions.csv", index_col=0)
        new, drop = corrected(p["label"], audit)
        keep = ~p.index.isin(drop)
        for name, y, pred in (
            ("original", p["label"], p["pred"]),
            ("reviewed", new[keep], p["pred"][keep]),
        ):
            rows.append({"model": m, "labels": name, "maps": len(y),
                         "macro_f1": evaluate.macro_f1(y, pred, CLASSES)})  # fmt: skip
            recall[(m, name)] = evaluate.per_class(y, pred, CLASSES)["recall"]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "label_rescore_summary.csv", index=False)
    rec = pd.DataFrame(recall)
    rec.columns = [f"{m}_{n}" for m, n in rec.columns]
    rec.to_csv(OUT / "label_rescore_recall.csv")

    kinds = audit["verdict"].str.split(":").str[0].value_counts()
    kinds.rename("candidates").to_csv(OUT / "label_audit_verdict_counts.csv")
    print(table.round(4).to_string(index=False))
    print(rec.round(3).to_string())
    print(kinds.to_string())


if __name__ == "__main__":
    main()
