"""when to trust the model and when to send a wafer to an engineer
works on any model that gives class probabilities so the forest and the cnn
are judged the same way
probability tables have one column per class named p_ plus the class"""

import numpy as np
import pandas as pd

from wm811k import evaluate


def _proba(table: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return table[[f"p_{c}" for c in classes]].to_numpy()


def none_gate(table: pd.DataFrame, classes: list[str], threshold: float) -> pd.Series:
    """labels where a wafer is only called none when p none is at least threshold
    otherwise it gets the most likely defect class
    d44 raising the threshold makes the model slower to call a wafer clean so
    fewer real defects slip through at the price of clean wafers sent to review"""
    defects = [c for c in classes if c != "none"]
    p = _proba(table, defects)
    best_defect = np.asarray(defects)[p.argmax(1)]
    clean = table["p_none"].to_numpy() >= threshold
    return pd.Series(np.where(clean, "none", best_defect), index=table.index)


def gate_curve(table: pd.DataFrame, classes: list[str], thresholds: np.ndarray) -> pd.DataFrame:
    """one row per threshold with the two numbers a fab trades against each other
    missed defect share is the share of defect wafers called none
    clean flagged per 1000 is how many clean wafers out of 1000 get flagged"""
    y = table["label"]
    defect = y != "none"
    rows = []
    for t in thresholds:
        pred = none_gate(table, classes, t)
        rows.append(
            {
                "threshold": float(t),
                "missed_defect_share": float((pred[defect] == "none").mean()),
                "clean_flagged_per_1000": float(1000 * (pred[~defect] != "none").mean()),
                "macro_f1": evaluate.macro_f1(y, pred, classes),
            }
        )
    return pd.DataFrame(rows)


def pick_gate(val_curve: pd.DataFrame, max_missed: float) -> float:
    """lowest threshold on validation that keeps missed defects at or under the target
    the lowest one is picked because it flags the fewest clean wafers"""
    ok = val_curve[val_curve["missed_defect_share"] <= max_missed]
    if ok.empty:
        # no threshold reaches the target so the strictest one is the closest
        return float(val_curve["threshold"].max())
    return float(ok["threshold"].min())


def coverage_curve(table: pd.DataFrame, classes: list[str], coverages: np.ndarray) -> pd.DataFrame:
    """macro f1 on the wafers the model keeps when the least sure ones go to review
    coverage 0.9 means the model decides 90 percent of wafers by itself and the
    10 percent it is least sure about go to an engineer"""
    p = _proba(table, classes)
    conf = p.max(1)
    pred = pd.Series(np.asarray(classes)[p.argmax(1)], index=table.index)
    y = table["label"]
    # most sure first so keeping the top k is keeping the k easiest wafers
    order = np.argsort(-conf, kind="stable")
    rows = []
    for c in coverages:
        keep = order[: round(float(c) * len(order))]
        yk, pk = y.iloc[keep], pred.iloc[keep]
        dk = yk != "none"
        # only classes still present among the kept wafers are averaged so a class
        # that went entirely to review does not count as a zero
        present = [k for k in classes if (yk == k).any()]
        rows.append(
            {
                "coverage": float(c),
                "macro_f1_kept": evaluate.macro_f1(yk, pk, present),
                "missed_defect_share_kept": float((pk[dk] == "none").mean()) if dk.any() else 0.0,
                "defects_sent_to_review": int((y != "none").sum() - dk.sum()),
            }
        )
    return pd.DataFrame(rows)
