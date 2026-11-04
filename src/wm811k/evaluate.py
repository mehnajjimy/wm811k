"""scoring shared by every model so rf  cnn and vlm are judged the same way"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def per_class(y_true: pd.Series, y_pred: pd.Series, classes: list[str]) -> pd.DataFrame:
    """precision recall f1 and how many true maps for each class"""
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": n},
        index=pd.Index(classes, name="label"),
    )


def macro_f1(y_true: pd.Series, y_pred: pd.Series, classes: list[str]) -> float:
    """average f1 over classes with every class counting the same
    d04 this is the headline number because accuracy is mostly the none class"""
    return float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))


def lot_bootstrap(
    y_true: pd.Series,
    y_pred: pd.Series,
    lots: pd.Series,
    classes: list[str],
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """95 percent intervals for macro f1 and per class recall
    whole lots are resampled not single wafers because wafers in a lot are
    correlated and resampling them one by one would make the intervals too narrow"""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(lots)
    # positions of each lot so a resampled lot brings all of its wafers along
    members = pd.Series(np.arange(len(codes))).groupby(codes).apply(np.array).to_list()
    yt, yp = y_true.to_numpy(), y_pred.to_numpy()
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        _, r, _, _ = precision_recall_fscore_support(
            yt[idx], yp[idx], labels=classes, zero_division=0
        )
        rows.append([macro_f1(yt[idx], yp[idx], classes), *r])
    boot = pd.DataFrame(rows, columns=["macro_f1", *[f"recall_{c}" for c in classes]])
    return boot.quantile([0.025, 0.975]).T.rename(columns={0.025: "lo", 0.975: "hi"})


def confusion(y_true: pd.Series, y_pred: pd.Series, classes: list[str]) -> pd.DataFrame:
    """rows are the true class and columns the predicted class as counts"""
    m = confusion_matrix(y_true, y_pred, labels=classes)
    return pd.DataFrame(m, index=pd.Index(classes, name="true"), columns=classes)


def paired_lot_bootstrap(
    y_true: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    lots: pd.Series,
    classes: list[str],
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """95 percent intervals for model a minus model b on the same maps
    both models are scored on the same resampled lots each round so the
    interval is about the gap between them and not about which lots got drawn"""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(lots)
    members = pd.Series(np.arange(len(codes))).groupby(codes).apply(np.array).to_list()
    yt, ya, yb = y_true.to_numpy(), pred_a.to_numpy(), pred_b.to_numpy()

    def scores(idx: np.ndarray) -> list[float]:
        _, ra, _, _ = precision_recall_fscore_support(
            yt[idx], ya[idx], labels=classes, zero_division=0
        )
        _, rb, _, _ = precision_recall_fscore_support(
            yt[idx], yb[idx], labels=classes, zero_division=0
        )
        f = macro_f1(yt[idx], ya[idx], classes) - macro_f1(yt[idx], yb[idx], classes)
        return [f, *(ra - rb)]

    point = scores(np.arange(len(yt)))
    rows = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows.append(scores(np.concatenate([members[i] for i in pick])))
    names = ["macro_f1", *[f"recall_{c}" for c in classes]]
    boot = pd.DataFrame(rows, columns=names)
    out = boot.quantile([0.025, 0.975]).T.rename(columns={0.025: "lo", 0.975: "hi"})
    out.insert(0, "diff", point)
    # a gap is only called real when the whole interval sits on one side of zero
    out["clear"] = (out["lo"] > 0) | (out["hi"] < 0)
    return out


def missed_as_none(y_true: pd.Series, y_pred: pd.Series, defects: list[str]) -> pd.Series:
    """share of each defect class that the model called none
    this is the costly error in a fab because the excursion is never looked at"""
    return pd.Series(
        {c: float((y_pred[y_true == c] == "none").mean()) for c in defects}, name="missed_as_none"
    )
