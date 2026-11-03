import numpy as np
import pandas as pd

from wm811k.split import make_split, split_groups


def _toy() -> pd.DataFrame:
    """small fake dataset with 40 lots and one defect map copied across two lots"""
    rng = np.random.default_rng(0)
    rows = []
    labels = ["none", "Center", "Scratch", "Edge-Ring"]
    for i in range(40):
        for w in range(5):
            m = rng.integers(0, 3, size=(6, 6))
            rows.append({"lot": f"lot{i}", "wafer_map": m, "label": labels[(i + w) % 4]})
    df = pd.DataFrame(rows)
    # plant the same defect map in lot0 and lot1 so they have to end up together
    copy = np.full((6, 6), 2)
    df.at[0, "wafer_map"] = copy
    df.at[0, "label"] = "Center"
    df.at[5, "wafer_map"] = copy
    df.at[5, "label"] = "Center"
    # one lot with no labels at all to check the unlabeled bucket
    df.loc[df["lot"] == "lot39", "label"] = None
    return df


def test_lots_never_cross_splits():
    df = _toy()
    df["split_ours"] = make_split(df, seed=1, n_folds=10, val_folds=2, test_folds=2)
    # every lot has to sit in exactly one split or the test score leaks
    assert df.groupby("lot")["split_ours"].nunique().max() == 1


def test_copied_defect_map_pulls_lots_together():
    df = _toy()
    groups = split_groups(df)
    assert groups[0] == groups[5]
    split = make_split(df, seed=1, n_folds=10, val_folds=2, test_folds=2)
    assert split[0] == split[5]


def test_split_is_reproducible_and_unlabeled_lot_is_set_aside():
    df = _toy()
    a = make_split(df, seed=7, n_folds=10, val_folds=2, test_folds=2)
    b = make_split(df, seed=7, n_folds=10, val_folds=2, test_folds=2)
    assert a.equals(b)
    assert set(a[df["lot"] == "lot39"]) == {"unlabeled_only"}
