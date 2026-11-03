"""train validation and test split that keeps lots apart"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wm811k.audit import map_hash


def _union_find_groups(lots: list[str], links: list[tuple[str, str]]) -> dict[str, str]:
    """joins lots into groups wherever a link says two lots belong together
    gives back a lookup from each lot to the name of its group"""
    parent = {lot: lot for lot in lots}

    # find walks up to the root of a lot and flattens the path on the way
    # so later lookups are quick
    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return {lot: find(lot) for lot in lots}


def split_groups(df: pd.DataFrame) -> pd.Series:
    """one group name per map
    a group is a lot unless lots share an identical defect map  then they are merged"""
    # d13 identical defect maps in different lots would leak across a plain lot
    # split so every lot touched by the same copy is pulled into one group
    defects = df[df["label"].notna() & (df["label"] != "none")]
    hashes = defects["wafer_map"].map(map_hash)
    links = []
    for lots in defects.groupby(hashes)["lot"].unique():
        # chain the lots of one copy group together  first to each of the rest
        links.extend((lots[0], other) for other in lots[1:])
    lookup = _union_find_groups(df["lot"].unique().tolist(), links)
    return df["lot"].map(lookup)


def make_split(
    df: pd.DataFrame, seed: int, n_folds: int = 20, val_folds: int = 3, test_folds: int = 3
) -> pd.Series:
    """gives each map a split  train val test or unlabeled_only
    20 folds with 3 for validation and 3 for test is a 70 15 15 split"""
    groups = split_groups(df)
    labeled = df["label"].notna()

    # d14 folds are built from labeled maps only so the class balance is set
    # by real labels  stratified group folds keep each group in one fold while
    # trying to give every fold the same mix of classes
    folds = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_group: dict[str, int] = {}
    x = np.zeros(labeled.sum())
    y = df.loc[labeled, "label"].to_numpy()
    g = groups[labeled].to_numpy()
    for fold, (_, idx) in enumerate(folds.split(x, y, g)):
        for name in np.unique(g[idx]):
            fold_of_group[name] = fold

    def name_for(fold: int | None) -> str:
        # groups with no labeled map never got a fold  they are only useful
        # later as an unlabeled pool for semi supervised work
        if fold is None:
            return "unlabeled_only"
        if fold < test_folds:
            return "test"
        if fold < test_folds + val_folds:
            return "val"
        return "train"

    # unlabeled maps follow their group so a test lot never feeds training
    # even through pseudo labels later on
    return groups.map(lambda name: name_for(fold_of_group.get(name))).rename("split_ours")
