"""does a pattern hit one wafer or the whole lot
a pattern that piles up inside lots points at a tool or chamber that every
wafer in the lot went through  a pattern spread evenly across lots looks more
like a random event on single wafers"""

import numpy as np
import pandas as pd


def analysis_pool(df: pd.DataFrame, min_labeled: int = 10) -> pd.DataFrame:
    """labeled wafers from lots with enough labels to say something about the lot"""
    # d20 a lot with two labeled wafers cannot show clustering so lots need
    # at least ten labeled wafers to count
    labeled = df[df["label"].notna()]
    per_lot = labeled.groupby("lot")["label"].transform("size")
    return labeled[per_lot >= min_labeled][["lot", "wafer_index", "label"]].reset_index(drop=True)


def _codes(pool: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lot = pd.factorize(pool["lot"])[0]
    lab = pool["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return lot, lab


def _same_lot_partners(lot: np.ndarray, lab: np.ndarray, n_classes: int) -> np.ndarray:
    """for each class the average number of other wafers in the same lot that
    carry the same class  one number per class"""
    # count wafers of each class in each lot then n times n minus 1 over n gives
    # how many same class partners an average wafer of that class has
    n_lots = lot.max() + 1
    counts = np.bincount(lot * n_classes + lab, minlength=n_lots * n_classes)
    counts = counts.reshape(n_lots, n_classes).astype(float)
    totals = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts * (counts - 1)).sum(axis=0) / totals


def lot_clustering(
    pool: pd.DataFrame, classes: list[str], n_perm: int = 200, seed: int = 0
) -> pd.DataFrame:
    """how much more a class piles up inside lots than chance would give
    chance is built by shuffling labels across all wafers while keeping lot sizes"""
    rng = np.random.default_rng(seed)
    lot, lab = _codes(pool, classes)
    k = len(classes)
    observed = _same_lot_partners(lot, lab, k)

    # d21 shuffling labels across the whole pool breaks any link between lot
    # and pattern but keeps how common each pattern is and how big each lot is
    null = np.array([_same_lot_partners(lot, rng.permutation(lab), k) for _ in range(n_perm)])
    expected = null.mean(axis=0)
    return pd.DataFrame(
        {
            "observed_partners": observed,
            "expected_partners": expected,
            "ratio": observed / expected,
            "null_ratio_lo": np.percentile(null, 2.5, axis=0) / expected,
            "null_ratio_hi": np.percentile(null, 97.5, axis=0) / expected,
            # share of shuffles at least as clustered as the real data
            "p_value": ((null >= observed).sum(axis=0) + 1) / (n_perm + 1),
        },
        index=pd.Index(classes, name="label"),
    )


def _next_slot_same(lot: np.ndarray, slot: np.ndarray, lab: np.ndarray, k: int) -> np.ndarray:
    """for each class the chance that the next slot in the lot has the same class
    only pairs of slots that sit right next to each other count"""
    order = np.lexsort((slot, lot))
    lot, slot, lab = lot[order], slot[order], lab[order]
    adjacent = (lot[1:] == lot[:-1]) & (slot[1:] == slot[:-1] + 1)
    first, second = lab[:-1][adjacent], lab[1:][adjacent]
    hits = np.bincount(first[first == second], minlength=k)
    pairs = np.bincount(first, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return hits / pairs


def slot_order(
    pool: pd.DataFrame, classes: list[str], n_perm: int = 200, seed: int = 0
) -> pd.DataFrame:
    """does a pattern run in neighbouring slots more than its lot mix explains
    chance is built by shuffling labels inside each lot so lot makeup stays fixed"""
    rng = np.random.default_rng(seed)
    lot, lab = _codes(pool, classes)
    slot = pool["wafer_index"].to_numpy()
    k = len(classes)
    observed = _next_slot_same(lot, slot, lab, k)

    # d22 shuffling inside each lot keeps which patterns a lot has and only
    # scrambles their order  so a ratio above one means neighbouring slots
    # match more than the lot mix alone would give  which fits a tool drifting
    # or failing partway through a lot
    null = []
    # positions grouped by lot in their original order
    by_lot = np.argsort(lot, kind="stable")
    for _ in range(n_perm):
        # positions grouped by lot in a random order inside each lot  both lists
        # walk through the lots in the same sequence so pairing them up hands
        # every wafer a label drawn from its own lot
        shuffle = np.lexsort((rng.random(len(lab)), lot))
        permuted = np.empty_like(lab)
        permuted[by_lot] = lab[shuffle]
        null.append(_next_slot_same(lot, slot, permuted, k))
    null = np.array(null)
    expected = np.nanmean(null, axis=0)
    return pd.DataFrame(
        {
            "observed_next_same": observed,
            "expected_next_same": expected,
            "ratio": observed / expected,
            "null_ratio_lo": np.nanpercentile(null, 2.5, axis=0) / expected,
            "null_ratio_hi": np.nanpercentile(null, 97.5, axis=0) / expected,
            "p_value": ((null >= observed).sum(axis=0) + 1) / (n_perm + 1),
        },
        index=pd.Index(classes, name="label"),
    )


def excursion_lots(pool: pd.DataFrame, share: float = 0.5) -> pd.DataFrame:
    """lots where one defect pattern covers at least half of the labeled wafers
    these are the lots a yield engineer would flag as an excursion"""
    defects = pool[pool["label"] != "none"]
    per_lot = pool.groupby("lot").size()
    top = defects.groupby(["lot", "label"]).size().rename("wafers").reset_index()
    top["labeled_in_lot"] = top["lot"].map(per_lot)
    top["share"] = top["wafers"] / top["labeled_in_lot"]
    return top[top["share"] >= share].sort_values("share", ascending=False)
