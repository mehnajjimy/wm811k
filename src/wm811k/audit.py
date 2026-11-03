"""first look at the data before any modelling
every check here answers a question that would change a later decision"""

import hashlib

import numpy as np
import pandas as pd


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """how many maps sit in each class plus the unlabeled ones"""
    # unlabeled maps have no label so fill them in as their own group
    # otherwise they vanish from the count and the 85 percent none figure
    # cannot be checked
    labels = df["label"].fillna("unlabeled")
    counts = labels.value_counts().rename_axis("label").reset_index(name="maps")
    labeled_total = counts.loc[counts["label"] != "unlabeled", "maps"].sum()
    counts["share_of_labeled"] = np.where(
        counts["label"] == "unlabeled", np.nan, counts["maps"] / labeled_total
    )
    return counts


def pixel_values(df: pd.DataFrame) -> set[int]:
    """every distinct pixel value across all maps
    should be only 0 1 2 or the yield formula is wrong"""
    seen: set[int] = set()
    for m in df["wafer_map"]:
        seen.update(np.unique(m).tolist())
    return seen


def map_hash(wafer_map: np.ndarray) -> str:
    """short fingerprint of a map so exact copies can be found fast"""
    # the shape goes into the hash too so two maps with the same pixels
    # laid out in different grids do not collide
    h = hashlib.sha1(str(wafer_map.shape).encode())
    h.update(np.ascontiguousarray(wafer_map, dtype=np.uint8).tobytes())
    return h.hexdigest()[:16]


def duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """groups of maps that are pixel for pixel identical
    gives back one row per copy with how many lots the copies span"""
    hashes = df["wafer_map"].map(map_hash)
    groups = df.assign(hash=hashes).groupby("hash")
    summary = groups.agg(
        copies=("lot", "size"),
        lots=("lot", "nunique"),
        labels=("label", lambda s: s.dropna().nunique()),
    )
    return summary[summary["copies"] > 1].sort_values("copies", ascending=False)


def defect_duplicates(df: pd.DataFrame) -> dict[str, int]:
    """exact copies among maps labeled with a real defect
    most duplicates in the data are clean maps with zero or one fail and those
    match by chance  a copied defect map with many fails is a real copy and
    could leak across any split that does not keep the copies together"""
    labeled = df[df["label"].notna() & (df["label"] != "none")].copy()
    labeled["hash"] = labeled["wafer_map"].map(map_hash)
    copies = labeled.groupby("hash").filter(lambda g: len(g) > 1)
    groups = copies.groupby("hash")
    return {
        "defect_maps_with_exact_copy": len(copies),
        "defect_copy_groups": int(groups.ngroups),
        # groups whose copies sit in different lots slip through a lot split
        "defect_copy_groups_spanning_lots": int((groups["lot"].nunique() > 1).sum()),
        # and these ones already straddle the dataset's own train and test sets
        "defect_copy_groups_across_builtin_split": int((groups["split"].nunique() > 1).sum()),
    }


def lot_overlap_in_builtin_split(df: pd.DataFrame) -> dict[str, int]:
    """checks if the dataset's own training and test sets share lots
    if they do that split leaks and cannot be used as is"""
    labeled = df[df["split"].notna()]
    train_lots = set(labeled.loc[labeled["split"] == "Training", "lot"])
    test_lots = set(labeled.loc[labeled["split"] == "Test", "lot"])
    shared = train_lots & test_lots
    return {
        "train_lots": len(train_lots),
        "test_lots": len(test_lots),
        "shared_lots": len(shared),
        # wafers in the test set whose lot also shows up in training
        "test_maps_in_shared_lots": int(
            labeled[(labeled["split"] == "Test") & labeled["lot"].isin(shared)].shape[0]
        ),
        "test_maps": int((labeled["split"] == "Test").sum()),
    }


def lot_summary(df: pd.DataFrame) -> dict[str, float]:
    """basic lot facts  how many lots  wafers per lot  odd lot names"""
    per_lot = df.groupby("lot").size()
    # a normal lot name looks like lot123  anything else is worth a look
    odd = ~df["lot"].str.fullmatch(r"lot\d+")
    return {
        "lots": int(per_lot.size),
        "wafers_per_lot_median": float(per_lot.median()),
        "wafers_per_lot_max": int(per_lot.max()),
        "maps_with_odd_lot_name": int(odd.sum()),
        # a lot has 25 slots so an index outside 1 to 25 is suspicious
        "wafer_index_min": int(df["wafer_index"].min()),
        "wafer_index_max": int(df["wafer_index"].max()),
    }


def size_summary(df: pd.DataFrame) -> pd.DataFrame:
    """how many distinct map grid sizes there are and how common each is"""
    sizes = df.groupby(["rows", "cols"]).size().rename("maps").reset_index()
    return sizes.sort_values("maps", ascending=False)
