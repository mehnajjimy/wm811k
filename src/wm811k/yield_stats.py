"""die yield per wafer and yield loss per defect pattern"""

import numpy as np
import pandas as pd


def die_yield(wafer_map: np.ndarray) -> float:
    """fraction of tested dies that passed
    0 is no die 1 is pass 2 is fail so only 1s and 2s count"""
    # dies off the wafer edge are 0 and are not tested so they are left out
    passed = (wafer_map == 1).sum()
    failed = (wafer_map == 2).sum()
    # a map with no tested dies has no yield so return nan instead of crashing
    if passed + failed == 0:
        return float("nan")
    return float(passed / (passed + failed))


def count_tested_dies(wafer_map: np.ndarray) -> int:
    """how many dies were actually probed on this wafer"""
    return int(((wafer_map == 1) | (wafer_map == 2)).sum())


def per_wafer(df: pd.DataFrame) -> pd.DataFrame:
    """adds yield  tested dies and failed dies to every map"""
    out = df.drop(columns="wafer_map").copy()
    out["tested_dies"] = df["wafer_map"].map(count_tested_dies)
    out["failed_dies"] = df["wafer_map"].map(lambda m: int((m == 2).sum()))
    out["yield"] = 1 - out["failed_dies"] / out["tested_dies"]
    return out


def clean_baseline(wafers: pd.DataFrame, min_clean: int = 30) -> tuple[pd.Series, float]:
    """the yield a clean wafer of the same product would normally get
    gives back one baseline per wafer and the overall clean median"""
    # even a wafer with no pattern loses some dies to random defects so the
    # median yield of none wafers is the fair baseline
    # d16 median not mean because a few very bad none wafers would drag the mean
    clean = wafers[wafers["label"] == "none"]
    overall = float(clean["yield"].median())

    # d17 different products run at different normal yields and the map grid
    # size is the best product marker in the data  so each wafer is compared
    # with clean wafers of its own grid size  sizes with too few clean wafers
    # for a stable median fall back to the overall clean median
    by_size = clean.groupby(["rows", "cols"])["yield"].agg(["median", "size"])
    by_size = by_size.loc[by_size["size"] >= min_clean, "median"]
    keys = pd.MultiIndex.from_frame(wafers[["rows", "cols"]])
    per_wafer_baseline = pd.Series(by_size.reindex(keys).to_numpy(), index=wafers.index)
    return per_wafer_baseline.fillna(overall), overall


def excess_loss(wafers: pd.DataFrame) -> tuple[pd.Series, float]:
    """dies lost on each wafer beyond what a clean wafer of the same product
    would lose anyway  gives back the excess per wafer and the overall clean median"""
    baseline, overall = clean_baseline(wafers)
    expected_fails = (1 - baseline) * wafers["tested_dies"]
    # can go negative when a patterned wafer still beats the clean median
    # and that is kept so averages are not biased upward
    return wafers["failed_dies"] - expected_fails, overall


def loss_by_class(wafers: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """one row per class with how often it shows up  how bad it is  and how
    much of the total excess loss it accounts for"""
    labeled = wafers[wafers["label"].notna()].copy()
    labeled["excess"], baseline = excess_loss(labeled)

    g = labeled.groupby("label")
    table = pd.DataFrame(
        {
            "maps": g.size(),
            "share_of_maps": g.size() / len(labeled),
            "yield_median": g["yield"].median(),
            # the middle half of wafers shows spread without the extremes
            "yield_q25": g["yield"].quantile(0.25),
            "yield_q75": g["yield"].quantile(0.75),
            "excess_dies_per_wafer": g["excess"].mean(),
            "excess_dies_total": g["excess"].sum(),
        }
    )
    # the pareto number  how many extra dies each pattern costs for every
    # thousand labeled wafers  this mixes how common it is with how bad it is
    table["excess_dies_per_1000_wafers"] = table["excess_dies_total"] / len(labeled) * 1000

    # share of all positive excess loss  none is left out because it is the baseline
    defects = table.drop(index="none", errors="ignore")
    total = defects["excess_dies_total"].clip(lower=0).sum()
    table["share_of_excess_loss"] = defects["excess_dies_total"].clip(lower=0) / total
    table = table.sort_values("excess_dies_per_1000_wafers", ascending=False)
    return table, baseline
