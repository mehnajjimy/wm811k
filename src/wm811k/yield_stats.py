"""die yield per wafer"""

import numpy as np


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
