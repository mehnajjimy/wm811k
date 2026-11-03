import math

import numpy as np

from wm811k.yield_stats import count_tested_dies, die_yield


def test_yield_ignores_empty_positions():
    # three passes one fail and the zeros are off the wafer so yield is 3 out of 4
    m = np.array([[0, 1, 1], [1, 2, 0]])
    assert die_yield(m) == 0.75
    assert count_tested_dies(m) == 4


def test_yield_of_map_with_no_tested_dies_is_nan():
    # an all zero map has nothing to measure so it must not read as 0 or 1
    assert math.isnan(die_yield(np.zeros((3, 3))))


def test_full_fail_and_full_pass():
    assert die_yield(np.full((2, 2), 2)) == 0.0
    assert die_yield(np.ones((2, 2))) == 1.0


def test_baseline_is_per_product():
    import pandas as pd

    from wm811k.yield_stats import clean_baseline

    # two products  the small one runs clean at 90 percent and the big one at 80
    # so a patterned wafer must be judged against its own product not the mix
    rows = []
    for _ in range(40):
        rows.append({"rows": 10, "cols": 10, "label": "none", "yield": 0.9})
        rows.append({"rows": 20, "cols": 20, "label": "none", "yield": 0.8})
    rows.append({"rows": 20, "cols": 20, "label": "Center", "yield": 0.6})
    # a product with too few clean wafers falls back to the overall median
    rows.append({"rows": 5, "cols": 5, "label": "Loc", "yield": 0.7})
    df = pd.DataFrame(rows)
    baseline, overall = clean_baseline(df)
    assert baseline.iloc[-2] == 0.8
    assert baseline.iloc[-1] == overall
