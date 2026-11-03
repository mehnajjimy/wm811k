import numpy as np
import pandas as pd

from wm811k.lots import excursion_lots, lot_clustering, slot_order

CLASSES = ["none", "Center"]


def _pool(clustered: bool) -> pd.DataFrame:
    """fake lots of 20 wafers  center either piles into a few lots or spreads out"""
    rows = []
    for lot in range(50):
        for slot in range(1, 21):
            if clustered:
                # every center wafer sits in the first five lots in a run of slots
                label = "Center" if lot < 5 and slot <= 10 else "none"
            else:
                label = "Center" if (lot * 20 + slot) % 20 == 0 else "none"
            rows.append({"lot": f"lot{lot}", "wafer_index": slot, "label": label})
    return pd.DataFrame(rows)


def test_clustered_pattern_scores_above_chance():
    result = lot_clustering(_pool(clustered=True), CLASSES, n_perm=50)
    assert result.loc["Center", "ratio"] > 3
    assert result.loc["Center", "p_value"] < 0.05


def test_spread_pattern_does_not_cluster():
    # one center wafer per lot means no wafer ever has a same class partner
    result = lot_clustering(_pool(clustered=False), CLASSES, n_perm=50)
    assert result.loc["Center", "ratio"] < 1


def test_slot_runs_show_up_and_lot_mix_is_kept():
    pool = _pool(clustered=True)
    result = slot_order(pool, CLASSES, n_perm=50)
    # center fills slots 1 to 10 in a row so neighbours match far above chance
    assert result.loc["Center", "ratio"] > 1.5


def test_excursion_lot_found():
    found = excursion_lots(_pool(clustered=True))
    assert set(found["lot"]) == {f"lot{i}" for i in range(5)}
    assert np.allclose(found["share"], 0.5)
