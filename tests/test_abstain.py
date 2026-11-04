import numpy as np
import pandas as pd

from wm811k.abstain import coverage_curve, gate_curve, none_gate, pick_gate

CLASSES = ["Loc", "Scratch", "none"]


def _table() -> pd.DataFrame:
    # two clean wafers the model is sure about  one scratch it almost calls clean
    # and one loc it gets right
    return pd.DataFrame(
        {
            "label": ["none", "none", "Scratch", "Loc"],
            "p_Loc": [0.02, 0.05, 0.10, 0.80],
            "p_Scratch": [0.03, 0.15, 0.35, 0.10],
            "p_none": [0.95, 0.80, 0.55, 0.10],
        }
    )


def test_gate_only_calls_none_above_the_threshold():
    t = _table()
    assert list(none_gate(t, CLASSES, 0.5)) == ["none", "none", "none", "Loc"]
    # at 0.6 the unsure scratch is no longer called clean and gets its best defect
    assert list(none_gate(t, CLASSES, 0.6)) == ["none", "none", "Scratch", "Loc"]


def test_stricter_gate_misses_fewer_defects_and_flags_more_clean():
    curve = gate_curve(_table(), CLASSES, np.array([0.5, 0.6, 0.9]))
    assert curve["missed_defect_share"].is_monotonic_decreasing
    assert curve["clean_flagged_per_1000"].is_monotonic_increasing
    assert curve.loc[1, "missed_defect_share"] == 0
    assert curve.loc[2, "clean_flagged_per_1000"] == 500


def test_pick_gate_takes_the_lowest_threshold_that_meets_the_target():
    curve = gate_curve(_table(), CLASSES, np.array([0.5, 0.6, 0.9]))
    assert pick_gate(curve, max_missed=0.0) == 0.6
    # an impossible target falls back to the strictest threshold
    impossible = curve.assign(missed_defect_share=1.0)
    assert pick_gate(impossible, max_missed=0.0) == 0.9


def test_coverage_drops_the_least_sure_wafers_first():
    curve = coverage_curve(_table(), CLASSES, np.array([0.5, 1.0]))
    # keeping the two surest wafers keeps the two clean ones
    assert curve.loc[0, "defects_sent_to_review"] == 2
    assert curve.loc[1, "defects_sent_to_review"] == 0
