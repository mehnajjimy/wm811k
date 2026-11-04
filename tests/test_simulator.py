"""tests for the process fault simulator
each test checks one promise the simulator makes to a process engineer"""

import math

import numpy as np
import pytest

from wm811k.simulator import (
    FAIL,
    GENERATORS,
    NO_DIE,
    PASS,
    WaferGrid,
    combine,
    edge_loc_fault,
    edge_ring_fault,
    generate,
    random_defects,
    scratch_fault,
)
from wm811k.yield_stats import die_yield

PATTERNS = sorted(GENERATORS)

# the strength knob for each class and a weak and strong setting
# every other input stays at its default so only the fault size changes
STRENGTH = {
    "Center": ("peak_deviation_pct", 6.0, 12.0),
    "Donut": ("peak_deviation_pct", 6.0, 12.0),
    "Edge-Loc": ("peak_deviation_pct", 15.0, 40.0),
    "Edge-Ring": ("peak_deviation_pct", 15.0, 40.0),
    "Loc": ("particle_count", 40.0, 300.0),
    "Near-full": ("kill_probability", 0.6, 0.97),
    "Random": ("d0_per_cm2", 0.2, 1.5),
    "Scratch": ("kill_probability", 0.3, 1.0),
    "none": ("d0_per_cm2", 0.01, 0.1),
}

# the most common wm811k map size from results audit_map_sizes csv
# it is not square so it checks the rectangular die path
COMMON_WM811K_SHAPE = (32, 29)


# shape and values
# a map must line up die for die with the grid and hold only the three codes


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("shape", [(8, 8), (26, 26), (52, 52), COMMON_WM811K_SHAPE, (64, 71)])
def test_shape_matches_grid(pattern: str, shape: tuple[int, int]) -> None:
    grid = WaferGrid(rows=shape[0], cols=shape[1])
    w = generate(pattern, grid, seed=1)
    assert w.wafer_map.shape == shape
    assert w.fail_probability.shape == shape


def test_labels_match_the_dataset() -> None:
    # synthetic and real maps must share one label set or a synthetic to real test cannot line up
    pytest.importorskip("pandas")
    from wm811k.data import CLASSES

    assert set(GENERATORS) == set(CLASSES)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_values_are_only_0_1_2(pattern: str) -> None:
    w = generate(pattern, seed=3)
    assert set(np.unique(w.wafer_map)) <= {NO_DIE, PASS, FAIL}
    assert w.wafer_map.dtype == np.uint8


@pytest.mark.parametrize("pattern", PATTERNS)
def test_no_die_mask_is_the_same_for_every_pattern(pattern: str) -> None:
    # the wafer outline comes from the grid alone so a fault must never add or remove dies
    grid = WaferGrid()
    w = generate(pattern, grid, seed=0)
    assert np.array_equal(w.wafer_map != NO_DIE, grid.on_wafer())


def test_grid_is_round() -> None:
    # corners of the square grid lie outside the disc so they hold no die
    on = WaferGrid(rows=30, cols=30).on_wafer()
    assert not on[0, 0] and not on[0, -1] and not on[-1, 0] and not on[-1, -1]
    assert on[15, 15]
    assert np.array_equal(on, on[::-1, :]) and np.array_equal(on, on[:, ::-1])


def test_bad_grid_is_rejected() -> None:
    with pytest.raises(ValueError):
        WaferGrid(rows=2)
    with pytest.raises(ValueError):
        WaferGrid(edge_exclusion_mm=-1.0)


# reproducibility
# a seed must pin the map exactly so any figure or result can be rebuilt


@pytest.mark.parametrize("pattern", PATTERNS)
def test_same_seed_same_map(pattern: str) -> None:
    a = generate(pattern, seed=42)
    b = generate(pattern, seed=42)
    assert np.array_equal(a.wafer_map, b.wafer_map)
    assert a.params == b.params


@pytest.mark.parametrize("pattern", PATTERNS)
def test_different_seed_different_map(pattern: str) -> None:
    a = generate(pattern, seed=1)
    b = generate(pattern, seed=2)
    assert not np.array_equal(a.wafer_map, b.wafer_map)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_params_rebuild_the_map(pattern: str) -> None:
    # random layout choices are written back into params so params alone rebuild the wafer
    w = generate(pattern, seed=7)
    grid_keys = ("rows", "cols", "wafer_diameter_mm", "edge_exclusion_mm")
    grid = WaferGrid(**{k: w.params[k] for k in grid_keys})
    skip = {"pattern", "seed", "wafer_d0_per_cm2", *grid_keys}
    if pattern in ("Random", "none"):
        # here the background density is the fault itself and is passed as d0
        skip.add("background_d0_per_cm2")
    rebuilt = generate(
        pattern, grid, seed=w.params["seed"], **{k: v for k, v in w.params.items() if k not in skip}
    )
    assert np.array_equal(w.wafer_map, rebuilt.wafer_map)


# yield direction
# a stronger fault must give lower yield on the same seed
# the die draws do not depend on fault size so this holds die by die not only on average


@pytest.mark.parametrize("pattern", PATTERNS)
def test_stronger_fault_lowers_yield(pattern: str) -> None:
    knob, weak, strong = STRENGTH[pattern]
    for seed in range(5):
        w_weak = generate(pattern, seed=seed, **{knob: weak})
        w_strong = generate(pattern, seed=seed, **{knob: strong})
        assert w_strong.die_yield < w_weak.die_yield
        assert w_strong.expected_yield < w_weak.expected_yield
        # every die that failed under the weak fault still fails under the strong one
        assert np.all(w_strong.wafer_map[w_weak.wafer_map == FAIL] == FAIL)


@pytest.mark.parametrize("pattern", ["Center", "Donut", "Edge-Ring", "Edge-Loc", "Loc"])
def test_zero_fault_leaves_only_background(pattern: str) -> None:
    knob = STRENGTH[pattern][0]
    w = generate(pattern, seed=0, background_d0_per_cm2=0.0, **{knob: 0.0})
    assert w.die_yield > 0.999


def test_longer_scratch_kills_more() -> None:
    kw = {
        "start_x_mm": -100.0,
        "start_y_mm": 0.0,
        "heading_deg": 0.0,
        "arc_radius_mm": None,
        "kill_probability": 1.0,
    }
    short = scratch_fault(length_mm=50.0, seed=0, **kw)
    long = scratch_fault(length_mm=180.0, seed=0, **kw)
    assert long.die_yield < short.die_yield


def test_wider_rolloff_lowers_edge_ring_yield() -> None:
    narrow = edge_ring_fault(rolloff_length_mm=5.0, seed=0)
    wide = edge_ring_fault(rolloff_length_mm=15.0, seed=0)
    assert wide.die_yield < narrow.die_yield


# physics checks
# these test that the shape of the damage matches the mechanism and not just the yield number


def test_poisson_yield_matches_formula() -> None:
    # with no spatial pattern the mean die yield should match exp of minus area times d0
    grid = WaferGrid()
    d0 = 0.5
    yields = [random_defects(grid, d0_per_cm2=d0, seed=s).die_yield for s in range(40)]
    assert abs(np.mean(yields) - math.exp(-grid.die_area_cm2 * d0)) < 0.01


def test_negative_binomial_spreads_wafer_yield() -> None:
    # small alpha means wafer to wafer density swings so yields spread wider than plain poisson
    plain = [random_defects(d0_per_cm2=0.5, seed=s).die_yield for s in range(40)]
    clustered = [
        random_defects(d0_per_cm2=0.5, cluster_alpha=1.0, seed=s).die_yield for s in range(40)
    ]
    assert np.std(clustered) > 3 * np.std(plain)


def test_edge_ring_is_symmetric_and_edge_loc_is_not() -> None:
    # compare fail chance on the left and right halves of the wafer
    grid = WaferGrid()
    x, _ = grid.die_centers_mm()
    on = grid.on_wafer()
    left, right = on & (x < 0), on & (x > 0)
    # background is switched off because it adds the same fail chance to both halves
    ring = edge_ring_fault(grid, background_d0_per_cm2=0.0, seed=0).fail_probability
    loc = edge_loc_fault(grid, angle_deg=0.0, background_d0_per_cm2=0.0, seed=0).fail_probability
    assert math.isclose(ring[left].sum(), ring[right].sum(), rel_tol=1e-9)
    # angle zero shifts the wafer so its right side lands in the bad edge zone
    assert loc[right].sum() > 5 * loc[left].sum()


def test_off_center_placement_alone_gives_one_sided_edge() -> None:
    # with no local arc the only thing breaking symmetry is the shifted wafer
    grid = WaferGrid()
    x, _ = grid.die_centers_mm()
    on = grid.on_wafer()
    shifted = edge_loc_fault(
        grid,
        peak_deviation_pct=9.0,
        arc_width_deg=None,
        placement_offset_mm=8.0,
        angle_deg=180.0,
        background_d0_per_cm2=0.0,
        seed=0,
    ).fail_probability
    assert shifted[on & (x < 0)].sum() > 5 * shifted[on & (x > 0)].sum()


def test_center_fails_concentrate_at_center() -> None:
    grid = WaferGrid()
    x, y = grid.die_centers_mm()
    r = np.hypot(x, y)
    w = generate("Center", grid, seed=0, background_d0_per_cm2=0.0)
    fails = w.wafer_map == FAIL
    assert r[fails].max() < 0.5 * grid.radius_mm


def test_donut_spares_the_center() -> None:
    grid = WaferGrid()
    x, y = grid.die_centers_mm()
    r = np.hypot(x, y)
    w = generate("Donut", grid, seed=0, background_d0_per_cm2=0.0)
    p = w.fail_probability
    assert p[r < 20].max() < 0.01
    assert p[np.abs(r - 80) < 5].mean() > 0.5


def test_loc_fails_sit_near_cluster_center() -> None:
    w = generate("Loc", seed=0, center_x_mm=30.0, center_y_mm=-20.0, background_d0_per_cm2=0.0)
    x, y = WaferGrid().die_centers_mm()
    dist = np.hypot(x - 30.0, y + 20.0)
    # past four cluster sigmas the particle cloud is too thin to kill
    sigma = w.params["cluster_sigma_mm"]
    assert w.fail_probability[dist > 4 * sigma].max() < 0.01
    assert w.fail_probability[dist < sigma].min() > 0.5


# helpers and inputs


def test_property_yield_uses_the_shared_formula() -> None:
    # the simulator must report yield the same way the real data analysis does
    w = generate("Loc", seed=4)
    assert w.die_yield == die_yield(w.wafer_map)


def test_rectangular_dies_fit_the_wafer() -> None:
    # with fewer columns than rows each die is wider than it is tall
    grid = WaferGrid(rows=COMMON_WM811K_SHAPE[0], cols=COMMON_WM811K_SHAPE[1])
    assert grid.pitch_x_mm > grid.pitch_y_mm
    assert math.isclose(grid.die_area_cm2, grid.pitch_x_mm * grid.pitch_y_mm / 100)
    on = grid.on_wafer()
    assert np.array_equal(on, on[::-1, :]) and np.array_equal(on, on[:, ::-1])


def test_combine_keeps_every_fail() -> None:
    a = generate("Scratch", seed=1)
    b = generate("Loc", seed=2)
    both = combine(a, b)
    assert np.all(both.wafer_map[(a.wafer_map == FAIL) | (b.wafer_map == FAIL)] == FAIL)
    assert both.die_yield <= min(a.die_yield, b.die_yield)
    assert both.pattern == "Scratch+Loc"


def test_combine_rejects_mismatched_grids() -> None:
    with pytest.raises(ValueError):
        combine(generate("Loc", WaferGrid(rows=20, cols=20), seed=0), generate("Loc", seed=0))


def test_unknown_pattern_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate("swirl", seed=0)


@pytest.mark.parametrize(
    ("pattern", "bad"),
    [
        ("Center", {"zone_width_mm": 0.0}),
        ("Donut", {"ring_radius_mm": 500.0}),
        ("Edge-Ring", {"die_sigma_pct": -1.0}),
        ("Edge-Loc", {"arc_width_deg": None, "placement_offset_mm": 0.0}),
        ("Scratch", {"kill_probability": 1.5}),
        ("Loc", {"kill_ratio": -0.1}),
        ("Random", {"d0_per_cm2": -1.0}),
        ("Random", {"cluster_alpha": 0.0}),
        ("Near-full", {"kill_probability": 2.0}),
    ],
)
def test_bad_physical_inputs_are_rejected(pattern: str, bad: dict) -> None:
    with pytest.raises(ValueError):
        generate(pattern, seed=0, **bad)


def test_synthetic_set_covers_every_class_and_respects_the_background_band():
    import pandas as pd

    from wm811k.data import CLASSES
    from wm811k.synthetic import make_set

    sizes = pd.DataFrame({"rows": [32, 49], "cols": [29, 39], "maps": [3, 1]})
    cfg = {"background_fail_share": [0.01, 0.15], "random_fail_share": [0.2, 0.6]}
    s = make_set(15, sizes, cfg, seed=0)
    assert sorted(s["label"].unique()) == sorted(CLASSES)
    fail = s["wafer_map"].map(lambda m: (m == 2).sum() / (m > 0).sum())
    by = fail.groupby(s["label"]).median()
    # clean wafers stay inside the background band and random sits above it
    assert 0.01 <= by["none"] <= 0.15
    assert by["Random"] > by["none"]
    assert by["Near-full"] > 0.7
