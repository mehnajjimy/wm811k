import numpy as np

from wm811k.features import extract
from wm811k.simulator import generate


def _avg(pattern: str, key: str, n: int = 10) -> float:
    """average one feature over a few simulated wafers of a class"""
    return float(np.mean([extract(generate(pattern, seed=s).wafer_map)[key] for s in range(n)]))


def test_center_fails_sit_in_the_middle_and_edge_ring_at_the_edge():
    # the inner ring is ring 0 and the outer ring is ring 4
    assert _avg("Center", "ring_0_density") > _avg("Center", "ring_4_density")
    assert _avg("Edge-Ring", "ring_4_density") > _avg("Edge-Ring", "ring_0_density")


def test_edge_loc_is_lopsided_and_edge_ring_is_not():
    # edge loc fails in one part of the edge so its sectors differ much more
    assert _avg("Edge-Loc", "edge_sector_spread") > _avg("Edge-Ring", "edge_sector_spread")


def test_scratch_cluster_is_long_and_thin():
    assert _avg("Scratch", "biggest_cluster_elongation") > _avg("Loc", "biggest_cluster_elongation")


def test_random_fails_are_mostly_alone_compared_with_loc():
    # loc fails come in a solid patch so most of them have failing neighbours
    # while random fails are scattered and split into many small clusters
    loc_n = _avg("Loc", "fail_with_3plus_fail_neighbours")
    random_n = _avg("Random", "fail_with_3plus_fail_neighbours")
    assert loc_n > random_n
    assert _avg("Random", "fail_clusters") > _avg("Loc", "fail_clusters")


def test_features_do_not_depend_on_map_size():
    # the same physical center fault on a small and a large grid should give
    # similar ring densities and the same worst ring because radius is normalised
    from wm811k.simulator import WaferGrid

    small = generate("Center", WaferGrid(rows=25, cols=27), seed=1).wafer_map
    large = generate("Center", WaferGrid(rows=49, cols=39), seed=1).wafer_map
    a, b = extract(small), extract(large)
    assert abs(a["ring_0_density"] - b["ring_0_density"]) < 0.25
    assert a["ring_peak_index"] == b["ring_peak_index"]


def test_clean_map_has_zero_cluster_features():
    m = np.ones((10, 10), dtype=int)
    f = extract(m)
    assert f["fail_fraction"] == 0 and f["fail_clusters"] == 0
