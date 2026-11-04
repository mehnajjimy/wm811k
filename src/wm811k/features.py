"""hand built features for the random forest baseline
each feature is something a yield engineer would look at by eye on a bin map
so the model stays explainable and size independent"""

import itertools

import numpy as np
import pandas as pd
from scipy import ndimage

# d37 radius is measured as a fraction of the wafer radius so a 25 by 27 map and
# a 49 by 39 map describe the same physical ring with the same number
RING_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
N_SECTORS = 8
OUTER_RING = 0.7


def _polar(wafer_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """normalised radius and angle of every die that sits on the wafer"""
    on = wafer_map > 0
    rows, cols = np.nonzero(on)
    # the centre is the middle of the dies that exist not the middle of the grid
    # because some maps have an empty row or column along one side
    cy, cx = rows.mean(), cols.mean()
    dy, dx = rows - cy, cols - cx
    # each axis is scaled by its own half width so rectangular grids still give
    # a radius of one at the edge in every direction
    ry = max(np.abs(dy).max(), 1.0)
    rx = max(np.abs(dx).max(), 1.0)
    r = np.sqrt((dy / ry) ** 2 + (dx / rx) ** 2)
    theta = np.arctan2(dy / ry, dx / rx)
    return r, theta, wafer_map[rows, cols] == 2


def _density(fails: np.ndarray, mask: np.ndarray) -> float:
    """share of dies in a region that failed  zero for an empty region"""
    n = mask.sum()
    return float(fails[mask].mean()) if n else 0.0


def extract(wafer_map: np.ndarray) -> dict[str, float]:
    """every feature for one map as a flat dict"""
    r, theta, fails = _polar(wafer_map)
    f: dict[str, float] = {"fail_fraction": float(fails.mean())}

    # radial profile  center  donut and edge ring all show up as which ring
    # carries the fails
    for i, (lo, hi) in enumerate(itertools.pairwise(RING_EDGES)):
        f[f"ring_{i}_density"] = _density(fails, (r >= lo) & (r < hi))
    rings = np.array([f[f"ring_{i}_density"] for i in range(len(RING_EDGES) - 1)])
    # how much more the worst ring fails than the wafer as a whole
    f["ring_peak_contrast"] = float(rings.max() - f["fail_fraction"])
    f["ring_peak_index"] = float(rings.argmax())

    # edge sectors  edge ring fails all the way round while edge loc fails in one
    # part of the edge  sectors are sorted so the feature does not care which
    # way the wafer was rotated
    outer = r >= OUTER_RING
    sector = ((theta + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    sectors = np.sort([_density(fails, outer & (sector == s)) for s in range(N_SECTORS)])
    for i, v in enumerate(sectors[::-1]):
        f[f"edge_sector_rank_{i}"] = float(v)
    f["edge_sector_spread"] = float(sectors.max() - sectors.min())

    # clusters of failing dies that touch each other
    fail_grid = wafer_map == 2
    labels, n = ndimage.label(fail_grid, structure=np.ones((3, 3)))
    f["fail_clusters"] = float(n)
    if n:
        sizes = np.bincount(labels.ravel())[1:]
        biggest = labels == (sizes.argmax() + 1)
        f["biggest_cluster_share"] = float(sizes.max() / fail_grid.sum())
        f["biggest_cluster_size_frac"] = float(sizes.max() / (wafer_map > 0).sum())
        f.update(_shape(biggest, wafer_map))
    else:
        f.update(
            biggest_cluster_share=0.0,
            biggest_cluster_size_frac=0.0,
            biggest_cluster_elongation=0.0,
            biggest_cluster_radius=0.0,
            biggest_cluster_fill=0.0,
        )

    # how many failing dies have failing neighbours  random fails are mostly
    # alone while systematic patterns come in solid patches
    neighbours = ndimage.convolve(fail_grid.astype(int), np.ones((3, 3)), mode="constant") - 1
    f["fail_with_3plus_fail_neighbours"] = (
        float((neighbours[fail_grid] >= 3).mean()) if fail_grid.any() else 0.0
    )
    return f


def _shape(cluster: np.ndarray, wafer_map: np.ndarray) -> dict[str, float]:
    """shape of the biggest fail cluster  a scratch is long and thin and a loc
    is a round blob away from the edge"""
    ys, xs = np.nonzero(cluster)
    if len(ys) < 3:
        return {
            "biggest_cluster_elongation": 0.0,
            "biggest_cluster_radius": 0.0,
            "biggest_cluster_fill": 0.0,
        }
    # the two spreads of the cluster along its own main axes  a ratio near one is
    # round and a big ratio is a line
    cov = np.cov(np.vstack([ys, xs]))
    eig = np.sort(np.linalg.eigvalsh(cov))
    elongation = float(np.sqrt(eig[1] / max(eig[0], 1e-3)))

    # where the cluster sits from center to edge
    on_rows, on_cols = np.nonzero(wafer_map > 0)
    cy, cx = on_rows.mean(), on_cols.mean()
    ry = max(np.abs(on_rows - cy).max(), 1.0)
    rx = max(np.abs(on_cols - cx).max(), 1.0)
    radius = float(np.hypot((ys.mean() - cy) / ry, (xs.mean() - cx) / rx))

    # how solid the cluster is inside its bounding box  a diagonal scratch fills
    # very little of its box while a blob fills most of it
    box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    fill = float(len(ys) / box)
    return {
        "biggest_cluster_elongation": elongation,
        "biggest_cluster_radius": radius,
        "biggest_cluster_fill": fill,
    }


def extract_all(maps: pd.Series) -> pd.DataFrame:
    """features for a column of maps  one row per map  same index"""
    return pd.DataFrame([extract(m) for m in maps], index=maps.index)
