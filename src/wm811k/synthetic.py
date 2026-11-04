"""synthetic training sets from the process fault simulator for item f prime
every map gets a grid size drawn from real map sizes and fault settings drawn
from wide ranges around the simulator defaults so a model trained on them
sees many versions of each fault instead of one tidy picture"""

import numpy as np
import pandas as pd

from wm811k.data import CLASSES
from wm811k.simulator import WaferGrid, generate


def _background_d0(grid: WaferGrid, fail_share: float) -> float:
    """defect density that makes a given share of dies fail on this grid
    poisson yield is exp of minus area times density so solve that for density"""
    return float(-np.log1p(-fail_share) / grid.die_area_cm2)


def random_settings(pattern: str, grid: WaferGrid, rng: np.random.Generator, cfg: dict) -> dict:
    """fault settings for one synthetic map
    d51 ranges are set by physics and by label free facts about the real maps
    and are not tuned on any labeled real wafer"""
    u = rng.uniform
    lo, hi = cfg["background_fail_share"]
    bg = {"background_d0_per_cm2": _background_d0(grid, u(lo, hi))}

    def scale() -> float:
        # strengths and sizes vary from 0.6 to 1.6 times the simulator default
        return u(0.6, 1.6)

    if pattern == "Center":
        return bg | {"peak_deviation_pct": 9 * scale(), "zone_width_mm": 30 * scale()}
    if pattern == "Donut":
        return bg | {
            "peak_deviation_pct": 9 * scale(),
            "ring_radius_mm": u(50, 110),
            "ring_width_mm": u(8, 20),
        }
    if pattern == "Edge-Ring":
        return bg | {"peak_deviation_pct": 30 * scale(), "rolloff_length_mm": 10 * scale()}
    if pattern == "Edge-Loc":
        return bg | {
            "peak_deviation_pct": 30 * scale(),
            "rolloff_length_mm": 15 * scale(),
            "arc_width_deg": u(15, 70),
        }
    if pattern == "Loc":
        return bg | {
            "particle_count": u(50, 400),
            "cluster_sigma_mm": u(6, 25),
            "kill_ratio": u(0.2, 0.7),
        }
    if pattern == "Scratch":
        # about a third straight for handling and the rest curved for cmp
        arc = None if rng.random() < 1 / 3 else u(100, 400)
        return bg | {
            "length_mm": u(60, 250),
            "arc_radius_mm": arc,
            "kill_probability": u(0.6, 0.95),
        }
    if pattern == "Near-full":
        return bg | {"kill_probability": u(0.75, 0.98)}
    if pattern == "Random":
        # random sits well above the background so its fail share is drawn
        # from a higher band and turned into a density the same way
        return {"d0_per_cm2": _background_d0(grid, u(*cfg["random_fail_share"]))}
    if pattern == "none":
        return {"d0_per_cm2": _background_d0(grid, u(lo, hi))}
    raise ValueError(f"unknown pattern {pattern!r}")


def make_set(n_per_class: int, sizes: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """n maps per class with grid sizes drawn from sizes in proportion to how
    often each size shows up  gives back one row per map with its settings"""
    rng = np.random.default_rng(seed)
    weights = sizes["maps"].to_numpy() / sizes["maps"].sum()
    rows = []
    for c in CLASSES:
        for _ in range(n_per_class):
            r, k = sizes.iloc[rng.choice(len(sizes), p=weights)][["rows", "cols"]]
            grid = WaferGrid(rows=int(r), cols=int(k))
            settings = random_settings(c, grid, rng, cfg)
            w = generate(c, grid, seed=int(rng.integers(0, 2**31)), **settings)
            rows.append({"label": c, "rows": int(r), "cols": int(k), "wafer_map": w.wafer_map,
                         "settings": settings})  # fmt: skip
    return pd.DataFrame(rows)
