"""process fault simulator for synthetic wafer maps

each generator turns one physical process fault into a fail probability for every die
then it draws pass or fail for each die from that probability with a fixed seed
map values are 0 no die 1 pass 2 fail
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from wm811k.yield_stats import die_yield

NO_DIE = 0
PASS = 1
FAIL = 2


# the wafer grid
# a wafer is a disc cut into a grid of rows and columns of dies
# rows and cols can differ because real dies are often not square
# wm811k maps come in hundreds of sizes so the grid has to match any of them
# d29 a die only exists if all four corners sit inside the usable disc
# the usable disc is the wafer minus an edge exclusion ring where no product is printed
# real fabs set the exclusion in mm so the grid takes it in mm too


@dataclass(frozen=True)
class WaferGrid:
    """die layout on a round wafer
    sizes are in mm and the die pitch is the wafer diameter over the die count on that axis"""

    rows: int = 40
    cols: int = 40
    wafer_diameter_mm: float = 300.0
    edge_exclusion_mm: float = 3.0

    def __post_init__(self) -> None:
        if self.rows < 4 or self.cols < 4:
            raise ValueError("rows and cols must be at least 4")
        if self.wafer_diameter_mm <= 0:
            raise ValueError("wafer_diameter_mm must be positive")
        if not 0 <= self.edge_exclusion_mm < self.wafer_diameter_mm / 4:
            raise ValueError("edge_exclusion_mm must be between 0 and a quarter of the diameter")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def radius_mm(self) -> float:
        return self.wafer_diameter_mm / 2

    @property
    def pitch_x_mm(self) -> float:
        return self.wafer_diameter_mm / self.cols

    @property
    def pitch_y_mm(self) -> float:
        return self.wafer_diameter_mm / self.rows

    @property
    def die_area_cm2(self) -> float:
        # defect density is quoted per cm2 so die area has to be in cm2 too
        return (self.pitch_x_mm / 10) * (self.pitch_y_mm / 10)

    def die_centers_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """x and y of every die center in mm with the wafer center at zero
        row 0 is the top of the wafer so y is flipped"""
        xs = (np.arange(self.cols) + 0.5) * self.pitch_x_mm - self.radius_mm
        ys = (np.arange(self.rows) + 0.5) * self.pitch_y_mm - self.radius_mm
        x, y = np.meshgrid(xs, ys[::-1])
        return x, y

    def on_wafer(self) -> np.ndarray:
        """true where a full die fits inside the usable disc"""
        x, y = self.die_centers_mm()
        # the corner farthest from the wafer center decides if the die is whole
        far_corner = np.hypot(np.abs(x) + self.pitch_x_mm / 2, np.abs(y) + self.pitch_y_mm / 2)
        return far_corner <= self.radius_mm - self.edge_exclusion_mm


# the grid is frozen so one shared default is safe to reuse across calls
DEFAULT_GRID = WaferGrid()


# the result object
# d30 every generator returns the map and the exact parameters that made it
# random choices like where a scratch starts are written back into params
# so any map can be rebuilt later from its params alone


@dataclass(frozen=True)
class SimulatedWafer:
    """one synthetic wafer
    fail_probability is the chance each die fails before the coin flip"""

    pattern: str
    wafer_map: np.ndarray
    fail_probability: np.ndarray
    params: dict[str, Any]

    @property
    def die_yield(self) -> float:
        return die_yield(self.wafer_map)

    @property
    def expected_yield(self) -> float:
        """yield you would get on average over many seeds with the same fault"""
        on = self.wafer_map != NO_DIE
        return float(np.mean(1.0 - self.fail_probability[on]))


# shared helpers
# d24 randomness is split into two streams from one seed
# the layout stream decides where a fault sits like a cluster center or scratch heading
# the die stream holds one uniform number per die and a die fails if its number is below its fail probability
# because the die numbers do not depend on fault strength a stronger fault can only add fails never remove them


def _streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    layout, dies = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(layout), np.random.default_rng(dies)


_erf = np.vectorize(math.erf, otypes=[float])


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    # numpy has no erf so the standard library one is vectorised
    # grids are small so the speed cost does not matter here
    return 0.5 * (1.0 + _erf(np.asarray(z, dtype=float) / math.sqrt(2.0)))


def _poisson_kill(expected_killers: np.ndarray) -> np.ndarray:
    # a die dies if it gets one or more killer defects
    # with poisson counts the chance of zero is exp of minus the mean
    return 1.0 - np.exp(-expected_killers)


def _background(grid: WaferGrid, d0_per_cm2: float) -> np.ndarray:
    # every real wafer has some random killer defects on top of any pattern
    return np.full(grid.shape, 1.0 - math.exp(-grid.die_area_cm2 * d0_per_cm2))


def _either(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    # two independent ways to fail combine as one minus the chance of surviving both
    return 1.0 - (1.0 - p_a) * (1.0 - p_b)


def _out_of_window(
    deviation_pct: np.ndarray, spec_limit_pct: float, die_sigma_pct: float
) -> np.ndarray:
    # each die sees the local mean deviation plus its own small random scatter
    # it fails if it lands outside the plus or minus spec window
    # this is the chance of landing above the upper limit plus below the lower limit
    upper = 1.0 - _norm_cdf((spec_limit_pct - deviation_pct) / die_sigma_pct)
    lower = _norm_cdf((-spec_limit_pct - deviation_pct) / die_sigma_pct)
    return upper + lower


def _finish(
    pattern: str,
    grid: WaferGrid,
    p_fault: np.ndarray,
    background_d0_per_cm2: float,
    die_rng: np.random.Generator,
    params: dict[str, Any],
) -> SimulatedWafer:
    p_fail = _either(p_fault, _background(grid, background_d0_per_cm2))
    on = grid.on_wafer()
    p_fail = np.where(on, p_fail, 0.0)
    draws = die_rng.random(on.shape)
    wafer_map = np.where(on, np.where(draws < p_fail, FAIL, PASS), NO_DIE).astype(np.uint8)
    full_params = {
        "pattern": pattern,
        **asdict(grid),
        **params,
        "background_d0_per_cm2": background_d0_per_cm2,
    }
    return SimulatedWafer(pattern, wafer_map, p_fail, full_params)


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if not value > 0:
            raise ValueError(f"{name} must be positive")


def _check_fraction(**values: float) -> None:
    for name, value in values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")


def _check_nonnegative(**values: float) -> None:
    for name, value in values.items():
        if not value >= 0:
            raise ValueError(f"{name} must be zero or more")


# radial film or etch non uniformity
# deposition etch and cmp all have a removal or growth rate that varies with radius
# here the film thickness or etched depth is written as a percent deviation from target
# d25 a die fails when its local deviation leaves the spec window
# center donut and edge ring are the same physics with the bad zone at a different radius


def center_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    peak_deviation_pct: float = 9.0,
    zone_width_mm: float = 30.0,
    spec_limit_pct: float = 5.0,
    die_sigma_pct: float = 1.5,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """center pattern from a film or etch profile that is off target at the wafer center
    deviation falls off as a gaussian with radius"""
    _check_positive(
        zone_width_mm=zone_width_mm, spec_limit_pct=spec_limit_pct, die_sigma_pct=die_sigma_pct
    )
    _check_nonnegative(
        peak_deviation_pct=peak_deviation_pct, background_d0_per_cm2=background_d0_per_cm2
    )
    _, die_rng = _streams(seed)
    x, y = grid.die_centers_mm()
    r = np.hypot(x, y)
    deviation = peak_deviation_pct * np.exp(-0.5 * (r / zone_width_mm) ** 2)
    p_fault = _out_of_window(deviation, spec_limit_pct, die_sigma_pct)
    params = {
        "peak_deviation_pct": peak_deviation_pct,
        "zone_width_mm": zone_width_mm,
        "spec_limit_pct": spec_limit_pct,
        "die_sigma_pct": die_sigma_pct,
        "seed": seed,
    }
    return _finish("Center", grid, p_fault, background_d0_per_cm2, die_rng, params)


def donut_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    peak_deviation_pct: float = 9.0,
    ring_radius_mm: float = 80.0,
    ring_width_mm: float = 12.0,
    spec_limit_pct: float = 5.0,
    die_sigma_pct: float = 1.5,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """donut pattern from a profile that is off target in a ring at mid radius
    the ring sits at ring_radius_mm with a gaussian cross section"""
    _check_positive(
        ring_width_mm=ring_width_mm, spec_limit_pct=spec_limit_pct, die_sigma_pct=die_sigma_pct
    )
    _check_nonnegative(
        peak_deviation_pct=peak_deviation_pct, background_d0_per_cm2=background_d0_per_cm2
    )
    if not 0 < ring_radius_mm < grid.radius_mm:
        raise ValueError("ring_radius_mm must sit inside the wafer")
    _, die_rng = _streams(seed)
    x, y = grid.die_centers_mm()
    r = np.hypot(x, y)
    deviation = peak_deviation_pct * np.exp(-0.5 * ((r - ring_radius_mm) / ring_width_mm) ** 2)
    p_fault = _out_of_window(deviation, spec_limit_pct, die_sigma_pct)
    params = {
        "peak_deviation_pct": peak_deviation_pct,
        "ring_radius_mm": ring_radius_mm,
        "ring_width_mm": ring_width_mm,
        "spec_limit_pct": spec_limit_pct,
        "die_sigma_pct": die_sigma_pct,
        "seed": seed,
    }
    return _finish("Donut", grid, p_fault, background_d0_per_cm2, die_rng, params)


# edge roll off
# near the wafer edge the plasma sheath bends and gas flow and temperature change
# so etch rate and film thickness drift away from target over the last few mm
# the drift is modelled as an exponential that is largest at the physical wafer edge
# when the drift is the same all the way round the result is edge ring
# edge loc is the same drift confined to one part of the edge
# two things can confine it here
# a local hardware feature like a clamp finger or gas inlet or a local edge bead problem gives an arc
# an off center wafer pushes one side deeper into the edge zone and pulls the other side out
# d26 so edge ring and edge loc share one mechanism and differ in symmetry


def _edge_deviation(
    grid: WaferGrid,
    peak_deviation_pct: float,
    rolloff_length_mm: float,
    angle_deg: float,
    arc_width_deg: float | None,
    placement_offset_mm: float,
) -> np.ndarray:
    x, y = grid.die_centers_mm()
    theta = math.radians(angle_deg)
    # die position seen from the chamber center once the wafer is shifted toward angle_deg
    r_chamber = np.hypot(
        x + placement_offset_mm * math.cos(theta), y + placement_offset_mm * math.sin(theta)
    )
    # distance in from the edge of a chamber that is sized for a centered wafer
    inset_mm = grid.radius_mm - r_chamber
    deviation = peak_deviation_pct * np.exp(-inset_mm / rolloff_length_mm)
    if arc_width_deg is None:
        return deviation
    # angle from each die to the center of the bad arc wrapped into plus or minus 180
    d_theta = np.degrees(np.angle(np.exp(1j * (np.arctan2(y, x) - theta))))
    return deviation * np.exp(-0.5 * (d_theta / arc_width_deg) ** 2)


def edge_ring_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    peak_deviation_pct: float = 30.0,
    rolloff_length_mm: float = 10.0,
    spec_limit_pct: float = 5.0,
    die_sigma_pct: float = 1.5,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """edge ring pattern from edge roll off that is the same all the way round
    peak_deviation_pct is the deviation at the physical wafer edge"""
    _check_positive(
        rolloff_length_mm=rolloff_length_mm,
        spec_limit_pct=spec_limit_pct,
        die_sigma_pct=die_sigma_pct,
    )
    _check_nonnegative(
        peak_deviation_pct=peak_deviation_pct, background_d0_per_cm2=background_d0_per_cm2
    )
    _, die_rng = _streams(seed)
    deviation = _edge_deviation(grid, peak_deviation_pct, rolloff_length_mm, 0.0, None, 0.0)
    p_fault = _out_of_window(deviation, spec_limit_pct, die_sigma_pct)
    params = {
        "peak_deviation_pct": peak_deviation_pct,
        "rolloff_length_mm": rolloff_length_mm,
        "spec_limit_pct": spec_limit_pct,
        "die_sigma_pct": die_sigma_pct,
        "seed": seed,
    }
    return _finish("Edge-Ring", grid, p_fault, background_d0_per_cm2, die_rng, params)


def edge_loc_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    peak_deviation_pct: float = 30.0,
    rolloff_length_mm: float = 15.0,
    angle_deg: float | None = None,
    arc_width_deg: float | None = 25.0,
    placement_offset_mm: float = 0.0,
    spec_limit_pct: float = 5.0,
    die_sigma_pct: float = 1.5,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """edge loc pattern from edge roll off confined to one part of the edge
    angle_deg is where the bad zone sits and is drawn from the seed when not given
    arc_width_deg is the angular spread of a local edge feature and none turns it off
    placement_offset_mm shifts the wafer toward angle_deg in the chamber"""
    _check_positive(
        rolloff_length_mm=rolloff_length_mm,
        spec_limit_pct=spec_limit_pct,
        die_sigma_pct=die_sigma_pct,
    )
    _check_nonnegative(
        peak_deviation_pct=peak_deviation_pct,
        placement_offset_mm=placement_offset_mm,
        background_d0_per_cm2=background_d0_per_cm2,
    )
    if arc_width_deg is not None:
        _check_positive(arc_width_deg=arc_width_deg)
    if arc_width_deg is None and placement_offset_mm == 0:
        raise ValueError(
            "edge loc needs an arc_width_deg or a placement_offset_mm or it is just edge ring"
        )
    layout_rng, die_rng = _streams(seed)
    drawn_angle = float(layout_rng.uniform(0.0, 360.0))
    if angle_deg is None:
        angle_deg = drawn_angle
    deviation = _edge_deviation(
        grid, peak_deviation_pct, rolloff_length_mm, angle_deg, arc_width_deg, placement_offset_mm
    )
    p_fault = _out_of_window(deviation, spec_limit_pct, die_sigma_pct)
    params = {
        "peak_deviation_pct": peak_deviation_pct,
        "rolloff_length_mm": rolloff_length_mm,
        "angle_deg": float(angle_deg),
        "arc_width_deg": arc_width_deg,
        "placement_offset_mm": placement_offset_mm,
        "spec_limit_pct": spec_limit_pct,
        "die_sigma_pct": die_sigma_pct,
        "seed": seed,
    }
    return _finish("Edge-Loc", grid, p_fault, background_d0_per_cm2, die_rng, params)


# handling or polishing scratch
# a scratch is a narrow track cut by something dragged across the surface
# a robot blade or a slipped wafer leaves a straight line
# debris trapped under a cmp pad leaves an arc because the wafer and pad both rotate
# any die whose square footprint the track crosses is at risk
# not every crossed die fails because the scratch depth varies and contact skips


def _scratch_path(
    start_x_mm: float,
    start_y_mm: float,
    heading_deg: float,
    length_mm: float,
    arc_radius_mm: float | None,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.arange(0.0, length_mm + step_mm, step_mm)
    s = np.minimum(s, length_mm)
    h = math.radians(heading_deg)
    if arc_radius_mm is None:
        return start_x_mm + s * math.cos(h), start_y_mm + s * math.sin(h)
    # the arc bends to the left of the heading
    # its center is one radius away at a right angle to the heading
    cx = start_x_mm - arc_radius_mm * math.sin(h)
    cy = start_y_mm + arc_radius_mm * math.cos(h)
    phi = h - math.pi / 2 + s / arc_radius_mm
    return cx + arc_radius_mm * np.cos(phi), cy + arc_radius_mm * np.sin(phi)


def scratch_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    length_mm: float = 140.0,
    arc_radius_mm: float | None = 220.0,
    scratch_width_mm: float = 0.5,
    kill_probability: float = 0.85,
    start_x_mm: float | None = None,
    start_y_mm: float | None = None,
    heading_deg: float | None = None,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """scratch pattern from a track dragged across the wafer
    arc_radius_mm of none gives a straight handling scratch
    start and heading are drawn from the seed when not given"""
    _check_positive(length_mm=length_mm, scratch_width_mm=scratch_width_mm)
    _check_fraction(kill_probability=kill_probability)
    _check_nonnegative(background_d0_per_cm2=background_d0_per_cm2)
    if arc_radius_mm is not None:
        _check_positive(arc_radius_mm=arc_radius_mm)
    layout_rng, die_rng = _streams(seed)
    # the layout draws happen in a fixed order even when values are given
    # so changing one geometry input does not shift the others
    r0, a0, h0 = (
        layout_rng.uniform(0.0, 0.7),
        layout_rng.uniform(0.0, 360.0),
        layout_rng.uniform(0.0, 360.0),
    )
    if start_x_mm is None or start_y_mm is None:
        # sqrt keeps start points spread evenly by area not bunched at the center
        rad = grid.radius_mm * math.sqrt(r0)
        start_x_mm = rad * math.cos(math.radians(a0))
        start_y_mm = rad * math.sin(math.radians(a0))
    if heading_deg is None:
        heading_deg = float(h0)
    step_mm = min(scratch_width_mm, grid.pitch_x_mm / 10, grid.pitch_y_mm / 10)
    px, py = _scratch_path(
        float(start_x_mm), float(start_y_mm), heading_deg, length_mm, arc_radius_mm, step_mm
    )
    x, y = grid.die_centers_mm()
    # the die footprint is a rectangle so a track point hits a die when it is inside on both axes
    reach_x = grid.pitch_x_mm / 2 + scratch_width_mm / 2
    reach_y = grid.pitch_y_mm / 2 + scratch_width_mm / 2
    inside = (np.abs(x[..., None] - px) <= reach_x) & (np.abs(y[..., None] - py) <= reach_y)
    crossed = inside.any(axis=-1)
    p_fault = np.where(crossed, kill_probability, 0.0)
    params = {
        "length_mm": length_mm,
        "arc_radius_mm": arc_radius_mm,
        "scratch_width_mm": scratch_width_mm,
        "kill_probability": kill_probability,
        "start_x_mm": float(start_x_mm),
        "start_y_mm": float(start_y_mm),
        "heading_deg": float(heading_deg),
        "seed": seed,
    }
    return _finish("Scratch", grid, p_fault, background_d0_per_cm2, die_rng, params)


# local particle cluster
# a flake off a chamber wall or a burst of particles lands as a cloud not as single points
# the cloud is a 2d gaussian of particle density around one spot
# only some particles kill a die because a particle has to land on a critical feature
# d27 kill_ratio is that fraction which yield engineers measure by overlaying inspection maps on bin maps
# each die gets the expected number of killer particles inside its square and poisson turns that into a fail chance


def loc_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    particle_count: float = 150.0,
    cluster_sigma_mm: float = 12.0,
    kill_ratio: float = 0.4,
    center_x_mm: float | None = None,
    center_y_mm: float | None = None,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """loc pattern from one particle cluster
    particle_count is the mean number of particles in the whole cloud
    the cluster center is drawn from the seed when not given"""
    _check_positive(cluster_sigma_mm=cluster_sigma_mm)
    _check_nonnegative(particle_count=particle_count, background_d0_per_cm2=background_d0_per_cm2)
    _check_fraction(kill_ratio=kill_ratio)
    layout_rng, die_rng = _streams(seed)
    r0, a0 = layout_rng.uniform(0.0, 0.55), layout_rng.uniform(0.0, 360.0)
    if center_x_mm is None or center_y_mm is None:
        # kept off the outer edge so the result reads as loc and not edge loc
        rad = grid.radius_mm * math.sqrt(r0)
        center_x_mm = rad * math.cos(math.radians(a0))
        center_y_mm = rad * math.sin(math.radians(a0))
    x, y = grid.die_centers_mm()
    hx, hy = grid.pitch_x_mm / 2, grid.pitch_y_mm / 2
    s = cluster_sigma_mm
    # share of the gaussian cloud that falls inside each die rectangle
    # x and y are independent so the share is the product of two one dimensional pieces
    share_x = _norm_cdf((x + hx - center_x_mm) / s) - _norm_cdf((x - hx - center_x_mm) / s)
    share_y = _norm_cdf((y + hy - center_y_mm) / s) - _norm_cdf((y - hy - center_y_mm) / s)
    expected_killers = particle_count * kill_ratio * share_x * share_y
    p_fault = _poisson_kill(expected_killers)
    params = {
        "particle_count": particle_count,
        "cluster_sigma_mm": cluster_sigma_mm,
        "kill_ratio": kill_ratio,
        "center_x_mm": float(center_x_mm),
        "center_y_mm": float(center_y_mm),
        "seed": seed,
    }
    return _finish("Loc", grid, p_fault, background_d0_per_cm2, die_rng, params)


# random background defects
# killer defects that land with no spatial preference follow a poisson process
# die yield is then exp of minus die area times defect density which is the classic poisson yield model
# d28 real fabs see defect density change from wafer to wafer so an optional gamma spread is added
# averaging poisson over a gamma spread gives the negative binomial yield model with cluster parameter alpha
# a small alpha means density swings a lot between wafers and a large alpha goes back to plain poisson


def random_defects(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    d0_per_cm2: float = 0.6,
    cluster_alpha: float | None = None,
    seed: int,
) -> SimulatedWafer:
    """random pattern from spatially uniform killer defects
    d0_per_cm2 is the mean killer defect density
    cluster_alpha turns on wafer to wafer density spread"""
    _check_nonnegative(d0_per_cm2=d0_per_cm2)
    layout_rng, die_rng = _streams(seed)
    # one standard gamma draw is always taken so the die stream lines up with or without alpha
    spread = float(layout_rng.standard_gamma(cluster_alpha if cluster_alpha is not None else 1.0))
    if cluster_alpha is None:
        wafer_d0 = d0_per_cm2
    else:
        _check_positive(cluster_alpha=cluster_alpha)
        # scaling a standard gamma by mean over alpha gives a gamma with the right mean
        wafer_d0 = spread * d0_per_cm2 / cluster_alpha
    params = {
        "d0_per_cm2": d0_per_cm2,
        "cluster_alpha": cluster_alpha,
        "wafer_d0_per_cm2": wafer_d0,
        "seed": seed,
    }
    zero = np.zeros(grid.shape)
    return _finish("Random", grid, zero, wafer_d0, die_rng, params)


def no_pattern(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    d0_per_cm2: float = 0.03,
    seed: int,
) -> SimulatedWafer:
    """none pattern which is a healthy wafer with only a low random background
    the line between none and random is a density threshold not a different physics"""
    _check_nonnegative(d0_per_cm2=d0_per_cm2)
    _, die_rng = _streams(seed)
    zero = np.zeros(grid.shape)
    return _finish(
        "none", grid, zero, d0_per_cm2, die_rng, {"d0_per_cm2": d0_per_cm2, "seed": seed}
    )


# gross failure
# a misprocessed wafer like a skipped step wrong recipe or bad implant dose kills almost every die
# the failure is not spatial so every die gets the same high fail chance
# a few dies survive by chance which is what separates near full from a fully dead wafer


def near_full_fault(
    grid: WaferGrid = DEFAULT_GRID,
    *,
    kill_probability: float = 0.92,
    background_d0_per_cm2: float = 0.02,
    seed: int,
) -> SimulatedWafer:
    """near full pattern from a gross wafer level failure
    kill_probability is the fail chance every die gets"""
    _check_fraction(kill_probability=kill_probability)
    _check_nonnegative(background_d0_per_cm2=background_d0_per_cm2)
    _, die_rng = _streams(seed)
    p_fault = np.full(grid.shape, kill_probability)
    params = {"kill_probability": kill_probability, "seed": seed}
    return _finish("Near-full", grid, p_fault, background_d0_per_cm2, die_rng, params)


# lookup and mixing
# d32 keys are the exact wm811k label strings so synthetic and real maps share one label set
# d31 combine stacks faults on one wafer because real wafers often carry more than one at once


GENERATORS: dict[str, Callable[..., SimulatedWafer]] = {
    "Center": center_fault,
    "Donut": donut_fault,
    "Edge-Loc": edge_loc_fault,
    "Edge-Ring": edge_ring_fault,
    "Loc": loc_fault,
    "Near-full": near_full_fault,
    "Random": random_defects,
    "Scratch": scratch_fault,
    "none": no_pattern,
}


def generate(
    pattern: str, grid: WaferGrid = DEFAULT_GRID, *, seed: int, **params: Any
) -> SimulatedWafer:
    """build one wafer for a class label with any physical parameters overridden"""
    if pattern not in GENERATORS:
        raise ValueError(f"unknown pattern {pattern!r} expected one of {sorted(GENERATORS)}")
    return GENERATORS[pattern](grid, seed=seed, **params)


def combine(*wafers: SimulatedWafer) -> SimulatedWafer:
    """put several faults on one wafer
    a die fails if any of the faults failed it"""
    if not wafers:
        raise ValueError("combine needs at least one wafer")
    first = wafers[0]
    for w in wafers[1:]:
        if w.wafer_map.shape != first.wafer_map.shape or not np.array_equal(
            w.wafer_map == NO_DIE, first.wafer_map == NO_DIE
        ):
            raise ValueError("all wafers must share the same grid")
    # fail is 2 and pass is 1 so the max keeps any fail
    wafer_map = np.maximum.reduce([w.wafer_map for w in wafers])
    survive = np.prod([1.0 - w.fail_probability for w in wafers], axis=0)
    pattern = "+".join(w.pattern for w in wafers)
    params = {"pattern": pattern, "components": [w.params for w in wafers]}
    return SimulatedWafer(pattern, wafer_map, 1.0 - survive, params)
