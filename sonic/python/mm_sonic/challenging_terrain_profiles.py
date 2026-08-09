"""Deterministic HCT-style height fields for fixed-terrain kinematic pilots.

These profiles mirror the useful geometry families from the Humanoid
Challenging Terrain benchmark: coarse random roughness, a rough slope, and
discrete obstacles.  They are target terrains only; no pose is authored from
their geometry, so a motion is admitted only after the independent rigid-sole
foothold planner, G1 IK, contact audit, and exact collision audit succeed.
"""

from __future__ import annotations

import numpy as np

from .canonical_terrain_matcher import RegularGridHeightField


PROFILES = (
    "hct_rough_flat",
    "hct_rough_slope",
    "hct_discrete_blocks",
)


def _smootherstep(value: object) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def _coarse_random_height(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    coarse_spacing_m: float = 0.20,
    maximum_height_m: float = 0.05,
    height_quantum_m: float = 0.005,
) -> np.ndarray:
    """Interpolate deterministic, quantized 20-cm terrain samples.

    This follows the important geometry of the benchmark's
    ``random_uniform_terrain`` setup (five-centimetre range, five-millimetre
    levels, 20-cm downsample scale) without depending on Isaac Gym.
    """

    coarse_x = np.arange(
        float(x[0]) - coarse_spacing_m,
        float(x[-1]) + 1.5 * coarse_spacing_m,
        coarse_spacing_m,
    )
    coarse_y = np.arange(
        float(y[0]) - coarse_spacing_m,
        float(y[-1]) + 1.5 * coarse_spacing_m,
        coarse_spacing_m,
    )
    rng = np.random.default_rng(int(seed))
    levels = int(round(maximum_height_m / height_quantum_m))
    coarse = rng.integers(
        -levels,
        levels + 1,
        size=(len(coarse_y), len(coarse_x)),
    ).astype(np.float64)
    coarse *= float(height_quantum_m)
    along_x = np.asarray(
        [np.interp(x, coarse_x, row) for row in coarse], dtype=np.float64
    )
    return np.asarray(
        [np.interp(y, coarse_y, along_x[:, column]) for column in range(len(x))],
        dtype=np.float64,
    ).T


def build_profile(
    profile: str,
    *,
    spacing_m: float = 0.04,
    x_range_m: tuple[float, float] = (-0.8, 4.0),
    y_range_m: tuple[float, float] = (-1.2, 1.2),
    active_range_m: tuple[float, float] = (0.30, 3.45),
) -> tuple[RegularGridHeightField, dict[str, object]]:
    """Build one fixed, flat-entry/exit challenging-terrain target."""

    name = str(profile)
    if name not in PROFILES:
        raise ValueError(f"unknown challenging terrain profile {name!r}")
    spacing = float(spacing_m)
    x_min, x_max = map(float, x_range_m)
    y_min, y_max = map(float, y_range_m)
    active_start, active_stop = map(float, active_range_m)
    if (
        spacing <= 0.0
        or not x_min < active_start < active_stop < x_max
        or not y_min < y_max
        or active_stop - active_start < 1.0
    ):
        raise ValueError("challenging terrain ranges are invalid")

    x = np.arange(x_min, x_max + spacing / 2.0, spacing)
    y = np.arange(y_min, y_max + spacing / 2.0, spacing)
    grid_x, grid_y = np.meshgrid(x, y)
    active_length = active_stop - active_start
    transition_length = min(0.50, 0.18 * active_length)
    enter = _smootherstep((grid_x - active_start) / transition_length)
    leave = _smootherstep((active_stop - grid_x) / transition_length)
    window = enter * leave
    roughness = _coarse_random_height(x, y, seed=1701)

    if name == "hct_rough_flat":
        height = window * roughness
        description = (
            "HCT-style five-centimetre coarse random roughness with flat transitions"
        )
    elif name == "hct_rough_slope":
        phase = np.clip((grid_x - active_start) / active_length, 0.0, 1.0)
        # A traversable pyramid section: rise to a fourteen-centimetre crown,
        # then descend, with the benchmark-style coarse roughness on top.
        pyramid = 0.14 * (1.0 - np.abs(2.0 * phase - 1.0))
        height = window * (pyramid + roughness)
        description = (
            "fourteen-centimetre pyramid slope with HCT-style coarse roughness"
        )
    else:
        height = np.zeros_like(grid_x, dtype=np.float64)
        # Broad, axis-aligned pieces deliberately leave several valid routes.
        # Their five-to-ten-centimetre heights match the easier half of the
        # HCT discrete-obstacle curriculum while still requiring real swing
        # clearance and whole-sole landings.
        blocks = (
            (0.14, 0.30, -0.72, 0.18, 0.055),
            (0.32, 0.50, -0.08, 0.76, 0.085),
            (0.52, 0.68, -0.78, 0.05, -0.040),
            (0.70, 0.88, -0.18, 0.70, 0.100),
        )
        for start_fraction, stop_fraction, y0, y1, block_height in blocks:
            block_start = active_start + start_fraction * active_length
            block_stop = active_start + stop_fraction * active_length
            mask = (
                (grid_x >= block_start)
                & (grid_x <= block_stop)
                & (grid_y >= y0)
                & (grid_y <= y1)
            )
            height[mask] = float(block_height)
        description = "HCT-style discrete five-to-ten-centimetre rectangular blocks"

    field = RegularGridHeightField(
        height,
        spacing_m=spacing,
        origin_xy=(float(x[0]), float(y[0])),
    )
    metadata: dict[str, object] = {
        "profile": name,
        "description": description,
        "family": "humanoid_challenging_terrain",
        "spacing_m": spacing,
        "origin_xy": [float(x[0]), float(y[0])],
        "shape": list(height.shape),
        "x_range_m": [float(x[0]), float(x[-1])],
        "y_range_m": [float(y[0]), float(y[-1])],
        "active_range_m": [active_start, active_stop],
        "height_range_m": [float(np.min(height)), float(np.max(height))],
        "flat_entry_exit": True,
    }
    return field, metadata
