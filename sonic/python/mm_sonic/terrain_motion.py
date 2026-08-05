"""Robot-centric contracts for terrain-aware, contact-safe motion matching.

This module deliberately contains no renderer, simulator, or global-localizer
state.  A runtime caller supplies the stair pose and foot locations in the
robot's *current* frame (normally estimated from the current depth map).  The
database stores the corresponding quantities in each recorded root frame.
Consequently matching never requires global root position or odometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .joints import ContractError


TERRAIN_ROWS = 48
TERRAIN_COLS = 32
TERRAIN_FORWARD_RANGE_M = (-0.4, 1.9)
TERRAIN_LATERAL_RANGE_M = (-0.75, 0.75)


class MotionMode(IntEnum):
    FLAT = 0
    STAIR_APPROACH = 1
    STAIR_COMMITTED = 2
    STAIR_EXIT = 3
    SAFE_STOP = 4


class ContactPhase(IntEnum):
    FLIGHT = 0
    LEFT = 1
    RIGHT = 2
    DOUBLE = 3


def contact_phase(contact: np.ndarray | tuple[bool, bool]) -> ContactPhase:
    left, right = bool(contact[0]), bool(contact[1])
    return ContactPhase((1 if left else 0) | (2 if right else 0))


def complete_route_ready_frames(
    database: object,
    *,
    tread_rise_m: float = 0.1778,
    foot_center_to_sole_m: float = 0.035,
    maximum_sole_residual_m: float = 0.010,
) -> dict[int, int]:
    """Find complete routes with a geometrically supported landing.

    Contact annotations alone are insufficient for deciding that a stair
    traversal can hand back to flat locomotion: some recovered captures label
    double support while their soles hover centimetres above the landing.  If
    foot geometry is present, require both labelled support feet to lie within
    a small signed-height residual of their labelled tread.  Lightweight
    legacy test/database objects without geometry retain the old label-only
    behavior.
    """

    rise = float(tread_rise_m)
    sole_offset = float(foot_center_to_sole_m)
    residual_limit = float(maximum_sole_residual_m)
    if (
        not np.isfinite((rise, sole_offset, residual_limit)).all()
        or rise <= 0.0
        or sole_offset < 0.0
        or residual_limit < 0.0
    ):
        raise ContractError("route landing support thresholds are invalid")

    source_clip = np.asarray(database.source_clip)
    source_frame = np.asarray(database.source_frame)
    exit_flag = np.asarray(database.exit, dtype=bool)
    contact = np.asarray(database.contact, dtype=bool)
    geometry_available = (
        hasattr(database, "feet_xyz_stair")
        and hasattr(database, "tread_id")
    )
    if geometry_available:
        feet = np.asarray(database.feet_xyz_stair, dtype=np.float64)
        tread = np.asarray(database.tread_id, dtype=np.int64)
        if feet.shape != (len(source_clip), 2, 3):
            raise ContractError(
                "route feet_xyz_stair must have shape [N,2,3]"
            )
        if tread.shape != (len(source_clip), 2):
            raise ContractError("route tread_id must have shape [N,2]")

    ready: dict[int, int] = {}
    for clip in sorted(set(int(value) for value in source_clip)):
        rows = np.flatnonzero(source_clip == clip)
        exit_rows = rows[exit_flag[rows]]
        if exit_rows.size == 0:
            continue
        last_exit = int(np.max(source_frame[exit_rows]))
        supported_mask = (
            (source_frame[rows] > last_exit)
            & np.all(contact[rows], axis=1)
        )
        if geometry_available:
            row_tread = tread[rows]
            expected_foot_z = (
                sole_offset + rise * row_tread.astype(np.float64)
            )
            residual = np.abs(feet[rows, :, 2] - expected_foot_z)
            supported_mask &= (
                np.all(row_tread >= 0, axis=1)
                & np.all(residual <= residual_limit, axis=1)
            )
        supported = rows[supported_mask]
        if supported.size:
            ready[clip] = int(np.min(source_frame[supported]))
    return ready


def _finite(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ContractError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ContractError(f"{name} must contain finite numeric values")
    return np.ascontiguousarray(array)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    value = np.asarray(angle)
    return np.arctan2(np.sin(value), np.cos(value))


@dataclass(frozen=True)
class TerrainObservation:
    """Causal terrain patch in the robot's current heading-aligned frame."""

    height_m: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    normal_xyz: np.ndarray
    traversable: np.ndarray

    def __post_init__(self) -> None:
        shape = (TERRAIN_ROWS, TERRAIN_COLS)
        height = np.asarray(self.height_m)
        if height.shape != shape:
            raise ContractError(f"height_m must have shape {shape}")
        observed = np.asarray(self.observed, dtype=bool)
        if observed.shape != shape:
            raise ContractError(f"observed must have shape {shape}")
        if not np.isfinite(height[observed]).all():
            raise ContractError("observed height cells must be finite")
        confidence = _finite("confidence", self.confidence, shape)
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ContractError("confidence must lie in [0, 1]")
        normals = _finite("normal_xyz", self.normal_xyz, shape + (3,))
        traversable = np.asarray(self.traversable, dtype=bool)
        if traversable.shape != shape:
            raise ContractError(f"traversable must have shape {shape}")
        object.__setattr__(self, "height_m", np.ascontiguousarray(height))
        object.__setattr__(self, "observed", np.ascontiguousarray(observed))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "normal_xyz", normals)
        object.__setattr__(self, "traversable", np.ascontiguousarray(traversable))


def terrain_observation_from_height_map(
    height_m: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    maximum_tread_slope_rad: float = np.deg2rad(35.0),
) -> TerrainObservation:
    """Convert a raw NaN-masked 48x32 ego height map into runtime channels."""
    height = np.asarray(height_m, dtype=np.float32)
    shape = (TERRAIN_ROWS, TERRAIN_COLS)
    if height.shape != shape:
        raise ContractError(f"height map must have shape {shape}")
    observed = np.isfinite(height)
    if not np.any(observed):
        raise ContractError("height map has no observed cells")
    conf = observed.astype(np.float32) if confidence is None else np.asarray(
        confidence, dtype=np.float32)
    if conf.shape != shape:
        raise ContractError(f"confidence must have shape {shape}")
    # Unknowns remain explicitly masked in the returned height channel.  A
    # median fill is used only for local finite-difference calculations.
    filled = np.where(observed, height, np.nanmedian(height[observed]))
    df = (TERRAIN_FORWARD_RANGE_M[1] - TERRAIN_FORWARD_RANGE_M[0]) / (TERRAIN_ROWS - 1)
    dl = (TERRAIN_LATERAL_RANGE_M[1] - TERRAIN_LATERAL_RANGE_M[0]) / (TERRAIN_COLS - 1)
    grad_f, grad_l = np.gradient(filled, df, dl)
    normal = np.stack((-grad_f, -grad_l, np.ones_like(filled)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    slope = np.arctan(np.hypot(grad_f, grad_l))
    traversable = observed & (slope <= maximum_tread_slope_rad) & (conf >= 0.5)
    return TerrainObservation(height, observed, conf, normal.astype(np.float32), traversable)


def detect_stair_geometry(
    observation: TerrainObservation,
    *,
    minimum_rise_m: float = 0.10,
    maximum_rise_m: float = 0.25,
    minimum_edge_cells: int = 8,
) -> StairGeometry | None:
    """Fit a compact staircase to a causal ego height map.

    This is the deterministic exact-Justin bootstrap detector, not the final
    learned perception model.  Horizontal tread surfaces provide the primary
    fit.  Unlike a sum of height gradients, their principal axes are not
    biased by the two long side edges of a finite staircase.  The older
    repeated-riser fit remains as a fallback when too little tread area is
    visible.
    """
    height = np.asarray(observation.height_m)
    observed = np.asarray(observation.observed)
    filled = np.where(observed, height, np.nanmedian(height[observed]))
    forward = np.linspace(*TERRAIN_FORWARD_RANGE_M, TERRAIN_ROWS)
    lateral = np.linspace(*TERRAIN_LATERAL_RANGE_M, TERRAIN_COLS)
    df, dl = float(forward[1] - forward[0]), float(lateral[1] - lateral[0])

    # Quantize the horizontal surfaces before looking at gradients.  On the
    # exact simulator bootstrap these are discrete levels; the gap threshold
    # also tolerates bounded within-tread depth noise.  Ground and the highest
    # landing are intentionally excluded from the pose fit because they may
    # extend far beyond the finite staircase and bias their centroids.
    sorted_height = np.sort(filled[observed])
    level_splits = (
        np.flatnonzero(np.diff(sorted_height) > minimum_rise_m * 0.5)
        + 1
    )
    level_centers = np.asarray(
        [
            np.median(part)
            for part in np.split(sorted_height, level_splits)
            if len(part) >= minimum_edge_cells
        ],
        dtype=np.float64,
    )
    level_rises = np.diff(level_centers)
    valid_level_rises = level_rises[
        (level_rises >= minimum_rise_m)
        & (level_rises <= maximum_rise_m)
    ]
    ff, ll = np.meshgrid(forward, lateral, indexing="ij")
    grid_points = np.stack((ff, ll), axis=-1)
    if len(level_centers) >= 3 and len(valid_level_rises) >= 2:
        nearest_level = np.argmin(
            np.abs(filled[..., None] - level_centers), axis=-1
        )
        tread_surfaces: list[
            tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        # With four or more levels the highest surface is the landing/top
        # tread and can be much larger than the staircase.  If only ground
        # plus two treads are visible, however, that last observed level is
        # still needed to orient the lower tread.
        surface_stop = (
            len(level_centers)
            if len(level_centers) == 3
            else len(level_centers) - 1
        )
        for level_index in range(1, surface_stop):
            mask = (
                observed
                & (observation.confidence >= 0.5)
                & (nearest_level == level_index)
            )
            points = grid_points[mask]
            if len(points) < minimum_edge_cells:
                continue
            centroid = np.mean(points, axis=0)
            covariance = np.cov((points - centroid).T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if (
                eigenvalues.shape != (2,)
                or not np.isfinite(eigenvalues).all()
                or eigenvalues[0] <= 0.0
            ):
                continue
            tread_surfaces.append(
                (
                    level_index,
                    points,
                    centroid,
                    eigenvalues,
                    eigenvectors,
                )
            )
        if len(tread_surfaces) >= 2:
            ordered = sorted(tread_surfaces, key=lambda item: item[0])
            direction_hint = np.zeros(2, dtype=np.float64)
            for lower, upper in zip(ordered[:-1], ordered[1:]):
                level_delta = upper[0] - lower[0]
                if level_delta > 0:
                    direction_hint += (
                        upper[2] - lower[2]
                    ) / float(level_delta)
            hint_norm = float(np.linalg.norm(direction_hint))
            if hint_norm > 1.0e-6:
                direction_hint /= hint_norm
                # Prefer the least-truncated tread.  At an oblique edge of
                # the sensor frustum, a clipped upper tread can have a badly
                # shifted centroid while the lower tread remains complete.
                best_surface = max(
                    tread_surfaces, key=lambda item: len(item[1])
                )
                (
                    level_index,
                    points,
                    centroid,
                    eigenvalues,
                    eigenvectors,
                ) = best_surface
                axis_index = int(
                    np.argmax(
                        np.abs(eigenvectors.T @ direction_hint)
                    )
                )
                ascent = eigenvectors[:, axis_index].astype(
                    np.float64, copy=True
                )
                if float(ascent @ direction_hint) < 0.0:
                    ascent *= -1.0
                side = np.asarray((-ascent[1], ascent[0]))
                run = float(
                    np.sqrt(12.0 * eigenvalues[axis_index])
                )
                side_index = 1 - axis_index
                width = float(
                    np.sqrt(12.0 * eigenvalues[side_index])
                )
                if 0.15 <= run <= 0.6 and width >= 0.1:
                    origin = (
                        centroid
                        - (float(level_index) - 0.5)
                        * run
                        * ascent
                    )
                    rise = float(np.median(valid_level_rises))
                    tread_count = int(len(valid_level_rises))
                    repetition = min(
                        1.0, len(valid_level_rises) / 3.0
                    )
                    expected_cell_count = max(
                        1.0, run * width / (df * dl)
                    )
                    coverage = min(
                        1.0, len(points) / expected_cell_count
                    )
                    confidence = float(
                        0.5 * repetition + 0.5 * coverage
                    )
                    return StairGeometry(
                        run,
                        rise,
                        width,
                        tread_count,
                        origin.astype(np.float32),
                        float(np.arctan2(ascent[1], ascent[0])),
                        confidence,
                    )

    grad_f, grad_l = np.gradient(filled, df, dl)
    # A riser is steep relative to a tread. Threshold scales with the smallest
    # acceptable physical rise rather than a scene-specific pixel value.
    magnitude = np.hypot(grad_f, grad_l)
    edge = observed & (observation.confidence >= 0.5) & (
        magnitude >= minimum_rise_m / (2.5 * max(df, dl)))
    if np.count_nonzero(edge) < minimum_edge_cells:
        return None
    gradients = np.stack((grad_f[edge], grad_l[edge]), axis=1)
    ascent = np.sum(gradients / np.maximum(
        np.linalg.norm(gradients, axis=1, keepdims=True), 1e-8), axis=0)
    ascent_norm = float(np.linalg.norm(ascent))
    if ascent_norm < 1e-5:
        return None
    ascent /= ascent_norm
    # A finite staircase also produces strong gradients at its two side edges
    # and at the edge of its landing.  Those edges form a continuous band in
    # the ascent coordinate and would merge the separate risers into one
    # cluster.  Retain only gradients that agree with the fitted ascent axis
    # before estimating run/width.
    unit_gradients = gradients / np.maximum(
        np.linalg.norm(gradients, axis=1, keepdims=True), 1e-8
    )
    aligned_edge = (unit_gradients @ ascent) >= np.cos(np.deg2rad(35.0))
    if np.count_nonzero(aligned_edge) < minimum_edge_cells:
        return None
    side = np.asarray((-ascent[1], ascent[0]))
    points = np.stack((ff[edge], ll[edge]), axis=1)[aligned_edge]
    along = np.sort(points @ ascent)
    # Each thick rasterized riser creates a small bundle; split only at gaps
    # large enough to represent distinct treads.
    split = np.flatnonzero(np.diff(along) > 0.12) + 1
    clusters = [part for part in np.split(along, split) if len(part) >= 2]
    if len(clusters) < 2:
        return None
    centers = np.asarray([np.median(part) for part in clusters])
    runs = np.diff(centers)
    runs = runs[(runs >= 0.15) & (runs <= 0.6)]
    if runs.size == 0:
        return None
    run = float(np.median(runs))
    rises = np.diff(level_centers)
    rises = rises[(rises >= minimum_rise_m) & (rises <= maximum_rise_m)]
    if rises.size == 0:
        return None
    rise = float(np.median(rises))
    # Riser edge clusters are sensitive to view truncation and may include a
    # landing boundary. Distinct physical height levels directly determine
    # how many upward steps are present in the observed staircase.
    tread_count = int(rises.size)
    side_coordinate = points @ side
    width = float(np.percentile(side_coordinate, 95) - np.percentile(side_coordinate, 5))
    origin = ascent * centers[0] + side * float(np.median(side_coordinate))
    repetition = min(1.0, (len(centers) - 1) / 3.0)
    coverage = min(1.0, np.count_nonzero(edge) / 40.0)
    confidence = float(0.5 * repetition + 0.5 * coverage)
    return StairGeometry(run, rise, max(width, 0.1), tread_count,
                         origin.astype(np.float32),
                         float(np.arctan2(ascent[1], ascent[0])), confidence)


@dataclass(frozen=True)
class StairGeometry:
    """Canonical staircase geometry and its current pose in the robot frame."""

    run_m: float
    rise_m: float
    width_m: float
    tread_count: int
    origin_robot_xy: np.ndarray
    ascent_yaw_robot: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.run_m <= 0 or self.rise_m <= 0 or self.width_m <= 0:
            raise ContractError("stair run, rise, and width must be positive")
        if self.tread_count < 1:
            raise ContractError("stair tread_count must be positive")
        origin = _finite("origin_robot_xy", self.origin_robot_xy, (2,))
        if not np.isfinite(self.ascent_yaw_robot):
            raise ContractError("ascent_yaw_robot must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("stair confidence must lie in [0, 1]")
        object.__setattr__(self, "origin_robot_xy", origin.astype(np.float32))


@dataclass(frozen=True)
class TerrainFrameDatabase:
    """One row per searchable clean kinematic frame."""

    feature: np.ndarray
    feature_scale: np.ndarray
    source_clip: np.ndarray
    source_frame: np.ndarray
    stair_origin_root_xy: np.ndarray
    stair_ascent_yaw_root: np.ndarray
    root_xy_stair: np.ndarray
    root_height_above_stair_base_m: np.ndarray
    contact: np.ndarray
    tread_id: np.ndarray
    feet_xyz_stair: np.ndarray
    future_root_xy: np.ndarray
    future_facing_xy: np.ndarray
    future_feet_xyz_stair: np.ndarray
    future_contact: np.ndarray
    future_tread_id: np.ndarray
    inferred_travel_velocity_local_xy: np.ndarray
    inferred_facing_local_xy: np.ndarray
    entry: np.ndarray
    exit: np.ndarray

    def __post_init__(self) -> None:
        feature = np.asarray(self.feature, dtype=np.float32)
        if feature.ndim != 2 or feature.shape[0] == 0:
            raise ContractError("feature must be a non-empty [N,D] array")
        n, d = feature.shape
        scale = _finite("feature_scale", self.feature_scale, (d,)).astype(np.float32)
        if np.any(scale <= 0):
            raise ContractError("feature_scale must be positive")
        expected = {
            "source_clip": (n,), "source_frame": (n,),
            "stair_origin_root_xy": (n, 2), "stair_ascent_yaw_root": (n,),
            "root_xy_stair": (n, 2), "root_height_above_stair_base_m": (n,),
            "contact": (n, 2),
            "tread_id": (n, 2), "feet_xyz_stair": (n, 2, 3),
            "inferred_travel_velocity_local_xy": (n, 2),
            "inferred_facing_local_xy": (n, 2),
            "entry": (n,), "exit": (n,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ContractError(f"{name} must have shape {shape}, got {value.shape}")
        future_root = np.asarray(self.future_root_xy, dtype=np.float32)
        future_facing = np.asarray(self.future_facing_xy, dtype=np.float32)
        if (future_root.ndim != 3 or future_root.shape[0] != n or
                future_root.shape[2] != 2 or future_facing.shape != future_root.shape):
            raise ContractError("future trajectories must both have shape [N,K,2]")
        k = future_root.shape[1]
        future_feet = np.asarray(self.future_feet_xyz_stair)
        future_contact = np.asarray(self.future_contact, dtype=bool)
        future_tread = np.asarray(self.future_tread_id)
        if future_feet.shape != (n, k, 2, 3):
            raise ContractError("future_feet_xyz_stair must have shape [N,K,2,3]")
        if future_contact.shape != (n, k, 2) or future_tread.shape != (n, k, 2):
            raise ContractError("future contact/tread arrays must have shape [N,K,2]")
        if np.any(future_tread[~future_contact] != -1):
            raise ContractError("a non-contacting future foot must have tread_id -1")
        numeric = (feature, future_root, future_facing,
                   np.asarray(self.stair_origin_root_xy),
                   np.asarray(self.stair_ascent_yaw_root),
                   np.asarray(self.root_xy_stair),
                   np.asarray(self.root_height_above_stair_base_m),
                   np.asarray(self.feet_xyz_stair), future_feet,
                   np.asarray(self.inferred_travel_velocity_local_xy),
                   np.asarray(self.inferred_facing_local_xy))
        if not all(np.isfinite(value).all() for value in numeric):
            raise ContractError("terrain database contains non-finite values")
        contacts = np.asarray(self.contact, dtype=bool)
        tread = np.asarray(self.tread_id, dtype=np.int16)
        if np.any(tread[~contacts] != -1):
            raise ContractError("a non-contacting foot must have tread_id -1")
        facing_norm = np.linalg.norm(future_facing, axis=-1)
        if np.max(np.abs(facing_norm - 1.0)) > 1e-3:
            raise ContractError("future_facing_xy must contain unit vectors")
        inferred_facing_norm = np.linalg.norm(self.inferred_facing_local_xy, axis=-1)
        if np.max(np.abs(inferred_facing_norm - 1.0)) > 1e-3:
            raise ContractError("inferred_facing_local_xy must contain unit vectors")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, name, np.ascontiguousarray(value))

    @property
    def row_count(self) -> int:
        return int(self.feature.shape[0])


@dataclass(frozen=True)
class TerrainMatchQuery:
    feature: np.ndarray
    future_root_xy: np.ndarray
    future_facing_xy: np.ndarray
    stair: StairGeometry
    contact: np.ndarray
    tread_id: np.ndarray
    feet_xyz_stair: np.ndarray
    mode: MotionMode
    root_height_above_stair_base_m: float | None = None
    active_clip: int = -1
    active_frame: int = -1


@dataclass(frozen=True)
class TerrainMatcherConfig:
    feature_weight: float = 1.0
    trajectory_weight: float = 2.0
    facing_weight: float = 1.0
    stair_pose_weight: float = 8.0
    planted_foot_weight: float = 20.0
    continuation_bonus: float = 0.5
    maximum_stair_origin_error_m: float = 0.18
    maximum_stair_yaw_error_rad: float = np.deg2rad(15.0)
    maximum_root_height_error_m: float = 0.04
    maximum_planted_foot_error_m: float = 0.035
    minimum_stair_confidence: float = 0.65


@dataclass(frozen=True)
class TerrainMatch:
    accepted: bool
    row: int
    source_clip: int
    source_frame: int
    cost: float
    reason: str
    candidate_count: int


@dataclass(frozen=True)
class TerrainSafetyReport:
    maximum_forbidden_penetration_m: float
    maximum_planted_foot_drift_m: float
    ik_feasible: bool
    collision_free: bool


@dataclass(frozen=True)
class PreparedTerrainTransition:
    token: int
    match: TerrainMatch
    prior_clip: int
    prior_frame: int


@dataclass(frozen=True)
class TerrainControllerState:
    active_clip: int = -1
    active_frame: int = -1
    committed_transitions: int = 0
    rejected_transitions: int = 0


class TerrainMotionMatcher:
    """Exact matcher with fixed-world and contact-topology hard gates."""

    def __init__(self, database: TerrainFrameDatabase,
                 config: TerrainMatcherConfig = TerrainMatcherConfig()) -> None:
        self.database = database
        self.config = config

    def select(self, query: TerrainMatchQuery) -> TerrainMatch:
        db = self.database
        cfg = self.config
        feature = _finite("query feature", query.feature, (db.feature.shape[1],))
        future_root = _finite("query future_root_xy", query.future_root_xy,
                              db.future_root_xy.shape[1:])
        future_facing = _finite("query future_facing_xy", query.future_facing_xy,
                                db.future_facing_xy.shape[1:])
        contact = np.asarray(query.contact, dtype=bool)
        tread = np.asarray(query.tread_id, dtype=np.int16)
        feet = _finite("query feet_xyz_stair", query.feet_xyz_stair, (2, 3))
        if contact.shape != (2,) or tread.shape != (2,):
            raise ContractError("query contact and tread_id must have shape [2]")
        if np.any(tread[~contact] != -1):
            raise ContractError("a non-contacting query foot must have tread_id -1")
        query_root_height = query.root_height_above_stair_base_m
        if query_root_height is not None:
            query_root_height = float(query_root_height)
            if not np.isfinite(query_root_height):
                raise ContractError(
                    "query root_height_above_stair_base_m must be finite"
                )
        if query.stair.confidence < cfg.minimum_stair_confidence:
            return self._reject("stair confidence below safe threshold")

        keep = np.ones(db.row_count, dtype=bool)
        if query.mode == MotionMode.STAIR_APPROACH:
            keep &= np.asarray(db.entry, dtype=bool)
        elif query.mode == MotionMode.STAIR_COMMITTED:
            # Matching in flight is too weakly constrained.  Continue the
            # current source instead; reconsider only at a supported frame.
            if contact_phase(contact) == ContactPhase.FLIGHT:
                return self._continuation_or_reject(query, "flight phase")
            keep &= np.all(np.asarray(db.contact, dtype=bool) == contact, axis=1)
            for foot in range(2):
                if contact[foot]:
                    keep &= db.tread_id[:, foot] == tread[foot]
        elif query.mode == MotionMode.STAIR_EXIT:
            keep &= np.asarray(db.exit, dtype=bool)
        elif query.mode in (MotionMode.FLAT, MotionMode.SAFE_STOP):
            return self._reject("terrain matcher is inactive in flat/safe-stop mode")

        # Justin stair clips are finite, non-looping traversals. A search
        # chooses a future pose: selecting the active or an earlier frame from
        # the same clip can deadlock into a two-frame end-of-clip loop.
        # Sequential continuation remains a distinct candidate at
        # active_frame + 1, while a contact-compatible different clip may still
        # be selected in either of its valid phases.
        if query.active_clip >= 0 and query.active_frame >= 0:
            keep &= ~(
                (db.source_clip == query.active_clip)
                & (db.source_frame <= query.active_frame)
            )

        # The observed staircase is fixed in the current root frame.  Candidate
        # rows describe where that same fixed staircase was in the recorded
        # root frame; this is an environment gate, not a normalized pose term.
        origin_error = np.linalg.norm(
            db.stair_origin_root_xy - query.stair.origin_robot_xy[None, :], axis=1)
        yaw_error = np.abs(wrap_angle(
            db.stair_ascent_yaw_root - query.stair.ascent_yaw_robot))
        height_error = np.zeros(db.row_count, dtype=np.float32)
        if query_root_height is not None:
            height_error = np.abs(
                db.root_height_above_stair_base_m - query_root_height
            )
        keep &= origin_error <= cfg.maximum_stair_origin_error_m
        keep &= yaw_error <= cfg.maximum_stair_yaw_error_rad
        keep &= height_error <= cfg.maximum_root_height_error_m

        planted_error = np.zeros(db.row_count, dtype=np.float32)
        if query.mode == MotionMode.STAIR_COMMITTED:
            for foot in range(2):
                if contact[foot]:
                    error = np.linalg.norm(
                        db.feet_xyz_stair[:, foot] - feet[foot][None, :], axis=1)
                    planted_error += error.astype(np.float32)
                    keep &= error <= cfg.maximum_planted_foot_error_m

        rows = np.flatnonzero(keep)
        if rows.size == 0:
            return self._continuation_or_reject(query, "no contact-safe fixed-world candidate")
        feature_cost = np.sum(
            ((db.feature[rows] - feature) / db.feature_scale) ** 2, axis=1)
        trajectory_cost = np.mean(
            np.sum((db.future_root_xy[rows] - future_root) ** 2, axis=2), axis=1)
        facing_cost = np.mean(
            1.0 - np.sum(db.future_facing_xy[rows] * future_facing, axis=2), axis=1)
        cost = (cfg.feature_weight * feature_cost +
                cfg.trajectory_weight * trajectory_cost +
                cfg.facing_weight * facing_cost +
                cfg.stair_pose_weight * (
                    origin_error[rows] ** 2
                    + yaw_error[rows] ** 2
                    + height_error[rows] ** 2
                ) +
                cfg.planted_foot_weight * planted_error[rows] ** 2)
        continuation = ((db.source_clip[rows] == query.active_clip) &
                        (db.source_frame[rows] == query.active_frame + 1))
        cost = cost - cfg.continuation_bonus * continuation.astype(np.float32)
        chosen_at = int(np.argmin(cost))
        row = int(rows[chosen_at])
        return TerrainMatch(True, row, int(db.source_clip[row]),
                            int(db.source_frame[row]), float(cost[chosen_at]),
                            "matched", int(rows.size))

    def _continuation_or_reject(self, query: TerrainMatchQuery, reason: str) -> TerrainMatch:
        rows = np.flatnonzero(
            (self.database.source_clip == query.active_clip) &
            (self.database.source_frame == query.active_frame + 1))
        if rows.size:
            row = int(rows[0])
            return TerrainMatch(True, row, int(self.database.source_clip[row]),
                                int(self.database.source_frame[row]), 0.0,
                                f"sequential fallback: {reason}", 1)
        return self._reject(reason)

    @staticmethod
    def _reject(reason: str) -> TerrainMatch:
        return TerrainMatch(False, -1, -1, -1, float("inf"), reason, 0)


class TransactionalTerrainController:
    """Prepare/validate/commit wrapper around :class:`TerrainMotionMatcher`.

    Collision and IK are intentionally external: the exact runtime backend
    evaluates the proposed frame against its actual robot and terrain model,
    then supplies the resulting report.  Until ``commit`` succeeds the active
    source frame is unchanged.
    """

    MAXIMUM_PENETRATION_M = 0.005
    MAXIMUM_PLANTED_DRIFT_M = 0.010

    def __init__(self, matcher: TerrainMotionMatcher) -> None:
        self.matcher = matcher
        self.state = TerrainControllerState()
        self._next_token = 1
        self._pending: PreparedTerrainTransition | None = None

    def prepare(self, query: TerrainMatchQuery) -> PreparedTerrainTransition:
        if self._pending is not None:
            raise ContractError("a terrain transition is already pending")
        # The controller, not its caller, owns continuity identity.
        owned_query = TerrainMatchQuery(
            feature=query.feature, future_root_xy=query.future_root_xy,
            future_facing_xy=query.future_facing_xy, stair=query.stair,
            contact=query.contact, tread_id=query.tread_id,
            feet_xyz_stair=query.feet_xyz_stair, mode=query.mode,
            active_clip=self.state.active_clip, active_frame=self.state.active_frame,
        )
        prepared = PreparedTerrainTransition(
            token=self._next_token, match=self.matcher.select(owned_query),
            prior_clip=self.state.active_clip, prior_frame=self.state.active_frame)
        self._next_token += 1
        self._pending = prepared
        return prepared

    def commit(self, token: int, safety: TerrainSafetyReport) -> bool:
        pending = self._require_pending(token)
        safe = (
            pending.match.accepted
            and safety.ik_feasible
            and safety.collision_free
            and safety.maximum_forbidden_penetration_m <= self.MAXIMUM_PENETRATION_M
            and safety.maximum_planted_foot_drift_m <= self.MAXIMUM_PLANTED_DRIFT_M
        )
        if safe:
            self.state = TerrainControllerState(
                active_clip=pending.match.source_clip,
                active_frame=pending.match.source_frame,
                committed_transitions=self.state.committed_transitions + 1,
                rejected_transitions=self.state.rejected_transitions,
            )
        else:
            self.state = TerrainControllerState(
                active_clip=self.state.active_clip,
                active_frame=self.state.active_frame,
                committed_transitions=self.state.committed_transitions,
                rejected_transitions=self.state.rejected_transitions + 1,
            )
        self._pending = None
        return safe

    def rollback(self, token: int) -> None:
        self._require_pending(token)
        self.state = TerrainControllerState(
            active_clip=self.state.active_clip,
            active_frame=self.state.active_frame,
            committed_transitions=self.state.committed_transitions,
            rejected_transitions=self.state.rejected_transitions + 1,
        )
        self._pending = None

    def _require_pending(self, token: int) -> PreparedTerrainTransition:
        if self._pending is None or self._pending.token != token:
            raise ContractError("terrain transition token is not pending")
        return self._pending
