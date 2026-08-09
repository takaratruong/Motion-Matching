"""Conservative fixed-terrain foothold planning for rigid G1 soles.

This module is deliberately a planning layer, not a replacement for the
existing MuJoCo IK, contact, or whole-body collision audits.  It accepts the
authored alternating stance windows and a representative sole footprint for
each window, searches a bounded neighbourhood on an exact terrain mesh, and
returns one rigid transform per accepted foothold.  Every accepted candidate
has downward terrain support over a dense sampling of the complete convex sole
footprint; centre-only or toe-only support is never sufficient.

The accompanying pelvis corridor is a conservative geometric pre-check.  It
intersects the height ranges implied by every active stance and enforces a
per-frame height-rate bound.  The downstream G1 leg IK must still prove exact
reachability, and the existing contact/collision gates remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol, Sequence

import numpy as np

from mm_sonic.joints import ContractError

from .terrain_oracle.stair_foothold_anchors import (
    NominalFootSolePose,
    StanceSpan,
)


_DOWN = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
_EPSILON = 1.0e-10


class _RayHitLike(Protocol):
    position_world: np.ndarray
    normal_world: np.ndarray


class TerrainMeshIndexCompatible(Protocol):
    """Minimal terrain query consumed by :func:`plan_terrain_footholds`."""

    vertices_world: np.ndarray

    def raycast(self, origin: object, direction: object) -> _RayHitLike | None:
        ...


def _readonly(value: object, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    result.flags.writeable = False
    return result


def _finite_vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{name} must be finite with shape {shape}")
    return array


@dataclass(frozen=True)
class FootholdIntent:
    """One authored planted interval and its nominal rigid sole footprint.

    ``nominal_pose.sole_support_points_world`` must describe the boundary of
    the contact footprint (or a conservative set whose convex hull is that
    boundary), at the sole surface rather than at ankle height.  The G1
    integration should use ``_G1FootfallAdapter.sole_support_points_for_pose``
    or subtract the exact sole-sphere radii from sphere centres.
    """

    span: StanceSpan
    nominal_pose: NominalFootSolePose

    def __post_init__(self) -> None:
        if not isinstance(self.span, StanceSpan):
            raise ContractError("span must be StanceSpan")
        if not isinstance(self.nominal_pose, NominalFootSolePose):
            raise ContractError("nominal_pose must be NominalFootSolePose")


@dataclass(frozen=True)
class TerrainFootholdPlannerConfig:
    """Bounded search, exact support, and pelvis-corridor thresholds."""

    maximum_longitudinal_adjustment_m: float = 0.12
    maximum_lateral_adjustment_m: float = 0.12
    maximum_yaw_adjustment_rad: float = math.radians(15.0)
    maximum_vertical_adjustment_m: float = 0.30
    longitudinal_samples: int = 5
    lateral_samples: int = 5
    yaw_samples: int = 5
    footprint_sample_spacing_m: float = 0.025
    maximum_nominal_plane_residual_m: float = 0.004
    maximum_surface_plane_residual_m: float = 0.004
    maximum_surface_normal_spread_rad: float = math.radians(10.0)
    maximum_surface_slope_rad: float = math.radians(25.0)
    support_height_tolerance_m: float = 0.006
    collision_clearance_m: float = 0.0025
    ray_origin_margin_m: float = 1.0
    pelvis_reach_slack_m: float = 0.12
    maximum_pelvis_height_adjustment_m: float = 0.35
    maximum_pelvis_height_step_m: float = 0.025
    pelvis_smoothing_weight: float = 8.0
    pelvis_smoothing_iterations: int = 100
    require_alternating_feet: bool = True
    maximum_planar_reach_change_m: float = 0.09
    maximum_yaw_change_rad: float = math.radians(7.5)
    maximum_pelvis_planar_adjustment_step_m: float = 0.015
    foothold_sequence_smoothing_weight: float = 4.0

    def __post_init__(self) -> None:
        nonnegative = (
            self.maximum_longitudinal_adjustment_m,
            self.maximum_lateral_adjustment_m,
            self.maximum_yaw_adjustment_rad,
            self.maximum_vertical_adjustment_m,
            self.maximum_nominal_plane_residual_m,
            self.maximum_surface_plane_residual_m,
            self.maximum_surface_normal_spread_rad,
            self.maximum_surface_slope_rad,
            self.support_height_tolerance_m,
            self.collision_clearance_m,
            self.pelvis_reach_slack_m,
            self.maximum_pelvis_height_adjustment_m,
            self.maximum_planar_reach_change_m,
            self.maximum_yaw_change_rad,
            self.foothold_sequence_smoothing_weight,
            self.pelvis_smoothing_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ContractError("planner limits must be finite and nonnegative")
        positive = (
            self.footprint_sample_spacing_m,
            self.ray_origin_margin_m,
            self.maximum_pelvis_height_step_m,
            self.maximum_pelvis_planar_adjustment_step_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ContractError("planner spacing and height-step limits must be positive")
        if self.maximum_surface_slope_rad >= math.pi / 2.0:
            raise ContractError("maximum surface slope must be less than pi/2")
        if self.maximum_surface_normal_spread_rad >= math.pi:
            raise ContractError("surface normal spread must be less than pi")
        if any(
            not isinstance(value, (int, np.integer)) or int(value) < 1
            for value in (
                self.longitudinal_samples,
                self.lateral_samples,
                self.yaw_samples,
                self.pelvis_smoothing_iterations,
            )
        ):
            raise ContractError("planner sample and iteration counts must be positive")
        if type(self.require_alternating_feet) is not bool:
            raise ContractError("require_alternating_feet must be bool")


@dataclass(frozen=True)
class FootholdSearchDiagnostics:
    """Auditable search outcome for one stance interval."""

    span: StanceSpan
    candidate_count: int
    accepted_candidate_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    longitudinal_adjustment_m: float | None
    lateral_adjustment_m: float | None
    yaw_adjustment_rad: float | None
    vertical_adjustment_m: float | None
    surface_slope_rad: float | None
    maximum_surface_residual_m: float | None
    maximum_normal_spread_rad: float | None


@dataclass(frozen=True)
class PlannedFoothold:
    """One accepted rigid sole placement for an entire stance span."""

    intent: FootholdIntent
    sole_center_world: np.ndarray
    sole_support_points_world: np.ndarray
    dense_support_points_world: np.ndarray
    surface_normal_world: np.ndarray
    rotation_world_from_nominal: np.ndarray
    translation_world: np.ndarray
    diagnostics: FootholdSearchDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.intent, FootholdIntent):
            raise ContractError("intent must be FootholdIntent")
        center = _finite_vector(self.sole_center_world, (3,), "sole_center_world")
        normal = _finite_vector(self.surface_normal_world, (3,), "surface_normal_world")
        rotation = _finite_vector(
            self.rotation_world_from_nominal,
            (3, 3),
            "rotation_world_from_nominal",
        )
        translation = _finite_vector(self.translation_world, (3,), "translation_world")
        support = np.asarray(self.sole_support_points_world, dtype=np.float64)
        dense = np.asarray(self.dense_support_points_world, dtype=np.float64)
        for array, name in ((support, "sole_support_points_world"), (dense, "dense_support_points_world")):
            if array.ndim != 2 or array.shape[0] < 3 or array.shape[1:] != (3,):
                raise ContractError(f"{name} must have shape [N>=3,3]")
            if not np.isfinite(array).all():
                raise ContractError(f"{name} must be finite")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
            raise ContractError("foothold rotation must be proper")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
            raise ContractError("foothold rotation must be orthonormal")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= _EPSILON:
            raise ContractError("surface normal must be nonzero")
        object.__setattr__(self, "sole_center_world", _readonly(center))
        object.__setattr__(self, "sole_support_points_world", _readonly(support))
        object.__setattr__(self, "dense_support_points_world", _readonly(dense))
        object.__setattr__(self, "surface_normal_world", _readonly(normal / normal_norm))
        object.__setattr__(self, "rotation_world_from_nominal", _readonly(rotation))
        object.__setattr__(self, "translation_world", _readonly(translation))

    def transform_points(self, points_world: object) -> np.ndarray:
        """Apply this accepted rigid edit to sphere centres or foot hulls."""

        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ContractError("points_world must be finite with trailing shape [3]")
        return points @ self.rotation_world_from_nominal.T + self.translation_world


@dataclass(frozen=True)
class PelvisHeightCorridor:
    """Per-frame feasible bounds and one smooth selected pelvis height path."""

    minimum_height_world_m: np.ndarray
    maximum_height_world_m: np.ndarray
    selected_height_world_m: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.minimum_height_world_m, dtype=np.float64)
        upper = np.asarray(self.maximum_height_world_m, dtype=np.float64)
        selected = np.asarray(self.selected_height_world_m, dtype=np.float64)
        if lower.ndim != 1 or upper.shape != lower.shape or selected.shape != lower.shape:
            raise ContractError("pelvis corridor arrays must share one-dimensional shape")
        if not np.isfinite((lower, upper, selected)).all():
            raise ContractError("pelvis corridor must be finite")
        if np.any(lower > upper + 1.0e-10):
            raise ContractError("pelvis corridor has inverted bounds")
        if np.any(selected < lower - 1.0e-9) or np.any(selected > upper + 1.0e-9):
            raise ContractError("selected pelvis height leaves its corridor")
        object.__setattr__(self, "minimum_height_world_m", _readonly(lower))
        object.__setattr__(self, "maximum_height_world_m", _readonly(upper))
        object.__setattr__(self, "selected_height_world_m", _readonly(selected))


@dataclass(frozen=True)
class TerrainFootholdPlan:
    footholds: tuple[PlannedFoothold, ...]
    pelvis_height_corridor: PelvisHeightCorridor
    pelvis_planar_offset_world_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pelvis_height_corridor, PelvisHeightCorridor):
            raise ContractError("pelvis_height_corridor must be PelvisHeightCorridor")
        frame_count = len(self.pelvis_height_corridor.selected_height_world_m)
        if self.pelvis_planar_offset_world_m is None:
            planar = np.zeros((frame_count, 2), dtype=np.float64)
        else:
            planar = np.asarray(
                self.pelvis_planar_offset_world_m, dtype=np.float64
            )
        if planar.shape != (frame_count, 2) or not np.isfinite(planar).all():
            raise ContractError(
                "pelvis_planar_offset_world_m must be finite with shape [T,2]"
            )
        object.__setattr__(
            self, "pelvis_planar_offset_world_m", _readonly(planar)
        )


@dataclass(frozen=True)
class TerrainFootholdPlanningDiagnostics:
    accepted: bool
    reason_code: str | None
    message: str
    rejected_span: StanceSpan | None
    foothold_searches: tuple[FootholdSearchDiagnostics, ...]
    rejected_frame: int | None = None
    sequence_search_stage: str | None = None
    candidate_pool_sizes: tuple[int, ...] = ()
    candidate_evaluation_count: int = 0
    candidate_cache_hit_count: int = 0
    sequence_transition_evaluation_count: int = 0
    sequence_feasible_transition_count: int = 0
    selected_sequence_cost: float | None = None
    maximum_selected_world_xy_shift_change_m: float | None = None
    maximum_selected_yaw_change_rad: float | None = None
    maximum_selected_pelvis_planar_step_m: float | None = None
    planning_runtime_seconds: float = 0.0


@dataclass(frozen=True)
class TerrainFootholdPlanningResult:
    """Either an accepted plan or a structured, expected rejection."""

    plan: TerrainFootholdPlan | None
    diagnostics: TerrainFootholdPlanningDiagnostics

    def __post_init__(self) -> None:
        if self.diagnostics.accepted != (self.plan is not None):
            raise ContractError("planning result and diagnostics disagree")

    @property
    def accepted(self) -> bool:
        return self.plan is not None


@dataclass(frozen=True)
class _Plane:
    coefficients: np.ndarray
    normal_world: np.ndarray
    maximum_residual_m: float

    def height(self, xy: np.ndarray) -> np.ndarray:
        values = np.asarray(xy, dtype=np.float64)
        return (
            values[..., 0] * self.coefficients[0]
            + values[..., 1] * self.coefficients[1]
            + self.coefficients[2]
        )


@dataclass(frozen=True)
class _Candidate:
    longitudinal_m: float
    lateral_m: float
    yaw_adjustment_rad: float
    vertical_adjustment_m: float
    slope_rad: float
    maximum_surface_residual_m: float
    maximum_normal_spread_rad: float
    center_world: np.ndarray
    support_points_world: np.ndarray
    dense_support_points_world: np.ndarray
    normal_world: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    cost: float


@dataclass
class _CandidateEvaluationMetrics:
    evaluation_count: int = 0
    cache_hit_count: int = 0
    transition_evaluation_count: int = 0
    feasible_transition_count: int = 0


@dataclass
class _PreparedFootholdSearch:
    intent: FootholdIntent
    nominal_plane: _Plane
    nominal_basis: np.ndarray
    nominal_dense_relative_xy: np.ndarray
    cache: dict[tuple[float, float, float], tuple[_Candidate | None, str]]

    def evaluate(
        self,
        terrain: TerrainMeshIndexCompatible,
        *,
        longitudinal_m: float,
        lateral_m: float,
        yaw_adjustment_rad: float,
        ray_origin_height_m: float,
        config: TerrainFootholdPlannerConfig,
        metrics: _CandidateEvaluationMetrics,
    ) -> tuple[_Candidate | None, str]:
        key = (
            float(longitudinal_m),
            float(lateral_m),
            float(yaw_adjustment_rad),
        )
        cached = self.cache.get(key)
        if cached is not None:
            metrics.cache_hit_count += 1
            return cached
        metrics.evaluation_count += 1
        result = _evaluate_candidate(
            terrain,
            self.intent,
            self.nominal_plane,
            self.nominal_basis,
            self.nominal_dense_relative_xy,
            longitudinal_m=key[0],
            lateral_m=key[1],
            yaw_adjustment_rad=key[2],
            ray_origin_height_m=ray_origin_height_m,
            config=config,
        )
        self.cache[key] = result
        return result

    def accepted_candidates(self) -> tuple[_Candidate, ...]:
        return tuple(
            sorted(
                (
                    candidate
                    for candidate, _ in self.cache.values()
                    if candidate is not None
                ),
                key=_candidate_rank,
            )
        )


def _convex_hull(points: object) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (2,) or not np.isfinite(values).all():
        raise ContractError("footprint points must be finite with shape [N,2]")
    unique = sorted({(float(point[0]), float(point[1])) for point in values})
    if len(unique) < 3:
        raise ContractError("sole footprint needs at least three distinct XY points")

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= _EPSILON:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= _EPSILON:
            upper.pop()
        upper.append(point)
    result = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(result) < 3:
        raise ContractError("sole footprint has no positive XY area")
    return result


def _inside_convex(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    cross = (
        edges[None, :, 0] * relative[:, :, 1]
        - edges[None, :, 1] * relative[:, :, 0]
    )
    return np.all(cross >= -1.0e-9, axis=1)


def _dense_polygon_samples(polygon: object, spacing_m: float) -> np.ndarray:
    """Sample vertices, edges, and interior of a convex XY footprint."""

    hull = _convex_hull(polygon)
    samples: list[np.ndarray] = [point.copy() for point in hull]
    for start, stop in zip(hull, np.roll(hull, -1, axis=0)):
        length = float(np.linalg.norm(stop - start))
        count = max(1, int(math.ceil(length / float(spacing_m))))
        for index in range(1, count):
            samples.append(start + (index / count) * (stop - start))
    lower = np.min(hull, axis=0)
    upper = np.max(hull, axis=0)
    x_count = max(2, int(math.ceil((upper[0] - lower[0]) / spacing_m)) + 1)
    y_count = max(2, int(math.ceil((upper[1] - lower[1]) / spacing_m)) + 1)
    grid_x, grid_y = np.meshgrid(
        np.linspace(lower[0], upper[0], x_count),
        np.linspace(lower[1], upper[1], y_count),
    )
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    samples.extend(grid[_inside_convex(hull, grid)])
    samples.append(np.mean(hull, axis=0))
    rounded = {
        (round(float(point[0]), 12), round(float(point[1]), 12))
        for point in samples
    }
    return np.asarray(sorted(rounded), dtype=np.float64)


def _fit_plane(points_world: object) -> _Plane:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1:] != (3,):
        raise ContractError("plane fit needs at least three 3D points")
    design = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    if rank < 3:
        # A narrow sole can be nearly collinear in XY.  SVD still gives its
        # plane, but such a footprint cannot prove area support and is rejected
        # earlier by the convex-hull check.
        raise ContractError("sole footprint cannot define an XY terrain plane")
    predicted = design @ coefficients
    normal = np.asarray((-coefficients[0], -coefficients[1], 1.0), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return _Plane(
        coefficients=_readonly(coefficients),
        normal_world=_readonly(normal),
        maximum_residual_m=float(np.max(np.abs(points[:, 2] - predicted))),
    )


def _basis_from_plane(normal_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    normal = np.array(normal_world, dtype=np.float64, copy=True)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal *= -1.0
    forward = np.asarray((math.cos(yaw_rad), math.sin(yaw_rad), 0.0), dtype=np.float64)
    forward -= float(np.dot(forward, normal)) * normal
    length = float(np.linalg.norm(forward))
    if length <= _EPSILON:
        raise ContractError("sole yaw is parallel to its plane normal")
    forward /= length
    lateral = np.cross(normal, forward)
    lateral /= np.linalg.norm(lateral)
    return np.stack((forward, lateral, normal), axis=1)


def _symmetric_samples(limit: float, count: int) -> tuple[float, ...]:
    if limit == 0.0 or count == 1:
        return (0.0,)
    values = set(float(value) for value in np.linspace(-limit, limit, count))
    values.add(0.0)
    return tuple(sorted(values, key=lambda value: (abs(value), value)))


def _terrain_raycast_height_and_normal(
    terrain: TerrainMeshIndexCompatible,
    xy: np.ndarray,
    ray_origin_height_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    hit = terrain.raycast(
        np.asarray((xy[0], xy[1], ray_origin_height_m), dtype=np.float64),
        _DOWN,
    )
    if hit is None:
        return None
    position = np.asarray(hit.position_world, dtype=np.float64)
    normal = np.asarray(hit.normal_world, dtype=np.float64)
    if position.shape != (3,) or normal.shape != (3,) or not np.isfinite((position, normal)).all():
        raise ContractError("terrain raycast returned malformed hit data")
    length = float(np.linalg.norm(normal))
    if length <= _EPSILON:
        raise ContractError("terrain raycast returned zero normal")
    normal = normal / length
    if normal[2] < 0.0:
        normal *= -1.0
    return position, normal


def _surface_samples(
    terrain: TerrainMeshIndexCompatible,
    sample_xy: np.ndarray,
    ray_origin_height_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    for xy in sample_xy:
        hit = _terrain_raycast_height_and_normal(terrain, xy, ray_origin_height_m)
        if hit is None:
            return None
        positions.append(hit[0])
        normals.append(hit[1])
    return np.asarray(positions), np.asarray(normals)


def _normal_spread(normals: np.ndarray, reference: np.ndarray) -> float:
    dot = np.clip(normals @ reference, -1.0, 1.0)
    return float(np.max(np.arccos(dot)))


def _candidate_rank(candidate: _Candidate) -> tuple[float, ...]:
    return (
        candidate.cost,
        abs(candidate.longitudinal_m),
        abs(candidate.lateral_m),
        abs(candidate.yaw_adjustment_rad),
        candidate.longitudinal_m,
        candidate.lateral_m,
        candidate.yaw_adjustment_rad,
    )


def _normalized_square(value: float, limit: float) -> float:
    return 0.0 if limit == 0.0 else float((value / limit) ** 2)


def _evaluate_candidate(
    terrain: TerrainMeshIndexCompatible,
    intent: FootholdIntent,
    nominal_plane: _Plane,
    nominal_basis: np.ndarray,
    nominal_dense_relative_xy: np.ndarray,
    *,
    longitudinal_m: float,
    lateral_m: float,
    yaw_adjustment_rad: float,
    ray_origin_height_m: float,
    config: TerrainFootholdPlannerConfig,
) -> tuple[_Candidate | None, str]:
    pose = intent.nominal_pose
    center = np.asarray(pose.sole_center_world, dtype=np.float64)
    forward = np.asarray((math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)), dtype=np.float64)
    lateral_axis = np.asarray((-forward[1], forward[0]), dtype=np.float64)
    candidate_center_xy = (
        center[:2]
        + float(longitudinal_m) * forward
        + float(lateral_m) * lateral_axis
    )
    cosine = math.cos(yaw_adjustment_rad)
    sine = math.sin(yaw_adjustment_rad)
    yaw_rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    provisional_xy = candidate_center_xy + nominal_dense_relative_xy @ yaw_rotation.T
    sampled = _surface_samples(terrain, provisional_xy, ray_origin_height_m)
    if sampled is None:
        return None, "missing_surface_support"
    surface_points, surface_normals = sampled
    surface_plane = _fit_plane(surface_points)
    slope = math.acos(float(np.clip(surface_plane.normal_world[2], -1.0, 1.0)))
    if slope > config.maximum_surface_slope_rad:
        return None, "surface_too_steep"
    normal_spread = _normal_spread(surface_normals, surface_plane.normal_world)
    if normal_spread > config.maximum_surface_normal_spread_rad:
        return None, "surface_normal_discontinuity"
    if surface_plane.maximum_residual_m > config.maximum_surface_plane_residual_m:
        return None, "surface_not_planar"

    target_basis = _basis_from_plane(
        surface_plane.normal_world,
        pose.yaw_rad + yaw_adjustment_rad,
    )
    rotation = target_basis @ nominal_basis.T
    support = np.asarray(pose.sole_support_points_world, dtype=np.float64)
    relative = support - center
    rotated_relative = relative @ rotation.T
    mean_relative = np.mean(rotated_relative, axis=0)
    mean_xy = candidate_center_xy + mean_relative[:2]
    center_z = float(surface_plane.height(mean_xy) - mean_relative[2])
    target_center = np.asarray(
        (candidate_center_xy[0], candidate_center_xy[1], center_z),
        dtype=np.float64,
    )
    translation = target_center - center @ rotation.T
    target_support = support @ rotation.T + translation
    vertical_adjustment = float(np.mean(target_support[:, 2] - support[:, 2]))
    if abs(vertical_adjustment) > config.maximum_vertical_adjustment_m:
        return None, "vertical_adjustment_exceeds_bound"

    target_dense_xy = _dense_polygon_samples(
        target_support[:, :2], config.footprint_sample_spacing_m
    )
    target_dense = np.column_stack(
        (target_dense_xy, surface_plane.height(target_dense_xy))
    )
    # Validate both the interpolated footprint and the actual rigidly moved
    # sole probes.  The former prevents an unsupported interior/edge; the
    # latter prevents a slightly non-coplanar authored probe from penetrating
    # even when the best-fit plane itself is valid.
    validation_target = np.concatenate((target_dense, target_support), axis=0)
    final_samples = _surface_samples(
        terrain, validation_target[:, :2], ray_origin_height_m
    )
    if final_samples is None:
        return None, "missing_surface_support"
    final_surface, final_normals = final_samples
    support_gap = validation_target[:, 2] - final_surface[:, 2]
    penetration = final_surface[:, 2] - validation_target[:, 2]
    if float(np.max(penetration)) > config.collision_clearance_m:
        return None, "sole_penetration"
    if float(np.max(support_gap)) > config.support_height_tolerance_m:
        return None, "unsupported_footprint"
    final_normal_spread = _normal_spread(final_normals, surface_plane.normal_world)
    if final_normal_spread > config.maximum_surface_normal_spread_rad:
        return None, "surface_normal_discontinuity"
    final_residual = float(
        np.max(np.abs(final_surface[:, 2] - validation_target[:, 2]))
    )
    if final_residual > config.maximum_surface_plane_residual_m:
        return None, "surface_not_planar"
    if pose.collision_envelope_points_world is not None:
        envelope = (
            np.asarray(pose.collision_envelope_points_world, dtype=np.float64)
            @ rotation.T
            + translation
        )
        for point in envelope:
            hit = _terrain_raycast_height_and_normal(
                terrain, point[:2], ray_origin_height_m
            )
            if (
                hit is not None
                and float(hit[0][2])
                > float(point[2]) + config.collision_clearance_m
            ):
                return None, "foot_envelope_collision"

    cost = (
        _normalized_square(
            longitudinal_m, config.maximum_longitudinal_adjustment_m
        )
        + _normalized_square(lateral_m, config.maximum_lateral_adjustment_m)
        + _normalized_square(
            yaw_adjustment_rad, config.maximum_yaw_adjustment_rad
        )
        + _normalized_square(
            vertical_adjustment, config.maximum_vertical_adjustment_m
        )
        + (final_residual / max(config.maximum_surface_plane_residual_m, 1.0e-9)) ** 2
        + (slope / max(config.maximum_surface_slope_rad, 1.0e-9)) ** 2
    )
    return (
        _Candidate(
            longitudinal_m=float(longitudinal_m),
            lateral_m=float(lateral_m),
            yaw_adjustment_rad=float(yaw_adjustment_rad),
            vertical_adjustment_m=vertical_adjustment,
            slope_rad=slope,
            maximum_surface_residual_m=final_residual,
            maximum_normal_spread_rad=final_normal_spread,
            center_world=target_center,
            support_points_world=target_support,
            dense_support_points_world=target_dense,
            normal_world=np.asarray(surface_plane.normal_world),
            rotation=rotation,
            translation=translation,
            cost=float(cost),
        ),
        "accepted",
    )


def _plan_one_foothold(
    terrain: TerrainMeshIndexCompatible,
    intent: FootholdIntent,
    *,
    ray_origin_height_m: float,
    config: TerrainFootholdPlannerConfig,
) -> tuple[PlannedFoothold | None, FootholdSearchDiagnostics]:
    pose = intent.nominal_pose
    support = np.asarray(pose.sole_support_points_world, dtype=np.float64)
    # This call also proves that the supplied points have a non-zero full-foot
    # area.  Dense support uses the hull rather than only the original probes.
    dense_nominal_xy = _dense_polygon_samples(
        support[:, :2], config.footprint_sample_spacing_m
    )
    nominal_plane = _fit_plane(support)
    if nominal_plane.maximum_residual_m > config.maximum_nominal_plane_residual_m:
        diagnostics = FootholdSearchDiagnostics(
            span=intent.span,
            candidate_count=0,
            accepted_candidate_count=0,
            rejection_counts=(("nominal_sole_not_rigid_planar", 1),),
            longitudinal_adjustment_m=None,
            lateral_adjustment_m=None,
            yaw_adjustment_rad=None,
            vertical_adjustment_m=None,
            surface_slope_rad=None,
            maximum_surface_residual_m=None,
            maximum_normal_spread_rad=None,
        )
        return None, diagnostics
    nominal_basis = _basis_from_plane(nominal_plane.normal_world, pose.yaw_rad)
    dense_relative_xy = dense_nominal_xy - np.asarray(pose.sole_center_world[:2])

    rejection_counts: dict[str, int] = {}
    accepted_count = 0
    candidate_count = 0
    best: _Candidate | None = None
    for longitudinal in _symmetric_samples(
        config.maximum_longitudinal_adjustment_m,
        config.longitudinal_samples,
    ):
        for lateral in _symmetric_samples(
            config.maximum_lateral_adjustment_m,
            config.lateral_samples,
        ):
            for yaw in _symmetric_samples(
                config.maximum_yaw_adjustment_rad,
                config.yaw_samples,
            ):
                candidate_count += 1
                candidate, reason = _evaluate_candidate(
                    terrain,
                    intent,
                    nominal_plane,
                    nominal_basis,
                    dense_relative_xy,
                    longitudinal_m=longitudinal,
                    lateral_m=lateral,
                    yaw_adjustment_rad=yaw,
                    ray_origin_height_m=ray_origin_height_m,
                    config=config,
                )
                if candidate is None:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                accepted_count += 1
                if best is None or _candidate_rank(candidate) < _candidate_rank(best):
                    best = candidate
                # The symmetric sample order evaluates the unchanged authored
                # foothold first.  If its complete footprint and collision
                # envelope are already valid, preserve it exactly: searching
                # for a cosmetically flatter displaced patch would introduce
                # needless gait edits and makes smooth-terrain planning much
                # more expensive.
                if (
                    abs(longitudinal) <= _EPSILON
                    and abs(lateral) <= _EPSILON
                    and abs(yaw) <= _EPSILON
                ):
                    diagnostics = FootholdSearchDiagnostics(
                        span=intent.span,
                        candidate_count=candidate_count,
                        accepted_candidate_count=accepted_count,
                        rejection_counts=tuple(sorted(rejection_counts.items())),
                        longitudinal_adjustment_m=candidate.longitudinal_m,
                        lateral_adjustment_m=candidate.lateral_m,
                        yaw_adjustment_rad=candidate.yaw_adjustment_rad,
                        vertical_adjustment_m=candidate.vertical_adjustment_m,
                        surface_slope_rad=candidate.slope_rad,
                        maximum_surface_residual_m=(
                            candidate.maximum_surface_residual_m
                        ),
                        maximum_normal_spread_rad=(
                            candidate.maximum_normal_spread_rad
                        ),
                    )
                    return (
                        PlannedFoothold(
                            intent=intent,
                            sole_center_world=candidate.center_world,
                            sole_support_points_world=(
                                candidate.support_points_world
                            ),
                            dense_support_points_world=(
                                candidate.dense_support_points_world
                            ),
                            surface_normal_world=candidate.normal_world,
                            rotation_world_from_nominal=candidate.rotation,
                            translation_world=candidate.translation,
                            diagnostics=diagnostics,
                        ),
                        diagnostics,
                    )

    if best is None:
        diagnostics = FootholdSearchDiagnostics(
            span=intent.span,
            candidate_count=candidate_count,
            accepted_candidate_count=0,
            rejection_counts=tuple(sorted(rejection_counts.items())),
            longitudinal_adjustment_m=None,
            lateral_adjustment_m=None,
            yaw_adjustment_rad=None,
            vertical_adjustment_m=None,
            surface_slope_rad=None,
            maximum_surface_residual_m=None,
            maximum_normal_spread_rad=None,
        )
        return None, diagnostics

    diagnostics = FootholdSearchDiagnostics(
        span=intent.span,
        candidate_count=candidate_count,
        accepted_candidate_count=accepted_count,
        rejection_counts=tuple(sorted(rejection_counts.items())),
        longitudinal_adjustment_m=best.longitudinal_m,
        lateral_adjustment_m=best.lateral_m,
        yaw_adjustment_rad=best.yaw_adjustment_rad,
        vertical_adjustment_m=best.vertical_adjustment_m,
        surface_slope_rad=best.slope_rad,
        maximum_surface_residual_m=best.maximum_surface_residual_m,
        maximum_normal_spread_rad=best.maximum_normal_spread_rad,
    )
    return (
        PlannedFoothold(
            intent=intent,
            sole_center_world=best.center_world,
            sole_support_points_world=best.support_points_world,
            dense_support_points_world=best.dense_support_points_world,
            surface_normal_world=best.normal_world,
            rotation_world_from_nominal=best.rotation,
            translation_world=best.translation,
            diagnostics=diagnostics,
        ),
        diagnostics,
    )


def _prepare_foothold_search(
    intent: FootholdIntent,
    config: TerrainFootholdPlannerConfig,
) -> tuple[_PreparedFootholdSearch | None, FootholdSearchDiagnostics | None]:
    pose = intent.nominal_pose
    support = np.asarray(pose.sole_support_points_world, dtype=np.float64)
    dense_nominal_xy = _dense_polygon_samples(
        support[:, :2], config.footprint_sample_spacing_m
    )
    nominal_plane = _fit_plane(support)
    if nominal_plane.maximum_residual_m > config.maximum_nominal_plane_residual_m:
        return None, FootholdSearchDiagnostics(
            span=intent.span,
            candidate_count=0,
            accepted_candidate_count=0,
            rejection_counts=(("nominal_sole_not_rigid_planar", 1),),
            longitudinal_adjustment_m=None,
            lateral_adjustment_m=None,
            yaw_adjustment_rad=None,
            vertical_adjustment_m=None,
            surface_slope_rad=None,
            maximum_surface_residual_m=None,
            maximum_normal_spread_rad=None,
        )
    return (
        _PreparedFootholdSearch(
            intent=intent,
            nominal_plane=nominal_plane,
            nominal_basis=_basis_from_plane(
                nominal_plane.normal_world, pose.yaw_rad
            ),
            nominal_dense_relative_xy=(
                dense_nominal_xy
                - np.asarray(pose.sole_center_world[:2], dtype=np.float64)
            ),
            cache={},
        ),
        None,
    )


def _search_diagnostics(
    search: _PreparedFootholdSearch,
    selected: _Candidate | None,
) -> FootholdSearchDiagnostics:
    rejection_counts: dict[str, int] = {}
    accepted_count = 0
    for candidate, reason in search.cache.values():
        if candidate is None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        else:
            accepted_count += 1
    return FootholdSearchDiagnostics(
        span=search.intent.span,
        candidate_count=len(search.cache),
        accepted_candidate_count=accepted_count,
        rejection_counts=tuple(sorted(rejection_counts.items())),
        longitudinal_adjustment_m=(
            None if selected is None else selected.longitudinal_m
        ),
        lateral_adjustment_m=None if selected is None else selected.lateral_m,
        yaw_adjustment_rad=(
            None if selected is None else selected.yaw_adjustment_rad
        ),
        vertical_adjustment_m=(
            None if selected is None else selected.vertical_adjustment_m
        ),
        surface_slope_rad=None if selected is None else selected.slope_rad,
        maximum_surface_residual_m=(
            None if selected is None else selected.maximum_surface_residual_m
        ),
        maximum_normal_spread_rad=(
            None if selected is None else selected.maximum_normal_spread_rad
        ),
    )


def _planned_foothold(
    search: _PreparedFootholdSearch,
    candidate: _Candidate,
    diagnostics: FootholdSearchDiagnostics,
) -> PlannedFoothold:
    return PlannedFoothold(
        intent=search.intent,
        sole_center_world=candidate.center_world,
        sole_support_points_world=candidate.support_points_world,
        dense_support_points_world=candidate.dense_support_points_world,
        surface_normal_world=candidate.normal_world,
        rotation_world_from_nominal=candidate.rotation,
        translation_world=candidate.translation,
        diagnostics=diagnostics,
    )


def _candidate_world_xy_shift(
    search: _PreparedFootholdSearch,
    candidate: _Candidate,
) -> np.ndarray:
    return np.asarray(candidate.center_world[:2], dtype=np.float64) - np.asarray(
        search.intent.nominal_pose.sole_center_world[:2], dtype=np.float64
    )


def _wrapped_angle_difference(left: float, right: float) -> float:
    return abs((float(left) - float(right) + math.pi) % (2.0 * math.pi) - math.pi)


def _stance_anchor_frame(intent: FootholdIntent) -> float:
    return 0.5 * float(intent.span.start_frame + intent.span.stop_frame - 1)


def _select_candidate_sequence(
    searches: Sequence[_PreparedFootholdSearch],
    pools: Sequence[Sequence[_Candidate]],
    config: TerrainFootholdPlannerConfig,
    metrics: _CandidateEvaluationMetrics,
) -> tuple[tuple[_Candidate, ...] | None, float | None, int | None]:
    """Viterbi-select one coherent world-frame adjustment per stance."""

    if len(searches) != len(pools) or not searches:
        raise ContractError("candidate pools must match nonempty foothold searches")
    if any(not pool for pool in pools):
        return None, None, next(
            index for index, pool in enumerate(pools) if not pool
        )

    costs = np.asarray([candidate.cost for candidate in pools[0]], dtype=np.float64)
    backpointers: list[np.ndarray] = []
    for sequence_index in range(1, len(searches)):
        previous_search = searches[sequence_index - 1]
        current_search = searches[sequence_index]
        previous_pool = pools[sequence_index - 1]
        current_pool = pools[sequence_index]
        next_costs = np.full(len(current_pool), np.inf, dtype=np.float64)
        predecessors = np.full(len(current_pool), -1, dtype=np.int64)
        previous_span = previous_search.intent.span
        current_span = current_search.intent.span
        if current_span.start_frame < previous_span.stop_frame:
            # During double support the pelvis anchor is the mean of the two
            # rigid foothold shifts.  Entering or leaving that overlap changes
            # the anchor by half their difference in one frame, so a coherent
            # pair may differ by at most twice the per-frame pelvis bound.
            maximum_rate_limited_change = (
                2.0 * config.maximum_pelvis_planar_adjustment_step_m
            )
        else:
            # Spans are stop-exclusive.  From the final frame of the previous
            # plant to the first frame of the next plant there are this many
            # frame-to-frame intervals over which a swing-gap interpolation
            # can move the pelvis anchor.
            transition_intervals = max(
                1,
                current_span.start_frame - (previous_span.stop_frame - 1),
            )
            maximum_rate_limited_change = (
                config.maximum_pelvis_planar_adjustment_step_m
                * transition_intervals
            )
        maximum_planar_change = min(
            config.maximum_planar_reach_change_m,
            maximum_rate_limited_change,
        )
        for current_index, current in enumerate(current_pool):
            current_shift = _candidate_world_xy_shift(current_search, current)
            for previous_index, previous in enumerate(previous_pool):
                metrics.transition_evaluation_count += 1
                if not math.isfinite(float(costs[previous_index])):
                    continue
                previous_shift = _candidate_world_xy_shift(
                    previous_search, previous
                )
                planar_change = float(
                    np.linalg.norm(current_shift - previous_shift)
                )
                yaw_change = _wrapped_angle_difference(
                    current.yaw_adjustment_rad,
                    previous.yaw_adjustment_rad,
                )
                if (
                    planar_change
                    > maximum_planar_change + _EPSILON
                    or yaw_change > config.maximum_yaw_change_rad + _EPSILON
                ):
                    continue
                metrics.feasible_transition_count += 1
                planar_scale = max(
                    config.maximum_planar_reach_change_m, 1.0e-9
                )
                yaw_scale = max(config.maximum_yaw_change_rad, 1.0e-9)
                smoothing_cost = config.foothold_sequence_smoothing_weight * (
                    (planar_change / planar_scale) ** 2
                    + (yaw_change / yaw_scale) ** 2
                )
                proposed = (
                    float(costs[previous_index])
                    + current.cost
                    + smoothing_cost
                )
                if proposed < next_costs[current_index] - 1.0e-12:
                    next_costs[current_index] = proposed
                    predecessors[current_index] = previous_index
        if not np.isfinite(next_costs).any():
            return None, None, sequence_index
        costs = next_costs
        backpointers.append(predecessors)

    final_index = int(np.argmin(costs))
    selected_indices = [final_index]
    for predecessors in reversed(backpointers):
        final_index = int(predecessors[final_index])
        if final_index < 0:
            raise ContractError("Viterbi path contains a missing predecessor")
        selected_indices.append(final_index)
    selected_indices.reverse()
    selected = tuple(
        pool[index] for pool, index in zip(pools, selected_indices, strict=True)
    )
    return selected, float(np.min(costs)), None


def _selected_sequence_change_metrics(
    searches: Sequence[_PreparedFootholdSearch],
    selected: Sequence[_Candidate],
) -> tuple[float, float]:
    maximum_planar = 0.0
    maximum_yaw = 0.0
    for index in range(1, len(selected)):
        previous_shift = _candidate_world_xy_shift(
            searches[index - 1], selected[index - 1]
        )
        current_shift = _candidate_world_xy_shift(searches[index], selected[index])
        maximum_planar = max(
            maximum_planar,
            float(np.linalg.norm(current_shift - previous_shift)),
        )
        maximum_yaw = max(
            maximum_yaw,
            _wrapped_angle_difference(
                selected[index].yaw_adjustment_rad,
                selected[index - 1].yaw_adjustment_rad,
            ),
        )
    return maximum_planar, maximum_yaw


def _pelvis_planar_offset_values(
    frame_count: int,
    searches: Sequence[_PreparedFootholdSearch],
    selected: Sequence[_Candidate],
) -> np.ndarray:
    accumulated = np.zeros((frame_count, 2), dtype=np.float64)
    active_count = np.zeros(frame_count, dtype=np.int64)
    for search, candidate in zip(searches, selected, strict=True):
        shift = _candidate_world_xy_shift(search, candidate)
        span = search.intent.span
        accumulated[span.start_frame : span.stop_frame] += shift
        active_count[span.start_frame : span.stop_frame] += 1

    anchored = active_count > 0
    if not np.any(anchored):
        raise ContractError("selected footholds contain no active stance frames")
    path = np.zeros((frame_count, 2), dtype=np.float64)
    path[anchored] = accumulated[anchored] / active_count[anchored, None]
    # Preserve the exact active-stance mean.  Only unsupported swing gaps are
    # interpolated; changing the pelvis anchor during a plant creates a false
    # sole reach error even when the selected rigid foothold itself is valid.
    anchor_frames = np.flatnonzero(anchored)
    all_frames = np.arange(frame_count, dtype=np.float64)
    for axis in range(2):
        path[~anchored, axis] = np.interp(
            all_frames[~anchored],
            anchor_frames.astype(np.float64),
            path[anchor_frames, axis],
        )
    return path


def _pelvis_planar_offset_path(
    frame_count: int,
    searches: Sequence[_PreparedFootholdSearch],
    selected: Sequence[_Candidate],
    maximum_step_m: float,
) -> np.ndarray:
    path = _pelvis_planar_offset_values(frame_count, searches, selected)
    maximum_realized_step = (
        0.0
        if frame_count < 2
        else float(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    if maximum_realized_step > maximum_step_m + 1.0e-10:
        raise ContractError(
            "selected foothold sequence exceeds the pelvis planar step bound"
        )
    return path


def _pelvis_planar_step_constraints(
    frame_count: int,
    searches: Sequence[_PreparedFootholdSearch],
) -> tuple[tuple[tuple[int, float], ...], ...]:
    """Express every pelvis-path step as weights on selected foothold shifts.

    Chronological stance spans are not necessarily a first-order chain.  A
    short plant can end while an older opposite-foot plant remains active, so
    the next unsupported-gap interpolation can depend on footholds two or more
    entries apart.  These sparse linear constraints exactly reproduce
    :func:`_pelvis_planar_offset_values` without assuming adjacency in the
    sorted intent list.
    """

    active: list[list[int]] = [[] for _ in range(frame_count)]
    for index, search in enumerate(searches):
        span = search.intent.span
        for frame in range(span.start_frame, span.stop_frame):
            active[frame].append(index)
    anchored = np.asarray([bool(indices) for indices in active], dtype=bool)
    anchor_frames = np.flatnonzero(anchored)
    if not len(anchor_frames):
        raise ContractError("foothold searches contain no active stance frames")

    weights: list[dict[int, float]] = [dict() for _ in range(frame_count)]
    for frame in anchor_frames:
        amount = 1.0 / float(len(active[int(frame)]))
        weights[int(frame)] = {
            index: amount for index in active[int(frame)]
        }
    first = int(anchor_frames[0])
    last = int(anchor_frames[-1])
    for frame in range(first):
        weights[frame] = dict(weights[first])
    for frame in range(last + 1, frame_count):
        weights[frame] = dict(weights[last])
    for left, right in zip(anchor_frames[:-1], anchor_frames[1:], strict=True):
        left = int(left)
        right = int(right)
        if right == left + 1:
            continue
        for frame in range(left + 1, right):
            alpha = float(frame - left) / float(right - left)
            value: dict[int, float] = {}
            for index, coefficient in weights[left].items():
                value[index] = value.get(index, 0.0) + (1.0 - alpha) * coefficient
            for index, coefficient in weights[right].items():
                value[index] = value.get(index, 0.0) + alpha * coefficient
            weights[frame] = value

    constraints: list[tuple[tuple[int, float], ...]] = []
    for previous, current in zip(weights[:-1], weights[1:], strict=True):
        difference: dict[int, float] = {}
        for index, coefficient in current.items():
            difference[index] = difference.get(index, 0.0) + coefficient
        for index, coefficient in previous.items():
            difference[index] = difference.get(index, 0.0) - coefficient
        constraints.append(
            tuple(
                (index, coefficient)
                for index, coefficient in sorted(difference.items())
                if abs(coefficient) > 1.0e-12
            )
        )
    return tuple(constraints)


def _select_exact_rate_candidate_sequence(
    searches: Sequence[_PreparedFootholdSearch],
    pools: Sequence[Sequence[_Candidate]],
    frame_count: int,
    config: TerrainFootholdPlannerConfig,
    metrics: _CandidateEvaluationMetrics,
    *,
    beam_width: int = 4096,
) -> tuple[tuple[_Candidate, ...] | None, float | None, int | None]:
    """Fallback beam search using the exact realized pelvis-path constraints."""

    constraints = _pelvis_planar_step_constraints(frame_count, searches)
    constraints_by_latest: list[list[tuple[tuple[int, float], ...]]] = [
        [] for _ in searches
    ]
    for constraint in constraints:
        if constraint:
            constraints_by_latest[max(index for index, _ in constraint)].append(
                constraint
            )

    # Each state stores the full short foothold sequence.  In practice this
    # fallback is entered only for overlapping support schedules; the ordinary
    # zero/first-order paths remain the fast path.
    states: list[tuple[float, tuple[_Candidate, ...]]] = [
        (candidate.cost, (candidate,)) for candidate in pools[0]
    ]
    for sequence_index in range(len(searches)):
        if sequence_index > 0:
            expanded: list[tuple[float, tuple[_Candidate, ...]]] = []
            current_search = searches[sequence_index]
            previous_search = searches[sequence_index - 1]
            for previous_cost, previous_selected in states:
                previous = previous_selected[-1]
                previous_shift = _candidate_world_xy_shift(
                    previous_search, previous
                )
                for current in pools[sequence_index]:
                    metrics.transition_evaluation_count += 1
                    current_shift = _candidate_world_xy_shift(
                        current_search, current
                    )
                    planar_change = float(
                        np.linalg.norm(current_shift - previous_shift)
                    )
                    yaw_change = _wrapped_angle_difference(
                        current.yaw_adjustment_rad,
                        previous.yaw_adjustment_rad,
                    )
                    if (
                        planar_change
                        > config.maximum_planar_reach_change_m + _EPSILON
                        or yaw_change
                        > config.maximum_yaw_change_rad + _EPSILON
                    ):
                        continue
                    selected = previous_selected + (current,)
                    feasible = True
                    for constraint in constraints_by_latest[sequence_index]:
                        step = np.zeros(2, dtype=np.float64)
                        for index, coefficient in constraint:
                            step += coefficient * _candidate_world_xy_shift(
                                searches[index], selected[index]
                            )
                        if (
                            float(np.linalg.norm(step))
                            > config.maximum_pelvis_planar_adjustment_step_m
                            + _EPSILON
                        ):
                            feasible = False
                            break
                    if not feasible:
                        continue
                    metrics.feasible_transition_count += 1
                    planar_scale = max(
                        config.maximum_planar_reach_change_m, 1.0e-9
                    )
                    yaw_scale = max(config.maximum_yaw_change_rad, 1.0e-9)
                    smoothing_cost = config.foothold_sequence_smoothing_weight * (
                        (planar_change / planar_scale) ** 2
                        + (yaw_change / yaw_scale) ** 2
                    )
                    expanded.append(
                        (
                            previous_cost + current.cost + smoothing_cost,
                            selected,
                        )
                    )
            if not expanded:
                return None, None, sequence_index
            expanded.sort(
                key=lambda state: (
                    state[0],
                    tuple(_candidate_rank(value) for value in state[1]),
                )
            )
            states = expanded[:beam_width]
        else:
            filtered: list[tuple[float, tuple[_Candidate, ...]]] = []
            for state in states:
                candidate = state[1][0]
                shift = _candidate_world_xy_shift(searches[0], candidate)
                feasible = all(
                    float(
                        np.linalg.norm(
                            sum(
                                (
                                    coefficient * shift
                                    for _index, coefficient in constraint
                                ),
                                np.zeros(2, dtype=np.float64),
                            )
                        )
                    )
                    <= config.maximum_pelvis_planar_adjustment_step_m + _EPSILON
                    for constraint in constraints_by_latest[0]
                )
                if feasible:
                    filtered.append(state)
            states = filtered
            if not states:
                return None, None, 0
    best_cost, best = states[0]
    return best, float(best_cost), None


def _tighten_rate_feasible_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    maximum_step_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    low = np.array(lower, copy=True)
    high = np.array(upper, copy=True)
    for frame in range(1, len(low)):
        low[frame] = max(low[frame], low[frame - 1] - maximum_step_m)
        high[frame] = min(high[frame], high[frame - 1] + maximum_step_m)
        if low[frame] > high[frame] + 1.0e-10:
            return None, None, frame
    for frame in range(len(low) - 2, -1, -1):
        low[frame] = max(low[frame], low[frame + 1] - maximum_step_m)
        high[frame] = min(high[frame], high[frame + 1] + maximum_step_m)
        if low[frame] > high[frame] + 1.0e-10:
            return None, None, frame
    return low, high, None


def _smooth_height_path(
    desired: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    maximum_step_m: float,
    smoothing_weight: float,
    iterations: int,
) -> np.ndarray:
    height = np.empty_like(desired)
    height[0] = float(np.clip(desired[0], lower[0], upper[0]))
    for frame in range(1, len(height)):
        local_lower = max(lower[frame], height[frame - 1] - maximum_step_m)
        local_upper = min(upper[frame], height[frame - 1] + maximum_step_m)
        if local_lower > local_upper + 1.0e-10:
            raise ContractError("rate-feasible corridor could not produce a height path")
        height[frame] = float(np.clip(desired[frame], local_lower, local_upper))

    if len(height) == 1 or smoothing_weight == 0.0:
        return height
    for _ in range(iterations):
        for order in (range(len(height)), range(len(height) - 1, -1, -1)):
            for frame in order:
                neighbours: list[float] = []
                local_lower = lower[frame]
                local_upper = upper[frame]
                if frame > 0:
                    neighbours.append(float(height[frame - 1]))
                    local_lower = max(local_lower, height[frame - 1] - maximum_step_m)
                    local_upper = min(local_upper, height[frame - 1] + maximum_step_m)
                if frame + 1 < len(height):
                    neighbours.append(float(height[frame + 1]))
                    local_lower = max(local_lower, height[frame + 1] - maximum_step_m)
                    local_upper = min(local_upper, height[frame + 1] + maximum_step_m)
                if local_lower > local_upper + 1.0e-10:
                    raise ContractError("pelvis smoothing left the feasible corridor")
                proposed = (
                    float(desired[frame])
                    + smoothing_weight * sum(neighbours)
                ) / (1.0 + smoothing_weight * len(neighbours))
                height[frame] = float(np.clip(proposed, local_lower, local_upper))
    return height


def _plan_pelvis_height_corridor(
    nominal_height_world_m: np.ndarray,
    footholds: Sequence[PlannedFoothold],
    config: TerrainFootholdPlannerConfig,
) -> tuple[PelvisHeightCorridor | None, str | None, int | None]:
    nominal = np.asarray(nominal_height_world_m, dtype=np.float64)
    lower = nominal - config.maximum_pelvis_height_adjustment_m
    upper = nominal + config.maximum_pelvis_height_adjustment_m
    active_shifts: list[list[float]] = [[] for _ in range(len(nominal))]

    for foothold in footholds:
        pose = foothold.intent.nominal_pose
        shift = float(
            np.mean(foothold.sole_support_points_world[:, 2])
            - np.mean(pose.sole_support_points_world[:, 2])
        )
        span = foothold.intent.span
        for frame in range(span.start_frame, span.stop_frame):
            lower[frame] = max(
                lower[frame],
                nominal[frame] + shift - config.pelvis_reach_slack_m,
            )
            upper[frame] = min(
                upper[frame],
                nominal[frame] + shift + config.pelvis_reach_slack_m,
            )
            active_shifts[frame].append(shift)
            if lower[frame] > upper[frame] + 1.0e-10:
                return None, "pelvis_reach_intervals_do_not_overlap", frame

    desired_shift = np.full(len(nominal), np.nan, dtype=np.float64)
    for frame, values in enumerate(active_shifts):
        if values:
            desired_shift[frame] = float(np.mean(values))
    fixed = np.flatnonzero(np.isfinite(desired_shift))
    if len(fixed):
        desired_shift = np.interp(
            np.arange(len(nominal), dtype=np.float64),
            fixed.astype(np.float64),
            desired_shift[fixed],
        )
    else:
        desired_shift.fill(0.0)
    desired = nominal + desired_shift

    tightened_lower, tightened_upper, failed_frame = _tighten_rate_feasible_bounds(
        lower,
        upper,
        config.maximum_pelvis_height_step_m,
    )
    if tightened_lower is None or tightened_upper is None:
        return None, "pelvis_height_rate_is_infeasible", failed_frame
    selected = _smooth_height_path(
        desired,
        tightened_lower,
        tightened_upper,
        maximum_step_m=config.maximum_pelvis_height_step_m,
        smoothing_weight=config.pelvis_smoothing_weight,
        iterations=config.pelvis_smoothing_iterations,
    )
    if len(selected) > 1 and float(np.max(np.abs(np.diff(selected)))) > (
        config.maximum_pelvis_height_step_m + 1.0e-9
    ):
        raise ContractError("selected pelvis path violates its height-step bound")
    return (
        PelvisHeightCorridor(
            minimum_height_world_m=tightened_lower,
            maximum_height_world_m=tightened_upper,
            selected_height_world_m=selected,
        ),
        None,
        None,
    )


def _validated_intents(
    intents: Sequence[FootholdIntent],
    frame_count: int,
    config: TerrainFootholdPlannerConfig,
) -> tuple[FootholdIntent, ...]:
    values = tuple(intents)
    if not values:
        raise ContractError("at least one foothold intent is required")
    if not all(isinstance(value, FootholdIntent) for value in values):
        raise ContractError("intents must contain only FootholdIntent values")
    ordered = tuple(
        sorted(
            values,
            key=lambda value: (
                value.span.start_frame,
                value.span.stop_frame,
                value.span.foot_index,
            ),
        )
    )
    for intent in ordered:
        if intent.span.stop_frame > frame_count:
            raise ContractError("stance span lies outside the pelvis trajectory")
    for foot in (0, 1):
        spans = [value.span for value in ordered if value.span.foot_index == foot]
        for previous, current in zip(spans, spans[1:]):
            if current.start_frame < previous.stop_frame:
                raise ContractError("stance windows for one foot must not overlap")
    if config.require_alternating_feet:
        for previous, current in zip(ordered, ordered[1:]):
            if previous.span.foot_index == current.span.foot_index:
                raise ContractError("chronological stance windows must alternate feet")
    return ordered


def _validated_terrain_height(terrain: object) -> float:
    if not callable(getattr(terrain, "raycast", None)):
        raise ContractError("terrain must provide raycast(origin, direction)")
    vertices = np.asarray(getattr(terrain, "vertices_world", None), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ContractError("terrain must provide nonempty vertices_world [N,3]")
    if not np.isfinite(vertices).all():
        raise ContractError("terrain vertices must be finite")
    return float(np.max(vertices[:, 2]))


def plan_terrain_footholds(
    terrain: TerrainMeshIndexCompatible,
    intents: Sequence[FootholdIntent],
    nominal_pelvis_height_world_m: object,
    *,
    config: TerrainFootholdPlannerConfig | None = None,
) -> TerrainFootholdPlanningResult:
    """Plan rigid footholds and a smooth pelvis-height corridor.

    Expected geometric failures are returned as diagnostics.  Malformed input
    remains a :class:`ContractError`.  Accepted transforms should be applied to
    the corresponding nominal sole-sphere centres and collision-envelope
    points before running the existing IK and exact collision/contact audits.
    """

    started_at = time.perf_counter()
    settings = config if config is not None else TerrainFootholdPlannerConfig()
    if not isinstance(settings, TerrainFootholdPlannerConfig):
        raise ContractError("config must be TerrainFootholdPlannerConfig")
    terrain_maximum_height = _validated_terrain_height(terrain)
    pelvis = np.asarray(nominal_pelvis_height_world_m, dtype=np.float64)
    if pelvis.ndim != 1 or not len(pelvis) or not np.isfinite(pelvis).all():
        raise ContractError("nominal pelvis height must be a finite nonempty vector")
    ordered = _validated_intents(intents, len(pelvis), settings)
    maximum_nominal_height = max(
        float(np.max(value.nominal_pose.sole_support_points_world[:, 2]))
        for value in ordered
    )
    ray_height = (
        max(terrain_maximum_height, maximum_nominal_height)
        + settings.ray_origin_margin_m
    )

    metrics = _CandidateEvaluationMetrics()
    searches: list[_PreparedFootholdSearch] = []
    for intent in ordered:
        search, preparation_failure = _prepare_foothold_search(intent, settings)
        if search is None:
            assert preparation_failure is not None
            previous_diagnostics = tuple(
                _search_diagnostics(previous, None) for previous in searches
            )
            return TerrainFootholdPlanningResult(
                plan=None,
                diagnostics=TerrainFootholdPlanningDiagnostics(
                    accepted=False,
                    reason_code="foothold_has_no_rigid_support_patch",
                    message=(
                        "nominal sole cannot enter exact-support search for "
                        f"foot {intent.span.foot_index}, frames "
                        f"{intent.span.start_frame}:{intent.span.stop_frame}"
                    ),
                    rejected_span=intent.span,
                    foothold_searches=(
                        *previous_diagnostics,
                        preparation_failure,
                    ),
                    sequence_search_stage="nominal_geometry",
                    candidate_pool_sizes=tuple(0 for _ in ordered),
                    planning_runtime_seconds=time.perf_counter() - started_at,
                ),
            )
        searches.append(search)

    zero_candidates: list[_Candidate] = []
    zero_fast_path = True
    for search in searches:
        candidate, _ = search.evaluate(
            terrain,
            longitudinal_m=0.0,
            lateral_m=0.0,
            yaw_adjustment_rad=0.0,
            ray_origin_height_m=ray_height,
            config=settings,
            metrics=metrics,
        )
        if candidate is None:
            zero_fast_path = False
        else:
            zero_candidates.append(candidate)

    selected: tuple[_Candidate, ...] | None = None
    selected_cost: float | None = None
    pelvis_planar_offset: np.ndarray | None = None
    failed_sequence_index: int | None = None
    sequence_stage = "zero_only"
    candidate_pool_sizes: tuple[int, ...] = tuple(
        int(search.cache[(0.0, 0.0, 0.0)][0] is not None)
        for search in searches
    )
    if zero_fast_path:
        selected = tuple(zero_candidates)
        selected_cost = float(sum(candidate.cost for candidate in selected))
        pelvis_planar_offset = _pelvis_planar_offset_path(
            len(pelvis),
            searches,
            selected,
            settings.maximum_pelvis_planar_adjustment_step_m,
        )
    else:
        longitudinal_values = _symmetric_samples(
            settings.maximum_longitudinal_adjustment_m,
            settings.longitudinal_samples,
        )
        lateral_values = _symmetric_samples(
            settings.maximum_lateral_adjustment_m,
            settings.lateral_samples,
        )
        yaw_values = _symmetric_samples(
            settings.maximum_yaw_adjustment_rad,
            settings.yaw_samples,
        )
        stage_parameters: list[
            tuple[str, tuple[tuple[float, float, float], ...]]
        ] = []

        def append_stage(
            name: str, parameters: Sequence[tuple[float, float, float]]
        ) -> None:
            unique = tuple(dict.fromkeys(parameters))
            if not stage_parameters or unique != stage_parameters[-1][1]:
                stage_parameters.append((name, unique))

        append_stage(
            "longitudinal_only",
            tuple((value, 0.0, 0.0) for value in longitudinal_values),
        )
        append_stage(
            "planar",
            tuple(
                (longitudinal, lateral, 0.0)
                for longitudinal in longitudinal_values
                for lateral in lateral_values
            ),
        )
        append_stage(
            "full_yaw",
            tuple(
                (longitudinal, lateral, yaw)
                for longitudinal in longitudinal_values
                for lateral in lateral_values
                for yaw in yaw_values
            ),
        )

        final_pools: tuple[tuple[_Candidate, ...], ...] = ()
        for sequence_stage, parameters in stage_parameters:
            for search in searches:
                for longitudinal, lateral, yaw in parameters:
                    search.evaluate(
                        terrain,
                        longitudinal_m=longitudinal,
                        lateral_m=lateral,
                        yaw_adjustment_rad=yaw,
                        ray_origin_height_m=ray_height,
                        config=settings,
                        metrics=metrics,
                    )
            final_pools = tuple(
                search.accepted_candidates() for search in searches
            )
            candidate_pool_sizes = tuple(len(pool) for pool in final_pools)
            selected, selected_cost, failed_sequence_index = (
                _select_candidate_sequence(
                    searches,
                    final_pools,
                    settings,
                    metrics,
                )
            )
            if selected is not None:
                try:
                    pelvis_planar_offset = _pelvis_planar_offset_path(
                        len(pelvis),
                        searches,
                        selected,
                        settings.maximum_pelvis_planar_adjustment_step_m,
                    )
                except ContractError:
                    # Adjacent chronological footholds are insufficient when
                    # a short plant ends inside a longer opposite-foot plant.
                    # Re-select with the exact realized per-frame path rather
                    # than allowing the builder to discover a one-frame jump.
                    selected, selected_cost, failed_sequence_index = (
                        _select_exact_rate_candidate_sequence(
                            searches,
                            final_pools,
                            len(pelvis),
                            settings,
                            metrics,
                        )
                    )
                    if selected is not None:
                        pelvis_planar_offset = _pelvis_planar_offset_path(
                            len(pelvis),
                            searches,
                            selected,
                            settings.maximum_pelvis_planar_adjustment_step_m,
                        )
                        sequence_stage = f"{sequence_stage}_exact_rate"
                if selected is not None:
                    break

        if selected is None:
            search_diagnostics = tuple(
                _search_diagnostics(search, None) for search in searches
            )
            empty_pool_index = next(
                (
                    index
                    for index, size in enumerate(candidate_pool_sizes)
                    if size == 0
                ),
                None,
            )
            if empty_pool_index is not None:
                rejected_index = empty_pool_index
                rejected = searches[rejected_index]
                summary = ", ".join(
                    f"{reason}={count}"
                    for reason, count in search_diagnostics[
                        rejected_index
                    ].rejection_counts
                )
                reason_code = "foothold_has_no_rigid_support_patch"
                message = (
                    "no bounded full-footprint support candidate for "
                    f"foot {rejected.intent.span.foot_index}, frames "
                    f"{rejected.intent.span.start_frame}:"
                    f"{rejected.intent.span.stop_frame}"
                    + (f" ({summary})" if summary else "")
                )
            else:
                rejected_index = (
                    failed_sequence_index
                    if failed_sequence_index is not None
                    else len(searches) - 1
                )
                rejected = searches[rejected_index]
                reason_code = "foothold_sequence_is_infeasible"
                message = (
                    "each stance has exact full-foot support, but no sequence "
                    "satisfies the world-frame planar/yaw continuity bounds at "
                    f"foot {rejected.intent.span.foot_index}, frames "
                    f"{rejected.intent.span.start_frame}:"
                    f"{rejected.intent.span.stop_frame}"
                )
            return TerrainFootholdPlanningResult(
                plan=None,
                diagnostics=TerrainFootholdPlanningDiagnostics(
                    accepted=False,
                    reason_code=reason_code,
                    message=message,
                    rejected_span=rejected.intent.span,
                    foothold_searches=search_diagnostics,
                    sequence_search_stage=sequence_stage,
                    candidate_pool_sizes=candidate_pool_sizes,
                    candidate_evaluation_count=metrics.evaluation_count,
                    candidate_cache_hit_count=metrics.cache_hit_count,
                    sequence_transition_evaluation_count=(
                        metrics.transition_evaluation_count
                    ),
                    sequence_feasible_transition_count=(
                        metrics.feasible_transition_count
                    ),
                    planning_runtime_seconds=time.perf_counter() - started_at,
                ),
            )

    assert selected is not None
    search_diagnostics = tuple(
        _search_diagnostics(search, candidate)
        for search, candidate in zip(searches, selected, strict=True)
    )
    planned = tuple(
        _planned_foothold(search, candidate, diagnostics)
        for search, candidate, diagnostics in zip(
            searches, selected, search_diagnostics, strict=True
        )
    )
    maximum_planar_change, maximum_yaw_change = (
        _selected_sequence_change_metrics(searches, selected)
    )
    assert pelvis_planar_offset is not None
    maximum_pelvis_planar_step = (
        0.0
        if len(pelvis_planar_offset) < 2
        else float(
            np.max(
                np.linalg.norm(np.diff(pelvis_planar_offset, axis=0), axis=1)
            )
        )
    )

    corridor, corridor_reason, rejected_frame = _plan_pelvis_height_corridor(
        pelvis,
        planned,
        settings,
    )
    if corridor is None:
        return TerrainFootholdPlanningResult(
            plan=None,
            diagnostics=TerrainFootholdPlanningDiagnostics(
                accepted=False,
                reason_code=corridor_reason,
                message=(
                    "footholds are supported but pelvis height is infeasible "
                    f"at frame {rejected_frame}"
                ),
                rejected_span=None,
                rejected_frame=rejected_frame,
                foothold_searches=search_diagnostics,
                sequence_search_stage=sequence_stage,
                candidate_pool_sizes=candidate_pool_sizes,
                candidate_evaluation_count=metrics.evaluation_count,
                candidate_cache_hit_count=metrics.cache_hit_count,
                sequence_transition_evaluation_count=(
                    metrics.transition_evaluation_count
                ),
                sequence_feasible_transition_count=(
                    metrics.feasible_transition_count
                ),
                selected_sequence_cost=selected_cost,
                maximum_selected_world_xy_shift_change_m=(
                    maximum_planar_change
                ),
                maximum_selected_yaw_change_rad=maximum_yaw_change,
                maximum_selected_pelvis_planar_step_m=(
                    maximum_pelvis_planar_step
                ),
                planning_runtime_seconds=time.perf_counter() - started_at,
            ),
        )

    return TerrainFootholdPlanningResult(
        plan=TerrainFootholdPlan(
            footholds=tuple(planned),
            pelvis_height_corridor=corridor,
            pelvis_planar_offset_world_m=pelvis_planar_offset,
        ),
        diagnostics=TerrainFootholdPlanningDiagnostics(
            accepted=True,
            reason_code=None,
            message="all stance footprints and the pelvis-height corridor are feasible",
            rejected_span=None,
            foothold_searches=search_diagnostics,
            sequence_search_stage=sequence_stage,
            candidate_pool_sizes=candidate_pool_sizes,
            candidate_evaluation_count=metrics.evaluation_count,
            candidate_cache_hit_count=metrics.cache_hit_count,
            sequence_transition_evaluation_count=(
                metrics.transition_evaluation_count
            ),
            sequence_feasible_transition_count=(
                metrics.feasible_transition_count
            ),
            selected_sequence_cost=selected_cost,
            maximum_selected_world_xy_shift_change_m=maximum_planar_change,
            maximum_selected_yaw_change_rad=maximum_yaw_change,
            maximum_selected_pelvis_planar_step_m=maximum_pelvis_planar_step,
            planning_runtime_seconds=time.perf_counter() - started_at,
        ),
    )


__all__ = (
    "FootholdIntent",
    "FootholdSearchDiagnostics",
    "PelvisHeightCorridor",
    "PlannedFoothold",
    "TerrainFootholdPlan",
    "TerrainFootholdPlannerConfig",
    "TerrainFootholdPlanningDiagnostics",
    "TerrainFootholdPlanningResult",
    "TerrainMeshIndexCompatible",
    "plan_terrain_footholds",
)
