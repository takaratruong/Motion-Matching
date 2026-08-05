"""Terrain-valid planted-foot anchors for an exact stair support route.

The solver deliberately works from sampled sole points and downward queries on
the canonical target mesh.  It is therefore safe to invoke for every authored
stance span without depending on a renderer, a controller, or source-clip
heuristics.  A candidate must support the requested number of sole points on
the selected target tread.  A ray hit above a sole sample is an exact
target-mesh collision with a higher next tread and disqualifies that candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mm_sonic.joints import ContractError

from .stair_support_route import StairSupportRoute
from .terrain_mesh import TerrainMeshIndex


_DOWN = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)


def _readonly_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{name} must be finite with shape {shape}")
    result = np.ascontiguousarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class StanceSpan:
    """Half-open authored contact interval for one planted foot."""

    foot_index: int
    start_frame: int
    stop_frame: int

    def __post_init__(self) -> None:
        if self.foot_index not in (0, 1):
            raise ContractError("foot_index must be 0 (left) or 1 (right)")
        if self.start_frame < 0 or self.stop_frame <= self.start_frame:
            raise ContractError("stance span must be nonempty with nonnegative frames")


@dataclass(frozen=True)
class NominalFootSolePose:
    """Nominal planted-foot pose represented by world-space sole samples.

    The solver rotates only the XY offsets around ``sole_center_world``.  It
    retains every support point's relative Z coordinate, so nominal pitch is
    preserved rather than silently flattening the foot onto a tread.
    """

    sole_center_world: np.ndarray
    sole_support_points_world: np.ndarray
    yaw_rad: float
    collision_envelope_points_world: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sole_center_world",
            _readonly_array(self.sole_center_world, (3,), "sole_center_world"),
        )
        points = np.asarray(self.sole_support_points_world, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[0] < 1
            or points.shape[1:] != (3,)
            or not np.isfinite(points).all()
        ):
            raise ContractError("sole_support_points_world must be finite with shape [N,3]")
        result = np.ascontiguousarray(points).copy()
        result.flags.writeable = False
        object.__setattr__(self, "sole_support_points_world", result)
        if self.collision_envelope_points_world is not None:
            envelope = np.asarray(
                self.collision_envelope_points_world, dtype=np.float64
            )
            if (
                envelope.ndim != 2
                or envelope.shape[0] < 1
                or envelope.shape[1:] != (3,)
                or not np.isfinite(envelope).all()
            ):
                raise ContractError(
                    "collision_envelope_points_world must be finite with shape [N,3]"
                )
            envelope_result = np.ascontiguousarray(envelope).copy()
            envelope_result.flags.writeable = False
            object.__setattr__(
                self,
                "collision_envelope_points_world",
                envelope_result,
            )
        if not math.isfinite(self.yaw_rad):
            raise ContractError("yaw_rad must be finite")


@dataclass(frozen=True)
class FootholdAnchorConfig:
    """Bounded local-search and exact support thresholds for one stance."""

    max_longitudinal_adjustment_m: float = 0.12
    max_lateral_adjustment_m: float = 0.40
    max_yaw_adjustment_rad: float = 0.20
    max_vertical_adjustment_m: float = 0.08
    longitudinal_samples: int = 13
    lateral_samples: int = 7
    yaw_samples: int = 7
    support_height_tolerance_m: float = 0.015
    collision_clearance_m: float = 0.0025
    minimum_support_points: int | None = 2
    lateral_seed_offsets_m: tuple[float, ...] = (-0.11, 0.0, 0.11)
    route_lateral_offset_m: float = 0.11
    route_alignment_weight: float = 10.0
    tread_edge_clearance_m: float = 0.01

    def __post_init__(self) -> None:
        limits = (
            self.max_longitudinal_adjustment_m,
            self.max_lateral_adjustment_m,
            self.max_yaw_adjustment_rad,
            self.max_vertical_adjustment_m,
            self.support_height_tolerance_m,
            self.collision_clearance_m,
            self.route_lateral_offset_m,
            self.route_alignment_weight,
            self.tread_edge_clearance_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in limits):
            raise ContractError("foothold anchor limits must be finite and nonnegative")
        if any(samples < 1 for samples in (
            self.longitudinal_samples,
            self.lateral_samples,
            self.yaw_samples,
        )):
            raise ContractError("foothold anchor sample counts must be positive")
        if self.minimum_support_points is not None and self.minimum_support_points < 1:
            raise ContractError("minimum_support_points must be positive when provided")
        if not self.lateral_seed_offsets_m or not all(
            math.isfinite(value) for value in self.lateral_seed_offsets_m
        ):
            raise ContractError("lateral_seed_offsets_m must be nonempty and finite")


@dataclass(frozen=True)
class FootholdAnchorDiagnostics:
    """Auditable result of a local planted-foot placement search."""

    span: StanceSpan
    target_level_index: int
    candidate_count: int
    best_support_point_count: int
    required_support_point_count: int
    nominal_support_point_count: int
    collision_point_count: int
    longitudinal_adjustment_m: float
    lateral_adjustment_m: float
    yaw_adjustment_rad: float
    vertical_adjustment_m: float
    cost: float | None
    rejection_reason: str | None


@dataclass(frozen=True)
class FootholdAnchor:
    """A target-tread pose to apply identically to every frame in ``span``."""

    span: StanceSpan
    target_level_index: int
    pose: NominalFootSolePose
    diagnostics: FootholdAnchorDiagnostics


class FootholdAnchorRejected(ValueError):
    """Raised when no bounded adjustment can provide genuine tread support."""

    def __init__(self, diagnostics: FootholdAnchorDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(
            "no terrain-valid foothold for stance span "
            f"{diagnostics.span.start_frame}:{diagnostics.span.stop_frame}"
        )


@dataclass(frozen=True)
class _Candidate:
    longitudinal_m: float
    lateral_m: float
    yaw_rad: float
    vertical_m: float
    support_point_count: int
    collision_point_count: int
    cost: float
    pose: NominalFootSolePose


def _samples(limit: float, count: int) -> np.ndarray:
    if count == 1 or limit == 0.0:
        return np.asarray((0.0,), dtype=np.float64)
    return np.linspace(-limit, limit, count, dtype=np.float64)


def _offset_samples(
    limit: float,
    count: int,
    extra: tuple[float, ...],
) -> np.ndarray:
    values = list(_samples(limit, count))
    values.extend(value for value in extra if abs(value) <= limit)
    return np.asarray(sorted(set(float(value) for value in values)), dtype=np.float64)


def _lateral_samples(
    settings: FootholdAnchorConfig,
    extra: tuple[float, ...] = (),
) -> np.ndarray:
    values = list(
        _samples(settings.max_lateral_adjustment_m, settings.lateral_samples)
    )
    values.extend(
        offset
        for offset in settings.lateral_seed_offsets_m
        if abs(offset) <= settings.max_lateral_adjustment_m
    )
    values.extend(
        offset
        for offset in extra
        if abs(offset) <= settings.max_lateral_adjustment_m
    )
    return np.asarray(sorted(set(float(value) for value in values)), dtype=np.float64)


def _route_axes(route: StairSupportRoute) -> tuple[np.ndarray, np.ndarray]:
    start = np.asarray(route.start_xy, dtype=np.float64)
    end = np.asarray(route.end_xy, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite((start, end)).all():
        raise ContractError("stair route endpoints must be finite XY vectors")
    route_vector = end - start
    length = float(np.linalg.norm(route_vector))
    if length <= 1.0e-9:
        raise ContractError("stair route must have nonzero length")
    longitudinal = route_vector / length
    return longitudinal, np.asarray((-longitudinal[1], longitudinal[0]), dtype=np.float64)


def _foot_collision_envelope_points(
    support_points: object,
    *,
    foot_forward_extension_m: float = 0.025,
    foot_backward_extension_m: float = 0.045,
    foot_lateral_extension_m: float = 0.010,
    foot_collision_height_m: float = 0.060,
) -> np.ndarray:
    """Return a conservative oriented box around the complete G1 foot mesh."""

    support = np.asarray(support_points, dtype=np.float64)
    if (
        support.ndim != 2
        or support.shape[0] < 4
        or support.shape[1:] != (3,)
        or not np.isfinite(support).all()
    ):
        raise ContractError("foot support points must be finite with shape [N>=4,3]")
    split = len(support) // 2
    foot_forward = (
        np.mean(support[split:], axis=0)
        - np.mean(support[:split], axis=0)
    )
    forward_norm = float(np.linalg.norm(foot_forward))
    if forward_norm <= 1.0e-9:
        raise ContractError("foot support points have no forward axis")
    foot_forward /= forward_norm
    foot_lateral = 0.5 * (
        support[0] - support[1]
        + support[split] - support[split + 1]
    )
    foot_lateral -= np.dot(foot_lateral, foot_forward) * foot_forward
    lateral_norm = float(np.linalg.norm(foot_lateral))
    if lateral_norm <= 1.0e-9:
        raise ContractError("foot support points have no lateral axis")
    foot_lateral /= lateral_norm
    foot_up = np.cross(foot_forward, foot_lateral)
    foot_up /= np.linalg.norm(foot_up)
    if foot_up[2] < 0.0:
        foot_lateral *= -1.0
        foot_up *= -1.0

    origin = np.mean(support, axis=0)
    relative = support - origin
    coordinates = np.stack(
        (
            relative @ foot_forward,
            relative @ foot_lateral,
            relative @ foot_up,
        ),
        axis=1,
    )
    lower = np.min(coordinates, axis=0)
    upper = np.max(coordinates, axis=0)
    lower[0] -= float(foot_backward_extension_m)
    upper[0] += float(foot_forward_extension_m)
    lower[1] -= float(foot_lateral_extension_m)
    upper[1] += float(foot_lateral_extension_m)
    upper[2] += float(foot_collision_height_m)
    basis = np.stack((foot_forward, foot_lateral, foot_up), axis=1)
    return np.asarray(
        [
            origin + basis @ np.asarray((forward, lateral, vertical))
            for forward in (lower[0], upper[0])
            for lateral in (lower[1], upper[1])
            for vertical in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def _support_and_collision_counts(
    mesh: TerrainMeshIndex,
    points: np.ndarray,
    *,
    target_height_m: float,
    ray_origin_height_m: float,
    support_tolerance_m: float,
    collision_tolerance_m: float,
    ground_fallback_height_m: float | None,
) -> tuple[int, int]:
    support_count = 0
    collision_count = 0
    for point in points:
        hit = mesh.raycast(
            np.asarray((point[0], point[1], ray_origin_height_m), dtype=np.float64),
            _DOWN,
        )
        if hit is None:
            if ground_fallback_height_m is None:
                continue
            hit_height = float(ground_fallback_height_m)
        else:
            hit_height = float(hit.position_world[2])
        if hit_height > float(point[2]) + collision_tolerance_m:
            collision_count += 1
        if (
            abs(hit_height - target_height_m) <= support_tolerance_m
            and abs(float(point[2]) - hit_height) <= support_tolerance_m
        ):
            support_count += 1
    return support_count, collision_count


def _support_and_full_foot_collision_counts(
    mesh: TerrainMeshIndex,
    points: np.ndarray,
    *,
    target_height_m: float,
    ray_origin_height_m: float,
    support_tolerance_m: float,
    collision_tolerance_m: float,
    ground_fallback_height_m: float | None,
    collision_envelope_points: np.ndarray | None = None,
) -> tuple[int, int]:
    support_count, sole_collision_count = _support_and_collision_counts(
        mesh,
        points,
        target_height_m=target_height_m,
        ray_origin_height_m=ray_origin_height_m,
        support_tolerance_m=support_tolerance_m,
        collision_tolerance_m=collision_tolerance_m,
        ground_fallback_height_m=ground_fallback_height_m,
    )
    envelope_collision_count = 0
    has_exact_collision_envelope = collision_envelope_points is not None
    envelope = (
        _foot_collision_envelope_points(points)
        if collision_envelope_points is None
        else np.asarray(collision_envelope_points, dtype=np.float64)
    )
    for point in envelope:
        hit = mesh.raycast(
            np.asarray(
                (point[0], point[1], ray_origin_height_m),
                dtype=np.float64,
            ),
            _DOWN,
        )
        if hit is None:
            continue
        hit_height = float(hit.position_world[2])
        if (
            hit_height > float(point[2]) + collision_tolerance_m
            and (
                has_exact_collision_envelope
                or hit_height > target_height_m + collision_tolerance_m
            )
        ):
            envelope_collision_count += 1
    return support_count, sole_collision_count + envelope_collision_count


def _candidate_pose(
    nominal: NominalFootSolePose,
    longitudinal_axis: np.ndarray,
    lateral_axis: np.ndarray,
    longitudinal_m: float,
    lateral_m: float,
    yaw_rad: float,
    vertical_m: float,
) -> NominalFootSolePose:
    offset = np.asarray((longitudinal_m, lateral_m), dtype=np.float64)
    translation_xy = offset[0] * longitudinal_axis + offset[1] * lateral_axis
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    relative_xy = nominal.sole_support_points_world[:, :2] - nominal.sole_center_world[:2]
    points = np.array(nominal.sole_support_points_world, copy=True)
    points[:, :2] = nominal.sole_center_world[:2] + relative_xy @ rotation.T + translation_xy
    points[:, 2] += vertical_m
    center = np.array(nominal.sole_center_world, copy=True)
    center[:2] += translation_xy
    center[2] += vertical_m
    envelope = None
    if nominal.collision_envelope_points_world is not None:
        envelope = np.array(
            nominal.collision_envelope_points_world, copy=True
        )
        envelope_relative_xy = (
            envelope[:, :2] - nominal.sole_center_world[:2]
        )
        envelope[:, :2] = (
            nominal.sole_center_world[:2]
            + envelope_relative_xy @ rotation.T
            + translation_xy
        )
        envelope[:, 2] += vertical_m
    return NominalFootSolePose(
        sole_center_world=center,
        sole_support_points_world=points,
        yaw_rad=nominal.yaw_rad + yaw_rad,
        collision_envelope_points_world=envelope,
    )


def _normalized_square(value: float, limit: float) -> float:
    return 0.0 if limit == 0.0 else (value / limit) ** 2


def _candidate_rank(candidate: _Candidate) -> tuple[float, ...]:
    """Prefer support coverage first, then the smallest deterministic edit."""

    return (
        -float(candidate.support_point_count),
        candidate.cost,
        abs(candidate.longitudinal_m),
        abs(candidate.lateral_m),
        abs(candidate.yaw_rad),
        candidate.longitudinal_m,
        candidate.lateral_m,
        candidate.yaw_rad,
    )


def anchor_planted_foot(
    mesh: TerrainMeshIndex,
    route: StairSupportRoute,
    span: StanceSpan,
    nominal_pose: NominalFootSolePose,
    *,
    target_level_index: int,
    ground_fallback_height_m: float | None = None,
    config: FootholdAnchorConfig | None = None,
) -> FootholdAnchor:
    """Solve one authored stance span against a selected exact mesh tread.

    The local grid is deterministic and includes the unmodified nominal pose.
    It only changes route-longitudinal/lateral position, world yaw, and a
    uniform vertical translation.  The latter is zero whenever the nominal
    sole is already at the target height, preserving source Z and pitch unless
    that shift is necessary for support.
    """

    if not isinstance(mesh, TerrainMeshIndex):
        raise ContractError("mesh must be TerrainMeshIndex")
    if not isinstance(route, StairSupportRoute):
        raise ContractError("route must be StairSupportRoute")
    if not isinstance(span, StanceSpan):
        raise ContractError("span must be StanceSpan")
    if not isinstance(nominal_pose, NominalFootSolePose):
        raise ContractError("nominal_pose must be NominalFootSolePose")
    settings = config if config is not None else FootholdAnchorConfig()
    if not isinstance(settings, FootholdAnchorConfig):
        raise ContractError("config must be FootholdAnchorConfig")
    if not isinstance(target_level_index, (int, np.integer)):
        raise ContractError("target_level_index must be an integer")
    target_index = int(target_level_index)
    if target_index < 0 or target_index >= len(route.levels):
        raise ContractError("target_level_index is outside the stair support route")

    target_height = float(route.levels[target_index].height_m)
    if not math.isfinite(target_height):
        raise ContractError("target tread height must be finite")
    longitudinal_axis, lateral_axis = _route_axes(route)
    route_start = np.asarray(route.start_xy, dtype=np.float64)
    nominal_xy = np.asarray(nominal_pose.sole_center_world[:2], dtype=np.float64)
    nominal_relative = nominal_xy - route_start
    nominal_longitudinal = float(nominal_relative @ longitudinal_axis)
    nominal_lateral = float(nominal_relative @ lateral_axis)
    level = route.levels[target_index]
    side_foothold = (
        level.left_foothold_center_xy
        if span.foot_index == 0
        else level.right_foothold_center_xy
    )
    if side_foothold is not None:
        desired_xy = np.asarray(side_foothold, dtype=np.float64)
        desired_relative = desired_xy - nominal_xy
        desired_longitudinal_adjustment = float(
            desired_relative @ longitudinal_axis
        )
        desired_lateral_adjustment = float(desired_relative @ lateral_axis)
    else:
        desired_longitudinal = float(
            np.clip(
                nominal_longitudinal,
                float(level.route_start_distance_m),
                float(level.route_stop_distance_m),
            )
        )
        desired_lateral = (
            settings.route_lateral_offset_m
            if span.foot_index == 0
            else -settings.route_lateral_offset_m
        )
        desired_longitudinal_adjustment = (
            desired_longitudinal - nominal_longitudinal
        )
        desired_lateral_adjustment = desired_lateral - nominal_lateral
    nominal_collision_envelope = (
        _foot_collision_envelope_points(
            nominal_pose.sole_support_points_world
        )
        if nominal_pose.collision_envelope_points_world is None
        else np.asarray(
            nominal_pose.collision_envelope_points_world,
            dtype=np.float64,
        )
    )
    complete_foot_points = np.concatenate(
        (
            nominal_pose.sole_support_points_world,
            nominal_collision_envelope,
        ),
        axis=0,
    )
    relative_longitudinal = (
        complete_foot_points[:, :2] - nominal_xy
    ) @ longitudinal_axis
    edge_seed_adjustments = (
        float(level.route_start_distance_m)
        + settings.tread_edge_clearance_m
        - float(np.min(relative_longitudinal))
        - nominal_longitudinal,
        float(level.route_stop_distance_m)
        - settings.tread_edge_clearance_m
        - float(np.max(relative_longitudinal))
        - nominal_longitudinal,
    )
    longitudinal_values = _offset_samples(
        settings.max_longitudinal_adjustment_m,
        settings.longitudinal_samples,
        (desired_longitudinal_adjustment,) + edge_seed_adjustments,
    )
    lateral_values = _lateral_samples(
        settings, (desired_lateral_adjustment,)
    )
    fallback = (
        None
        if bool(level.visible_in_mesh)
        else (
            target_height
            if ground_fallback_height_m is None
            else float(ground_fallback_height_m)
        )
    )
    point_count = len(nominal_pose.sole_support_points_world)
    required_support = (
        point_count
        if settings.minimum_support_points is None
        else min(point_count, settings.minimum_support_points)
    )
    nominal_count, nominal_collision_count = _support_and_full_foot_collision_counts(
        mesh,
        nominal_pose.sole_support_points_world,
        target_height_m=target_height,
        ray_origin_height_m=max(
            target_height,
            float(np.max(nominal_pose.sole_support_points_world[:, 2])),
            float(np.max(mesh.vertices_world[:, 2])) if len(mesh.vertices_world) else target_height,
        ) + 1.0,
        support_tolerance_m=settings.support_height_tolerance_m,
        collision_tolerance_m=settings.collision_clearance_m,
        ground_fallback_height_m=fallback,
        collision_envelope_points=nominal_collision_envelope,
    )
    median_height = float(np.median(nominal_pose.sole_support_points_world[:, 2]))
    median_vertical = target_height - median_height
    vertical_values = {
        0.0,
        float(median_vertical),
        *(
            float(target_height - height)
            for height in nominal_pose.sole_support_points_world[:, 2]
        ),
        *(
            float(target_height - height)
            for height in nominal_collision_envelope[:, 2]
            if height < target_height
        ),
    }
    vertical_samples = tuple(
        sorted(
            value
            for value in vertical_values
            if abs(value) <= settings.max_vertical_adjustment_m
        )
    )

    candidate_count = 0
    best_support_count = nominal_count
    least_collision_count = nominal_collision_count
    best: _Candidate | None = None
    if vertical_samples:
        ray_origin_height = max(
            target_height,
            float(
                np.max(nominal_pose.sole_support_points_world[:, 2])
                + max(vertical_samples)
            ),
            float(np.max(mesh.vertices_world[:, 2])) if len(mesh.vertices_world) else target_height,
        ) + 1.0
        for longitudinal_m in longitudinal_values:
            for lateral_m in lateral_values:
                for yaw_m in _samples(settings.max_yaw_adjustment_rad, settings.yaw_samples):
                    for vertical_m in vertical_samples:
                        candidate_count += 1
                        pose = _candidate_pose(
                            nominal_pose,
                            longitudinal_axis,
                            lateral_axis,
                            float(longitudinal_m),
                            float(lateral_m),
                            float(yaw_m),
                            vertical_m,
                        )
                        (
                            support_count,
                            collision_count,
                        ) = _support_and_full_foot_collision_counts(
                            mesh,
                            pose.sole_support_points_world,
                            target_height_m=target_height,
                            ray_origin_height_m=ray_origin_height,
                            support_tolerance_m=settings.support_height_tolerance_m,
                            collision_tolerance_m=settings.collision_clearance_m,
                            ground_fallback_height_m=fallback,
                            collision_envelope_points=(
                                pose.collision_envelope_points_world
                            ),
                        )
                        best_support_count = max(best_support_count, support_count)
                        least_collision_count = min(least_collision_count, collision_count)
                        cost = (
                            _normalized_square(longitudinal_m, settings.max_longitudinal_adjustment_m)
                            + _normalized_square(lateral_m, settings.max_lateral_adjustment_m)
                            + _normalized_square(yaw_m, settings.max_yaw_adjustment_rad)
                            + _normalized_square(vertical_m, settings.max_vertical_adjustment_m)
                            + settings.route_alignment_weight
                            * (
                                _normalized_square(
                                    float(longitudinal_m)
                                    - desired_longitudinal_adjustment,
                                    settings.max_longitudinal_adjustment_m,
                                )
                                + _normalized_square(
                                    float(lateral_m)
                                    - desired_lateral_adjustment,
                                    settings.max_lateral_adjustment_m,
                                )
                            )
                        )
                        candidate = _Candidate(
                            longitudinal_m=float(longitudinal_m),
                            lateral_m=float(lateral_m),
                            yaw_rad=float(yaw_m),
                            vertical_m=vertical_m,
                            support_point_count=support_count,
                            collision_point_count=collision_count,
                            cost=cost,
                            pose=pose,
                        )
                        if collision_count == 0 and support_count >= required_support and (
                            best is None
                            or _candidate_rank(candidate) < _candidate_rank(best)
                        ):
                            best = candidate

    if best is None:
        reason = (
            "vertical alignment exceeds the configured bound"
            if not vertical_samples
            else (
                "every supported candidate collides with a higher target-mesh tread"
                if best_support_count >= required_support and least_collision_count > 0
                else "bounded pose adjustments cannot support the required sole samples"
            )
        )
        diagnostics = FootholdAnchorDiagnostics(
            span=span,
            target_level_index=target_index,
            candidate_count=candidate_count,
            best_support_point_count=best_support_count,
            required_support_point_count=required_support,
            nominal_support_point_count=nominal_count,
            collision_point_count=least_collision_count,
            longitudinal_adjustment_m=0.0,
            lateral_adjustment_m=0.0,
            yaw_adjustment_rad=0.0,
            vertical_adjustment_m=median_vertical,
            cost=None,
            rejection_reason=reason,
        )
        raise FootholdAnchorRejected(diagnostics)

    diagnostics = FootholdAnchorDiagnostics(
        span=span,
        target_level_index=target_index,
        candidate_count=candidate_count,
        best_support_point_count=best.support_point_count,
        required_support_point_count=required_support,
        nominal_support_point_count=nominal_count,
        collision_point_count=best.collision_point_count,
        longitudinal_adjustment_m=best.longitudinal_m,
        lateral_adjustment_m=best.lateral_m,
        yaw_adjustment_rad=best.yaw_rad,
        vertical_adjustment_m=best.vertical_m,
        cost=best.cost,
        rejection_reason=None,
    )
    return FootholdAnchor(
        span=span,
        target_level_index=target_index,
        pose=best.pose,
        diagnostics=diagnostics,
    )


__all__ = (
    "FootholdAnchor",
    "FootholdAnchorConfig",
    "FootholdAnchorDiagnostics",
    "FootholdAnchorRejected",
    "NominalFootSolePose",
    "StanceSpan",
    "anchor_planted_foot",
)
