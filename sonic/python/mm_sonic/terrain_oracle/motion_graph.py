"""Mechanically compatible fragment adjacency and bounded route search.

This module is intentionally a small practical graph layer.  Fragment placement
can absorb root translation and yaw at a seam, but it cannot hide a support-foot,
lower-body pose, or velocity discontinuity.  Those quantities therefore form the
hard edge tests and edge cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .fragments import EndpointSummary, FragmentDescriptor, SupportPhase


# IsaacLab canonical order: legs plus the three waist joints.  Arm posture is
# deliberately not a hard transition gate; reconstruction can inertialize it.
_LOWER_BODY_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18)
_STOP_TAG_HINTS = ("idle", "stand", "stop")


@dataclass(frozen=True)
class TransitionConfig:
    """Thresholds used to admit and rank fragment-to-fragment seams."""

    max_root_height_delta_m: float = 0.18
    max_root_tilt_delta_rad: float = 0.35
    max_joint_position_rmse_rad: float = 0.45
    max_root_linear_velocity_delta_m_s: float = 0.9
    max_root_angular_velocity_delta_rad_s: float = 2.0
    max_joint_velocity_rmse_rad_s: float = 4.0
    max_support_sole_delta_m: float = 0.18
    max_same_support_gap_s: float = 0.8
    max_neighbors_per_fragment: int = 32
    safe_stop_max_planar_speed_m_s: float = 0.25
    safe_stop_max_yaw_rate_rad_s: float = 0.5
    pose_cost_weight: float = 1.0
    velocity_cost_weight: float = 0.7
    sole_cost_weight: float = 1.2

    def __post_init__(self) -> None:
        thresholds = (
            self.max_root_height_delta_m,
            self.max_root_tilt_delta_rad,
            self.max_joint_position_rmse_rad,
            self.max_root_linear_velocity_delta_m_s,
            self.max_root_angular_velocity_delta_rad_s,
            self.max_joint_velocity_rmse_rad_s,
            self.max_support_sole_delta_m,
            self.max_same_support_gap_s,
            self.safe_stop_max_planar_speed_m_s,
            self.safe_stop_max_yaw_rate_rad_s,
        )
        if any(value <= 0.0 for value in thresholds):
            raise ValueError("motion-graph thresholds must be positive")
        if self.max_neighbors_per_fragment < 1:
            raise ValueError("max_neighbors_per_fragment must be positive")


@dataclass(frozen=True)
class FragmentStepConstraint:
    """Optional displacement envelope for one selected contact fragment."""

    min_planar_displacement_m: float = 0.0
    max_planar_displacement_m: float = math.inf
    min_vertical_displacement_m: float = -math.inf
    max_vertical_displacement_m: float = math.inf
    terrain_tag: str | None = None

    def __post_init__(self) -> None:
        if (
            self.min_planar_displacement_m < 0.0
            or self.min_planar_displacement_m
            > self.max_planar_displacement_m
            or self.min_vertical_displacement_m
            > self.max_vertical_displacement_m
        ):
            raise ValueError("invalid fragment step displacement envelope")


@dataclass(frozen=True)
class RouteRequest:
    """One robot-entry-frame target for the bounded sequence search."""

    displacement_local_xyz: tuple[float, float, float]
    yaw_delta_rad: float
    duration_s: float
    terrain_tag: str | None = None
    step_constraints: tuple[FragmentStepConstraint | None, ...] = ()
    start_endpoint: EndpointSummary | None = None


@dataclass(frozen=True)
class SearchConfig:
    """Small deterministic beam-search budget and endpoint tolerances."""

    max_fragments: int = 8
    beam_width: int = 256
    max_expansions: int = 50_000
    max_successors_per_expansion: int = 64
    position_tolerance_m: float = 0.3
    yaw_tolerance_rad: float = 0.35
    duration_tolerance_s: float = 0.75
    position_scale_m: float = 0.5
    yaw_scale_rad: float = 0.5
    duration_scale_s: float = 1.0
    transition_cost_weight: float = 0.25
    terrain_missing_penalty: float = 4.0
    require_terrain_match: bool = True
    require_safe_stop: bool = True

    def __post_init__(self) -> None:
        if (
            self.max_fragments < 1
            or self.beam_width < 1
            or self.max_expansions < 1
            or self.max_successors_per_expansion < 1
        ):
            raise ValueError("search budgets must be positive")
        values = (
            self.position_tolerance_m,
            self.yaw_tolerance_rad,
            self.duration_tolerance_s,
            self.position_scale_m,
            self.yaw_scale_rad,
            self.duration_scale_s,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("search tolerances and scales must be positive")


@dataclass(frozen=True)
class TransitionAssessment:
    compatible: bool
    cost: float
    reason: str | None
    root_height_delta_m: float
    root_tilt_delta_rad: float
    joint_position_rmse_rad: float
    root_linear_velocity_delta_m_s: float
    root_angular_velocity_delta_rad_s: float
    joint_velocity_rmse_rad_s: float
    support_sole_delta_m: float


@dataclass(frozen=True)
class SkippedFragment:
    fragment_id: str
    reason: str


@dataclass(frozen=True)
class TransitionEdge:
    source_id: str
    target_id: str
    cost: float
    pose_cost: float
    velocity_cost: float
    sole_cost: float


@dataclass(frozen=True)
class MotionGraph:
    descriptors: Mapping[str, FragmentDescriptor]
    adjacency: Mapping[str, tuple[TransitionEdge, ...]]
    skipped_fragments: tuple[SkippedFragment, ...]
    safe_stop_fragment_ids: tuple[str, ...]
    transition_config: TransitionConfig

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self.adjacency.values())

    def is_safe_stop(self, fragment_id: str) -> bool:
        return fragment_id in self.safe_stop_fragment_ids


@dataclass(frozen=True)
class UnsupportedReason:
    code: str
    message: str


@dataclass(frozen=True)
class MotionPlan:
    fragment_ids: tuple[str, ...]
    achieved_displacement_local_xyz: tuple[float, float, float]
    achieved_yaw_delta_rad: float
    achieved_duration_s: float
    transition_cost: float
    total_cost: float


@dataclass(frozen=True)
class SearchResult:
    supported: bool
    plan: MotionPlan | None
    reason: UnsupportedReason | None
    expansions: int


@dataclass(frozen=True)
class _EndpointFeature:
    root_height: float
    root_tilt_up: np.ndarray
    root_linear_velocity: np.ndarray
    root_angular_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    sole_position: np.ndarray


@dataclass(frozen=True)
class _SearchState:
    fragment_ids: tuple[str, ...]
    last_fragment_id: str
    position_xyz: tuple[float, float, float]
    yaw: float
    duration: float
    transition_cost: float
    terrain_covered: bool


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_about_z(value: np.ndarray, yaw: float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    result = np.array(source, copy=True)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    result[..., 0] = cosine * source[..., 0] - sine * source[..., 1]
    result[..., 1] = sine * source[..., 0] + cosine * source[..., 1]
    return result


def _quaternion_up_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dtype=np.float64,
    )


def _lower_body(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    indices = tuple(index for index in _LOWER_BODY_JOINT_INDICES if index < len(array))
    return array[np.asarray(indices, dtype=np.int64)]


def _endpoint_feature(endpoint: EndpointSummary) -> _EndpointFeature:
    yaw = _yaw_wxyz(endpoint.root_quaternion_local_wxyz)
    root_position = np.asarray(endpoint.root_position_local_xyz, dtype=np.float64)
    sole = np.asarray(endpoint.sole_position_local_xyz, dtype=np.float64)
    sole_root_local = _rotate_about_z(sole - root_position[None, :], -yaw)
    support_indices = _support_indices(endpoint.phase)
    support_relative_height = (
        -float(np.mean(sole_root_local[np.asarray(support_indices), 2]))
        if support_indices
        else float(endpoint.root_height_m)
    )
    return _EndpointFeature(
        root_height=support_relative_height,
        root_tilt_up=_rotate_about_z(
            _quaternion_up_wxyz(endpoint.root_quaternion_local_wxyz), -yaw
        ),
        root_linear_velocity=_rotate_about_z(
            np.asarray(endpoint.root_linear_velocity_local_xyz, dtype=np.float64),
            -yaw,
        ),
        root_angular_velocity=_rotate_about_z(
            np.asarray(endpoint.root_angular_velocity_local_xyz, dtype=np.float64),
            -yaw,
        ),
        joint_position=_lower_body(endpoint.joint_position),
        joint_velocity=_lower_body(endpoint.joint_velocity),
        sole_position=sole_root_local,
    )


def _angular_rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.arctan2(np.sin(left - right), np.cos(left - right))
    return float(np.sqrt(np.mean(np.square(difference))))


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _support_indices(phase: SupportPhase) -> tuple[int, ...]:
    if phase == SupportPhase.LEFT:
        return (0,)
    if phase == SupportPhase.RIGHT:
        return (1,)
    if phase == SupportPhase.DOUBLE:
        return (0, 1)
    return ()


def _rejected(reason: str, values: tuple[float, ...]) -> TransitionAssessment:
    return TransitionAssessment(False, math.inf, reason, *values)


def assess_transition(
    source: FragmentDescriptor,
    target: FragmentDescriptor,
    config: TransitionConfig = TransitionConfig(),
) -> TransitionAssessment:
    """Compare the source exit with a yaw/translation-aligned target entry."""

    zeros = (0.0,) * 7
    if source.exit.phase != target.entry.phase:
        return _rejected("support_phase_mismatch", zeros)
    source_feature = _endpoint_feature(source.exit)
    target_feature = _endpoint_feature(target.entry)
    if (
        source_feature.joint_position.shape != target_feature.joint_position.shape
        or source_feature.joint_velocity.shape != target_feature.joint_velocity.shape
    ):
        return _rejected("joint_dimension_mismatch", zeros)

    height = abs(source_feature.root_height - target_feature.root_height)
    tilt = math.acos(
        float(
            np.clip(
                np.dot(source_feature.root_tilt_up, target_feature.root_tilt_up),
                -1.0,
                1.0,
            )
        )
    )
    joint_position = _angular_rmse(
        source_feature.joint_position, target_feature.joint_position
    )
    linear_velocity = float(
        np.linalg.norm(
            source_feature.root_linear_velocity
            - target_feature.root_linear_velocity
        )
    )
    angular_velocity = float(
        np.linalg.norm(
            source_feature.root_angular_velocity
            - target_feature.root_angular_velocity
        )
    )
    joint_velocity = _rmse(
        source_feature.joint_velocity, target_feature.joint_velocity
    )
    support_indices = _support_indices(source.exit.phase)
    sole = (
        max(
            float(
                np.linalg.norm(
                    source_feature.sole_position[index]
                    - target_feature.sole_position[index]
                )
            )
            for index in support_indices
        )
        if support_indices
        else 0.0
    )
    values = (
        height,
        tilt,
        joint_position,
        linear_velocity,
        angular_velocity,
        joint_velocity,
        sole,
    )
    thresholds_and_reasons = (
        (height, config.max_root_height_delta_m, "root_height_delta"),
        (tilt, config.max_root_tilt_delta_rad, "root_tilt_delta"),
        (
            joint_position,
            config.max_joint_position_rmse_rad,
            "joint_position_rmse",
        ),
        (
            linear_velocity,
            config.max_root_linear_velocity_delta_m_s,
            "root_linear_velocity_delta",
        ),
        (
            angular_velocity,
            config.max_root_angular_velocity_delta_rad_s,
            "root_angular_velocity_delta",
        ),
        (
            joint_velocity,
            config.max_joint_velocity_rmse_rad_s,
            "joint_velocity_rmse",
        ),
        (sole, config.max_support_sole_delta_m, "support_sole_delta"),
    )
    for value, threshold, reason in thresholds_and_reasons:
        if value > threshold:
            return _rejected(reason, values)

    pose_cost = math.sqrt(
        (
            (height / config.max_root_height_delta_m) ** 2
            + (tilt / config.max_root_tilt_delta_rad) ** 2
            + (joint_position / config.max_joint_position_rmse_rad) ** 2
        )
        / 3.0
    )
    velocity_cost = math.sqrt(
        (
            (linear_velocity / config.max_root_linear_velocity_delta_m_s) ** 2
            + (
                angular_velocity / config.max_root_angular_velocity_delta_rad_s
            )
            ** 2
            + (joint_velocity / config.max_joint_velocity_rmse_rad_s) ** 2
        )
        / 3.0
    )
    sole_cost = sole / config.max_support_sole_delta_m
    total = (
        config.pose_cost_weight * pose_cost
        + config.velocity_cost_weight * velocity_cost
        + config.sole_cost_weight * sole_cost
    )
    return TransitionAssessment(True, total, None, *values)


def _edge_from_assessment(
    source_id: str,
    target_id: str,
    assessment: TransitionAssessment,
    config: TransitionConfig,
) -> TransitionEdge:
    pose_cost = math.sqrt(
        (
            (
                assessment.root_height_delta_m
                / config.max_root_height_delta_m
            )
            ** 2
            + (
                assessment.root_tilt_delta_rad
                / config.max_root_tilt_delta_rad
            )
            ** 2
            + (
                assessment.joint_position_rmse_rad
                / config.max_joint_position_rmse_rad
            )
            ** 2
        )
        / 3.0
    )
    velocity_cost = math.sqrt(
        (
            (
                assessment.root_linear_velocity_delta_m_s
                / config.max_root_linear_velocity_delta_m_s
            )
            ** 2
            + (
                assessment.root_angular_velocity_delta_rad_s
                / config.max_root_angular_velocity_delta_rad_s
            )
            ** 2
            + (
                assessment.joint_velocity_rmse_rad_s
                / config.max_joint_velocity_rmse_rad_s
            )
            ** 2
        )
        / 3.0
    )
    return TransitionEdge(
        source_id=source_id,
        target_id=target_id,
        cost=assessment.cost,
        pose_cost=pose_cost,
        velocity_cost=velocity_cost,
        sole_cost=(
            assessment.support_sole_delta_m / config.max_support_sole_delta_m
        ),
    )


def _pathological_reason(
    descriptor: FragmentDescriptor, config: TransitionConfig
) -> str | None:
    if descriptor.entry.phase != descriptor.exit.phase:
        return None
    if descriptor.duration_s <= config.max_same_support_gap_s:
        return None
    phase = descriptor.entry.phase
    # A genuinely sustained idle may be long and same-support.  The pathology is
    # the long unclassified/intermittent gap between two observations of a phase.
    if any(span.phase != phase for span in descriptor.contact_phases):
        return "pathological_long_same_support_gap"
    return None


def _has_stop_tag(descriptor: FragmentDescriptor) -> bool:
    tags = tuple(tag.lower() for tag in descriptor.action_tags)
    return any(hint in tag for tag in tags for hint in _STOP_TAG_HINTS)


def _is_safe_stop(
    descriptor: FragmentDescriptor, config: TransitionConfig
) -> bool:
    if descriptor.exit.phase != SupportPhase.DOUBLE:
        return False
    exit_feature = _endpoint_feature(descriptor.exit)
    if (
        float(np.linalg.norm(exit_feature.root_linear_velocity[:2]))
        > config.safe_stop_max_planar_speed_m_s
        or abs(float(exit_feature.root_angular_velocity[2]))
        > config.safe_stop_max_yaw_rate_rad_s
    ):
        return False
    if _has_stop_tag(descriptor):
        return True
    planar_distance = math.hypot(
        descriptor.root_displacement_local_xyz[0],
        descriptor.root_displacement_local_xyz[1],
    )
    mean_speed = planar_distance / max(descriptor.duration_s, 1.0e-9)
    return mean_speed <= config.safe_stop_max_planar_speed_m_s


def build_compatibility_graph(
    descriptors: Sequence[FragmentDescriptor],
    config: TransitionConfig = TransitionConfig(),
) -> MotionGraph:
    """Build bounded deterministic adjacency from descriptor seam mechanics."""

    sorted_descriptors = sorted(descriptors, key=lambda item: item.fragment_id)
    usable: dict[str, FragmentDescriptor] = {}
    skipped: list[SkippedFragment] = []
    for descriptor in sorted_descriptors:
        reason = _pathological_reason(descriptor, config)
        if reason is not None:
            skipped.append(SkippedFragment(descriptor.fragment_id, reason))
            continue
        usable[descriptor.fragment_id] = descriptor

    by_entry_phase: dict[SupportPhase, list[FragmentDescriptor]] = {
        phase: [] for phase in SupportPhase
    }
    for descriptor in usable.values():
        by_entry_phase[descriptor.entry.phase].append(descriptor)

    entry_features = {
        fragment_id: _endpoint_feature(descriptor.entry)
        for fragment_id, descriptor in usable.items()
    }
    exit_features = {
        fragment_id: _endpoint_feature(descriptor.exit)
        for fragment_id, descriptor in usable.items()
    }
    adjacency: dict[str, tuple[TransitionEdge, ...]] = {}
    for source_id, source in usable.items():
        targets = by_entry_phase[source.exit.phase]
        if not targets:
            adjacency[source_id] = ()
            continue
        target_ids = tuple(target.fragment_id for target in targets)
        target_features = tuple(entry_features[target_id] for target_id in target_ids)
        source_feature = exit_features[source_id]
        heights = np.asarray(
            [feature.root_height for feature in target_features], dtype=np.float64
        )
        tilts = np.stack(
            [feature.root_tilt_up for feature in target_features], axis=0
        )
        root_linear_velocities = np.stack(
            [feature.root_linear_velocity for feature in target_features], axis=0
        )
        root_angular_velocities = np.stack(
            [feature.root_angular_velocity for feature in target_features], axis=0
        )
        joint_positions = np.stack(
            [feature.joint_position for feature in target_features], axis=0
        )
        joint_velocities = np.stack(
            [feature.joint_velocity for feature in target_features], axis=0
        )
        soles = np.stack(
            [feature.sole_position for feature in target_features], axis=0
        )

        height = np.abs(heights - source_feature.root_height)
        tilt = np.arccos(
            np.clip(tilts @ source_feature.root_tilt_up, -1.0, 1.0)
        )
        joint_position_difference = np.arctan2(
            np.sin(joint_positions - source_feature.joint_position[None, :]),
            np.cos(joint_positions - source_feature.joint_position[None, :]),
        )
        joint_position = np.sqrt(
            np.mean(np.square(joint_position_difference), axis=1)
        )
        linear_velocity = np.linalg.norm(
            root_linear_velocities
            - source_feature.root_linear_velocity[None, :],
            axis=1,
        )
        angular_velocity = np.linalg.norm(
            root_angular_velocities
            - source_feature.root_angular_velocity[None, :],
            axis=1,
        )
        joint_velocity = np.sqrt(
            np.mean(
                np.square(
                    joint_velocities - source_feature.joint_velocity[None, :]
                ),
                axis=1,
            )
        )
        support_indices = _support_indices(source.exit.phase)
        sole = (
            np.max(
                np.linalg.norm(
                    soles[:, np.asarray(support_indices), :]
                    - source_feature.sole_position[
                        np.asarray(support_indices), :
                    ][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            if support_indices
            else np.zeros(len(target_ids), dtype=np.float64)
        )
        compatible = (
            (height <= config.max_root_height_delta_m)
            & (tilt <= config.max_root_tilt_delta_rad)
            & (joint_position <= config.max_joint_position_rmse_rad)
            & (
                linear_velocity
                <= config.max_root_linear_velocity_delta_m_s
            )
            & (
                angular_velocity
                <= config.max_root_angular_velocity_delta_rad_s
            )
            & (joint_velocity <= config.max_joint_velocity_rmse_rad_s)
            & (sole <= config.max_support_sole_delta_m)
        )
        pose_cost = np.sqrt(
            (
                np.square(height / config.max_root_height_delta_m)
                + np.square(tilt / config.max_root_tilt_delta_rad)
                + np.square(
                    joint_position / config.max_joint_position_rmse_rad
                )
            )
            / 3.0
        )
        velocity_cost = np.sqrt(
            (
                np.square(
                    linear_velocity
                    / config.max_root_linear_velocity_delta_m_s
                )
                + np.square(
                    angular_velocity
                    / config.max_root_angular_velocity_delta_rad_s
                )
                + np.square(
                    joint_velocity / config.max_joint_velocity_rmse_rad_s
                )
            )
            / 3.0
        )
        sole_cost = sole / config.max_support_sole_delta_m
        total_cost = (
            config.pose_cost_weight * pose_cost
            + config.velocity_cost_weight * velocity_cost
            + config.sole_cost_weight * sole_cost
        )
        compatible_indices = np.flatnonzero(compatible)
        ordered_indices = sorted(
            (int(index) for index in compatible_indices),
            key=lambda index: (float(total_cost[index]), target_ids[index]),
        )
        kept_indices = ordered_indices[: config.max_neighbors_per_fragment]
        candidates: list[TransitionEdge] = []
        for index in kept_indices:
            candidates.append(
                TransitionEdge(
                    source_id=source_id,
                    target_id=target_ids[index],
                    cost=float(total_cost[index]),
                    pose_cost=float(pose_cost[index]),
                    velocity_cost=float(velocity_cost[index]),
                    sole_cost=float(sole_cost[index]),
                )
            )
        adjacency[source_id] = tuple(candidates)

    safe_stop_ids = tuple(
        fragment_id
        for fragment_id, descriptor in usable.items()
        if _is_safe_stop(descriptor, config)
    )
    return MotionGraph(
        descriptors=usable,
        adjacency=adjacency,
        skipped_fragments=tuple(skipped),
        safe_stop_fragment_ids=safe_stop_ids,
        transition_config=config,
    )


def _normalize_tag(tag: str) -> str:
    return "".join(character for character in tag.lower() if character.isalnum())


def _matches_terrain(descriptor: FragmentDescriptor, requested: str | None) -> bool:
    if requested is None:
        return True
    wanted = _normalize_tag(requested)
    available = tuple(_normalize_tag(tag) for tag in descriptor.terrain_tags)
    if wanted == "terrain":
        return any(tag != "flat" for tag in available)
    return any(wanted in tag or tag in wanted for tag in available)


def _matches_step_constraint(
    descriptor: FragmentDescriptor,
    constraint: FragmentStepConstraint | None,
) -> bool:
    if constraint is None:
        return True
    dx, dy, dz = descriptor.root_displacement_local_xyz
    planar = math.hypot(dx, dy)
    return (
        constraint.min_planar_displacement_m
        <= planar
        <= constraint.max_planar_displacement_m
        and constraint.min_vertical_displacement_m
        <= dz
        <= constraint.max_vertical_displacement_m
        and _matches_terrain(descriptor, constraint.terrain_tag)
    )


def _virtual_source(endpoint: EndpointSummary) -> FragmentDescriptor:
    return FragmentDescriptor(
        fragment_id="__request_start__",
        root_displacement_local_xyz=(0.0, 0.0, 0.0),
        root_yaw_delta_rad=0.0,
        duration_s=0.0,
        entry=endpoint,
        exit=endpoint,
        contact_phases=(),
        terrain_tags=(),
        action_tags=(),
    )


def _request_successors(
    graph: MotionGraph,
    state: _SearchState,
    request: RouteRequest,
    search_config: SearchConfig,
    source_id: str,
    constraint: FragmentStepConstraint | None,
    *,
    required_terrain: str | None,
    require_safe_stop: bool,
    remaining_duration_s: float,
    limit: int,
) -> tuple[TransitionEdge, ...]:
    """Rank a bounded request-specific successor set without using stored truncation."""

    source = graph.descriptors[source_id]
    candidates: list[TransitionEdge] = []
    for target_id, target in graph.descriptors.items():
        if (
            target.entry.phase != source.exit.phase
            or target.duration_s > remaining_duration_s
            or not _matches_step_constraint(target, constraint)
            or (
                required_terrain is not None
                and not _matches_terrain(target, required_terrain)
            )
            or (require_safe_stop and not graph.is_safe_stop(target_id))
        ):
            continue
        assessment = assess_transition(
            source, target, graph.transition_config
        )
        if assessment.compatible:
            candidates.append(
                _edge_from_assessment(
                    source_id,
                    target_id,
                    assessment,
                    graph.transition_config,
                )
            )
    def rank(edge: TransitionEdge) -> tuple[int, float, str]:
        target = graph.descriptors[edge.target_id]
        next_state = _append_fragment(
            state, target, edge.cost, request.terrain_tag
        )
        completes_request = (
            search_config.require_safe_stop
            and graph.is_safe_stop(edge.target_id)
            and _is_complete(
                graph, next_state, request, search_config
            )
        )
        return (0 if completes_request else 1, edge.cost, edge.target_id)

    candidates.sort(key=rank)
    return tuple(candidates[:limit])


def _append_fragment(
    state: _SearchState | None,
    descriptor: FragmentDescriptor,
    transition_cost: float,
    requested_terrain: str | None,
) -> _SearchState:
    if state is None:
        position = (0.0, 0.0, 0.0)
        yaw = 0.0
        duration = 0.0
        fragment_ids: tuple[str, ...] = ()
        accumulated_transition_cost = 0.0
        terrain_covered = False
    else:
        position = state.position_xyz
        yaw = state.yaw
        duration = state.duration
        fragment_ids = state.fragment_ids
        accumulated_transition_cost = state.transition_cost
        terrain_covered = state.terrain_covered
    displacement = descriptor.root_displacement_local_xyz
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    position = (
        position[0] + cosine * displacement[0] - sine * displacement[1],
        position[1] + sine * displacement[0] + cosine * displacement[1],
        position[2] + displacement[2],
    )
    return _SearchState(
        fragment_ids=fragment_ids + (descriptor.fragment_id,),
        last_fragment_id=descriptor.fragment_id,
        position_xyz=position,
        yaw=_wrap_angle(yaw + descriptor.root_yaw_delta_rad),
        duration=duration + descriptor.duration_s,
        transition_cost=accumulated_transition_cost + transition_cost,
        terrain_covered=terrain_covered
        or _matches_terrain(descriptor, requested_terrain),
    )


def _endpoint_errors(
    state: _SearchState, request: RouteRequest
) -> tuple[float, float, float]:
    position = float(
        np.linalg.norm(
            np.asarray(state.position_xyz, dtype=np.float64)
            - np.asarray(request.displacement_local_xyz, dtype=np.float64)
        )
    )
    yaw = abs(_wrap_angle(state.yaw - request.yaw_delta_rad))
    duration = abs(state.duration - request.duration_s)
    return position, yaw, duration


def _state_score(
    state: _SearchState, request: RouteRequest, config: SearchConfig
) -> float:
    position, yaw, duration = _endpoint_errors(state, request)
    terrain_penalty = (
        config.terrain_missing_penalty
        if request.terrain_tag is not None and not state.terrain_covered
        else 0.0
    )
    return (
        (position / config.position_scale_m) ** 2
        + (yaw / config.yaw_scale_rad) ** 2
        + (duration / config.duration_scale_s) ** 2
        + config.transition_cost_weight * state.transition_cost
        + terrain_penalty
    )


def _is_complete(
    graph: MotionGraph,
    state: _SearchState,
    request: RouteRequest,
    config: SearchConfig,
) -> bool:
    if request.step_constraints and len(state.fragment_ids) != len(
        request.step_constraints
    ):
        return False
    position, yaw, duration = _endpoint_errors(state, request)
    if (
        position > config.position_tolerance_m
        or yaw > config.yaw_tolerance_rad
        or duration > config.duration_tolerance_s
    ):
        return False
    if (
        config.require_terrain_match
        and request.terrain_tag is not None
        and not state.terrain_covered
    ):
        return False
    return not config.require_safe_stop or graph.is_safe_stop(
        state.last_fragment_id
    )


def search_complete_route(
    graph: MotionGraph,
    request: RouteRequest,
    config: SearchConfig = SearchConfig(),
) -> SearchResult:
    """Return the best complete bounded route, never a matching dead-end prefix."""

    if not graph.descriptors:
        return SearchResult(
            False,
            None,
            UnsupportedReason("empty_graph", "no usable fragments"),
            0,
        )
    if config.require_safe_stop and not graph.safe_stop_fragment_ids:
        return SearchResult(
            False,
            None,
            UnsupportedReason(
                "no_safe_terminal_stop",
                "the graph contains no low-velocity double-support stop fragment",
            ),
            0,
        )

    virtual_source = (
        _virtual_source(request.start_endpoint)
        if request.start_endpoint is not None
        else None
    )
    initial: list[_SearchState] = []
    for descriptor in graph.descriptors.values():
        if (
            descriptor.duration_s
            > request.duration_s + config.duration_tolerance_s
            or not _matches_step_constraint(
                descriptor,
                request.step_constraints[0]
                if request.step_constraints
                else None,
            )
        ):
            continue
        transition_cost = 0.0
        if virtual_source is not None:
            assessment = assess_transition(
                virtual_source, descriptor, graph.transition_config
            )
            if not assessment.compatible:
                continue
            transition_cost = assessment.cost
        initial.append(
            _append_fragment(
                None, descriptor, transition_cost, request.terrain_tag
            )
        )
    initial.sort(
        key=lambda state: (
            _state_score(state, request, config),
            state.fragment_ids,
        )
    )
    frontier = initial[: config.beam_width]
    best: _SearchState | None = None
    best_key: tuple[float, tuple[str, ...]] | None = None
    expansions = 0
    successor_cache: dict[
        tuple[object, ...], tuple[TransitionEdge, ...]
    ] = {}

    maximum_depth = (
        min(config.max_fragments, len(request.step_constraints))
        if request.step_constraints
        else config.max_fragments
    )
    for depth in range(1, maximum_depth + 1):
        for state in frontier:
            if not _is_complete(graph, state, request, config):
                continue
            key = (_state_score(state, request, config), state.fragment_ids)
            if best_key is None or key < best_key:
                best = state
                best_key = key
        if depth == maximum_depth or expansions >= config.max_expansions:
            break

        expanded: list[_SearchState] = []
        for state in frontier:
            constraint = (
                request.step_constraints[depth]
                if request.step_constraints
                else None
            )
            terminal_expansion = depth + 1 == maximum_depth
            required_terrain = (
                request.terrain_tag
                if (
                    terminal_expansion
                    and config.require_terrain_match
                    and request.terrain_tag is not None
                    and not state.terrain_covered
                )
                else None
            )
            remaining_duration = (
                request.duration_s
                + config.duration_tolerance_s
                - state.duration
            )
            cache_key = (
                state.last_fragment_id,
                state.position_xyz,
                round(state.yaw, 9),
                round(state.duration, 9),
                state.terrain_covered,
                len(state.fragment_ids),
                constraint,
                required_terrain,
                terminal_expansion and config.require_safe_stop,
                round(remaining_duration, 9),
            )
            edges = successor_cache.get(cache_key)
            if edges is None:
                edges = _request_successors(
                    graph,
                    state,
                    request,
                    config,
                    state.last_fragment_id,
                    constraint,
                    required_terrain=required_terrain,
                    require_safe_stop=(
                        terminal_expansion and config.require_safe_stop
                    ),
                    remaining_duration_s=remaining_duration,
                    limit=config.max_successors_per_expansion,
                )
                successor_cache[cache_key] = edges
            for edge in edges:
                if expansions >= config.max_expansions:
                    break
                descriptor = graph.descriptors[edge.target_id]
                next_state = _append_fragment(
                    state, descriptor, edge.cost, request.terrain_tag
                )
                expansions += 1
                if (
                    next_state.duration
                    > request.duration_s + config.duration_tolerance_s
                ):
                    continue
                expanded.append(next_state)
            if expansions >= config.max_expansions:
                break
        expanded.sort(
            key=lambda state: (
                _state_score(state, request, config),
                state.fragment_ids,
            )
        )
        frontier = expanded[: config.beam_width]
        if not frontier:
            break

    if best is None:
        code = (
            "no_safe_terminal_stop"
            if config.require_safe_stop
            else "no_complete_route"
        )
        return SearchResult(
            False,
            None,
            UnsupportedReason(
                code,
                "bounded search found no complete route within endpoint tolerances",
            ),
            expansions,
        )
    plan = MotionPlan(
        fragment_ids=best.fragment_ids,
        achieved_displacement_local_xyz=best.position_xyz,
        achieved_yaw_delta_rad=best.yaw,
        achieved_duration_s=best.duration,
        transition_cost=best.transition_cost,
        total_cost=_state_score(best, request, config),
    )
    return SearchResult(True, plan, None, expansions)


__all__ = (
    "FragmentStepConstraint",
    "MotionGraph",
    "MotionPlan",
    "RouteRequest",
    "SearchConfig",
    "SearchResult",
    "SkippedFragment",
    "TransitionAssessment",
    "TransitionConfig",
    "TransitionEdge",
    "UnsupportedReason",
    "assess_transition",
    "build_compatibility_graph",
    "search_complete_route",
)
