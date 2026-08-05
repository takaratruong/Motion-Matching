"""Globally select one semantic motion fragment for each sampled stair riser."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .fragments import (
    ContactPhaseSpan,
    EndpointSummary,
    FragmentDescriptor,
    SupportPhase,
)
from .motion_graph import TransitionConfig, assess_transition
from .stair_support_route import StairSupportRoute, VisibleTread
from .stairs500_fragments import FragmentRole, StairFragmentRecord


@dataclass(frozen=True)
class StairFragmentSequenceConfig:
    """Geometry envelopes and seam weighting for stair-fragment retrieval."""

    max_rise_error_m: float = 0.08
    max_run_error_m: float = 0.15
    geometry_cost_weight: float = 1.0
    seam_cost_weight: float = 1.0
    require_boundary_roles: bool = True
    excluded_archive_clip_indices: frozenset[int] = frozenset()
    transition_config: TransitionConfig = TransitionConfig()

    def __post_init__(self) -> None:
        if self.max_rise_error_m <= 0.0 or self.max_run_error_m <= 0.0:
            raise ValueError("stair geometry error limits must be positive")
        if self.geometry_cost_weight < 0.0 or self.seam_cost_weight < 0.0:
            raise ValueError("stair sequence cost weights must be nonnegative")


@dataclass(frozen=True)
class StairFragmentTarget:
    """One exact consecutive-level transition and its sampled footholds."""

    transition_index: int
    source_level_index: int
    target_level_index: int
    compatible_roles: tuple[FragmentRole, ...]
    source_height_m: float
    target_height_m: float
    rise_m: float
    run_m: float
    entry_run_m: float | None
    exit_run_m: float | None
    source_left_foothold_center_xy: np.ndarray | None
    source_right_foothold_center_xy: np.ndarray | None
    target_left_foothold_center_xy: np.ndarray | None
    target_right_foothold_center_xy: np.ndarray | None


@dataclass(frozen=True)
class StairFragmentRejection:
    """Why one record or graph edge was excluded from a target layer."""

    target_transition_index: int
    fragment_id: str
    reason: str
    source_fragment_id: str | None = None


@dataclass(frozen=True)
class StairFragmentSequenceDiagnostics:
    rejections: tuple[StairFragmentRejection, ...]


@dataclass(frozen=True)
class StairFragmentSequenceReason:
    code: str
    message: str
    target_transition_index: int | None


@dataclass(frozen=True)
class StairFragmentSelection:
    target: StairFragmentTarget
    fragment: StairFragmentRecord
    matched_run_m: float
    geometry_cost: float
    seam_cost: float


@dataclass(frozen=True)
class StairFragmentSequencePlan:
    supported: bool
    selections: tuple[StairFragmentSelection, ...]
    total_geometry_cost: float
    total_seam_cost: float
    total_cost: float
    diagnostics: StairFragmentSequenceDiagnostics
    reason: StairFragmentSequenceReason | None


@dataclass(frozen=True)
class _Candidate:
    fragment: StairFragmentRecord
    matched_run_m: float
    geometry_cost: float


@dataclass(frozen=True)
class _Path:
    candidates: tuple[_Candidate, ...]
    seam_costs: tuple[float, ...]
    geometry_cost: float
    seam_cost: float
    weighted_cost: float

    @property
    def fragment_ids(self) -> tuple[str, ...]:
        return tuple(item.fragment.fragment_id for item in self.candidates)


def _float_tuple(value: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


def _endpoint_from_dict(row: dict[str, Any]) -> EndpointSummary:
    return EndpointSummary(
        phase=SupportPhase(str(row["phase"])),
        root_position_local_xyz=_float_tuple(row["root_position_local_xyz"]),
        root_quaternion_local_wxyz=_float_tuple(
            row["root_quaternion_local_wxyz"]
        ),
        root_height_m=float(row["root_height_m"]),
        root_linear_velocity_local_xyz=_float_tuple(
            row["root_linear_velocity_local_xyz"]
        ),
        root_angular_velocity_local_xyz=_float_tuple(
            row["root_angular_velocity_local_xyz"]
        ),
        joint_position=_float_tuple(row["joint_position"]),
        joint_velocity=_float_tuple(row["joint_velocity"]),
        sole_position_local_xyz=tuple(
            _float_tuple(value)
            for value in row["sole_position_local_xyz"]
        ),
    )


def _descriptor_from_dict(row: dict[str, Any]) -> FragmentDescriptor:
    return FragmentDescriptor(
        fragment_id=str(row["fragment_id"]),
        root_displacement_local_xyz=_float_tuple(
            row["root_displacement_local_xyz"]
        ),
        root_yaw_delta_rad=float(row["root_yaw_delta_rad"]),
        duration_s=float(row["duration_s"]),
        entry=_endpoint_from_dict(row["entry"]),
        exit=_endpoint_from_dict(row["exit"]),
        contact_phases=tuple(
            ContactPhaseSpan(
                SupportPhase(str(value["phase"])),
                int(value["start_offset"]),
                int(value["stop_offset"]),
            )
            for value in row["contact_phases"]
        ),
        terrain_tags=tuple(str(value) for value in row["terrain_tags"]),
        action_tags=tuple(str(value) for value in row["action_tags"]),
    )


def _record_from_dict(row: dict[str, Any]) -> StairFragmentRecord:
    source_start, source_stop = row["source_range"]
    context_start, context_stop = row["context_range"]
    return StairFragmentRecord(
        fragment_id=str(row["fragment_id"]),
        archive_clip_index=int(row["archive_clip_index"]),
        source_clip_name=str(row["source_clip_name"]),
        source_start_frame=int(source_start),
        source_stop_frame=int(source_stop),
        context_start_frame=int(context_start),
        context_stop_frame=int(context_stop),
        entry_phase=SupportPhase(str(row["entry_phase"])),
        exit_phase=SupportPhase(str(row["exit_phase"])),
        role=FragmentRole(str(row["role"])),
        segmentation_basis=str(row["segmentation_basis"]),
        start_tread_index=(
            None
            if row["start_tread_index"] is None
            else int(row["start_tread_index"])
        ),
        end_tread_index=(
            None
            if row["end_tread_index"] is None
            else int(row["end_tread_index"])
        ),
        riser_transition_count=int(row["riser_transition_count"]),
        entry_contact_confirmation=float(row["entry_contact_confirmation"]),
        exit_contact_confirmation=float(row["exit_contact_confirmation"]),
        entry_kinematic_support_evidence=bool(
            row["entry_kinematic_support_evidence"]
        ),
        exit_kinematic_support_evidence=bool(
            row["exit_kinematic_support_evidence"]
        ),
        source_direction=str(row["source_direction"]),
        source_rise_m=float(row["source_rise_m"]),
        source_tread_m=float(row["source_tread_m"]),
        source_step_count=int(row["source_step_count"]),
        duration_s=float(row["duration_s"]),
        root_displacement_local_xyz=_float_tuple(
            row["root_displacement_local_xyz"]
        ),
        terrain_displacement_local_xyz=_float_tuple(
            row["terrain_displacement_local_xyz"]
        ),
        vertical_delta_m=float(row["vertical_delta_m"]),
        approximate_run_m=float(row["approximate_run_m"]),
        approximate_lateral_m=float(row["approximate_lateral_m"]),
        approximate_yaw_rad=float(row["approximate_yaw_rad"]),
        start_speed_local_xyz=_float_tuple(
            row["start_speed_local_xyz"]
        ),
        end_speed_local_xyz=_float_tuple(
            row["end_speed_local_xyz"]
        ),
        start_yaw_rate_rad_s=float(row["start_yaw_rate_rad_s"]),
        end_yaw_rate_rad_s=float(row["end_yaw_rate_rad_s"]),
        source_robot_path=str(row["source_robot_path"]),
        terrain_usd_path=str(row["terrain_usd_path"]),
        descriptor=_descriptor_from_dict(row["descriptor"]),
    )


def load_stair_fragment_records_jsonl(
    path: str | Path,
) -> tuple[StairFragmentRecord, ...]:
    """Load fragment-bank JSONL rows into motion-graph-ready records."""

    with Path(path).open() as stream:
        return tuple(
            _record_from_dict(json.loads(line))
            for line in stream
            if line.strip()
        )


def _level_center_distance(level: VisibleTread) -> float:
    return 0.5 * (
        float(level.route_start_distance_m)
        + float(level.route_stop_distance_m)
    )


def _compatible_roles(
    transition_index: int,
    transition_count: int,
    *,
    require_boundary_roles: bool,
) -> tuple[FragmentRole, ...]:
    if transition_count == 1:
        return (
            FragmentRole.ENTRY,
            FragmentRole.MIDDLE,
            FragmentRole.EXIT,
        )
    if transition_index == 0:
        return (
            (FragmentRole.ENTRY,)
            if require_boundary_roles
            else (FragmentRole.ENTRY, FragmentRole.MIDDLE)
        )
    if transition_index == transition_count - 1:
        return (
            (FragmentRole.EXIT,)
            if require_boundary_roles
            else (FragmentRole.MIDDLE, FragmentRole.EXIT)
        )
    return (FragmentRole.MIDDLE,)


def _targets(
    route: StairSupportRoute,
    *,
    require_boundary_roles: bool,
) -> tuple[StairFragmentTarget, ...]:
    transition_count = max(0, len(route.levels) - 1)
    route_length = float(
        np.linalg.norm(
            np.asarray(route.end_xy, dtype=np.float64)
            - np.asarray(route.start_xy, dtype=np.float64)
        )
    )
    targets: list[StairFragmentTarget] = []
    for index, (source, target) in enumerate(
        zip(route.levels, route.levels[1:])
    ):
        targets.append(
            StairFragmentTarget(
                transition_index=index,
                source_level_index=index,
                target_level_index=index + 1,
                compatible_roles=_compatible_roles(
                    index,
                    transition_count,
                    require_boundary_roles=require_boundary_roles,
                ),
                source_height_m=float(source.height_m),
                target_height_m=float(target.height_m),
                rise_m=float(target.height_m) - float(source.height_m),
                run_m=(
                    _level_center_distance(target)
                    - _level_center_distance(source)
                ),
                entry_run_m=(
                    _level_center_distance(target) if index == 0 else None
                ),
                exit_run_m=(
                    route_length - _level_center_distance(source)
                    if index == transition_count - 1
                    else None
                ),
                source_left_foothold_center_xy=(
                    source.left_foothold_center_xy
                ),
                source_right_foothold_center_xy=(
                    source.right_foothold_center_xy
                ),
                target_left_foothold_center_xy=(
                    target.left_foothold_center_xy
                ),
                target_right_foothold_center_xy=(
                    target.right_foothold_center_xy
                ),
            )
        )
    return tuple(targets)


def _direction(rise_m: float) -> str | None:
    if rise_m > 1.0e-9:
        return "up"
    if rise_m < -1.0e-9:
        return "down"
    return None


def _source_rise(record: StairFragmentRecord) -> float:
    return (
        -float(record.source_rise_m)
        if record.source_direction == "down"
        else float(record.source_rise_m)
    )


def _target_run(
    target: StairFragmentTarget, role: FragmentRole
) -> float:
    if role is FragmentRole.ENTRY and target.entry_run_m is not None:
        return target.entry_run_m
    if role is FragmentRole.EXIT and target.exit_run_m is not None:
        return target.exit_run_m
    return target.run_m


def _candidate(
    target: StairFragmentTarget,
    record: StairFragmentRecord,
    config: StairFragmentSequenceConfig,
) -> tuple[_Candidate | None, str | None]:
    if record.archive_clip_index in config.excluded_archive_clip_indices:
        return None, "source_clip_excluded"
    direction = _direction(target.rise_m)
    if direction is None:
        return None, "zero_rise"
    if record.source_direction != direction:
        return None, "direction_mismatch"
    if record.role not in target.compatible_roles:
        return None, "role_mismatch"
    if record.riser_transition_count != 1:
        return None, "riser_count_mismatch"
    rise_error = abs(_source_rise(record) - target.rise_m)
    target_run = _target_run(target, record.role)
    run_error = abs(float(record.approximate_run_m) - target_run)
    if (
        rise_error > config.max_rise_error_m
        or run_error > config.max_run_error_m
    ):
        return None, "geometry_mismatch"
    cost = math.hypot(
        rise_error / config.max_rise_error_m,
        run_error / config.max_run_error_m,
    )
    return _Candidate(record, target_run, cost), None


def _unsupported(
    code: str,
    message: str,
    transition_index: int | None,
    rejections: list[StairFragmentRejection],
) -> StairFragmentSequencePlan:
    return StairFragmentSequencePlan(
        supported=False,
        selections=(),
        total_geometry_cost=math.inf,
        total_seam_cost=math.inf,
        total_cost=math.inf,
        diagnostics=StairFragmentSequenceDiagnostics(tuple(rejections)),
        reason=StairFragmentSequenceReason(
            code=code,
            message=message,
            target_transition_index=transition_index,
        ),
    )


def plan_stair_fragment_sequences(
    route: StairSupportRoute,
    fragments: Sequence[StairFragmentRecord],
    config: StairFragmentSequenceConfig = StairFragmentSequenceConfig(),
    *,
    maximum_plans: int = 8,
    maximum_paths_per_layer: int | None = None,
) -> tuple[StairFragmentSequencePlan, ...]:
    """Return ranked complete paths, optionally using a bounded global beam."""

    if maximum_plans < 1:
        raise ValueError("maximum_plans must be positive")
    if (
        maximum_paths_per_layer is not None
        and maximum_paths_per_layer < maximum_plans
    ):
        raise ValueError("path beam must be at least maximum_plans")
    targets = _targets(
        route,
        require_boundary_roles=config.require_boundary_roles,
    )
    rejections: list[StairFragmentRejection] = []
    if not targets:
        return (
            _unsupported(
                "no_risers",
                "stair support route has fewer than two terrain levels",
                None,
                rejections,
            ),
        )

    layers: list[tuple[_Candidate, ...]] = []
    ordered_fragments = tuple(
        sorted(fragments, key=lambda item: item.fragment_id)
    )
    for target in targets:
        accepted: list[_Candidate] = []
        for fragment in ordered_fragments:
            candidate, reason = _candidate(target, fragment, config)
            if candidate is not None:
                accepted.append(candidate)
            else:
                rejections.append(
                    StairFragmentRejection(
                        target.transition_index,
                        fragment.fragment_id,
                        str(reason),
                    )
                )
        if not accepted:
            return (
                _unsupported(
                    "no_candidate",
                    (
                        "no fragment matches direction, semantic role, and geometry "
                        f"for target transition {target.transition_index}"
                    ),
                    target.transition_index,
                    rejections,
                ),
            )
        layers.append(tuple(accepted))

    paths = tuple(
        _Path(
            candidates=(candidate,),
            seam_costs=(0.0,),
            geometry_cost=candidate.geometry_cost,
            seam_cost=0.0,
            weighted_cost=(
                config.geometry_cost_weight * candidate.geometry_cost
            ),
        )
        for candidate in layers[0]
    )
    if maximum_paths_per_layer is not None:
        paths = tuple(
            sorted(
                paths,
                key=lambda path: (
                    path.weighted_cost,
                    path.fragment_ids,
                ),
            )[:maximum_paths_per_layer]
        )
    assessment_cache: dict[tuple[str, str], object] = {}
    rejected_edges: set[tuple[int, str, str]] = set()
    for target, candidates in zip(targets[1:], layers[1:]):
        next_paths: list[_Path] = []
        for candidate in candidates:
            proposals: list[_Path] = []
            for path in paths:
                source = path.candidates[-1].fragment
                key = (source.fragment_id, candidate.fragment.fragment_id)
                assessment = assessment_cache.get(key)
                if assessment is None:
                    assessment = assess_transition(
                        source.descriptor,
                        candidate.fragment.descriptor,
                        config.transition_config,
                    )
                    assessment_cache[key] = assessment
                if not assessment.compatible:
                    rejection_key = (
                        target.transition_index,
                        source.fragment_id,
                        candidate.fragment.fragment_id,
                    )
                    if rejection_key not in rejected_edges:
                        rejected_edges.add(rejection_key)
                        rejections.append(
                            StairFragmentRejection(
                                target.transition_index,
                                candidate.fragment.fragment_id,
                                f"seam_{assessment.reason}",
                                source_fragment_id=source.fragment_id,
                            )
                        )
                    continue
                geometry_cost = (
                    path.geometry_cost + candidate.geometry_cost
                )
                seam_cost = path.seam_cost + assessment.cost
                proposal = _Path(
                    candidates=path.candidates + (candidate,),
                    seam_costs=path.seam_costs + (assessment.cost,),
                    geometry_cost=geometry_cost,
                    seam_cost=seam_cost,
                    weighted_cost=(
                        config.geometry_cost_weight * geometry_cost
                        + config.seam_cost_weight * seam_cost
                    ),
                )
                proposals.append(proposal)
            next_paths.extend(
                sorted(
                    proposals,
                    key=lambda path: (
                        path.weighted_cost,
                        path.fragment_ids,
                    ),
                )[:maximum_plans]
            )
        if not next_paths:
            return (
                _unsupported(
                    "no_compatible_seam",
                    (
                        "geometry-matched fragments have no compatible incoming "
                        f"seam for target transition {target.transition_index}"
                    ),
                    target.transition_index,
                    rejections,
                ),
            )
        paths = tuple(next_paths)
        if maximum_paths_per_layer is not None:
            paths = tuple(
                sorted(
                    paths,
                    key=lambda path: (
                        path.weighted_cost,
                        path.fragment_ids,
                    ),
                )[:maximum_paths_per_layer]
            )

    diagnostics = StairFragmentSequenceDiagnostics(tuple(rejections))
    ranked = sorted(
        paths,
        key=lambda path: (path.weighted_cost, path.fragment_ids),
    )[:maximum_plans]
    return tuple(
        StairFragmentSequencePlan(
            supported=True,
            selections=tuple(
                StairFragmentSelection(
                    target=target,
                    fragment=candidate.fragment,
                    matched_run_m=candidate.matched_run_m,
                    geometry_cost=candidate.geometry_cost,
                    seam_cost=seam_cost,
                )
                for target, candidate, seam_cost in zip(
                    targets, path.candidates, path.seam_costs
                )
            ),
            total_geometry_cost=path.geometry_cost,
            total_seam_cost=path.seam_cost,
            total_cost=path.weighted_cost,
            diagnostics=diagnostics,
            reason=None,
        )
        for path in ranked
    )


def plan_stair_fragment_sequence(
    route: StairSupportRoute,
    fragments: Sequence[StairFragmentRecord],
    config: StairFragmentSequenceConfig = StairFragmentSequenceConfig(),
) -> StairFragmentSequencePlan:
    """Select the minimum-cost seam-compatible fragment for every route riser."""

    return plan_stair_fragment_sequences(
        route,
        fragments,
        config,
        maximum_plans=1,
    )[0]


__all__ = (
    "StairFragmentRejection",
    "StairFragmentSelection",
    "StairFragmentSequenceConfig",
    "StairFragmentSequenceDiagnostics",
    "StairFragmentSequencePlan",
    "StairFragmentSequenceReason",
    "StairFragmentTarget",
    "load_stair_fragment_records_jsonl",
    "plan_stair_fragment_sequence",
    "plan_stair_fragment_sequences",
)
