"""Receding-horizon execution for the offline terrain contact oracle."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
import tempfile
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .joints import ContractError
from .torch_contact_oracle_actions import (
    ContactPhaseActionIndex,
    with_mirrored_actions,
)
from .torch_contact_oracle_search import (
    CommandSchedule,
    ContactOracleExperimentConfig,
    FeasibilityResult,
    OracleConstraints,
    OracleCost,
    OraclePlan,
    OracleSearchFailure,
    OracleState,
    advance_state,
    search_contact_plan,
    validate_placement,
)
from .torch_g1_fk import target_state_qpos
from .torch_terrain_omni_metrics import OmniRouteMetrics, evaluate_omni_route
from .torch_terrain_omni_rollout import (
    RouteOutcomeEvaluation,
    evaluate_route_outcome,
)
from .torch_terrain_omni_routes import (
    OmniRoute,
    StairFrame,
    same_stair_routes,
    world_commands,
)


ORACLE_ARRAY_SHAPES = {
    "command_velocity_world_xy": (2,),
    "command_heading_world_yaw": (),
    "command_segment_index": (),
    "qpos": (36,),
    "joint_position": (29,),
    "joint_velocity": (29,),
    "root_position_world": (3,),
    "root_yaw_world": (),
    "root_orientation_world_wxyz": (4,),
    "foot_position_world": (2, 3),
    "foot_surface_height_m": (2,),
    "selected_clip_path": (),
    "selected_source_frame": (),
    "selected_action_index": (),
    "planned_horizon": (),
    "plan_total_cost": (),
    "plan_time_ns": (),
    "stance_error_m": (),
    "landing_error_m": (),
    "minimum_swing_clearance_m": (),
}
ORACLE_IMPLEMENTATION_ID = "g1-contact-oracle/all-flight-clearance-v7"


@dataclass(frozen=True)
class OracleRouteFailure:
    stage: str
    frame_index: int | None
    exception_type: str
    message: str
    rejected_by_reason: Mapping[str, int]

    def __post_init__(self) -> None:
        rejected = dict(self.rejected_by_reason)
        if any(
            not isinstance(reason, str)
            or not reason
            or type(count) is not int
            or count < 1
            for reason, count in rejected.items()
        ):
            raise ContractError("contact oracle route failure is invalid")
        object.__setattr__(
            self, "rejected_by_reason", MappingProxyType(rejected)
        )


@dataclass(frozen=True)
class OraclePlanEvent:
    route_frame: int
    action_indices: tuple[int, ...]
    source_keys: tuple[tuple[int, int, int, bool], ...]
    step_costs: tuple[OracleCost, ...]
    total_cost: OracleCost
    planned_landing_feet: tuple[int, ...]
    planned_landing_world_xyz: tuple[tuple[float, float, float], ...]
    emitted_landing_world_xyz: tuple[float, float, float] | None
    entry_joint_position_error_rad: tuple[float, ...]
    entry_joint_velocity_error_rad_s: tuple[float, ...]
    rejected_by_reason: Mapping[str, int]
    beam_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        rejected = dict(self.rejected_by_reason)
        if (
            type(self.route_frame) is not int
            or self.route_frame < 0
            or not self.action_indices
            or len(self.action_indices) != len(self.source_keys)
            or len(self.step_costs) != len(self.action_indices)
            or any(not isinstance(cost, OracleCost) for cost in self.step_costs)
            or not isinstance(self.total_cost, OracleCost)
            or len(self.planned_landing_feet) != len(self.action_indices)
            or any(foot not in (0, 1) for foot in self.planned_landing_feet)
            or len(self.planned_landing_world_xyz) != len(self.action_indices)
            or len(self.entry_joint_position_error_rad) != len(self.action_indices)
            or len(self.entry_joint_velocity_error_rad_s) != len(self.action_indices)
            or any(
                len(position) != 3
                or not all(math.isfinite(float(value)) for value in position)
                for position in self.planned_landing_world_xyz
            )
            or (
                self.emitted_landing_world_xyz is not None
                and (
                    len(self.emitted_landing_world_xyz) != 3
                    or not all(
                        math.isfinite(float(value))
                        for value in self.emitted_landing_world_xyz
                    )
                )
            )
            or any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in (
                    *self.entry_joint_position_error_rad,
                    *self.entry_joint_velocity_error_rad_s,
                )
            )
        ):
            raise ContractError("contact oracle plan event is invalid")
        object.__setattr__(self, "rejected_by_reason", MappingProxyType(rejected))


@dataclass(frozen=True)
class OracleRouteRun:
    route: OmniRoute
    arrays: Mapping[str, np.ndarray]
    plan_events: tuple[OraclePlanEvent, ...]
    outcome: RouteOutcomeEvaluation
    metrics: OmniRouteMetrics | None
    completed_frames: int
    completed_without_exception: bool
    failure: OracleRouteFailure | None
    deterministic_sha256: str


@dataclass(frozen=True)
class OracleMatrix:
    runs: tuple[OracleRouteRun, ...]
    dataset_identity: str
    config_identity: str
    action_count: int
    action_rejections: Mapping[str, int]
    matrix_pass: bool
    deterministic_sha256: str

    def __post_init__(self) -> None:
        rejected = dict(self.action_rejections)
        if (
            not isinstance(self.runs, tuple)
            or any(not isinstance(run, OracleRouteRun) for run in self.runs)
            or not self.dataset_identity
            or not self.config_identity
            or type(self.action_count) is not int
            or self.action_count < 0
            or type(self.matrix_pass) is not bool
            or len(self.deterministic_sha256) != 64
        ):
            raise ContractError("contact oracle matrix is invalid")
        object.__setattr__(self, "action_rejections", MappingProxyType(rejected))


def _command_arrays(
    route: OmniRoute,
    stair_frame: StairFrame,
    device: torch.device,
    origin_world_xy: torch.Tensor,
) -> tuple[CommandSchedule, torch.Tensor]:
    expanded = world_commands(stair_frame, route)
    velocities = []
    headings = []
    segments = []
    for segment_index, command in enumerate(expanded):
        velocities.extend([command.velocity_world_xy] * command.frames)
        headings.extend([command.heading_world_yaw] * command.frames)
        segments.extend([segment_index] * command.frames)
    if not velocities:
        raise ContractError("contact oracle route has no command frames")
    return (
        CommandSchedule(
            velocity_world_xy=torch.tensor(
                velocities, dtype=torch.float32, device=device
            ),
            heading_world_yaw=torch.tensor(
                headings, dtype=torch.float32, device=device
            ),
            origin_world_xy=origin_world_xy,
        ),
        torch.tensor(segments, dtype=torch.int64, device=device),
    )


def _surface(
    terrain_sampler: Callable[[torch.Tensor], torch.Tensor], feet: torch.Tensor
) -> torch.Tensor:
    values = terrain_sampler(feet[:, :2])
    if (
        not isinstance(values, torch.Tensor)
        or tuple(values.shape) != (2,)
        or values.dtype != feet.dtype
        or values.device != feet.device
        or not torch.isfinite(values).all()
    ):
        raise ContractError("contact oracle rollout terrain sample is invalid")
    return values


def _append_frame(
    rows: dict[str, list],
    *,
    schedule: CommandSchedule,
    segment_indices: torch.Tensor,
    route_frame: int,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    root_position: torch.Tensor,
    root_yaw: torch.Tensor,
    root_orientation: torch.Tensor,
    feet: torch.Tensor,
    surface: torch.Tensor,
    clip_path: str,
    source_frame: int,
    action_index: int,
    planned_horizon: int,
    plan_total_cost: float,
    plan_time_ns: int,
    feasibility: FeasibilityResult,
) -> None:
    qpos = target_state_qpos(
        joint_position, root_position, root_orientation
    )
    rows["command_velocity_world_xy"].append(
        schedule.velocity_world_xy[route_frame].detach().cpu().numpy()
    )
    rows["command_heading_world_yaw"].append(
        float(schedule.heading_world_yaw[route_frame].item())
    )
    rows["command_segment_index"].append(
        int(segment_indices[route_frame].item())
    )
    rows["qpos"].append(qpos)
    rows["joint_position"].append(joint_position.detach().cpu().numpy())
    rows["joint_velocity"].append(joint_velocity.detach().cpu().numpy())
    rows["root_position_world"].append(root_position.detach().cpu().numpy())
    rows["root_yaw_world"].append(float(root_yaw.item()))
    rows["root_orientation_world_wxyz"].append(
        root_orientation.detach().cpu().numpy()
    )
    rows["foot_position_world"].append(feet.detach().cpu().numpy())
    rows["foot_surface_height_m"].append(surface.detach().cpu().numpy())
    rows["selected_clip_path"].append(clip_path)
    rows["selected_source_frame"].append(source_frame)
    rows["selected_action_index"].append(action_index)
    rows["planned_horizon"].append(planned_horizon)
    rows["plan_total_cost"].append(plan_total_cost)
    rows["plan_time_ns"].append(plan_time_ns)
    rows["stance_error_m"].append(feasibility.stance_error_m)
    rows["landing_error_m"].append(feasibility.landing_error_m)
    rows["minimum_swing_clearance_m"].append(
        feasibility.minimum_swing_clearance_m
    )


def _append_hold(
    rows: dict[str, list],
    *,
    state: OracleState,
    schedule: CommandSchedule,
    segment_indices: torch.Tensor,
    route_frame: int,
    terrain_sampler: Callable[[torch.Tensor], torch.Tensor],
) -> OracleState:
    feasibility = FeasibilityResult(True, None, 0.0, 0.0, 0.0)
    _append_frame(
        rows,
        schedule=schedule,
        segment_indices=segment_indices,
        route_frame=route_frame,
        joint_position=state.joint_position,
        joint_velocity=torch.zeros_like(state.joint_velocity),
        root_position=state.root_position_world,
        root_yaw=state.root_yaw_world,
        root_orientation=state.root_orientation_world_wxyz,
        feet=state.foot_position_world,
        surface=_surface(terrain_sampler, state.foot_position_world),
        clip_path="<hold>",
        source_frame=-1,
        action_index=-1,
        planned_horizon=1,
        plan_total_cost=0.0,
        plan_time_ns=0,
        feasibility=feasibility,
    )
    return replace(
        state,
        joint_velocity=torch.zeros_like(state.joint_velocity),
        route_frame=route_frame,
    )


def _finalize_rows(rows: Mapping[str, list]) -> Mapping[str, np.ndarray]:
    output = {}
    frame_count = len(rows["qpos"])
    integer = {
        "command_segment_index",
        "selected_source_frame",
        "selected_action_index",
        "planned_horizon",
        "plan_time_ns",
    }
    for name, tail in ORACLE_ARRAY_SHAPES.items():
        if name == "selected_clip_path":
            array = np.asarray(rows[name], dtype=np.str_)
        elif name in integer:
            array = np.asarray(rows[name], dtype=np.int64)
        else:
            array = np.asarray(rows[name], dtype=np.float64)
        expected = (frame_count, *tail)
        if array.shape != expected:
            if frame_count == 0:
                array = np.empty(expected, dtype=array.dtype)
            else:
                raise ContractError(f"contact oracle rollout {name} shape is invalid")
        if name != "selected_clip_path" and not np.isfinite(array).all():
            raise ContractError(f"contact oracle rollout {name} is non-finite")
        array.setflags(write=False)
        output[name] = array
    return MappingProxyType(output)


def _hash_run(
    route: OmniRoute,
    arrays: Mapping[str, np.ndarray],
    plan_events: Sequence[OraclePlanEvent],
    failure: OracleRouteFailure | None,
    dataset_identity: str,
    config_identity: str,
) -> str:
    digest = hashlib.sha256(b"g1-contact-oracle-route/v1")
    for value in (route.name, dataset_identity, config_identity):
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
    for name in sorted(arrays):
        if name == "plan_time_ns":
            continue
        array = arrays[name]
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        if array.dtype.kind in "US":
            digest.update("\0".join(array.tolist()).encode("utf-8"))
        else:
            digest.update(array.tobytes())
    digest.update(b"\0plan-events\0")
    digest.update(
        _canonical_json(
            [_plan_event_json(event) for event in plan_events]
        )
    )
    if failure is not None:
        digest.update(b"\0failure\0")
        digest.update(_canonical_json(_failure_json(failure)))
    return digest.hexdigest()


def oracle_arrays_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash the complete saved array payload without pickle serialization."""

    if set(arrays) != set(ORACLE_ARRAY_SHAPES):
        raise ContractError("contact oracle array inventory is invalid")
    digest = hashlib.sha256(b"g1-contact-oracle-arrays/v1")
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(array.shape).encode("ascii"))
        if array.dtype.kind in "US":
            digest.update("\0".join(array.reshape(-1).tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_oracle_route(
    *,
    route: OmniRoute,
    stair_frame: StairFrame,
    initial_state: OracleState,
    planner: Callable[[OracleState, CommandSchedule], OraclePlan],
    terrain_sampler: Callable[[torch.Tensor], torch.Tensor],
    clip_paths: Sequence[str],
    dataset_identity: str,
    config_identity: str,
    constraints: OracleConstraints = OracleConstraints(),
) -> OracleRouteRun:
    """Execute only the first planned edge and replan at its terminal contact."""

    if (
        not isinstance(route, OmniRoute)
        or not isinstance(stair_frame, StairFrame)
        or not isinstance(initial_state, OracleState)
        or not callable(planner)
        or not callable(terrain_sampler)
        or not dataset_identity
        or not config_identity
        or not isinstance(constraints, OracleConstraints)
    ):
        raise ContractError("contact oracle rollout inputs are invalid")
    if (
        not route.commands
        or not route.commands[0].reset_before
        or any(command.reset_before for command in route.commands[1:])
    ):
        raise ContractError("contact oracle route reset contract is invalid")
    paths = tuple(clip_paths)
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ContractError("contact oracle clip paths are invalid")
    schedule, segment_indices = _command_arrays(
        route,
        stair_frame,
        initial_state.root_position_world.device,
        initial_state.root_position_world[:2],
    )
    rows = {name: [] for name in ORACLE_ARRAY_SHAPES}
    events: list[OraclePlanEvent] = []
    state = initial_state
    failure = None
    stage = "setup"
    try:
        while len(rows["qpos"]) < schedule.frame_count:
            next_frame = len(rows["qpos"])
            command_speed = float(
                torch.linalg.vector_norm(schedule.velocity(next_frame)).item()
            )
            if command_speed <= 1e-6:
                stage = "hold"
                state = _append_hold(
                    rows,
                    state=state,
                    schedule=schedule,
                    segment_indices=segment_indices,
                    route_frame=next_frame,
                    terrain_sampler=terrain_sampler,
                )
                continue
            stage = "oracle-search"
            start_ns = time.perf_counter_ns()
            plan = planner(state, schedule)
            elapsed_ns = time.perf_counter_ns() - start_ns
            if not isinstance(plan, OraclePlan) or not plan.placements:
                raise ContractError("contact oracle planner returned no first edge")
            action_index = plan.action_indices[0]
            placed = plan.placements[0]
            if not 0 <= placed.action.clip_index < len(paths):
                raise ContractError("contact oracle selected clip is outside inventory")
            stage = "terrain-validation"
            feasibility = validate_placement(
                placed=placed,
                state=state,
                sample_surface=terrain_sampler,
                constraints=constraints,
            )
            if not feasibility.accepted:
                raise ContractError(
                    f"planned first edge failed revalidation: {feasibility.reason}"
                )
            first_local_frame = 0 if not rows["qpos"] else 1
            terminal_route_frame = (
                state.route_frame + placed.action.frame_count - 1
            )
            command_window = schedule.velocity_world_xy[
                next_frame : min(
                    terminal_route_frame + 1, schedule.frame_count
                )
            ]
            if bool(
                (
                    torch.linalg.vector_norm(command_window, dim=1)
                    <= 1e-6
                ).any().item()
            ):
                stage = "command-boundary-hold"
                state = _append_hold(
                    rows,
                    state=state,
                    schedule=schedule,
                    segment_indices=segment_indices,
                    route_frame=next_frame,
                    terrain_sampler=terrain_sampler,
                )
                continue
            plan_state = state
            position_errors = []
            velocity_errors = []
            for plan_placement in plan.placements:
                position_errors.append(
                    float(
                        torch.linalg.vector_norm(
                            plan_placement.action.joint_position[0]
                            - plan_state.joint_position
                        ).item()
                    )
                )
                velocity_errors.append(
                    float(
                        torch.linalg.vector_norm(
                            plan_placement.action.joint_velocity[0]
                            - plan_state.joint_velocity
                        ).item()
                    )
                )
                plan_state = advance_state(plan_state, plan_placement)
            stage = "record"
            for local_frame in range(
                first_local_frame, placed.action.frame_count
            ):
                route_frame = state.route_frame + local_frame
                if route_frame >= schedule.frame_count:
                    break
                feet = placed.foot_position_world[local_frame]
                _append_frame(
                    rows,
                    schedule=schedule,
                    segment_indices=segment_indices,
                    route_frame=route_frame,
                    joint_position=placed.action.joint_position[local_frame],
                    joint_velocity=placed.action.joint_velocity[local_frame],
                    root_position=placed.root_position_world[local_frame],
                    root_yaw=placed.root_yaw_world[local_frame],
                    root_orientation=(
                        placed.root_orientation_world_wxyz[local_frame]
                    ),
                    feet=feet,
                    surface=_surface(terrain_sampler, feet),
                    clip_path=paths[placed.action.clip_index],
                    source_frame=placed.action.start_frame + local_frame,
                    action_index=action_index,
                    planned_horizon=plan.horizon_landings,
                    plan_total_cost=plan.total_cost.total,
                    plan_time_ns=elapsed_ns,
                    feasibility=feasibility,
                )
            completed_landing = (
                route_frame == state.route_frame + placed.action.frame_count - 1
            )
            events.append(
                OraclePlanEvent(
                    route_frame=state.route_frame,
                    action_indices=plan.action_indices,
                    source_keys=tuple(
                        placement.action.source_key
                        for placement in plan.placements
                    ),
                    step_costs=plan.step_costs,
                    total_cost=plan.total_cost,
                    planned_landing_feet=tuple(
                        placement.action.swing_foot
                        for placement in plan.placements
                    ),
                    planned_landing_world_xyz=tuple(
                        tuple(
                            float(value)
                            for value in placement.foot_position_world[
                                -1, placement.action.swing_foot
                            ].detach().cpu().tolist()
                        )
                        for placement in plan.placements
                    ),
                    emitted_landing_world_xyz=(
                        tuple(
                            float(value)
                            for value in placed.foot_position_world[
                                -1, placed.action.swing_foot
                            ].detach().cpu().tolist()
                        )
                        if completed_landing
                        else None
                    ),
                    entry_joint_position_error_rad=tuple(position_errors),
                    entry_joint_velocity_error_rad_s=tuple(velocity_errors),
                    rejected_by_reason=plan.expansion.rejected_by_reason,
                    beam_sizes=plan.expansion.beam_sizes,
                )
            )
            if len(rows["qpos"]) >= schedule.frame_count:
                break
            state = advance_state(state, placed)
    except Exception as error:
        failure = OracleRouteFailure(
            stage=stage,
            frame_index=len(rows["qpos"]),
            exception_type=type(error).__name__,
            message=str(error),
            rejected_by_reason=(
                error.rejected_by_reason
                if isinstance(error, OracleSearchFailure)
                else {}
            ),
        )

    arrays = _finalize_rows(rows)
    outcome = evaluate_route_outcome(route, arrays)
    metrics = None
    if arrays["qpos"].shape[0]:
        metrics = evaluate_omni_route(
            root_xy=arrays["root_position_world"][:, :2],
            root_yaw=arrays["root_yaw_world"],
            foot_position_world=arrays["foot_position_world"],
            foot_surface_height_m=arrays["foot_surface_height_m"],
            command_velocity_world_xy=arrays["command_velocity_world_xy"],
            command_heading_world_yaw=arrays["command_heading_world_yaw"],
            selected_clip_id=arrays["selected_clip_path"],
            selected_source_frame=arrays["selected_source_frame"],
            required_outcome_completed=outcome.completed,
        )
    return OracleRouteRun(
        route=route,
        arrays=arrays,
        plan_events=tuple(events),
        outcome=outcome,
        metrics=metrics,
        completed_frames=int(arrays["qpos"].shape[0]),
        completed_without_exception=failure is None,
        failure=failure,
        deterministic_sha256=_hash_run(
            route,
            arrays,
            tuple(events),
            failure,
            dataset_identity,
            config_identity,
        ),
    )


def run_oracle_matrix(
    *,
    routes: Sequence[OmniRoute],
    stair_frame: StairFrame,
    initial_state: OracleState,
    planner: Callable[[OracleState, CommandSchedule], OraclePlan],
    terrain_sampler: Callable[[torch.Tensor], torch.Tensor],
    clip_paths: Sequence[str],
    dataset_identity: str,
    config_identity: str,
    action_index: ContactPhaseActionIndex,
    constraints: OracleConstraints,
) -> OracleMatrix:
    inventory = tuple(routes)
    if not inventory or not isinstance(action_index, ContactPhaseActionIndex):
        raise ContractError("contact oracle matrix inventory is invalid")
    runs = tuple(
        run_oracle_route(
            route=route,
            stair_frame=stair_frame,
            initial_state=initial_state,
            planner=planner,
            terrain_sampler=terrain_sampler,
            clip_paths=clip_paths,
            dataset_identity=dataset_identity,
            config_identity=config_identity,
            constraints=constraints,
        )
        for route in inventory
    )
    digest = hashlib.sha256(b"g1-contact-oracle-matrix/v1")
    digest.update(dataset_identity.encode("utf-8"))
    digest.update(config_identity.encode("utf-8"))
    for run in runs:
        digest.update(run.route.name.encode("utf-8"))
        digest.update(run.deterministic_sha256.encode("ascii"))
    return OracleMatrix(
        runs=runs,
        dataset_identity=dataset_identity,
        config_identity=config_identity,
        action_count=len(action_index.actions),
        action_rejections=action_index.inventory.rejected_by_reason,
        matrix_pass=all(
            run.completed_without_exception and run.outcome.completed
            for run in runs
        ),
        deterministic_sha256=digest.hexdigest(),
    )


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _resolved_initial_state(resolved, segment_index, foot_kinematics) -> OracleState:
    reset_path = resolved.resolved_config["reset_clip"]
    matches = [
        (clip_index, clip)
        for clip_index, clip in enumerate(resolved.dataset.folder.clips)
        if clip.relative_path == reset_path
    ]
    if len(matches) != 1:
        raise ContractError("contact oracle reset clip must resolve exactly once")
    clip_index, clip = matches[0]
    support = segment_index.support_mask(clip_index).to(resolved.device)
    velocity_sq = np.sum(np.square(clip.joint_velocity), axis=1)
    supported = support.any(dim=1).detach().cpu().numpy()
    candidates = np.flatnonzero(supported)
    if candidates.size == 0:
        raise ContractError("contact oracle reset clip has no supported frame")
    frame = int(candidates[np.argmin(velocity_sq[candidates])])
    layout = resolved.dataset.folder.layout
    root_index = int(layout.root_body_index)
    joint_position = torch.tensor(
        clip.joint_position[frame], dtype=torch.float32, device=resolved.device
    )
    joint_velocity = torch.tensor(
        clip.joint_velocity[frame], dtype=torch.float32, device=resolved.device
    )
    root = torch.tensor(
        clip.body_position_world[frame, root_index],
        dtype=torch.float32,
        device=resolved.device,
    )
    from .torch_terrain_omni_rollout import resolved_stair_reset_position

    reset_xy = resolved_stair_reset_position(resolved)
    root[:2] = torch.tensor(reset_xy, dtype=torch.float32, device=resolved.device)
    quaternion = torch.tensor(
        clip.body_quaternion_world_wxyz[frame, root_index],
        dtype=torch.float32,
        device=resolved.device,
    )
    feet = torch.tensor(
        foot_kinematics.foot_positions(
            joint_position[None, :], root[None, :], quaternion[None, :]
        )[0],
        dtype=torch.float32,
        device=resolved.device,
    )
    return OracleState(
        root_position_world=root,
        root_yaw_world=_yaw_from_wxyz(quaternion),
        root_orientation_world_wxyz=quaternion,
        foot_position_world=feet,
        support_mask=support[frame],
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        route_frame=0,
    )


def run_resolved_oracle_matrix(
    *,
    dataset: str,
    experiment: ContactOracleExperimentConfig,
    g1_xml: str,
    device: str | torch.device = "cuda",
    routes: Sequence[OmniRoute] | None = None,
) -> OracleMatrix:
    """Build the privileged-height oracle and run the fixed route matrix."""

    if not isinstance(experiment, ContactOracleExperimentConfig):
        raise ContractError("contact oracle resolved experiment is invalid")
    from .torch_contact_segment_rollout import (
        build_contact_segment_policy,
        load_contact_segment_config,
        resolve_contact_segment_config,
    )
    from .torch_g1_fk import MujocoG1FootKinematics

    terrain_raw = load_contact_segment_config(experiment.terrain_config)
    resolved = resolve_contact_segment_config(
        dataset, terrain_raw, device=device
    )
    contact_policy = build_contact_segment_policy(resolved, g1_xml)
    foot_kinematics = MujocoG1FootKinematics(g1_xml)
    action_index = with_mirrored_actions(
        ContactPhaseActionIndex.from_dataset(
            resolved.dataset, contact_policy.index, foot_kinematics
        )
    )
    initial_state = _resolved_initial_state(
        resolved, contact_policy.index, foot_kinematics
    )
    measurement = resolved.measurement_extension

    def terrain_sampler(points: torch.Tensor) -> torch.Tensor:
        return measurement.query_grid.sample_xy(
            measurement.alignment.matcher_to_scene_xy(points)
        )

    direction = np.asarray(
        resolved.resolved_config["reference_direction_matcher_xy"],
        dtype=np.float64,
    )
    stair_frame = StairFrame(
        origin_world_xy=(0.0, 0.0),
        ascent_world_yaw=math.atan2(float(direction[1]), float(direction[0])),
        width_m=0.6223,
        tread_depth_m=0.3302,
        riser_height_m=0.1778,
        tread_count=3,
    )

    def planner(state: OracleState, schedule: CommandSchedule) -> OraclePlan:
        return search_contact_plan(
            initial_state=state,
            actions=action_index.actions,
            command_schedule=schedule,
            sample_surface=terrain_sampler,
            config=experiment.search,
        )

    config_payload = {
        "implementation": ORACLE_IMPLEMENTATION_ID,
        "terrain": resolved.base_config_sha256,
        "search": repr(experiment.search),
    }
    config_identity = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return run_oracle_matrix(
        routes=tuple(same_stair_routes() if routes is None else routes),
        stair_frame=stair_frame,
        initial_state=initial_state,
        planner=planner,
        terrain_sampler=terrain_sampler,
        clip_paths=tuple(
            clip.relative_path for clip in resolved.dataset.folder.clips
        ),
        dataset_identity=resolved.dataset.manifest_sha256,
        config_identity=config_identity,
        action_index=action_index,
        constraints=experiment.search.constraints,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _failure_json(failure: OracleRouteFailure | None) -> dict | None:
    if failure is None:
        return None
    return {
        "stage": failure.stage,
        "frame_index": failure.frame_index,
        "exception_type": failure.exception_type,
        "message": failure.message,
        "rejected_by_reason": dict(failure.rejected_by_reason),
    }


def _cost_json(cost: OracleCost) -> dict[str, float]:
    return {
        name: float(getattr(cost, name))
        for name in OracleCost.__dataclass_fields__
    } | {"total": cost.total}


def _plan_event_json(event: OraclePlanEvent) -> dict:
    return {
        "route_frame": event.route_frame,
        "action_indices": list(event.action_indices),
        "source_keys": [list(key) for key in event.source_keys],
        "step_costs": [_cost_json(cost) for cost in event.step_costs],
        "total_cost": _cost_json(event.total_cost),
        "planned_landing_feet": list(event.planned_landing_feet),
        "planned_landing_world_xyz": [
            list(position) for position in event.planned_landing_world_xyz
        ],
        "emitted_landing_world_xyz": (
            list(event.emitted_landing_world_xyz)
            if event.emitted_landing_world_xyz is not None
            else None
        ),
        "entry_joint_position_error_rad": list(
            event.entry_joint_position_error_rad
        ),
        "entry_joint_velocity_error_rad_s": list(
            event.entry_joint_velocity_error_rad_s
        ),
        "rejected_by_reason": dict(event.rejected_by_reason),
        "beam_sizes": list(event.beam_sizes),
    }


def _outcome_json(outcome: RouteOutcomeEvaluation) -> dict:
    final_heading = float(outcome.final_heading_error_rad)
    return {
        "completed": outcome.completed,
        "segment_progress_ratio": [
            {"segment": segment, "ratio": ratio}
            for segment, ratio in outcome.segment_progress_ratio
        ],
        "elevated_foot_sample_count": outcome.elevated_foot_sample_count,
        "final_heading_error_rad": (
            final_heading if math.isfinite(final_heading) else None
        ),
        "failure_reasons": list(outcome.failure_reasons),
    }


def _metrics_json(metrics: OmniRouteMetrics | None) -> dict | None:
    if metrics is None:
        return None

    def distribution(value) -> dict:
        return {
            "count": len(value.samples),
            "minimum": value.minimum,
            "maximum": value.maximum,
            "mean": value.mean,
            "p95": value.p95,
        }

    return {
        "stance_frame_count_per_foot": np.sum(
            metrics.stance_mask, axis=0
        ).astype(int).tolist(),
        "support_height_error_m": distribution(
            metrics.support_height_error_m
        ),
        "support_height_difference_m": distribution(
            metrics.support_height_difference_m
        ),
        "penetration_depth_m": distribution(metrics.penetration_depth_m),
        "stance_slide_m": {
            "per_foot": list(metrics.stance_slide_m.per_foot),
            "total": metrics.stance_slide_m.total,
        },
        "heading_error_rad": distribution(metrics.heading_error_rad),
        "root_progress_m": metrics.root_progress_m,
        "root_jerk_m_s3": distribution(metrics.root_jerk_m_s3),
        "root_velocity_error_mps": distribution(
            metrics.root_velocity_error_mps
        ),
        "stalled_moving_frame_count": int(
            np.sum(metrics.stalled_moving_mask)
        ),
        "stalled_moving_fraction": metrics.stalled_moving_fraction,
        "longest_stall_frames": metrics.longest_stall_frames,
        "transition_count": metrics.transition_count,
        "rescue_cycle_count": len(metrics.rescue_cycles),
        "rescue_cycles": [
            {
                "event_indices": list(cycle.event_indices),
                "clip_pair": list(cycle.clip_pair),
                "root_progress_m": cycle.root_progress_m,
            }
            for cycle in metrics.rescue_cycles
        ],
        "required_outcome_completed": metrics.required_outcome_completed,
    }


def save_oracle_matrix(matrix: OracleMatrix, output: str | Path) -> None:
    """Atomically publish a pickle-free contact-oracle matrix."""

    if not isinstance(matrix, OracleMatrix):
        raise TypeError("matrix must be an OracleMatrix")
    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"contact oracle output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        routes_dir = staging / "routes"
        routes_dir.mkdir()
        route_summaries = []
        for run in matrix.runs:
            route_dir = routes_dir / run.route.name
            route_dir.mkdir()
            np.savez_compressed(route_dir / "rollout.npz", **run.arrays)
            events = [_plan_event_json(event) for event in run.plan_events]
            diagnostics = {
                "schema": "g1-contact-space-oracle-route/v1",
                "route": run.route.name,
                "required_outcome": run.route.required_outcome,
                "completed_frames": run.completed_frames,
                "completed_without_exception": run.completed_without_exception,
                "failure": _failure_json(run.failure),
                "outcome": _outcome_json(run.outcome),
                "metrics": _metrics_json(run.metrics),
                "plan_events": events,
                "arrays_sha256": oracle_arrays_sha256(run.arrays),
                "deterministic_sha256": run.deterministic_sha256,
            }
            (route_dir / "diagnostics.json").write_bytes(
                _canonical_json(diagnostics) + b"\n"
            )
            route_summaries.append(
                {
                    "name": run.route.name,
                    "required_outcome": run.route.required_outcome,
                    "completed_frames": run.completed_frames,
                    "completed_without_exception": (
                        run.completed_without_exception
                    ),
                    "outcome_completed": run.outcome.completed,
                    "failure": _failure_json(run.failure),
                    "deterministic_sha256": run.deterministic_sha256,
                }
            )
        summary = {
            "schema": "g1-contact-space-oracle-matrix/v1",
            "dataset_identity": matrix.dataset_identity,
            "config_identity": matrix.config_identity,
            "action_count": matrix.action_count,
            "action_rejections": dict(matrix.action_rejections),
            "matrix_pass": matrix.matrix_pass,
            "deterministic_sha256": matrix.deterministic_sha256,
            "routes": route_summaries,
        }
        (staging / "summary.json").write_bytes(
            _canonical_json(summary) + b"\n"
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
