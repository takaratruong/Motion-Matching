"""Exhaustive, multi-objective ranking for frozen terrain quality states."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .torch_contact_oracle_actions import ContactPhaseActionIndex
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M
from .torch_contact_oracle_search import (
    OracleConstraints,
    OracleState,
    advance_state,
    place_action,
    validate_placement,
)
from .torch_terrain_action_quality import NativeActionQuality
from .torch_terrain_quality_states import FrozenQualityState


@dataclass(frozen=True)
class QualityActionCache:
    """Padded immutable tensors used by every frozen-state exhaustive query."""

    source_keys: tuple[tuple[int, int, int, bool], ...]
    entry_support: torch.Tensor
    entry_joint_position: torch.Tensor
    entry_joint_velocity: torch.Tensor
    entry_foot_local: torch.Tensor
    terminal_root_local_xy: torch.Tensor
    terminal_yaw_local: torch.Tensor
    terminal_foot_local: torch.Tensor
    frame_steps: torch.Tensor
    foot_trajectory_local: torch.Tensor
    support_trajectory: torch.Tensor
    valid_frames: torch.Tensor
    surface_delta_m: torch.Tensor
    swing_foot: torch.Tensor


def build_quality_action_cache(index: ContactPhaseActionIndex) -> QualityActionCache:
    """Pad the complete inventory once so per-state feasibility stays batched."""

    if not isinstance(index, ContactPhaseActionIndex):
        raise ValueError("terrain quality cache requires ContactPhaseActionIndex")
    actions = index.actions
    if not actions:
        empty_float = torch.empty((0,), dtype=torch.float32)
        return QualityActionCache(
            source_keys=(),
            entry_support=torch.empty((0, 2), dtype=torch.bool),
            entry_joint_position=torch.empty((0, 29)),
            entry_joint_velocity=torch.empty((0, 29)),
            entry_foot_local=torch.empty((0, 2, 3)),
            terminal_root_local_xy=torch.empty((0, 2)),
            terminal_yaw_local=empty_float,
            terminal_foot_local=torch.empty((0, 3)),
            frame_steps=torch.empty((0,), dtype=torch.int64),
            foot_trajectory_local=torch.empty((0, 1, 2, 3)),
            support_trajectory=torch.empty((0, 1, 2), dtype=torch.bool),
            valid_frames=torch.empty((0, 1), dtype=torch.bool),
            surface_delta_m=torch.empty((0, 1, 2)),
            swing_foot=torch.empty((0,), dtype=torch.int64),
        )
    device = actions[0].root_position_local.device
    dtype = actions[0].root_position_local.dtype
    count = len(actions)
    frame_count = max(action.frame_count for action in actions)
    feet = torch.zeros((count, frame_count, 2, 3), dtype=dtype, device=device)
    support = torch.zeros((count, frame_count, 2), dtype=torch.bool, device=device)
    valid = torch.zeros((count, frame_count), dtype=torch.bool, device=device)
    surface = torch.zeros((count, frame_count, 2), dtype=dtype, device=device)
    for action_index, action in enumerate(actions):
        length = action.frame_count
        feet[action_index, :length] = action.foot_position_local
        support[action_index, :length] = action.support_mask
        valid[action_index, :length] = True
        surface[action_index, :length] = action.foot_surface_delta_m
    swing = torch.tensor(
        [action.swing_foot for action in actions], dtype=torch.int64, device=device
    )
    rows = torch.arange(count, dtype=torch.int64, device=device)
    return QualityActionCache(
        source_keys=tuple(action.source_key for action in actions),
        entry_support=torch.stack([action.entry_support for action in actions]),
        entry_joint_position=torch.stack(
            [action.joint_position[0] for action in actions]
        ),
        entry_joint_velocity=torch.stack(
            [action.joint_velocity[0] for action in actions]
        ),
        entry_foot_local=torch.stack(
            [action.foot_position_local[0] for action in actions]
        ),
        terminal_root_local_xy=torch.stack(
            [action.root_position_local[-1, :2] for action in actions]
        ),
        terminal_yaw_local=torch.stack(
            [action.root_yaw_local[-1] for action in actions]
        ),
        terminal_foot_local=torch.stack(
            [action.foot_position_local[-1, action.swing_foot] for action in actions]
        ),
        frame_steps=torch.tensor(
            [action.frame_count - 1 for action in actions],
            dtype=torch.int64,
            device=device,
        ),
        foot_trajectory_local=feet,
        support_trajectory=support,
        valid_frames=valid,
        surface_delta_m=surface,
        swing_foot=swing,
    )


@dataclass(frozen=True)
class QualityCandidateScore:
    action_index: int
    source_key: tuple[int, int, int, bool]
    feasible: bool
    feasibility_reason: str | None
    landing_error_m: float
    support_mismatch_count: int
    joint_position_boundary_error_rad: float
    joint_velocity_boundary_error_rad_s: float
    command_displacement_error_m: float
    command_facing_error_rad: float
    native_stance_drift_m: float
    clearance_margin_m: float
    total_cost: float
    exact_successor_index: int | None
    two_step_total_cost: float | None

    def __post_init__(self) -> None:
        if (
            type(self.action_index) is not int
            or self.action_index < 0
            or not isinstance(self.source_key, tuple)
            or len(self.source_key) != 4
            or type(self.feasible) is not bool
            or self.feasible != (self.feasibility_reason is None)
            or type(self.support_mismatch_count) is not int
            or self.support_mismatch_count < 0
            or (
                self.exact_successor_index is not None
                and (
                    type(self.exact_successor_index) is not int
                    or self.exact_successor_index < 0
                )
            )
        ):
            raise ValueError("terrain quality candidate identity is invalid")
        values = (
            self.landing_error_m,
            self.joint_position_boundary_error_rad,
            self.joint_velocity_boundary_error_rad_s,
            self.command_displacement_error_m,
            self.command_facing_error_rad,
            self.native_stance_drift_m,
            self.clearance_margin_m,
            self.total_cost,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ) or any(float(value) < 0.0 for value in values[:6]):
            raise ValueError("terrain quality candidate costs are invalid")
        if self.two_step_total_cost is not None and (
            isinstance(self.two_step_total_cost, bool)
            or not isinstance(self.two_step_total_cost, (int, float))
            or not math.isfinite(float(self.two_step_total_cost))
            or float(self.two_step_total_cost) < 0.0
        ):
            raise ValueError("terrain quality two-step cost is invalid")


@dataclass(frozen=True)
class QualityOracleResult:
    state_id: str
    evaluated_action_count: int
    rejected_by_reason: Mapping[str, int]
    best_landing: tuple[QualityCandidateScore, ...]
    best_continuity: tuple[QualityCandidateScore, ...]
    best_combined: tuple[QualityCandidateScore, ...]
    best_two_step: tuple[QualityCandidateScore, ...]
    deterministic_sha256: str

    def __post_init__(self) -> None:
        rejected = dict(self.rejected_by_reason)
        rankings = (
            self.best_landing,
            self.best_continuity,
            self.best_combined,
            self.best_two_step,
        )
        if (
            not isinstance(self.state_id, str)
            or not self.state_id
            or type(self.evaluated_action_count) is not int
            or self.evaluated_action_count < 0
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count < 1
                for reason, count in rejected.items()
            )
            or any(
                not isinstance(rows, tuple)
                or any(not isinstance(row, QualityCandidateScore) for row in rows)
                for rows in rankings
            )
            or not isinstance(self.deterministic_sha256, str)
            or len(self.deterministic_sha256) != 64
        ):
            raise ValueError("terrain quality oracle result is invalid")
        object.__setattr__(
            self, "rejected_by_reason", MappingProxyType(dict(sorted(rejected.items())))
        )


def _vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _oracle_state(
    state: FrozenQualityState, *, dtype: torch.dtype, device: torch.device
) -> OracleState:
    quaternion = np.asarray(state.qpos[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        half = state.root_yaw_world / 2.0
        quaternion = np.array((math.cos(half), 0.0, 0.0, math.sin(half)))
    else:
        quaternion = quaternion / norm

    def tensor(value: object) -> torch.Tensor:
        return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)

    return OracleState(
        root_position_world=tensor(state.root_position_world),
        root_yaw_world=tensor(state.root_yaw_world),
        root_orientation_world_wxyz=tensor(quaternion),
        foot_position_world=tensor(state.foot_position_world),
        support_mask=torch.as_tensor(
            np.array(state.source_support_mask, copy=True),
            dtype=torch.bool,
            device=device,
        ),
        joint_position=tensor(state.joint_position),
        joint_velocity=tensor(state.joint_velocity),
        route_frame=state.route_frame,
    )


def _angle_error(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def _combined_cost(
    *,
    landing: float,
    joint_position: float,
    joint_velocity: float,
    displacement: float,
    facing: float,
    stance_drift: float,
    clearance_margin: float,
) -> float:
    """Dimensionless diagnostic score; individual components remain observable."""

    return float(
        (landing / 0.25) ** 2
        + (joint_position / 1.50) ** 2
        + (joint_velocity / 8.0) ** 2
        + (displacement / 0.50) ** 2
        + (facing / 0.50) ** 2
        + (stance_drift / 0.10) ** 2
        + (max(0.0, -clearance_margin) / 0.03) ** 2
    )


def _candidate_payload(row: QualityCandidateScore) -> dict[str, object]:
    return {
        "action_index": row.action_index,
        "source_key": list(row.source_key),
        "feasible": row.feasible,
        "feasibility_reason": row.feasibility_reason,
        "landing_error_m": row.landing_error_m,
        "support_mismatch_count": row.support_mismatch_count,
        "joint_position_boundary_error_rad": row.joint_position_boundary_error_rad,
        "joint_velocity_boundary_error_rad_s": row.joint_velocity_boundary_error_rad_s,
        "command_displacement_error_m": row.command_displacement_error_m,
        "command_facing_error_rad": row.command_facing_error_rad,
        "native_stance_drift_m": row.native_stance_drift_m,
        "clearance_margin_m": row.clearance_margin_m,
        "total_cost": row.total_cost,
        "exact_successor_index": row.exact_successor_index,
        "two_step_total_cost": row.two_step_total_cost,
    }


def _digest(
    state_id: str,
    evaluated: int,
    rejected: Mapping[str, int],
    rankings: Mapping[str, Sequence[QualityCandidateScore]],
) -> str:
    payload = {
        "schema": "g1-terrain-quality-oracle/v1",
        "state_id": state_id,
        "evaluated_action_count": evaluated,
        "rejected_by_reason": dict(sorted(rejected.items())),
        "rankings": {
            name: [_candidate_payload(row) for row in rows]
            for name, rows in sorted(rankings.items())
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rotate_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * values[..., 0] - sine * values[..., 1],
            sine * values[..., 0] + cosine * values[..., 1],
        ),
        dim=-1,
    )


def _surface_values(sample_surface, points: torch.Tensor) -> torch.Tensor:
    try:
        values = sample_surface(points)
    except Exception as error:
        raise ValueError("terrain quality surface sampling failed") from error
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != points.shape[:-1]
        or values.dtype != points.dtype
        or values.device != points.device
        or not torch.isfinite(values).all()
    ):
        raise ValueError("terrain quality surface sampler returned invalid heights")
    return values


def _batch_feasible_indices(
    *,
    cache: QualityActionCache,
    current: OracleState,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    constraints: OracleConstraints,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, Counter[str]]:
    count = len(cache.source_keys)
    device = current.root_position_world.device
    rejected: Counter[str] = Counter()
    eligible = torch.ones(count, dtype=torch.bool, device=device)

    exact_support = (cache.entry_support == current.support_mask[None]).all(dim=1)
    rejected["entry-support"] += int((eligible & ~exact_support).sum().item())
    eligible &= exact_support
    entry_xy = (
        _rotate_xy(cache.entry_foot_local[..., :2], current.root_yaw_world)
        + current.root_position_world[None, None, :2]
    )
    entry_z = cache.entry_foot_local[..., 2:] + current.root_position_world[2]
    entry_feet = torch.cat((entry_xy, entry_z), dim=2)
    entry_error = torch.linalg.vector_norm(
        entry_feet - current.foot_position_world[None], dim=2
    )
    foot_ok = ~(
        entry_error[:, current.support_mask]
        > float(constraints.maximum_entry_foot_error_m)
    ).any(dim=1)
    rejected["entry-foot-error"] += int((eligible & ~foot_ok).sum().item())
    eligible &= foot_ok
    position_ok = torch.linalg.vector_norm(
        cache.entry_joint_position - current.joint_position[None], dim=1
    ) <= float(constraints.maximum_joint_position_error_rad)
    rejected["joint-position"] += int((eligible & ~position_ok).sum().item())
    eligible &= position_ok
    velocity_ok = torch.linalg.vector_norm(
        cache.entry_joint_velocity - current.joint_velocity[None], dim=1
    ) <= float(constraints.maximum_joint_velocity_error_rad_s)
    rejected["joint-velocity"] += int((eligible & ~velocity_ok).sum().item())
    eligible &= velocity_ok
    entry_candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    accepted_chunks = []
    clearance_chunks = []
    terrain_names = (
        "stance-height",
        "landing-height",
        "landing-edge-margin",
        "swing-penetration",
        "height-deformation",
    )
    for start in range(0, int(entry_candidates.numel()), chunk_size):
        candidates = entry_candidates[start : start + chunk_size]
        local = cache.foot_trajectory_local[candidates]
        foot_xy = (
            _rotate_xy(local[..., :2], current.root_yaw_world)
            + current.root_position_world[None, None, None, :2]
        )
        foot_z = local[..., 2] + current.root_position_world[2]
        foot_world = torch.cat((foot_xy, foot_z[..., None]), dim=3)
        surface = _surface_values(sample_surface, foot_world[..., :2])
        clearance = foot_world[..., 2] - surface - float(ANKLE_ORIGIN_SOLE_M)
        support = cache.support_trajectory[candidates]
        valid = cache.valid_frames[candidates]
        swing = cache.swing_foot[candidates]
        rows = torch.arange(candidates.numel(), dtype=torch.int64, device=device)
        terminal = cache.frame_steps[candidates]
        stance = support & valid[..., None]
        stance[rows, terminal, swing] = False
        stance_error = torch.where(
            stance, torch.abs(clearance), torch.zeros_like(clearance)
        ).amax(dim=(1, 2))
        landing_error = torch.abs(clearance[rows, terminal, swing])
        margin = float(constraints.edge_margin_m)
        offsets = torch.tensor(
            (
                (0.0, 0.0),
                (margin, 0.0),
                (-margin, 0.0),
                (0.0, margin),
                (0.0, -margin),
            ),
            dtype=foot_world.dtype,
            device=device,
        )
        landing_xy = foot_world[rows, terminal, swing, :2]
        edge = _surface_values(
            sample_surface, landing_xy[:, None] + offsets[None]
        )
        edge_range = edge.amax(dim=1) - edge.amin(dim=1)
        swing_samples = valid[..., None] & ~support
        minimum_swing = torch.where(
            swing_samples, clearance, torch.full_like(clearance, torch.inf)
        ).amin(dim=(1, 2))
        minimum_swing = torch.where(
            swing_samples.any(dim=(1, 2)), minimum_swing, landing_error
        )
        query_delta = surface[rows, terminal, swing] - surface[rows, 0, swing]
        source_delta = cache.surface_delta_m[candidates][rows, terminal, swing]
        deformation = torch.abs(query_delta - source_delta)
        reason = torch.zeros(candidates.numel(), dtype=torch.int64, device=device)
        checks = (
            stance_error > float(constraints.stance_height_tolerance_m),
            landing_error > float(constraints.landing_height_tolerance_m),
            edge_range > float(constraints.maximum_edge_height_range_m),
            minimum_swing < float(constraints.minimum_swing_clearance_m),
            deformation > float(constraints.maximum_height_deformation_m),
        )
        for code, failed in enumerate(checks, start=1):
            reason = torch.where((reason == 0) & failed, code, reason)
        counts = torch.bincount(reason, minlength=6).detach().cpu().tolist()
        for code, name in enumerate(terrain_names, start=1):
            if counts[code]:
                rejected[name] += int(counts[code])
        accepted = reason == 0
        accepted_chunks.append(candidates[accepted])
        clearance_chunks.append(minimum_swing[accepted])
    rejected += Counter()
    return (
        torch.cat(accepted_chunks) if accepted_chunks else entry_candidates[:0],
        torch.cat(clearance_chunks)
        if clearance_chunks
        else torch.empty(0, dtype=current.root_position_world.dtype, device=device),
        Counter({name: count for name, count in rejected.items() if count}),
    )


def rank_quality_actions(
    *,
    state: FrozenQualityState,
    index: ContactPhaseActionIndex,
    native_quality: Sequence[NativeActionQuality],
    desired_landing_world_xyz: object,
    command_target_world_xy: object,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    constraints: OracleConstraints,
    quality_cache: QualityActionCache | None = None,
    top_k: int = 5,
) -> QualityOracleResult:
    """Evaluate the complete action inventory without online shortlist pruning."""

    if not isinstance(state, FrozenQualityState):
        raise ValueError("terrain quality state is invalid")
    if not isinstance(index, ContactPhaseActionIndex):
        raise ValueError("terrain quality action index is invalid")
    if not isinstance(constraints, OracleConstraints) or not callable(sample_surface):
        raise ValueError("terrain quality feasibility inputs are invalid")
    desired = _vector(desired_landing_world_xyz, (3,), "desired landing")
    command_target = _vector(command_target_world_xy, (2,), "command target")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("terrain quality top_k must be positive")
    qualities = tuple(native_quality)
    if len(qualities) != len(index.actions) or any(
        not isinstance(quality, NativeActionQuality)
        or quality.action_index != action_index
        or quality.source_key != index.actions[action_index].source_key
        for action_index, quality in enumerate(qualities)
    ):
        raise ValueError("terrain quality native descriptors do not match actions")
    cache = build_quality_action_cache(index) if quality_cache is None else quality_cache
    if (
        not isinstance(cache, QualityActionCache)
        or cache.source_keys != tuple(action.source_key for action in index.actions)
    ):
        raise ValueError("terrain quality action cache does not match actions")

    if index.actions:
        reference = index.actions[0].root_position_local
        dtype, device = reference.dtype, reference.device
    else:
        dtype, device = torch.float32, torch.device("cpu")
    current = _oracle_state(state, dtype=dtype, device=device)
    desired_tensor = torch.as_tensor(desired, dtype=dtype, device=device)
    target_tensor = torch.as_tensor(command_target, dtype=dtype, device=device)
    accepted, minimum_swing, rejected = _batch_feasible_indices(
        cache=cache,
        current=current,
        sample_surface=sample_surface,
        constraints=constraints,
    )
    if accepted.numel():
        terminal_xy = (
            _rotate_xy(cache.terminal_foot_local[accepted, :2], current.root_yaw_world)
            + current.root_position_world[None, :2]
        )
        terminal_z = (
            cache.terminal_foot_local[accepted, 2] + current.root_position_world[2]
        )
        terminal_feet = torch.cat((terminal_xy, terminal_z[:, None]), dim=1)
        landing_cost = torch.linalg.vector_norm(
            terminal_feet - desired_tensor[None], dim=1
        )
        joint_position_cost = torch.linalg.vector_norm(
            cache.entry_joint_position[accepted] - current.joint_position[None], dim=1
        )
        joint_velocity_cost = torch.linalg.vector_norm(
            cache.entry_joint_velocity[accepted] - current.joint_velocity[None], dim=1
        )
        terminal_root_xy = (
            _rotate_xy(cache.terminal_root_local_xy[accepted], current.root_yaw_world)
            + current.root_position_world[None, :2]
        )
        displacement_cost = torch.linalg.vector_norm(
            terminal_root_xy - target_tensor[None], dim=1
        )
        facing_delta = (
            cache.terminal_yaw_local[accepted]
            + current.root_yaw_world
            - float(state.command_heading_world_yaw)
        )
        facing_cost = torch.abs(torch.atan2(torch.sin(facing_delta), torch.cos(facing_delta)))
        native_stance = torch.tensor(
            [qualities[int(value)].source_stance_drift_m for value in accepted.detach().cpu().tolist()],
            dtype=dtype,
            device=device,
        )
        clearance_margin = minimum_swing - float(constraints.minimum_swing_clearance_m)
        total_cost = (
            torch.square(landing_cost / 0.25)
            + torch.square(joint_position_cost / 1.50)
            + torch.square(joint_velocity_cost / 8.0)
            + torch.square(displacement_cost / 0.50)
            + torch.square(facing_cost / 0.50)
            + torch.square(native_stance / 0.10)
            + torch.square(torch.clamp(-clearance_margin, min=0.0) / 0.03)
        )
        metrics = torch.stack(
            (
                landing_cost,
                joint_position_cost,
                joint_velocity_cost,
                displacement_cost,
                facing_cost,
                native_stance,
                clearance_margin,
                total_cost,
            ),
            dim=1,
        ).detach().cpu().numpy()
        accepted_indices = accepted.detach().cpu().tolist()
    else:
        metrics = np.empty((0, 8), dtype=np.float64)
        accepted_indices = []
    feasible = [
        QualityCandidateScore(
            action_index=int(action_index),
            source_key=index.actions[int(action_index)].source_key,
            feasible=True,
            feasibility_reason=None,
            landing_error_m=float(values[0]),
            support_mismatch_count=0,
            joint_position_boundary_error_rad=float(values[1]),
            joint_velocity_boundary_error_rad_s=float(values[2]),
            command_displacement_error_m=float(values[3]),
            command_facing_error_rad=float(values[4]),
            native_stance_drift_m=float(values[5]),
            clearance_margin_m=float(values[6]),
            total_cost=float(values[7]),
            exact_successor_index=qualities[int(action_index)].exact_successor_index,
            two_step_total_cost=None,
        )
        for action_index, values in zip(accepted_indices, metrics)
    ]
    for row_index, row in enumerate(feasible):
        successor_index = row.exact_successor_index
        if successor_index is None:
            continue
        action = index.actions[row.action_index]
        placed = place_action(action, current)
        advanced = advance_state(current, placed)
        successor = index.actions[successor_index]
        if not bool(torch.equal(successor.entry_support, advanced.support_mask)):
            continue
        successor_placed = place_action(successor, advanced)
        successor_validation = validate_placement(
            placed=successor_placed,
            state=advanced,
            sample_surface=sample_surface,
            constraints=constraints,
        )
        if not successor_validation.accepted:
            continue
        successor_native = qualities[successor_index]
        successor_position = float(
            torch.linalg.vector_norm(
                successor.joint_position[0] - advanced.joint_position
            ).item()
        )
        successor_velocity = float(
            torch.linalg.vector_norm(
                successor.joint_velocity[0] - advanced.joint_velocity
            ).item()
        )
        successor_margin = (
            float(successor_validation.minimum_swing_clearance_m)
            - float(constraints.minimum_swing_clearance_m)
        )
        feasible[row_index] = replace(
            row,
            two_step_total_cost=row.total_cost
            + _combined_cost(
                landing=0.0,
                joint_position=successor_position,
                joint_velocity=successor_velocity,
                displacement=0.0,
                facing=0.0,
                stance_drift=successor_native.source_stance_drift_m,
                clearance_margin=successor_margin,
            ),
        )

    source_tie = lambda row: row.source_key
    landing_rows = tuple(
        sorted(feasible, key=lambda row: (row.landing_error_m, source_tie(row)))[:top_k]
    )
    continuity_rows = tuple(
        sorted(
            feasible,
            key=lambda row: (
                row.joint_position_boundary_error_rad,
                row.joint_velocity_boundary_error_rad_s,
                source_tie(row),
            ),
        )[:top_k]
    )
    combined_rows = tuple(
        sorted(feasible, key=lambda row: (row.total_cost, source_tie(row)))[:top_k]
    )
    two_step_rows = tuple(
        sorted(
            (row for row in feasible if row.two_step_total_cost is not None),
            key=lambda row: (float(row.two_step_total_cost), source_tie(row)),
        )[:top_k]
    )
    rankings = {
        "best_landing": landing_rows,
        "best_continuity": continuity_rows,
        "best_combined": combined_rows,
        "best_two_step": two_step_rows,
    }
    digest = _digest(state.state_id, len(index.actions), rejected, rankings)
    return QualityOracleResult(
        state_id=state.state_id,
        evaluated_action_count=len(index.actions),
        rejected_by_reason=rejected,
        best_landing=landing_rows,
        best_continuity=continuity_rows,
        best_combined=combined_rows,
        best_two_step=two_step_rows,
        deterministic_sha256=digest,
    )
