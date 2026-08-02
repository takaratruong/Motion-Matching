"""Exhaustive, multi-objective ranking for frozen terrain quality states."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from .torch_contact_oracle_actions import ContactPhaseActionIndex
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


def rank_quality_actions(
    *,
    state: FrozenQualityState,
    index: ContactPhaseActionIndex,
    native_quality: Sequence[NativeActionQuality],
    desired_landing_world_xyz: object,
    command_target_world_xy: object,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    constraints: OracleConstraints,
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

    if index.actions:
        reference = index.actions[0].root_position_local
        dtype, device = reference.dtype, reference.device
    else:
        dtype, device = torch.float32, torch.device("cpu")
    current = _oracle_state(state, dtype=dtype, device=device)
    desired_tensor = torch.as_tensor(desired, dtype=dtype, device=device)
    target_tensor = torch.as_tensor(command_target, dtype=dtype, device=device)
    rejected: Counter[str] = Counter()
    feasible: list[QualityCandidateScore] = []

    for action_index, (action, native) in enumerate(zip(index.actions, qualities)):
        support_mismatch = int(
            torch.count_nonzero(action.entry_support != current.support_mask).item()
        )
        if support_mismatch:
            rejected["entry-support"] += 1
            continue
        placed = place_action(action, current)
        validation = validate_placement(
            placed=placed,
            state=current,
            sample_surface=sample_surface,
            constraints=constraints,
        )
        if not validation.accepted:
            assert validation.reason is not None
            rejected[validation.reason] += 1
            continue

        landing_error = float(
            torch.linalg.vector_norm(
                placed.foot_position_world[-1, action.swing_foot] - desired_tensor
            ).item()
        )
        joint_position = float(
            torch.linalg.vector_norm(action.joint_position[0] - current.joint_position).item()
        )
        joint_velocity = float(
            torch.linalg.vector_norm(action.joint_velocity[0] - current.joint_velocity).item()
        )
        displacement = float(
            torch.linalg.vector_norm(placed.root_position_world[-1, :2] - target_tensor).item()
        )
        facing = _angle_error(
            float(placed.root_yaw_world[-1].item()),
            state.command_heading_world_yaw,
        )
        clearance_margin = (
            float(validation.minimum_swing_clearance_m)
            - float(constraints.minimum_swing_clearance_m)
        )
        total = _combined_cost(
            landing=landing_error,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            displacement=displacement,
            facing=facing,
            stance_drift=native.source_stance_drift_m,
            clearance_margin=clearance_margin,
        )

        two_step_total = None
        successor_index = native.exact_successor_index
        if successor_index is not None:
            successor = index.actions[successor_index]
            advanced = advance_state(current, placed)
            if bool(torch.equal(successor.entry_support, advanced.support_mask)):
                successor_placed = place_action(successor, advanced)
                successor_validation = validate_placement(
                    placed=successor_placed,
                    state=advanced,
                    sample_surface=sample_surface,
                    constraints=constraints,
                )
                if successor_validation.accepted:
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
                    two_step_total = total + _combined_cost(
                        landing=0.0,
                        joint_position=successor_position,
                        joint_velocity=successor_velocity,
                        displacement=0.0,
                        facing=0.0,
                        stance_drift=successor_native.source_stance_drift_m,
                        clearance_margin=successor_margin,
                    )

        feasible.append(
            QualityCandidateScore(
                action_index=action_index,
                source_key=action.source_key,
                feasible=True,
                feasibility_reason=None,
                landing_error_m=landing_error,
                support_mismatch_count=0,
                joint_position_boundary_error_rad=joint_position,
                joint_velocity_boundary_error_rad_s=joint_velocity,
                command_displacement_error_m=displacement,
                command_facing_error_rad=facing,
                native_stance_drift_m=native.source_stance_drift_m,
                clearance_margin_m=clearance_margin,
                total_cost=total,
                exact_successor_index=successor_index,
                two_step_total_cost=two_step_total,
            )
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
