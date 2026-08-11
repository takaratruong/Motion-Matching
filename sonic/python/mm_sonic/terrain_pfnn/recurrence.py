"""Differentiable recurrent trajectory planning for terrain PFNNs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from .layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
    TRAJECTORY_TIMES_S,
    VectorLayout,
)


_DT = 1.0 / 30.0
_TWO_PI = 2.0 * math.pi
_STATIONARY_SPEED_M_S = 0.05
_DIRECTION_EPSILON = 1.0e-12


def _validate_tensor(
    name: str,
    value: Tensor,
    shape: tuple[int, ...],
    *,
    reference: Tensor | None = None,
) -> None:
    if not isinstance(value, Tensor) or value.shape != shape:
        raise ValueError(f"{name} must be a tensor with shape {shape}")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if reference is not None and (
        value.dtype != reference.dtype or value.device != reference.device
    ):
        raise ValueError("all tensors must use a shared floating dtype and device")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _validate_directions(name: str, value: Tensor) -> None:
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise ValueError(f"{name} must contain normalized directions")


def _validate_semantics(name: str, value: Tensor) -> None:
    if not bool(torch.all((value == 0.0) | (value == 1.0))) or not bool(
        torch.all(torch.sum(value, dim=-1) == 1.0)
    ):
        raise ValueError(f"{name} must contain one-hot semantics")


def _rotation_2d(yaw: Tensor) -> Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            torch.stack((cosine, -sine), dim=-1),
            torch.stack((sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def _world_from_local(vector: Tensor, yaw: Tensor) -> Tensor:
    return torch.einsum("b...i,bji->b...j", vector, _rotation_2d(yaw))


def _local_from_world(vector: Tensor, yaw: Tensor) -> Tensor:
    return torch.einsum("b...i,bij->b...j", vector, _rotation_2d(yaw))


def _integrate_root_motion(
    state: RecurrentTrajectoryState,
    physical_output: Tensor,
) -> tuple[Tensor, Tensor]:
    local_velocity = physical_output[:, OUTPUT_LAYOUT["root_planar_velocity"]]
    root_world_xy = state.root_world_xy + _DT * _world_from_local(
        local_velocity, state.root_yaw_world
    )
    yaw_velocity = physical_output[:, OUTPUT_LAYOUT["root_yaw_velocity"]][:, 0]
    root_yaw_world = state.root_yaw_world + _DT * yaw_velocity
    return root_world_xy, root_yaw_world


@dataclass(frozen=True)
class RecurrentTrajectoryState:
    root_world_xy: Tensor
    root_yaw_world: Tensor
    history_position_world_xy: Tensor
    history_direction_world_xy: Tensor
    history_semantic_intent: Tensor
    predicted_position_world_xy: Tensor
    predicted_direction_world_xy: Tensor
    previous_body_position_local: Tensor
    previous_body_velocity_local: Tensor
    joint_position: Tensor
    joint_velocity: Tensor
    phase: Tensor

    def __post_init__(self) -> None:
        batch_size = (
            int(self.root_world_xy.shape[0])
            if isinstance(self.root_world_xy, Tensor) and self.root_world_xy.ndim >= 1
            else -1
        )
        reference = self.root_world_xy
        _validate_tensor("root_world_xy", self.root_world_xy, (batch_size, 2))
        fields = (
            ("root_yaw_world", self.root_yaw_world, (batch_size,)),
            (
                "history_position_world_xy",
                self.history_position_world_xy,
                (batch_size, 31, 2),
            ),
            (
                "history_direction_world_xy",
                self.history_direction_world_xy,
                (batch_size, 31, 2),
            ),
            (
                "history_semantic_intent",
                self.history_semantic_intent,
                (batch_size, 31, 2),
            ),
            (
                "predicted_position_world_xy",
                self.predicted_position_world_xy,
                (batch_size, 12, 2),
            ),
            (
                "predicted_direction_world_xy",
                self.predicted_direction_world_xy,
                (batch_size, 12, 2),
            ),
            (
                "previous_body_position_local",
                self.previous_body_position_local,
                (batch_size, 30, 3),
            ),
            (
                "previous_body_velocity_local",
                self.previous_body_velocity_local,
                (batch_size, 30, 3),
            ),
            ("joint_position", self.joint_position, (batch_size, 29)),
            ("joint_velocity", self.joint_velocity, (batch_size, 29)),
            ("phase", self.phase, (batch_size,)),
        )
        for name, value, shape in fields:
            _validate_tensor(name, value, shape, reference=reference)
        _validate_directions(
            "history_direction_world_xy", self.history_direction_world_xy
        )
        _validate_directions(
            "predicted_direction_world_xy", self.predicted_direction_world_xy
        )
        _validate_semantics("history_semantic_intent", self.history_semantic_intent)


@dataclass(frozen=True)
class PlannedTrajectory:
    position_world_xy: Tensor
    direction_world_xy: Tensor
    semantic_intent: Tensor

    def __post_init__(self) -> None:
        batch_size = (
            int(self.position_world_xy.shape[0])
            if isinstance(self.position_world_xy, Tensor)
            and self.position_world_xy.ndim >= 1
            else -1
        )
        reference = self.position_world_xy
        _validate_tensor(
            "position_world_xy", self.position_world_xy, (batch_size, 12, 2)
        )
        _validate_tensor(
            "direction_world_xy",
            self.direction_world_xy,
            (batch_size, 12, 2),
            reference=reference,
        )
        _validate_tensor(
            "semantic_intent",
            self.semantic_intent,
            (batch_size, 12, 2),
            reference=reference,
        )
        _validate_directions("direction_world_xy", self.direction_world_xy)
        _validate_semantics("semantic_intent", self.semantic_intent)


def initialize_recurrent_state(
    *,
    trajectory_position_local: Tensor,
    trajectory_direction_local: Tensor,
    semantic_intent: Tensor,
    previous_body_position_local: Tensor,
    previous_body_velocity_local: Tensor,
    phase: Tensor,
    root_world_xy: Tensor,
    root_yaw_world: Tensor,
    joint_position: Tensor | None = None,
    joint_velocity: Tensor | None = None,
) -> RecurrentTrajectoryState:
    if not isinstance(trajectory_position_local, Tensor) or trajectory_position_local.ndim != 3:
        raise ValueError("trajectory_position_local must have shape (B, 12, 2)")
    batch_size = int(trajectory_position_local.shape[0])
    reference = trajectory_position_local
    _validate_tensor(
        "trajectory_position_local",
        trajectory_position_local,
        (batch_size, 12, 2),
    )
    fields = (
        (
            "trajectory_direction_local",
            trajectory_direction_local,
            (batch_size, 12, 2),
        ),
        ("semantic_intent", semantic_intent, (batch_size, 12, 2)),
        (
            "previous_body_position_local",
            previous_body_position_local,
            (batch_size, 30, 3),
        ),
        (
            "previous_body_velocity_local",
            previous_body_velocity_local,
            (batch_size, 30, 3),
        ),
        ("phase", phase, (batch_size,)),
        ("root_world_xy", root_world_xy, (batch_size, 2)),
        ("root_yaw_world", root_yaw_world, (batch_size,)),
    )
    for name, value, shape in fields:
        _validate_tensor(name, value, shape, reference=reference)
    _validate_directions("trajectory_direction_local", trajectory_direction_local)
    _validate_semantics("semantic_intent", semantic_intent)
    if joint_position is None:
        joint_position = torch.zeros(
            (batch_size, 29), dtype=reference.dtype, device=reference.device
        )
    if joint_velocity is None:
        joint_velocity = torch.zeros(
            (batch_size, 29), dtype=reference.dtype, device=reference.device
        )
    _validate_tensor(
        "joint_position",
        joint_position,
        (batch_size, 29),
        reference=reference,
    )
    _validate_tensor(
        "joint_velocity",
        joint_velocity,
        (batch_size, 29),
        reference=reference,
    )

    rotation = _rotation_2d(root_yaw_world)
    predicted_position_world_xy = root_world_xy[:, None] + torch.matmul(
        trajectory_position_local, rotation.transpose(-1, -2)
    )
    predicted_direction_world_xy = torch.matmul(
        trajectory_direction_local, rotation.transpose(-1, -2)
    )
    past_position = predicted_position_world_xy[:, :7]
    past_direction = predicted_direction_world_xy[:, :7]
    past_semantic = semantic_intent[:, :7]
    sample_index = torch.arange(31, device=reference.device)
    left_index = torch.clamp(torch.div(sample_index, 5, rounding_mode="floor"), max=5)
    alpha = ((sample_index - 5 * left_index).to(reference.dtype) / 5.0).reshape(
        1, 31, 1
    )
    right_index = left_index + 1
    history_position = (
        (1.0 - alpha) * past_position[:, left_index]
        + alpha * past_position[:, right_index]
    )
    interpolated_direction = (
        (1.0 - alpha) * past_direction[:, left_index]
        + alpha * past_direction[:, right_index]
    )
    direction_norm = torch.linalg.vector_norm(
        interpolated_direction, dim=-1, keepdim=True
    )
    if bool(torch.any(direction_norm < _DIRECTION_EPSILON)):
        raise ValueError("interpolated history directions must be nonzero")
    history_direction = interpolated_direction / direction_norm
    interpolated_semantic = (
        (1.0 - alpha) * past_semantic[:, left_index]
        + alpha * past_semantic[:, right_index]
    )
    history_semantic = torch.nn.functional.one_hot(
        torch.argmax(interpolated_semantic, dim=-1), num_classes=2
    ).to(reference.dtype)
    return RecurrentTrajectoryState(
        root_world_xy=root_world_xy,
        root_yaw_world=root_yaw_world,
        history_position_world_xy=history_position,
        history_direction_world_xy=history_direction,
        history_semantic_intent=history_semantic,
        predicted_position_world_xy=predicted_position_world_xy,
        predicted_direction_world_xy=predicted_direction_world_xy,
        previous_body_position_local=previous_body_position_local,
        previous_body_velocity_local=previous_body_velocity_local,
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        phase=phase,
    )


def derive_training_desired_velocity(
    trajectory_position_local: Tensor,
    semantic_intent: Tensor,
    root_yaw_world: Tensor,
) -> Tensor:
    if not isinstance(trajectory_position_local, Tensor) or trajectory_position_local.ndim != 3:
        raise ValueError("trajectory_position_local must have shape (B, 12, 2)")
    batch_size = int(trajectory_position_local.shape[0])
    reference = trajectory_position_local
    _validate_tensor(
        "trajectory_position_local",
        trajectory_position_local,
        (batch_size, 12, 2),
    )
    _validate_tensor(
        "semantic_intent",
        semantic_intent,
        (batch_size, 12, 2),
        reference=reference,
    )
    _validate_tensor(
        "root_yaw_world",
        root_yaw_world,
        (batch_size,),
        reference=reference,
    )
    _validate_semantics("semantic_intent", semantic_intent)
    duration = float(TRAJECTORY_TIMES_S[11] - TRAJECTORY_TIMES_S[6])
    desired_local = (
        trajectory_position_local[:, 11] - trajectory_position_local[:, 6]
    ) / duration
    idle = semantic_intent[:, 11, 0:1] == 1.0
    desired_local = torch.where(idle, torch.zeros_like(desired_local), desired_local)
    return _world_from_local(desired_local, root_yaw_world)


def plan_recurrent_trajectory(
    state: RecurrentTrajectoryState,
    desired_velocity_world: Tensor,
) -> PlannedTrajectory:
    if not isinstance(state, RecurrentTrajectoryState):
        raise TypeError("state must be RecurrentTrajectoryState")
    batch_size = int(state.root_world_xy.shape[0])
    _validate_tensor(
        "desired_velocity_world",
        desired_velocity_world,
        (batch_size, 2),
        reference=state.root_world_xy,
    )
    history_index = torch.arange(0, 31, 5, device=state.root_world_xy.device)
    past_position = state.history_position_world_xy[:, history_index]
    past_direction = state.history_direction_world_xy[:, history_index]
    past_semantic = state.history_semantic_intent[:, history_index]

    desired_speed = torch.linalg.vector_norm(
        desired_velocity_world, dim=-1, keepdim=True
    )
    desired_direction = torch.where(
        desired_speed >= _DIRECTION_EPSILON,
        desired_velocity_world / torch.clamp_min(desired_speed, _DIRECTION_EPSILON),
        past_direction[:, 6],
    )
    current_semantic = torch.nn.functional.one_hot(
        (desired_speed[:, 0] >= _STATIONARY_SPEED_M_S).to(torch.long),
        num_classes=2,
    ).to(state.root_world_xy.dtype)
    position_values = [past_position[:, index] for index in range(6)]
    position_values.append(state.root_world_xy)
    direction_values = [past_direction[:, index] for index in range(7)]
    semantic_values = [past_semantic[:, index] for index in range(6)]
    semantic_values.append(current_semantic)

    times = torch.as_tensor(
        TRAJECTORY_TIMES_S,
        dtype=state.root_world_xy.dtype,
        device=state.root_world_xy.device,
    )
    for index in range(7, 12):
        knot_dt = times[index] - times[index - 1]
        predicted_velocity = (
            state.predicted_position_world_xy[:, index]
            - state.predicted_position_world_xy[:, index - 1]
        ) / knot_dt
        u = times[index] / times[-1]
        # Holden et al.'s non-responsive trajectory bias: future knots nearer
        # the character retain more of the PFNN prediction, while distant
        # knots converge to the requested velocity/direction.
        velocity_weight = 1.0 - torch.sqrt(1.0 - u)
        velocity = (
            (1.0 - velocity_weight) * predicted_velocity
            + velocity_weight * desired_velocity_world
        )
        position_values.append(position_values[-1] + knot_dt * velocity)
        facing_weight = 1.0 - (1.0 - u).square()
        facing = (
            (1.0 - facing_weight) * state.predicted_direction_world_xy[:, index]
            + facing_weight * desired_direction
        )
        facing_norm = torch.linalg.vector_norm(facing, dim=-1, keepdim=True)
        normalized_facing = torch.where(
            facing_norm >= _DIRECTION_EPSILON,
            facing / torch.clamp_min(facing_norm, _DIRECTION_EPSILON),
            desired_direction,
        )
        direction_values.append(normalized_facing)
        semantic_values.append(
            torch.nn.functional.one_hot(
                (
                    torch.linalg.vector_norm(velocity, dim=-1)
                    >= _STATIONARY_SPEED_M_S
                ).to(torch.long),
                num_classes=2,
            ).to(state.root_world_xy.dtype)
        )
    return PlannedTrajectory(
        position_world_xy=torch.stack(position_values, dim=1),
        direction_world_xy=torch.stack(direction_values, dim=1),
        semantic_intent=torch.stack(semantic_values, dim=1),
    )


def pack_recurrent_input(
    *,
    state: RecurrentTrajectoryState,
    planned: PlannedTrajectory,
    terrain_height: Tensor,
    x_mean: Tensor,
    x_std: Tensor,
    body_scale: float = 0.1,
    input_layout: VectorLayout = INPUT_LAYOUT,
) -> Tensor:
    if not isinstance(state, RecurrentTrajectoryState):
        raise TypeError("state must be RecurrentTrajectoryState")
    if not isinstance(planned, PlannedTrajectory):
        raise TypeError("planned must be PlannedTrajectory")
    if input_layout not in (INPUT_LAYOUT, CLASSIC_G1_INPUT_LAYOUT_V3):
        raise ValueError("recurrent input layout is unsupported")
    batch_size = int(state.root_world_xy.shape[0])
    reference = state.root_world_xy
    fields = (
        ("planned position_world_xy", planned.position_world_xy, (batch_size, 12, 2)),
        (
            "planned direction_world_xy",
            planned.direction_world_xy,
            (batch_size, 12, 2),
        ),
        ("planned semantic_intent", planned.semantic_intent, (batch_size, 12, 2)),
        ("terrain_height", terrain_height, (batch_size, 12, 3)),
        ("x_mean", x_mean, (input_layout.size,)),
        ("x_std", x_std, (input_layout.size,)),
    )
    for name, value, shape in fields:
        _validate_tensor(name, value, shape, reference=reference)
    if not bool(torch.all(x_std > 0.0)):
        raise ValueError("x_std must be positive")
    if (
        isinstance(body_scale, bool)
        or not isinstance(body_scale, Real)
        or not math.isfinite(float(body_scale))
        or float(body_scale) <= 0.0
    ):
        raise ValueError("body_scale must be finite and positive")
    local_position = _local_from_world(
        planned.position_world_xy - state.root_world_xy[:, None],
        state.root_yaw_world,
    )
    local_direction = _local_from_world(
        planned.direction_world_xy, state.root_yaw_world
    )
    fields = [
        local_position.reshape(batch_size, -1),
        local_direction.reshape(batch_size, -1),
        terrain_height.reshape(batch_size, -1),
        planned.semantic_intent.reshape(batch_size, -1),
        state.previous_body_position_local.reshape(batch_size, -1),
        state.previous_body_velocity_local.reshape(batch_size, -1),
    ]
    if input_layout == CLASSIC_G1_INPUT_LAYOUT_V3:
        fields.extend((state.joint_position, state.joint_velocity))
    raw = torch.cat(fields, dim=-1)
    normalized = (raw - x_mean) / x_std
    packed = normalized.clone()
    for field in ("previous_body_position", "previous_body_velocity"):
        packed[:, input_layout[field]] *= float(body_scale)
    if not bool(torch.isfinite(packed).all()):
        raise ValueError("packed recurrent input must be finite")
    return packed


def advance_recurrent_state(
    state: RecurrentTrajectoryState,
    planned: PlannedTrajectory,
    physical_output: Tensor,
    *,
    phase_advance_cap: Tensor,
) -> RecurrentTrajectoryState:
    if not isinstance(state, RecurrentTrajectoryState):
        raise TypeError("state must be RecurrentTrajectoryState")
    if not isinstance(planned, PlannedTrajectory):
        raise TypeError("planned must be PlannedTrajectory")
    batch_size = int(state.root_world_xy.shape[0])
    reference = state.root_world_xy
    fields = (
        ("planned position_world_xy", planned.position_world_xy, (batch_size, 12, 2)),
        (
            "planned direction_world_xy",
            planned.direction_world_xy,
            (batch_size, 12, 2),
        ),
        ("planned semantic_intent", planned.semantic_intent, (batch_size, 12, 2)),
        ("physical_output", physical_output, (batch_size, OUTPUT_LAYOUT.size)),
        ("phase_advance_cap", phase_advance_cap, (batch_size,)),
    )
    for name, value, shape in fields:
        _validate_tensor(name, value, shape, reference=reference)
    if not bool(torch.all(phase_advance_cap >= 0.0)):
        raise ValueError("phase_advance_cap must be nonnegative")

    new_root_world_xy, new_root_yaw_world = _integrate_root_motion(
        state, physical_output
    )
    local_position = physical_output[
        :, OUTPUT_LAYOUT["trajectory_position"]
    ].reshape(batch_size, 12, 2)
    local_direction = physical_output[
        :, OUTPUT_LAYOUT["trajectory_direction"]
    ].reshape(batch_size, 12, 2)
    local_direction_norm = torch.linalg.vector_norm(
        local_direction, dim=-1, keepdim=True
    )
    if bool(torch.any(local_direction_norm < _DIRECTION_EPSILON)):
        raise ValueError("physical trajectory directions must be nonzero")
    local_direction = local_direction / local_direction_norm
    new_predicted_position_world_xy = new_root_world_xy[:, None] + _world_from_local(
        local_position, new_root_yaw_world
    )
    new_predicted_direction_world_xy = _world_from_local(
        local_direction, new_root_yaw_world
    )
    realized_facing = new_predicted_direction_world_xy[:, 6]
    realized_semantic = planned.semantic_intent[:, 6]
    phase_advance = torch.minimum(
        torch.clamp_min(
            physical_output[:, OUTPUT_LAYOUT["phase_advance"]][:, 0], 0.0
        ),
        phase_advance_cap,
    )
    phase = torch.remainder(state.phase + phase_advance, _TWO_PI)
    return RecurrentTrajectoryState(
        root_world_xy=new_root_world_xy,
        root_yaw_world=new_root_yaw_world,
        history_position_world_xy=torch.cat(
            (state.history_position_world_xy[:, 1:], new_root_world_xy[:, None]),
            dim=1,
        ),
        history_direction_world_xy=torch.cat(
            (state.history_direction_world_xy[:, 1:], realized_facing[:, None]),
            dim=1,
        ),
        history_semantic_intent=torch.cat(
            (state.history_semantic_intent[:, 1:], realized_semantic[:, None]),
            dim=1,
        ),
        predicted_position_world_xy=new_predicted_position_world_xy,
        predicted_direction_world_xy=new_predicted_direction_world_xy,
        previous_body_position_local=physical_output[
            :, OUTPUT_LAYOUT["body_position"]
        ].reshape(batch_size, 30, 3),
        previous_body_velocity_local=physical_output[
            :, OUTPUT_LAYOUT["body_velocity"]
        ].reshape(batch_size, 30, 3),
        joint_position=physical_output[:, OUTPUT_LAYOUT["joint_position"]],
        joint_velocity=(
            physical_output[:, OUTPUT_LAYOUT["joint_position"]]
            - state.joint_position
        )
        / _DT,
        phase=phase,
    )
