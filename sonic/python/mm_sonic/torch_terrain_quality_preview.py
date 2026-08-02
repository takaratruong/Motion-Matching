"""Stage-isolated previews for terrain contact-action transitions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from .torch_contact_oracle_actions import ContactPhaseAction
from .torch_contact_oracle_search import OracleState, place_action
from .torch_motion_matcher import decay_spring_offsets
from .torch_terrain_quality_states import FrozenQualityState


def _readonly(value: object, shape: tuple[int, ...], name: str, *, boolean=False):
    array = np.array(value, dtype=np.bool_ if boolean else np.float64, copy=True)
    if array.shape != shape or (not boolean and not np.isfinite(array).all()):
        raise ValueError(f"terrain quality preview {name} is invalid")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class QualityPreviewStage:
    qpos: np.ndarray
    foot_position_world: np.ndarray
    source_support_mask: np.ndarray
    joint_boundary_jump_rad: float
    joint_velocity_boundary_jump_rad_s: float
    root_boundary_jump_m: float
    source_stance_drift_m: float

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or qpos.shape[0] < 2:
            raise ValueError("terrain quality preview qpos is invalid")
        frames = qpos.shape[0]
        object.__setattr__(self, "qpos", _readonly(qpos, (frames, 36), "qpos"))
        object.__setattr__(
            self,
            "foot_position_world",
            _readonly(
                self.foot_position_world, (frames, 2, 3), "foot positions"
            ),
        )
        object.__setattr__(
            self,
            "source_support_mask",
            _readonly(
                self.source_support_mask,
                (frames, 2),
                "source support",
                boolean=True,
            ),
        )
        for name in (
            "joint_boundary_jump_rad",
            "joint_velocity_boundary_jump_rad_s",
            "root_boundary_jump_m",
            "source_stance_drift_m",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"terrain quality preview {name} is invalid")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class QualityPreview:
    native: QualityPreviewStage
    placed: QualityPreviewStage
    composed: QualityPreviewStage

    def __post_init__(self) -> None:
        if any(
            not isinstance(stage, QualityPreviewStage)
            for stage in (self.native, self.placed, self.composed)
        ):
            raise ValueError("terrain quality preview stages are invalid")


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_inverse(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[..., :1], -value[..., 1:]), dim=-1)


def _quat_to_scaled_axis(value: torch.Tensor) -> torch.Tensor:
    value = _quat_normalize(value)
    value = torch.where(value[..., :1] < 0.0, -value, value)
    vector_length = torch.linalg.vector_norm(value[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_length, value[..., :1])
    return value[..., 1:] * (angle / vector_length.clamp_min(1e-8))


def _quat_from_scaled_axis(value: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = torch.sin(angle / 2.0) / angle.clamp_min(1e-8)
    quaternion = torch.cat((torch.cos(angle / 2.0), value * scale), dim=-1)
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    return _quat_normalize(torch.where(angle < 1e-8, identity, quaternion))


def _state_as_oracle(
    state: FrozenQualityState, reference: torch.Tensor
) -> OracleState:
    dtype, device = reference.dtype, reference.device

    def tensor(value: object) -> torch.Tensor:
        return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)

    quaternion = tensor(state.qpos[3:7])
    if float(torch.linalg.vector_norm(quaternion).item()) < 1e-8:
        yaw = tensor(state.root_yaw_world)
        zero = torch.zeros_like(yaw)
        quaternion = torch.stack(
            (torch.cos(yaw / 2.0), zero, zero, torch.sin(yaw / 2.0))
        )
    quaternion = _quat_normalize(quaternion)
    return OracleState(
        root_position_world=tensor(state.root_position_world),
        root_yaw_world=tensor(state.root_yaw_world),
        root_orientation_world_wxyz=quaternion,
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


def _feet(
    foot_kinematics: Any,
    joints: torch.Tensor,
    roots: torch.Tensor,
    quaternions: torch.Tensor,
) -> np.ndarray:
    function = getattr(foot_kinematics, "foot_positions", None)
    if not callable(function):
        raise ValueError("terrain quality preview foot kinematics is invalid")
    try:
        value = function(
            joints.detach().cpu().numpy(),
            roots.detach().cpu().numpy(),
            quaternions.detach().cpu().numpy(),
        )
    except Exception as error:
        raise ValueError("terrain quality preview foot kinematics failed") from error
    feet = np.asarray(value, dtype=np.float64)
    if feet.shape != (joints.shape[0], 2, 3) or not np.isfinite(feet).all():
        raise ValueError("terrain quality preview foot kinematics returned invalid feet")
    return feet


def _stance_drift(feet: np.ndarray, support: np.ndarray) -> float:
    displacement = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
    intervals = support[:-1] & support[1:]
    return float(np.where(intervals, displacement, 0.0).sum())


def _stage(
    *,
    state: FrozenQualityState,
    joints: torch.Tensor,
    velocities: torch.Tensor,
    roots: torch.Tensor,
    quaternions: torch.Tensor,
    support: np.ndarray,
    foot_kinematics: Any,
) -> QualityPreviewStage:
    qpos = torch.cat((roots, quaternions, joints), dim=1).detach().cpu().numpy()
    feet = _feet(foot_kinematics, joints, roots, quaternions)
    joints_np = joints.detach().cpu().numpy()
    velocity_np = velocities.detach().cpu().numpy()
    roots_np = roots.detach().cpu().numpy()
    return QualityPreviewStage(
        qpos=qpos,
        foot_position_world=feet,
        source_support_mask=support,
        joint_boundary_jump_rad=float(
            np.linalg.norm(joints_np[0] - state.joint_position)
        ),
        joint_velocity_boundary_jump_rad_s=float(
            np.linalg.norm(velocity_np[0] - state.joint_velocity)
        ),
        root_boundary_jump_m=float(
            np.linalg.norm(roots_np[0] - state.root_position_world)
        ),
        source_stance_drift_m=_stance_drift(feet, support),
    )


def build_quality_preview(
    *,
    state: FrozenQualityState,
    action: ContactPhaseAction,
    foot_kinematics: Any,
    inertialization_halflife_s: float = 0.10,
) -> QualityPreview:
    """Build native, rigidly placed, and inertialized 50 Hz action stages."""

    if not isinstance(state, FrozenQualityState) or not isinstance(
        action, ContactPhaseAction
    ):
        raise ValueError("terrain quality preview inputs are invalid")
    if (
        isinstance(inertialization_halflife_s, bool)
        or not isinstance(inertialization_halflife_s, (int, float))
        or not math.isfinite(float(inertialization_halflife_s))
        or float(inertialization_halflife_s) <= 0.0
    ):
        raise ValueError("terrain quality preview halflife must be positive")

    current = _state_as_oracle(state, action.joint_position)
    placed = place_action(action, current)
    support = action.support_mask.detach().cpu().numpy().astype(bool, copy=True)
    native = _stage(
        state=state,
        joints=action.joint_position,
        velocities=action.joint_velocity,
        roots=action.root_position_local,
        quaternions=action.root_orientation_local_wxyz,
        support=support,
        foot_kinematics=foot_kinematics,
    )
    rigid = _stage(
        state=state,
        joints=action.joint_position,
        velocities=action.joint_velocity,
        roots=placed.root_position_world,
        quaternions=placed.root_orientation_world_wxyz,
        support=support,
        foot_kinematics=foot_kinematics,
    )

    frames = action.frame_count
    times = torch.arange(
        frames,
        dtype=action.joint_position.dtype,
        device=action.joint_position.device,
    ) / 50.0
    joint_position_offset = current.joint_position - action.joint_position[0]
    joint_velocity_offset = current.joint_velocity - action.joint_velocity[0]
    root_position_offset = current.root_position_world - placed.root_position_world[0]
    if frames > 1:
        placed_root_velocity = (
            placed.root_position_world[1] - placed.root_position_world[0]
        ) * 50.0
        yaw_rate = torch.atan2(
            torch.sin(placed.root_yaw_world[1] - placed.root_yaw_world[0]),
            torch.cos(placed.root_yaw_world[1] - placed.root_yaw_world[0]),
        ) * 50.0
    else:
        placed_root_velocity = torch.zeros_like(root_position_offset)
        yaw_rate = torch.zeros_like(current.root_yaw_world)
    current_root_velocity = torch.cat(
        (
            torch.as_tensor(
                np.array(state.command_velocity_world_xy, copy=True),
                dtype=action.joint_position.dtype,
                device=action.joint_position.device,
            ),
            torch.zeros(1, dtype=action.joint_position.dtype, device=action.joint_position.device),
        )
    )
    root_velocity_offset = current_root_velocity - placed_root_velocity
    rotation_offset = _quat_to_scaled_axis(
        _quat_mul(
            current.root_orientation_world_wxyz,
            _quat_inverse(placed.root_orientation_world_wxyz[0]),
        )
    )
    angular_velocity_offset = torch.stack(
        (torch.zeros_like(yaw_rate), torch.zeros_like(yaw_rate), -yaw_rate)
    )
    joint_position_decay, joint_velocity_decay = decay_spring_offsets(
        joint_position_offset,
        joint_velocity_offset,
        halflife_s=float(inertialization_halflife_s),
        time_s=times,
    )
    root_position_decay, _ = decay_spring_offsets(
        root_position_offset,
        root_velocity_offset,
        halflife_s=float(inertialization_halflife_s),
        time_s=times,
    )
    rotation_decay, _ = decay_spring_offsets(
        rotation_offset,
        angular_velocity_offset,
        halflife_s=float(inertialization_halflife_s),
        time_s=times,
    )
    composed_joints = action.joint_position + joint_position_decay
    composed_velocities = action.joint_velocity + joint_velocity_decay
    composed_roots = placed.root_position_world + root_position_decay
    composed_quaternions = _quat_normalize(
        _quat_mul(
            _quat_from_scaled_axis(rotation_decay),
            placed.root_orientation_world_wxyz,
        )
    )
    composed = _stage(
        state=state,
        joints=composed_joints,
        velocities=composed_velocities,
        roots=composed_roots,
        quaternions=composed_quaternions,
        support=support,
        foot_kinematics=foot_kinematics,
    )
    return QualityPreview(native=native, placed=rigid, composed=composed)
