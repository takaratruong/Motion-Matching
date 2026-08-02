"""Contact-anchored placement and composition for terrain contact actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .torch_contact_oracle_actions import ContactPhaseAction
from .torch_contact_oracle_search import OracleState, PlacedContactPhase


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


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack(
        (torch.cos(yaw / 2.0), zero, zero, torch.sin(yaw / 2.0))
    )


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
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


@dataclass(frozen=True)
class ContactAnchoredPlacement:
    placed: PlacedContactPhase
    yaw_world: torch.Tensor
    translation_world: torch.Tensor
    maximum_entry_support_error_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.placed, PlacedContactPhase):
            raise ValueError("contact-anchored placed action is invalid")
        reference = self.placed.root_position_world
        if (
            not isinstance(self.yaw_world, torch.Tensor)
            or self.yaw_world.shape != torch.Size([])
            or self.yaw_world.dtype != reference.dtype
            or self.yaw_world.device != reference.device
            or not torch.isfinite(self.yaw_world)
            or not isinstance(self.translation_world, torch.Tensor)
            or tuple(self.translation_world.shape) != (3,)
            or self.translation_world.dtype != reference.dtype
            or self.translation_world.device != reference.device
            or not torch.isfinite(self.translation_world).all()
            or isinstance(self.maximum_entry_support_error_m, bool)
            or not isinstance(self.maximum_entry_support_error_m, (int, float))
            or not math.isfinite(float(self.maximum_entry_support_error_m))
            or float(self.maximum_entry_support_error_m) < 0.0
        ):
            raise ValueError("contact-anchored placement metadata is invalid")
        object.__setattr__(self, "yaw_world", self.yaw_world.detach().clone())
        object.__setattr__(
            self, "translation_world", self.translation_world.detach().clone()
        )
        object.__setattr__(
            self,
            "maximum_entry_support_error_m",
            float(self.maximum_entry_support_error_m),
        )


@dataclass(frozen=True)
class ContactTargetTrajectory:
    position_world: torch.Tensor
    solve_mask: torch.Tensor
    swing_warp_weight: torch.Tensor

    def __post_init__(self) -> None:
        position = self.position_world
        if (
            not isinstance(position, torch.Tensor)
            or position.ndim != 3
            or tuple(position.shape[1:]) != (2, 3)
            or not position.dtype.is_floating_point
            or not torch.isfinite(position).all()
        ):
            raise ValueError("contact target positions are invalid")
        frames = position.shape[0]
        if frames < 2:
            raise ValueError("contact target trajectory is too short")
        if (
            not isinstance(self.solve_mask, torch.Tensor)
            or tuple(self.solve_mask.shape) != (frames, 2)
            or self.solve_mask.dtype != torch.bool
            or self.solve_mask.device != position.device
            or not isinstance(self.swing_warp_weight, torch.Tensor)
            or tuple(self.swing_warp_weight.shape) != (frames,)
            or self.swing_warp_weight.dtype != position.dtype
            or self.swing_warp_weight.device != position.device
            or not torch.isfinite(self.swing_warp_weight).all()
            or bool((self.swing_warp_weight < 0.0).any())
            or bool((self.swing_warp_weight > 1.0).any())
        ):
            raise ValueError("contact target metadata is invalid")
        object.__setattr__(self, "position_world", position.detach().clone())
        object.__setattr__(self, "solve_mask", self.solve_mask.detach().clone())
        object.__setattr__(
            self,
            "swing_warp_weight",
            self.swing_warp_weight.detach().clone(),
        )


@dataclass(frozen=True)
class ContactProjectionResult:
    joint_position: torch.Tensor
    foot_position_world: torch.Tensor
    maximum_target_error_m: float
    maximum_joint_deformation_rad: float
    rms_joint_deformation_rad: float

    def __post_init__(self) -> None:
        joints = self.joint_position
        if (
            not isinstance(joints, torch.Tensor)
            or joints.ndim != 2
            or tuple(joints.shape[1:]) != (29,)
            or not joints.dtype.is_floating_point
            or not torch.isfinite(joints).all()
            or not isinstance(self.foot_position_world, torch.Tensor)
            or tuple(self.foot_position_world.shape) != (joints.shape[0], 2, 3)
            or self.foot_position_world.dtype != joints.dtype
            or self.foot_position_world.device != joints.device
            or not torch.isfinite(self.foot_position_world).all()
        ):
            raise ValueError("contact projection output is invalid")
        for name in (
            "maximum_target_error_m",
            "maximum_joint_deformation_rad",
            "rms_joint_deformation_rad",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("contact projection metrics are invalid")
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "joint_position", joints.detach().clone())
        object.__setattr__(
            self,
            "foot_position_world",
            self.foot_position_world.detach().clone(),
        )


def _fit_entry_support(
    action: ContactPhaseAction, state: OracleState
) -> tuple[torch.Tensor, torch.Tensor]:
    support_indices = torch.nonzero(state.support_mask, as_tuple=False).flatten()
    source = action.foot_position_local[0, support_indices]
    target = state.foot_position_world[support_indices]
    if support_indices.numel() == 1:
        yaw = state.root_yaw_world
    elif support_indices.numel() == 2:
        source_delta = source[1, :2] - source[0, :2]
        target_delta = target[1, :2] - target[0, :2]
        if (
            float(torch.linalg.vector_norm(source_delta).item()) <= 1e-6
            or float(torch.linalg.vector_norm(target_delta).item()) <= 1e-6
        ):
            raise ValueError("contact-anchored double support is degenerate")
        source_angle = torch.atan2(source_delta[1], source_delta[0])
        target_angle = torch.atan2(target_delta[1], target_delta[0])
        yaw = torch.atan2(
            torch.sin(target_angle - source_angle),
            torch.cos(target_angle - source_angle),
        )
    else:
        raise ValueError("contact-anchored placement requires entry support")
    rotated_xy = _rotate_xy(source[:, :2], yaw)
    translation_xy = torch.mean(target[:, :2] - rotated_xy, dim=0)
    translation_z = torch.mean(target[:, 2] - source[:, 2])
    return yaw, torch.cat((translation_xy, translation_z.reshape(1)))


def place_action_contact_anchored(
    action: ContactPhaseAction, state: OracleState
) -> ContactAnchoredPlacement:
    """Rigidly fit the complete action to current entry support contacts."""

    if not isinstance(action, ContactPhaseAction) or not isinstance(
        state, OracleState
    ):
        raise ValueError("contact-anchored placement inputs are invalid")
    if action.root_position_local.device != state.root_position_world.device:
        raise ValueError("contact-anchored placement devices do not match")
    if not bool(torch.equal(action.entry_support, state.support_mask)):
        raise ValueError("contact-anchored entry support must match exactly")
    yaw, translation = _fit_entry_support(action, state)
    root_xy = _rotate_xy(action.root_position_local[:, :2], yaw) + translation[:2]
    root_z = action.root_position_local[:, 2:] + translation[2]
    feet_xy = _rotate_xy(action.foot_position_local[..., :2], yaw) + translation[:2]
    feet_z = action.foot_position_local[..., 2:] + translation[2]
    root_yaw = action.root_yaw_local + yaw
    root_yaw = torch.atan2(torch.sin(root_yaw), torch.cos(root_yaw))
    orientation = _quaternion_multiply(
        _yaw_quaternion(yaw), action.root_orientation_local_wxyz
    )
    placed = PlacedContactPhase(
        action=action,
        root_position_world=torch.cat((root_xy, root_z), dim=1),
        root_yaw_world=root_yaw,
        root_orientation_world_wxyz=orientation,
        foot_position_world=torch.cat((feet_xy, feet_z), dim=2),
    )
    entry_error = torch.linalg.vector_norm(
        placed.foot_position_world[0, state.support_mask]
        - state.foot_position_world[state.support_mask],
        dim=1,
    )
    return ContactAnchoredPlacement(
        placed=placed,
        yaw_world=yaw,
        translation_world=translation,
        maximum_entry_support_error_m=float(entry_error.max().item()),
    )


def build_contact_target_trajectory(
    raw_foot_position_world: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    swing_foot: int,
    entry_foot_position_world: torch.Tensor,
    landing_target_world: torch.Tensor,
) -> ContactTargetTrajectory:
    """Lock stance contacts and smoothly warp one swing endpoint."""

    raw = raw_foot_position_world
    if (
        not isinstance(raw, torch.Tensor)
        or raw.ndim != 3
        or tuple(raw.shape[1:]) != (2, 3)
        or raw.shape[0] < 2
        or not raw.dtype.is_floating_point
        or not torch.isfinite(raw).all()
    ):
        raise ValueError("contact target raw feet are invalid")
    frames = raw.shape[0]
    if (
        not isinstance(support_mask, torch.Tensor)
        or tuple(support_mask.shape) != (frames, 2)
        or support_mask.dtype != torch.bool
        or support_mask.device != raw.device
        or type(swing_foot) is not int
        or swing_foot not in (0, 1)
        or not isinstance(entry_foot_position_world, torch.Tensor)
        or tuple(entry_foot_position_world.shape) != (2, 3)
        or entry_foot_position_world.dtype != raw.dtype
        or entry_foot_position_world.device != raw.device
        or not torch.isfinite(entry_foot_position_world).all()
        or not isinstance(landing_target_world, torch.Tensor)
        or tuple(landing_target_world.shape) != (3,)
        or landing_target_world.dtype != raw.dtype
        or landing_target_world.device != raw.device
        or not torch.isfinite(landing_target_world).all()
    ):
        raise ValueError("contact target inputs are invalid")

    swing_support = support_mask[:, swing_foot]
    unsupported = torch.nonzero(~swing_support, as_tuple=False).flatten()
    if unsupported.numel() == 0:
        raise ValueError("contact target swing interval is missing")
    flight_start = int(unsupported[0].item())
    touchdown_candidates = torch.nonzero(
        swing_support[flight_start:], as_tuple=False
    ).flatten()
    if touchdown_candidates.numel() == 0:
        raise ValueError("contact target touchdown is missing")
    touchdown = flight_start + int(touchdown_candidates[0].item())

    targets = raw.clone()
    solve = support_mask.clone()
    weights = torch.zeros((frames,), dtype=raw.dtype, device=raw.device)

    for foot in range(2):
        if bool(support_mask[0, foot]):
            releases = torch.nonzero(~support_mask[:, foot], as_tuple=False).flatten()
            release = int(releases[0].item()) if releases.numel() else frames
            targets[:release, foot] = entry_foot_position_world[foot]

    flight_length = touchdown - flight_start
    phase = torch.linspace(
        0.0,
        1.0,
        flight_length + 1,
        dtype=raw.dtype,
        device=raw.device,
    )
    smoothstep = phase.square() * (3.0 - 2.0 * phase)
    weights[flight_start : touchdown + 1] = smoothstep
    landing_delta = landing_target_world - raw[touchdown, swing_foot]
    targets[flight_start : touchdown + 1, swing_foot] = (
        raw[flight_start : touchdown + 1, swing_foot]
        + smoothstep[:, None] * landing_delta
    )
    solve[flight_start : touchdown + 1, swing_foot] = True

    releases = torch.nonzero(
        ~swing_support[touchdown:], as_tuple=False
    ).flatten()
    release = touchdown + (int(releases[0].item()) if releases.numel() else frames - touchdown)
    targets[touchdown:release, swing_foot] = landing_target_world
    weights[touchdown:release] = 1.0
    return ContactTargetTrajectory(
        position_world=targets,
        solve_mask=solve,
        swing_warp_weight=weights,
    )


def project_contact_trajectory(
    *,
    joint_position: torch.Tensor,
    root_position_world: torch.Tensor,
    root_orientation_world_wxyz: torch.Tensor,
    targets: ContactTargetTrajectory,
    foot_kinematics: object,
) -> ContactProjectionResult:
    """Project selected feet onto contact targets while holding the root fixed."""

    joints = joint_position
    if (
        not isinstance(joints, torch.Tensor)
        or joints.ndim != 2
        or tuple(joints.shape[1:]) != (29,)
        or joints.shape[0] < 2
        or not joints.dtype.is_floating_point
        or not torch.isfinite(joints).all()
        or not isinstance(root_position_world, torch.Tensor)
        or tuple(root_position_world.shape) != (joints.shape[0], 3)
        or root_position_world.dtype != joints.dtype
        or root_position_world.device != joints.device
        or not torch.isfinite(root_position_world).all()
        or not isinstance(root_orientation_world_wxyz, torch.Tensor)
        or tuple(root_orientation_world_wxyz.shape) != (joints.shape[0], 4)
        or root_orientation_world_wxyz.dtype != joints.dtype
        or root_orientation_world_wxyz.device != joints.device
        or not torch.isfinite(root_orientation_world_wxyz).all()
        or not isinstance(targets, ContactTargetTrajectory)
        or targets.position_world.dtype != joints.dtype
        or targets.position_world.device != joints.device
        or targets.position_world.shape[0] != joints.shape[0]
    ):
        raise ValueError("contact projection inputs are invalid")
    solve = getattr(foot_kinematics, "solve_leg_positions", None)
    forward = getattr(foot_kinematics, "foot_positions", None)
    if not callable(solve) or not callable(forward):
        raise ValueError("contact projection foot kinematics is invalid")

    source = joints.detach().cpu().numpy()
    roots = root_position_world.detach().cpu().numpy()
    quaternions = root_orientation_world_wxyz.detach().cpu().numpy()
    masks = targets.solve_mask.detach().cpu().numpy()
    target_feet = targets.position_world.detach().cpu().numpy()
    projected = np.array(source, dtype=np.float64, copy=True)
    for frame in range(joints.shape[0]):
        if not bool(masks[frame].any()):
            continue
        try:
            solved = np.asarray(
                solve(
                    projected[frame],
                    roots[frame],
                    quaternions[frame],
                    masks[frame],
                    target_feet[frame],
                ),
                dtype=np.float64,
            )
        except Exception as error:
            raise ValueError(
                f"contact projection failed at frame {frame}"
            ) from error
        if solved.shape != (29,) or not np.isfinite(solved).all():
            raise ValueError(
                f"contact projection returned invalid joints at frame {frame}"
            )
        projected[frame] = solved

    try:
        feet = np.asarray(
            forward(projected, roots, quaternions),
            dtype=np.float64,
        )
    except Exception as error:
        raise ValueError("contact projection forward kinematics failed") from error
    if feet.shape != (joints.shape[0], 2, 3) or not np.isfinite(feet).all():
        raise ValueError("contact projection forward kinematics returned invalid feet")
    selected_error = np.linalg.norm(feet - target_feet, axis=2)[masks]
    deformation = projected - source
    result_joints = torch.as_tensor(
        projected,
        dtype=joints.dtype,
        device=joints.device,
    )
    result_feet = torch.as_tensor(
        feet,
        dtype=joints.dtype,
        device=joints.device,
    )
    return ContactProjectionResult(
        joint_position=result_joints,
        foot_position_world=result_feet,
        maximum_target_error_m=(
            float(selected_error.max()) if selected_error.size else 0.0
        ),
        maximum_joint_deformation_rad=float(np.abs(deformation).max()),
        rms_joint_deformation_rad=float(np.sqrt(np.mean(np.square(deformation)))),
    )
