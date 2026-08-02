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
    maximum_joint_correction_speed_rad_s: float

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
            "maximum_joint_correction_speed_rad_s",
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


@dataclass(frozen=True)
class StanceRootProjectionResult:
    root_position_world: torch.Tensor
    joint_position: torch.Tensor
    foot_position_world: torch.Tensor
    maximum_target_error_m: float
    maximum_root_correction_m: float
    maximum_root_correction_speed_m_s: float
    maximum_joint_deformation_rad: float
    rms_joint_deformation_rad: float
    maximum_joint_correction_speed_rad_s: float

    def __post_init__(self) -> None:
        roots = self.root_position_world
        if (
            not isinstance(roots, torch.Tensor)
            or roots.ndim != 2
            or tuple(roots.shape[1:]) != (3,)
            or roots.shape[0] < 2
            or not roots.dtype.is_floating_point
            or not torch.isfinite(roots).all()
            or not isinstance(self.joint_position, torch.Tensor)
            or tuple(self.joint_position.shape) != (roots.shape[0], 29)
            or self.joint_position.dtype != roots.dtype
            or self.joint_position.device != roots.device
            or not torch.isfinite(self.joint_position).all()
            or not isinstance(self.foot_position_world, torch.Tensor)
            or tuple(self.foot_position_world.shape) != (roots.shape[0], 2, 3)
            or self.foot_position_world.dtype != roots.dtype
            or self.foot_position_world.device != roots.device
            or not torch.isfinite(self.foot_position_world).all()
        ):
            raise ValueError("stance-root projection output is invalid")
        for name in (
            "maximum_target_error_m",
            "maximum_root_correction_m",
            "maximum_root_correction_speed_m_s",
            "maximum_joint_deformation_rad",
            "rms_joint_deformation_rad",
            "maximum_joint_correction_speed_rad_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("stance-root projection metrics are invalid")
            object.__setattr__(self, name, float(value))
        for name in (
            "root_position_world",
            "joint_position",
            "foot_position_world",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())


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
        maximum_joint_correction_speed_rad_s=float(
            np.linalg.norm(np.diff(deformation, axis=0), axis=1).max(initial=0.0)
            * 50.0
        ),
    )


def project_contact_trajectory_with_stance_root(
    *,
    joint_position: torch.Tensor,
    root_position_world: torch.Tensor,
    root_orientation_world_wxyz: torch.Tensor,
    raw_foot_position_world: torch.Tensor,
    support_mask: torch.Tensor,
    swing_foot: int,
    entry_foot_position_world: torch.Tensor,
    landing_target_world: torch.Tensor,
    foot_kinematics: object,
    root_correction_scale: float = 1.0,
    root_smoothing_passes: int = 0,
    joint_smoothing_passes: int = 0,
    reproject_smoothed_joints: bool = False,
    joint_projection_enabled: bool = True,
) -> StanceRootProjectionResult:
    """Lock the persistent stance with root translation, then warp the swing leg."""

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
        or not isinstance(raw_foot_position_world, torch.Tensor)
        or tuple(raw_foot_position_world.shape) != (joints.shape[0], 2, 3)
        or raw_foot_position_world.dtype != joints.dtype
        or raw_foot_position_world.device != joints.device
        or not torch.isfinite(raw_foot_position_world).all()
        or not isinstance(support_mask, torch.Tensor)
        or tuple(support_mask.shape) != (joints.shape[0], 2)
        or support_mask.dtype != torch.bool
        or support_mask.device != joints.device
        or type(swing_foot) is not int
        or swing_foot not in (0, 1)
        or not isinstance(entry_foot_position_world, torch.Tensor)
        or tuple(entry_foot_position_world.shape) != (2, 3)
        or entry_foot_position_world.dtype != joints.dtype
        or entry_foot_position_world.device != joints.device
        or not torch.isfinite(entry_foot_position_world).all()
        or not isinstance(landing_target_world, torch.Tensor)
        or tuple(landing_target_world.shape) != (3,)
        or landing_target_world.dtype != joints.dtype
        or landing_target_world.device != joints.device
        or not torch.isfinite(landing_target_world).all()
        or isinstance(root_correction_scale, bool)
        or not isinstance(root_correction_scale, (int, float))
        or not math.isfinite(float(root_correction_scale))
        or not 0.0 <= float(root_correction_scale) <= 1.0
        or type(root_smoothing_passes) is not int
        or not 0 <= root_smoothing_passes <= 10
        or type(joint_smoothing_passes) is not int
        or not 0 <= joint_smoothing_passes <= 10
        or type(reproject_smoothed_joints) is not bool
        or type(joint_projection_enabled) is not bool
        or (not joint_projection_enabled and joint_smoothing_passes != 0)
        or (not joint_projection_enabled and reproject_smoothed_joints)
    ):
        raise ValueError("stance-root projection inputs are invalid")
    provisional = build_contact_target_trajectory(
        raw_foot_position_world,
        support_mask,
        swing_foot=swing_foot,
        entry_foot_position_world=entry_foot_position_world,
        landing_target_world=landing_target_world,
    )
    known = support_mask.any(dim=1)
    known_indices = torch.nonzero(known, as_tuple=False).flatten()
    if known_indices.numel() == 0:
        raise ValueError("stance-root projection requires support")
    root_correction = torch.zeros_like(root_position_world)
    for frame in known_indices.detach().cpu().tolist():
        supported = support_mask[frame]
        root_correction[frame] = torch.mean(
            provisional.position_world[frame, supported]
            - raw_foot_position_world[frame, supported],
            dim=0,
        )
    first = int(known_indices[0].item())
    last = int(known_indices[-1].item())
    root_correction[:first] = root_correction[first]
    root_correction[last + 1 :] = root_correction[last]
    known_list = known_indices.detach().cpu().tolist()
    for left, right in zip(known_list[:-1], known_list[1:]):
        if right == left + 1:
            continue
        phase = torch.linspace(
            0.0,
            1.0,
            right - left + 1,
            dtype=joints.dtype,
            device=joints.device,
        )
        smoothstep = phase.square() * (3.0 - 2.0 * phase)
        root_correction[left : right + 1] = (
            root_correction[left][None] * (1.0 - smoothstep[:, None])
            + root_correction[right][None] * smoothstep[:, None]
        )
    boundary_start = root_correction[0].clone()
    weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    for _ in range(root_smoothing_passes):
        padded = torch.cat(
            (
                root_correction[:1].expand(2, 3),
                root_correction,
                root_correction[-1:].expand(2, 3),
            ),
            dim=0,
        )
        root_correction = sum(
            weight * padded[offset : offset + joints.shape[0]]
            for offset, weight in enumerate(weights)
        ) / sum(weights)
        root_correction[0] = boundary_start
    root_correction = root_correction * float(root_correction_scale)
    corrected_roots = root_position_world + root_correction
    shifted_feet = raw_foot_position_world + root_correction[:, None, :]
    targets = build_contact_target_trajectory(
        shifted_feet,
        support_mask,
        swing_foot=swing_foot,
        entry_foot_position_world=entry_foot_position_world,
        landing_target_world=landing_target_world,
    )
    if joint_projection_enabled:
        projection = project_contact_trajectory(
            joint_position=joints,
            root_position_world=corrected_roots,
            root_orientation_world_wxyz=root_orientation_world_wxyz,
            targets=ContactTargetTrajectory(
                position_world=targets.position_world,
                solve_mask=targets.solve_mask,
                swing_warp_weight=targets.swing_warp_weight,
            ),
            foot_kinematics=foot_kinematics,
        )
        projected_joints = projection.joint_position
        projected_feet = projection.foot_position_world
        maximum_projection_error = projection.maximum_target_error_m
    else:
        projected_joints = joints
        projected_feet = shifted_feet
        support_error = torch.linalg.vector_norm(
            projected_feet - targets.position_world, dim=2
        )[support_mask]
        maximum_projection_error = (
            float(support_error.max().item()) if support_error.numel() else 0.0
        )
    total_root_correction = root_correction
    if joint_smoothing_passes:
        joint_correction = projected_joints - joints
        joint_boundary_start = joint_correction[0].clone()
        joint_boundary_end = joint_correction[-1].clone()
        for _ in range(joint_smoothing_passes):
            padded = torch.cat(
                (
                    joint_correction[:1].expand(2, 29),
                    joint_correction,
                    joint_correction[-1:].expand(2, 29),
                ),
                dim=0,
            )
            joint_correction = sum(
                weight * padded[offset : offset + joints.shape[0]]
                for offset, weight in enumerate(weights)
            ) / sum(weights)
            joint_correction[0] = joint_boundary_start
            joint_correction[-1] = joint_boundary_end
        projected_joints = joints + joint_correction
        forward = getattr(foot_kinematics, "foot_positions", None)
        if not callable(forward):
            raise ValueError("stance-root projection foot kinematics is invalid")
        try:
            smoothed_feet_numpy = np.asarray(
                forward(
                    projected_joints.detach().cpu().numpy(),
                    corrected_roots.detach().cpu().numpy(),
                    root_orientation_world_wxyz.detach().cpu().numpy(),
                ),
                dtype=np.float64,
            )
        except Exception as error:
            raise ValueError(
                "stance-root projection smoothing forward kinematics failed"
            ) from error
        if (
            smoothed_feet_numpy.shape != (joints.shape[0], 2, 3)
            or not np.isfinite(smoothed_feet_numpy).all()
        ):
            raise ValueError(
                "stance-root projection smoothing returned invalid feet"
            )
        projected_feet = torch.as_tensor(
            smoothed_feet_numpy, dtype=joints.dtype, device=joints.device
        )
        if reproject_smoothed_joints:
            polished = project_contact_trajectory(
                joint_position=projected_joints,
                root_position_world=corrected_roots,
                root_orientation_world_wxyz=root_orientation_world_wxyz,
                targets=targets,
                foot_kinematics=foot_kinematics,
            )
            projected_joints = polished.joint_position
            projected_feet = polished.foot_position_world
            maximum_projection_error = max(
                maximum_projection_error,
                polished.maximum_target_error_m,
            )
        else:
            residual_root = torch.zeros_like(root_correction)
            residual_known = support_mask.any(dim=1)
            residual_indices = torch.nonzero(
                residual_known, as_tuple=False
            ).flatten()
            for frame in residual_indices.detach().cpu().tolist():
                supported = support_mask[frame]
                residual_root[frame] = torch.mean(
                    targets.position_world[frame, supported]
                    - projected_feet[frame, supported],
                    dim=0,
                )
            residual_list = residual_indices.detach().cpu().tolist()
            for left, right in zip(residual_list[:-1], residual_list[1:]):
                if right == left + 1:
                    continue
                phase = torch.linspace(
                    0.0,
                    1.0,
                    right - left + 1,
                    dtype=joints.dtype,
                    device=joints.device,
                )
                smoothstep = phase.square() * (3.0 - 2.0 * phase)
                residual_root[left : right + 1] = (
                    residual_root[left][None] * (1.0 - smoothstep[:, None])
                    + residual_root[right][None] * smoothstep[:, None]
                )
            corrected_roots = corrected_roots + residual_root
            projected_feet = projected_feet + residual_root[:, None, :]
            total_root_correction = root_correction + residual_root
    target_error = torch.linalg.vector_norm(
        projected_feet - targets.position_world,
        dim=2,
    )[support_mask]
    joint_deformation = projected_joints - joints
    return StanceRootProjectionResult(
        root_position_world=corrected_roots,
        joint_position=projected_joints,
        foot_position_world=projected_feet,
        maximum_target_error_m=max(
            maximum_projection_error,
            float(target_error.max().item()) if target_error.numel() else 0.0,
        ),
        maximum_root_correction_m=float(
            torch.linalg.vector_norm(total_root_correction, dim=1).max().item()
        ),
        maximum_root_correction_speed_m_s=float(
            torch.linalg.vector_norm(
                torch.diff(total_root_correction, dim=0), dim=1
            ).max().item()
            * 50.0
        ),
        maximum_joint_deformation_rad=float(
            torch.abs(joint_deformation).max().item()
        ),
        rms_joint_deformation_rad=float(
            torch.sqrt(torch.mean(joint_deformation.square())).item()
        ),
        maximum_joint_correction_speed_rad_s=float(
            torch.linalg.vector_norm(
                torch.diff(joint_deformation, dim=0), dim=1
            ).max().item()
            * 50.0
        ),
    )
