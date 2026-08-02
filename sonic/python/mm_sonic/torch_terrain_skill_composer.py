"""Atomic sequential playback of one selected terrain skill."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .joints import ContractError
from .torch_motion_matcher import decay_spring_offsets
from .torch_terrain_skills import TerrainSkill


@dataclass(frozen=True)
class TerrainSkillPose:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    root_linear_velocity_world: torch.Tensor
    root_angular_velocity_world: torch.Tensor


@dataclass(frozen=True)
class TerrainSkillFrame:
    qpos: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    root_linear_velocity_world: torch.Tensor
    root_angular_velocity_world: torch.Tensor
    support: torch.Tensor
    skill_index: int
    clip_index: int
    source_frame: int
    inertialization_residual: float


@dataclass(frozen=True)
class _Offsets:
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_velocity: torch.Tensor
    root_rotation_axis: torch.Tensor
    root_angular_velocity: torch.Tensor


@dataclass(frozen=True)
class TerrainSkillState:
    folder: Any
    skill: TerrainSkill
    selected_entry_frame: int
    next_source_frame: int
    yaw_offset: torch.Tensor
    translation_world: torch.Tensor
    offsets: _Offsets
    halflife_s: float


@dataclass(frozen=True)
class TerrainSkillStep:
    frame: TerrainSkillFrame
    state: TerrainSkillState
    completed: bool


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _quat_inverse(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[:1], -q[1:]))


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q).clamp_min(1e-8)


def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind()
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)))


def _quat_to_scaled_axis(q: torch.Tensor) -> torch.Tensor:
    q = _quat_normalize(q)
    q = torch.where(q[:1] < 0, -q, q)
    length = torch.linalg.vector_norm(q[1:])
    angle = 2.0 * torch.atan2(length, q[0])
    return q[1:] * (angle / length.clamp_min(1e-8))


def _quat_from_scaled_axis(value: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(value)
    half = angle / 2
    scale = torch.sin(half) / angle.clamp_min(1e-8)
    q = torch.cat((torch.cos(half).reshape(1), value * scale))
    identity = torch.zeros_like(q)
    identity[0] = 1
    return _quat_normalize(torch.where(angle < 1e-8, identity, q))


def _rotate_z(value: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (c * value[0] - s * value[1], s * value[0] + c * value[1], value[2])
    )


def _source_pose(folder: Any, skill: TerrainSkill, frame: int, reference: torch.Tensor):
    try:
        clip = folder.clips[skill.clip_index]
        root_index = int(folder.layout.root_body_index)
        values = (
            clip.joint_position[frame],
            clip.joint_velocity[frame],
            clip.body_position_world[frame, root_index],
            clip.body_quaternion_world_wxyz[frame, root_index],
            clip.body_linear_velocity_world[frame, root_index],
            clip.body_angular_velocity_world[frame, root_index],
        )
    except (AttributeError, IndexError, TypeError) as error:
        raise ContractError("terrain skill source frame is unavailable") from error
    return tuple(
        torch.tensor(value, dtype=reference.dtype, device=reference.device)
        for value in values
    )


def _validate_current(current: TerrainSkillPose) -> None:
    if not isinstance(current, TerrainSkillPose):
        raise ContractError("terrain skill current pose is invalid")
    reference = current.joint_position
    expected = {
        "joint_velocity": tuple(reference.shape),
        "root_position_world": (3,),
        "root_orientation_world_wxyz": (4,),
        "root_linear_velocity_world": (3,),
        "root_angular_velocity_world": (3,),
    }
    if reference.ndim != 1 or not reference.dtype.is_floating_point:
        raise ContractError("terrain skill joints must be a floating vector")
    for name, shape in expected.items():
        value = getattr(current, name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not torch.isfinite(value).all()
        ):
            raise ContractError(f"terrain skill {name} is invalid")


def start_skill(
    folder: Any,
    skill: TerrainSkill,
    *,
    selected_entry_frame: int,
    current: TerrainSkillPose,
    halflife_s: float = 0.10,
) -> TerrainSkillState:
    """Place a skill at the current root and initialize exact pose offsets."""

    _validate_current(current)
    if not isinstance(skill, TerrainSkill):
        raise ContractError("terrain skill is invalid")
    if (
        type(selected_entry_frame) is not int
        or not skill.interval.entry_start
        <= selected_entry_frame
        < skill.interval.playback_stop
    ):
        raise ContractError("selected frame is outside the terrain skill entry window")
    if not math.isfinite(float(halflife_s)) or halflife_s <= 0:
        raise ContractError("terrain skill inertialization halflife must be positive")

    jp, jv, root, quat, root_v, root_w = _source_pose(
        folder, skill, selected_entry_frame, current.joint_position
    )
    if tuple(jp.shape) != tuple(current.joint_position.shape):
        raise ContractError("terrain skill source joint layout does not match current pose")
    yaw_offset = _quat_yaw(current.root_orientation_world_wxyz) - _quat_yaw(quat)
    yaw_quat = _quat_from_yaw(yaw_offset)
    placed_root = _rotate_z(root, yaw_offset)
    translation = current.root_position_world - placed_root
    placed_quat = _quat_normalize(_quat_mul(yaw_quat, quat))
    placed_root_v = _rotate_z(root_v, yaw_offset)
    placed_root_w = _rotate_z(root_w, yaw_offset)
    rotation_offset = _quat_mul(
        current.root_orientation_world_wxyz, _quat_inverse(placed_quat)
    )
    return TerrainSkillState(
        folder=folder,
        skill=skill,
        selected_entry_frame=selected_entry_frame,
        next_source_frame=selected_entry_frame,
        yaw_offset=yaw_offset,
        translation_world=translation,
        offsets=_Offsets(
            joint_position=current.joint_position - jp,
            joint_velocity=current.joint_velocity - jv,
            root_position=current.root_position_world - (placed_root + translation),
            root_velocity=current.root_linear_velocity_world - placed_root_v,
            root_rotation_axis=_quat_to_scaled_axis(rotation_offset),
            root_angular_velocity=current.root_angular_velocity_world - placed_root_w,
        ),
        halflife_s=float(halflife_s),
    )


def can_interrupt_skill(skill: TerrainSkill, source_frame: int) -> bool:
    if not isinstance(skill, TerrainSkill) or type(source_frame) is not int:
        raise ContractError("terrain skill interrupt query is invalid")
    if not 0 <= source_frame < int(skill.support_mask.shape[0]):
        raise ContractError("terrain skill interrupt frame is invalid")
    return bool(skill.support_mask[source_frame].all().item())


def advance_skill(state: TerrainSkillState) -> TerrainSkillStep:
    """Emit exactly one source frame; no search occurs while advancing."""

    if not isinstance(state, TerrainSkillState):
        raise ContractError("terrain skill state is invalid")
    frame_index = state.next_source_frame
    if frame_index >= state.skill.interval.playback_stop:
        raise ContractError("terrain skill playback is already complete")
    reference = state.offsets.joint_position
    jp, jv, root, quat, root_v, root_w = _source_pose(
        state.folder, state.skill, frame_index, reference
    )
    yaw_quat = _quat_from_yaw(state.yaw_offset)
    placed_root = _rotate_z(root, state.yaw_offset) + state.translation_world
    placed_quat = _quat_normalize(_quat_mul(yaw_quat, quat))
    placed_root_v = _rotate_z(root_v, state.yaw_offset)
    placed_root_w = _rotate_z(root_w, state.yaw_offset)
    time_s = torch.as_tensor(
        (frame_index - state.selected_entry_frame) / 50.0,
        dtype=reference.dtype,
        device=reference.device,
    )
    jpo, jvo = decay_spring_offsets(
        state.offsets.joint_position,
        state.offsets.joint_velocity,
        halflife_s=state.halflife_s,
        time_s=time_s,
    )
    rpo, rvo = decay_spring_offsets(
        state.offsets.root_position,
        state.offsets.root_velocity,
        halflife_s=state.halflife_s,
        time_s=time_s,
    )
    rao, rwo = decay_spring_offsets(
        state.offsets.root_rotation_axis,
        state.offsets.root_angular_velocity,
        halflife_s=state.halflife_s,
        time_s=time_s,
    )
    out_jp = jp + jpo
    out_jv = jv + jvo
    out_root = placed_root + rpo
    out_quat = _quat_normalize(_quat_mul(_quat_from_scaled_axis(rao), placed_quat))
    out_root_v = placed_root_v + rvo
    out_root_w = placed_root_w + rwo
    residual = float(
        torch.sqrt(
            torch.sum(jpo.square())
            + torch.sum(rpo.square())
            + torch.sum(rao.square())
        ).item()
    )
    frame = TerrainSkillFrame(
        qpos=torch.cat((out_root, out_quat, out_jp)),
        joint_position=out_jp,
        joint_velocity=out_jv,
        root_position_world=out_root,
        root_orientation_world_wxyz=out_quat,
        root_linear_velocity_world=out_root_v,
        root_angular_velocity_world=out_root_w,
        support=state.skill.support_mask[frame_index].clone(),
        skill_index=state.skill.skill_index,
        clip_index=state.skill.clip_index,
        source_frame=frame_index,
        inertialization_residual=residual,
    )
    next_state = TerrainSkillState(
        folder=state.folder,
        skill=state.skill,
        selected_entry_frame=state.selected_entry_frame,
        next_source_frame=frame_index + 1,
        yaw_offset=state.yaw_offset,
        translation_world=state.translation_world,
        offsets=state.offsets,
        halflife_s=state.halflife_s,
    )
    return TerrainSkillStep(
        frame=frame,
        state=next_state,
        completed=next_state.next_source_frame >= state.skill.interval.playback_stop,
    )
