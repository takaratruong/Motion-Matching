"""Atomic sequential playback of one selected terrain skill."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    playback_stop: int
    yaw_offset: torch.Tensor
    translation_world: torch.Tensor
    offsets: _Offsets
    halflife_s: float
    endpoint_translation_warp_world_xy: torch.Tensor
    endpoint_yaw_warp_rad: torch.Tensor


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
    playback_stop: int | None = None,
    target_displacement_local_xy: torch.Tensor | None = None,
    target_yaw_delta_rad: torch.Tensor | None = None,
    maximum_translation_warp_m: float = 0.0,
    maximum_yaw_warp_rad: float = 0.0,
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
    resolved_stop = (
        skill.interval.playback_stop
        if playback_stop is None
        else playback_stop
    )
    if (
        type(resolved_stop) is not int
        or not selected_entry_frame < resolved_stop <= skill.interval.playback_stop
        or not bool(skill.support_mask[resolved_stop - 1].all().item())
    ):
        raise ContractError("terrain skill playback endpoint must be stable double support")
    if not math.isfinite(float(halflife_s)) or halflife_s <= 0:
        raise ContractError("terrain skill inertialization halflife must be positive")
    if (
        (target_displacement_local_xy is None)
        != (target_yaw_delta_rad is None)
        or not math.isfinite(float(maximum_translation_warp_m))
        or float(maximum_translation_warp_m) < 0.0
        or not math.isfinite(float(maximum_yaw_warp_rad))
        or float(maximum_yaw_warp_rad) < 0.0
    ):
        raise ContractError("terrain skill endpoint warp is invalid")
    if target_displacement_local_xy is not None and (
        not isinstance(target_displacement_local_xy, torch.Tensor)
        or tuple(target_displacement_local_xy.shape) != (2,)
        or target_displacement_local_xy.dtype != current.joint_position.dtype
        or target_displacement_local_xy.device != current.joint_position.device
        or not torch.isfinite(target_displacement_local_xy).all()
        or not isinstance(target_yaw_delta_rad, torch.Tensor)
        or target_yaw_delta_rad.numel() != 1
        or target_yaw_delta_rad.dtype != current.joint_position.dtype
        or target_yaw_delta_rad.device != current.joint_position.device
        or not torch.isfinite(target_yaw_delta_rad).all()
    ):
        raise ContractError("terrain skill endpoint target is invalid")

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
    offsets = _Offsets(
        joint_position=current.joint_position - jp,
        joint_velocity=current.joint_velocity - jv,
        root_position=current.root_position_world - (placed_root + translation),
        root_velocity=current.root_linear_velocity_world - placed_root_v,
        root_rotation_axis=_quat_to_scaled_axis(rotation_offset),
        root_angular_velocity=current.root_angular_velocity_world - placed_root_w,
    )
    translation_warp = torch.zeros(
        2, dtype=current.joint_position.dtype, device=current.joint_position.device
    )
    yaw_warp = torch.zeros(
        (), dtype=current.joint_position.dtype, device=current.joint_position.device
    )
    if target_displacement_local_xy is not None:
        _, _, endpoint_root, endpoint_quat, _, _ = _source_pose(
            folder, skill, resolved_stop - 1, current.joint_position
        )
        endpoint_time_s = torch.as_tensor(
            (resolved_stop - 1 - selected_entry_frame) / 50.0,
            dtype=current.joint_position.dtype,
            device=current.joint_position.device,
        )
        endpoint_root_offset, _ = decay_spring_offsets(
            offsets.root_position,
            offsets.root_velocity,
            halflife_s=float(halflife_s),
            time_s=endpoint_time_s,
        )
        endpoint_rotation_offset, _ = decay_spring_offsets(
            offsets.root_rotation_axis,
            offsets.root_angular_velocity,
            halflife_s=float(halflife_s),
            time_s=endpoint_time_s,
        )
        current_yaw = _quat_yaw(current.root_orientation_world_wxyz)
        target_world = _rotate_z(
            torch.cat(
                (
                    target_displacement_local_xy,
                    torch.zeros(
                        1,
                        dtype=current.joint_position.dtype,
                        device=current.joint_position.device,
                    ),
                )
            ),
            current_yaw,
        )[:2]
        source_world = (
            _rotate_z(endpoint_root - root, yaw_offset) + endpoint_root_offset
        )[:2]
        translation_warp = target_world - source_world
        length = torch.linalg.vector_norm(translation_warp)
        limit = torch.as_tensor(
            maximum_translation_warp_m,
            dtype=translation_warp.dtype,
            device=translation_warp.device,
        )
        translation_warp = translation_warp * torch.clamp(
            limit / length.clamp_min(1e-8), max=1.0
        )
        placed_endpoint_quat = _quat_normalize(
            _quat_mul(yaw_quat, endpoint_quat)
        )
        baseline_endpoint_quat = _quat_normalize(
            _quat_mul(
                _quat_from_scaled_axis(endpoint_rotation_offset),
                placed_endpoint_quat,
            )
        )
        source_yaw_delta = (
            _quat_yaw(baseline_endpoint_quat)
            - _quat_yaw(current.root_orientation_world_wxyz)
        )
        yaw_warp = torch.atan2(
            torch.sin(target_yaw_delta_rad.reshape(()) - source_yaw_delta),
            torch.cos(target_yaw_delta_rad.reshape(()) - source_yaw_delta),
        ).clamp(-maximum_yaw_warp_rad, maximum_yaw_warp_rad)
    return TerrainSkillState(
        folder=folder,
        skill=skill,
        selected_entry_frame=selected_entry_frame,
        next_source_frame=selected_entry_frame,
        playback_stop=resolved_stop,
        yaw_offset=yaw_offset,
        translation_world=translation,
        offsets=offsets,
        halflife_s=float(halflife_s),
        endpoint_translation_warp_world_xy=translation_warp,
        endpoint_yaw_warp_rad=yaw_warp,
    )


def can_interrupt_skill(skill: TerrainSkill, source_frame: int) -> bool:
    if not isinstance(skill, TerrainSkill) or type(source_frame) is not int:
        raise ContractError("terrain skill interrupt query is invalid")
    if not 0 <= source_frame < int(skill.support_mask.shape[0]):
        raise ContractError("terrain skill interrupt frame is invalid")
    return bool(skill.support_mask[source_frame].all().item())


def extend_skill_state(
    state: TerrainSkillState, *, playback_stop: int
) -> TerrainSkillState:
    """Extend a completed chunk without restarting its world placement."""

    if not isinstance(state, TerrainSkillState):
        raise ContractError("terrain skill extension state is invalid")
    if state.next_source_frame != state.playback_stop:
        raise ContractError("terrain skill extension requires a completed chunk")
    if (
        type(playback_stop) is not int
        or not state.playback_stop
        < playback_stop
        <= state.skill.interval.playback_stop
    ):
        raise ContractError("terrain skill extension endpoint is invalid")
    if not bool(state.skill.support_mask[playback_stop - 1].all().item()):
        raise ContractError(
            "terrain skill extension endpoint must be stable double support"
        )
    has_translation_warp = bool(
        state.endpoint_translation_warp_world_xy.abs().max().item()
    )
    has_yaw_warp = bool(state.endpoint_yaw_warp_rad.abs().item())
    if not has_translation_warp and not has_yaw_warp:
        return replace(state, playback_stop=playback_stop)

    _, _, endpoint_root, _, _, _ = _source_pose(
        state.folder,
        state.skill,
        state.playback_stop - 1,
        state.offsets.joint_position,
    )
    folded_yaw = state.yaw_offset + state.endpoint_yaw_warp_rad
    old_endpoint = (
        _rotate_z(endpoint_root, state.yaw_offset)
        + state.translation_world
    )
    new_endpoint = _rotate_z(endpoint_root, folded_yaw)
    folded_translation = state.translation_world.clone()
    folded_translation[:2] = (
        old_endpoint[:2]
        + state.endpoint_translation_warp_world_xy
        - new_endpoint[:2]
    )
    return replace(
        state,
        playback_stop=playback_stop,
        yaw_offset=folded_yaw,
        translation_world=folded_translation,
        endpoint_translation_warp_world_xy=torch.zeros_like(
            state.endpoint_translation_warp_world_xy
        ),
        endpoint_yaw_warp_rad=torch.zeros_like(state.endpoint_yaw_warp_rad),
    )


def advance_skill(state: TerrainSkillState) -> TerrainSkillStep:
    """Emit exactly one source frame; no search occurs while advancing."""

    if not isinstance(state, TerrainSkillState):
        raise ContractError("terrain skill state is invalid")
    frame_index = state.next_source_frame
    if frame_index >= state.playback_stop:
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
    duration_frames = state.playback_stop - 1 - state.selected_entry_frame
    progress = torch.as_tensor(
        0.0
        if duration_frames == 0
        else (frame_index - state.selected_entry_frame) / duration_frames,
        dtype=reference.dtype,
        device=reference.device,
    )
    smooth_progress = progress * progress * (3.0 - 2.0 * progress)
    smooth_rate = (
        torch.zeros_like(progress)
        if duration_frames == 0
        else 6.0 * progress * (1.0 - progress) / (duration_frames / 50.0)
    )
    warp_xy = smooth_progress * state.endpoint_translation_warp_world_xy
    warp_yaw = smooth_progress * state.endpoint_yaw_warp_rad
    out_jp = jp + jpo
    out_jv = jv + jvo
    out_root = placed_root + rpo
    out_root = out_root.clone()
    out_root[:2] += warp_xy
    out_quat = _quat_normalize(
        _quat_mul(
            _quat_from_yaw(warp_yaw),
            _quat_mul(_quat_from_scaled_axis(rao), placed_quat),
        )
    )
    out_root_v = placed_root_v + rvo
    out_root_v = out_root_v.clone()
    out_root_v[:2] += smooth_rate * state.endpoint_translation_warp_world_xy
    out_root_w = placed_root_w + rwo
    out_root_w = out_root_w.clone()
    out_root_w[2] += smooth_rate * state.endpoint_yaw_warp_rad
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
        playback_stop=state.playback_stop,
        yaw_offset=state.yaw_offset,
        translation_world=state.translation_world,
        offsets=state.offsets,
        halflife_s=state.halflife_s,
        endpoint_translation_warp_world_xy=(
            state.endpoint_translation_warp_world_xy
        ),
        endpoint_yaw_warp_rad=state.endpoint_yaw_warp_rad,
    )
    return TerrainSkillStep(
        frame=frame,
        state=next_state,
        completed=next_state.next_source_frame >= state.playback_stop,
    )
