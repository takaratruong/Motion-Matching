"""Read-only contact validation of inertialized terrain-skill playback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch

from .joints import ContractError
from .torch_terrain_contact_feasibility import (
    TerrainContactFeasibilityConfig,
    TerrainContactFeasibilityResult,
    validate_placed_contact_trace,
)
from .torch_terrain_skill_composer import (
    TerrainSkillPose,
    TerrainSkillState,
    advance_skill,
    start_skill,
)
from .torch_terrain_skills import TerrainSkill


def _rejected(reason: str) -> TerrainContactFeasibilityResult:
    return TerrainContactFeasibilityResult(
        accepted=False,
        reason=reason,
        maximum_stance_error_m=0.0,
        landing_error_m=0.0,
        minimum_swing_clearance_m=0.0,
        minimum_sole_clearance_m=0.0,
        maximum_footprint_height_range_m=0.0,
        maximum_height_deformation_m=0.0,
    )


def _validate_emitted_frames(
    *,
    folder: Any,
    skill: TerrainSkill,
    frames: tuple[Any, ...],
    current: TerrainSkillPose,
    foot_kinematics: Any,
    sole_kinematics: Any | None,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: TerrainContactFeasibilityConfig,
    result_filter: Any | None,
    new_chunk: bool,
) -> TerrainContactFeasibilityResult:
    if result_filter is not None:
        clip = folder.clips[skill.clip_index]
        clip_path = str(getattr(clip, "relative_path", skill.clip_index))
        raw_results = tuple(
            SimpleNamespace(
                joint_position=frame.joint_position,
                joint_velocity=frame.joint_velocity,
                root_position_world=frame.root_position_world,
                root_orientation_world_wxyz=(
                    frame.root_orientation_world_wxyz
                ),
                root_linear_velocity_world=frame.root_linear_velocity_world,
                diagnostics=SimpleNamespace(
                    selected_clip_path=clip_path,
                    selected_frame=frame.source_frame,
                    terrain_chunk_start=(new_chunk and index == 0),
                ),
            )
            for index, frame in enumerate(frames)
        )
        try:
            filtered = tuple(result_filter.preview(raw_results))
        except Exception as error:
            raise ContractError("emitted contact filter preview failed") from error
        if len(filtered) != len(frames):
            raise ContractError("emitted contact filter preview length is invalid")
    else:
        filtered = frames

    reference = current.joint_position
    joints = torch.stack([frame.joint_position for frame in filtered])
    roots = torch.stack([frame.root_position_world for frame in filtered])
    orientations = torch.stack(
        [frame.root_orientation_world_wxyz for frame in filtered]
    )
    try:
        feet_numpy = foot_kinematics.foot_positions(
            joints.detach().cpu().numpy(),
            roots.detach().cpu().numpy(),
            orientations.detach().cpu().numpy(),
        )
    except Exception as error:
        raise ContractError("emitted contact preview FK failed") from error
    feet = torch.as_tensor(
        np.asarray(feet_numpy, dtype=np.float64),
        dtype=reference.dtype,
        device=reference.device,
    )
    if (
        tuple(feet.shape) != (len(frames), 2, 3)
        or not torch.isfinite(feet).all()
    ):
        raise ContractError("emitted contact preview FK result is invalid")
    sole_points = None
    if sole_kinematics is not None:
        try:
            sole_numpy = sole_kinematics.sole_points(
                joints.detach().cpu().numpy(),
                roots.detach().cpu().numpy(),
                orientations.detach().cpu().numpy(),
            )
        except Exception as error:
            raise ContractError("emitted contact preview sole FK failed") from error
        sole_points = torch.as_tensor(
            np.asarray(sole_numpy, dtype=np.float64),
            dtype=reference.dtype,
            device=reference.device,
        )
        if (
            sole_points.ndim != 4
            or tuple(sole_points.shape[:2]) != (len(frames), 2)
            or sole_points.shape[2] < 3
            or sole_points.shape[3] != 3
            or not torch.isfinite(sole_points).all()
        ):
            raise ContractError("emitted contact preview sole FK result is invalid")

    support = torch.stack([frame.support for frame in frames]).to(
        device=reference.device
    )
    source_frames = torch.tensor(
        [frame.source_frame for frame in frames],
        dtype=torch.long,
        device=reference.device,
    )
    source_surface = skill.foot_surface_height_m[source_frames].to(
        dtype=reference.dtype, device=reference.device
    )
    return validate_placed_contact_trace(
        foot_position_world=feet,
        support_mask=support,
        source_surface_height_m=source_surface,
        sample_surface=sample_surface,
        config=config,
        align_initial_support=False,
        landing_footprint_position_world=sole_points,
    )


def preview_continued_emitted_contact_trace(
    *,
    folder: Any,
    state: TerrainSkillState,
    current: TerrainSkillPose,
    foot_kinematics: Any,
    sole_kinematics: Any | None = None,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: TerrainContactFeasibilityConfig,
    result_filter: Any | None = None,
) -> TerrainContactFeasibilityResult:
    """Validate the exact filtered frames emitted by a skill extension."""

    if (
        not isinstance(state, TerrainSkillState)
        or state.next_source_frame >= state.playback_stop
        or not callable(getattr(foot_kinematics, "foot_positions", None))
        or (
            sole_kinematics is not None
            and not callable(getattr(sole_kinematics, "sole_points", None))
        )
        or not callable(sample_surface)
        or not isinstance(config, TerrainContactFeasibilityConfig)
        or (
            result_filter is not None
            and not callable(getattr(result_filter, "preview", None))
        )
    ):
        raise ContractError("continued emitted contact preview inputs are invalid")

    frames = []
    preview_state = state
    while preview_state.next_source_frame < preview_state.playback_stop:
        step = advance_skill(preview_state)
        frames.append(step.frame)
        preview_state = step.state
    return _validate_emitted_frames(
        folder=folder,
        skill=state.skill,
        frames=tuple(frames),
        current=current,
        foot_kinematics=foot_kinematics,
        sole_kinematics=sole_kinematics,
        sample_surface=sample_surface,
        config=config,
        result_filter=result_filter,
        new_chunk=False,
    )


def preview_emitted_contact_trace(
    *,
    folder: Any,
    skill: TerrainSkill,
    selected_entry_frame: int,
    endpoint_frame_exclusive: int,
    current: TerrainSkillPose,
    halflife_s: float,
    foot_kinematics: Any,
    sole_kinematics: Any | None = None,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: TerrainContactFeasibilityConfig,
    result_filter: Any | None = None,
    target_displacement_local_xy: torch.Tensor | None = None,
    target_yaw_delta_rad: torch.Tensor | None = None,
    maximum_translation_warp_m: float = 0.0,
    maximum_yaw_warp_rad: float = 0.0,
    entry_foot_position_world: torch.Tensor | None = None,
    entry_support: torch.Tensor | None = None,
    maximum_contact_anchor_yaw_rad: float = 0.0,
) -> TerrainContactFeasibilityResult:
    """Advance an isolated skill and validate the kinematics it would emit."""

    if (
        not isinstance(skill, TerrainSkill)
        or type(selected_entry_frame) is not int
        or not 0 <= selected_entry_frame < skill.support_mask.shape[0]
        or not callable(getattr(foot_kinematics, "foot_positions", None))
        or (
            sole_kinematics is not None
            and not callable(getattr(sole_kinematics, "sole_points", None))
        )
        or not callable(sample_surface)
        or not isinstance(config, TerrainContactFeasibilityConfig)
        or (
            result_filter is not None
            and not callable(getattr(result_filter, "preview", None))
        )
    ):
        raise ContractError("emitted contact preview inputs are invalid")
    if not bool(skill.support_mask[selected_entry_frame].any().item()):
        return _rejected("unsupported-entry")

    state = start_skill(
        folder,
        skill,
        selected_entry_frame=selected_entry_frame,
        current=current,
        halflife_s=halflife_s,
        playback_stop=endpoint_frame_exclusive,
        target_displacement_local_xy=target_displacement_local_xy,
        target_yaw_delta_rad=target_yaw_delta_rad,
        maximum_translation_warp_m=maximum_translation_warp_m,
        maximum_yaw_warp_rad=maximum_yaw_warp_rad,
        entry_foot_position_world=entry_foot_position_world,
        entry_support=entry_support,
        maximum_contact_anchor_yaw_rad=maximum_contact_anchor_yaw_rad,
    )
    frames = []
    while state.next_source_frame < state.playback_stop:
        step = advance_skill(state)
        frames.append(step.frame)
        state = step.state
    if not frames:
        raise ContractError("emitted contact preview produced no frames")
    return _validate_emitted_frames(
        folder=folder,
        skill=skill,
        frames=tuple(frames),
        current=current,
        foot_kinematics=foot_kinematics,
        sole_kinematics=sole_kinematics,
        sample_surface=sample_surface,
        config=config,
        result_filter=result_filter,
        new_chunk=True,
    )
