"""Read-only contact validation of inertialized terrain-skill playback."""

from __future__ import annotations

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
        maximum_footprint_height_range_m=0.0,
        maximum_height_deformation_m=0.0,
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
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    config: TerrainContactFeasibilityConfig,
) -> TerrainContactFeasibilityResult:
    """Advance an isolated skill and validate the kinematics it would emit."""

    if (
        not isinstance(skill, TerrainSkill)
        or type(selected_entry_frame) is not int
        or not 0 <= selected_entry_frame < skill.support_mask.shape[0]
        or not callable(getattr(foot_kinematics, "foot_positions", None))
        or not callable(sample_surface)
        or not isinstance(config, TerrainContactFeasibilityConfig)
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
    )
    frames = []
    while state.next_source_frame < state.playback_stop:
        step = advance_skill(state)
        frames.append(step.frame)
        state = step.state
    if not frames:
        raise ContractError("emitted contact preview produced no frames")

    reference = current.joint_position
    joints = torch.stack([frame.joint_position for frame in frames])
    roots = torch.stack([frame.root_position_world for frame in frames])
    orientations = torch.stack(
        [frame.root_orientation_world_wxyz for frame in frames]
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
    )
