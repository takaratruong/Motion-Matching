"""Source-conditioned contact metrics for terrain-motion quality diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .torch_contact_oracle_actions import ContactPhaseAction
from .torch_terrain_contact_composition import (
    build_contact_target_trajectory,
    place_action_contact_anchored,
    project_contact_trajectory,
    project_contact_trajectory_with_stance_root,
)
from .torch_terrain_quality_preview import (
    build_quality_preview_from_placement,
    quality_state_as_oracle,
)
from .torch_terrain_quality_states import FrozenQualityState


ANKLE_ORIGIN_SOLE_M = 0.035
STANCE_CLEARANCE_TOLERANCE_M = 0.02
STANCE_VERTICAL_SPEED_MAX_MPS = 0.12
MOVING_COMMAND_SPEED_MIN_MPS = 0.10
TRANSITION_NEIGHBORHOOD_FRAMES = 15


def _readonly(value: np.ndarray, *, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ContactQualityMetrics:
    emitted_contact_mask: np.ndarray
    expected_stance_floating_mask: np.ndarray
    contact_agreement_fraction: float
    expected_stance_floating_fraction: tuple[float, float]
    maximum_no_contact_frames: int
    unload_count: tuple[int, int]
    touchdown_count: tuple[int, int]
    complete_step_count: int
    complete_steps_per_m: float
    source_stance_drift_m: tuple[float, float]
    transition_source_stance_drift_m: float
    steady_source_stance_drift_m: float
    command_to_unload_frames: tuple[int, ...]
    command_to_touchdown_frames: tuple[int, ...]

    def __post_init__(self) -> None:
        emitted = np.asarray(self.emitted_contact_mask)
        floating = np.asarray(self.expected_stance_floating_mask)
        if (
            emitted.dtype != np.bool_
            or floating.dtype != np.bool_
            or emitted.ndim != 2
            or emitted.shape[1:] != (2,)
            or floating.shape != emitted.shape
        ):
            raise ValueError("contact quality masks must have boolean shape (T, 2)")
        object.__setattr__(
            self, "emitted_contact_mask", _readonly(emitted, dtype=np.bool_)
        )
        object.__setattr__(
            self,
            "expected_stance_floating_mask",
            _readonly(floating, dtype=np.bool_),
        )


def _floating_array(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _boolean_array(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must have boolean shape {shape}")
    return array


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask.tolist():
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _event_indices(
    support: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    unload = support[:-1] & ~support[1:]
    touchdown = ~support[:-1] & support[1:]
    unload_indices = tuple(
        np.flatnonzero(unload[:, foot]).astype(np.int64, copy=False) + 1
        for foot in range(2)
    )
    touchdown_indices = tuple(
        np.flatnonzero(touchdown[:, foot]).astype(np.int64, copy=False) + 1
        for foot in range(2)
    )
    return unload_indices, touchdown_indices


def _complete_step_count(
    support: np.ndarray,
) -> int:
    waiting_for_touchdown = [False, False]
    count = 0
    for frame in range(1, support.shape[0]):
        for foot in range(2):
            if support[frame - 1, foot] and not support[frame, foot]:
                waiting_for_touchdown[foot] = True
            elif (
                waiting_for_touchdown[foot]
                and not support[frame - 1, foot]
                and support[frame, foot]
            ):
                count += 1
                waiting_for_touchdown[foot] = False
    return count


def _next_event_delays(
    moving: np.ndarray,
    events: tuple[np.ndarray, np.ndarray],
) -> tuple[int, ...]:
    onset = moving & ~np.concatenate((np.zeros(1, dtype=bool), moving[:-1]))
    combined = np.sort(np.concatenate(events))
    delays: list[int] = []
    for frame in np.flatnonzero(onset):
        future = combined[combined >= frame]
        if future.size:
            delays.append(int(future[0] - frame))
    return tuple(delays)


def evaluate_contact_quality(
    *,
    foot_position_world: np.ndarray,
    foot_surface_height_m: np.ndarray,
    source_support_mask: np.ndarray,
    root_position_world: np.ndarray,
    command_velocity_world_xy: np.ndarray,
    transition_start_mask: np.ndarray,
    dt_s: float = 0.02,
) -> ContactQualityMetrics:
    """Measure emitted contact against the source motion's expected contacts."""

    feet_input = np.asarray(foot_position_world)
    if feet_input.ndim != 3 or feet_input.shape[1:] != (2, 3):
        raise ValueError("foot_position_world must have finite shape (T, 2, 3)")
    frame_count = int(feet_input.shape[0])
    if frame_count < 1:
        raise ValueError("foot_position_world must contain at least one frame")
    feet = _floating_array(
        feet_input, "foot_position_world", (frame_count, 2, 3)
    )
    surface = _floating_array(
        foot_surface_height_m,
        "foot_surface_height_m",
        (frame_count, 2),
    )
    support = _boolean_array(
        source_support_mask, "source_support_mask", (frame_count, 2)
    )
    root = _floating_array(
        root_position_world, "root_position_world", (frame_count, 3)
    )
    command = _floating_array(
        command_velocity_world_xy,
        "command_velocity_world_xy",
        (frame_count, 2),
    )
    transitions = _boolean_array(
        transition_start_mask, "transition_start_mask", (frame_count,)
    )
    if (
        isinstance(dt_s, bool)
        or not isinstance(dt_s, (int, float))
        or not math.isfinite(float(dt_s))
        or float(dt_s) <= 0.0
    ):
        raise ValueError("dt_s must be finite and positive")

    clearance = feet[:, :, 2] - surface
    vertical_speed = np.zeros((frame_count, 2), dtype=np.float64)
    if frame_count > 1:
        vertical_speed[1:] = np.diff(feet[:, :, 2], axis=0) / float(dt_s)
    emitted = (
        np.abs(clearance - ANKLE_ORIGIN_SOLE_M)
        <= STANCE_CLEARANCE_TOLERANCE_M
    ) & (np.abs(vertical_speed) <= STANCE_VERTICAL_SPEED_MAX_MPS)
    floating = support & ~emitted
    support_counts = np.sum(support, axis=0)
    floating_counts = np.sum(floating, axis=0)
    floating_fraction = tuple(
        float(floating_counts[foot] / support_counts[foot])
        if support_counts[foot]
        else 0.0
        for foot in range(2)
    )

    unload, touchdown = _event_indices(support)
    complete_steps = _complete_step_count(support)
    path_length = (
        float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum())
        if frame_count > 1
        else 0.0
    )
    steps_per_m = complete_steps / path_length if path_length > 1e-9 else 0.0

    if frame_count > 1:
        horizontal_step = np.linalg.norm(
            np.diff(feet[:, :, :2], axis=0), axis=2
        )
        stance_intervals = support[:-1] & support[1:]
        stance_drift = np.where(stance_intervals, horizontal_step, 0.0)
    else:
        stance_drift = np.zeros((0, 2), dtype=np.float64)
    source_drift = tuple(float(value) for value in stance_drift.sum(axis=0))

    transition_frames = np.zeros(frame_count, dtype=bool)
    for start in np.flatnonzero(transitions):
        transition_frames[start : start + TRANSITION_NEIGHBORHOOD_FRAMES] = True
    transition_intervals = transition_frames[1:]
    transition_drift = float(stance_drift[transition_intervals].sum())
    steady_drift = float(stance_drift[~transition_intervals].sum())

    moving = np.linalg.norm(command, axis=1) >= MOVING_COMMAND_SPEED_MIN_MPS
    return ContactQualityMetrics(
        emitted_contact_mask=emitted,
        expected_stance_floating_mask=floating,
        contact_agreement_fraction=float(np.mean(emitted == support)),
        expected_stance_floating_fraction=floating_fraction,
        maximum_no_contact_frames=_longest_true_run(~emitted.any(axis=1)),
        unload_count=(int(unload[0].size), int(unload[1].size)),
        touchdown_count=(int(touchdown[0].size), int(touchdown[1].size)),
        complete_step_count=complete_steps,
        complete_steps_per_m=float(steps_per_m),
        source_stance_drift_m=source_drift,
        transition_source_stance_drift_m=transition_drift,
        steady_source_stance_drift_m=steady_drift,
        command_to_unload_frames=_next_event_delays(moving, unload),
        command_to_touchdown_frames=_next_event_delays(moving, touchdown),
    )


@dataclass(frozen=True)
class ContactQualityAblationResult:
    unprojected_qpos: np.ndarray
    projected_qpos: np.ndarray
    unprojected_foot_position_world: np.ndarray
    projected_foot_position_world: np.ndarray
    source_support_mask: np.ndarray
    placement_entry_error_m: float
    placed_stance_drift_m: float
    unprojected_stance_drift_m: float
    projected_stance_drift_m: float
    projected_landing_error_m: float
    maximum_target_error_m: float
    maximum_root_correction_m: float
    maximum_root_correction_speed_m_s: float
    maximum_joint_deformation_rad: float
    rms_joint_deformation_rad: float
    maximum_joint_correction_speed_rad_s: float

    def __post_init__(self) -> None:
        qpos = np.asarray(self.unprojected_qpos)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or qpos.shape[0] < 2:
            raise ValueError("contact quality ablation qpos is invalid")
        frames = qpos.shape[0]
        arrays = (
            ("unprojected_qpos", (frames, 36), np.float64),
            ("projected_qpos", (frames, 36), np.float64),
            ("unprojected_foot_position_world", (frames, 2, 3), np.float64),
            ("projected_foot_position_world", (frames, 2, 3), np.float64),
            ("source_support_mask", (frames, 2), np.bool_),
        )
        for name, shape, dtype in arrays:
            value = np.asarray(getattr(self, name))
            if value.shape != shape or (
                dtype != np.bool_ and not np.isfinite(value).all()
            ):
                raise ValueError("contact quality ablation arrays are invalid")
            object.__setattr__(self, name, _readonly(value, dtype=dtype))
        for name in (
            "placement_entry_error_m",
            "placed_stance_drift_m",
            "unprojected_stance_drift_m",
            "projected_stance_drift_m",
            "projected_landing_error_m",
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
                raise ValueError("contact quality ablation metrics are invalid")
            object.__setattr__(self, name, float(value))


def _source_stance_drift(feet: np.ndarray, support: np.ndarray) -> float:
    displacement = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
    return float(np.where(support[:-1] & support[1:], displacement, 0.0).sum())


def build_contact_quality_ablation(
    *,
    state: FrozenQualityState,
    action: ContactPhaseAction,
    desired_landing_foot: int,
    desired_landing_world_xyz: object,
    foot_kinematics: object,
    inertialization_halflife_s: float = 0.10,
    projection_strategy: str = "joint-only",
    root_correction_scale: float = 1.0,
    root_smoothing_passes: int = 0,
    joint_smoothing_passes: int = 0,
    reproject_smoothed_joints: bool = False,
) -> ContactQualityAblationResult:
    """Measure contact anchoring and projected composition on one action."""

    if not isinstance(state, FrozenQualityState) or not isinstance(
        action, ContactPhaseAction
    ):
        raise ValueError("contact quality ablation inputs are invalid")
    if (
        type(desired_landing_foot) is not int
        or desired_landing_foot not in (0, 1)
        or action.swing_foot != desired_landing_foot
    ):
        raise ValueError("contact quality ablation landing foot does not match action")
    desired = np.asarray(desired_landing_world_xyz, dtype=np.float64)
    if desired.shape != (3,) or not np.isfinite(desired).all():
        raise ValueError("contact quality ablation desired landing is invalid")
    if projection_strategy not in ("joint-only", "stance-root"):
        raise ValueError("contact quality ablation projection strategy is invalid")

    current = quality_state_as_oracle(state, action.joint_position)
    anchored = place_action_contact_anchored(action, current)
    preview = build_quality_preview_from_placement(
        state=state,
        action=action,
        placement=anchored.placed,
        foot_kinematics=foot_kinematics,
        inertialization_halflife_s=inertialization_halflife_s,
    )
    dtype = action.joint_position.dtype
    device = action.joint_position.device
    raw_qpos = np.asarray(preview.composed.qpos, dtype=np.float64)
    raw_feet = torch.as_tensor(
        np.array(preview.composed.foot_position_world, copy=True),
        dtype=dtype,
        device=device,
    )
    support = action.support_mask
    entry_feet = torch.as_tensor(
        np.array(state.foot_position_world, copy=True),
        dtype=dtype,
        device=device,
    )
    landing = torch.as_tensor(desired, dtype=dtype, device=device)
    roots = torch.as_tensor(
        np.array(raw_qpos[:, :3], copy=True), dtype=dtype, device=device
    )
    quaternions = torch.as_tensor(
        np.array(raw_qpos[:, 3:7], copy=True), dtype=dtype, device=device
    )
    raw_joints = torch.as_tensor(
        np.array(raw_qpos[:, 7:], copy=True), dtype=dtype, device=device
    )
    if projection_strategy == "joint-only":
        targets = build_contact_target_trajectory(
            raw_feet,
            support,
            swing_foot=action.swing_foot,
            entry_foot_position_world=entry_feet,
            landing_target_world=landing,
        )
        projection = project_contact_trajectory(
            joint_position=raw_joints,
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            targets=targets,
            foot_kinematics=foot_kinematics,
        )
        projected_roots = roots
        maximum_root_correction_m = 0.0
        maximum_root_correction_speed_m_s = 0.0
    else:
        projection = project_contact_trajectory_with_stance_root(
            joint_position=raw_joints,
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            raw_foot_position_world=raw_feet,
            support_mask=support,
            swing_foot=action.swing_foot,
            entry_foot_position_world=entry_feet,
            landing_target_world=landing,
            foot_kinematics=foot_kinematics,
            root_correction_scale=root_correction_scale,
            root_smoothing_passes=root_smoothing_passes,
            joint_smoothing_passes=joint_smoothing_passes,
            reproject_smoothed_joints=reproject_smoothed_joints,
        )
        projected_roots = projection.root_position_world
        maximum_root_correction_m = projection.maximum_root_correction_m
        maximum_root_correction_speed_m_s = (
            projection.maximum_root_correction_speed_m_s
        )
    projected_qpos = torch.cat(
        (projected_roots, quaternions, projection.joint_position), dim=1
    ).detach().cpu().numpy()
    projected_feet = projection.foot_position_world.detach().cpu().numpy()
    support_numpy = support.detach().cpu().numpy().astype(bool, copy=True)
    return ContactQualityAblationResult(
        unprojected_qpos=raw_qpos,
        projected_qpos=projected_qpos,
        unprojected_foot_position_world=raw_feet.detach().cpu().numpy(),
        projected_foot_position_world=projected_feet,
        source_support_mask=support_numpy,
        placement_entry_error_m=anchored.maximum_entry_support_error_m,
        placed_stance_drift_m=preview.placed.source_stance_drift_m,
        unprojected_stance_drift_m=preview.composed.source_stance_drift_m,
        projected_stance_drift_m=_source_stance_drift(
            projected_feet, support_numpy
        ),
        projected_landing_error_m=float(
            np.linalg.norm(projected_feet[-1, action.swing_foot] - desired)
        ),
        maximum_target_error_m=projection.maximum_target_error_m,
        maximum_root_correction_m=maximum_root_correction_m,
        maximum_root_correction_speed_m_s=maximum_root_correction_speed_m_s,
        maximum_joint_deformation_rad=projection.maximum_joint_deformation_rad,
        rms_joint_deformation_rad=projection.rms_joint_deformation_rad,
        maximum_joint_correction_speed_rad_s=(
            projection.maximum_joint_correction_speed_rad_s
        ),
    )
