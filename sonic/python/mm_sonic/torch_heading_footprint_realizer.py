"""Bounded realization of constant-heading footprint action plans."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .joints import ContractError
from .torch_flat_gait_contact_overlay import (
    anticipate_touchdown_positions,
    phase_contact_weights,
    source_relative_com_targets,
)
from .torch_heading_footprint_search import (
    FootprintActionEdge,
    HeadingFootprintPlan,
)
from .resample import shortest_path_slerp


@dataclass(frozen=True)
class RealizedHeadingTraversal:
    joint_position: np.ndarray
    root_position_world: np.ndarray
    root_orientation_world_wxyz: np.ndarray
    source_support_mask: np.ndarray
    source_frame_provenance: np.ndarray
    maximum_stance_error_m: float
    minimum_sole_clearance_m: float

    def __post_init__(self) -> None:
        joints = np.asarray(self.joint_position, dtype=np.float64)
        root = np.asarray(self.root_position_world, dtype=np.float64)
        quaternion = np.asarray(
            self.root_orientation_world_wxyz, dtype=np.float64
        )
        support = np.asarray(self.source_support_mask)
        provenance = np.asarray(self.source_frame_provenance)
        frames = len(joints)
        metrics = (
            self.maximum_stance_error_m,
            self.minimum_sole_clearance_m,
        )
        if (
            joints.shape != (frames, 29)
            or root.shape != (frames, 3)
            or quaternion.shape != (frames, 4)
            or support.shape != (frames, 2)
            or support.dtype != np.bool_
            or provenance.shape != (frames, 2)
            or provenance.dtype.kind not in "iu"
            or frames < 1
            or not bool(support.any(axis=1).all())
            or not all(
                np.isfinite(value).all()
                for value in (joints, root, quaternion)
            )
            or any(not math.isfinite(float(value)) for value in metrics)
        ):
            raise ContractError("realized heading traversal is invalid")
        object.__setattr__(
            self, "joint_position", np.ascontiguousarray(joints)
        )
        object.__setattr__(
            self, "root_position_world", np.ascontiguousarray(root)
        )
        object.__setattr__(
            self,
            "root_orientation_world_wxyz",
            np.ascontiguousarray(quaternion),
        )
        object.__setattr__(
            self, "source_support_mask", np.ascontiguousarray(support)
        )
        object.__setattr__(
            self,
            "source_frame_provenance",
            np.ascontiguousarray(provenance, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "maximum_stance_error_m",
            float(self.maximum_stance_error_m),
        )
        object.__setattr__(
            self,
            "minimum_sole_clearance_m",
            float(self.minimum_sole_clearance_m),
        )


class RealizationFailure(ContractError):
    def __init__(
        self,
        *,
        code: str,
        edge_index: int,
        source_action_key: tuple[int, int],
        reasons: tuple[str, ...],
    ) -> None:
        if (
            not isinstance(code, str)
            or not code
            or type(edge_index) is not int
            or edge_index < 0
            or not isinstance(source_action_key, tuple)
            or len(source_action_key) != 2
            or any(
                type(value) is not int or value < 0
                for value in source_action_key
            )
            or not isinstance(reasons, tuple)
            or not reasons
            or any(not isinstance(item, str) or not item for item in reasons)
        ):
            raise ContractError("heading realization failure is invalid")
        self.code = code
        self.edge_index = edge_index
        self.source_action_key = source_action_key
        self.reasons = reasons
        super().__init__(
            f"{code} at edge {edge_index} {source_action_key}: "
            f"{'; '.join(reasons)}"
        )


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _quaternion_multiply(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _rotate_xy(values: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    return np.asarray(values, dtype=np.float64) @ rotation.T


def _time_stretch_action_window(
    *,
    frames: np.ndarray,
    joint_position: np.ndarray,
    root_position_world: np.ndarray,
    foot_position_world: np.ndarray,
    root_orientation_world_wxyz: np.ndarray,
    support_mask: np.ndarray,
    scale: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Resample a source action at 50 Hz without changing contact order."""

    count = len(frames)
    target_count = int(round((count - 1) * float(scale))) + 1
    target_time = np.linspace(0.0, count - 1, target_count)
    source_time = np.arange(count, dtype=np.float64)

    def interpolate(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(count, -1)
        output = np.stack(
            [
                np.interp(target_time, source_time, flat[:, column])
                for column in range(flat.shape[1])
            ],
            axis=1,
        )
        return output.reshape((target_count, *values.shape[1:]))

    joints = interpolate(joint_position)
    root = interpolate(root_position_world)
    feet = interpolate(foot_position_world)
    quaternion = np.empty((target_count, 4), dtype=np.float64)
    for index, time in enumerate(target_time):
        left = int(math.floor(time))
        right = min(count - 1, left + 1)
        quaternion[index] = shortest_path_slerp(
            root_orientation_world_wxyz[left],
            root_orientation_world_wxyz[right],
            float(time - left),
        )
    nearest = np.clip(
        np.floor(target_time + 0.5).astype(np.int64), 0, count - 1
    )
    return (
        np.ascontiguousarray(frames[nearest]),
        np.ascontiguousarray(joints),
        np.ascontiguousarray(root),
        np.ascontiguousarray(feet),
        np.ascontiguousarray(quaternion),
        np.ascontiguousarray(support_mask[nearest]),
    )


def report_support_after_touchdown_settles(
    support_mask: object,
) -> np.ndarray:
    """Delay newly declared contact until one complete solved frame later."""

    support = np.asarray(support_mask)
    if (
        support.ndim != 2
        or support.shape[1] != 2
        or support.dtype != np.bool_
        or len(support) < 2
        or not bool(support.any(axis=1).all())
    ):
        raise ContractError("reported support mask is invalid")
    output = support.copy()
    touchdown = (~support[:-1]) & support[1:]
    for frame, foot in np.argwhere(touchdown):
        contact_frame = int(frame) + 1
        if contact_frame < len(output) - 1:
            output[contact_frame, int(foot)] = False
    return np.ascontiguousarray(output)


def blend_action_boundary(
    *,
    joint_position: object,
    root_position_world: object,
    root_orientation_world_wxyz: object,
    previous_joint_position: object,
    previous_root_position_world: object,
    previous_root_orientation_world_wxyz: object,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend a source action onto the exact previous kinematic state."""

    joints = np.asarray(joint_position, dtype=np.float64)
    root = np.asarray(root_position_world, dtype=np.float64)
    quaternion = np.asarray(
        root_orientation_world_wxyz, dtype=np.float64
    )
    previous_joint = np.asarray(
        previous_joint_position, dtype=np.float64
    )
    previous_root = np.asarray(
        previous_root_position_world, dtype=np.float64
    )
    previous_quaternion = np.asarray(
        previous_root_orientation_world_wxyz, dtype=np.float64
    )
    frames = len(joints)
    if (
        joints.shape != (frames, 29)
        or root.shape != (frames, 3)
        or quaternion.shape != (frames, 4)
        or previous_joint.shape != (29,)
        or previous_root.shape != (3,)
        or previous_quaternion.shape != (4,)
        or type(blend_frames) is not int
        or not 2 <= blend_frames <= frames
        or not all(
            np.isfinite(value).all()
            for value in (
                joints,
                root,
                quaternion,
                previous_joint,
                previous_root,
                previous_quaternion,
            )
        )
    ):
        raise ContractError("heading action boundary blend is invalid")
    output_joints = joints.copy()
    output_root = root.copy()
    output_quaternion = quaternion.copy()
    linear = np.linspace(0.0, 1.0, blend_frames)
    smooth = linear * linear * (3.0 - 2.0 * linear)
    output_joints[:blend_frames] = (
        (1.0 - smooth[:, None]) * previous_joint
        + smooth[:, None] * joints[:blend_frames]
    )
    output_root[:blend_frames] = (
        (1.0 - smooth[:, None]) * previous_root
        + smooth[:, None] * root[:blend_frames]
    )
    for frame, fraction in enumerate(smooth):
        output_quaternion[frame] = shortest_path_slerp(
            previous_quaternion,
            quaternion[frame],
            float(fraction),
        )
    return (
        np.ascontiguousarray(output_joints),
        np.ascontiguousarray(output_root),
        np.ascontiguousarray(output_quaternion),
    )


def _source_inventory(source: object):
    try:
        clips = source.clips
        root_index = int(source.root_body_index)
        foot_indices = tuple(int(value) for value in source.foot_body_indices)
        action_index = source.action_index
        support_for_clip = source.support_mask
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError(
            "heading realization source is invalid"
        ) from error
    if (
        len(foot_indices) != 2
        or not callable(support_for_clip)
        or not hasattr(action_index, "entry")
    ):
        raise ContractError("heading realization source is invalid")
    return (
        clips,
        root_index,
        foot_indices,
        action_index,
        support_for_clip,
    )


def _surface_sampler(terrain: object):
    sampler = getattr(terrain, "sample_surface", None)
    if sampler is None and callable(terrain):
        sampler = terrain
    if not callable(sampler):
        raise ContractError("heading realization terrain is invalid")
    return sampler


def _scene_to_world_xy(terrain: object, points: np.ndarray) -> np.ndarray:
    transform = getattr(terrain, "scene_to_world_xy", None)
    if transform is None:
        return np.asarray(points, dtype=np.float64)
    output = np.asarray(transform(points), dtype=np.float64)
    if output.shape != np.asarray(points).shape or not np.isfinite(
        output
    ).all():
        raise ContractError("heading realization scene transform is invalid")
    return output


def _scene_heading_to_world(
    terrain: object, heading: np.ndarray
) -> np.ndarray:
    transform = getattr(terrain, "scene_heading_to_world", None)
    output = (
        np.asarray(heading, dtype=np.float64)
        if transform is None
        else np.asarray(transform(heading), dtype=np.float64)
    )
    if (
        output.shape != (2,)
        or not np.isfinite(output).all()
        or np.linalg.norm(output) <= 1.0e-6
    ):
        raise ContractError("heading realization heading transform is invalid")
    return output / np.linalg.norm(output)


def _support_anchors(
    *,
    support: np.ndarray,
    transformed_source_feet: np.ndarray,
    touchdown_targets: dict[tuple[int, int], np.ndarray],
    initial_anchor: np.ndarray,
) -> np.ndarray:
    anchors = np.full_like(transformed_source_feet, np.nan)
    last_anchor = initial_anchor.copy()
    for foot in range(2):
        active_start: int | None = None
        for frame in range(len(support)):
            if support[frame, foot] and active_start is None:
                active_start = frame
                if frame == 0:
                    anchor = initial_anchor[foot]
                else:
                    key = (frame, foot)
                    anchor = touchdown_targets.get(
                        key, last_anchor[foot]
                    )
                last_anchor[foot] = anchor
            if support[frame, foot]:
                anchors[frame, foot] = anchor
            else:
                active_start = None
    return anchors


def lift_swing_targets_over_terrain(
    *,
    target_foot_position_world: object,
    support_mask: object,
    target_heading_world_xy: object,
    sample_surface: object,
    ankle_origin_sole_m: float,
    minimum_clearance_m: float,
    sample_uncertainty_m: float = 0.0,
) -> np.ndarray:
    """Lift swing ankles above the highest terrain under the complete sole."""

    targets = np.asarray(
        target_foot_position_world, dtype=np.float64
    )
    support = np.asarray(support_mask)
    heading = np.asarray(target_heading_world_xy, dtype=np.float64)
    if (
        targets.ndim != 3
        or targets.shape[1:] != (2, 3)
        or support.shape != targets.shape[:2]
        or support.dtype != np.bool_
        or heading.shape != (2,)
        or not np.isfinite(targets).all()
        or not np.isfinite(heading).all()
        or np.linalg.norm(heading) <= 1.0e-6
        or not callable(sample_surface)
        or ankle_origin_sole_m <= 0.0
        or minimum_clearance_m <= 0.0
        or not math.isfinite(float(sample_uncertainty_m))
        or sample_uncertainty_m < 0.0
    ):
        raise ContractError("swing terrain-clearance input is invalid")
    forward = heading / np.linalg.norm(heading)
    lateral = np.array((-forward[1], forward[0]))
    local_offsets = np.array(
        (
            (0.035, 0.0),
            (0.12, 0.0),
            (-0.05, 0.0),
            (0.12, 0.04),
            (0.12, -0.04),
            (-0.05, 0.04),
            (-0.05, -0.04),
        ),
        dtype=np.float64,
    )
    world_offsets = (
        local_offsets[:, :1] * forward
        + local_offsets[:, 1:] * lateral
    )
    uncertainty = float(sample_uncertainty_m)
    uncertainty_offsets = np.stack(
        (
            np.zeros(2, dtype=np.float64),
            uncertainty * forward,
            -uncertainty * forward,
            uncertainty * lateral,
            -uncertainty * lateral,
        )
    )
    world_offsets = (
        world_offsets[:, None, :]
        + uncertainty_offsets[None, :, :]
    ).reshape(-1, 2)
    sample_points = (
        targets[:, :, None, :2] + world_offsets[None, None]
    )
    surface = np.asarray(
        sample_surface(sample_points.reshape(-1, 2)),
        dtype=np.float64,
    ).reshape(sample_points.shape[:-1])
    if not np.isfinite(surface).all():
        raise ContractError("swing terrain-clearance samples are invalid")
    required_ankle_height = (
        surface.max(axis=2)
        + float(ankle_origin_sole_m)
        + float(minimum_clearance_m)
    )
    output = targets.copy()
    swing = ~support
    output[..., 2][swing] = np.maximum(
        output[..., 2][swing],
        required_ankle_height[swing],
    )
    return np.ascontiguousarray(output)


def delay_swing_progress_for_terrain(
    *,
    target_foot_position_world: object,
    support_mask: object,
    raw_terrain_lift_m: object,
    lift_threshold_m: float,
    maximum_delay_frames: int,
) -> np.ndarray:
    """Hold swing XY near liftoff while a large stair-face lift develops."""

    targets = np.asarray(
        target_foot_position_world, dtype=np.float64
    )
    support = np.asarray(support_mask)
    lift = np.asarray(raw_terrain_lift_m, dtype=np.float64)
    if (
        targets.ndim != 3
        or targets.shape[1:] != (2, 3)
        or support.shape != targets.shape[:2]
        or support.dtype != np.bool_
        or lift.shape != support.shape
        or not np.isfinite(targets).all()
        or not np.isfinite(lift).all()
        or bool((lift < 0.0).any())
        or not math.isfinite(float(lift_threshold_m))
        or lift_threshold_m <= 0.0
        or type(maximum_delay_frames) is not int
        or maximum_delay_frames < 1
    ):
        raise ContractError("swing terrain-progress delay input is invalid")
    output = targets.copy()
    for foot in range(2):
        frame = 0
        while frame < len(targets):
            if support[frame, foot]:
                frame += 1
                continue
            start = frame
            while frame < len(targets) and not support[frame, foot]:
                frame += 1
            stop = frame
            if (
                float(lift[start:stop, foot].max(initial=0.0))
                < float(lift_threshold_m)
            ):
                continue
            left = max(0, start - 1)
            right = min(len(targets) - 1, stop)
            span = right - left
            delay = min(maximum_delay_frames, max(1, span - 2))
            if span <= delay:
                continue
            source_xy = targets[left : right + 1, foot, :2]
            source_frames = np.arange(span + 1, dtype=np.float64)
            for target_frame in range(start, stop):
                local = target_frame - left
                warped = max(
                    0.0,
                    (local - delay) * span / (span - delay),
                )
                output[target_frame, foot, 0] = np.interp(
                    warped, source_frames, source_xy[:, 0]
                )
                output[target_frame, foot, 1] = np.interp(
                    warped, source_frames, source_xy[:, 1]
                )
    return np.ascontiguousarray(output)


def smooth_swing_terrain_height(
    *,
    required_height_m: object,
    support_mask: object,
    maximum_height_step_m: float,
) -> np.ndarray:
    """Build the least swing-height majorant with a bounded absolute slope."""

    required = np.asarray(required_height_m, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        required.ndim != 2
        or required.shape[1] != 2
        or support.shape != required.shape
        or support.dtype != np.bool_
        or not np.isfinite(required).all()
        or not math.isfinite(float(maximum_height_step_m))
        or maximum_height_step_m <= 0.0
    ):
        raise ContractError("swing terrain-height envelope is invalid")
    output = required.copy()
    step = float(maximum_height_step_m)
    for foot in range(2):
        frame = 0
        while frame < len(required):
            if support[frame, foot]:
                frame += 1
                continue
            start = frame
            while frame < len(required) and not support[frame, foot]:
                frame += 1
            stop = frame
            left = max(0, start - 1)
            right = min(len(required), stop + 1)
            segment = required[left:right, foot].copy()
            left_height = segment[0]
            right_height = segment[-1]
            left_capacity = (
                left_height + np.arange(len(segment)) * step
                if support[left, foot]
                else np.full(len(segment), np.inf)
            )
            right_capacity = (
                right_height
                + np.arange(len(segment) - 1, -1, -1) * step
                if support[right - 1, foot]
                else np.full(len(segment), np.inf)
            )
            capacity = np.minimum(left_capacity, right_capacity)
            exceeded = np.argwhere(segment > capacity + 1.0e-3)
            if len(exceeded):
                local_frame = int(exceeded[0, 0])
                raise ContractError(
                    "swing terrain height exceeds the authenticated phase "
                    f"for foot {foot} at frame {left + local_frame}: "
                    f"required {segment[local_frame]:.6f} m, "
                    f"capacity {capacity[local_frame]:.6f} m, "
                    f"swing interval [{start}, {stop})"
                )
            for index in range(1, len(segment)):
                segment[index] = max(
                    segment[index], segment[index - 1] - step
                )
            for index in range(len(segment) - 2, -1, -1):
                segment[index] = max(
                    segment[index], segment[index + 1] - step
                )
            output[left:right, foot] = np.maximum(
                output[left:right, foot], segment
            )
    output[support] = required[support]
    return np.ascontiguousarray(output)


def smooth_swing_terrain_lift(
    *,
    raw_lift_m: object,
    support_mask: object,
    maximum_height_step_m: float,
) -> np.ndarray:
    """Spread obstacle clearance through each authenticated swing interval."""

    raw = np.asarray(raw_lift_m, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        raw.ndim != 2
        or raw.shape[1] != 2
        or support.shape != raw.shape
        or support.dtype != np.bool_
        or not np.isfinite(raw).all()
        or bool((raw < 0.0).any())
        or not math.isfinite(float(maximum_height_step_m))
        or maximum_height_step_m <= 0.0
    ):
        raise ContractError("swing terrain-lift envelope is invalid")
    output = np.zeros_like(raw)
    step = float(maximum_height_step_m)
    for foot in range(2):
        frame = 0
        while frame < len(raw):
            if support[frame, foot]:
                frame += 1
                continue
            start = frame
            while frame < len(raw) and not support[frame, foot]:
                frame += 1
            stop = frame
            left = max(0, start - 1)
            right = min(len(raw), stop + 1)
            segment = raw[left:right, foot].copy()
            if support[left, foot]:
                segment[0] = 0.0
            if support[right - 1, foot]:
                segment[-1] = 0.0
            left_capacity = (
                np.arange(len(segment)) * step
                if support[left, foot]
                else np.full(len(segment), np.inf)
            )
            right_capacity = (
                np.arange(len(segment) - 1, -1, -1) * step
                if support[right - 1, foot]
                else np.full(len(segment), np.inf)
            )
            capacity = np.minimum(left_capacity, right_capacity)
            exceeded = np.argwhere(segment > capacity + 1.0e-9)
            if len(exceeded):
                local_frame = int(exceeded[0, 0])
                raise ContractError(
                    "swing terrain lift exceeds the authenticated phase "
                    f"for foot {foot} at frame {left + local_frame}: "
                    f"required {segment[local_frame]:.6f} m, "
                    f"capacity {capacity[local_frame]:.6f} m"
                )
            for index in range(1, len(segment)):
                segment[index] = max(
                    segment[index], segment[index - 1] - step
                )
            for index in range(len(segment) - 2, -1, -1):
                segment[index] = max(
                    segment[index], segment[index + 1] - step
                )
            output[left:right, foot] = np.maximum(
                output[left:right, foot], segment
            )
    output[support] = 0.0
    return np.ascontiguousarray(output)


def realize_heading_footprint_plan(
    *,
    plan: HeadingFootprintPlan,
    source: object,
    terrain: object,
    kinematics: object,
    retargeter: object,
    ankle_origin_sole_m: float = 0.035,
    touchdown_blend_frames: int = 8,
    boundary_blend_frames: int = 8,
    maximum_joint_step_rad: float = 0.30,
    maximum_root_step_m: float = 0.05,
    maximum_stance_error_m: float = 0.02,
    swing_sample_uncertainty_m: float | None = None,
    action_time_scale: float = 1.0,
    extend_terminal_swing: bool = False,
    departure_action_key: tuple[int, int] | None = None,
    departure_swing_clearance_m: float = 0.24,
    departure_swing_target_xy_world: np.ndarray | None = None,
) -> RealizedHeadingTraversal:
    """Rotate, contact-retarget, and concatenate selected source actions."""

    if (
        not isinstance(plan, HeadingFootprintPlan)
        or not callable(getattr(kinematics, "foot_positions", None))
        or not callable(
            getattr(kinematics, "center_of_mass_positions", None)
        )
        or not callable(getattr(retargeter, "solve_frame", None))
        or type(touchdown_blend_frames) is not int
        or touchdown_blend_frames < 1
        or type(boundary_blend_frames) is not int
        or boundary_blend_frames < 2
        or (
            swing_sample_uncertainty_m is not None
            and (
                isinstance(swing_sample_uncertainty_m, bool)
                or not isinstance(
                    swing_sample_uncertainty_m, (int, float)
                )
                or not math.isfinite(
                    float(swing_sample_uncertainty_m)
                )
                or float(swing_sample_uncertainty_m) < 0.0
            )
        )
        or (
            isinstance(action_time_scale, bool)
            or not isinstance(action_time_scale, (int, float))
            or not math.isfinite(float(action_time_scale))
            or float(action_time_scale) < 1.0
        )
        or type(extend_terminal_swing) is not bool
        or (
            departure_action_key is not None
            and (
                not isinstance(departure_action_key, tuple)
                or len(departure_action_key) != 2
                or any(
                    type(value) is not int or value < 0
                    for value in departure_action_key
                )
            )
        )
        or (
            extend_terminal_swing
            and departure_action_key is not None
        )
        or (
            departure_swing_target_xy_world is not None
            and (
                departure_action_key is None
                or np.asarray(
                    departure_swing_target_xy_world
                ).shape
                != (2,)
                or not np.isfinite(
                    np.asarray(departure_swing_target_xy_world)
                ).all()
            )
        )
    ):
        raise ContractError("heading realization input is invalid")
    bounds = (
        ankle_origin_sole_m,
        maximum_joint_step_rad,
        maximum_root_step_m,
        maximum_stance_error_m,
        action_time_scale,
        departure_swing_clearance_m,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in bounds
    ):
        raise ContractError("heading realization bounds are invalid")
    sampler = _surface_sampler(terrain)
    swing_uncertainty = (
        maximum_stance_error_m
        if swing_sample_uncertainty_m is None
        else float(swing_sample_uncertainty_m)
    )
    (
        clips,
        root_body_index,
        foot_body_indices,
        action_index,
        support_for_clip,
    ) = _source_inventory(source)
    target_heading_world = _scene_heading_to_world(
        terrain, plan.heading_scene_xy.detach().cpu().numpy()
    )
    target_yaw = math.atan2(
        float(target_heading_world[1]),
        float(target_heading_world[0]),
    )
    target_lateral_world = np.array(
        (-target_heading_world[1], target_heading_world[0], 0.0),
        dtype=np.float64,
    )
    target_foot_rotation = np.tile(
        np.stack(
            (
                np.append(target_heading_world, 0.0),
                target_lateral_world,
                np.array((0.0, 0.0, 1.0)),
            ),
            axis=1,
        ),
        (2, 1, 1),
    )

    all_joints: list[np.ndarray] = []
    all_roots: list[np.ndarray] = []
    all_quaternions: list[np.ndarray] = []
    all_support: list[np.ndarray] = []
    all_provenance: list[np.ndarray] = []
    previous_joint: np.ndarray | None = None
    previous_root: np.ndarray | None = None
    previous_quaternion: np.ndarray | None = None
    previous_support: np.ndarray | None = None
    previous_feet: np.ndarray | None = None
    maximum_stance = 0.0
    minimum_clearance = math.inf

    realization_edges = list(plan.edges)
    if departure_action_key is not None:
        realization_edges.append(
            FootprintActionEdge(
                action_key=departure_action_key,
                candidate_indices=(0, 0),
                descriptor_cost=0.0,
                transition_cost=0.0,
            )
        )
    for edge_index, edge in enumerate(realization_edges):
        departure_edge = edge_index >= len(plan.edges)
        action = action_index.entry(*edge.action_key)
        if action is None:
            raise RealizationFailure(
                code="missing_source_action",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=("selected action is absent from the source index",),
            )
        try:
            clip = clips[action.clip_index]
            raw_support = support_for_clip(action.clip_index)
            if isinstance(raw_support, torch.Tensor):
                raw_support = raw_support.detach().cpu().numpy()
            raw_support = np.asarray(raw_support, dtype=np.bool_)
            window_end_frame = action.end_frame
            terminal_extension = (
                extend_terminal_swing
                and edge_index == len(plan.edges) - 1
            )
            if departure_edge:
                window_end_frame = (
                    action.start_frame
                    + action.landing_frame_offsets[0]
                )
            elif terminal_extension:
                onset = (~raw_support[:-1]) & raw_support[1:]
                future_landings = (
                    np.argwhere(onset)[:, 0] + 1
                    if bool(onset.any())
                    else np.empty(0, dtype=np.int64)
                )
                future_landings = future_landings[
                    future_landings >= action.end_frame
                ]
                if len(future_landings):
                    window_end_frame = int(future_landings.min())
            frames = np.arange(
                action.start_frame, window_end_frame, dtype=np.int64
            )
            joints = np.asarray(
                clip.joint_position[frames], dtype=np.float64
            ).copy()
            body_position = np.asarray(
                clip.body_position_world[frames], dtype=np.float64
            )
            root = body_position[:, root_body_index].copy()
            source_feet = body_position[:, foot_body_indices].copy()
            quaternion = np.asarray(
                clip.body_quaternion_world_wxyz[
                    frames, root_body_index
                ],
                dtype=np.float64,
            ).copy()
            support = np.asarray(raw_support)[frames].astype(
                np.bool_, copy=True
            )
            height_changing_action = bool(
                (
                    torch.abs(action.landing_height_delta_m)
                    > 0.05
                ).any().item()
            )
            planned_on_terrain = (
                not departure_edge
                and any(
                    footprint.surface_height_m > 0.05
                    for footprint in plan.footprints[
                        2 * edge_index : 2 * edge_index + 2
                    ]
                )
            )
            if action_time_scale > 1.0 and (
                height_changing_action or planned_on_terrain
            ):
                (
                    frames,
                    joints,
                    root,
                    source_feet,
                    quaternion,
                    support,
                ) = _time_stretch_action_window(
                    frames=frames,
                    joint_position=joints,
                    root_position_world=root,
                    foot_position_world=source_feet,
                    root_orientation_world_wxyz=quaternion,
                    support_mask=support,
                    scale=action_time_scale,
                )
        except (IndexError, TypeError, ValueError, AttributeError) as error:
            raise RealizationFailure(
                code="invalid_source_window",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=("source arrays do not cover the selected action",),
            ) from error
        if (
            joints.shape != (len(frames), 29)
            or root.shape != (len(frames), 3)
            or source_feet.shape != (len(frames), 2, 3)
            or quaternion.shape != (len(frames), 4)
            or support.shape != (len(frames), 2)
            or len(frames) < 2
            or not bool(support.any(axis=1).all())
        ):
            raise RealizationFailure(
                code="invalid_source_window",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=("source support or kinematics are malformed",),
            )
        terminal_extension_applied = (
            terminal_extension
            and window_end_frame > action.end_frame
        )
        if terminal_extension and not terminal_extension_applied:
            raise RealizationFailure(
                code="terminal_mid_swing",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "source clip has no authenticated departure swing",
                ),
            )
        if (
            not departure_edge
            and
            not terminal_extension_applied
            and not bool(support[-1].all())
        ):
            raise RealizationFailure(
                code="terminal_mid_swing",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "selected source window does not end in complete support",
                ),
            )
        if (
            (terminal_extension_applied or departure_edge)
            and (
                len(support) < 4
                or not bool(support[-4:].any(axis=1).all())
                or bool(support[-4:].all(axis=1).any())
                or not bool((support[-4:] == support[-1]).all())
            )
        ):
            raise RealizationFailure(
                code="terminal_mid_swing",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "departure extension lacks four stable swing frames",
                ),
            )
        if previous_support is not None and not np.array_equal(
            support[0], previous_support
        ):
            raise RealizationFailure(
                code="support_phase_mismatch",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "source entry support differs from prior terminal support",
                ),
            )

        source_yaw = _yaw_from_wxyz(quaternion[0])
        yaw_delta = target_yaw - source_yaw
        root_start_xy = root[0, :2].copy()
        root[:, :2] = _rotate_xy(
            root[:, :2] - root_start_xy, yaw_delta
        )
        source_feet[:, :, :2] = _rotate_xy(
            source_feet[:, :, :2] - root_start_xy, yaw_delta
        )
        yaw_quaternion = np.array(
            (
                math.cos(yaw_delta / 2.0),
                0.0,
                0.0,
                math.sin(yaw_delta / 2.0),
            ),
            dtype=np.float64,
        )
        quaternion = _quaternion_multiply(
            np.broadcast_to(yaw_quaternion, quaternion.shape),
            quaternion,
        )

        if previous_feet is None:
            requested_root = getattr(
                source, "start_root_position_world", None
            )
            translation = (
                np.zeros(3, dtype=np.float64)
                if requested_root is None
                else np.asarray(requested_root, dtype=np.float64) - root[0]
            )
        else:
            translation = (
                previous_feet[support[0]]
                - source_feet[0, support[0]]
            ).mean(axis=0)
        root += translation
        source_feet += translation
        if previous_joint is not None:
            assert previous_root is not None
            assert previous_quaternion is not None
            blend_count = min(boundary_blend_frames, len(frames))
            joints, root, quaternion = blend_action_boundary(
                joint_position=joints,
                root_position_world=root,
                root_orientation_world_wxyz=quaternion,
                previous_joint_position=previous_joint,
                previous_root_position_world=previous_root,
                previous_root_orientation_world_wxyz=(
                    previous_quaternion
                ),
                blend_frames=blend_count,
            )
            source_feet = np.asarray(
                kinematics.foot_positions(joints, root, quaternion),
                dtype=np.float64,
            )
        if previous_feet is None:
            requested_start_feet = getattr(
                source, "start_foot_position_world", None
            )
            initial_anchor = (
                source_feet[0].copy()
                if requested_start_feet is None
                else np.asarray(
                    requested_start_feet, dtype=np.float64
                ).copy()
            )
            if (
                initial_anchor.shape != (2, 3)
                or not np.isfinite(initial_anchor).all()
            ):
                raise ContractError(
                    "heading realization start feet are invalid"
                )
        else:
            initial_anchor = previous_feet.copy()

        onset = (~support[:-1]) & support[1:]
        events = tuple(
            (int(frame) + 1, int(foot))
            for frame, foot in np.argwhere(onset)
        )
        expected_events_list: list[tuple[int, int]] = []
        if not departure_edge:
            previous_event_frame = -1
            for landing_foot in action.landing_feet:
                match = next(
                    (
                        event
                        for event in events
                        if event[0] > previous_event_frame
                        and event[1] == landing_foot
                    ),
                    None,
                )
                if match is None:
                    break
                expected_events_list.append(match)
                previous_event_frame = match[0]
        expected_events = tuple(expected_events_list)
        if not departure_edge and len(expected_events) != 2:
            expected_events = tuple(
                (
                    action.landing_frame_offsets[index],
                    action.landing_feet[index],
                )
                for index in range(2)
            )
        if any(event not in events for event in expected_events):
            raise RealizationFailure(
                code="source_contact_mismatch",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    "authenticated support events differ from action metadata",
                ),
            )
        edge_footprints = (
            ()
            if departure_edge
            else plan.footprints[
                2 * edge_index : 2 * edge_index + 2
            ]
        )
        touchdown_targets = {}
        for event, footprint in zip(expected_events, edge_footprints):
            target_world_xy = _scene_to_world_xy(
                terrain,
                footprint.center_scene_xy.detach().cpu().numpy(),
            )
            touchdown_targets[event] = np.array(
                (
                    float(target_world_xy[0]),
                    float(target_world_xy[1]),
                    footprint.surface_height_m
                    + float(ankle_origin_sole_m),
                ),
                dtype=np.float64,
            )
        try:
            stance_anchors = _support_anchors(
                support=support,
                transformed_source_feet=source_feet,
                touchdown_targets=touchdown_targets,
                initial_anchor=initial_anchor,
            )
            targets = anticipate_touchdown_positions(
                foot_position_target_world=source_feet,
                stance_anchor_world=stance_anchors,
                support_mask=support,
                blend_frames=touchdown_blend_frames,
            )
            targets[support] = stance_anchors[support]
            weights = 10.0 * phase_contact_weights(
                support_mask=support,
                minimum_swing_weight=0.1,
                blend_frames=touchdown_blend_frames,
            )
            if departure_edge:
                swing_foot = action.landing_feet[0]
                terminal_xy = (
                    source_feet[-1, swing_foot, :2]
                    if departure_swing_target_xy_world is None
                    else np.asarray(
                        departure_swing_target_xy_world,
                        dtype=np.float64,
                    )
                )
                terminal_surface = np.asarray(
                    sampler(terminal_xy[None]),
                    dtype=np.float64,
                )
                if (
                    terminal_surface.shape != (1,)
                    or not np.isfinite(terminal_surface).all()
                ):
                    raise ContractError(
                        "departure swing surface is invalid"
                    )
                target_z = (
                    float(terminal_surface[0])
                    + float(ankle_origin_sole_m)
                    + float(departure_swing_clearance_m)
                )
                correction_frames = min(8, len(frames))
                alpha = np.linspace(
                    0.0, 1.0, correction_frames
                )
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                start = len(frames) - correction_frames
                targets[start:, swing_foot, :2] = (
                    (1.0 - smooth[:, None])
                    * targets[start:, swing_foot, :2]
                    + smooth[:, None] * terminal_xy
                )
                targets[start:, swing_foot, 2] = (
                    (1.0 - smooth)
                    * targets[start:, swing_foot, 2]
                    + smooth * target_z
                )
                weights[start:, swing_foot] = 10.0
                # The late-swing route correction competes with the planted
                # leg through the shared pelvis. Keep the authenticated stance
                # contact dominant so moving the swing target cannot drag the
                # support foot across the tread.
                weights[support] = np.maximum(weights[support], 50.0)
            targets_before_terrain_lift = targets.copy()
            initially_lifted_targets = lift_swing_targets_over_terrain(
                target_foot_position_world=targets_before_terrain_lift,
                support_mask=support,
                target_heading_world_xy=target_heading_world,
                sample_surface=sampler,
                ankle_origin_sole_m=ankle_origin_sole_m,
                minimum_clearance_m=0.02,
                sample_uncertainty_m=swing_uncertainty,
            )
            targets_before_terrain_lift = delay_swing_progress_for_terrain(
                target_foot_position_world=targets_before_terrain_lift,
                support_mask=support,
                raw_terrain_lift_m=np.maximum(
                    0.0,
                    initially_lifted_targets[..., 2]
                    - targets_before_terrain_lift[..., 2],
                ),
                lift_threshold_m=0.08,
                maximum_delay_frames=32,
            )
            targets = lift_swing_targets_over_terrain(
                target_foot_position_world=targets_before_terrain_lift,
                support_mask=support,
                target_heading_world_xy=target_heading_world,
                sample_surface=sampler,
                ankle_origin_sole_m=ankle_origin_sole_m,
                minimum_clearance_m=0.02,
                sample_uncertainty_m=swing_uncertainty,
            )
            terrain_height_envelope = smooth_swing_terrain_height(
                required_height_m=targets[..., 2],
                support_mask=support,
                maximum_height_step_m=0.02,
            )
            targets[..., 2] = terrain_height_envelope
            terrain_lift_envelope = np.maximum(
                0.0,
                targets[..., 2]
                - targets_before_terrain_lift[..., 2],
            )
            terrain_lifted = terrain_lift_envelope > 1.0e-9
            weights[terrain_lifted] = np.maximum(
                weights[terrain_lifted], 20.0
            )
            source_com = kinematics.center_of_mass_positions(
                joints, root, quaternion
            )
            com_targets = source_relative_com_targets(
                source_center_of_mass_world=source_com,
                source_foot_position_world=source_feet,
                support_mask=support,
                stance_anchor_world=stance_anchors,
            )
        except ContractError as error:
            raise RealizationFailure(
                code="invalid_contact_targets",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(str(error),),
            ) from error

        solved_joints = []
        solved_roots = []
        window_feet = []
        for local_frame in range(len(frames)):
            required_target_feet = support[local_frame].copy()
            required_target_feet |= terrain_lifted[local_frame]
            if departure_edge and local_frame >= len(frames) - 4:
                # MotionBricks consumes these four poses as exact context.
                # Certify the planned swing geometry as well as the planted
                # contact instead of handing the model a missed route target.
                required_target_feet[:] = True
            kwargs = dict(
                joint_position=joints[local_frame],
                root_position_world=root[local_frame],
                root_orientation_world_wxyz=quaternion[local_frame],
                solve_feet=np.ones(2, dtype=np.bool_),
                enforce_target_error_feet=required_target_feet,
                target_foot_position_world=targets[local_frame],
                target_foot_position_weights=weights[local_frame],
                target_foot_rotation_world=target_foot_rotation,
                target_center_of_mass_world_xy=com_targets[local_frame],
                level_feet=np.ones(2, dtype=np.bool_),
            )
            if previous_joint is not None:
                kwargs["initial_joint_position"] = previous_joint
                kwargs["initial_root_position_world"] = previous_root
            try:
                solved_joint, solved_root = retargeter.solve_frame(
                    **kwargs
                )
            except (ContractError, ValueError, RuntimeError) as error:
                raise RealizationFailure(
                    code="retarget_unreachable",
                    edge_index=edge_index,
                    source_action_key=edge.action_key,
                    reasons=(
                        f"frame {local_frame}: {error}; "
                        "targets="
                        f"{targets[local_frame].tolist()}, "
                        "source_feet="
                        f"{source_feet[local_frame].tolist()}, "
                        "terrain_lift="
                        f"{terrain_lift_envelope[local_frame].tolist()}, "
                        f"support={support[local_frame].tolist()}",
                    ),
                ) from error
            solved_joint = np.asarray(
                solved_joint, dtype=np.float64
            )
            solved_root = np.asarray(solved_root, dtype=np.float64)
            actual_feet = np.asarray(
                kinematics.foot_positions(
                    solved_joint[None],
                    solved_root[None],
                    quaternion[local_frame : local_frame + 1],
                )[0],
                dtype=np.float64,
            )
            if previous_joint is not None:
                joint_step = float(
                    np.max(np.abs(solved_joint - previous_joint))
                )
                root_step = float(
                    np.linalg.norm(solved_root - previous_root)
                )
                if (
                    joint_step > maximum_joint_step_rad
                    or root_step > maximum_root_step_m
                ):
                    raise RealizationFailure(
                        code="boundary_discontinuity",
                        edge_index=edge_index,
                        source_action_key=edge.action_key,
                        reasons=(
                            f"joint step {joint_step:.6f} rad, "
                            f"root step {root_step:.6f} m",
                        ),
                    )
            stance_error = np.linalg.norm(
                actual_feet[support[local_frame]]
                - stance_anchors[local_frame, support[local_frame]],
                axis=1,
            )
            maximum_stance = max(
                maximum_stance, float(stance_error.max())
            )
            surface_height = np.asarray(
                sampler(actual_feet[:, :2]), dtype=np.float64
            )
            if surface_height.shape != (2,) or not np.isfinite(
                surface_height
            ).all():
                raise ContractError(
                    "heading realization terrain samples are invalid"
                )
            clearance = (
                actual_feet[:, 2]
                - float(ankle_origin_sole_m)
                - surface_height
            )
            minimum_clearance = min(
                minimum_clearance, float(clearance.min())
            )
            solved_joints.append(solved_joint.copy())
            solved_roots.append(solved_root.copy())
            window_feet.append(actual_feet.copy())
            previous_joint = solved_joint
            previous_root = solved_root
        if maximum_stance > maximum_stance_error_m:
            raise RealizationFailure(
                code="stance_error",
                edge_index=edge_index,
                source_action_key=edge.action_key,
                reasons=(
                    f"maximum stance error {maximum_stance:.6f} m",
                ),
            )

        # The first solved frame of every action is the boundary bridge.  It
        # may be close to the prior terminal state, but it is not guaranteed
        # identical; dropping it can collapse two bounded solver steps into
        # one discontinuous emitted step.
        keep = slice(None)
        all_joints.extend(solved_joints[keep])
        all_roots.extend(solved_roots[keep])
        all_quaternions.extend(quaternion[keep])
        all_support.extend(
            report_support_after_touchdown_settles(support)[keep]
        )
        all_provenance.extend(
            np.stack(
                (
                    np.full(len(frames), action.clip_index),
                    frames,
                ),
                axis=1,
            )[keep]
        )
        previous_support = support[-1].copy()
        previous_feet = window_feet[-1].copy()
        previous_quaternion = quaternion[-1].copy()

    return RealizedHeadingTraversal(
        joint_position=np.asarray(all_joints),
        root_position_world=np.asarray(all_roots),
        root_orientation_world_wxyz=np.asarray(all_quaternions),
        source_support_mask=np.asarray(all_support),
        source_frame_provenance=np.asarray(all_provenance),
        maximum_stance_error_m=maximum_stance,
        minimum_sole_clearance_m=minimum_clearance,
    )
