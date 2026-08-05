"""Deterministic global search over terrain-contact motion windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .joints import ContractError


_ROLES = ("approach", "entry", "uneven_walk", "exit", "departure")
_MAXIMUM_HEIGHT_PATTERN_ERROR_M = 0.04
_MAXIMUM_HEADING_ERROR_RAD = math.radians(10.0)


def _vector(value: object, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 1
        or len(array) < 1
        or not np.isfinite(array).all()
    ):
        raise ContractError(f"contact window {name} is invalid")
    return tuple(float(item) for item in array)


def _support(value: object) -> tuple[bool, bool]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not bool for item in value)
        or not any(value)
    ):
        raise ContractError("contact window support is invalid")
    return value


@dataclass(frozen=True, order=True)
class ContactWindow:
    window_id: str
    role: str
    source_clip: str
    start_frame: int
    stop_frame: int
    support_start: tuple[bool, bool]
    support_stop: tuple[bool, bool]
    height_pattern_error_m: float
    timing_error_frames: float
    sole_transform_error_m: float
    heading_error_rad: float
    pelvis_error_m: float
    start_pose: tuple[float, ...]
    stop_pose: tuple[float, ...]
    start_velocity: tuple[float, ...]
    stop_velocity: tuple[float, ...]

    def __post_init__(self) -> None:
        metrics = (
            self.height_pattern_error_m,
            self.timing_error_frames,
            self.sole_transform_error_m,
            self.heading_error_rad,
            self.pelvis_error_m,
        )
        if (
            not isinstance(self.window_id, str)
            or not self.window_id
            or self.role not in _ROLES
            or not isinstance(self.source_clip, str)
            or not self.source_clip
            or type(self.start_frame) is not int
            or type(self.stop_frame) is not int
            or not 0 <= self.start_frame < self.stop_frame
            or any(
                not math.isfinite(float(value)) or value < 0.0
                for value in metrics
            )
        ):
            raise ContractError("contact window is invalid")
        object.__setattr__(self, "support_start", _support(self.support_start))
        object.__setattr__(self, "support_stop", _support(self.support_stop))
        for name in (
            "start_pose",
            "stop_pose",
            "start_velocity",
            "stop_velocity",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if (
            len(self.start_pose) != len(self.stop_pose)
            or len(self.start_velocity) != len(self.stop_velocity)
            or len(self.start_pose) != len(self.start_velocity)
        ):
            raise ContractError("contact window boundary dimensions differ")


@dataclass(frozen=True)
class ContactWindowPath:
    windows: tuple[ContactWindow, ...]
    total_cost: float

    def __post_init__(self) -> None:
        if (
            not self.windows
            or any(
                not isinstance(window, ContactWindow)
                for window in self.windows
            )
            or not math.isfinite(float(self.total_cost))
            or self.total_cost < 0.0
        ):
            raise ContractError("contact-window path is invalid")


def contact_window_cost(window: ContactWindow) -> float:
    """Return the intrinsic target-contact mismatch of one source window."""

    if not isinstance(window, ContactWindow):
        raise ContractError("contact window cost input is invalid")
    if window.height_pattern_error_m > _MAXIMUM_HEIGHT_PATTERN_ERROR_M:
        raise ContractError("contact window height pattern mismatch")
    if window.heading_error_rad > _MAXIMUM_HEADING_ERROR_RAD:
        raise ContractError("contact window heading mismatch")
    return (
        12.0 * window.height_pattern_error_m
        + 0.02 * window.timing_error_frames
        + 6.0 * window.sole_transform_error_m
        + 2.0 * window.heading_error_rad
        + 3.0 * window.pelvis_error_m
    )


def _transition_cost(left: ContactWindow, right: ContactWindow) -> float:
    if left.support_stop != right.support_start:
        raise ContractError("contact window support identity mismatch")
    left_pose = np.asarray(left.stop_pose, dtype=np.float64)
    right_pose = np.asarray(right.start_pose, dtype=np.float64)
    left_velocity = np.asarray(left.stop_velocity, dtype=np.float64)
    right_velocity = np.asarray(right.start_velocity, dtype=np.float64)
    if (
        left_pose.shape != right_pose.shape
        or left_velocity.shape != right_velocity.shape
        or left_pose.shape != left_velocity.shape
    ):
        raise ContractError("contact window boundary dimensions differ")
    return float(
        4.0 * np.linalg.norm(left_pose - right_pose)
        + 2.0 * np.linalg.norm(left_velocity - right_velocity)
    )


def best_contact_window_path(
    roles: Sequence[str], candidates: Sequence[ContactWindow]
) -> ContactWindowPath:
    """Solve one candidate per ordered role with deterministic dynamic programming."""

    ordered_roles = tuple(roles)
    windows = tuple(candidates)
    if (
        not ordered_roles
        or any(role not in _ROLES for role in ordered_roles)
        or len(set(ordered_roles)) != len(ordered_roles)
        or any(not isinstance(window, ContactWindow) for window in windows)
    ):
        raise ContractError("contact-window path input is invalid")
    by_role = {
        role: tuple(sorted((item for item in windows if item.role == role)))
        for role in ordered_roles
    }
    if any(not by_role[role] for role in ordered_roles):
        raise ContractError("no contact-window path")

    states: dict[str, tuple[float, tuple[str, ...], tuple[ContactWindow, ...]]] = {}
    for candidate in by_role[ordered_roles[0]]:
        try:
            cost = contact_window_cost(candidate)
        except ContractError:
            continue
        states[candidate.window_id] = (
            cost,
            (candidate.window_id,),
            (candidate,),
        )
    for role in ordered_roles[1:]:
        following: dict[
            str, tuple[float, tuple[str, ...], tuple[ContactWindow, ...]]
        ] = {}
        for candidate in by_role[role]:
            try:
                intrinsic = contact_window_cost(candidate)
            except ContractError:
                continue
            choices = []
            for cost, identifiers, path in states.values():
                try:
                    edge = _transition_cost(path[-1], candidate)
                except ContractError:
                    continue
                choices.append(
                    (
                        cost + intrinsic + edge,
                        identifiers + (candidate.window_id,),
                        path + (candidate,),
                    )
                )
            if choices:
                following[candidate.window_id] = min(
                    choices, key=lambda item: (item[0], item[1])
                )
        states = following
        if not states:
            raise ContractError("no contact-window path")
    cost, _, path = min(states.values(), key=lambda item: (item[0], item[1]))
    return ContactWindowPath(windows=path, total_cost=float(cost))


def _yaw_wxyz(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.unwrap(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def _touchdowns(support: np.ndarray) -> list[tuple[int, int]]:
    onset = (~support[:-1]) & support[1:]
    frame_foot = np.argwhere(onset)
    return sorted((int(frame) + 1, int(foot)) for frame, foot in frame_foot)


def contact_windows_from_source(
    *,
    source_clip: str,
    joint_position: object,
    root_position_world: object,
    root_orientation_world_wxyz: object,
    generalized_velocity: object,
    foot_position_world: object,
    support_mask: object,
    foot_surface_height_m: object,
    target_split_height_m: float,
    nominal_step_frames: float = 18.0,
) -> tuple[ContactWindow, ...]:
    """Extract sustained split-height windows from one authenticated clip."""

    joints = np.asarray(joint_position, dtype=np.float64)
    roots = np.asarray(root_position_world, dtype=np.float64)
    quaternions = np.asarray(
        root_orientation_world_wxyz, dtype=np.float64
    )
    velocities = np.asarray(generalized_velocity, dtype=np.float64)
    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    surface = np.asarray(foot_surface_height_m, dtype=np.float64)
    frames = len(joints)
    if (
        not isinstance(source_clip, str)
        or not source_clip
        or joints.ndim != 2
        or joints.shape[0] != frames
        or joints.shape[1] < 2
        or roots.shape != (frames, 3)
        or quaternions.shape != (frames, 4)
        or velocities.ndim != 2
        or velocities.shape[0] != frames
        or velocities.shape[1] < 2
        or feet.shape != (frames, 2, 3)
        or support.shape != (frames, 2)
        or support.dtype != np.bool_
        or surface.shape != (frames, 2)
        or frames < 8
        or not all(
            np.isfinite(value).all()
            for value in (
                joints,
                roots,
                quaternions,
                velocities,
                feet,
                surface,
            )
        )
        or np.any(
            np.abs(np.linalg.norm(quaternions, axis=1) - 1.0) > 1.0e-4
        )
        or not math.isfinite(float(target_split_height_m))
        or not 0.08 <= target_split_height_m <= 0.35
        or not math.isfinite(float(nominal_step_frames))
        or nominal_step_frames <= 0.0
    ):
        raise ContractError("source contact-window arrays are invalid")
    dimension = min(joints.shape[1], velocities.shape[1])
    yaw = _yaw_wxyz(quaternions)
    events = _touchdowns(support)
    output = []
    for event_index in range(max(0, len(events) - 3)):
        selected = events[event_index : event_index + 4]
        if (
            len(selected) != 4
            or tuple(foot for _, foot in selected)
            not in ((0, 1, 0, 1), (1, 0, 1, 0))
        ):
            continue
        start = selected[0][0]
        stop = min(frames, selected[-1][0] + 10)
        if (
            stop - start < 12
            or not bool(support[start:stop].any(axis=1).all())
        ):
            continue
        supported_levels = []
        for foot in (0, 1):
            mask = support[start:stop, foot]
            if int(mask.sum()) < 3:
                break
            supported_levels.append(
                float(np.median(surface[start:stop, foot][mask]))
            )
        if len(supported_levels) != 2:
            continue
        split = abs(supported_levels[1] - supported_levels[0])
        height_error = abs(split - float(target_split_height_m))
        if split < 0.08 or height_error > _MAXIMUM_HEIGHT_PATTERN_ERROR_M:
            continue
        progress = roots[stop - 1, :2] - roots[start, :2]
        distance = float(np.linalg.norm(progress))
        if distance < 0.20:
            continue
        travel_yaw = math.atan2(float(progress[1]), float(progress[0]))
        heading_error = float(
            np.max(
                np.abs(
                    np.arctan2(
                        np.sin(yaw[start:stop] - travel_yaw),
                        np.cos(yaw[start:stop] - travel_yaw),
                    )
                )
            )
        )
        if heading_error > _MAXIMUM_HEADING_ERROR_RAD:
            continue
        event_frames = np.asarray(
            [frame for frame, _ in selected], dtype=np.float64
        )
        timing_error = float(
            np.mean(np.abs(np.diff(event_frames) - nominal_step_frames))
        )
        lateral_axis = np.array(
            (-math.sin(travel_yaw), math.cos(travel_yaw)),
            dtype=np.float64,
        )
        relative_feet = feet[start:stop, :, :2] - roots[
            start:stop, None, :2
        ]
        lateral = relative_feet @ lateral_axis
        sole_error = float(
            abs(np.median(lateral[:, 0]) - np.median(lateral[:, 1]))
            * 0.05
        )
        pelvis_error = float(
            np.ptp(roots[start:stop, 2])
        )
        output.append(
            ContactWindow(
                window_id=(
                    f"{source_clip}:{start}:{stop}:uneven_walk"
                ),
                role="uneven_walk",
                source_clip=source_clip,
                start_frame=start,
                stop_frame=stop,
                support_start=tuple(
                    bool(value) for value in support[start]
                ),
                support_stop=tuple(
                    bool(value) for value in support[stop - 1]
                ),
                height_pattern_error_m=height_error,
                timing_error_frames=timing_error,
                sole_transform_error_m=sole_error,
                heading_error_rad=heading_error,
                pelvis_error_m=pelvis_error,
                start_pose=tuple(joints[start, :dimension]),
                stop_pose=tuple(joints[stop - 1, :dimension]),
                start_velocity=tuple(velocities[start, :dimension]),
                stop_velocity=tuple(velocities[stop - 1, :dimension]),
            )
        )

    for event_index in range(max(0, len(events) - 1)):
        first, second = events[event_index : event_index + 2]
        first_frame, first_foot = first
        second_frame, second_foot = second
        if first_foot == second_foot:
            continue
        previous_frames = []
        for frame, foot in (first, second):
            prior = np.flatnonzero(support[:frame, foot])
            if not len(prior):
                break
            previous_frames.append(int(prior[-1]))
        if len(previous_frames) != 2:
            continue
        before = np.array(
            (
                surface[previous_frames[0], first_foot],
                surface[previous_frames[1], second_foot],
            ),
            dtype=np.float64,
        )
        after = np.array(
            (
                surface[first_frame, first_foot],
                surface[second_frame, second_foot],
            ),
            dtype=np.float64,
        )
        initial_split = abs(float(before[1] - before[0]))
        height_error = abs(initial_split - float(target_split_height_m))
        if (
            height_error > _MAXIMUM_HEIGHT_PATTERN_ERROR_M
            or abs(float(after[1] - after[0])) > 0.04
            or not bool((before - after >= 0.08).all())
        ):
            continue
        start = max(0, min(previous_frames) - 5)
        stop = min(frames, second_frame + 10)
        if (
            stop - start < 12
            or not bool(support[start:stop].any(axis=1).all())
            or not bool(support[stop - 1].all())
        ):
            continue
        progress = roots[stop - 1, :2] - roots[start, :2]
        distance = float(np.linalg.norm(progress))
        if distance < 0.20:
            continue
        travel_yaw = math.atan2(float(progress[1]), float(progress[0]))
        heading_error = float(
            np.max(
                np.abs(
                    np.arctan2(
                        np.sin(yaw[start:stop] - travel_yaw),
                        np.cos(yaw[start:stop] - travel_yaw),
                    )
                )
            )
        )
        if heading_error > _MAXIMUM_HEADING_ERROR_RAD:
            continue
        timing_error = abs(
            float(second_frame - first_frame) - nominal_step_frames
        )
        output.append(
            ContactWindow(
                window_id=f"{source_clip}:{start}:{stop}:exit",
                role="exit",
                source_clip=source_clip,
                start_frame=start,
                stop_frame=stop,
                support_start=tuple(bool(value) for value in support[start]),
                support_stop=tuple(
                    bool(value) for value in support[stop - 1]
                ),
                height_pattern_error_m=height_error,
                timing_error_frames=timing_error,
                sole_transform_error_m=abs(
                    float(
                        np.linalg.norm(
                            (feet[first_frame, first_foot, :2]
                             - feet[second_frame, second_foot, :2])
                        )
                        - 0.24
                    )
                ),
                heading_error_rad=heading_error,
                pelvis_error_m=float(np.ptp(roots[start:stop, 2])),
                start_pose=tuple(joints[start, :dimension]),
                stop_pose=tuple(joints[stop - 1, :dimension]),
                start_velocity=tuple(velocities[start, :dimension]),
                stop_velocity=tuple(velocities[stop - 1, :dimension]),
            )
        )
    return tuple(sorted(output))
