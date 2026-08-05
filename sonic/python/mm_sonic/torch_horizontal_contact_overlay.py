"""Sole-aware contact overlays for one horizontal staircase crossing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable

import numpy as np

from .joints import ContractError


_SOLE_HALF_LENGTH_M = 0.11
_SOLE_HALF_WIDTH_M = 0.045
_SURFACE_TOLERANCE_M = 0.025


def _finite_tuple(values: object, length: int) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ContractError("horizontal contact vector is invalid")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, order=True)
class ContactInterval:
    foot: int
    start_frame: int
    stop_frame: int
    sole_center_scene_xyz: tuple[float, float, float]
    sole_yaw_rad: float
    role: str

    def __post_init__(self) -> None:
        center = _finite_tuple(self.sole_center_scene_xyz, 3)
        if (
            self.foot not in (0, 1)
            or type(self.start_frame) is not int
            or type(self.stop_frame) is not int
            or not 0 <= self.start_frame < self.stop_frame
            or not math.isfinite(float(self.sole_yaw_rad))
            or not isinstance(self.role, str)
            or not self.role
        ):
            raise ContractError("horizontal contact interval is invalid")
        object.__setattr__(self, "sole_center_scene_xyz", center)

    def sole_corners_scene_xy(self) -> np.ndarray:
        center = np.asarray(self.sole_center_scene_xyz[:2], dtype=np.float64)
        local = np.array(
            (
                (-_SOLE_HALF_LENGTH_M, -_SOLE_HALF_WIDTH_M),
                (-_SOLE_HALF_LENGTH_M, _SOLE_HALF_WIDTH_M),
                (_SOLE_HALF_LENGTH_M, -_SOLE_HALF_WIDTH_M),
                (_SOLE_HALF_LENGTH_M, _SOLE_HALF_WIDTH_M),
            ),
            dtype=np.float64,
        )
        cosine = math.cos(self.sole_yaw_rad)
        sine = math.sin(self.sole_yaw_rad)
        rotation = np.array(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        return local @ rotation.T + center


@dataclass(frozen=True, order=True)
class ContactPhase:
    role: str
    start_frame: int
    stop_frame: int

    def __post_init__(self) -> None:
        if (
            self.role
            not in ("approach", "entry", "uneven_walk", "exit", "departure")
            or type(self.start_frame) is not int
            or type(self.stop_frame) is not int
            or not 0 <= self.start_frame < self.stop_frame
        ):
            raise ContractError("horizontal contact phase is invalid")


@dataclass(frozen=True)
class HorizontalContactPlan:
    frames_per_second: float
    frame_count: int
    direction_scene_xy: tuple[float, float]
    heading_scene_yaw_rad: float
    leading_foot: int
    intervals: tuple[ContactInterval, ...]
    phases: tuple[ContactPhase, ...]

    def __post_init__(self) -> None:
        direction = _finite_tuple(self.direction_scene_xy, 2)
        if (
            not math.isfinite(float(self.frames_per_second))
            or self.frames_per_second <= 0.0
            or type(self.frame_count) is not int
            or self.frame_count < 2
            or direction != (1.0, 0.0)
            or not math.isfinite(float(self.heading_scene_yaw_rad))
            or abs(float(self.heading_scene_yaw_rad)) > 1.0e-8
            or self.leading_foot not in (0, 1)
            or not self.intervals
            or any(
                not isinstance(interval, ContactInterval)
                or interval.stop_frame > self.frame_count
                for interval in self.intervals
            )
            or tuple(phase.role for phase in self.phases)
            != ("approach", "entry", "uneven_walk", "exit", "departure")
            or self.phases[0].start_frame != 0
            or self.phases[-1].stop_frame != self.frame_count
            or any(
                left.stop_frame != right.start_frame
                for left, right in zip(self.phases[:-1], self.phases[1:])
            )
        ):
            raise ContractError("horizontal contact plan is invalid")
        object.__setattr__(self, "direction_scene_xy", direction)

    @property
    def plan_id(self) -> str:
        payload = {
            "fps": self.frames_per_second,
            "frames": self.frame_count,
            "heading": self.heading_scene_yaw_rad,
            "lead": self.leading_foot,
            "intervals": [
                (
                    item.foot,
                    item.start_frame,
                    item.stop_frame,
                    item.sole_center_scene_xyz,
                    item.sole_yaw_rad,
                    item.role,
                )
                for item in self.intervals
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def support_mask(self) -> np.ndarray:
        mask = np.zeros((self.frame_count, 2), dtype=np.bool_)
        for interval in self.intervals:
            mask[interval.start_frame : interval.stop_frame, interval.foot] = True
        return mask

    def overlapping_intervals(
        self,
    ) -> tuple[tuple[ContactInterval, ContactInterval], ...]:
        output = []
        left = [item for item in self.intervals if item.foot == 0]
        right = [item for item in self.intervals if item.foot == 1]
        for first in left:
            for second in right:
                if max(first.start_frame, second.start_frame) < min(
                    first.stop_frame, second.stop_frame
                ):
                    output.append((first, second))
        return tuple(output)


def _sole_surface(
    sample_height: Callable[[np.ndarray], object],
    *,
    center_xy: np.ndarray,
    heading_yaw: float,
) -> float:
    probe = ContactInterval(
        foot=0,
        start_frame=0,
        stop_frame=1,
        sole_center_scene_xyz=(float(center_xy[0]), float(center_xy[1]), 0.0),
        sole_yaw_rad=heading_yaw,
        role="probe",
    )
    heights = np.asarray(
        sample_height(probe.sole_corners_scene_xy()), dtype=np.float64
    )
    if heights.shape != (4,) or not np.isfinite(heights).all():
        raise ContractError("horizontal contact terrain sample is invalid")
    if float(np.ptp(heights)) > _SURFACE_TOLERANCE_M:
        raise ContractError("horizontal contact sole straddles a riser")
    return float(np.median(heights))


def _nearest_safe_x(
    sample_height: Callable[[np.ndarray], object],
    *,
    proposed_x: float,
    lane_y: float,
    heading_yaw: float,
    minimum_x: float,
) -> tuple[float, float]:
    offsets = np.concatenate(
        (
            np.array((0.0,), dtype=np.float64),
            np.linspace(0.005, 0.14, 28, dtype=np.float64),
            -np.linspace(0.005, 0.14, 28, dtype=np.float64),
        )
    )
    offsets = sorted(
        offsets.tolist(), key=lambda value: (abs(value), -value)
    )
    for offset in offsets:
        candidate_x = float(proposed_x + offset)
        if candidate_x <= minimum_x + 1.0e-6:
            continue
        try:
            height = _sole_surface(
                sample_height,
                center_xy=np.array((candidate_x, lane_y), dtype=np.float64),
                heading_yaw=heading_yaw,
            )
        except ContractError as error:
            if "straddles a riser" not in str(error):
                raise
            continue
        return candidate_x, height
    raise ContractError("horizontal contact sole straddles a riser")


def _roles_for_touchdowns(
    contacts: list[dict[str, object]],
) -> list[str]:
    latest_height = [float(contacts[0]["height"]), float(contacts[1]["height"])]
    ever_both_elevated = all(value > 0.04 for value in latest_height)
    began_exit = False
    roles = ["approach", "approach"]
    for contact in contacts[2:]:
        foot = int(contact["foot"])
        height = float(contact["height"])
        previous_both_elevated = all(value > 0.04 for value in latest_height)
        latest_height[foot] = height
        both_elevated = all(value > 0.04 for value in latest_height)
        both_ground = all(value <= 0.04 for value in latest_height)
        if began_exit:
            role = "departure" if both_ground else "exit"
        elif ever_both_elevated and height <= 0.04:
            began_exit = True
            role = "departure" if both_ground else "exit"
        elif both_elevated and (ever_both_elevated or previous_both_elevated):
            ever_both_elevated = True
            role = "uneven_walk"
        elif both_elevated:
            ever_both_elevated = True
            role = "entry"
        elif any(value > 0.04 for value in latest_height):
            role = "entry"
        else:
            role = "approach"
        roles.append(role)
    if not began_exit or roles.count("uneven_walk") < 2:
        raise ContractError("horizontal contact route lacks complete phases")
    return roles


def _phase_bounds(
    frame_count: int,
    contacts: list[dict[str, object]],
    roles: list[str],
) -> tuple[ContactPhase, ...]:
    required = ("approach", "entry", "uneven_walk", "exit", "departure")
    starts = []
    previous = 0
    for role in required:
        indices = [
            index
            for index, value in enumerate(roles)
            if value == role and index >= previous
        ]
        if not indices:
            raise ContractError("horizontal contact route lacks complete phases")
        index = indices[0]
        starts.append(int(contacts[index]["frame"]))
        previous = index
    starts[0] = 0
    bounds = starts + [frame_count]
    return tuple(
        ContactPhase(role, bounds[index], bounds[index + 1])
        for index, role in enumerate(required)
    )


def _one_plan(
    *,
    sample_height: Callable[[np.ndarray], object],
    route_start_scene_x: float,
    route_stop_scene_x: float,
    foot_lane_scene_y: tuple[float, float],
    step_length_m: float,
    step_frames: int,
    double_support_frames: int,
    leading_foot: int,
    phase_offset_frames: int,
    frames_per_second: float,
    heading_scene_yaw_rad: float,
) -> HorizontalContactPlan:
    lanes = _finite_tuple(foot_lane_scene_y, 2)
    # Reject a lane that crosses a riser even when the sole is well inside the
    # staircase footprint.
    middle_x = 0.5 * (route_start_scene_x + route_stop_scene_x)
    for lane in lanes:
        _sole_surface(
            sample_height,
            center_xy=np.array((middle_x, lane), dtype=np.float64),
            heading_yaw=heading_scene_yaw_rad,
        )

    contacts: list[dict[str, object]] = []
    initial_offsets = (
        (0.0, 0.35 * step_length_m)
        if leading_foot == 0
        else (0.35 * step_length_m, 0.0)
    )
    for foot in (0, 1):
        x, height = _nearest_safe_x(
            sample_height,
            proposed_x=route_start_scene_x + initial_offsets[foot],
            lane_y=lanes[foot],
            heading_yaw=heading_scene_yaw_rad,
            minimum_x=-math.inf,
        )
        contacts.append(
            {"frame": 0, "foot": foot, "x": x, "height": height}
        )

    next_x = max(float(item["x"]) for item in contacts) + step_length_m
    next_frame = step_frames + phase_offset_frames
    foot = leading_foot
    last_x = max(float(item["x"]) for item in contacts)
    while last_x < route_stop_scene_x:
        x, height = _nearest_safe_x(
            sample_height,
            proposed_x=next_x,
            lane_y=lanes[foot],
            heading_yaw=heading_scene_yaw_rad,
            minimum_x=last_x,
        )
        contacts.append(
            {"frame": next_frame, "foot": foot, "x": x, "height": height}
        )
        last_x = x
        next_x = x + step_length_m
        next_frame += step_frames
        foot = 1 - foot
        if len(contacts) > 64:
            raise ContractError("horizontal contact route does not terminate")

    # Add a final opposite-foot touchdown so the departure ends in support.
    if int(contacts[-1]["foot"]) == leading_foot:
        x, height = _nearest_safe_x(
            sample_height,
            proposed_x=float(contacts[-1]["x"]) + step_length_m,
            lane_y=lanes[1 - leading_foot],
            heading_yaw=heading_scene_yaw_rad,
            minimum_x=float(contacts[-1]["x"]),
        )
        contacts.append(
            {
                "frame": next_frame,
                "foot": 1 - leading_foot,
                "x": x,
                "height": height,
            }
        )
        next_frame += step_frames

    roles = _roles_for_touchdowns(contacts)
    frame_count = int(contacts[-1]["frame"]) + step_frames
    intervals: list[ContactInterval] = []
    for index, contact in enumerate(contacts):
        foot = int(contact["foot"])
        start = int(contact["frame"])
        later_opposite = next(
            (
                item
                for item in contacts[index + 1 :]
                if int(item["foot"]) != foot
                and int(item["frame"]) > start
            ),
            None,
        )
        stop = (
            min(
                frame_count,
                int(later_opposite["frame"]) + double_support_frames,
            )
            if later_opposite is not None
            else frame_count
        )
        # The foot selected to swing first only owns a short initial stance.
        if index < 2 and foot == leading_foot:
            stop = min(stop, max(1, double_support_frames))
        if stop <= start:
            continue
        intervals.append(
            ContactInterval(
                foot=foot,
                start_frame=start,
                stop_frame=stop,
                sole_center_scene_xyz=(
                    float(contact["x"]),
                    lanes[foot],
                    float(contact["height"]),
                ),
                sole_yaw_rad=heading_scene_yaw_rad,
                role=roles[index],
            )
        )
    plan = HorizontalContactPlan(
        frames_per_second=frames_per_second,
        frame_count=frame_count,
        direction_scene_xy=(1.0, 0.0),
        heading_scene_yaw_rad=heading_scene_yaw_rad,
        leading_foot=leading_foot,
        intervals=tuple(sorted(intervals)),
        phases=_phase_bounds(frame_count, contacts, roles),
    )
    if not bool(plan.support_mask().any(axis=1).all()):
        raise ContractError("horizontal contact plan has unsupported frames")
    return plan


def horizontal_contact_plan_variants(
    *,
    sample_height: Callable[[np.ndarray], object],
    route_start_scene_x: float,
    route_stop_scene_x: float,
    foot_lane_scene_y: object,
    nominal_step_length_m: float,
    nominal_step_frames: int,
    nominal_double_support_frames: int,
    frames_per_second: float = 50.0,
    heading_scene_yaw_rad: float = 0.0,
) -> tuple[HorizontalContactPlan, ...]:
    """Overlay bounded normal-walking contact variants on one staircase."""

    if (
        not callable(sample_height)
        or not math.isfinite(float(route_start_scene_x))
        or not math.isfinite(float(route_stop_scene_x))
        or route_stop_scene_x - route_start_scene_x < 0.8
        or not math.isfinite(float(nominal_step_length_m))
        or not 0.10 <= nominal_step_length_m <= 0.30
        or type(nominal_step_frames) is not int
        or not 8 <= nominal_step_frames <= 40
        or type(nominal_double_support_frames) is not int
        or not 1 <= nominal_double_support_frames < nominal_step_frames
        or not math.isfinite(float(frames_per_second))
        or frames_per_second <= 0.0
        or not math.isfinite(float(heading_scene_yaw_rad))
        or abs(float(heading_scene_yaw_rad)) > 1.0e-8
    ):
        raise ContractError("horizontal contact heading or gait is invalid")
    _finite_tuple(foot_lane_scene_y, 2)

    plans = []
    for leading_foot in (0, 1):
        for length_scale, phase_offset in (
            (0.90, 0),
            (1.00, 0),
            (1.10, 2),
        ):
            plans.append(
                _one_plan(
                    sample_height=sample_height,
                    route_start_scene_x=float(route_start_scene_x),
                    route_stop_scene_x=float(route_stop_scene_x),
                    foot_lane_scene_y=_finite_tuple(foot_lane_scene_y, 2),
                    step_length_m=nominal_step_length_m * length_scale,
                    step_frames=nominal_step_frames,
                    double_support_frames=nominal_double_support_frames,
                    leading_foot=leading_foot,
                    phase_offset_frames=phase_offset,
                    frames_per_second=float(frames_per_second),
                    heading_scene_yaw_rad=float(heading_scene_yaw_rad),
                )
            )
    return tuple(sorted(plans, key=lambda plan: plan.plan_id))
