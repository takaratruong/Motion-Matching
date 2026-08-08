"""Object-local multi-heading traversal field geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .joints import ContractError


@dataclass(frozen=True)
class MotionFieldLine:
    family_index: int
    lane_index: int
    line_id: str
    heading_degrees: float
    heading_scene_xy: tuple[float, float]
    lateral_offset_m: float
    start_scene_xy: tuple[float, float]
    stop_scene_xy: tuple[float, float]

    def __post_init__(self) -> None:
        heading = np.asarray(self.heading_scene_xy, dtype=np.float64)
        start = np.asarray(self.start_scene_xy, dtype=np.float64)
        stop = np.asarray(self.stop_scene_xy, dtype=np.float64)
        if (
            type(self.family_index) is not int
            or self.family_index < 0
            or type(self.lane_index) is not int
            or self.lane_index < 0
            or not isinstance(self.line_id, str)
            or not self.line_id
            or heading.shape != (2,)
            or start.shape != (2,)
            or stop.shape != (2,)
            or not np.isfinite(heading).all()
            or not np.isfinite(start).all()
            or not np.isfinite(stop).all()
            or not math.isfinite(float(self.heading_degrees))
            or not math.isfinite(float(self.lateral_offset_m))
            or abs(float(np.linalg.norm(heading)) - 1.0) > 1.0e-9
            or float(np.linalg.norm(stop - start)) <= 0.0
        ):
            raise ContractError("motion field line is invalid")


@dataclass(frozen=True)
class MotionFieldIntersection:
    first_line_id: str
    second_line_id: str
    first_heading_degrees: float
    second_heading_degrees: float
    scene_xy: tuple[float, float]
    first_progress_m: float
    second_progress_m: float

    def __post_init__(self) -> None:
        point = np.asarray(self.scene_xy, dtype=np.float64)
        values = (
            self.first_heading_degrees,
            self.second_heading_degrees,
            self.first_progress_m,
            self.second_progress_m,
        )
        if (
            not self.first_line_id
            or not self.second_line_id
            or self.first_line_id == self.second_line_id
            or point.shape != (2,)
            or not np.isfinite(point).all()
            or not all(math.isfinite(float(value)) for value in values)
            or self.first_progress_m < 0.0
            or self.second_progress_m < 0.0
        ):
            raise ContractError("motion field intersection is invalid")


@dataclass(frozen=True)
class MotionFieldLineAssignment:
    line_id: str
    lane_index: int
    heading_degrees: float
    projected_scene_xy: tuple[float, float]
    progress_m: float
    lateral_distance_m: float

    def __post_init__(self) -> None:
        point = np.asarray(self.projected_scene_xy, dtype=np.float64)
        if (
            not self.line_id
            or type(self.lane_index) is not int
            or self.lane_index < 0
            or point.shape != (2,)
            or not np.isfinite(point).all()
            or not all(
                math.isfinite(float(value))
                for value in (
                    self.heading_degrees,
                    self.progress_m,
                    self.lateral_distance_m,
                )
            )
            or self.progress_m < 0.0
            or self.lateral_distance_m < 0.0
        ):
            raise ContractError("motion field line assignment is invalid")


@dataclass(frozen=True)
class MotionFieldRouteTrace:
    line_id: str
    root_scene_xy: np.ndarray
    support_mask: np.ndarray
    surface_height_m: np.ndarray

    def __post_init__(self) -> None:
        roots = np.asarray(self.root_scene_xy, dtype=np.float64)
        support = np.asarray(self.support_mask)
        height = np.asarray(self.surface_height_m, dtype=np.float64)
        frames = len(roots)
        if (
            not isinstance(self.line_id, str)
            or not self.line_id
            or frames < 1
            or roots.shape != (frames, 2)
            or support.shape != (frames, 2)
            or support.dtype != np.bool_
            or height.shape != (frames,)
            or not np.isfinite(roots).all()
            or not np.isfinite(height).all()
        ):
            raise ContractError("motion field route trace is invalid")
        owned_roots = np.ascontiguousarray(roots).copy()
        owned_support = np.ascontiguousarray(support).copy()
        owned_height = np.ascontiguousarray(height).copy()
        for value in (owned_roots, owned_support, owned_height):
            value.setflags(write=False)
        object.__setattr__(self, "root_scene_xy", owned_roots)
        object.__setattr__(self, "support_mask", owned_support)
        object.__setattr__(self, "surface_height_m", owned_height)


@dataclass(frozen=True)
class TransitionOpportunity:
    first_line_id: str
    second_line_id: str
    scene_xy: tuple[float, float]
    first_frame: int
    second_frame: int
    first_distance_m: float
    second_distance_m: float
    height_difference_m: float
    first_support: tuple[bool, bool]
    second_support: tuple[bool, bool]
    status: str
    reason: str

    def __post_init__(self) -> None:
        point = np.asarray(self.scene_xy, dtype=np.float64)
        distances = (
            self.first_distance_m,
            self.second_distance_m,
            self.height_difference_m,
        )
        if (
            not self.first_line_id
            or not self.second_line_id
            or self.first_line_id == self.second_line_id
            or point.shape != (2,)
            or not np.isfinite(point).all()
            or type(self.first_frame) is not int
            or type(self.second_frame) is not int
            or self.first_frame < -1
            or self.second_frame < -1
            or not all(math.isfinite(float(value)) for value in distances)
            or any(float(value) < 0.0 for value in distances)
            or len(self.first_support) != 2
            or len(self.second_support) != 2
            or any(type(value) is not bool for value in self.first_support)
            or any(type(value) is not bool for value in self.second_support)
            or self.status not in {"admitted", "rejected"}
            or not self.reason
        ):
            raise ContractError("motion field transition opportunity is invalid")


def _heading_id(value: float) -> str:
    sign = "m" if value < 0.0 else "p"
    magnitude = f"{abs(value):07.3f}".replace(".", "p")
    return f"{sign}{magnitude}"


def rasterized_motion_field(
    *,
    elevated_scene_xy: object,
    heading_degrees: Sequence[float],
    spacing_m: float,
    approach_margin_m: float,
    exit_margin_m: float,
) -> tuple[MotionFieldLine, ...]:
    """Rasterize object-crossing parallel line families from its footprint."""

    points = np.asarray(elevated_scene_xy, dtype=np.float64)
    headings = tuple(float(value) for value in heading_degrees)
    scalars = (spacing_m, approach_margin_m, exit_margin_m)
    canonical = tuple(round(value % 360.0, 9) for value in headings)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or not np.isfinite(points).all()
        or not headings
        or not np.isfinite(headings).all()
        or len(set(canonical)) != len(canonical)
        or not all(math.isfinite(float(value)) for value in scalars)
        or spacing_m <= 0.0
        or approach_margin_m <= 0.0
        or exit_margin_m <= 0.0
    ):
        raise ContractError("motion field geometry input is invalid")

    output = []
    for family_index, degrees in enumerate(headings):
        radians = math.radians(degrees)
        forward = np.array(
            (math.cos(radians), math.sin(radians)), dtype=np.float64
        )
        lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
        forward_projection = points @ forward
        lateral_projection = points @ lateral
        lower = float(lateral_projection.min())
        upper = float(lateral_projection.max())
        lane_count = max(
            1,
            math.floor((upper - lower) / spacing_m + 1.0e-9) + 1,
        )
        first = (
            0.5 * (lower + upper)
            - 0.5 * (lane_count - 1) * spacing_m
        )
        forward_start = float(forward_projection.min()) - approach_margin_m
        forward_stop = float(forward_projection.max()) + exit_margin_m
        for lane_index in range(lane_count):
            offset = float(first + lane_index * spacing_m)
            start = forward_start * forward + offset * lateral
            stop = forward_stop * forward + offset * lateral
            output.append(
                MotionFieldLine(
                    family_index=family_index,
                    lane_index=lane_index,
                    line_id=(
                        f"heading-{_heading_id(degrees)}-lane-{lane_index:03d}"
                    ),
                    heading_degrees=degrees,
                    heading_scene_xy=tuple(float(value) for value in forward),
                    lateral_offset_m=offset,
                    start_scene_xy=tuple(float(value) for value in start),
                    stop_scene_xy=tuple(float(value) for value in stop),
                )
            )
    return tuple(output)


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def motion_field_intersections(
    lines: Sequence[MotionFieldLine],
    *,
    endpoint_margin_m: float = 0.0,
) -> tuple[MotionFieldIntersection, ...]:
    """Return exact finite-segment intersections between heading families."""

    owned = tuple(lines)
    if (
        not owned
        or any(not isinstance(line, MotionFieldLine) for line in owned)
        or len({line.line_id for line in owned}) != len(owned)
        or not math.isfinite(float(endpoint_margin_m))
        or endpoint_margin_m < 0.0
    ):
        raise ContractError("motion field intersection input is invalid")
    output = []
    tolerance = 1.0e-10
    for first_index, first in enumerate(owned):
        p = np.asarray(first.start_scene_xy, dtype=np.float64)
        r = np.asarray(first.stop_scene_xy, dtype=np.float64) - p
        first_length = float(np.linalg.norm(r))
        for second in owned[first_index + 1 :]:
            if first.family_index == second.family_index:
                continue
            q = np.asarray(second.start_scene_xy, dtype=np.float64)
            s = np.asarray(second.stop_scene_xy, dtype=np.float64) - q
            denominator = _cross(r, s)
            if abs(denominator) <= tolerance:
                continue
            displacement = q - p
            first_fraction = _cross(displacement, s) / denominator
            second_fraction = _cross(displacement, r) / denominator
            second_length = float(np.linalg.norm(s))
            first_progress = first_fraction * first_length
            second_progress = second_fraction * second_length
            if (
                first_progress < endpoint_margin_m - tolerance
                or first_progress > first_length - endpoint_margin_m + tolerance
                or second_progress < endpoint_margin_m - tolerance
                or second_progress > second_length - endpoint_margin_m + tolerance
            ):
                continue
            point = p + first_fraction * r
            output.append(
                MotionFieldIntersection(
                    first_line_id=first.line_id,
                    second_line_id=second.line_id,
                    first_heading_degrees=first.heading_degrees,
                    second_heading_degrees=second.heading_degrees,
                    scene_xy=tuple(float(value) for value in point),
                    first_progress_m=float(first_progress),
                    second_progress_m=float(second_progress),
                )
            )
    return tuple(output)


def assign_endpoint_to_heading_line(
    *,
    lines: Sequence[MotionFieldLine],
    heading_degrees: float,
    endpoint_scene_xy: object,
) -> MotionFieldLineAssignment:
    """Assign a finite-radius motion endpoint to its nearest heading lane."""

    owned = tuple(lines)
    point = np.asarray(endpoint_scene_xy, dtype=np.float64)
    if (
        not owned
        or any(not isinstance(line, MotionFieldLine) for line in owned)
        or len({line.line_id for line in owned}) != len(owned)
        or isinstance(heading_degrees, bool)
        or not isinstance(heading_degrees, (int, float))
        or not math.isfinite(float(heading_degrees))
        or point.shape != (2,)
        or not np.isfinite(point).all()
    ):
        raise ContractError("motion field endpoint assignment input is invalid")

    def wrapped(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0

    candidates = []
    for line in owned:
        if abs(wrapped(line.heading_degrees - float(heading_degrees))) > 1.0e-6:
            continue
        start = np.asarray(line.start_scene_xy, dtype=np.float64)
        displacement = (
            np.asarray(line.stop_scene_xy, dtype=np.float64) - start
        )
        squared_length = float(displacement @ displacement)
        fraction = float(
            np.clip(((point - start) @ displacement) / squared_length, 0.0, 1.0)
        )
        projected = start + fraction * displacement
        distance = float(np.linalg.norm(point - projected))
        length = math.sqrt(squared_length)
        assignment = MotionFieldLineAssignment(
            line_id=line.line_id,
            lane_index=line.lane_index,
            heading_degrees=line.heading_degrees,
            projected_scene_xy=tuple(float(value) for value in projected),
            progress_m=fraction * length,
            lateral_distance_m=distance,
        )
        candidates.append((distance, line.line_id, assignment))
    if not candidates:
        raise ContractError("motion field endpoint heading is unavailable")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def transition_opportunities(
    *,
    intersections: Sequence[MotionFieldIntersection],
    route_traces: Sequence[MotionFieldRouteTrace],
    search_radius_m: float,
    maximum_height_difference_m: float,
) -> tuple[TransitionOpportunity, ...]:
    """Find deterministic contact-compatible frame pairs near intersections."""

    crossings = tuple(intersections)
    traces = tuple(route_traces)
    if (
        not crossings
        or any(
            not isinstance(item, MotionFieldIntersection)
            for item in crossings
        )
        or any(not isinstance(item, MotionFieldRouteTrace) for item in traces)
        or len({item.line_id for item in traces}) != len(traces)
        or not math.isfinite(float(search_radius_m))
        or search_radius_m <= 0.0
        or not math.isfinite(float(maximum_height_difference_m))
        or maximum_height_difference_m < 0.0
    ):
        raise ContractError("motion field transition input is invalid")
    by_line = {item.line_id: item for item in traces}
    output = []

    def record(
        crossing: MotionFieldIntersection,
        *,
        first: MotionFieldRouteTrace | None,
        second: MotionFieldRouteTrace | None,
        first_frame: int,
        second_frame: int,
        first_distance: float,
        second_distance: float,
        height_difference: float,
        status: str,
        reason: str,
    ) -> TransitionOpportunity:
        first_support = (
            tuple(bool(value) for value in first.support_mask[first_frame])
            if first is not None and first_frame >= 0
            else (False, False)
        )
        second_support = (
            tuple(bool(value) for value in second.support_mask[second_frame])
            if second is not None and second_frame >= 0
            else (False, False)
        )
        return TransitionOpportunity(
            first_line_id=crossing.first_line_id,
            second_line_id=crossing.second_line_id,
            scene_xy=crossing.scene_xy,
            first_frame=first_frame,
            second_frame=second_frame,
            first_distance_m=first_distance,
            second_distance_m=second_distance,
            height_difference_m=height_difference,
            first_support=first_support,
            second_support=second_support,
            status=status,
            reason=reason,
        )

    for crossing in crossings:
        first = by_line.get(crossing.first_line_id)
        second = by_line.get(crossing.second_line_id)
        if first is None or second is None:
            output.append(
                record(
                    crossing,
                    first=first,
                    second=second,
                    first_frame=-1,
                    second_frame=-1,
                    first_distance=0.0,
                    second_distance=0.0,
                    height_difference=0.0,
                    status="rejected",
                    reason="missing_route_trace",
                )
            )
            continue
        point = np.asarray(crossing.scene_xy, dtype=np.float64)
        first_distance = np.linalg.norm(first.root_scene_xy - point, axis=1)
        second_distance = np.linalg.norm(second.root_scene_xy - point, axis=1)
        first_near = np.flatnonzero(first_distance <= search_radius_m)
        second_near = np.flatnonzero(second_distance <= search_radius_m)
        if not len(first_near) or not len(second_near):
            first_frame = int(np.argmin(first_distance))
            second_frame = int(np.argmin(second_distance))
            output.append(
                record(
                    crossing,
                    first=first,
                    second=second,
                    first_frame=first_frame,
                    second_frame=second_frame,
                    first_distance=float(first_distance[first_frame]),
                    second_distance=float(second_distance[second_frame]),
                    height_difference=float(
                        abs(
                            first.surface_height_m[first_frame]
                            - second.surface_height_m[second_frame]
                        )
                    ),
                    status="rejected",
                    reason="no_nearby_frame",
                )
            )
            continue
        supported_first = [
            int(frame)
            for frame in first_near
            if bool(first.support_mask[frame].any())
        ]
        supported_second = [
            int(frame)
            for frame in second_near
            if bool(second.support_mask[frame].any())
        ]
        if not supported_first or not supported_second:
            first_frame = int(first_near[np.argmin(first_distance[first_near])])
            second_frame = int(second_near[np.argmin(second_distance[second_near])])
            output.append(
                record(
                    crossing,
                    first=first,
                    second=second,
                    first_frame=first_frame,
                    second_frame=second_frame,
                    first_distance=float(first_distance[first_frame]),
                    second_distance=float(second_distance[second_frame]),
                    height_difference=float(
                        abs(
                            first.surface_height_m[first_frame]
                            - second.surface_height_m[second_frame]
                        )
                    ),
                    status="rejected",
                    reason="unsupported_near_intersection",
                )
            )
            continue
        pairs = [
            (
                int(first_frame),
                int(second_frame),
                float(
                    abs(
                        first.surface_height_m[first_frame]
                        - second.surface_height_m[second_frame]
                    )
                ),
            )
            for first_frame in supported_first
            for second_frame in supported_second
        ]
        height_pairs = [
            item
            for item in pairs
            if item[2] <= maximum_height_difference_m
        ]
        compatible = [
            item
            for item in height_pairs
            if np.array_equal(
                first.support_mask[item[0]], second.support_mask[item[1]]
            )
        ]
        candidates = compatible or height_pairs or pairs
        first_frame, second_frame, height_difference = min(
            candidates,
            key=lambda item: (
                float(first_distance[item[0]] + second_distance[item[1]]),
                item[2],
                item[0],
                item[1],
            ),
        )
        if compatible:
            status = "admitted"
            reason = "phase_and_height_compatible"
        elif not height_pairs:
            status = "rejected"
            reason = "terrain_height_mismatch"
        else:
            status = "rejected"
            reason = "support_phase_mismatch"
        output.append(
            record(
                crossing,
                first=first,
                second=second,
                first_frame=first_frame,
                second_frame=second_frame,
                first_distance=float(first_distance[first_frame]),
                second_distance=float(second_distance[second_frame]),
                height_difference=height_difference,
                status=status,
                reason=reason,
            )
        )
    return tuple(output)
