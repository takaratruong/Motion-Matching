"""Pure contracts for parallel path grids and kinematic playlists."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .joints import ContractError


_CONNECTOR_KEYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
)


def _lane_id(value: float) -> str:
    if abs(value) <= 1.0e-12:
        return "lane-zero-0p0"
    prefix = "neg" if value < 0.0 else "pos"
    magnitude = f"{abs(value):.1f}".replace(".", "p")
    return f"lane-{prefix}-{magnitude}"


@dataclass(frozen=True)
class HorizontalGridLane:
    index: int
    lane_id: str
    center_y_m: float
    start_scene_xy: tuple[float, float]
    stop_scene_xy: tuple[float, float]

    def __post_init__(self) -> None:
        values = (
            self.center_y_m,
            *self.start_scene_xy,
            *self.stop_scene_xy,
        )
        if (
            type(self.index) is not int
            or self.index < 0
            or not isinstance(self.lane_id, str)
            or not self.lane_id
            or len(self.start_scene_xy) != 2
            or len(self.stop_scene_xy) != 2
            or not all(math.isfinite(float(value)) for value in values)
            or self.stop_scene_xy[0] <= self.start_scene_xy[0]
            or abs(self.start_scene_xy[1] - self.center_y_m) > 1.0e-9
            or abs(self.stop_scene_xy[1] - self.center_y_m) > 1.0e-9
        ):
            raise ContractError("horizontal grid lane is invalid")


def horizontal_grid_lanes(
    *,
    start_x: float,
    stop_x: float,
    minimum_y: float,
    maximum_y: float,
    spacing_m: float,
) -> tuple[HorizontalGridLane, ...]:
    values = (start_x, stop_x, minimum_y, maximum_y, spacing_m)
    if (
        not all(math.isfinite(float(value)) for value in values)
        or stop_x <= start_x
        or maximum_y < minimum_y
        or spacing_m <= 0.0
    ):
        raise ContractError("horizontal grid bounds are invalid")
    interval_count = int(round((maximum_y - minimum_y) / spacing_m))
    if abs(
        minimum_y + interval_count * spacing_m - maximum_y
    ) > 1.0e-9:
        raise ContractError("horizontal grid spacing misses upper bound")
    output = []
    for index in range(interval_count + 1):
        center = float(minimum_y + index * spacing_m)
        if abs(center) <= 1.0e-12:
            center = 0.0
        output.append(
            HorizontalGridLane(
                index=index,
                lane_id=_lane_id(center),
                center_y_m=center,
                start_scene_xy=(float(start_x), center),
                stop_scene_xy=(float(stop_x), center),
            )
        )
    return tuple(output)


def classify_lane(
    *,
    path_length_m: float,
    covered_intervals: object,
    has_certified_placement: bool,
) -> str:
    if (
        not math.isfinite(float(path_length_m))
        or path_length_m <= 0.0
        or type(has_certified_placement) is not bool
        or not isinstance(covered_intervals, (list, tuple))
    ):
        raise ContractError("grid lane classification input is invalid")
    intervals = tuple(
        (float(start), float(stop)) for start, stop in covered_intervals
    )
    if any(
        not math.isfinite(start)
        or not math.isfinite(stop)
        or not 0.0 <= start < stop <= path_length_m + 1.0e-6
        for start, stop in intervals
    ):
        raise ContractError("grid lane coverage interval is invalid")
    if not has_certified_placement:
        if intervals:
            raise ContractError("infeasible lane cannot have coverage")
        return "infeasible"
    if not intervals:
        raise ContractError("certified lane must have coverage")
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1] + 1.0e-6:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    if (
        merged[0][0] <= 1.0e-6
        and merged[-1][1] >= path_length_m - 1.0e-6
        and all(
            left[1] >= right[0] - 1.0e-6
            for left, right in zip(merged[:-1], merged[1:])
        )
    ):
        return "full"
    return "partial"


def _validated_connector(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or set(value) != set(_CONNECTOR_KEYS):
        raise ContractError("grid playlist connector inventory is invalid")
    arrays = {
        name: np.asarray(value[name], dtype=np.float64)
        for name in _CONNECTOR_KEYS
    }
    frames = len(arrays["joint_position"])
    if (
        frames < 1
        or arrays["joint_position"].shape != (frames, 29)
        or arrays["root_position_world"].shape != (frames, 3)
        or arrays["root_orientation_world_wxyz"].shape != (frames, 4)
        or not all(np.isfinite(array).all() for array in arrays.values())
        or np.any(
            np.abs(
                np.linalg.norm(
                    arrays["root_orientation_world_wxyz"], axis=1
                )
                - 1.0
            )
            > 1.0e-4
        )
    ):
        raise ContractError("grid playlist connector arrays are invalid")
    return arrays


def build_grid_playlist(
    accepted_lanes: object, *, hold_frames: int = 20
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if (
        not isinstance(accepted_lanes, (list, tuple))
        or not accepted_lanes
        or type(hold_frames) is not int
        or hold_frames < 0
    ):
        raise ContractError("grid playlist input is invalid")
    chunks = {name: [] for name in _CONNECTOR_KEYS}
    segments = []
    teleport_boundaries = []
    cursor = 0
    previous_index = -1
    for item_index, item in enumerate(accepted_lanes):
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], HorizontalGridLane)
            or item[1] not in ("full", "partial")
            or item[0].index <= previous_index
        ):
            raise ContractError("grid playlist lane entry is invalid")
        lane, classification, connector_value = item
        connector = _validated_connector(connector_value)
        frames = len(connector["joint_position"])
        if item_index > 0:
            teleport_boundaries.append(cursor)
        for name, array in connector.items():
            chunks[name].extend(
                (
                    np.repeat(array[0:1], hold_frames, axis=0),
                    np.array(array, copy=True),
                    np.repeat(array[-1:], hold_frames, axis=0),
                )
            )
        motion_start = cursor + hold_frames
        motion_stop = motion_start + frames
        segment_stop = motion_stop + hold_frames
        segments.append(
            {
                "lane_id": lane.lane_id,
                "lane_index": lane.index,
                "center_y_m": lane.center_y_m,
                "classification": classification,
                "segment_frames": [cursor, segment_stop],
                "motion_frames": [motion_start, motion_stop],
            }
        )
        cursor = segment_stop
        previous_index = lane.index
    arrays = {
        name: np.concatenate(value, axis=0)
        for name, value in chunks.items()
    }
    metadata = {
        "schema": "g1-horizontal-grid-playlist/v1",
        "hold_frames": hold_frames,
        "frame_count": cursor,
        "segments": segments,
        "teleport_boundaries": teleport_boundaries,
    }
    return arrays, metadata
