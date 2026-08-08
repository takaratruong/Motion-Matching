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


def _path_id(index: int, value: float) -> str:
    sign = "zero" if abs(value) <= 1.0e-12 else (
        "neg" if value < 0.0 else "pos"
    )
    magnitude = f"{abs(value):.6f}".rstrip("0").rstrip(".")
    if not magnitude:
        magnitude = "0"
    return f"path-{index:03d}-{sign}-{magnitude.replace('.', 'p')}"


@dataclass(frozen=True)
class ParallelPath:
    """One member of a heading-local parallel path grid."""

    index: int
    path_id: str
    lateral_offset_m: float
    start_scene_xy: tuple[float, float]
    stop_scene_xy: tuple[float, float]

    def __post_init__(self) -> None:
        start = np.asarray(self.start_scene_xy, dtype=np.float64)
        stop = np.asarray(self.stop_scene_xy, dtype=np.float64)
        if (
            type(self.index) is not int
            or self.index < 0
            or not isinstance(self.path_id, str)
            or not self.path_id
            or start.shape != (2,)
            or stop.shape != (2,)
            or not math.isfinite(float(self.lateral_offset_m))
            or not np.isfinite(start).all()
            or not np.isfinite(stop).all()
            or float(np.linalg.norm(stop - start)) <= 0.0
        ):
            raise ContractError("parallel path is invalid")


@dataclass(frozen=True)
class StaircasePathContract:
    """Terrain-derived admission contract for one parallel path."""

    path: ParallelPath
    classification: str
    ordered_surface_heights_m: tuple[float, ...]
    elevated_intervals_m: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        allowed = (
            "staircase_intersecting",
            "flat_only_excluded",
            "missing_approach",
            "missing_opposite_exit",
        )
        heights = np.asarray(
            self.ordered_surface_heights_m, dtype=np.float64
        )
        intervals = np.asarray(self.elevated_intervals_m, dtype=np.float64)
        if intervals.size == 0:
            intervals = intervals.reshape(0, 2)
        if (
            not isinstance(self.path, ParallelPath)
            or self.classification not in allowed
            or heights.ndim != 1
            or not len(heights)
            or not np.isfinite(heights).all()
            or intervals.ndim != 2
            or intervals.shape[1:] != (2,)
            or not np.isfinite(intervals).all()
            or np.any(intervals[:, 0] < 0.0)
            or np.any(intervals[:, 1] < intervals[:, 0])
            or (
                self.classification == "flat_only_excluded"
                and len(intervals)
            )
            or (
                self.classification != "flat_only_excluded"
                and not len(intervals)
            )
        ):
            raise ContractError("staircase path contract is invalid")


def parallel_path_grid(
    *,
    center_start_scene_xy: tuple[float, float],
    heading_scene_xy: tuple[float, float],
    path_length_m: float,
    minimum_lateral_offset_m: float,
    maximum_lateral_offset_m: float,
    spacing_m: float,
) -> tuple[ParallelPath, ...]:
    """Generate parallel paths in the frame defined by a runtime heading."""

    origin = np.asarray(center_start_scene_xy, dtype=np.float64)
    heading = np.asarray(heading_scene_xy, dtype=np.float64)
    scalars = (
        path_length_m,
        minimum_lateral_offset_m,
        maximum_lateral_offset_m,
        spacing_m,
    )
    heading_norm = float(np.linalg.norm(heading))
    if (
        origin.shape != (2,)
        or heading.shape != (2,)
        or not np.isfinite(origin).all()
        or not np.isfinite(heading).all()
        or not all(math.isfinite(float(value)) for value in scalars)
        or heading_norm <= 0.0
        or path_length_m <= 0.0
        or maximum_lateral_offset_m < minimum_lateral_offset_m
        or spacing_m <= 0.0
    ):
        raise ContractError("parallel path grid bounds are invalid")
    interval_count = int(
        round(
            (maximum_lateral_offset_m - minimum_lateral_offset_m)
            / spacing_m
        )
    )
    if abs(
        minimum_lateral_offset_m
        + interval_count * spacing_m
        - maximum_lateral_offset_m
    ) > 1.0e-9:
        raise ContractError("parallel path grid spacing misses upper bound")
    forward = heading / heading_norm
    lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
    output = []
    for index in range(interval_count + 1):
        offset = float(minimum_lateral_offset_m + index * spacing_m)
        if abs(offset) <= 1.0e-12:
            offset = 0.0
        start = origin + offset * lateral
        stop = start + float(path_length_m) * forward
        output.append(
            ParallelPath(
                index=index,
                path_id=_path_id(index, offset),
                lateral_offset_m=offset,
                start_scene_xy=tuple(float(value) for value in start),
                stop_scene_xy=tuple(float(value) for value in stop),
            )
        )
    return tuple(output)


def staircase_parallel_path_grid(
    *,
    center_start_scene_xy: tuple[float, float],
    heading_scene_xy: tuple[float, float],
    path_length_m: float,
    elevated_scene_xy: object,
    spacing_m: float,
) -> tuple[ParallelPath, ...]:
    """Center a parallel lattice strictly inside an elevated terrain band."""

    points = np.asarray(elevated_scene_xy, dtype=np.float64)
    heading = np.asarray(heading_scene_xy, dtype=np.float64)
    origin = np.asarray(center_start_scene_xy, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or not np.isfinite(points).all()
        or heading.shape != (2,)
        or origin.shape != (2,)
        or not np.isfinite(heading).all()
        or not np.isfinite(origin).all()
        or not math.isfinite(float(path_length_m))
        or float(path_length_m) <= 0.0
        or not math.isfinite(float(spacing_m))
        or float(spacing_m) <= 0.0
    ):
        raise ContractError("staircase terrain footprint is invalid")
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 0.0:
        raise ContractError("staircase terrain footprint is invalid")
    forward = heading / heading_norm
    lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
    projected = (points - origin) @ lateral
    lower = float(projected.min())
    upper = float(projected.max())
    count = max(
        1,
        int(
            math.floor(
                (upper - lower) / float(spacing_m) + 1.0e-9
            )
        ),
    )
    first = (
        0.5 * (lower + upper)
        - 0.5 * (count - 1) * float(spacing_m)
    )
    return parallel_path_grid(
        center_start_scene_xy=tuple(float(value) for value in origin),
        heading_scene_xy=tuple(float(value) for value in forward),
        path_length_m=float(path_length_m),
        minimum_lateral_offset_m=first,
        maximum_lateral_offset_m=(
            first + (count - 1) * float(spacing_m)
        ),
        spacing_m=float(spacing_m),
    )


def classify_staircase_path(
    *,
    path: ParallelPath,
    sample_surface,
    sample_spacing_m: float = 0.02,
    elevated_threshold_m: float = 0.05,
    minimum_ground_run_m: float = 0.15,
    minimum_elevated_run_m: float = 0.08,
) -> StaircasePathContract:
    """Classify a finite path from its sampled ground/elevated profile."""

    values = (
        sample_spacing_m,
        elevated_threshold_m,
        minimum_ground_run_m,
        minimum_elevated_run_m,
    )
    if (
        not isinstance(path, ParallelPath)
        or not callable(sample_surface)
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in values
        )
    ):
        raise ContractError("staircase path classification input is invalid")
    start = np.asarray(path.start_scene_xy, dtype=np.float64)
    stop = np.asarray(path.stop_scene_xy, dtype=np.float64)
    displacement = stop - start
    length = float(np.linalg.norm(displacement))
    sample_count = int(math.ceil(length / sample_spacing_m)) + 1
    distance = np.linspace(0.0, length, sample_count)
    points = start + distance[:, None] * displacement[None, :] / length
    height = np.asarray(sample_surface(points), dtype=np.float64)
    if height.shape != (sample_count,) or not np.isfinite(height).all():
        raise ContractError("staircase surface profile is invalid")

    ground = float(min(height[0], height[-1]))
    elevated = height > ground + float(elevated_threshold_m)
    ordered = [float(height[0])]
    for value in height[1:]:
        value = float(value)
        if abs(value - ordered[-1]) > float(elevated_threshold_m):
            ordered.append(value)

    padded = np.pad(elevated.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    intervals = tuple(
        (float(distance[begin]), float(distance[end - 1]))
        for begin, end in zip(starts, stops)
    )
    classification = "staircase_intersecting"
    if (
        not intervals
        or max(stop_m - start_m for start_m, stop_m in intervals)
        < float(minimum_elevated_run_m)
    ):
        classification = "flat_only_excluded"
        intervals = ()
    elif intervals[0][0] < float(minimum_ground_run_m):
        classification = "missing_approach"
    elif length - intervals[-1][1] < float(minimum_ground_run_m):
        classification = "missing_opposite_exit"
    return StaircasePathContract(
        path=path,
        classification=classification,
        ordered_surface_heights_m=tuple(ordered),
        elevated_intervals_m=intervals,
    )


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


def build_parallel_grid_playlist(
    accepted_paths: object, *, hold_frames: int = 20
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Package validated arbitrary-heading paths for sequential inspection."""

    if (
        not isinstance(accepted_paths, (list, tuple))
        or not accepted_paths
        or type(hold_frames) is not int
        or hold_frames < 0
    ):
        raise ContractError("parallel grid playlist input is invalid")
    chunks = {name: [] for name in _CONNECTOR_KEYS}
    segments = []
    teleport_boundaries = []
    cursor = 0
    previous_index = -1
    for item_index, item in enumerate(accepted_paths):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], ParallelPath)
            or item[0].index <= previous_index
        ):
            raise ContractError("parallel grid playlist entry is invalid")
        path, connector_value = item
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
                "path_id": path.path_id,
                "path_index": path.index,
                "lateral_offset_m": path.lateral_offset_m,
                "start_scene_xy": list(path.start_scene_xy),
                "stop_scene_xy": list(path.stop_scene_xy),
                "segment_frames": [cursor, segment_stop],
                "motion_frames": [motion_start, motion_stop],
            }
        )
        cursor = segment_stop
        previous_index = path.index
    arrays = {
        name: np.concatenate(value, axis=0)
        for name, value in chunks.items()
    }
    metadata = {
        "schema": "g1-parallel-path-playlist/v1",
        "hold_frames": hold_frames,
        "frame_count": cursor,
        "segments": segments,
        "teleport_boundaries": teleport_boundaries,
    }
    return arrays, metadata
