"""Label-independent raw motion retrieval for directed terrain paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from .joints import ContractError
from .torch_heading_footprint_path import NominalFootprintPath


def _finite_tuple(value: object, length: int, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ContractError(f"path motion {label} is invalid")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class PathContactSignature:
    foot_order: tuple[int, ...]
    forward_m: tuple[float, ...]
    lateral_m: tuple[float, ...]
    contact_frame: tuple[int, ...]
    height_pattern_m: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.foot_order)
        if (
            count < 1
            or any(foot not in (0, 1) for foot in self.foot_order)
            or len(self.contact_frame) != count
            or any(type(frame) is not int or frame < 1 for frame in self.contact_frame)
        ):
            raise ContractError("path contact signature metadata is invalid")
        for name in ("forward_m", "lateral_m", "height_pattern_m"):
            object.__setattr__(
                self, name, _finite_tuple(getattr(self, name), count, name)
            )


@dataclass(frozen=True, order=True)
class RawContactEvent:
    frame: int
    foot: int
    position_world_xy: tuple[float, float]
    surface_height_m: float

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not int
            or self.frame < 0
            or self.foot not in (0, 1)
            or not math.isfinite(float(self.surface_height_m))
        ):
            raise ContractError("raw contact event is invalid")
        object.__setattr__(
            self,
            "position_world_xy",
            _finite_tuple(self.position_world_xy, 2, "event position"),
        )
        object.__setattr__(
            self, "surface_height_m", float(self.surface_height_m)
        )


@dataclass(frozen=True, order=True)
class RawMotionWindow:
    source_clip: str
    start_frame: int
    stop_frame: int
    events: tuple[RawContactEvent, ...]
    forward_progress_m: float
    heading_error_rad: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_clip, str)
            or not self.source_clip
            or type(self.start_frame) is not int
            or type(self.stop_frame) is not int
            or not 0 <= self.start_frame < self.stop_frame
            or not isinstance(self.events, tuple)
            or len(self.events) < 2
            or any(not isinstance(event, RawContactEvent) for event in self.events)
            or tuple(sorted(self.events)) != self.events
            or not math.isfinite(float(self.forward_progress_m))
            or self.forward_progress_m <= 0.0
            or not math.isfinite(float(self.heading_error_rad))
            or self.heading_error_rad < 0.0
        ):
            raise ContractError("raw motion window is invalid")
        object.__setattr__(
            self, "forward_progress_m", float(self.forward_progress_m)
        )
        object.__setattr__(
            self, "heading_error_rad", float(self.heading_error_rad)
        )


def path_contact_signature(
    *,
    path: NominalFootprintPath,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
) -> PathContactSignature:
    """Turn approximate path footprints into a relative-height search query."""

    if not isinstance(path, NominalFootprintPath) or not callable(sample_surface):
        raise ContractError("path contact signature inputs are invalid")
    if not path.footprints:
        raise ContractError("path contact signature has no footprints")
    points = torch.stack(
        [footprint.center_scene_xy for footprint in path.footprints]
    )
    try:
        height = sample_surface(points)
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("path contact terrain sampling failed") from error
    if (
        not isinstance(height, torch.Tensor)
        or tuple(height.shape) != (len(path.footprints),)
        or height.device != points.device
        or height.dtype != points.dtype
        or not torch.isfinite(height).all()
    ):
        raise ContractError("path contact terrain sampler output is invalid")
    relative = height - height[0]
    return PathContactSignature(
        foot_order=tuple(item.foot for item in path.footprints),
        forward_m=tuple(
            float(item.center_heading_xy[0].item())
            for item in path.footprints
        ),
        lateral_m=tuple(
            float(item.center_heading_xy[1].item())
            for item in path.footprints
        ),
        contact_frame=tuple(item.contact_frame for item in path.footprints),
        height_pattern_m=tuple(float(value) for value in relative.tolist()),
    )


def _yaw_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.unwrap(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def extract_raw_motion_windows(
    *,
    source_clip: str,
    root_position_world: object,
    root_orientation_world_wxyz: object,
    foot_position_world: object,
    support_mask: object,
    foot_surface_height_m: object,
    minimum_events: int = 4,
    maximum_events: int = 12,
    minimum_progress_m: float = 0.40,
) -> tuple[RawMotionWindow, ...]:
    """Extract alternating variable-length contact windows from one raw clip."""

    roots = np.asarray(root_position_world, dtype=np.float64)
    quaternion = np.asarray(root_orientation_world_wxyz, dtype=np.float64)
    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    surface = np.asarray(foot_surface_height_m, dtype=np.float64)
    frames = len(roots)
    if (
        not isinstance(source_clip, str)
        or not source_clip
        or roots.shape != (frames, 3)
        or quaternion.shape != (frames, 4)
        or feet.shape != (frames, 2, 3)
        or support.shape != (frames, 2)
        or support.dtype != np.bool_
        or surface.shape != (frames, 2)
        or frames < 4
        or type(minimum_events) is not int
        or type(maximum_events) is not int
        or not 2 <= minimum_events <= maximum_events
        or maximum_events > 32
        or not math.isfinite(float(minimum_progress_m))
        or minimum_progress_m <= 0.0
        or not all(
            np.isfinite(value).all()
            for value in (roots, quaternion, feet, surface)
        )
        or np.any(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0) > 1.0e-4)
    ):
        raise ContractError("raw motion source profiles are invalid")

    raw: list[RawContactEvent] = []
    for foot in (0, 1):
        if bool(support[0, foot]):
            raw.append(
                RawContactEvent(
                    0,
                    foot,
                    tuple(feet[0, foot, :2]),
                    float(surface[0, foot]),
                )
            )
    onset = (~support[:-1]) & support[1:]
    for frame, foot in np.argwhere(onset):
        index = int(frame) + 1
        raw.append(
            RawContactEvent(
                index,
                int(foot),
                tuple(feet[index, foot, :2]),
                float(surface[index, foot]),
            )
        )
    raw.sort()

    events: list[RawContactEvent] = []
    latest: dict[int, RawContactEvent] = {}
    for event in raw:
        previous = latest.get(event.foot)
        if previous is not None:
            distance = float(
                np.linalg.norm(
                    np.asarray(event.position_world_xy)
                    - np.asarray(previous.position_world_xy)
                )
            )
            height = abs(event.surface_height_m - previous.surface_height_m)
            if distance < 0.07 and height < 0.04:
                continue
        events.append(event)
        latest[event.foot] = event

    yaw = _yaw_wxyz(quaternion)
    output: list[RawMotionWindow] = []
    for count in range(minimum_events, maximum_events + 1):
        for begin in range(0, len(events) - count + 1):
            selected = tuple(events[begin : begin + count])
            if any(
                left.foot == right.foot
                for left, right in zip(selected[:-1], selected[1:])
            ):
                continue
            start = selected[0].frame
            stop = min(frames, selected[-1].frame + 11)
            if stop - start < 4:
                continue
            displacement = roots[stop - 1, :2] - roots[start, :2]
            progress = float(np.linalg.norm(displacement))
            if progress < float(minimum_progress_m):
                continue
            travel_yaw = math.atan2(float(displacement[1]), float(displacement[0]))
            error = np.abs(
                np.arctan2(
                    np.sin(yaw[start:stop] - travel_yaw),
                    np.cos(yaw[start:stop] - travel_yaw),
                )
            )
            output.append(
                RawMotionWindow(
                    source_clip=source_clip,
                    start_frame=start,
                    stop_frame=stop,
                    events=selected,
                    forward_progress_m=progress,
                    heading_error_rad=float(np.quantile(error, 0.95)),
                )
            )
    return tuple(sorted(output))


def contact_signature_cost(
    query: PathContactSignature, window: RawMotionWindow
) -> float:
    """Return a coarse, label-independent path/contact mismatch score."""

    if not isinstance(query, PathContactSignature) or not isinstance(
        window, RawMotionWindow
    ):
        raise ContractError("contact signature cost inputs are invalid")
    if len(query.foot_order) != len(window.events):
        return math.inf
    event_xy = np.asarray(
        [event.position_world_xy for event in window.events], dtype=np.float64
    )
    displacement = event_xy[-1] - event_xy[0]
    norm = float(np.linalg.norm(displacement))
    if norm <= 1.0e-6:
        return math.inf
    forward = displacement / norm
    lateral = np.array((-forward[1], forward[0]), dtype=np.float64)
    relative = event_xy - event_xy[0]
    source_forward = relative @ forward
    source_lateral = relative @ lateral
    source_height = np.asarray(
        [event.surface_height_m for event in window.events], dtype=np.float64
    )
    source_height -= source_height[0]
    source_time = np.asarray(
        [event.frame for event in window.events], dtype=np.float64
    )
    source_time -= source_time[0]
    query_forward = np.asarray(query.forward_m, dtype=np.float64)
    query_forward -= query_forward[0]
    query_lateral = np.asarray(query.lateral_m, dtype=np.float64)
    query_height = np.asarray(query.height_pattern_m, dtype=np.float64)
    query_time = np.asarray(query.contact_frame, dtype=np.float64)
    query_time -= query_time[0]
    foot_error = sum(
        int(left != right.foot)
        for left, right in zip(query.foot_order, window.events)
    )
    return float(
        20.0 * foot_error
        + 60.0 * np.mean(np.square(source_height - query_height))
        + 2.0 * np.mean(np.square(source_forward - query_forward))
        + 2.0 * np.mean(np.square(source_lateral - query_lateral))
        + 0.002 * np.mean(np.square(source_time - query_time))
        + 2.0 * window.heading_error_rad**2
    )
