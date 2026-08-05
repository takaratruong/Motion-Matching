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
    root_start_world_xy: tuple[float, float]
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
            self,
            "root_start_world_xy",
            _finite_tuple(
                self.root_start_world_xy, 2, "window root start"
            ),
        )
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
            starts = sorted(
                {
                    max(0, selected[0].frame - lead_in)
                    for lead_in in (0, 20, 40, 60)
                }
            )
            stops = sorted(
                {
                    min(frames, selected[-1].frame + tail)
                    for tail in (11, 20, 40)
                }
            )
            for start in starts:
                for stop in stops:
                    if stop - start < 4:
                        continue
                    displacement = roots[stop - 1, :2] - roots[start, :2]
                    progress = float(np.linalg.norm(displacement))
                    if progress < float(minimum_progress_m):
                        continue
                    travel_yaw = math.atan2(
                        float(displacement[1]), float(displacement[0])
                    )
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
                            root_start_world_xy=tuple(roots[start, :2]),
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
    relative = event_xy - np.asarray(
        window.root_start_world_xy, dtype=np.float64
    )
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
        + 2.0 * (window.forward_progress_m - query_forward[-1]) ** 2
        + 2.0 * window.heading_error_rad**2
    )


class PlacementRejected(ContractError):
    """Stable rejection from rigid path placement."""

    def __init__(self, reason: str):
        if not isinstance(reason, str) or not reason:
            raise ContractError("path placement rejection reason is invalid")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class PlacementMetrics:
    covered_start_m: float
    covered_stop_m: float
    heading_error_p95_rad: float
    maximum_lateral_error_m: float
    maximum_stance_error_m: float
    minimum_sole_clearance_m: float

    def __post_init__(self) -> None:
        values = tuple(float(getattr(self, name)) for name in self.__dataclass_fields__)
        if (
            not all(math.isfinite(value) for value in values)
            or values[0] < -1.0e-6
            or values[1] <= values[0]
            or any(value < 0.0 for value in values[2:5])
        ):
            raise ContractError("path placement metrics are invalid")
        for name, value in zip(self.__dataclass_fields__, values):
            object.__setattr__(self, name, value)


def _owned_array(
    value: object, shape: tuple[int, ...], label: str
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.number):
        raise ContractError(f"path placement {label} is invalid")
    owned = np.array(array, copy=True)
    if not np.isfinite(owned).all():
        raise ContractError(f"path placement {label} is invalid")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True)
class PlacedRawWindow:
    window: RawMotionWindow
    yaw_scene_rad: float
    translation_scene_xyz: tuple[float, float, float]
    joint_position: np.ndarray
    root_position_scene: np.ndarray
    root_orientation_scene_wxyz: np.ndarray
    metrics: PlacementMetrics

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window, RawMotionWindow)
            or not math.isfinite(float(self.yaw_scene_rad))
            or not isinstance(self.metrics, PlacementMetrics)
        ):
            raise ContractError("placed raw window metadata is invalid")
        frames = self.window.stop_frame - self.window.start_frame
        object.__setattr__(
            self,
            "translation_scene_xyz",
            _finite_tuple(
                self.translation_scene_xyz, 3, "placement translation"
            ),
        )
        object.__setattr__(
            self,
            "joint_position",
            _owned_array(self.joint_position, (frames, 29), "joints"),
        )
        object.__setattr__(
            self,
            "root_position_scene",
            _owned_array(self.root_position_scene, (frames, 3), "roots"),
        )
        quaternion = _owned_array(
            self.root_orientation_scene_wxyz,
            (frames, 4),
            "orientations",
        )
        if np.any(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0) > 1.0e-4):
            raise ContractError("path placement orientations are invalid")
        object.__setattr__(self, "root_orientation_scene_wxyz", quaternion)
        object.__setattr__(self, "yaw_scene_rad", float(self.yaw_scene_rad))


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def place_raw_window_on_path(
    *,
    window: RawMotionWindow,
    joint_position: object,
    root_position_world: object,
    root_orientation_world_wxyz: object,
    foot_position_world: object,
    sole_position_world: object,
    support_mask: object,
    path_start_scene_xy: object,
    path_heading_scene_xy: object,
    sample_surface: Callable[[np.ndarray], np.ndarray],
    maximum_heading_error_rad: float = math.radians(15.0),
    maximum_lateral_error_m: float = 0.15,
    maximum_stance_error_m: float = 0.03,
    minimum_sole_clearance_m: float = -0.03,
) -> PlacedRawWindow:
    """Rigidly align one raw window to a path and certify actual contacts."""

    if not isinstance(window, RawMotionWindow) or not callable(sample_surface):
        raise ContractError("path placement inputs are invalid")
    joints = np.asarray(joint_position)
    roots = np.asarray(root_position_world, dtype=np.float64)
    quaternion = np.asarray(root_orientation_world_wxyz, dtype=np.float64)
    feet = np.asarray(foot_position_world, dtype=np.float64)
    soles = np.asarray(sole_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    total = len(joints)
    sole_points = soles.shape[2] if soles.ndim == 4 else 0
    start, stop = window.start_frame, window.stop_frame
    heading = np.asarray(path_heading_scene_xy, dtype=np.float64)
    path_start = np.asarray(path_start_scene_xy, dtype=np.float64)
    thresholds = (
        maximum_heading_error_rad,
        maximum_lateral_error_m,
        maximum_stance_error_m,
    )
    if (
        joints.shape != (total, 29)
        or roots.shape != (total, 3)
        or quaternion.shape != (total, 4)
        or feet.shape != (total, 2, 3)
        or soles.shape != (total, 2, sole_points, 3)
        or sole_points < 1
        or support.shape != (total, 2)
        or support.dtype != np.bool_
        or not 0 <= start < stop <= total
        or path_start.shape != (2,)
        or heading.shape != (2,)
        or not all(
            np.isfinite(value).all()
            for value in (joints, roots, quaternion, feet, soles, path_start, heading)
        )
        or np.any(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0) > 1.0e-4)
        or not all(math.isfinite(float(value)) and value > 0.0 for value in thresholds)
        or not math.isfinite(float(minimum_sole_clearance_m))
    ):
        raise ContractError("path placement source profiles are invalid")
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1.0e-6:
        raise ContractError("path placement heading is zero")
    heading /= heading_norm
    source_displacement = roots[stop - 1, :2] - roots[start, :2]
    source_distance = float(np.linalg.norm(source_displacement))
    if source_distance <= 1.0e-6:
        raise PlacementRejected("insufficient-progress")
    source_yaw = math.atan2(
        float(source_displacement[1]), float(source_displacement[0])
    )
    target_yaw = math.atan2(float(heading[1]), float(heading[0]))
    yaw_delta = target_yaw - source_yaw
    cosine, sine = math.cos(yaw_delta), math.sin(yaw_delta)
    rotation = np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    source_origin = roots[start, :2]
    base_translation_xy = path_start - source_origin @ rotation.T

    selection = slice(start, stop)
    base_roots = roots[selection].copy()
    base_feet = feet[selection].copy()
    base_soles = soles[selection].copy()
    base_roots[:, :2] = (
        roots[selection, :2] @ rotation.T + base_translation_xy
    )
    base_feet[..., :2] = (
        feet[selection, ..., :2] @ rotation.T + base_translation_xy
    )
    base_soles[..., :2] = (
        soles[selection, ..., :2] @ rotation.T + base_translation_xy
    )
    placed_support = support[selection]
    lateral_axis = np.array((-heading[1], heading[0]), dtype=np.float64)
    base_relative_root = base_roots[:, :2] - path_start
    base_lateral = base_relative_root @ lateral_axis
    lower_offset = float(-maximum_lateral_error_m - base_lateral.min())
    upper_offset = float(maximum_lateral_error_m - base_lateral.max())
    if lower_offset > upper_offset + 1.0e-9:
        raise PlacementRejected("lateral-path")

    rotated_quaternion = _quaternion_multiply_wxyz(
        np.array(
            (
                math.cos(yaw_delta / 2.0),
                0.0,
                0.0,
                math.sin(yaw_delta / 2.0),
            )
        ),
        quaternion[selection],
    )
    character_yaw = _yaw_wxyz(rotated_quaternion)
    heading_error = np.abs(
        np.arctan2(
            np.sin(character_yaw - target_yaw),
            np.cos(character_yaw - target_yaw),
        )
    )
    heading_p95 = float(np.quantile(heading_error, 0.95))
    if heading_p95 > float(maximum_heading_error_rad):
        raise PlacementRejected("heading")

    step = 0.01
    first_step = int(math.ceil((lower_offset - 1.0e-12) / step))
    last_step = int(math.floor((upper_offset + 1.0e-12) / step))
    offsets = [
        float(index * step)
        for index in range(first_step, last_step + 1)
    ]
    offsets.extend((lower_offset, upper_offset))
    if lower_offset <= 0.0 <= upper_offset:
        offsets.append(0.0)
    lateral_offsets = np.asarray(
        sorted({round(value, 12) for value in offsets}),
        dtype=np.float64,
    )
    foot_offset_xy = (
        lateral_offsets[:, None, None, None]
        * lateral_axis[None, None, None, :]
    )
    sole_offset_xy = (
        lateral_offsets[:, None, None, None, None]
        * lateral_axis[None, None, None, None, :]
    )
    candidate_feet_xy = base_feet[None, ..., :2] + foot_offset_xy
    candidate_soles_xy = base_soles[None, ..., :2] + sole_offset_xy
    try:
        foot_surface = np.asarray(
            sample_surface(candidate_feet_xy), dtype=np.float64
        )
    except PlacementRejected:
        raise
    except Exception as error:
        raise ContractError("path placement terrain sampling failed") from error
    if (
        foot_surface.shape
        != (len(lateral_offsets),) + placed_support.shape
        or not np.isfinite(foot_surface).all()
        or not bool(placed_support.any())
    ):
        raise ContractError("path placement terrain samples are invalid")
    height_residual = (
        foot_surface + 0.035 - base_feet[None, ..., 2]
    )
    translation_z = np.median(
        height_residual[:, placed_support], axis=1
    )
    foot_clearance = (
        base_feet[None, ..., 2]
        + translation_z[:, None, None]
        - foot_surface
        - 0.035
    )
    stance_error = np.max(
        np.abs(foot_clearance[:, placed_support]), axis=1
    )
    try:
        sole_surface = np.asarray(
            sample_surface(candidate_soles_xy), dtype=np.float64
        )
    except Exception as error:
        raise ContractError("path placement sole sampling failed") from error
    if (
        sole_surface.shape
        != (len(lateral_offsets),) + base_soles.shape[:-1]
        or not np.isfinite(sole_surface).all()
    ):
        raise ContractError("path placement sole samples are invalid")
    sole_clearance = (
        base_soles[None, ..., 2]
        + translation_z[:, None, None, None]
        - sole_surface
    )
    minimum_clearance = np.min(
        sole_clearance, axis=tuple(range(1, sole_clearance.ndim))
    )
    stance_valid = stance_error <= float(maximum_stance_error_m)
    if not bool(stance_valid.any()):
        raise PlacementRejected("stance-height")
    fully_valid = stance_valid & (
        minimum_clearance >= float(minimum_sole_clearance_m)
    )
    if not bool(fully_valid.any()):
        raise PlacementRejected("sole-penetration")
    valid_indices = np.flatnonzero(fully_valid)
    chosen = min(
        valid_indices,
        key=lambda index: (
            abs(float(lateral_offsets[index])),
            float(stance_error[index]),
            -float(minimum_clearance[index]),
            float(lateral_offsets[index]),
        ),
    )
    lateral_offset = float(lateral_offsets[chosen])
    selected_translation_z = float(translation_z[chosen])
    translation_xy = (
        base_translation_xy + lateral_offset * lateral_axis
    )
    placed_roots = base_roots.copy()
    placed_feet = base_feet.copy()
    placed_soles = base_soles.copy()
    placed_roots[:, :2] += lateral_offset * lateral_axis
    placed_feet[..., :2] += lateral_offset * lateral_axis
    placed_soles[..., :2] += lateral_offset * lateral_axis
    placed_roots[:, 2] += selected_translation_z
    placed_feet[..., 2] += selected_translation_z
    placed_soles[..., 2] += selected_translation_z
    relative_root = placed_roots[:, :2] - path_start
    along = relative_root @ heading
    lateral = relative_root @ lateral_axis
    maximum_lateral = float(np.max(np.abs(lateral)))
    covered_start = max(0.0, float(along.min()))
    covered_stop = float(along.max())
    if covered_stop <= covered_start:
        raise PlacementRejected("insufficient-progress")
    return PlacedRawWindow(
        window=window,
        yaw_scene_rad=yaw_delta,
        translation_scene_xyz=(
            float(translation_xy[0]),
            float(translation_xy[1]),
            selected_translation_z,
        ),
        joint_position=joints[selection],
        root_position_scene=placed_roots,
        root_orientation_scene_wxyz=rotated_quaternion,
        metrics=PlacementMetrics(
            covered_start_m=covered_start,
            covered_stop_m=covered_stop,
            heading_error_p95_rad=heading_p95,
            maximum_lateral_error_m=maximum_lateral,
            maximum_stance_error_m=float(stance_error[chosen]),
            minimum_sole_clearance_m=float(minimum_clearance[chosen]),
        ),
    )
