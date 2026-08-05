"""Pure terrain-phase contracts for straight path traversal."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .joints import ContractError


_PHASE_KINDS = ("mount", "interior", "dismount")


def _pair(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ContractError(f"terrain phase {label} is invalid")
    return array


@dataclass(frozen=True)
class TerrainPathPhase:
    kind: str
    start_m: float
    stop_m: float
    start_scene_xy: tuple[float, float]
    stop_scene_xy: tuple[float, float]
    mean_support_height_m: float
    normalized_support_height_m: tuple[float, float]

    def __post_init__(self) -> None:
        start = _pair(self.start_scene_xy, "start")
        stop = _pair(self.stop_scene_xy, "stop")
        normalized = _pair(
            self.normalized_support_height_m, "normalized support"
        )
        scalars = (
            self.start_m,
            self.stop_m,
            self.mean_support_height_m,
        )
        if (
            self.kind not in _PHASE_KINDS
            or not all(math.isfinite(float(value)) for value in scalars)
            or self.start_m < 0.0
            or self.stop_m <= self.start_m
            or float(np.linalg.norm(stop - start)) <= 1.0e-9
            or abs(float(normalized.mean())) > 1.0e-8
        ):
            raise ContractError("terrain path phase is invalid")
        object.__setattr__(
            self, "start_scene_xy", tuple(float(value) for value in start)
        )
        object.__setattr__(
            self, "stop_scene_xy", tuple(float(value) for value in stop)
        )
        object.__setattr__(
            self,
            "normalized_support_height_m",
            tuple(float(value) for value in normalized),
        )
        object.__setattr__(
            self, "mean_support_height_m", float(self.mean_support_height_m)
        )


def _components(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    padded = np.pad(np.asarray(mask, dtype=np.bool_), (1, 1))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple(
        (int(start), int(stop - 1))
        for start, stop in zip(changes[::2], changes[1::2])
    )


def _phase(
    *,
    kind: str,
    start_m: float,
    stop_m: float,
    path_start: np.ndarray,
    heading: np.ndarray,
    support_height: np.ndarray,
) -> TerrainPathPhase:
    mean = float(np.mean(support_height))
    return TerrainPathPhase(
        kind=kind,
        start_m=float(start_m),
        stop_m=float(stop_m),
        start_scene_xy=tuple(path_start + start_m * heading),
        stop_scene_xy=tuple(path_start + stop_m * heading),
        mean_support_height_m=mean,
        normalized_support_height_m=tuple(support_height - mean),
    )


def segment_path_surface(
    *,
    path_start_scene_xy: object,
    path_stop_scene_xy: object,
    step_width_m: float,
    sample_surface: Callable[[np.ndarray], np.ndarray],
    sample_spacing_m: float = 0.01,
    ground_tolerance_m: float = 0.03,
    stable_tolerance_m: float = 0.03,
    boundary_context_m: float = 0.40,
) -> tuple[TerrainPathPhase, ...]:
    """Partition a ground/object/ground path into reusable motion phases."""

    start = _pair(path_start_scene_xy, "path start")
    stop = _pair(path_stop_scene_xy, "path stop")
    values = (
        step_width_m,
        sample_spacing_m,
        ground_tolerance_m,
        stable_tolerance_m,
        boundary_context_m,
    )
    if (
        not callable(sample_surface)
        or not all(math.isfinite(float(value)) and value > 0.0 for value in values)
    ):
        raise ContractError("terrain phase options are invalid")
    delta = stop - start
    length = float(np.linalg.norm(delta))
    if length <= 2.0 * sample_spacing_m:
        raise ContractError("terrain phase path is too short")
    heading = delta / length
    lateral = np.array((-heading[1], heading[0]), dtype=np.float64)
    sample_count = int(math.ceil(length / sample_spacing_m)) + 1
    progress = np.linspace(0.0, length, sample_count)
    centers = start[None, :] + progress[:, None] * heading[None, :]
    feet = np.stack(
        (
            centers + 0.5 * step_width_m * lateral,
            centers - 0.5 * step_width_m * lateral,
        ),
        axis=1,
    )
    try:
        height = np.asarray(sample_surface(feet), dtype=np.float64)
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("terrain phase sampling failed") from error
    if height.shape != (sample_count, 2) or not np.isfinite(height).all():
        raise ContractError("terrain phase samples are invalid")

    endpoint_height = 0.5 * (height[0] + height[-1])
    if np.max(np.abs(height[0] - height[-1])) > ground_tolerance_m:
        raise ContractError("terrain phase endpoint grounds differ")
    elevation = np.max(np.abs(height - endpoint_height[None, :]), axis=1)
    elevated_components = _components(elevation > ground_tolerance_m)
    if not elevated_components:
        support = np.median(height, axis=0)
        return (
            _phase(
                kind="interior",
                start_m=0.0,
                stop_m=length,
                path_start=start,
                heading=heading,
                support_height=support,
            ),
        )

    elevated_start, elevated_stop = max(
        elevated_components, key=lambda bounds: bounds[1] - bounds[0]
    )
    component_height = height[elevated_start : elevated_stop + 1]
    trim = max(1, len(component_height) // 4)
    core = (
        component_height[trim:-trim]
        if len(component_height) > 2 * trim
        else component_height
    )
    support = np.median(core, axis=0)
    stable = np.max(
        np.abs(height - support[None, :]), axis=1
    ) <= stable_tolerance_m
    stable[:elevated_start] = False
    stable[elevated_stop + 1 :] = False
    stable_components = _components(stable)
    if not stable_components:
        raise ContractError("terrain phase has no stable interior")
    stable_start, stable_stop = max(
        stable_components, key=lambda bounds: bounds[1] - bounds[0]
    )
    interior_start = float(progress[stable_start])
    interior_stop = float(progress[stable_stop])
    if interior_stop - interior_start <= sample_spacing_m:
        raise ContractError("terrain phase interior is too short")

    mount_stop = min(
        length, interior_start + float(boundary_context_m)
    )
    dismount_start = max(
        0.0, interior_stop - float(boundary_context_m)
    )
    return (
        _phase(
            kind="mount",
            start_m=0.0,
            stop_m=mount_stop,
            path_start=start,
            heading=heading,
            support_height=support,
        ),
        _phase(
            kind="interior",
            start_m=interior_start,
            stop_m=interior_stop,
            path_start=start,
            heading=heading,
            support_height=support,
        ),
        _phase(
            kind="dismount",
            start_m=dismount_start,
            stop_m=length,
            path_start=start,
            heading=heading,
            support_height=support,
        ),
    )


def classify_phase_coverage(
    *, required_kinds: object, certified_kinds: object
) -> str:
    required = tuple(required_kinds) if isinstance(
        required_kinds, (list, tuple)
    ) else ()
    certified = tuple(certified_kinds) if isinstance(
        certified_kinds, (list, tuple)
    ) else ()
    if (
        not required
        or len(set(required)) != len(required)
        or any(kind not in _PHASE_KINDS for kind in required)
        or len(set(certified)) != len(certified)
        or any(kind not in required for kind in certified)
    ):
        raise ContractError("terrain phase coverage input is invalid")
    if set(certified) == set(required):
        return "full"
    if certified:
        return "partial"
    return "infeasible"
