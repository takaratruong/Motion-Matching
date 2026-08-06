"""Ordered terrain-level contracts for directed path motion chains."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .joints import ContractError


class OrderedLevelRejected(ContractError):
    """Stable rejection from ordered terrain-level matching."""

    def __init__(self, reason: str):
        if not isinstance(reason, str) or not reason:
            raise ContractError("ordered level rejection reason is invalid")
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class RequiredTerrainLevel:
    index: int
    height_m: float
    start_m: float
    stop_m: float
    requires_double_support: bool = False

    def __post_init__(self) -> None:
        values = (self.height_m, self.start_m, self.stop_m)
        if (
            type(self.index) is not int
            or self.index < 0
            or not all(math.isfinite(float(value)) for value in values)
            or self.start_m < 0.0
            or self.stop_m <= self.start_m
            or type(self.requires_double_support) is not bool
        ):
            raise ContractError("required terrain level is invalid")


@dataclass(frozen=True)
class OrderedTerrainLevelContract:
    path_start_scene_xy: tuple[float, float]
    path_heading_scene_xy: tuple[float, float]
    path_length_m: float
    levels: tuple[RequiredTerrainLevel, ...]

    def __post_init__(self) -> None:
        start = np.asarray(self.path_start_scene_xy, dtype=np.float64)
        heading = np.asarray(self.path_heading_scene_xy, dtype=np.float64)
        if (
            start.shape != (2,)
            or heading.shape != (2,)
            or not np.isfinite(start).all()
            or not np.isfinite(heading).all()
            or abs(float(np.linalg.norm(heading)) - 1.0) > 1.0e-6
            or not math.isfinite(float(self.path_length_m))
            or self.path_length_m <= 0.0
            or not isinstance(self.levels, tuple)
            or len(self.levels) < 2
            or any(
                not isinstance(level, RequiredTerrainLevel)
                or level.index != index
                for index, level in enumerate(self.levels)
            )
            or not self.levels[-1].requires_double_support
        ):
            raise ContractError("ordered terrain level contract is invalid")


def derive_ordered_terrain_levels(
    *,
    path_start_scene_xy: object,
    path_stop_scene_xy: object,
    sample_surface: Callable[[np.ndarray], object],
    sample_spacing_m: float = 0.01,
    height_tolerance_m: float = 0.03,
) -> OrderedTerrainLevelContract:
    """Derive the ordered centerline terrain levels along one path."""

    start = np.asarray(path_start_scene_xy, dtype=np.float64)
    stop = np.asarray(path_stop_scene_xy, dtype=np.float64)
    if (
        start.shape != (2,)
        or stop.shape != (2,)
        or not np.isfinite(start).all()
        or not np.isfinite(stop).all()
        or not callable(sample_surface)
        or not math.isfinite(float(sample_spacing_m))
        or sample_spacing_m <= 0.0
        or not math.isfinite(float(height_tolerance_m))
        or height_tolerance_m <= 0.0
    ):
        raise ContractError("ordered terrain level inputs are invalid")
    displacement = stop - start
    length = float(np.linalg.norm(displacement))
    if length <= 2.0 * sample_spacing_m:
        raise ContractError("ordered terrain level path is too short")
    heading = displacement / length
    sample_count = int(math.ceil(length / sample_spacing_m)) + 1
    progress = np.linspace(0.0, length, sample_count)
    points = start[None, :] + progress[:, None] * heading[None, :]
    try:
        height = np.asarray(sample_surface(points), dtype=np.float64)
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("ordered terrain sampling failed") from error
    if height.shape != (sample_count,) or not np.isfinite(height).all():
        raise ContractError("ordered terrain samples are invalid")

    runs: list[tuple[int, int]] = []
    run_start = 0
    reference = float(height[0])
    for index in range(1, sample_count):
        if abs(float(height[index]) - reference) <= height_tolerance_m:
            reference = float(np.median(height[run_start : index + 1]))
            continue
        runs.append((run_start, index))
        run_start = index
        reference = float(height[index])
    runs.append((run_start, sample_count))
    runs = [
        bounds for bounds in runs if bounds[1] - bounds[0] >= 2
    ]
    if len(runs) < 2:
        raise ContractError("ordered terrain path has insufficient levels")

    levels = []
    for level_index, (first, stop_index) in enumerate(runs):
        stop_m = (
            length
            if stop_index == sample_count
            else float(progress[stop_index])
        )
        levels.append(
            RequiredTerrainLevel(
                index=level_index,
                height_m=float(np.median(height[first:stop_index])),
                start_m=float(progress[first]),
                stop_m=stop_m,
                requires_double_support=level_index == len(runs) - 1,
            )
        )
    return OrderedTerrainLevelContract(
        path_start_scene_xy=tuple(float(value) for value in start),
        path_heading_scene_xy=tuple(float(value) for value in heading),
        path_length_m=length,
        levels=tuple(levels),
    )


def match_ordered_touchdown_levels(
    contract: OrderedTerrainLevelContract,
    *,
    touchdown_progress_m: object,
    touchdown_height_m: object,
    touchdown_foot: object,
    height_tolerance_m: float = 0.03,
    progress_tolerance_m: float = 0.08,
) -> tuple[int, ...]:
    """Match actual touchdowns to every required terrain level in order."""

    progress = np.asarray(touchdown_progress_m, dtype=np.float64)
    height = np.asarray(touchdown_height_m, dtype=np.float64)
    feet = np.asarray(touchdown_foot)
    if (
        not isinstance(contract, OrderedTerrainLevelContract)
        or progress.ndim != 1
        or height.shape != progress.shape
        or feet.shape != progress.shape
        or len(progress) < 2
        or not np.isfinite(progress).all()
        or not np.isfinite(height).all()
        or not np.issubdtype(feet.dtype, np.integer)
        or not np.isin(feet, (0, 1)).all()
        or np.any(np.diff(progress) < -1.0e-6)
        or not math.isfinite(float(height_tolerance_m))
        or height_tolerance_m <= 0.0
        or not math.isfinite(float(progress_tolerance_m))
        or progress_tolerance_m < 0.0
    ):
        raise ContractError("ordered touchdown inputs are invalid")

    matched = []
    current = 0
    for event_progress, event_height in zip(progress, height):
        candidates = tuple(
            level.index
            for level in contract.levels
            if level.start_m - progress_tolerance_m
            <= float(event_progress)
            <= level.stop_m + progress_tolerance_m
            and abs(float(event_height) - level.height_m)
            <= height_tolerance_m
        )
        if current in candidates:
            selected = current
        elif current + 1 in candidates:
            current += 1
            selected = current
        elif any(index > current + 1 for index in candidates):
            raise OrderedLevelRejected("skipped-required-level")
        else:
            raise OrderedLevelRejected("unmatched-touchdown")
        matched.append(selected)

    final = len(contract.levels) - 1
    if current < final:
        raise OrderedLevelRejected("incomplete-required-levels")
    final_feet = [
        int(foot)
        for foot, level_index in zip(feet, matched)
        if level_index == final
    ]
    if (
        contract.levels[-1].requires_double_support
        and (
            len(final_feet) < 2
            or final_feet[-1] == final_feet[-2]
        )
    ):
        raise OrderedLevelRejected("missing-ground-exit")
    return tuple(matched)
