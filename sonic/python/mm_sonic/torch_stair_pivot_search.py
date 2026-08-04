"""Deterministic task-space candidates for stair pivot-step connectors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from .joints import ContractError
from .torch_stair_connector import pivot_stance_target


@dataclass(frozen=True)
class PivotTargetCandidate:
    pivot_foot: int
    yaw_delta_rad: float
    planar_offset_m: tuple[float, float]
    target_foot_position_world: np.ndarray


def pivot_target_lattice(
    foot_position_world: object,
    *,
    yaw_delta_rad: float,
    planar_offsets_m: Sequence[tuple[float, float]],
    sample_surface_height_m: Callable[[np.ndarray], object],
) -> tuple[PivotTargetCandidate, ...]:
    """Place rotated swing-foot candidates onto the queried terrain surface."""

    feet = np.asarray(foot_position_world, dtype=np.float64)
    offsets = tuple(planar_offsets_m)
    if (
        feet.shape != (2, 3)
        or not np.isfinite(feet).all()
        or isinstance(yaw_delta_rad, bool)
        or not isinstance(yaw_delta_rad, (int, float))
        or not math.isfinite(float(yaw_delta_rad))
        or not offsets
        or not callable(sample_surface_height_m)
    ):
        raise ContractError("stair pivot target lattice inputs are invalid")
    owned_offsets = []
    for offset in offsets:
        value = np.asarray(offset, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ContractError("stair pivot target lattice inputs are invalid")
        owned_offsets.append((float(value[0]), float(value[1])))

    candidates = []
    for pivot_foot in (0, 1):
        swing_foot = 1 - pivot_foot
        nominal = pivot_stance_target(
            feet,
            pivot_foot=pivot_foot,
            yaw_delta_rad=float(yaw_delta_rad),
        )
        query_points = np.asarray(
            (
                feet[swing_foot, :2],
                *(
                    nominal[swing_foot, :2] + np.asarray(offset)
                    for offset in owned_offsets
                ),
            ),
            dtype=np.float64,
        )
        try:
            surface = np.asarray(
                sample_surface_height_m(query_points),
                dtype=np.float64,
            )
        except Exception as error:
            raise ContractError("stair pivot terrain query failed") from error
        if (
            surface.shape != (len(query_points),)
            or not np.isfinite(surface).all()
        ):
            raise ContractError("stair pivot terrain query failed")
        source_surface = float(surface[0])
        for index, offset in enumerate(owned_offsets):
            target = nominal.copy()
            target[swing_foot, :2] += np.asarray(offset)
            target[swing_foot, 2] = (
                feet[swing_foot, 2]
                + float(surface[index + 1])
                - source_surface
            )
            target.flags.writeable = False
            candidates.append(
                PivotTargetCandidate(
                    pivot_foot=pivot_foot,
                    yaw_delta_rad=float(yaw_delta_rad),
                    planar_offset_m=offset,
                    target_foot_position_world=target,
                )
            )
    return tuple(candidates)
