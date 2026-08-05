"""Split one globally known route into reusable terrain traversal events."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .terrain_route_profile import TerrainRouteProfile


@dataclass(frozen=True)
class TerrainRouteEvent:
    kind: str
    start_distance_m: float
    end_distance_m: float
    start_xy: np.ndarray
    end_xy: np.ndarray


def _groups(indices: np.ndarray, coordinate: np.ndarray, gap_m: float) -> list[np.ndarray]:
    if not len(indices):
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        value = int(index)
        if float(coordinate[value] - coordinate[groups[-1][-1]]) <= float(gap_m):
            groups[-1].append(value)
        else:
            groups.append([value])
    return [np.asarray(group, dtype=np.int64) for group in groups]


def segment_terrain_route_profile(
    profile: TerrainRouteProfile,
    *,
    context_margin_m: float = 0.35,
    slope_context_margin_m: float = 0.12,
    maximum_step_gap_m: float = 0.70,
    maximum_slope_gap_m: float = 0.10,
) -> tuple[TerrainRouteEvent, ...]:
    """Return non-flat route events with flat context for portal handoffs."""

    distance = np.asarray(profile.distance_m, dtype=np.float64)
    height = np.asarray(profile.height_m, dtype=np.float64)
    edge_distance = 0.5 * (distance[:-1] + distance[1:])
    delta_distance = np.diff(distance)
    delta_height = np.diff(height)
    gradient = np.divide(
        delta_height,
        delta_distance,
        out=np.zeros_like(delta_height),
        where=delta_distance > 0.0,
    )
    jump = np.abs(delta_height) > 0.03
    slope = (~jump) & (np.abs(gradient) > math.tan(math.radians(3.0)))
    jump_indices = np.flatnonzero(jump)
    # Triangle edges immediately beside a vertical transition can look like a
    # one-sample ramp.  Keep those inside the stepped event instead.
    if len(jump_indices):
        jump_distance = edge_distance[jump_indices]
        near_jump = np.min(
            np.abs(edge_distance[:, None] - jump_distance[None, :]), axis=1
        ) <= 0.08
        slope &= ~near_jump

    raw: list[tuple[str, float, float]] = []
    for group in _groups(jump_indices, edge_distance, maximum_step_gap_m):
        raw.append(
            (
                "stepped",
                float(edge_distance[group[0]]),
                float(edge_distance[group[-1]]),
            )
        )
    for group in _groups(np.flatnonzero(slope), edge_distance, maximum_slope_gap_m):
        raw.append(
            (
                "slope",
                float(edge_distance[group[0]]),
                float(edge_distance[group[-1]]),
            )
        )
    raw.sort(key=lambda value: (value[1], value[2], value[0]))
    if not raw:
        return ()

    # Overlapping support evidence is one mixed event.  Otherwise retain
    # separate portals, leaving MotionBricks in charge of the flat interval.
    merged: list[tuple[str, float, float]] = []
    for kind, start, stop in raw:
        if merged and start <= merged[-1][2] + 0.02:
            previous_kind, previous_start, previous_stop = merged[-1]
            merged[-1] = (
                previous_kind if previous_kind == kind else "mixed",
                previous_start,
                max(previous_stop, stop),
            )
        else:
            merged.append((kind, start, stop))

    route_start = np.asarray(profile.start_xy, dtype=np.float64)
    route_end = np.asarray(profile.end_xy, dtype=np.float64)
    route_length = float(distance[-1])
    direction = (route_end - route_start) / route_length
    events: list[TerrainRouteEvent] = []
    for kind, start, stop in merged:
        margin = (
            float(slope_context_margin_m)
            if kind == "slope"
            else float(context_margin_m)
        )
        lo = max(0.0, start - margin)
        hi = min(route_length, stop + margin)
        if hi - lo < 0.20:
            centre = 0.5 * (lo + hi)
            lo = max(0.0, centre - 0.10)
            hi = min(route_length, centre + 0.10)
        events.append(
            TerrainRouteEvent(
                kind=kind,
                start_distance_m=lo,
                end_distance_m=hi,
                start_xy=np.asarray(route_start + lo * direction, dtype=np.float32),
                end_xy=np.asarray(route_start + hi * direction, dtype=np.float32),
            )
        )
    return tuple(events)


__all__ = ("TerrainRouteEvent", "segment_terrain_route_profile")
