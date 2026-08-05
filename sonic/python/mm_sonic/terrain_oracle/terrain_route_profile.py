"""Classify an exact-mesh route as flat, curb, stairs, slope, or mixed."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .terrain_mesh import TerrainMeshIndex


@dataclass(frozen=True)
class TerrainRouteProfile:
    start_xy: np.ndarray
    end_xy: np.ndarray
    distance_m: np.ndarray
    height_m: np.ndarray
    visible_in_mesh: np.ndarray
    kind: str
    height_range_m: float
    endpoint_height_delta_m: float
    jump_count: int
    continuous_slope_fraction: float
    maximum_continuous_slope_rad: float


def sample_terrain_route_profile(
    mesh: TerrainMeshIndex,
    start_xy: object,
    end_xy: object,
    *,
    ground_fallback_height_m: float = 0.0,
    sample_spacing_m: float = 0.01,
    jump_threshold_m: float = 0.03,
    slope_threshold_rad: float = math.radians(3.0),
    flat_height_range_m: float = 0.025,
) -> TerrainRouteProfile:
    """Ray-cast and classify a straight globally known route.

    A jump is a vertical change too large to be a plausible continuous ramp
    over one sample.  Remaining sustained gradients distinguish slopes from
    horizontal support.  This is geometry-only: no target robot motion is
    inspected.
    """

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError("route endpoints must be XY vectors")
    vector = end - start
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("route must have positive finite length")
    spacing = float(sample_spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("sample spacing must be positive and finite")
    direction = vector / length
    count = max(2, int(math.ceil(length / spacing)) + 1)
    distance = np.linspace(0.0, length, count)
    points = start[None] + distance[:, None] * direction[None]
    ray_z = max(
        float(np.max(mesh.vertices_world[:, 2])),
        float(ground_fallback_height_m),
    ) + 1.0
    height = np.empty(count, dtype=np.float64)
    visible = np.zeros(count, dtype=np.bool_)
    for index, point in enumerate(points):
        hit = mesh.raycast(
            (float(point[0]), float(point[1]), ray_z),
            (0.0, 0.0, -1.0),
        )
        if hit is None:
            height[index] = float(ground_fallback_height_m)
        else:
            height[index] = float(hit.position_world[2])
            visible[index] = True

    delta_distance = np.diff(distance)
    delta_height = np.diff(height)
    jumps = np.abs(delta_height) > float(jump_threshold_m)
    gradient = np.divide(
        delta_height,
        delta_distance,
        out=np.zeros_like(delta_height),
        where=delta_distance > 0.0,
    )
    continuous = ~jumps
    slope_edges = continuous & (
        np.abs(gradient) > math.tan(float(slope_threshold_rad))
    )
    slope_fraction = float(np.mean(slope_edges)) if len(slope_edges) else 0.0
    maximum_slope = float(
        np.max(np.arctan(np.abs(gradient[continuous])), initial=0.0)
    )
    height_range = float(np.max(height) - np.min(height))
    endpoint_delta = float(height[-1] - height[0])
    jump_count = int(np.count_nonzero(jumps))
    if jump_count:
        stepped_kind = "curb" if jump_count <= 2 else "stairs"
        kind = "mixed" if slope_fraction >= 0.12 else stepped_kind
    elif height_range <= float(flat_height_range_m):
        kind = "flat"
    elif slope_fraction >= 0.12:
        kind = "slope"
    else:
        kind = "mixed"
    return TerrainRouteProfile(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        distance_m=np.asarray(distance, dtype=np.float32),
        height_m=np.asarray(height, dtype=np.float32),
        visible_in_mesh=visible,
        kind=kind,
        height_range_m=height_range,
        endpoint_height_delta_m=endpoint_delta,
        jump_count=jump_count,
        continuous_slope_fraction=slope_fraction,
        maximum_continuous_slope_rad=maximum_slope,
    )


__all__ = ("TerrainRouteProfile", "sample_terrain_route_profile")
