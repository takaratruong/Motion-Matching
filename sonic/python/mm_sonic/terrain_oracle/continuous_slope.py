"""Continuous-ramp source windows and route scaffolds for mesh-only warping."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .stair_support_route import StairSupportRoute, VisibleTread
from .terrain_mesh import TerrainMeshIndex
from .terrain_route_profile import TerrainRouteProfile


@dataclass(frozen=True)
class MonotonicSlopeWindow:
    start_frame: int
    stop_frame: int
    traversal: str
    height_delta_m: float


def surface_height_along_trajectory(
    mesh: TerrainMeshIndex,
    root_position_world: object,
    *,
    ground_fallback_height_m: float = 0.0,
) -> np.ndarray:
    root = np.asarray(root_position_world, dtype=np.float64)
    if root.ndim != 2 or root.shape[1:] != (3,):
        raise ValueError("root positions must have shape [T,3]")
    ray_z = max(
        float(np.max(mesh.vertices_world[:, 2])),
        float(ground_fallback_height_m),
    ) + 1.0
    height = np.empty(len(root), dtype=np.float64)
    for index, point in enumerate(root):
        hit = mesh.raycast(
            (float(point[0]), float(point[1]), ray_z),
            (0.0, 0.0, -1.0),
        )
        height[index] = (
            float(ground_fallback_height_m)
            if hit is None
            else float(hit.position_world[2])
        )
    return height


def extract_monotonic_slope_window(
    root_position_world: object,
    surface_height_m: object,
    *,
    traversal: str,
    minimum_height_delta_m: float = 0.035,
    minimum_frames: int = 20,
    context_frames: int = 8,
    minimum_endpoint_displacement_m: float = 0.25,
) -> MonotonicSlopeWindow | None:
    """Extract the largest ordered up/down terrain excursion from one clip."""

    root = np.asarray(root_position_world, dtype=np.float64)
    height = np.asarray(surface_height_m, dtype=np.float64)
    if root.ndim != 2 or root.shape[1:] != (3,) or height.shape != (len(root),):
        raise ValueError("slope window inputs have incompatible shapes")
    if traversal not in {"up", "down"}:
        raise ValueError("slope traversal must be up or down")
    if len(height) < minimum_frames:
        return None
    padded = np.pad(height, (2, 2), mode="edge")
    smooth = np.convolve(padded, np.ones(5) / 5.0, mode="valid")
    sign = 1.0 if traversal == "up" else -1.0
    directed = sign * smooth
    best_delta = -math.inf
    best_start = 0
    best_stop = 0
    minimum_value = float(directed[0])
    minimum_index = 0
    for stop in range(1, len(directed)):
        delta = float(directed[stop] - minimum_value)
        if stop - minimum_index + 1 >= int(minimum_frames) and delta > best_delta:
            best_delta = delta
            best_start = minimum_index
            best_stop = stop
        if float(directed[stop]) < minimum_value:
            minimum_value = float(directed[stop])
            minimum_index = stop
    if best_delta < float(minimum_height_delta_m):
        return None
    start = max(0, best_start - int(context_frames))
    stop = min(len(root), best_stop + int(context_frames) + 1)
    if (
        stop - start < int(minimum_frames)
        or float(np.linalg.norm(root[stop - 1, :2] - root[start, :2]))
        < float(minimum_endpoint_displacement_m)
    ):
        return None
    return MonotonicSlopeWindow(
        start_frame=start,
        stop_frame=stop,
        traversal=traversal,
        height_delta_m=sign * float(smooth[best_stop] - smooth[best_start]),
    )


def continuous_profile_support_route(
    profile: TerrainRouteProfile,
    *,
    level_count: int = 24,
) -> StairSupportRoute:
    """Represent a continuous height curve with equal-count warp knots."""

    count = int(level_count)
    if count < 2:
        raise ValueError("continuous profile needs at least two warp levels")
    start = np.asarray(profile.start_xy, dtype=np.float64)
    end = np.asarray(profile.end_xy, dtype=np.float64)
    vector = end - start
    length = float(np.linalg.norm(vector))
    direction = vector / length
    boundaries = np.linspace(0.0, length, count + 1)
    levels = []
    for index in range(count):
        left = float(boundaries[index])
        right = float(boundaries[index + 1])
        centre = 0.5 * (left + right)
        height = float(
            np.interp(centre, profile.distance_m, profile.height_m)
        )
        visibility = bool(
            profile.visible_in_mesh[
                int(np.argmin(np.abs(profile.distance_m - centre)))
            ]
        )
        levels.append(
            VisibleTread(
                height_m=height,
                route_start_distance_m=left,
                route_stop_distance_m=right,
                route_start_xy=np.asarray(
                    start + left * direction, dtype=np.float32
                ),
                route_stop_xy=np.asarray(
                    start + right * direction, dtype=np.float32
                ),
                visible_in_mesh=visibility,
                left_foothold_center_xy=None,
                right_foothold_center_xy=None,
            )
        )
    return StairSupportRoute(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        levels=tuple(levels),
    )


__all__ = (
    "MonotonicSlopeWindow",
    "continuous_profile_support_route",
    "extract_monotonic_slope_window",
    "surface_height_along_trajectory",
)
