"""Route-local stair support discovery from exact terrain-mesh ray queries.

This is intentionally independent from clips, commands, and robot kinematics.
It turns a requested world-XY route into visible horizontal plateaus and places
two conservative, route-aligned sole centres only where sampled mesh support
covers the complete sole footprint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .terrain_mesh import TerrainMeshIndex


@dataclass(frozen=True)
class VisibleTread:
    """One route-contiguous height plateau and its safe bilateral footholds."""

    height_m: float
    route_start_distance_m: float
    route_stop_distance_m: float
    route_start_xy: np.ndarray
    route_stop_xy: np.ndarray
    visible_in_mesh: bool
    left_foothold_center_xy: np.ndarray | None
    right_foothold_center_xy: np.ndarray | None


@dataclass(frozen=True)
class StairSupportRoute:
    """Ordered support plateaus encountered while travelling along a route."""

    start_xy: np.ndarray
    end_xy: np.ndarray
    levels: tuple[VisibleTread, ...]


def extend_stair_route_to_requested_plateau_edge(
    scanned_route: StairSupportRoute,
    requested_end_xy: object,
    *,
    end_margin_m: float = 0.05,
) -> StairSupportRoute:
    """Finish the requested route on the plateau that already contains its end.

    The joystick path may reach a new landing only a few centimetres before
    its current finite horizon ends.  A longer privileged mesh scan reveals
    the rest of that same plateau.  Extending only to its far edge preserves
    the user's intended staircase instead of accidentally consuming another
    staircase farther along the same ray.
    """

    start = np.asarray(scanned_route.start_xy, dtype=np.float64)
    scanned_end = np.asarray(scanned_route.end_xy, dtype=np.float64)
    route_vector = scanned_end - start
    route_length = float(np.linalg.norm(route_vector))
    if route_length <= 0.0:
        raise ValueError("scanned stair route has zero length")
    direction = route_vector / route_length
    requested = np.asarray(requested_end_xy, dtype=np.float64)
    requested_distance = float((requested - start) @ direction)
    landing_index = next(
        (
            index
            for index, level in enumerate(scanned_route.levels)
            if (
                float(level.route_start_distance_m) - 1.0e-6
                <= requested_distance
                <= float(level.route_stop_distance_m) + 1.0e-6
            )
        ),
        None,
    )
    if landing_index is None:
        raise ValueError("requested route end is outside the scanned support")
    landing = scanned_route.levels[landing_index]
    end_distance = max(
        requested_distance,
        float(landing.route_stop_distance_m) - float(end_margin_m),
    )
    end_distance = min(end_distance, float(landing.route_stop_distance_m))
    end = start + end_distance * direction
    trimmed_landing = replace(
        landing,
        route_stop_distance_m=end_distance,
        route_stop_xy=np.asarray(end, dtype=np.float32),
    )
    return StairSupportRoute(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        levels=scanned_route.levels[:landing_index] + (trimmed_landing,),
    )


def truncate_stair_route_at_first_landing(
    route: StairSupportRoute,
    traversal: str,
    *,
    minimum_landing_length_m: float = 0.30,
    end_margin_m: float = 0.05,
    height_tolerance_m: float = 0.002,
) -> StairSupportRoute:
    """Stop a long global route on the first landing after one traversal.

    A privileged bootstrap may scan well beyond the requested staircase.  The
    first height reversal marks the far edge of its landing (for example the
    beginning of stairs down the other side of a platform).  Keeping the
    landing context avoids ending a reconstruction halfway through its final
    support transfer.
    """

    if traversal not in {"up", "down"}:
        raise ValueError("stair traversal must be 'up' or 'down'")
    direction_sign = 1.0 if traversal == "up" else -1.0
    desired_seen = False
    landing_index: int | None = None
    for index, (source, target) in enumerate(
        zip(route.levels, route.levels[1:])
    ):
        delta = float(target.height_m) - float(source.height_m)
        if direction_sign * delta > height_tolerance_m:
            desired_seen = True
        elif desired_seen and direction_sign * delta < -height_tolerance_m:
            landing_index = index
            break
    if not desired_seen:
        raise ValueError("sampled route contains no requested stair traversal")
    if landing_index is None:
        landing_index = len(route.levels) - 1

    landing = route.levels[landing_index]
    landing_length = (
        float(landing.route_stop_distance_m)
        - float(landing.route_start_distance_m)
    )
    if landing_length < minimum_landing_length_m:
        raise ValueError(
            "sampled route does not include enough far-landing context"
        )
    retained_landing = max(
        minimum_landing_length_m,
        landing_length - float(end_margin_m),
    )
    end_distance = (
        float(landing.route_start_distance_m) + retained_landing
    )
    start = np.asarray(route.start_xy, dtype=np.float64)
    original_end = np.asarray(route.end_xy, dtype=np.float64)
    direction = original_end - start
    direction /= np.linalg.norm(direction)
    end = start + end_distance * direction
    trimmed_landing = replace(
        landing,
        route_stop_distance_m=end_distance,
        route_stop_xy=np.asarray(end, dtype=np.float32),
    )
    return StairSupportRoute(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        levels=route.levels[:landing_index] + (trimmed_landing,),
    )


@dataclass(frozen=True)
class _RouteSample:
    distance_m: float
    xy: np.ndarray
    height_m: float
    visible_in_mesh: bool


def _downward_hit(
    mesh: TerrainMeshIndex,
    xy: np.ndarray,
    ray_origin_height_m: float,
) -> float | None:
    hit = mesh.raycast(
        np.array((xy[0], xy[1], ray_origin_height_m), dtype=np.float64),
        np.array((0.0, 0.0, -1.0), dtype=np.float64),
    )
    return None if hit is None else float(hit.position_world[2])


def _footprint_is_visible(
    mesh: TerrainMeshIndex,
    centre_xy: np.ndarray,
    direction_xy: np.ndarray,
    lateral_xy: np.ndarray,
    height_m: float,
    sole_half_length_m: float,
    sole_half_width_m: float,
    ray_origin_height_m: float,
    height_tolerance_m: float,
) -> bool:
    """Sample a 3x3 sole grid; every point must meet the observed tread height."""

    for along in (-sole_half_length_m, 0.0, sole_half_length_m):
        for sideways in (-sole_half_width_m, 0.0, sole_half_width_m):
            point = centre_xy + along * direction_xy + sideways * lateral_xy
            hit_height = _downward_hit(mesh, point, ray_origin_height_m)
            if hit_height is None or abs(hit_height - height_m) > height_tolerance_m:
                return False
    return True


def _safe_side_foothold(
    mesh: TerrainMeshIndex,
    route_centre_xy: np.ndarray,
    direction_xy: np.ndarray,
    lateral_xy: np.ndarray,
    height_m: float,
    sole_half_length_m: float,
    sole_half_width_m: float,
    ray_origin_height_m: float,
    height_tolerance_m: float,
    side: float,
    lateral_probe_limit_m: float,
    lateral_probe_spacing_m: float,
) -> np.ndarray | None:
    desired_offset = side * max(2.0 * sole_half_width_m, lateral_probe_spacing_m)
    offsets = np.arange(
        lateral_probe_spacing_m,
        lateral_probe_limit_m + 0.5 * lateral_probe_spacing_m,
        lateral_probe_spacing_m,
    )
    candidates = [
        side * offset
        for offset in offsets
        if _footprint_is_visible(
            mesh,
            route_centre_xy + side * offset * lateral_xy,
            direction_xy,
            lateral_xy,
            height_m,
            sole_half_length_m,
            sole_half_width_m,
            ray_origin_height_m,
            height_tolerance_m,
        )
    ]
    if not candidates:
        return None
    offset = min(candidates, key=lambda value: abs(value - desired_offset))
    return np.asarray(route_centre_xy + offset * lateral_xy, dtype=np.float32)


def _make_tread(
    mesh: TerrainMeshIndex,
    samples: list[_RouteSample],
    direction_xy: np.ndarray,
    lateral_xy: np.ndarray,
    sole_half_length_m: float,
    sole_half_width_m: float,
    ray_origin_height_m: float,
    height_tolerance_m: float,
    lateral_probe_limit_m: float,
    lateral_probe_spacing_m: float,
) -> VisibleTread:
    first = samples[0]
    last = samples[-1]
    height = float(np.mean([sample.height_m for sample in samples]))
    visible = all(sample.visible_in_mesh for sample in samples)
    left = None
    right = None
    interval_length = last.distance_m - first.distance_m
    if visible and interval_length >= 2.0 * sole_half_length_m:
        route_centre = 0.5 * (first.xy + last.xy)
        left = _safe_side_foothold(
            mesh,
            route_centre,
            direction_xy,
            lateral_xy,
            height,
            sole_half_length_m,
            sole_half_width_m,
            ray_origin_height_m,
            height_tolerance_m,
            side=1.0,
            lateral_probe_limit_m=lateral_probe_limit_m,
            lateral_probe_spacing_m=lateral_probe_spacing_m,
        )
        right = _safe_side_foothold(
            mesh,
            route_centre,
            direction_xy,
            lateral_xy,
            height,
            sole_half_length_m,
            sole_half_width_m,
            ray_origin_height_m,
            height_tolerance_m,
            side=-1.0,
            lateral_probe_limit_m=lateral_probe_limit_m,
            lateral_probe_spacing_m=lateral_probe_spacing_m,
        )
    return VisibleTread(
        height_m=height,
        route_start_distance_m=first.distance_m,
        route_stop_distance_m=last.distance_m,
        route_start_xy=np.asarray(first.xy, dtype=np.float32),
        route_stop_xy=np.asarray(last.xy, dtype=np.float32),
        visible_in_mesh=visible,
        left_foothold_center_xy=left,
        right_foothold_center_xy=right,
    )


def sample_stair_support_route(
    mesh: TerrainMeshIndex,
    start_xy: object,
    end_xy: object,
    ground_fallback_height_m: float,
    sole_half_length_m: float,
    sole_half_width_m: float,
    *,
    sample_spacing_m: float = 0.025,
    height_tolerance_m: float = 0.002,
    lateral_probe_limit_m: float = 0.50,
    lateral_probe_spacing_m: float = 0.01,
) -> StairSupportRoute:
    """Return visible stair treads and conservative route-aligned foot centres.

    A downward ray is cast at each route sample.  Ray misses use
    ``ground_fallback_height_m`` for level continuity but never create a safe
    foothold.  Equal-height adjacent samples collapse into one tread interval.
    """

    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    route = end - start
    route_length = float(np.linalg.norm(route))
    direction = route / route_length
    lateral = np.array((-direction[1], direction[0]), dtype=np.float64)
    sample_count = max(2, int(math.ceil(route_length / sample_spacing_m)) + 1)
    distances = np.linspace(0.0, route_length, sample_count)
    ray_origin = max(
        float(np.max(mesh.vertices_world[:, 2])) if len(mesh.vertices_world) else 0.0,
        float(ground_fallback_height_m),
    ) + 1.0
    samples: list[_RouteSample] = []
    for distance in distances:
        xy = start + distance * direction
        hit_height = _downward_hit(mesh, xy, ray_origin)
        samples.append(
            _RouteSample(
                distance_m=float(distance),
                xy=xy,
                height_m=(
                    float(ground_fallback_height_m)
                    if hit_height is None
                    else hit_height
                ),
                visible_in_mesh=hit_height is not None,
            )
        )

    plateaus: list[list[_RouteSample]] = [[samples[0]]]
    for sample in samples[1:]:
        previous = plateaus[-1][-1]
        if (
            sample.visible_in_mesh == previous.visible_in_mesh
            and abs(sample.height_m - previous.height_m) <= height_tolerance_m
        ):
            plateaus[-1].append(sample)
        else:
            plateaus.append([sample])
    levels = tuple(
        _make_tread(
            mesh,
            plateau,
            direction,
            lateral,
            sole_half_length_m,
            sole_half_width_m,
            ray_origin,
            height_tolerance_m,
            lateral_probe_limit_m,
            lateral_probe_spacing_m,
        )
        for plateau in plateaus
    )
    return StairSupportRoute(
        start_xy=np.asarray(start, dtype=np.float32),
        end_xy=np.asarray(end, dtype=np.float32),
        levels=levels,
    )
