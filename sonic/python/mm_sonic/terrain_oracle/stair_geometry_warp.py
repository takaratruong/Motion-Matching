"""Warp authored stair kinematics onto a route sampled from another mesh.

The target side deliberately contains no motion scaffold.  It consists only of
an exact terrain mesh plus a requested world-XY route.  An authored source
motion supplies timing and full-body style; corresponding support-level
boundaries provide a piecewise route-coordinate warp, and bounded leg IK
reconciles the warped sole placements.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .contact import CanonicalMeshQuery
from .math3d import (
    RigidTransform,
    quaternion_multiply_wxyz,
    unroll_quaternions_wxyz,
)
from .reference_stitch import _G1FootfallAdapter
from .source_grail import _load_usd_mesh
from .stair_support_route import (
    StairSupportRoute,
    sample_stair_support_route,
)
from .stitch import FrameProvenance, StitchedMotion
from .terrain_mesh import TerrainMeshIndex


DEFAULT_MOTION_ROUTE_PROFILE_CATALOG = (
    Path(__file__).resolve().parents[4]
    / "artifacts/privileged_online_matcher/stairs500_motion_route_profiles_v1.npz"
)
_ROUTE_PROFILE_CATALOG_CACHE: dict[
    Path, tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}


def _route_length(route: StairSupportRoute) -> float:
    return float(
        np.linalg.norm(
            np.asarray(route.end_xy, dtype=np.float64)
            - np.asarray(route.start_xy, dtype=np.float64)
        )
    )


def _level_boundaries(route: StairSupportRoute) -> np.ndarray:
    """Return route endpoints plus mid-gaps between adjacent plateaus."""

    if not route.levels:
        raise ValueError("stair route has no support levels")
    boundaries = np.empty(len(route.levels) + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[-1] = _route_length(route)
    for index in range(1, len(route.levels)):
        boundaries[index] = 0.5 * (
            float(route.levels[index - 1].route_stop_distance_m)
            + float(route.levels[index].route_start_distance_m)
        )
    if np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("stair support levels are not ordered along the route")
    return boundaries


def _linear_extrapolate(
    values: np.ndarray,
    source_knots: np.ndarray,
    target_knots: np.ndarray,
) -> np.ndarray:
    output = np.interp(values, source_knots, target_knots)
    before = values < source_knots[0]
    after = values > source_knots[-1]
    if np.any(before):
        output[before] = target_knots[0] + (
            values[before] - source_knots[0]
        )
    if np.any(after):
        # Beyond the requested route, retain metric distance instead of
        # extending the final (possibly very short) sampled tread's scale.
        # Swing feet commonly lead the pelvis past the route endpoint.
        output[after] = target_knots[-1] + (
            values[after] - source_knots[-1]
        )
    return output


@dataclass(frozen=True)
class StairGeometryWarp:
    """Piecewise route-coordinate map between two support profiles."""

    source_route: StairSupportRoute
    target_route: StairSupportRoute
    source_boundaries_m: np.ndarray
    target_boundaries_m: np.ndarray
    source_heights_m: np.ndarray
    target_heights_m: np.ndarray
    source_direction_xy: np.ndarray
    target_direction_xy: np.ndarray
    yaw_delta_rad: float

    def _height_delta(
        self, source_distance_m: np.ndarray, *, smooth: bool
    ) -> np.ndarray:
        if not smooth:
            level = np.searchsorted(
                self.source_boundaries_m[1:-1],
                source_distance_m,
                side="right",
            )
            level = np.clip(level, 0, len(self.source_heights_m) - 1)
            return (
                self.target_heights_m[level]
                - self.source_heights_m[level]
            )

        # A short smooth transition prevents a root or swing foot from popping
        # when it crosses a vertical riser.  Away from the riser the offset is
        # exactly the target-level height minus the source-level height.
        output = np.full_like(
            source_distance_m,
            self.target_heights_m[0] - self.source_heights_m[0],
            dtype=np.float64,
        )
        intervals = np.diff(self.source_boundaries_m)
        for index in range(1, len(self.source_heights_m)):
            half_width = min(
                0.045,
                0.20 * float(min(intervals[index - 1], intervals[index])),
            )
            half_width = max(half_width, 1.0e-4)
            coordinate = np.clip(
                (
                    source_distance_m
                    - (self.source_boundaries_m[index] - half_width)
                )
                / (2.0 * half_width),
                0.0,
                1.0,
            )
            blend = coordinate * coordinate * (3.0 - 2.0 * coordinate)
            source_rise = (
                self.source_heights_m[index]
                - self.source_heights_m[index - 1]
            )
            target_rise = (
                self.target_heights_m[index]
                - self.target_heights_m[index - 1]
            )
            output += (target_rise - source_rise) * blend
        return output

    def warp_points(
        self, points_world: object, *, smooth_height: bool = False
    ) -> np.ndarray:
        """Warp world points while preserving their route-lateral offset."""

        points = np.asarray(points_world, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("points must be finite and end in XYZ")
        original_shape = points.shape
        flat = points.reshape((-1, 3))
        source_start = np.asarray(
            self.source_route.start_xy, dtype=np.float64
        )
        target_start = np.asarray(
            self.target_route.start_xy, dtype=np.float64
        )
        source_lateral = np.asarray(
            (-self.source_direction_xy[1], self.source_direction_xy[0])
        )
        target_lateral = np.asarray(
            (-self.target_direction_xy[1], self.target_direction_xy[0])
        )
        relative = flat[:, :2] - source_start
        source_distance = relative @ self.source_direction_xy
        lateral_distance = relative @ source_lateral
        target_distance = _linear_extrapolate(
            source_distance,
            self.source_boundaries_m,
            self.target_boundaries_m,
        )
        output = np.empty_like(flat)
        output[:, :2] = (
            target_start
            + target_distance[:, None] * self.target_direction_xy
            + lateral_distance[:, None] * target_lateral
        )
        output[:, 2] = flat[:, 2] + self._height_delta(
            source_distance, smooth=smooth_height
        )
        return output.reshape(original_shape)

    def warp_rigid_point_cloud(
        self, points_world: object, *, smooth_height: bool = True
    ) -> np.ndarray:
        """Warp a sole centre while retaining exact within-sole geometry."""

        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("rigid point cloud must have shape [N,3]")
        centre = np.mean(points, axis=0)
        warped_centre = self.warp_points(
            centre[None], smooth_height=smooth_height
        )[0]
        cosine = math.cos(self.yaw_delta_rad)
        sine = math.sin(self.yaw_delta_rad)
        rotation = np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        return warped_centre + (points - centre) @ rotation.T

    def rotate_quaternions_wxyz(self, values: object) -> np.ndarray:
        quaternions = np.asarray(values, dtype=np.float64)
        delta = np.asarray(
            (
                math.cos(0.5 * self.yaw_delta_rad),
                0.0,
                0.0,
                math.sin(0.5 * self.yaw_delta_rad),
            )
        )
        return np.asarray(
            unroll_quaternions_wxyz(
                quaternion_multiply_wxyz(delta, quaternions)
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class StairGeometryWarpResult:
    """Motion adapted from one source clip to a target mesh-only route."""

    motion: StitchedMotion
    source_route: StairSupportRoute
    target_route: StairSupportRoute
    maximum_joint_correction_rad: float
    maximum_foot_target_error_m: float
    maximum_sole_penetration_m: float
    maximum_triangle_sphere_penetration_m: float
    minimum_sole_clearance_m: float
    maximum_root_clearance_lift_m: float
    maximum_root_route_adjustment_m: float
    maximum_root_route_lateral_deviation_m: float
    minimum_stance_support_point_count: int
    per_frame_joint_correction_rad: np.ndarray
    per_frame_foot_target_error_m: np.ndarray
    per_frame_minimum_sole_clearance_m: np.ndarray
    per_frame_triangle_sphere_penetration_m: np.ndarray
    per_frame_root_clearance_lift_m: np.ndarray
    per_frame_stance_support_point_count: np.ndarray
    maximum_foothold_progress_shift_m: float = 0.0
    per_frame_foothold_progress_shift_m: np.ndarray | None = None
    maximum_foothold_yaw_adjustment_rad: float = 0.0
    per_frame_foothold_yaw_adjustment_rad: np.ndarray | None = None


@dataclass(frozen=True)
class AutomaticStairGeometryWarpSelection:
    """Best mechanically feasible source from a geometry-prefiltered search."""

    result: StairGeometryWarpResult
    source_clip_index: int
    target_clip_index: int
    score: float
    evaluated_candidate_count: int
    rejected_candidate_count: int


def build_stair_geometry_warp(
    source_route: StairSupportRoute,
    target_route: StairSupportRoute,
) -> StairGeometryWarp:
    """Pair corresponding terrain levels without using any target motion."""

    if len(source_route.levels) != len(target_route.levels):
        raise ValueError(
            "source and target routes need the same support-level count: "
            f"{len(source_route.levels)} != {len(target_route.levels)}"
        )
    if len(source_route.levels) < 2:
        raise ValueError("stair geometry warp needs at least two levels")
    source_vector = np.asarray(
        source_route.end_xy, dtype=np.float64
    ) - np.asarray(source_route.start_xy, dtype=np.float64)
    target_vector = np.asarray(
        target_route.end_xy, dtype=np.float64
    ) - np.asarray(target_route.start_xy, dtype=np.float64)
    source_direction = source_vector / np.linalg.norm(source_vector)
    target_direction = target_vector / np.linalg.norm(target_vector)
    yaw_delta = math.atan2(
        float(target_direction[1]), float(target_direction[0])
    ) - math.atan2(
        float(source_direction[1]), float(source_direction[0])
    )
    yaw_delta = math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))
    return StairGeometryWarp(
        source_route=source_route,
        target_route=target_route,
        source_boundaries_m=_level_boundaries(source_route),
        target_boundaries_m=_level_boundaries(target_route),
        source_heights_m=np.asarray(
            [level.height_m for level in source_route.levels],
            dtype=np.float64,
        ),
        target_heights_m=np.asarray(
            [level.height_m for level in target_route.levels],
            dtype=np.float64,
        ),
        source_direction_xy=source_direction,
        target_direction_xy=target_direction,
        yaw_delta_rad=yaw_delta,
    )


def _archive_terrain_index(
    archive: object, clip_index: int
) -> TerrainMeshIndex:
    terrain_path = Path(str(archive["terrain_usd_path"][clip_index]))
    mesh = _load_usd_mesh(
        terrain_path,
        source_asset_sha256="0" * 64,
    )
    transform = RigidTransform(
        np.asarray(archive["terrain_position_env"][clip_index]),
        np.asarray(archive["terrain_rotation_env_wxyz"][clip_index]),
    )
    return TerrainMeshIndex(mesh, transform)


def motion_conditioned_stair_support_route(
    archive: object, clip_index: int
) -> StairSupportRoute:
    """Sample the exact terrain route traversed by one motion clip."""

    index = int(clip_index)
    start = int(archive["clip_start_idx"][index])
    stop = int(archive["clip_end_idx"][index])
    root = np.asarray(
        archive["body_pos_w"][start:stop, 0], dtype=np.float64
    )
    return sample_stair_support_route(
        _archive_terrain_index(archive, index),
        root[0, :2],
        root[-1, :2],
        0.0,
        0.10,
        0.055,
        sample_spacing_m=0.01,
    )


def motion_conditioned_stair_route_profile(
    archive: object, clip_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return support-level widths/heights along one motion's root route."""

    route = motion_conditioned_stair_support_route(archive, clip_index)
    return (
        np.diff(_level_boundaries(route)),
        np.asarray(
            [level.height_m for level in route.levels], dtype=np.float64
        ),
    )


def _motion_route_profile_from_path(
    arguments: tuple[str, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    import zarr

    archive_path, clip_index = arguments
    try:
        archive = zarr.open_group(archive_path, mode="r")
        return motion_conditioned_stair_route_profile(
            archive, int(clip_index)
        )
    except (ValueError, RuntimeError):
        return None


def build_motion_route_profile_catalog(
    archive_path: str | Path,
    output_path: str | Path = DEFAULT_MOTION_ROUTE_PROFILE_CATALOG,
    *,
    worker_count: int = 8,
) -> dict[str, object]:
    """Persist the expensive mesh/root-route queries used by source ranking."""

    import zarr

    source = Path(archive_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    archive = zarr.open_group(str(source), mode="r")
    count = len(archive["clip_names"])
    workers = int(worker_count)
    if workers <= 0:
        raise ValueError("worker_count must be positive")
    if workers == 1:
        rows = [
            _motion_route_profile_from_path((str(source), clip_index))
            for clip_index in range(count)
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(
                executor.map(
                    _motion_route_profile_from_path,
                    ((str(source), clip_index) for clip_index in range(count)),
                    chunksize=4,
                )
            )
    maximum_levels = max(
        (len(row[0]) for row in rows if row is not None), default=0
    )
    counts = np.zeros(len(rows), dtype=np.int16)
    widths = np.full((len(rows), maximum_levels), np.nan, dtype=np.float32)
    heights = np.full((len(rows), maximum_levels), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        if row is None:
            continue
        count = len(row[0])
        counts[index] = count
        widths[index, :count] = row[0]
        heights[index, :count] = row[1]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray("motion_conditioned_stair_route_profiles/v1"),
        archive_path=np.asarray(str(source)),
        level_count=counts,
        level_width_m=widths,
        level_height_m=heights,
    )
    _ROUTE_PROFILE_CATALOG_CACHE.pop(output, None)
    return {
        "schema": "motion_conditioned_stair_route_profiles/v1",
        "archive_path": str(source),
        "output_path": str(output),
        "clip_count": len(rows),
        "valid_clip_count": int(np.count_nonzero(counts)),
        "maximum_level_count": int(maximum_levels),
        "worker_count": workers,
    }


def _load_route_profile_catalog(
    archive: object,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    resolved = path.expanduser().resolve()
    cached = _ROUTE_PROFILE_CATALOG_CACHE.get(resolved)
    if cached is not None:
        return cached
    if not resolved.is_file():
        return None
    with np.load(resolved, allow_pickle=False) as data:
        counts = np.asarray(data["level_count"], dtype=np.int16)
        widths = np.asarray(data["level_width_m"], dtype=np.float64)
        heights = np.asarray(data["level_height_m"], dtype=np.float64)
    if (
        counts.shape != (len(archive["clip_names"]),)
        or widths.shape != heights.shape
        or widths.shape[0] != len(counts)
    ):
        return None
    cached = (counts, widths, heights)
    _ROUTE_PROFILE_CATALOG_CACHE[resolved] = cached
    return cached


def rank_geometry_compatible_sources(
    archive: object,
    target_clip_index: int,
    *,
    ranking: str = "nominal",
    reverse_target_traversal: bool = False,
    route_profile_catalog: str | Path | None = (
        DEFAULT_MOTION_ROUTE_PROFILE_CATALOG
    ),
) -> tuple[tuple[float, int], ...]:
    """Rank coherent source traversals for one target stair geometry.

    ``nominal`` reproduces the original metadata prefilter.  ``exact``
    ray-casts each physical stair profile, requires the same number of support
    levels, and compares the per-level tread widths and rises that the warp
    will pair. ``route`` measures those quantities only between each motion's
    actual root endpoints, exactly matching the support routes consumed by the
    warper.
    """

    from ..stair_mesh_profile import measure_archive_stair_profile

    target = int(target_clip_index)
    count = len(archive["clip_names"])
    if not 0 <= target < count:
        raise ValueError("target archive clip index is out of range")
    mode = str(ranking)
    if mode not in {"nominal", "exact", "route"}:
        raise ValueError("geometry ranking must be nominal, exact, or route")

    authored_traversal = str(archive["clip_traversal"][target])
    target_traversal = (
        ("down" if authored_traversal == "up" else "up")
        if reverse_target_traversal
        else authored_traversal
    )
    if mode == "nominal":
        target_steps = int(archive["stair_n_steps"][target])
        target_rise = float(archive["stair_rise_m"][target])
        target_tread = float(archive["stair_tread_m"][target])
        proposals = [
            (
                2.0
                * abs(float(archive["stair_rise_m"][source]) - target_rise)
                + abs(float(archive["stair_tread_m"][source]) - target_tread),
                source,
            )
            for source in range(count)
            if source != target
            and str(archive["clip_traversal"][source]) == target_traversal
            and int(archive["stair_n_steps"][source]) == target_steps
        ]
        return tuple(sorted(proposals, key=lambda value: (value[0], value[1])))

    def rises(heights: np.ndarray, traversal: str) -> np.ndarray:
        if traversal == "up":
            return np.diff(np.concatenate(((0.0,), heights)))
        return -np.diff(np.concatenate((heights, (0.0,))))

    if mode == "exact":
        target_profile = measure_archive_stair_profile(archive, target)
        target_widths = np.diff(
            np.asarray(target_profile.tread_edges_m, dtype=np.float64)
        )
        target_heights = np.asarray(
            target_profile.tread_heights_m, dtype=np.float64
        )
        target_level_count = target_profile.level_count
        if reverse_target_traversal:
            target_widths = target_widths[::-1]
            target_heights = target_heights[::-1]
    else:
        catalog = (
            None
            if route_profile_catalog is None
            else _load_route_profile_catalog(
                archive, Path(route_profile_catalog)
            )
        )
        if catalog is None or int(catalog[0][target]) <= 0:
            target_widths, target_heights = (
                motion_conditioned_stair_route_profile(archive, target)
            )
        else:
            target_level_count = int(catalog[0][target])
            target_widths = catalog[1][target, :target_level_count].copy()
            target_heights = catalog[2][target, :target_level_count].copy()
        if reverse_target_traversal:
            target_widths = target_widths[::-1]
            target_heights = target_heights[::-1]
        target_level_count = len(target_widths)

    target_rises = rises(target_heights, target_traversal)
    proposals: list[tuple[float, int]] = []
    for source in range(count):
        if (
            source == target
            or str(archive["clip_traversal"][source]) != target_traversal
        ):
            continue
        try:
            if mode == "exact":
                profile = measure_archive_stair_profile(archive, source)
                widths = np.diff(
                    np.asarray(profile.tread_edges_m, dtype=np.float64)
                )
                heights = np.asarray(
                    profile.tread_heights_m, dtype=np.float64
                )
                level_count = profile.level_count
            else:
                if catalog is None or int(catalog[0][source]) <= 0:
                    widths, heights = (
                        motion_conditioned_stair_route_profile(
                            archive, source
                        )
                    )
                else:
                    level_count = int(catalog[0][source])
                    widths = catalog[1][source, :level_count]
                    heights = catalog[2][source, :level_count]
                level_count = len(widths)
        except (ValueError, RuntimeError):
            continue
        if level_count != target_level_count:
            continue
        source_rises = rises(heights, target_traversal)
        # Route warping pairs corresponding support levels.  Width mismatch
        # controls horizontal timing deformation; rise mismatch is weighted
        # twice because it also drives leg correction and clearance.
        cost = float(
            np.mean(np.abs(widths - target_widths))
            + 2.0 * np.mean(np.abs(source_rises - target_rises))
        )
        proposals.append((cost, source))
    return tuple(sorted(proposals, key=lambda value: (value[0], value[1])))


def warp_archive_clip_to_stair_geometry(
    archive_path: str | Path,
    *,
    source_clip_index: int,
    source_start_frame: int,
    source_stop_frame: int,
    target_clip_index: int | None = None,
    target_mesh: TerrainMeshIndex | None = None,
    target_route_start_xy: object,
    target_route_end_xy: object,
    model_path: str | Path,
    source_support_route: StairSupportRoute | None = None,
    target_support_route: StairSupportRoute | None = None,
    ground_fallback_height_m: float = 0.0,
    sole_half_length_m: float = 0.10,
    sole_half_width_m: float = 0.055,
    route_sample_spacing_m: float = 0.01,
    maximum_joint_correction_rad: float = 0.35,
    maximum_foot_target_error_m: float = 0.001,
    maximum_sole_penetration_m: float = 0.0025,
    maximum_root_clearance_lift_m: float = 0.005,
    maximum_foothold_progress_shift_m: float = 0.0,
    foothold_progress_search_step_m: float = 0.005,
    foothold_edge_clearance_margin_m: float = 0.0,
    maximum_foothold_yaw_adjustment_rad: float = 0.0,
    foothold_yaw_search_step_rad: float = math.radians(5.0),
    minimum_stance_support_points: int = 1,
    support_contact_tolerance_m: float = 0.02,
    support_maximum_sole_speed_m_s: float = 0.25,
) -> StairGeometryWarpResult:
    """Fit one authored clip onto a supplied mesh-only route.

    ``target_clip_index`` is retained as a convenience for archive-backed
    evaluations.  Runtime callers can instead provide any ``target_mesh``;
    target robot poses are never read.
    """

    import zarr

    archive = zarr.open_group(str(Path(archive_path)), mode="r")
    source_index = int(source_clip_index)
    clip_start = int(archive["clip_start_idx"][source_index])
    clip_stop = int(archive["clip_end_idx"][source_index])
    frame_start = int(source_start_frame)
    frame_stop = int(source_stop_frame)
    if not 0 <= frame_start < frame_stop <= clip_stop - clip_start:
        raise ValueError("source frame range exceeds its archive clip")

    source_positions = np.asarray(
        archive["body_pos_w"][
            clip_start + frame_start : clip_start + frame_stop, 0
        ],
        dtype=np.float64,
    )
    source_xyzw = np.asarray(
        archive["body_quat_w"][
            clip_start + frame_start : clip_start + frame_stop, 0
        ],
        dtype=np.float64,
    )
    source_quaternions = np.asarray(
        unroll_quaternions_wxyz(source_xyzw[..., (3, 0, 1, 2)]),
        dtype=np.float64,
    )
    source_joints = np.asarray(
        archive["joint_pos"][
            clip_start + frame_start : clip_start + frame_stop
        ],
        dtype=np.float64,
    )
    joint_names: Sequence[str] = tuple(
        str(value) for value in archive["joint_names"][:]
    )

    source_mesh = _archive_terrain_index(archive, source_index)
    if target_mesh is None:
        if target_clip_index is None:
            raise ValueError("target_mesh or target_clip_index is required")
        target_mesh = _archive_terrain_index(archive, int(target_clip_index))
    elif not isinstance(target_mesh, TerrainMeshIndex):
        raise TypeError("target_mesh must be a TerrainMeshIndex")
    if source_support_route is None:
        source_route = sample_stair_support_route(
            source_mesh,
            source_positions[0, :2],
            source_positions[-1, :2],
            ground_fallback_height_m,
            sole_half_length_m,
            sole_half_width_m,
            sample_spacing_m=route_sample_spacing_m,
        )
    elif isinstance(source_support_route, StairSupportRoute):
        source_route = source_support_route
    else:
        raise TypeError("source_support_route must be a StairSupportRoute")
    if target_support_route is None:
        target_route = sample_stair_support_route(
            target_mesh,
            target_route_start_xy,
            target_route_end_xy,
            ground_fallback_height_m,
            sole_half_length_m,
            sole_half_width_m,
            sample_spacing_m=route_sample_spacing_m,
        )
    elif isinstance(target_support_route, StairSupportRoute):
        target_route = target_support_route
    else:
        raise TypeError("target_support_route must be a StairSupportRoute")
    warp = build_stair_geometry_warp(source_route, target_route)
    requested_root = warp.warp_points(
        source_positions, smooth_height=True
    )
    warped_root = requested_root.copy()
    warped_quaternions = warp.rotate_quaternions_wxyz(
        source_quaternions
    )

    adapter = _G1FootfallAdapter(
        Path(model_path),
        joint_names,
        maximum_joint_correction_rad=maximum_joint_correction_rad,
    )
    adapted_joints = np.empty_like(source_joints)
    corrections = np.empty(len(source_joints), dtype=np.float64)
    errors = np.empty(len(source_joints), dtype=np.float64)
    ray_origin_height = (
        float(np.max(target_mesh.vertices_world[:, 2])) + 1.0
    )
    sphere_radii = adapter.sole_sphere_radii()
    minimum_support = int(minimum_stance_support_points)
    if minimum_support < 1 or minimum_support > min(
        len(value) for value in sphere_radii
    ):
        raise ValueError("minimum_stance_support_points is out of range")
    if (
        support_contact_tolerance_m <= 0.0
        or support_maximum_sole_speed_m_s <= 0.0
    ):
        raise ValueError("stance support thresholds must be positive")
    maximum_foothold_shift = float(maximum_foothold_progress_shift_m)
    foothold_search_step = float(foothold_progress_search_step_m)
    foothold_edge_margin = float(foothold_edge_clearance_margin_m)
    maximum_foothold_yaw = float(maximum_foothold_yaw_adjustment_rad)
    foothold_yaw_step = float(foothold_yaw_search_step_rad)
    if (
        maximum_foothold_shift < 0.0
        or foothold_search_step <= 0.0
        or foothold_edge_margin < 0.0
        or maximum_foothold_yaw < 0.0
        or foothold_yaw_step <= 0.0
    ):
        raise ValueError("foothold progress search bounds are invalid")
    target_progress_direction = np.asarray(
        target_route.end_xy, dtype=np.float64
    ) - np.asarray(target_route.start_xy, dtype=np.float64)
    target_progress_direction /= np.linalg.norm(target_progress_direction)
    scaffold_penetration_budget = max(
        0.0,
        float(maximum_sole_penetration_m)
        - float(maximum_foot_target_error_m),
    )

    def vertical_clearance(
        support_points: Sequence[np.ndarray],
    ) -> float:
        values = []
        for point in np.concatenate(tuple(support_points)):
            hit = target_mesh.raycast(
                np.asarray(
                    (point[0], point[1], ray_origin_height),
                    dtype=np.float64,
                ),
                np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
            )
            surface_height = (
                ground_fallback_height_m
                if hit is None
                else float(hit.position_world[2])
            )
            values.append(float(point[2] - surface_height))
        return min(values)

    def point_vertical_clearances(points: np.ndarray) -> np.ndarray:
        """Signed sole-point clearance from the uppermost exact surface."""

        values = []
        for point in np.asarray(points, dtype=np.float64):
            hit = target_mesh.raycast(
                np.asarray(
                    (point[0], point[1], ray_origin_height),
                    dtype=np.float64,
                ),
                np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
            )
            surface_height = (
                ground_fallback_height_m
                if hit is None
                else float(hit.position_world[2])
            )
            values.append(float(point[2] - surface_height))
        return np.asarray(values, dtype=np.float64)

    def clearance_safe_foothold_transform(
        sole_centres: np.ndarray,
        radii: np.ndarray,
        collision_envelope: np.ndarray,
        *,
        support_group_size: int | None = None,
    ) -> tuple[float, float]:
        """Move an overhanging foot on a short tread instead of lifting it.

        A sole can be longer than a tread while still having a valid foothold:
        the heel may overhang the lower side, but the toe may not enter the
        next vertical riser.  Route warping used to centre the sole and then
        try to repair that collision by raising the complete character.  This
        bounded search chooses the smallest exact-mesh-safe combination of
        fore/aft translation and diagonal foot yaw.  The later source-stance
        audit remains the authority on whether the planted foot is supported.
        """

        if maximum_foothold_shift <= 0.0 and maximum_foothold_yaw <= 0.0:
            return 0.0, 0.0
        support = np.asarray(sole_centres, dtype=np.float64) - np.asarray(
            (0.0, 0.0, 1.0), dtype=np.float64
        )[None] * np.asarray(radii, dtype=np.float64)[:, None]
        envelope = np.asarray(collision_envelope, dtype=np.float64)
        original_support = point_vertical_clearances(support)
        original_envelope = point_vertical_clearances(envelope)

        def has_support(clearances: np.ndarray) -> bool:
            contact = (
                np.abs(np.asarray(clearances, dtype=np.float64))
                <= support_contact_tolerance_m
            )
            if support_group_size is None:
                return int(np.count_nonzero(contact)) >= minimum_support
            group = int(support_group_size)
            if group < 1 or len(contact) % group != 0:
                raise ValueError("invalid foothold support grouping")
            counts = np.count_nonzero(contact.reshape(-1, group), axis=1)
            return bool(np.all(counts >= minimum_support))

        if (
            float(np.min(original_support))
            >= -scaffold_penetration_budget
            and float(np.min(original_envelope))
            >= -scaffold_penetration_budget
            and has_support(original_support)
        ):
            return 0.0, 0.0
        # Search in increasing displacement so the selected repair is the
        # least invasive one.  Test both signs because ascent collisions are
        # normally repaired backward while descent collisions are repaired
        # forward in route coordinates.
        pivot = np.mean(np.asarray(sole_centres, dtype=np.float64), axis=0)

        def transformed(
            points: np.ndarray,
            progress_shift: float,
            yaw_adjustment: float,
        ) -> np.ndarray:
            output = np.asarray(points, dtype=np.float64).copy()
            cosine = math.cos(yaw_adjustment)
            sine = math.sin(yaw_adjustment)
            rotation = np.asarray(
                ((cosine, -sine), (sine, cosine)), dtype=np.float64
            )
            output[:, :2] = (
                (output[:, :2] - pivot[None, :2]) @ rotation.T
                + pivot[None, :2]
                + progress_shift * target_progress_direction[None]
            )
            return output

        progress_count = int(
            math.ceil(maximum_foothold_shift / foothold_search_step)
        )
        progress_values = [0.0]
        for index in range(1, progress_count + 1):
            distance = min(
                maximum_foothold_shift, index * foothold_search_step
            )
            progress_values.extend((-distance, distance))
        yaw_count = int(
            math.ceil(maximum_foothold_yaw / foothold_yaw_step)
        )
        yaw_values = [0.0]
        for index in range(1, yaw_count + 1):
            angle = min(maximum_foothold_yaw, index * foothold_yaw_step)
            yaw_values.extend((-angle, angle))
        candidates = sorted(
            (
                (distance, yaw)
                for distance in progress_values
                for yaw in yaw_values
                if distance != 0.0 or yaw != 0.0
            ),
            key=lambda value: (
                # A fore/aft foothold correction preserves the authored foot
                # orientation and is much less invasive than rotating it.
                # Exhaust that one-dimensional solution before allowing yaw;
                # otherwise a cheap 10-degree yaw wins over a centimetre of
                # translation and can change discontinuously between frames.
                abs(value[1]) > 1.0e-12,
                abs(value[0]),
                abs(value[1]),
                value[0],
                value[1],
            ),
        )
        for signed_distance, yaw_adjustment in candidates:
            shifted = transformed(
                support, signed_distance, yaw_adjustment
            )
            shifted_envelope = transformed(
                envelope, signed_distance, yaw_adjustment
            )
            support_clearances = point_vertical_clearances(shifted)
            envelope_clearances = point_vertical_clearances(
                shifted_envelope
            )
            collision_free = (
                float(np.min(support_clearances))
                >= -scaffold_penetration_budget
                and float(np.min(envelope_clearances))
                >= -scaffold_penetration_budget
                and has_support(support_clearances)
            )
            if collision_free:
                # Stay a small distance back from the vertical edge.  The
                # ray-cast samples certify the sole points; this margin
                # also covers the continuous rendered foot hull between
                # those samples without lifting it off the tread.
                margin_distance = (
                    0.0
                    if signed_distance == 0.0
                    else math.copysign(
                        min(
                            maximum_foothold_shift,
                            abs(signed_distance)
                            + foothold_edge_margin,
                        ),
                        signed_distance,
                    )
                )
                if margin_distance != signed_distance:
                    margin_support = transformed(
                        support,
                        margin_distance,
                        yaw_adjustment,
                    )
                    margin_envelope = transformed(
                        envelope,
                        margin_distance,
                        yaw_adjustment,
                    )
                    margin_support_clearance = (
                        point_vertical_clearances(margin_support)
                    )
                    margin_envelope_clearance = (
                        point_vertical_clearances(margin_envelope)
                    )
                    margin_valid = (
                        float(np.min(margin_support_clearance))
                        >= -scaffold_penetration_budget
                        and float(np.min(margin_envelope_clearance))
                        >= -scaffold_penetration_budget
                        and has_support(margin_support_clearance)
                    )
                    if margin_valid:
                        return float(margin_distance), float(
                            yaw_adjustment
                        )
                return float(signed_distance), float(yaw_adjustment)
        return 0.0, 0.0

    def support_point_count(
        mesh: TerrainMeshIndex,
        ray_height: float,
        points: np.ndarray,
    ) -> int:
        """Count sole-bottom samples touching mesh or the explicit ground plane."""

        count = 0
        for point in np.asarray(points, dtype=np.float64):
            hit = mesh.raycast(
                np.asarray(
                    (point[0], point[1], ray_height),
                    dtype=np.float64,
                ),
                np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
            )
            surface_height = (
                ground_fallback_height_m
                if hit is None
                else float(hit.position_world[2])
            )
            if (
                abs(float(point[2]) - surface_height)
                <= support_contact_tolerance_m
            ):
                count += 1
        return count

    # Source contact labels must be known before foothold placement.  Inferring
    # stance from one coincident target point mislabels a swing sole while it
    # crosses a riser and prevents the collision-free fore/aft shift.
    source_support_by_frame = [
        adapter.sole_support_points_for_pose(
            root_position=source_positions[frame],
            root_quaternion_wxyz=source_quaternions[frame],
            joints=source_joints[frame],
        )
        for frame in range(len(source_joints))
    ]
    source_support = np.asarray(source_support_by_frame, dtype=np.float64)
    source_sole_centres = np.mean(source_support, axis=2)
    source_sole_speed = np.linalg.norm(
        np.gradient(source_sole_centres, axis=0)
        * float(archive["fps"][0]),
        axis=2,
    )
    source_ray_origin_height = (
        float(np.max(source_mesh.vertices_world[:, 2])) + 1.0
    )
    source_support_counts = np.empty(
        (len(source_joints), 2), dtype=np.int16
    )
    for frame in range(len(source_joints)):
        for foot in range(2):
            source_support_counts[frame, foot] = support_point_count(
                source_mesh,
                source_ray_origin_height,
                source_support[frame, foot],
            )
    source_stance = (
        (source_support_counts >= 2)
        & (
            source_sole_speed
            <= float(support_maximum_sole_speed_m_s)
        )
    )

    # Construct the uncorrected geometry-only scaffold first.  Foothold
    # placement is a contact-phase decision, not a per-frame collision hack:
    # one planted foot receives one constant target, and its transform changes
    # along a quintic curve only while that foot is in swing.  Independent
    # frame-wise choices produced one-frame pelvis jumps on short treads.
    raw_target_soles_by_frame: list[tuple[np.ndarray, np.ndarray]] = []
    raw_target_envelopes_by_frame: list[
        tuple[np.ndarray, np.ndarray]
    ] = []
    for frame in range(len(source_joints)):
        source_soles = adapter.sole_positions_for_pose(
            root_position=source_positions[frame],
            root_quaternion_wxyz=source_quaternions[frame],
            joints=source_joints[frame],
        )
        target_soles = tuple(
            warp.warp_rigid_point_cloud(sole, smooth_height=True)
            for sole in source_soles
        )
        source_envelopes = adapter.foot_collision_envelope_points_for_pose(
            root_position=source_positions[frame],
            root_quaternion_wxyz=source_quaternions[frame],
            joints=source_joints[frame],
        )
        target_envelopes = tuple(
            warp.warp_rigid_point_cloud(envelope, smooth_height=True)
            for envelope in source_envelopes
        )
        raw_target_soles_by_frame.append(
            (target_soles[0], target_soles[1])
        )
        raw_target_envelopes_by_frame.append(
            (target_envelopes[0], target_envelopes[1])
        )

    frame_count = len(source_joints)
    foothold_shift = np.zeros((frame_count, 2), dtype=np.float64)
    foothold_yaw = np.zeros((frame_count, 2), dtype=np.float64)

    def stance_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return stable [start, stop) contacts, closing tiny label gaps."""

        values = np.asarray(mask, dtype=bool).copy()
        false_runs: list[tuple[int, int]] = []
        start = None
        for index, active in enumerate(values):
            if not active and start is None:
                start = index
            if active and start is not None:
                false_runs.append((start, index))
                start = None
        if start is not None:
            false_runs.append((start, len(values)))
        # Contact-speed estimates can flicker for a few samples.  A genuine G1
        # swing in this 50 Hz bank is substantially longer than four frames.
        for gap_start, gap_stop in false_runs:
            if (
                gap_start > 0
                and gap_stop < len(values)
                and gap_stop - gap_start <= 4
            ):
                values[gap_start:gap_stop] = True
        intervals: list[tuple[int, int]] = []
        start = None
        for index, active in enumerate(values):
            if active and start is None:
                start = index
            if not active and start is not None:
                intervals.append((start, index))
                start = None
        if start is not None:
            intervals.append((start, len(values)))
        return intervals

    def quintic(value: np.ndarray) -> np.ndarray:
        return value**3 * (10.0 + value * (-15.0 + 6.0 * value))

    for foot in range(2):
        intervals = stance_intervals(source_stance[:, foot])
        if not intervals:
            intervals = [(0, frame_count)]
        transforms: list[tuple[float, float]] = []
        for start, stop in intervals:
            soles = np.concatenate(
                [
                    raw_target_soles_by_frame[frame][foot]
                    for frame in range(start, stop)
                ],
                axis=0,
            )
            envelopes = np.concatenate(
                [
                    raw_target_envelopes_by_frame[frame][foot]
                    for frame in range(start, stop)
                ],
                axis=0,
            )
            radii = np.tile(sphere_radii[foot], stop - start)
            transform = clearance_safe_foothold_transform(
                soles,
                radii,
                envelopes,
                support_group_size=len(sphere_radii[foot]),
            )
            transforms.append(transform)
            foothold_shift[start:stop, foot] = transform[0]
            foothold_yaw[start:stop, foot] = transform[1]
        first_start = intervals[0][0]
        foothold_shift[:first_start, foot] = transforms[0][0]
        foothold_yaw[:first_start, foot] = transforms[0][1]
        for interval_index in range(len(intervals) - 1):
            left_stop = intervals[interval_index][1]
            right_start = intervals[interval_index + 1][0]
            gap = right_start - left_stop
            if gap <= 0:
                continue
            blend = quintic(
                np.arange(1, gap + 1, dtype=np.float64) / (gap + 1.0)
            )
            left = transforms[interval_index]
            right = transforms[interval_index + 1]
            foothold_shift[left_stop:right_start, foot] = (
                left[0] + blend * (right[0] - left[0])
            )
            foothold_yaw[left_stop:right_start, foot] = (
                left[1] + blend * (right[1] - left[1])
            )
        last_stop = intervals[-1][1]
        foothold_shift[last_stop:, foot] = transforms[-1][0]
        foothold_yaw[last_stop:, foot] = transforms[-1][1]

    # Apply the phase-consistent transforms.  Any remaining tiny penetration
    # is lifted in both this scaffold and the root before IK, so the residual
    # reported below describes the returned motion.
    target_soles_by_frame: list[tuple[np.ndarray, np.ndarray]] = []
    foothold_shifts_by_frame: list[tuple[float, float]] = []
    foothold_yaws_by_frame: list[tuple[float, float]] = []
    scaffold_clearances = np.empty(frame_count, dtype=np.float64)
    for frame in range(frame_count):
        target_soles = raw_target_soles_by_frame[frame]
        foothold_shifts = tuple(float(value) for value in foothold_shift[frame])
        foothold_yaws = tuple(float(value) for value in foothold_yaw[frame])
        foothold_shifts_by_frame.append(foothold_shifts)
        foothold_yaws_by_frame.append(foothold_yaws)
        transformed_soles = []
        for foot in range(2):
            sole = np.asarray(target_soles[foot], dtype=np.float64).copy()
            pivot = np.mean(sole, axis=0)
            cosine = math.cos(foothold_yaws[foot])
            sine = math.sin(foothold_yaws[foot])
            rotation = np.asarray(
                ((cosine, -sine), (sine, cosine)), dtype=np.float64
            )
            sole[:, :2] = (
                (sole[:, :2] - pivot[None, :2]) @ rotation.T
                + pivot[None, :2]
                + foothold_shifts[foot]
                * target_progress_direction[None]
            )
            transformed_soles.append(sole)
        target_soles = (transformed_soles[0], transformed_soles[1])
        # Mapping the pelvis independently can raise it onto the next tread
        # before the trailing foot, or compress a leading swing on a short top
        # tread.  Centre its translation between the two warped feet so route
        # deformation cannot tear the skeleton apart.  This uses only source
        # FK and target geometry; no target pose sample participates.
        provisional_soles = adapter.sole_positions_for_pose(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            joints=source_joints[frame],
        )
        foot_shifts = np.stack(
            [
                np.mean(target_soles[foot], axis=0)
                - np.mean(provisional_soles[foot], axis=0)
                for foot in range(2)
            ]
        )
        pelvis_shift = np.mean(foot_shifts, axis=0)
        warped_root[frame] += pelvis_shift
        target_soles_by_frame.append(target_soles)
        target_support = tuple(
            target_soles[foot]
            - np.asarray((0.0, 0.0, 1.0))[None]
            * sphere_radii[foot][:, None]
            for foot in range(2)
        )
        scaffold_clearances[frame] = vertical_clearance(target_support)

    required_lift = np.maximum(
        0.0, -scaffold_penetration_budget - scaffold_clearances
    )
    clearance_lift = np.zeros_like(required_lift)
    active_frames = np.flatnonzero(required_lift > 0.0)
    frame_indices = np.arange(len(required_lift), dtype=np.float64)
    for active in active_frames:
        clearance_lift = np.maximum(
            clearance_lift,
            required_lift[active]
            * np.exp(-np.abs(frame_indices - active) / 4.0),
        )
    if float(np.max(clearance_lift)) > float(
        maximum_root_clearance_lift_m
    ):
        worst_frame = int(np.argmax(clearance_lift))
        worst_support = tuple(
            target_soles_by_frame[worst_frame][foot]
            - np.asarray((0.0, 0.0, 1.0))[None]
            * sphere_radii[foot][:, None]
            for foot in range(2)
        )
        worst_clearance = tuple(
            float(np.min(point_vertical_clearances(points)))
            for points in worst_support
        )
        raise ValueError(
            "geometry warp needs excessive root clearance lift: "
            f"{float(np.max(clearance_lift)):.6f} m at frame "
            f"{worst_frame}; foot clearances={worst_clearance}, "
            f"progress shifts={foothold_shifts_by_frame[worst_frame]}, "
            f"yaw adjustments={foothold_yaws_by_frame[worst_frame]}"
        )
    warped_root[:, 2] += clearance_lift
    for frame, lift in enumerate(clearance_lift):
        if lift > 0.0:
            target_soles_by_frame[frame] = tuple(
                sole + np.asarray((0.0, 0.0, lift))[None]
                for sole in target_soles_by_frame[frame]
            )  # type: ignore[assignment]

    minimum_clearances = np.empty(len(source_joints), dtype=np.float64)
    triangle_penetrations = np.empty(len(source_joints), dtype=np.float64)
    stance_support_counts = np.full(
        (len(source_joints), 2), -1, dtype=np.int16
    )
    unsupported_stance: list[tuple[int, int, int]] = []
    mesh_query = CanonicalMeshQuery(
        target_mesh.mesh, target_mesh.world_from_terrain
    )
    for frame in range(len(source_joints)):
        target_soles = target_soles_by_frame[frame]
        (
            adapted_joints[frame],
            corrections[frame],
            errors[frame],
        ) = adapter.adapt_to_targets(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            authored_joints=source_joints[frame],
            sole_targets_world=target_soles,
        )
        final_centres = adapter.sole_positions_for_pose(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            joints=adapted_joints[frame],
        )
        support_points = adapter.sole_support_points_for_pose(
            root_position=warped_root[frame],
            root_quaternion_wxyz=warped_quaternions[frame],
            joints=adapted_joints[frame],
        )
        minimum_clearances[frame] = vertical_clearance(support_points)
        for foot in range(2):
            if not source_stance[frame, foot]:
                continue
            count = support_point_count(
                target_mesh,
                ray_origin_height,
                support_points[foot],
            )
            stance_support_counts[frame, foot] = count
            if count < minimum_support:
                unsupported_stance.append((frame, foot, count))
        centres = np.concatenate(final_centres)
        radii = np.concatenate(sphere_radii)
        distances = np.asarray(
            mesh_query.query(centres).distance_m, dtype=np.float64
        )
        triangle_penetrations[frame] = float(
            np.max(np.maximum(0.0, radii - distances))
        )

    maximum_correction = float(np.max(corrections))
    maximum_error = float(np.max(errors))
    minimum_clearance = float(np.min(minimum_clearances))
    maximum_penetration = max(0.0, -minimum_clearance)
    maximum_triangle_penetration = float(np.max(triangle_penetrations))
    advertised_stance_counts = stance_support_counts[
        stance_support_counts >= 0
    ]
    minimum_stance_count = int(np.min(advertised_stance_counts))
    route_adjustment = np.linalg.norm(
        warped_root - requested_root, axis=1
    )
    target_start = np.asarray(target_route.start_xy, dtype=np.float64)
    target_lateral = np.asarray(
        (-warp.target_direction_xy[1], warp.target_direction_xy[0])
    )
    route_lateral_deviation = np.abs(
        (warped_root[:, :2] - target_start) @ target_lateral
    )
    if maximum_correction > maximum_joint_correction_rad + 1.0e-6:
        raise ValueError(
            "geometry warp exceeds joint-correction bound: "
            f"{maximum_correction:.6f} rad"
        )
    if maximum_error > maximum_foot_target_error_m:
        worst_error_frame = int(np.argmax(errors))
        raise ValueError(
            "geometry warp misses foot target: "
            f"{maximum_error:.6f} m at frame {worst_error_frame}; "
            "progress shifts="
            f"{foothold_shifts_by_frame[worst_error_frame]}, "
            "yaw adjustments="
            f"{foothold_yaws_by_frame[worst_error_frame]}"
        )
    if maximum_penetration > maximum_sole_penetration_m + 1.0e-6:
        raise ValueError(
            "geometry warp exceeds vertical sole-penetration bound: "
            f"{maximum_penetration:.6f} m"
        )
    if (
        maximum_triangle_penetration
        > maximum_sole_penetration_m + 1.0e-6
    ):
        raise ValueError(
            "geometry warp intersects a target mesh edge or riser: "
            f"{maximum_triangle_penetration:.6f} m"
        )
    if unsupported_stance:
        frame, foot, count = unsupported_stance[0]
        side = "left" if foot == 0 else "right"
        raise ValueError(
            "geometry warp leaves an unsupported stance foot: "
            f"frame {frame}, {side}, {count}/{minimum_support} "
            "required sole support points; progress shift="
            f"{foothold_shifts_by_frame[frame][foot]:.6f} m, yaw="
            f"{math.degrees(foothold_yaws_by_frame[frame][foot]):.3f} deg, "
            f"source support={int(source_support_counts[frame, foot])}, "
            f"source sole speed={float(source_sole_speed[frame, foot]):.4f} m/s"
        )

    clip_id = str(archive["clip_names"][source_index])
    provenance = tuple(
        FrameProvenance(source_index, frame, clip_id)
        for frame in range(frame_start, frame_stop)
    )
    motion = StitchedMotion(
        fps=float(archive["fps"][0]),
        root_position_world=np.asarray(warped_root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            warped_quaternions, dtype=np.float32
        ),
        joint_position=np.asarray(adapted_joints, dtype=np.float32),
        provenance=provenance,
        seam_indices=(),
    )
    return StairGeometryWarpResult(
        motion=motion,
        source_route=source_route,
        target_route=target_route,
        maximum_joint_correction_rad=maximum_correction,
        maximum_foot_target_error_m=maximum_error,
        maximum_sole_penetration_m=maximum_penetration,
        maximum_triangle_sphere_penetration_m=(
            maximum_triangle_penetration
        ),
        minimum_sole_clearance_m=minimum_clearance,
        maximum_root_clearance_lift_m=float(np.max(clearance_lift)),
        maximum_root_route_adjustment_m=float(
            np.max(route_adjustment)
        ),
        maximum_root_route_lateral_deviation_m=float(
            np.max(route_lateral_deviation)
        ),
        minimum_stance_support_point_count=minimum_stance_count,
        per_frame_joint_correction_rad=np.asarray(
            corrections, dtype=np.float32
        ),
        per_frame_foot_target_error_m=np.asarray(
            errors, dtype=np.float32
        ),
        per_frame_minimum_sole_clearance_m=np.asarray(
            minimum_clearances, dtype=np.float32
        ),
        per_frame_triangle_sphere_penetration_m=np.asarray(
            triangle_penetrations, dtype=np.float32
        ),
        per_frame_root_clearance_lift_m=np.asarray(
            clearance_lift, dtype=np.float32
        ),
        per_frame_stance_support_point_count=np.asarray(
            stance_support_counts, dtype=np.int16
        ),
        maximum_foothold_progress_shift_m=float(
            np.max(np.abs(np.asarray(foothold_shifts_by_frame)))
        ),
        per_frame_foothold_progress_shift_m=np.asarray(
            foothold_shifts_by_frame, dtype=np.float32
        ),
        maximum_foothold_yaw_adjustment_rad=float(
            np.max(np.abs(np.asarray(foothold_yaws_by_frame)))
        ),
        per_frame_foothold_yaw_adjustment_rad=np.asarray(
            foothold_yaws_by_frame, dtype=np.float32
        ),
    )


def select_best_geometry_compatible_clip(
    archive_path: str | Path,
    *,
    target_clip_index: int,
    target_route_start_xy: object,
    target_route_end_xy: object,
    model_path: str | Path,
    candidate_limit: int = 8,
    maximum_joint_correction_rad: float = 0.36,
    maximum_foot_target_error_m: float = 0.001,
    maximum_sole_penetration_m: float = 0.0025,
    maximum_root_clearance_lift_m: float = 0.005,
    minimum_stance_support_points: int = 1,
    geometry_ranking: str = "nominal",
) -> AutomaticStairGeometryWarpSelection:
    """Retrieve and mechanically rank full clips without source clip rules.

    Archive stair metadata is only a cheap prefilter.  Every retained candidate
    still has to pass exact target-mesh route sampling, bounded IK, vertical
    support, and sphere-to-triangle collision checks.
    """

    import zarr

    limit = int(candidate_limit)
    if limit <= 0:
        raise ValueError("candidate_limit must be positive")
    archive = zarr.open_group(str(Path(archive_path)), mode="r")
    target = int(target_clip_index)
    proposals = rank_geometry_compatible_sources(
        archive, target, ranking=geometry_ranking
    )

    feasible: list[
        tuple[float, int, StairGeometryWarpResult]
    ] = []
    rejected = 0
    evaluated = 0
    for cheap_cost, source in proposals[:limit]:
        evaluated += 1
        source_frames = int(archive["clip_end_idx"][source]) - int(
            archive["clip_start_idx"][source]
        )
        try:
            result = warp_archive_clip_to_stair_geometry(
                archive_path,
                source_clip_index=source,
                source_start_frame=0,
                source_stop_frame=source_frames,
                target_clip_index=target,
                target_route_start_xy=target_route_start_xy,
                target_route_end_xy=target_route_end_xy,
                model_path=model_path,
                maximum_joint_correction_rad=(
                    maximum_joint_correction_rad
                ),
                maximum_foot_target_error_m=(
                    maximum_foot_target_error_m
                ),
                maximum_sole_penetration_m=(
                    maximum_sole_penetration_m
                ),
                maximum_root_clearance_lift_m=(
                    maximum_root_clearance_lift_m
                ),
                minimum_stance_support_points=(
                    minimum_stance_support_points
                ),
            )
        except ValueError:
            rejected += 1
            continue
        score = (
            float(cheap_cost)
            + result.maximum_joint_correction_rad
            + 10.0 * result.maximum_foot_target_error_m
            + 2.0 * result.maximum_root_route_adjustment_m
            + 5.0 * result.maximum_triangle_sphere_penetration_m
        )
        feasible.append((score, source, result))
    if not feasible:
        raise ValueError(
            "no geometry-compatible source clip met the mechanical bounds"
        )
    score, source, result = min(
        feasible, key=lambda item: (item[0], item[1])
    )
    return AutomaticStairGeometryWarpSelection(
        result=result,
        source_clip_index=source,
        target_clip_index=target,
        score=float(score),
        evaluated_candidate_count=evaluated,
        rejected_candidate_count=rejected,
    )


__all__ = (
    "AutomaticStairGeometryWarpSelection",
    "DEFAULT_MOTION_ROUTE_PROFILE_CATALOG",
    "StairGeometryWarp",
    "StairGeometryWarpResult",
    "build_motion_route_profile_catalog",
    "build_stair_geometry_warp",
    "motion_conditioned_stair_route_profile",
    "motion_conditioned_stair_support_route",
    "rank_geometry_compatible_sources",
    "select_best_geometry_compatible_clip",
    "warp_archive_clip_to_stair_geometry",
)
