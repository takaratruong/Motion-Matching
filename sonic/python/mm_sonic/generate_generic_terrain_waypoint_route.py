"""Compile a globally known waypoint route into exact-safe terrain events.

The single-segment generic route generator remains the authority for terrain
classification, motion retrieval, geometry warping, and collision rejection.
This module only removes the straight-route restriction: each polyline leg is
generated independently, flat legs remain under live MotionBricks control,
and accepted non-flat events are collected into one runtime portal manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .generate_generic_stair_route import load_target_mesh
from .generate_generic_terrain_route import generate as generate_segment
from .generate_generic_terrain_route import _profile_dict
from .render_stitched_motion import load_stitched_motion_npz
from .terrain_oracle.math3d import quaternion_multiply_wxyz
from .terrain_oracle.stitch import StitchedMotion
from .terrain_oracle.terrain_route_profile import sample_terrain_route_profile


_PROFILE_COMPARISON_SAMPLES = 257
_MAXIMUM_REUSE_LENGTH_ERROR_M = 0.025
_MAXIMUM_REUSE_HEIGHT_ERROR_M = 0.003
_CACHE_PROFILE_SPACING_M = 0.10


def _compact_result(result: object) -> dict[str, object]:
    if not isinstance(result, dict):
        raise ValueError("generated route event has no result object")
    return {
        key: result[key]
        for key in ("status", "backend", "motion", "transition_count")
        if key in result
    }


def _segment_events(summary: dict[str, object]) -> list[dict[str, object]]:
    result = summary.get("result")
    if not isinstance(result, dict):
        raise ValueError("generated route segment has no result object")
    nested = result.get("events")
    if nested is not None:
        if not isinstance(nested, list):
            raise ValueError("generated route events must be a list")
        output = []
        for event in nested:
            if not isinstance(event, dict):
                raise ValueError("generated route event must be an object")
            output.append({**event, "result": _compact_result(event.get("result"))})
        return output
    if result.get("motion") is None:
        return []
    profile = summary.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("generated route segment has no profile object")
    return [
        {
            "event_index": 0,
            "kind": str(profile["kind"]),
            "backend": result.get("backend"),
            "start_distance_m": 0.0,
            "end_distance_m": float(profile["route_length_m"]),
            "start_xy": list(summary["start_xy"]),
            "end_xy": list(summary["end_xy"]),
            "result": _compact_result(result),
        }
    ]


def _wrapped_turn(previous: np.ndarray, current: np.ndarray) -> float:
    first = math.atan2(float(previous[1]), float(previous[0]))
    second = math.atan2(float(current[1]), float(current[0]))
    return math.atan2(math.sin(second - first), math.cos(second - first))


def _summary_matches_request(
    summary: dict[str, object],
    *,
    terrain_usd: Path,
    terrain_position: tuple[float, float, float],
    terrain_quaternion_wxyz: tuple[float, float, float, float],
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> bool:
    """Return whether an accepted on-disk leg is safe to resume."""

    try:
        return bool(
            summary.get("status") == "accepted"
            and Path(str(summary["terrain_usd"])).expanduser().resolve()
            == terrain_usd.expanduser().resolve()
            and np.allclose(
                summary["terrain_position_world"], terrain_position, atol=1.0e-8
            )
            and np.allclose(
                summary["terrain_quaternion_world_from_usd_wxyz"],
                terrain_quaternion_wxyz,
                atol=1.0e-8,
            )
            and np.allclose(summary["start_xy"], start_xy, atol=1.0e-6)
            and np.allclose(summary["end_xy"], end_xy, atol=1.0e-6)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _load_resumable_segment(
    segment_dir: Path,
    **request: object,
) -> dict[str, object] | None:
    summary_path = segment_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(summary, dict) or not _summary_matches_request(
        summary, **request
    ):
        return None
    return summary


def _profiles_equivalent(source: object, target: object) -> tuple[bool, float]:
    """Compare relative support-height signatures, independent of placement."""

    if str(source.kind) != str(target.kind):
        return False, math.inf
    source_length = float(source.distance_m[-1])
    target_length = float(target.distance_m[-1])
    if abs(source_length - target_length) > _MAXIMUM_REUSE_LENGTH_ERROR_M:
        return False, math.inf
    unit_distance = np.linspace(0.0, 1.0, _PROFILE_COMPARISON_SAMPLES)
    source_height = np.interp(
        unit_distance,
        np.asarray(source.distance_m, dtype=np.float64) / source_length,
        np.asarray(source.height_m, dtype=np.float64),
    )
    target_height = np.interp(
        unit_distance,
        np.asarray(target.distance_m, dtype=np.float64) / target_length,
        np.asarray(target.height_m, dtype=np.float64),
    )
    source_height -= source_height[0]
    target_height -= target_height[0]
    error = float(np.max(np.abs(source_height - target_height)))
    return error <= _MAXIMUM_REUSE_HEIGHT_ERROR_M, error


def _rigidly_place_motion(
    motion: StitchedMotion,
    *,
    source_start_xy: object,
    source_end_xy: object,
    target_start_xy: object,
    target_end_xy: object,
    vertical_offset_m: float,
) -> tuple[StitchedMotion, float]:
    """Rigidly instantiate one primitive on a route-equivalent portal."""

    source_start = np.asarray(source_start_xy, dtype=np.float64)
    source_end = np.asarray(source_end_xy, dtype=np.float64)
    target_start = np.asarray(target_start_xy, dtype=np.float64)
    target_end = np.asarray(target_end_xy, dtype=np.float64)
    source_vector = source_end - source_start
    target_vector = target_end - target_start
    source_yaw = math.atan2(float(source_vector[1]), float(source_vector[0]))
    target_yaw = math.atan2(float(target_vector[1]), float(target_vector[0]))
    yaw_delta = math.atan2(
        math.sin(target_yaw - source_yaw), math.cos(target_yaw - source_yaw)
    )
    cosine = math.cos(yaw_delta)
    sine = math.sin(yaw_delta)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)
    root = np.asarray(motion.root_position_world, dtype=np.float64).copy()
    root[:, :2] = (
        (root[:, :2] - source_start[None]) @ rotation.T
        + target_start[None]
    )
    root[:, 2] += float(vertical_offset_m)
    yaw_quaternion = np.asarray(
        (math.cos(0.5 * yaw_delta), 0.0, 0.0, math.sin(0.5 * yaw_delta)),
        dtype=np.float64,
    )
    quaternion = quaternion_multiply_wxyz(
        np.broadcast_to(yaw_quaternion, motion.root_quaternion_world_wxyz.shape),
        motion.root_quaternion_world_wxyz,
    )
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    return (
        StitchedMotion(
            fps=motion.fps,
            root_position_world=np.asarray(root, dtype=np.float32),
            root_quaternion_world_wxyz=np.asarray(quaternion, dtype=np.float32),
            joint_position=motion.joint_position,
            provenance=motion.provenance,
            seam_indices=motion.seam_indices,
        ),
        yaw_delta,
    )


def _save_motion(path: Path, motion: StitchedMotion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=np.asarray(motion.fps, dtype=np.float32),
        root_position_world=np.asarray(motion.root_position_world, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz, dtype=np.float32
        ),
        joint_position=np.asarray(motion.joint_position, dtype=np.float32),
        seam_indices=np.asarray(motion.seam_indices, dtype=np.int64),
        source_archive_clip_index=np.asarray(
            [value.archive_clip_index for value in motion.provenance], dtype=np.int64
        ),
        source_frame=np.asarray(
            [value.source_frame for value in motion.provenance], dtype=np.int64
        ),
        source_clip_id=np.asarray(
            [value.clip_id for value in motion.provenance], dtype=np.str_
        ),
    )


def _cache_events(summary: dict[str, object]) -> list[dict[str, object]]:
    result = summary.get("result")
    if not isinstance(result, dict):
        return []
    events = result.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    if result.get("motion") is None:
        return []
    return [
        {
            "kind": summary.get("profile", {}).get("kind"),
            "backend": result.get("backend"),
            "start_xy": summary.get("start_xy"),
            "end_xy": summary.get("end_xy"),
            "result": result,
        }
    ]


def _try_cached_primitive(
    *,
    cache_summaries: tuple[Path, ...],
    target_profile: object,
    target_start_xy: np.ndarray,
    target_end_xy: np.ndarray,
    segment_dir: Path,
) -> dict[str, object] | None:
    """Find and rigidly place an accepted route-equivalent primitive.

    The final portal composer performs the expensive exact collision audit on
    the assembled course.  Repeating it here would audit the same candidate
    twice and can itself exceed the route compiler's runtime budget.
    """

    for cache_path in cache_summaries:
        resolved_cache = cache_path.expanduser().resolve()
        try:
            cached_summary = json.loads(resolved_cache.read_text())
            if cached_summary.get("status") != "accepted":
                continue
            source_mesh = load_target_mesh(
                Path(str(cached_summary["terrain_usd"])),
                position_world=tuple(cached_summary["terrain_position_world"]),
                quaternion_world_from_usd_wxyz=tuple(
                    cached_summary[
                        "terrain_quaternion_world_from_usd_wxyz"
                    ]
                ),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        for event in _cache_events(cached_summary):
            result = event.get("result")
            if not isinstance(result, dict) or result.get("status") != "accepted":
                continue
            if result.get("motion") is None:
                continue
            try:
                source_start = np.asarray(event["start_xy"], dtype=np.float64)
                source_end = np.asarray(event["end_xy"], dtype=np.float64)
                source_profile = sample_terrain_route_profile(
                    source_mesh,
                    source_start,
                    source_end,
                    sample_spacing_m=_CACHE_PROFILE_SPACING_M,
                )
                equivalent, height_error = _profiles_equivalent(
                    source_profile, target_profile
                )
                if not equivalent:
                    continue
                source_motion = Path(str(result["motion"])).expanduser().resolve()
                motion = load_stitched_motion_npz(source_motion)
                placed, yaw_delta = _rigidly_place_motion(
                    motion,
                    source_start_xy=source_start,
                    source_end_xy=source_end,
                    target_start_xy=target_start_xy,
                    target_end_xy=target_end_xy,
                    vertical_offset_m=(
                        float(target_profile.height_m[0])
                        - float(source_profile.height_m[0])
                    ),
                )
                backend = str(event.get("backend") or result.get("backend") or "")
                motion_path = (segment_dir / "cached_rigid" / "motion.npz").resolve()
                _save_motion(motion_path, placed)
                result_summary = {
                    "status": "accepted",
                    "backend": backend,
                    "motion": str(motion_path),
                    "transition_count": int(result.get("transition_count", 0)),
                    "cache_reuse": {
                        "source_summary": str(resolved_cache),
                        "source_motion": str(source_motion),
                        "height_signature_error_m": height_error,
                        "yaw_delta_rad": yaw_delta,
                        "source_was_accepted": True,
                        "exact_collision_audit": "deferred_to_portal_composer",
                    },
                }
                summary = {
                    "schema": "generic-terrain-route/v1",
                    "terrain_usd": None,
                    "terrain_position_world": None,
                    "terrain_quaternion_world_from_usd_wxyz": None,
                    "start_xy": target_start_xy.tolist(),
                    "end_xy": target_end_xy.tolist(),
                    "profile": _profile_dict(target_profile),
                    "result": result_summary,
                    "status": "accepted",
                }
                return summary
            except (KeyError, OSError, TypeError, ValueError):
                continue
    return None


def generate(
    *,
    terrain_usd: Path,
    terrain_position: tuple[float, float, float],
    terrain_quaternion_wxyz: tuple[float, float, float, float],
    waypoints_xy: Iterable[tuple[float, float]],
    output_dir: Path,
    primitive_cache_summaries: Iterable[Path] = (),
) -> dict[str, object]:
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    points = np.asarray(tuple(waypoints_xy), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("waypoint route needs at least two finite XY points")
    if not np.all(np.isfinite(points)):
        raise ValueError("waypoint route must be finite")
    legs = np.diff(points, axis=0)
    lengths = np.linalg.norm(legs, axis=1)
    if np.any(lengths < 0.20):
        raise ValueError("every waypoint leg must be at least 0.20 m")

    turns = [
        _wrapped_turn(legs[index - 1], legs[index])
        for index in range(1, len(legs))
    ]
    segment_rows: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    distance_offset = 0.0
    resolved_terrain = terrain_usd.expanduser().resolve()
    cache_summaries = tuple(
        path.expanduser().resolve() for path in primitive_cache_summaries
    )
    target_mesh = None
    for segment_index, (start, stop, length) in enumerate(
        zip(points[:-1], points[1:], lengths, strict=True)
    ):
        segment_dir = destination / f"segment_{segment_index:02d}"
        request = {
            "terrain_usd": resolved_terrain,
            "terrain_position": terrain_position,
            "terrain_quaternion_wxyz": terrain_quaternion_wxyz,
            "start_xy": start,
            "end_xy": stop,
        }
        summary = _load_resumable_segment(segment_dir, **request)
        if summary is None and cache_summaries:
            if target_mesh is None:
                target_mesh = load_target_mesh(
                    resolved_terrain,
                    position_world=terrain_position,
                    quaternion_world_from_usd_wxyz=terrain_quaternion_wxyz,
                )
            target_profile = sample_terrain_route_profile(
                target_mesh,
                start,
                stop,
                sample_spacing_m=_CACHE_PROFILE_SPACING_M,
            )
            if str(target_profile.kind) == "flat":
                summary = {
                    "schema": "generic-terrain-route/v1",
                    "terrain_usd": str(resolved_terrain),
                    "terrain_position_world": list(terrain_position),
                    "terrain_quaternion_world_from_usd_wxyz": list(
                        terrain_quaternion_wxyz
                    ),
                    "start_xy": start.tolist(),
                    "end_xy": stop.tolist(),
                    "profile": _profile_dict(target_profile),
                    "result": {
                        "status": "accepted",
                        "backend": "live_motionbricks_flat",
                        "motion": None,
                        "note": (
                            "flat route remains under the live MotionBricks "
                            "controller"
                        ),
                    },
                    "status": "accepted",
                }
                segment_dir.mkdir(parents=True, exist_ok=True)
                (segment_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n"
                )
            else:
                summary = _try_cached_primitive(
                    cache_summaries=cache_summaries,
                    target_profile=target_profile,
                    target_start_xy=start,
                    target_end_xy=stop,
                    segment_dir=segment_dir,
                )
                if summary is not None:
                    summary.update(
                        {
                            "terrain_usd": str(resolved_terrain),
                            "terrain_position_world": list(terrain_position),
                            "terrain_quaternion_world_from_usd_wxyz": list(
                                terrain_quaternion_wxyz
                            ),
                        }
                    )
                    segment_dir.mkdir(parents=True, exist_ok=True)
                    (segment_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2, sort_keys=True) + "\n"
                    )
        if summary is None:
            summary = generate_segment(
                terrain_usd=resolved_terrain,
                terrain_position=terrain_position,
                terrain_quaternion_wxyz=terrain_quaternion_wxyz,
                start_xy=(float(start[0]), float(start[1])),
                end_xy=(float(stop[0]), float(stop[1])),
                output_dir=segment_dir,
            )
        segment_rows.append(
            {
                "segment_index": segment_index,
                "start_xy": [float(value) for value in start],
                "end_xy": [float(value) for value in stop],
                "length_m": float(length),
                "status": summary["status"],
                "profile_kind": summary["profile"]["kind"],
                "summary": str(segment_dir / "summary.json"),
            }
        )
        for local_event in _segment_events(summary):
            event = dict(local_event)
            event["event_index"] = len(events)
            event["segment_index"] = segment_index
            event["segment_summary"] = str(segment_dir / "summary.json")
            event["start_distance_m"] = distance_offset + float(
                local_event["start_distance_m"]
            )
            event["end_distance_m"] = distance_offset + float(
                local_event["end_distance_m"]
            )
            events.append(event)
        distance_offset += float(length)

    accepted = all(row["status"] == "accepted" for row in segment_rows)
    result = {
        "status": "accepted" if accepted else "rejected",
        "backend": "live_motionbricks_with_terrain_portals",
        "motion": None,
        "events": events,
    }
    summary = {
        "schema": "generic-terrain-waypoint-route/v1",
        "status": result["status"],
        "terrain_usd": str(terrain_usd.expanduser().resolve()),
        "terrain_position_world": list(terrain_position),
        "terrain_quaternion_world_from_usd_wxyz": list(
            terrain_quaternion_wxyz
        ),
        "start_xy": [float(value) for value in points[0]],
        "end_xy": [float(value) for value in points[-1]],
        "waypoints_xy": points.tolist(),
        "profile": {
            "kind": "polyline",
            "route_length_m": float(np.sum(lengths)),
            "segment_count": int(len(lengths)),
            "turn_count": int(len(turns)),
            "maximum_absolute_turn_rad": float(
                np.max(np.abs(turns), initial=0.0)
            ),
            "signed_turns_rad": [float(value) for value in turns],
            "lateral_span_m": float(np.ptp(points[:, 1])),
        },
        "segments": segment_rows,
        "result": result,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-usd", type=Path, required=True)
    parser.add_argument("--terrain-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument(
        "--terrain-quaternion-wxyz", type=float, nargs=4, default=(1, 0, 0, 0)
    )
    parser.add_argument(
        "--waypoint-xy",
        type=float,
        nargs=2,
        action="append",
        required=True,
        help="Repeat in traversal order; at least two are required.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--primitive-cache-summary",
        type=Path,
        action="append",
        default=[],
        help=(
            "Accepted route summary to search for route-equivalent primitives; "
            "repeat to add caches."
        ),
    )
    arguments = parser.parse_args(argv)
    summary = generate(
        terrain_usd=arguments.terrain_usd,
        terrain_position=tuple(arguments.terrain_position),
        terrain_quaternion_wxyz=tuple(arguments.terrain_quaternion_wxyz),
        waypoints_xy=tuple(tuple(value) for value in arguments.waypoint_xy),
        output_dir=arguments.output_dir,
        primitive_cache_summaries=tuple(arguments.primitive_cache_summary),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
