"""Build terrain portals for backward travel with unchanged body facing.

An ordinary reverse traversal turns the character around, so it cannot serve
the two-stick command "walk backward while still facing uphill".  This builder
first composes an exact-audited forward traversal with a genuine MotionBricks
flat exit, then reverses that continuous animation.  The reversed exit is a
long flat approach, and the terrain frames retain exactly the same collision
geometry while travel reverses independently of facing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import zarr

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as DEFAULT_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .compose_generic_terrain_portals import (
    DEFAULT_DESCENDING_EXITS,
    DEFAULT_EXIT,
    _exit_candidates,
)
from .compose_coherent_block_plan import _mechanically_accepted
from .compose_motionbricks_terrain_course import compose
from .compose_privileged_stair_route import _maximum_steps
from .generate_generic_stair_route import load_target_mesh
from .gear_action import mujoco_to_isaaclab_joint_vector
from .motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    _slerp_wxyz,
)
from .motionbricks_global_terrain_viewer import (
    MAXIMUM_SUPPORT_FOOT_CLEARANCE_M,
    _minimum_sole_clearance_m,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion


MAXIMUM_PORTAL_ENDPOINT_CLEARANCE_M = 0.020


def _support_foot_clearance_audit(
    motion_path: Path,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: object,
    ray_origin_z: float,
    maximum_clearance_m: float = MAXIMUM_SUPPORT_FOOT_CLEARANCE_M,
) -> dict[str, object]:
    """Reject collision-free courses that actually hover above terrain."""

    course = MotionBricksTerrainCourse.load(motion_path)
    values = np.asarray(
        [
            _minimum_sole_clearance_m(
                qpos,
                sole_adapter=sole_adapter,
                terrain=terrain,  # type: ignore[arg-type]
                ray_origin_z=ray_origin_z,
            )
            for qpos in course.native_mujoco_qpos()
        ],
        dtype=np.float64,
    )
    exceedances = np.flatnonzero(values > float(maximum_clearance_m))
    grounded = np.flatnonzero(
        values <= MAXIMUM_PORTAL_ENDPOINT_CLEARANCE_M
    )
    endpoint_accepted = bool(
        len(grounded) > 0
        and grounded[0] == 0
        and grounded[-1] == len(values) - 1
    )
    return {
        "accepted": len(exceedances) == 0 and endpoint_accepted,
        "threshold_m": float(maximum_clearance_m),
        "endpoint_threshold_m": MAXIMUM_PORTAL_ENDPOINT_CLEARANCE_M,
        "endpoint_accepted": endpoint_accepted,
        "first_grounded_frame_index": (
            None if len(grounded) == 0 else int(grounded[0])
        ),
        "last_grounded_frame_index": (
            None if len(grounded) == 0 else int(grounded[-1])
        ),
        "minimum_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.percentile(values, 95.0)),
        "maximum_m": float(np.max(values)),
        "threshold_exceedance_frame_indices": exceedances.tolist(),
    }


def _trim_hovering_flat_margins(
    motion_path: Path,
    support_audit: dict[str, object],
    *,
    minimum_approach_s: float = 0.82,
    minimum_landing_s: float = 0.16,
) -> dict[str, object] | None:
    """Remove only unsupported outer flat tails, never terrain frames."""

    bad = np.asarray(
        support_audit["threshold_exceedance_frame_indices"], dtype=np.int64
    )
    endpoint_accepted = bool(support_audit.get("endpoint_accepted", True))
    if len(bad) == 0 and endpoint_accepted:
        return {
            "applied": False,
            "frame_range": None,
        }
    with np.load(motion_path.expanduser().resolve(), allow_pickle=False) as arrays:
        payload = {name: np.asarray(arrays[name]) for name in arrays.files}
    frame_count = len(np.asarray(payload["root_position_world"]))
    seams = tuple(int(value) for value in payload["seam_indices"])
    if len(seams) != 2:
        return None
    entry, landing = seams
    # Hover inside the authored non-flat traversal cannot be repaired by
    # shortening a reusable flat margin.
    if np.any((bad >= entry) & (bad < landing)):
        return None
    approach_bad = bad[bad < entry]
    landing_bad = bad[bad >= landing]
    start = 0 if len(approach_bad) == 0 else int(approach_bad[-1] + 1)
    stop = frame_count if len(landing_bad) == 0 else int(landing_bad[0])
    first_grounded = support_audit.get("first_grounded_frame_index")
    last_grounded = support_audit.get("last_grounded_frame_index")
    if first_grounded is None or last_grounded is None:
        return None
    if int(first_grounded) > 0:
        start = max(start, int(first_grounded))
    if int(last_grounded) < frame_count - 1:
        stop = min(stop, int(last_grounded) + 1)
    fps = float(np.asarray(payload["fps"]).item())
    minimum_approach_frames = int(math.ceil(float(minimum_approach_s) * fps))
    minimum_landing_frames = int(math.ceil(float(minimum_landing_s) * fps))
    if (
        entry - start < minimum_approach_frames
        or stop - landing < minimum_landing_frames
        or start >= stop
    ):
        return None
    for name, value in tuple(payload.items()):
        if name == "seam_indices":
            continue
        if value.ndim >= 1 and len(value) == frame_count:
            payload[name] = value[start:stop].copy()
    payload["seam_indices"] = np.asarray(
        (entry - start, landing - start), dtype=np.int64
    )
    np.savez_compressed(motion_path, **payload)
    return {
        "applied": bool(start > 0 or stop < frame_count),
        "source_frame_count": frame_count,
        "frame_range": [start, stop],
        "frame_count": stop - start,
        "seam_indices": [entry - start, landing - start],
        "approach_duration_s": float((entry - start) / fps),
        "landing_duration_s": float((stop - landing) / fps),
    }


def _reverse_motion(source: Path, destination: Path) -> dict[str, object]:
    """Reverse a complete approach/terrain/exit course without changing poses."""

    with np.load(source.expanduser().resolve(), allow_pickle=False) as arrays:
        payload = {name: np.asarray(arrays[name]) for name in arrays.files}
    root = np.asarray(payload["root_position_world"])
    frame_count = len(root)
    seams = tuple(int(value) for value in payload["seam_indices"])
    if len(seams) != 2:
        raise ValueError("backward-facing synthesis requires a two-seam course")
    for name, value in tuple(payload.items()):
        if name == "seam_indices":
            continue
        if value.ndim >= 1 and len(value) == frame_count:
            payload[name] = value[::-1].copy()
    reversed_seams = tuple(frame_count - value for value in reversed(seams))
    fps = float(np.asarray(payload["fps"]).item())
    # The reusable flat recordings are deliberately long and may curve far
    # away from the authored route or even leave a finite landing.  Runtime
    # commitment needs only a one-second approach and 0.8-second landing;
    # retaining the rest turns an otherwise valid portal into a global path.
    approach_frames = int(round(1.0 * fps))
    landing_frames = int(round(0.8 * fps))
    trim_start = max(0, reversed_seams[0] - approach_frames)
    trim_stop = min(frame_count, reversed_seams[-1] + landing_frames)
    for name, value in tuple(payload.items()):
        if name == "seam_indices":
            continue
        if value.ndim >= 1 and len(value) == frame_count:
            payload[name] = value[trim_start:trim_stop].copy()
    trimmed_seams = tuple(value - trim_start for value in reversed_seams)
    payload["seam_indices"] = np.asarray(trimmed_seams, dtype=np.int64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    course = MotionBricksTerrainCourse.load(destination)
    if course.entry_frame_index(0.80) <= 0:
        raise ValueError("reversed MotionBricks exit is too short for JIT capture")
    return {
        "source_motion": str(source.expanduser().resolve()),
        "motion": str(destination.resolve()),
        "source_frame_count": frame_count,
        "frame_count": course.frame_count,
        "source_seam_indices": list(seams),
        "untrimmed_reversed_seam_indices": list(reversed_seams),
        "trimmed_source_frame_range": [trim_start, trim_stop],
        "seam_indices": list(trimmed_seams),
        "flat_approach_frame_count": int(trimmed_seams[0]),
        "flat_approach_duration_s": float(trimmed_seams[0] / course.fps),
        "entry_frame_index_at_0p8s": int(course.entry_frame_index(0.80)),
        "start_yaw_world_rad": float(course.start_yaw_world),
        "travel_direction_world_xy": course.travel_direction_world_xy(
            course.entry_frame_index(0.80)
        ).tolist(),
    }


def _resample_qpos(
    timestamps_s: np.ndarray, qpos: np.ndarray, *, fps: float
) -> np.ndarray:
    target = np.arange(
        0.0,
        float(timestamps_s[-1]) + 0.5 / float(fps),
        1.0 / float(fps),
        dtype=np.float64,
    )
    left = np.clip(
        np.searchsorted(timestamps_s, target, side="right") - 1,
        0,
        len(timestamps_s) - 1,
    )
    right = np.minimum(left + 1, len(timestamps_s) - 1)
    span = timestamps_s[right] - timestamps_s[left]
    alpha = np.divide(
        target - timestamps_s[left],
        span,
        out=np.zeros_like(target),
        where=span > 1.0e-9,
    )
    result = np.empty((len(target), 36), dtype=np.float64)
    result[:, :3] = (
        (1.0 - alpha[:, None]) * qpos[left, :3]
        + alpha[:, None] * qpos[right, :3]
    )
    result[:, 7:] = (
        (1.0 - alpha[:, None]) * qpos[left, 7:]
        + alpha[:, None] * qpos[right, 7:]
    )
    result[:, 3:7] = np.stack(
        [
            _slerp_wxyz(qpos[a, 3:7], qpos[b, 3:7], amount)
            for a, b, amount in zip(left, right, alpha, strict=True)
        ]
    )
    return result


def _append_live_rollout_exit(
    *,
    course_path: Path,
    rollout_trace: Path,
    course_mode: str,
    output_path: Path,
) -> tuple[StitchedMotion, dict[str, object]]:
    """Recover a phase-matched flat exit already exercised by live rollout."""

    course = MotionBricksTerrainCourse.load(course_path)
    source_qpos = course.native_mujoco_qpos()
    with np.load(rollout_trace.expanduser().resolve(), allow_pickle=False) as arrays:
        timestamps = np.asarray(arrays["timestamp_s"], dtype=np.float64)
        qpos = np.asarray(arrays["mujoco_qpos"], dtype=np.float64)
        modes = np.asarray(arrays["mode"])
    course_frames = np.flatnonzero(modes == str(course_mode))
    if len(course_frames) != course.frame_count:
        raise ValueError(
            f"rollout {course_mode} frame count does not match source course"
        )
    if np.max(np.abs(qpos[course_frames[-1]] - source_qpos[-1])) > 1.0e-5:
        raise ValueError("live rollout exit does not begin at the exact course end")
    possible = np.arange(course_frames[-1] + 1, len(modes), dtype=np.int64)
    first_nonflat = np.flatnonzero(modes[possible] != "flat")
    prefix_length = (
        len(possible) if len(first_nonflat) == 0 else int(first_nonflat[0])
    )
    trailing = possible[:prefix_length]
    if len(trailing) < 24:
        raise ValueError("live rollout has no contiguous flat exit continuation")
    exit_time = np.concatenate(
        ((timestamps[course_frames[-1]],), timestamps[trailing])
    )
    exit_time -= exit_time[0]
    exit_qpos = np.concatenate(
        (source_qpos[-1:], qpos[trailing]), axis=0
    )
    resampled_exit = _resample_qpos(exit_time, exit_qpos, fps=course.fps)
    combined = np.concatenate((source_qpos, resampled_exit[1:]), axis=0)
    joints = np.stack(
        [mujoco_to_isaaclab_joint_vector(value[7:]) for value in combined]
    )
    seams = (int(course.seam_indices[0]), int(course.frame_count))
    motion = StitchedMotion(
        fps=course.fps,
        root_position_world=np.asarray(combined[:, :3], dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            combined[:, 3:7], dtype=np.float32
        ),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=tuple(
            FrameProvenance(-1, frame, "live_phase_matched_exit")
            for frame in range(len(combined))
        ),
        seam_indices=seams,
    )
    mechanics = _maximum_steps(motion)
    if not _mechanically_accepted(mechanics):
        raise ValueError("live rollout exit fails the mechanical motion gate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fps=np.asarray(motion.fps, dtype=np.float32),
        root_position_world=motion.root_position_world,
        root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
        joint_position=motion.joint_position,
        seam_indices=np.asarray(seams, dtype=np.int64),
        source_archive_clip_index=np.full(len(combined), -1, dtype=np.int64),
        source_frame=np.arange(len(combined), dtype=np.int64),
        source_clip_id=np.full(
            len(combined), "live_phase_matched_exit", dtype=np.str_
        ),
    )
    return motion, {
        "rollout_trace": str(rollout_trace.expanduser().resolve()),
        "course_mode": str(course_mode),
        "source_course": str(course_path.expanduser().resolve()),
        "flat_exit_frame_count_30hz": int(len(trailing)),
        "flat_exit_frame_count_50hz": int(len(resampled_exit) - 1),
        "mechanics": mechanics,
    }


def build(
    *,
    forward_manifest: Path,
    output_dir: Path,
    event_indices: tuple[int, ...] | None = None,
    archive_path: Path = DEFAULT_ARCHIVE,
    model_path: Path = DEFAULT_G1_MJCF,
    live_rollout_fallback: Path | None = None,
    live_rollout_fallback_event_indices: tuple[int, ...] | None = None,
    event_live_rollout_fallbacks: dict[int, Path] | None = None,
    allow_partial: bool = False,
) -> dict[str, object]:
    source = forward_manifest.expanduser().resolve()
    manifest = json.loads(source.read_text())
    if (
        manifest.get("schema") != "generic-terrain-portal-manifest/v1"
        or manifest.get("status") != "accepted"
    ):
        raise ValueError("source portal manifest must be accepted")
    terrain = load_target_mesh(
        Path(str(manifest["terrain_usd"])),
        position_world=manifest["terrain_position_world"],
        quaternion_world_from_usd_wxyz=(
            manifest["terrain_quaternion_world_from_usd_wxyz"]
        ),
    )
    archive = zarr.open_group(str(archive_path.expanduser().resolve()), mode="r")
    sole_adapter = _G1FootfallAdapter(
        model_path.expanduser().resolve(),
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=0.1,
    )
    ray_origin_z = float(np.max(terrain.vertices_world[:, 2]) + 2.0)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    wanted = None if event_indices is None else set(event_indices)
    fallback_events = (
        None
        if live_rollout_fallback_event_indices is None
        else set(live_rollout_fallback_event_indices)
    )
    fallback_by_event = {
        int(index): Path(path).expanduser().resolve()
        for index, path in (event_live_rollout_fallbacks or {}).items()
    }
    rows: list[dict[str, object]] = []
    courses: list[str] = []
    for event in manifest["events"]:
        event_index = int(event["event_index"])
        if wanted is not None and event_index not in wanted:
            continue
        event_live_fallback = fallback_by_event.get(event_index)
        if (
            event_live_fallback is None
            and live_rollout_fallback is not None
            and (fallback_events is None or event_index in fallback_events)
        ):
            event_live_fallback = live_rollout_fallback
        selected = event.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(f"event {event_index} has no selected primitive")
        primitive = Path(str(selected["primitive_motion"])).expanduser().resolve()
        entry_summary = json.loads(Path(str(selected["summary"])).read_text())
        approach_raw = Path(str(entry_summary["approach_raw"])).expanduser().resolve()
        attempts: list[dict[str, object]] = []
        accepted: dict[str, object] | None = None
        exits = tuple(
            dict.fromkeys(
                (
                    *_exit_candidates(
                        primitive,
                        exit_raw=DEFAULT_EXIT,
                        descending_exit_raws=DEFAULT_DESCENDING_EXITS,
                    ),
                    *DEFAULT_DESCENDING_EXITS,
                    DEFAULT_EXIT,
                )
            )
        )
        # Reuse completed wide phase searches, but do not launch another
        # expensive one when an already-exercised live exit is available.
        # This preserves prior accepted events while allowing a prompt
        # fallback for the one event whose wider search never completed.
        compose_tiers = ((4, 8), (16, 128))
        for phase_limit, combination_limit in compose_tiers:
            for exit_index, exit_raw in enumerate(exits):
                trial_dir = (
                    destination
                    / f"event_{event_index:02d}"
                    / f"forward_with_exit_{exit_index:02d}_phase_{phase_limit:02d}"
                )
                summary_path = trial_dir / "summary.json"
                if summary_path.is_file():
                    result = json.loads(summary_path.read_text())
                elif event_live_fallback is not None and phase_limit > 4:
                    continue
                else:
                    result = compose(
                        approach_raw=approach_raw,
                        exit_raw=exit_raw,
                        stair_motion=primitive,
                        stairs_archive=archive_path.expanduser().resolve(),
                        target_clip_index=None,
                        target_mesh=terrain,
                        model_path=model_path.expanduser().resolve(),
                        output_dir=trial_dir,
                        phase_candidate_limit=phase_limit,
                        maximum_candidate_combinations=combination_limit,
                        render=False,
                        include_exit=True,
                    )
                attempt = {
                    "phase_candidate_limit": phase_limit,
                    "maximum_candidate_combinations": combination_limit,
                    "exit_raw": str(exit_raw),
                    "summary": str(summary_path),
                    "status": result["status"],
                }
                attempts.append(attempt)
                if result["status"] != "accepted":
                    continue
                forward_course = trial_dir / str(result["artifacts"]["motion"])
                reverse_path = (
                    destination / f"event_{event_index:02d}" / "motion.npz"
                )
                reverse = _reverse_motion(forward_course, reverse_path)
                support_audit = _support_foot_clearance_audit(
                    reverse_path,
                    sole_adapter=sole_adapter,
                    terrain=terrain,
                    ray_origin_z=ray_origin_z,
                )
                support_trim = None
                if not support_audit["accepted"]:
                    support_trim = _trim_hovering_flat_margins(
                        reverse_path, support_audit
                    )
                    if support_trim is not None:
                        support_audit = _support_foot_clearance_audit(
                            reverse_path,
                            sole_adapter=sole_adapter,
                            terrain=terrain,
                            ray_origin_z=ray_origin_z,
                        )
                        trimmed = MotionBricksTerrainCourse.load(reverse_path)
                        reverse["frame_count"] = trimmed.frame_count
                        reverse["seam_indices"] = list(trimmed.seam_indices)
                        reverse["flat_approach_frame_count"] = int(
                            trimmed.seam_indices[0]
                        )
                        reverse["flat_approach_duration_s"] = float(
                            trimmed.seam_indices[0] / trimmed.fps
                        )
                attempt["support_foot_clearance"] = support_audit
                attempt["support_margin_trim"] = support_trim
                if not support_audit["accepted"]:
                    attempt["status"] = "support_hover_rejected"
                    continue
                accepted = {
                    **attempt,
                    **reverse,
                    "construction": "time_reverse(exact_forward_course+MotionBricks_exit)",
                    "collision_audit_invariant_under_time_reversal": True,
                }
                courses.append(str(reverse_path.resolve()))
                break
            if accepted is not None:
                break
        if accepted is None and event_live_fallback is not None:
            fallback_dir = (
                destination / f"event_{event_index:02d}" / "live_rollout_exit"
            )
            forward_course = fallback_dir / "forward_motion.npz"
            motion, fallback = _append_live_rollout_exit(
                course_path=Path(str(selected["course_motion"])),
                rollout_trace=event_live_fallback,
                course_mode=f"course_{event_index}",
                output_path=forward_course,
            )
            audit = audit_stair_motion_collisions(
                motion,
                archive_path=archive_path.expanduser().resolve(),
                target_clip_index=None,
                target_mesh=terrain,
                model_path=model_path.expanduser().resolve(),
                maximum_foot_penetration_m=0.005,
                maximum_forbidden_body_penetration_m=1.0e-6,
            )
            fallback["full_body_audit"] = {
                "accepted": bool(audit.accepted),
                "maximum_foot_penetration_m": float(
                    audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    audit.maximum_forbidden_body_penetration_m
                ),
            }
            fallback["status"] = "accepted" if audit.accepted else "rejected"
            (fallback_dir / "summary.json").write_text(
                json.dumps(fallback, indent=2, sort_keys=True) + "\n"
            )
            attempts.append(
                {
                    "status": fallback["status"],
                    "construction": "accepted_live_phase_matched_exit",
                    "summary": str(fallback_dir / "summary.json"),
                }
            )
            if audit.accepted:
                reverse_path = (
                    destination / f"event_{event_index:02d}" / "motion.npz"
                )
                reverse = _reverse_motion(forward_course, reverse_path)
                support_audit = _support_foot_clearance_audit(
                    reverse_path,
                    sole_adapter=sole_adapter,
                    terrain=terrain,
                    ray_origin_z=ray_origin_z,
                )
                support_trim = None
                if not support_audit["accepted"]:
                    support_trim = _trim_hovering_flat_margins(
                        reverse_path, support_audit
                    )
                    if support_trim is not None:
                        support_audit = _support_foot_clearance_audit(
                            reverse_path,
                            sole_adapter=sole_adapter,
                            terrain=terrain,
                            ray_origin_z=ray_origin_z,
                        )
                        trimmed = MotionBricksTerrainCourse.load(reverse_path)
                        reverse["frame_count"] = trimmed.frame_count
                        reverse["seam_indices"] = list(trimmed.seam_indices)
                        reverse["flat_approach_frame_count"] = int(
                            trimmed.seam_indices[0]
                        )
                        reverse["flat_approach_duration_s"] = float(
                            trimmed.seam_indices[0] / trimmed.fps
                        )
                fallback["support_foot_clearance"] = support_audit
                fallback["support_margin_trim"] = support_trim
                if support_audit["accepted"]:
                    accepted = {
                        "status": "accepted",
                        "construction": (
                            "time_reverse(exact_forward_course+"
                            "accepted_live_phase_matched_exit)"
                        ),
                        "collision_audit_invariant_under_time_reversal": True,
                        "summary": str(fallback_dir / "summary.json"),
                        **fallback,
                        **reverse,
                    }
                    courses.append(str(reverse_path.resolve()))
                else:
                    attempts[-1]["status"] = "support_hover_rejected"
                    attempts[-1]["support_foot_clearance"] = support_audit
        rows.append(
            {
                "event_index": event_index,
                "kind": event["kind"],
                "backend": event["backend"],
                "start_distance_m": event["end_distance_m"],
                "end_distance_m": event["start_distance_m"],
                "start_xy": event["end_xy"],
                "end_xy": event["start_xy"],
                "status": "accepted" if accepted is not None else "rejected",
                "selected": accepted,
                "attempts": attempts,
            }
        )
    # Every synthesized course traverses its source event in reverse, so a
    # coherent end-to-start route must also visit the events in reverse order.
    rows.reverse()
    courses.reverse()
    accepted_all = bool(rows) and all(row["status"] == "accepted" for row in rows)
    rejected_events = [
        {
            "event_index": int(row["event_index"]),
            "kind": str(row["kind"]),
            "attempts": row["attempts"],
        }
        for row in rows
        if row["status"] != "accepted"
    ]
    published_rows = (
        [row for row in rows if row["status"] == "accepted"]
        if bool(allow_partial) and not accepted_all
        else rows
    )
    accepted_bundle = bool(accepted_all or (allow_partial and published_rows))
    result = {
        "schema": "generic-terrain-portal-manifest/v1",
        "status": "accepted" if accepted_bundle else "rejected",
        "construction": "backward-travel_with_forward-facing_body",
        "coverage_complete": accepted_all,
        "partial_bundle_allowed": bool(allow_partial),
        "rejected_events": rejected_events,
        "source_portal_manifest": str(source),
        "terrain_usd": manifest["terrain_usd"],
        "terrain_position_world": manifest["terrain_position_world"],
        "terrain_quaternion_world_from_usd_wxyz": (
            manifest["terrain_quaternion_world_from_usd_wxyz"]
        ),
        "events": published_rows,
        "course_motions": courses,
    }
    (destination / "portal_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-index", type=int, action="append")
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--live-rollout-fallback", type=Path)
    parser.add_argument(
        "--live-rollout-fallback-event-index", type=int, action="append"
    )
    parser.add_argument(
        "--event-live-rollout-fallback",
        action="append",
        metavar="EVENT_INDEX=TRACE_NPZ",
        help="event-specific phase-matched exit trace; repeat as needed",
    )
    parser.add_argument("--allow-partial", action="store_true")
    arguments = parser.parse_args(argv)
    fallback_by_event: dict[int, Path] = {}
    for specification in arguments.event_live_rollout_fallback or ():
        event_text, separator, path_text = specification.partition("=")
        if not separator or not event_text or not path_text:
            parser.error(
                "--event-live-rollout-fallback expects EVENT_INDEX=TRACE_NPZ"
            )
        try:
            event_index = int(event_text)
        except ValueError:
            parser.error(
                "--event-live-rollout-fallback event index must be an integer"
            )
        if event_index in fallback_by_event:
            parser.error(
                f"duplicate live-rollout fallback for event {event_index}"
            )
        fallback_by_event[event_index] = Path(path_text)
    result = build(
        forward_manifest=arguments.forward_manifest,
        output_dir=arguments.output_dir,
        event_indices=(
            None
            if arguments.event_index is None
            else tuple(arguments.event_index)
        ),
        archive_path=arguments.archive_path,
        model_path=arguments.model_path,
        live_rollout_fallback=arguments.live_rollout_fallback,
        live_rollout_fallback_event_indices=(
            None
            if arguments.live_rollout_fallback_event_index is None
            else tuple(arguments.live_rollout_fallback_event_index)
        ),
        event_live_rollout_fallbacks=fallback_by_event,
        allow_partial=arguments.allow_partial,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
