"""Headless no-match/retreat probe for the global MotionBricks runtime.

The robot starts on the last safe flat support before a terrain portal, with
its body deliberately facing ninety degrees away from every compatible portal
pose.  A held command into the obstacle must be stopped by the terrain guard;
an opposite command must then move the robot away, and a final reversal must
move it forward again without stale-state lockup.  No authored terrain course
is allowed to commit during this negative test.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys

import numpy as np

from .build_c490_curb_slope_archive import DEFAULT_OUTPUT as C490_ARCHIVE
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .generate_generic_stair_route import load_target_mesh
from .generate_motionbricks_terrain_transitions import _load_demo
from .motionbricks_global_terrain_rollout import (
    _motion_from_qpos,
    _resample_qpos,
    _save_motion,
)
from .motionbricks_global_terrain_viewer import (
    DEFAULT_MOTIONBRICKS_ROOT,
    FLAT_SUPPORT_TOLERANCE_M,
    MAXIMUM_SUPPORT_FOOT_CLEARANCE_M,
    MotionBricksCommandBufferInvalidator,
    _flat_pose_support_error,
    _guard_flat_velocity,
    _minimum_sole_clearance_m,
    _project_live_root_above_support,
    _submit_responsive_motionbricks,
    _submit_motionbricks,
    _terrain_surface_height,
)
from .motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    _yaw_wxyz,
    select_terrain_portal,
)
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)


def _left_multiply_yaw(quaternion_wxyz: object, yaw: float) -> np.ndarray:
    first = np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )
    second = np.asarray(quaternion_wxyz, dtype=np.float64)
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    result = np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def probe(
    *,
    portal_manifest: Path,
    output_dir: Path,
    motionbricks_root: Path = DEFAULT_MOTIONBRICKS_ROOT,
    model_path: Path = DEFAULT_G1_MJCF,
    seed: int = 31,
    course_index: int = 2,
    render: bool = True,
) -> dict[str, object]:
    manifest_path = portal_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "accepted":
        raise ValueError("safety probe needs an accepted portal manifest")
    courses = tuple(
        MotionBricksTerrainCourse.load(Path(value))
        for value in manifest["course_motions"]
    )
    selected = courses[int(course_index)]
    if selected.seam_indices[0] < 38:
        raise ValueError("selected safety-probe course has too little flat approach")
    terrain = load_target_mesh(
        Path(manifest["terrain_usd"]),
        position_world=manifest["terrain_position_world"],
        quaternion_world_from_usd_wxyz=(
            manifest["terrain_quaternion_world_from_usd_wxyz"]
        ),
    )
    ray_origin_z = float(np.max(terrain.vertices_world[:, 2]) + 2.0)
    import zarr

    archive = zarr.open_group(str(C490_ARCHIVE), mode="r")
    sole_adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=0.1,
    )
    root = motionbricks_root.expanduser().resolve()
    for value in (root, root / "scripts"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    full_agent, controller = _load_demo(root, seed=seed)
    full_agent.reset()

    start_frame = selected.seam_indices[0] - 15
    native = selected.native_mujoco_qpos()
    context = native[start_frame - 3 : start_frame + 1].copy()
    yaw_offset = 0.5 * math.pi
    for qpos in context:
        qpos[3:7] = _left_multiply_yaw(qpos[3:7], yaw_offset)
    current = context[-1].copy()
    facing_yaw = _yaw_wxyz(current[3:7])
    support_height = _terrain_surface_height(
        terrain, current[:2], ray_origin_z=ray_origin_z
    )
    for index, qpos in enumerate(context):
        context[index], _ = _project_live_root_above_support(
            qpos,
            sole_adapter=sole_adapter,
            terrain=terrain,
            ray_origin_z=ray_origin_z,
            support_height_world=support_height,
        )
    current = context[-1].copy()
    _submit_motionbricks(
        full_agent,
        controller,
        context_qpos=context,
        velocity_world_xy=np.zeros(2, dtype=np.float64),
        facing_yaw_world=facing_yaw,
        mode_name="idle",
        force=True,
        random_seed=0,
        support_height_world=support_height,
    )
    for _ in range(len(context)):
        full_agent.get_next_frame()
    command_buffer = MotionBricksCommandBufferInvalidator()
    command_buffer.observe_generated(
        velocity_world_xy=np.zeros(2, dtype=np.float64),
        facing_yaw_world=facing_yaw,
        mode_name="idle",
    )

    direction = selected.approach_direction_world_xy
    phases = (
        ("blocked_into_terrain", 36, 0.34 * direction),
        ("retreat", 72, -0.38 * direction),
        ("forward_after_retreat", 72, 0.38 * direction),
    )
    safe_history: deque[np.ndarray] = deque((current.copy(),), maxlen=4)
    qposes = [current.copy()]
    timestamps = [0.0]
    phase_names = ["initial"]
    requested = [np.zeros(2, dtype=np.float64)]
    blocked = [False]
    support_errors = [0.0]
    portal_acceptances = [False]
    recovery_count = 0
    invalidation_reasons: list[tuple[str, ...]] = []
    invalidated = [False]
    phase_rows: list[dict[str, object]] = []

    for phase_name, frame_count, velocity in phases:
        phase_start = current[:2].copy()
        blocked_count = 0
        accepted_count = 0
        for _ in range(frame_count):
            selection = select_terrain_portal(
                courses,
                current,
                velocity,
                entry_lead_time_s=0.80,
            )
            accepted_count += int(selection.capture.accepted)
            guarded, was_blocked = _guard_flat_velocity(
                terrain,
                current[:3],
                velocity,
                ray_origin_z=ray_origin_z,
            )
            blocked_count += int(was_blocked)
            speed = float(np.linalg.norm(guarded))
            mode = "idle" if speed < 0.06 else "slow_walk"
            submission = _submit_responsive_motionbricks(
                full_agent,
                controller,
                command_buffer=command_buffer,
                verified_history=safe_history,
                velocity_world_xy=guarded,
                facing_yaw_world=facing_yaw,
                mode_name=mode,
                random_seed=0,
                support_height_world=support_height,
            )
            if submission.invalidated:
                invalidation_reasons.append(submission.invalidation_reasons)
            proposed, _ = _project_live_root_above_support(
                np.asarray(full_agent.get_next_frame(), dtype=np.float64),
                sole_adapter=sole_adapter,
                terrain=terrain,
                ray_origin_z=ray_origin_z,
                support_height_world=support_height,
            )
            support_error = _flat_pose_support_error(
                proposed,
                sole_adapter=sole_adapter,
                terrain=terrain,
                ray_origin_z=ray_origin_z,
                support_height_world=support_height,
            )
            if support_error > FLAT_SUPPORT_TOLERANCE_M:
                recovery_count += 1
                recovery = np.stack(tuple(safe_history))
                if len(recovery) < 4:
                    recovery = np.concatenate(
                        (
                            np.repeat(recovery[:1], 4 - len(recovery), axis=0),
                            recovery,
                        ),
                        axis=0,
                    )
                current = recovery[-1].copy()
                _submit_motionbricks(
                    full_agent,
                    controller,
                    context_qpos=recovery,
                    velocity_world_xy=np.zeros(2, dtype=np.float64),
                    facing_yaw_world=facing_yaw,
                    mode_name="idle",
                    force=True,
                    random_seed=0,
                    support_height_world=support_height,
                )
                for _ in range(len(recovery)):
                    full_agent.get_next_frame()
                command_buffer.observe_generated(
                    velocity_world_xy=np.zeros(2, dtype=np.float64),
                    facing_yaw_world=facing_yaw,
                    mode_name="idle",
                )
            else:
                current = proposed
                safe_history.append(current.copy())
            timestamps.append(timestamps[-1] + 1.0 / 30.0)
            qposes.append(current.copy())
            phase_names.append(phase_name)
            requested.append(velocity.copy())
            blocked.append(was_blocked)
            support_errors.append(support_error)
            portal_acceptances.append(selection.capture.accepted)
            invalidated.append(submission.invalidated)
        displacement = current[:2] - phase_start
        progress_direction = -direction if phase_name == "retreat" else direction
        phase_rows.append(
            {
                "phase": phase_name,
                "frame_count": frame_count,
                "blocked_frame_count": blocked_count,
                "portal_acceptance_count": accepted_count,
                "signed_progress_m": float(
                    np.dot(displacement, progress_direction)
                ),
            }
        )

    _, resampled = _resample_qpos(
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(qposes, dtype=np.float64),
        fps=50.0,
    )
    motion = _motion_from_qpos(resampled, fps=50.0)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    motion_path = destination / "motion.npz"
    _save_motion(motion_path, motion)
    collision = audit_stair_motion_collisions(
        motion,
        archive_path=C490_ARCHIVE,
        target_clip_index=None,
        target_mesh=terrain,
        model_path=model_path,
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=1.0e-6,
    )
    clearance = np.asarray(
        [
            _minimum_sole_clearance_m(
                qpos,
                sole_adapter=sole_adapter,
                terrain=terrain,
                ray_origin_z=ray_origin_z,
            )
            for qpos in resampled
        ],
        dtype=np.float64,
    )
    joint_steps = np.max(np.abs(np.diff(resampled[:, 7:], axis=0)), axis=1)
    root_steps = np.linalg.norm(np.diff(resampled[:, :3], axis=0), axis=1)
    quaternion_dots = np.abs(
        np.sum(resampled[:-1, 3:7] * resampled[1:, 3:7], axis=1)
    )
    root_angular_steps = 2.0 * np.arccos(
        np.clip(quaternion_dots, 0.0, 1.0)
    )
    maximum_joint_step = float(np.max(joint_steps))
    maximum_root_step = float(np.max(root_steps))
    maximum_root_angular_step = float(np.max(root_angular_steps))
    continuity_accepted = bool(
        maximum_joint_step <= 0.25
        and maximum_root_step <= 0.08
        and maximum_root_angular_step <= 0.20
    )
    blocked_into = phase_rows[0]
    retreat = phase_rows[1]
    forward = phase_rows[2]
    accepted = bool(
        int(blocked_into["blocked_frame_count"]) >= 5
        and float(retreat["signed_progress_m"]) >= 0.25
        and float(forward["signed_progress_m"]) >= 0.20
        and not any(portal_acceptances)
        and recovery_count == 0
        and float(np.max(support_errors)) <= FLAT_SUPPORT_TOLERANCE_M
        and float(np.max(clearance)) <= MAXIMUM_SUPPORT_FOOT_CLEARANCE_M
        and collision.accepted
        and continuity_accepted
    )
    video_path = destination / "rollout_25fps.mp4"
    render_result = None
    if render:
        render_result = render_stitched_motion(
            motion,
            model_path=model_path,
            archive_path=C490_ARCHIVE,
            target_mesh=terrain,
            output_path=video_path,
            width=640,
            height=360,
            frame_stride=2,
        )
    np.savez_compressed(
        destination / "probe_trace.npz",
        timestamp_s=np.asarray(timestamps, dtype=np.float32),
        mujoco_qpos=np.asarray(qposes, dtype=np.float32),
        phase=np.asarray(phase_names, dtype=np.str_),
        requested_velocity_world_xy=np.asarray(requested, dtype=np.float32),
        terrain_guard_blocked=np.asarray(blocked, dtype=np.bool_),
        flat_support_error_m=np.asarray(support_errors, dtype=np.float32),
        portal_accepted=np.asarray(portal_acceptances, dtype=np.bool_),
        command_buffer_invalidated=np.asarray(invalidated, dtype=np.bool_),
        resampled_minimum_sole_clearance_m=np.asarray(
            clearance, dtype=np.float32
        ),
    )
    summary = {
        "schema": "motionbricks-global-terrain-safety-probe/v1",
        "status": "accepted" if accepted else "rejected",
        "portal_manifest": str(manifest_path),
        "course_index": int(course_index),
        "start_frame": int(start_frame),
        "yaw_offset_rad": yaw_offset,
        "phases": phase_rows,
        "portal_acceptance_count": int(sum(portal_acceptances)),
        "flat_support_recovery_count": int(recovery_count),
        "command_buffer_invalidation_count": len(invalidation_reasons),
        "command_buffer_invalidation_reasons": [
            list(value) for value in invalidation_reasons
        ],
        "flat_support_error_maximum_m": float(np.max(support_errors)),
        "minimum_sole_clearance_maximum_m": float(np.max(clearance)),
        "full_body_collision_audit": {
            "accepted": bool(collision.accepted),
            "maximum_foot_penetration_m": float(
                collision.maximum_foot_penetration_m
            ),
            "maximum_forbidden_body_penetration_m": float(
                collision.maximum_forbidden_body_penetration_m
            ),
        },
        "kinematic_continuity": {
            "accepted": continuity_accepted,
            "maximum_joint_step_rad": maximum_joint_step,
            "maximum_joint_step_threshold_rad": 0.25,
            "maximum_root_translation_step_m": maximum_root_step,
            "maximum_root_translation_step_threshold_m": 0.08,
            "maximum_root_angular_step_rad": maximum_root_angular_step,
            "maximum_root_angular_step_threshold_rad": 0.20,
        },
        "artifacts": {
            "motion": str(motion_path),
            "trace": str(destination / "probe_trace.npz"),
            "video": str(video_path) if render else None,
        },
        "render": render_result,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portal-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--motionbricks-root", type=Path, default=DEFAULT_MOTIONBRICKS_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--course-index", type=int, default=2)
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args(argv)
    result = probe(
        portal_manifest=arguments.portal_manifest,
        output_dir=arguments.output_dir,
        motionbricks_root=arguments.motionbricks_root,
        model_path=arguments.model_path,
        seed=arguments.seed,
        course_index=arguments.course_index,
        render=not arguments.no_render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
