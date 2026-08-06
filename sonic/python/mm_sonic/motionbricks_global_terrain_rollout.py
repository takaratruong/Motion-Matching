"""Headless held-out rollout of live MotionBricks between terrain portals.

This is a globally privileged evaluator, not a deployable command interface.
It uses the robot root only to emulate a human steering toward the next known
portal.  MotionBricks produces every flat frame; exact-audited course playback
is committed only while crossing a non-flat event.
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
from .gear_action import mujoco_to_isaaclab_joint_vector
from .motionbricks_global_terrain_viewer import (
    DEFAULT_MOTIONBRICKS_ROOT,
    EXIT_FOOT_LOCK_MAXIMUM_JOINT_STEP_RAD,
    EXIT_FOOT_LOCK_MAXIMUM_TARGET_ERROR_M,
    FLAT_SUPPORT_TOLERANCE_M,
    LANDING_SEAM_BLEND_CLEARANCE_M,
    MAXIMUM_SUPPORT_FOOT_CLEARANCE_M,
    MotionBricksCommandBufferInvalidator,
    MotionBricksExitFootLock,
    _flat_pose_support_error,
    _guard_flat_velocity,
    _per_foot_minimum_sole_clearance_m,
    _project_live_root_above_support,
    _submit_responsive_motionbricks,
    _submit_motionbricks,
    _submit_phase_matched_course_exit,
    _terrain_surface_height,
)
from .motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    TerrainCoursePlayback,
    _slerp_wxyz,
    _yaw_wxyz,
    select_terrain_seam_portal,
)
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion


def _wrap(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def _slew_vector(
    current: np.ndarray,
    target: np.ndarray,
    *,
    maximum_delta: float,
) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(
        current, dtype=np.float64
    )
    length = float(np.linalg.norm(delta))
    if length > float(maximum_delta) > 0.0:
        delta *= float(maximum_delta) / length
    return np.asarray(current, dtype=np.float64) + delta


def _next_flat_target(
    courses: tuple[MotionBricksTerrainCourse, ...],
    pending_course: int,
    final_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if pending_course < len(courses):
        course = courses[pending_course]
        direction = course.approach_direction_world_xy
        return course.root_position_world[0, :2].copy(), direction
    previous = courses[-1].root_position_world[-1, :2]
    direction = np.asarray(final_xy, dtype=np.float64) - previous
    direction /= max(float(np.linalg.norm(direction)), 1.0e-8)
    return np.asarray(final_xy, dtype=np.float64), direction


def _course_exit_command(
    course: MotionBricksTerrainCourse,
) -> tuple[np.ndarray, float, str]:
    """Continue the authored landing gait before steering elsewhere."""

    span = min(course.frame_count - 1, max(4, int(round(0.50 * course.fps))))
    velocity = (
        course.root_position_world[-1, :2]
        - course.root_position_world[-1 - span, :2]
    ) / (span / course.fps)
    speed = float(np.linalg.norm(velocity))
    if speed < 1.0e-5:
        velocity = 0.18 * course.approach_direction_world_xy
        speed = 0.18
    elif speed > 0.58:
        velocity *= 0.58 / speed
        speed = 0.58
    mode = "slow_walk" if speed < 0.38 else "walk"
    return velocity, _yaw_wxyz(course.root_quaternion_world_wxyz[-1]), mode


def _point_to_polyline_distance(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    starts = line[:-1]
    vectors = line[1:] - starts
    length_squared = np.sum(vectors * vectors, axis=1)
    relative = points[:, None, :] - starts[None, :, :]
    alpha = np.divide(
        np.sum(relative * vectors[None, :, :], axis=2),
        length_squared[None, :],
        out=np.zeros((len(points), len(vectors)), dtype=np.float64),
        where=length_squared[None, :] > 1.0e-12,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = starts[None] + alpha[:, :, None] * vectors[None]
    return np.min(np.linalg.norm(points[:, None] - closest, axis=2), axis=1)


def _longest_true_run(values: object) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=np.bool_):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return int(longest)


def _resample_qpos(
    timestamps: np.ndarray,
    qpos: np.ndarray,
    *,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    target_time = np.arange(
        0.0, float(timestamps[-1]) + 0.5 / fps, 1.0 / fps, dtype=np.float64
    )
    left = np.searchsorted(timestamps, target_time, side="right") - 1
    left = np.clip(left, 0, len(timestamps) - 1)
    right = np.minimum(left + 1, len(timestamps) - 1)
    span = timestamps[right] - timestamps[left]
    alpha = np.divide(
        target_time - timestamps[left],
        span,
        out=np.zeros_like(target_time),
        where=span > 1.0e-12,
    )
    output = np.empty((len(target_time), qpos.shape[1]), dtype=np.float64)
    output[:, :3] = (
        (1.0 - alpha[:, None]) * qpos[left, :3]
        + alpha[:, None] * qpos[right, :3]
    )
    output[:, 7:] = (
        (1.0 - alpha[:, None]) * qpos[left, 7:]
        + alpha[:, None] * qpos[right, 7:]
    )
    output[:, 3:7] = np.stack(
        [
            _slerp_wxyz(qpos[a, 3:7], qpos[b, 3:7], amount)
            for a, b, amount in zip(left, right, alpha, strict=True)
        ]
    )
    return target_time, output


def _motion_from_qpos(qpos: np.ndarray, *, fps: float) -> StitchedMotion:
    joints = np.stack(
        [mujoco_to_isaaclab_joint_vector(value[7:]) for value in qpos]
    )
    provenance = tuple(
        FrameProvenance(-1, frame, "live_motionbricks_global_route")
        for frame in range(len(qpos))
    )
    return StitchedMotion(
        fps=float(fps),
        root_position_world=np.asarray(qpos[:, :3], dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(qpos[:, 3:7], dtype=np.float32),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=provenance,
        seam_indices=(),
    )


def _save_motion(path: Path, motion: StitchedMotion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=np.asarray(motion.fps, dtype=np.float32),
        root_position_world=motion.root_position_world,
        root_quaternion_world_wxyz=motion.root_quaternion_world_wxyz,
        joint_position=motion.joint_position,
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


def _save_route_figure(
    path: Path,
    *,
    intended: np.ndarray,
    actual: np.ndarray,
    courses: tuple[MotionBricksTerrainCourse, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(intended[:, 0], intended[:, 1], "r-", linewidth=2.5)
    axes[0].set_title("Intended global route")
    axes[1].plot(intended[:, 0], intended[:, 1], "r--", linewidth=2, label="intended")
    axes[1].plot(actual[:, 0], actual[:, 1], color="#1676d2", linewidth=2, label="actual")
    axes[1].set_title("Live MotionBricks + portal playback")
    for axis in axes:
        for index, course in enumerate(courses):
            route = course.root_position_world[:, :2]
            axis.plot(
                route[:, 0],
                route[:, 1],
                color="#f49c24",
                linewidth=4,
                alpha=0.45,
                label="terrain portal" if index == 0 and axis is axes[1] else None,
            )
        axis.scatter(intended[0, 0], intended[0, 1], marker="o", color="black")
        axis.scatter(intended[-1, 0], intended[-1, 1], marker="x", color="black")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def rollout(
    *,
    portal_manifest: Path,
    output_dir: Path,
    motionbricks_root: Path = DEFAULT_MOTIONBRICKS_ROOT,
    model_path: Path = DEFAULT_G1_MJCF,
    seed: int = 17,
    maximum_ticks: int = 2400,
    inter_course_dwell_s: float = 0.0,
    prefer_landing_seam_reentry: bool = False,
    course_indices: tuple[int, ...] | None = None,
    render: bool = True,
) -> dict[str, object]:
    manifest_path = portal_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "accepted":
        raise ValueError("portal manifest must be accepted")
    all_courses = tuple(
        MotionBricksTerrainCourse.load(Path(value))
        for value in manifest["course_motions"]
    )
    if not all_courses:
        raise ValueError("portal manifest has no courses")
    if course_indices is None:
        course_labels = tuple(range(len(all_courses)))
    else:
        course_labels = tuple(int(value) for value in course_indices)
        if (
            not course_labels
            or len(set(course_labels)) != len(course_labels)
            or any(value < 0 or value >= len(all_courses) for value in course_labels)
        ):
            raise ValueError("course indices must be unique in-range values")
    courses = tuple(all_courses[value] for value in course_labels)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
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
    exit_foot_lock_adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in archive["joint_names"][:]),
        maximum_joint_correction_rad=1.6,
        target_tolerance_m=5.0e-4,
        maximum_iterations=96,
        damping=0.008,
    )
    exit_foot_lock = MotionBricksExitFootLock(
        sole_adapter=sole_adapter,
        ik_adapter=exit_foot_lock_adapter,
        terrain=terrain,
        ray_origin_z=ray_origin_z,
    )

    root = motionbricks_root.expanduser().resolve()
    for value in (root, root / "scripts"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    full_agent, controller = _load_demo(root, seed=seed)
    full_agent.reset()
    first = courses[0]
    normal = np.asarray(
        (-first.approach_direction_world_xy[1], first.approach_direction_world_xy[0])
    )
    start_xy = (
        first.root_position_world[0, :2]
        - 0.90 * first.approach_direction_world_xy
        + 0.35 * normal
    )
    last = courses[-1]
    last_velocity, _, _ = _course_exit_command(last)
    last_direction = last_velocity / max(
        float(np.linalg.norm(last_velocity)), 1.0e-8
    )
    final_xy = last.root_position_world[-1, :2] + 0.95 * last_direction

    context = first.resampled_context_qpos(at_end=False, target_fps=30.0, frame_count=4)
    context[:, :2] += start_xy - first.root_position_world[0, :2]
    desired_facing = first.start_yaw_world
    desired_velocity = np.zeros(2, dtype=np.float64)
    flat_phase_seed = 0
    flat_support_height = _terrain_surface_height(
        terrain, start_xy, ray_origin_z=ray_origin_z
    )
    _submit_motionbricks(
        full_agent,
        controller,
        context_qpos=context,
        velocity_world_xy=desired_velocity,
        facing_yaw_world=desired_facing,
        mode_name="idle",
        force=True,
        random_seed=flat_phase_seed,
        support_height_world=flat_support_height,
    )
    current, initial_clearance_lift = _project_live_root_above_support(
        np.asarray(full_agent.get_next_frame(), dtype=np.float64),
        sole_adapter=sole_adapter,
        terrain=terrain,
        ray_origin_z=ray_origin_z,
        support_height_world=flat_support_height,
    )
    timestamps = [0.0]
    qposes = [current.copy()]
    modes = ["flat"]
    requested = [desired_velocity.copy()]
    clearance_lifts = [initial_clearance_lift]
    flat_support_errors = [
        _flat_pose_support_error(
            current,
            sole_adapter=sole_adapter,
            terrain=terrain,
            ray_origin_z=ray_origin_z,
            support_height_world=flat_support_height,
        )
    ]
    flat_support_rejected_errors = [0.0]
    exit_foot_lock_active = [False]
    exit_foot_lock_raw_clearances = [0.0]
    exit_foot_lock_corrected_clearances = [0.0]
    exit_foot_lock_joint_corrections = [0.0]
    exit_foot_lock_target_errors = [0.0]
    exit_foot_lock_joint_steps = [0.0]
    exit_foot_lock_trigger_count = 0
    exit_foot_lock_release_count = 0
    safe_flat_history: deque[np.ndarray] = deque((current.copy(),), maxlen=4)
    command_buffer = MotionBricksCommandBufferInvalidator()
    command_buffer.observe_generated(
        velocity_world_xy=desired_velocity,
        facing_yaw_world=desired_facing,
        mode_name="idle",
    )
    command_buffer_invalidations: list[tuple[str, ...]] = []
    flat_support_recovery_count = 0
    pending = 0
    playback: TerrainCoursePlayback | None = None
    active_course: MotionBricksTerrainCourse | None = None
    capture_rows: list[dict[str, object]] = []
    exit_phase_rows: list[dict[str, object]] = []
    blocked_count = 0
    dwell_ticks_remaining = 0
    completed = False

    for tick in range(int(maximum_ticks)):
        clearance_lift = 0.0
        flat_support_error = 0.0
        flat_support_rejected_error = 0.0
        handoff_active = False
        handoff_raw_clearance = 0.0
        handoff_corrected_clearance = 0.0
        handoff_joint_correction = 0.0
        handoff_target_error = 0.0
        handoff_joint_step = 0.0
        flat_target, terminal_direction = _next_flat_target(
            courses, pending, final_xy
        )
        delta = flat_target - current[:2]
        distance = float(np.linalg.norm(delta))
        target_direction = (
            delta / distance if distance > 1.0e-6 else terminal_direction
        )
        target_speed = min(0.58, max(0.04, 0.90 * distance))
        seam_reentry_ready = False
        if pending < len(courses) and bool(prefer_landing_seam_reentry):
            seam = courses[pending].seam_indices[0]
            seam_reentry_ready = bool(
                np.linalg.norm(
                    current[:2] - courses[pending].root_position_world[seam, :2]
                )
                <= 0.08
            )
        if dwell_ticks_remaining > 0:
            target_speed = 0.0
            target_velocity = np.zeros(2, dtype=np.float64)
        elif seam_reentry_ready:
            target_speed = 0.18
            target_velocity = (
                target_speed
                * courses[pending].travel_direction_world_xy(
                    courses[pending].seam_indices[0]
                )
            )
        elif pending >= len(courses) and distance < 0.10:
            target_speed = 0.0
            target_velocity = np.zeros(2, dtype=np.float64)
        elif pending < len(courses) and distance < 0.12:
            # After centring on a portal, emulate the operator nudging the
            # stick through it.  Feedback-to-the-point alone changes sign at
            # zero error and can never satisfy a forward-alignment gate.
            target_velocity = 0.18 * courses[pending].approach_direction_world_xy
        else:
            target_velocity = target_direction * target_speed
        desired_velocity = _slew_vector(
            desired_velocity, target_velocity, maximum_delta=1.15 / 30.0
        )
        target_yaw = (
            desired_facing
            if dwell_ticks_remaining > 0 or seam_reentry_ready
            else
            courses[pending].start_yaw_world
            if pending < len(courses)
            else math.atan2(float(target_direction[1]), float(target_direction[0]))
        )
        desired_facing += float(
            np.clip(_wrap(target_yaw - desired_facing), -1.4 / 30.0, 1.4 / 30.0)
        )
        desired_facing = _wrap(desired_facing)

        if (
            playback is None
            and pending < len(courses)
            and dwell_ticks_remaining == 0
        ):
            seam_selection = (
                select_terrain_seam_portal(
                    (courses[pending],), current, desired_velocity
                )
                if bool(prefer_landing_seam_reentry)
                else None
            )
            entry_frame = 0
            entry_kind = "flat_lead"
            blend_frames = 18
            if seam_selection is not None:
                capture = seam_selection.capture
                entry_frame = seam_selection.entry_frame_index
                entry_kind = "landing_seam"
                blend_frames = 18
            else:
                capture = courses[pending].portal_capture(
                    current, desired_velocity
                )
            if capture.accepted:
                exit_foot_lock.reset()
                active_course = courses[pending]
                playback = TerrainCoursePlayback(
                    active_course,
                    current,
                    blend_frames=blend_frames,
                    start_frame=entry_frame,
                    allow_terrain_seam_start=(entry_kind == "landing_seam"),
                )
                capture_rows.append(
                    {
                        "course_index": course_labels[pending],
                        "entry_kind": entry_kind,
                        "entry_frame_index": entry_frame,
                        "tick": tick,
                        "time_s": timestamps[-1],
                        "position_error_m": capture.position_error_m,
                        "yaw_error_rad": capture.yaw_error_rad,
                        "travel_alignment": capture.travel_alignment,
                        "lower_body_rmse_rad": capture.lower_body_rmse_rad,
                    }
                )

        if playback is not None:
            assert active_course is not None
            current = playback.next_qpos()
            if (
                playback.index
                <= playback.start_frame + playback.blend_frames
            ):
                current, clearance_lift = _project_live_root_above_support(
                    current,
                    sole_adapter=sole_adapter,
                    terrain=terrain,
                    ray_origin_z=ray_origin_z,
                    sole_clearance_m=(
                        LANDING_SEAM_BLEND_CLEARANCE_M
                        if playback.start_frame
                        == playback.course.seam_indices[0]
                        else 0.003
                    ),
                )
            dt = 1.0 / active_course.fps
            mode = f"course_{course_labels[pending]}"
            if playback.done:
                finished_course_index = pending
                pending += 1
                if (
                    pending < len(courses)
                    and float(inter_course_dwell_s) > 0.0
                ):
                    next_velocity = np.zeros(2, dtype=np.float64)
                    next_facing = _yaw_wxyz(current[3:7])
                    next_mode = "idle"
                    dwell_ticks_remaining = max(
                        1, int(round(float(inter_course_dwell_s) * 30.0))
                    )
                else:
                    next_velocity, next_facing, next_mode = _course_exit_command(
                        active_course
                    )
                context = active_course.resampled_context_qpos(
                    at_end=True, target_fps=30.0, frame_count=4
                )
                flat_support_height = _terrain_surface_height(
                    terrain, current[:2], ray_origin_z=ray_origin_z
                )
                phase_result = _submit_phase_matched_course_exit(
                    full_agent,
                    controller,
                    context_qpos=context,
                    velocity_world_xy=next_velocity,
                    facing_yaw_world=next_facing,
                    mode_name=next_mode,
                    support_height_world=flat_support_height,
                )
                flat_phase_seed = int(phase_result["selected"]["seed"])
                command_buffer.observe_generated(
                    velocity_world_xy=next_velocity,
                    facing_yaw_world=next_facing,
                    mode_name=next_mode,
                )
                lock_receipt = exit_foot_lock.arm(current)
                exit_phase_rows.append(
                    {
                        "course_index": course_labels[finished_course_index],
                        "support_height_world_m": flat_support_height,
                        "exit_foot_lock": lock_receipt,
                        **phase_result,
                    }
                )
                desired_velocity = next_velocity
                desired_facing = next_facing
                playback = None
                active_course = None
                safe_flat_history.clear()
                safe_flat_history.append(current.copy())
        else:
            guarded, blocked = _guard_flat_velocity(
                terrain,
                current[:3],
                desired_velocity,
                ray_origin_z=ray_origin_z,
            )
            blocked_count += int(blocked)
            speed = float(np.linalg.norm(guarded))
            mode_name = "idle" if speed < 0.06 else ("slow_walk" if speed < 0.38 else "walk")
            # Preserve the selected exit gait until its anchored support foot
            # has transferred fully onto the landing.  The latest guidance is
            # applied immediately after release from verified handoff poses.
            if not exit_foot_lock.armed:
                submission = _submit_responsive_motionbricks(
                    full_agent,
                    controller,
                    command_buffer=command_buffer,
                    verified_history=safe_flat_history,
                    velocity_world_xy=guarded,
                    facing_yaw_world=desired_facing,
                    mode_name=mode_name,
                    random_seed=flat_phase_seed,
                    support_height_world=flat_support_height,
                )
                if submission.invalidated:
                    command_buffer_invalidations.append(
                        submission.invalidation_reasons
                    )
            proposed, clearance_lift = _project_live_root_above_support(
                np.asarray(full_agent.get_next_frame(), dtype=np.float64),
                sole_adapter=sole_adapter,
                terrain=terrain,
                ray_origin_z=ray_origin_z,
                support_height_world=flat_support_height,
            )
            handoff = exit_foot_lock.apply(proposed)
            proposed = handoff.qpos
            handoff_active = bool(
                handoff.active or handoff.joint_correction_rad > 1.0e-10
            )
            if handoff_active:
                handoff_raw_clearance = handoff.raw_minimum_clearance_m
                handoff_corrected_clearance = (
                    handoff.corrected_minimum_clearance_m
                )
                handoff_joint_correction = handoff.joint_correction_rad
                handoff_target_error = handoff.foot_target_error_m
                handoff_joint_step = handoff.maximum_joint_step_rad
            exit_foot_lock_trigger_count += int(handoff.triggered)
            exit_foot_lock_release_count += int(handoff.released)
            proposed_support_error = (
                0.0
                if handoff.active
                else _flat_pose_support_error(
                    proposed,
                    sole_adapter=sole_adapter,
                    terrain=terrain,
                    ray_origin_z=ray_origin_z,
                    support_height_world=flat_support_height,
                )
            )
            if proposed_support_error > FLAT_SUPPORT_TOLERANCE_M:
                # Match the interactive runtime: never publish a flat frame
                # that has wandered onto a ramp, curb, or stair without a
                # compatible portal.  Hold the last verified pose and re-seed
                # MotionBricks there so the operator can still back away.
                flat_support_rejected_error = proposed_support_error
                flat_support_recovery_count += 1
                exit_foot_lock.reset()
                context = np.stack(tuple(safe_flat_history))
                if len(context) < 4:
                    context = np.concatenate(
                        (
                            np.repeat(context[:1], 4 - len(context), axis=0),
                            context,
                        ),
                        axis=0,
                    )
                current = context[-1].copy()
                _submit_motionbricks(
                    full_agent,
                    controller,
                    context_qpos=context,
                    velocity_world_xy=np.zeros(2, dtype=np.float64),
                    facing_yaw_world=_yaw_wxyz(current[3:7]),
                    mode_name="idle",
                    force=True,
                    random_seed=flat_phase_seed,
                    support_height_world=flat_support_height,
                )
                for _ in range(len(context)):
                    full_agent.get_next_frame()
                command_buffer.observe_generated(
                    velocity_world_xy=np.zeros(2, dtype=np.float64),
                    facing_yaw_world=_yaw_wxyz(current[3:7]),
                    mode_name="idle",
                )
                clearance_lift = 0.0
                flat_support_error = _flat_pose_support_error(
                    current,
                    sole_adapter=sole_adapter,
                    terrain=terrain,
                    ray_origin_z=ray_origin_z,
                    support_height_world=flat_support_height,
                )
            else:
                current = proposed
                flat_support_error = proposed_support_error
                safe_flat_history.append(current.copy())
            dt = 1.0 / 30.0
            mode = "flat"
            if dwell_ticks_remaining > 0:
                dwell_ticks_remaining -= 1

        timestamps.append(timestamps[-1] + dt)
        qposes.append(current.copy())
        modes.append(mode)
        requested.append(desired_velocity.copy())
        clearance_lifts.append(clearance_lift)
        flat_support_errors.append(flat_support_error)
        flat_support_rejected_errors.append(flat_support_rejected_error)
        exit_foot_lock_active.append(handoff_active)
        exit_foot_lock_raw_clearances.append(handoff_raw_clearance)
        exit_foot_lock_corrected_clearances.append(
            handoff_corrected_clearance
        )
        exit_foot_lock_joint_corrections.append(
            handoff_joint_correction
        )
        exit_foot_lock_target_errors.append(handoff_target_error)
        exit_foot_lock_joint_steps.append(handoff_joint_step)
        if pending >= len(courses) and float(np.linalg.norm(final_xy - current[:2])) < 0.10:
            completed = True
            break

    time_array, resampled = _resample_qpos(
        np.asarray(timestamps), np.asarray(qposes), fps=50.0
    )
    motion = _motion_from_qpos(resampled, fps=50.0)
    motion_path = destination / "motion.npz"
    _save_motion(motion_path, motion)
    collision_audit = audit_stair_motion_collisions(
        motion,
        archive_path=C490_ARCHIVE,
        target_clip_index=None,
        target_mesh=terrain,
        model_path=model_path,
        maximum_foot_penetration_m=0.005,
        maximum_forbidden_body_penetration_m=1.0e-6,
    )
    per_foot_clearances = np.asarray(
        [
            _per_foot_minimum_sole_clearance_m(
                qpos,
                sole_adapter=sole_adapter,
                terrain=terrain,
                ray_origin_z=ray_origin_z,
            )
            for qpos in resampled
        ],
        dtype=np.float64,
    )
    support_foot_clearances = np.min(per_foot_clearances, axis=1)
    support_hover_indices = np.flatnonzero(
        support_foot_clearances > MAXIMUM_SUPPORT_FOOT_CLEARANCE_M
    )
    joint_steps = np.max(np.abs(np.diff(resampled[:, 7:], axis=0)), axis=1)
    root_steps = np.linalg.norm(np.diff(resampled[:, :3], axis=0), axis=1)
    quaternion_dots = np.abs(
        np.sum(resampled[:-1, 3:7] * resampled[1:, 3:7], axis=1)
    )
    quaternion_dots = np.clip(quaternion_dots, 0.0, 1.0)
    root_angular_steps = 2.0 * np.arccos(quaternion_dots)
    repeated_pose = (root_steps < 1.0e-6) & (joint_steps < 1.0e-6)
    maximum_joint_step = float(np.max(joint_steps))
    maximum_root_step = float(np.max(root_steps))
    maximum_root_angular_step = float(np.max(root_angular_steps))
    longest_repeated_pose_run = _longest_true_run(repeated_pose)
    continuity_accepted = bool(
        maximum_joint_step <= 0.25
        and maximum_root_step <= 0.08
        and maximum_root_angular_step <= 0.20
        and longest_repeated_pose_run <= 6
    )
    planned_parts = [start_xy[None]]
    for course in courses:
        if not np.allclose(planned_parts[-1][-1], course.root_position_world[0, :2]):
            planned_parts.append(course.root_position_world[0:1, :2])
        planned_parts.append(course.root_position_world[:, :2])
    planned_parts.append(final_xy[None])
    intended = np.concatenate(planned_parts, axis=0)
    actual = np.asarray(qposes)[:, :2]
    deviation = _point_to_polyline_distance(actual, intended)
    figure_path = destination / "intended_vs_actual.png"
    _save_route_figure(
        figure_path, intended=intended, actual=actual, courses=courses
    )
    np.savez_compressed(
        destination / "rollout_trace.npz",
        timestamp_s=np.asarray(timestamps, dtype=np.float32),
        mujoco_qpos=np.asarray(qposes, dtype=np.float32),
        requested_velocity_world_xy=np.asarray(requested, dtype=np.float32),
        terrain_root_clearance_lift_m=np.asarray(
            clearance_lifts, dtype=np.float32
        ),
        flat_support_error_m=np.asarray(
            flat_support_errors, dtype=np.float32
        ),
        flat_support_rejected_error_m=np.asarray(
            flat_support_rejected_errors, dtype=np.float32
        ),
        exit_foot_lock_active=np.asarray(exit_foot_lock_active, dtype=np.bool_),
        exit_foot_lock_raw_clearance_m=np.asarray(
            exit_foot_lock_raw_clearances, dtype=np.float32
        ),
        exit_foot_lock_corrected_clearance_m=np.asarray(
            exit_foot_lock_corrected_clearances, dtype=np.float32
        ),
        exit_foot_lock_joint_correction_rad=np.asarray(
            exit_foot_lock_joint_corrections, dtype=np.float32
        ),
        exit_foot_lock_target_error_m=np.asarray(
            exit_foot_lock_target_errors, dtype=np.float32
        ),
        exit_foot_lock_joint_step_rad=np.asarray(
            exit_foot_lock_joint_steps, dtype=np.float32
        ),
        mode=np.asarray(modes, dtype=np.str_),
        resampled_timestamp_s=np.asarray(time_array, dtype=np.float32),
        minimum_sole_clearance_m=np.asarray(
            support_foot_clearances, dtype=np.float32
        ),
        intended_polyline_xy=np.asarray(intended, dtype=np.float32),
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
    flat_support_violation_count = int(
        np.sum(
            np.asarray(flat_support_errors, dtype=np.float64)
            > FLAT_SUPPORT_TOLERANCE_M
        )
    )
    summary = {
        "schema": "motionbricks-global-terrain-rollout/v1",
        "status": (
            "accepted"
            if completed
            and pending == len(courses)
            and flat_support_violation_count == 0
            and collision_audit.accepted
            and len(support_hover_indices) == 0
            and continuity_accepted
            and not exit_foot_lock.armed
            and exit_foot_lock_release_count == len(exit_phase_rows)
            else "incomplete"
        ),
        "portal_manifest": str(manifest_path),
        "seed": int(seed),
        "inter_course_dwell_s": float(inter_course_dwell_s),
        "prefer_landing_seam_reentry": bool(prefer_landing_seam_reentry),
        "course_indices": list(course_labels),
        "completed_all_courses": pending == len(courses),
        "reached_final_waypoint": completed,
        "portal_capture_count": len(capture_rows),
        "portal_captures": capture_rows,
        "course_exit_phase_matches": exit_phase_rows,
        "flat_frame_count": int(sum(value == "flat" for value in modes)),
        "course_frame_count": int(sum(value != "flat" for value in modes)),
        "terrain_guard_blocked_frame_count": int(blocked_count),
        "command_buffer_invalidation_count": len(
            command_buffer_invalidations
        ),
        "command_buffer_invalidation_reasons": [
            list(value) for value in command_buffer_invalidations
        ],
        "terrain_root_clearance_lift_maximum_m": float(
            np.max(clearance_lifts)
        ),
        "terrain_root_clearance_lift_p95_m": float(
            np.percentile(clearance_lifts, 95.0)
        ),
        "flat_support_error_maximum_m": float(
            np.max(flat_support_errors)
        ),
        "flat_support_violation_frame_count": flat_support_violation_count,
        "flat_support_tolerance_m": FLAT_SUPPORT_TOLERANCE_M,
        "flat_support_recovery_count": int(flat_support_recovery_count),
        "flat_support_rejected_error_maximum_m": float(
            np.max(flat_support_rejected_errors)
        ),
        "course_exit_support_handoff": {
            "trigger_count": int(exit_foot_lock_trigger_count),
            "release_count": int(exit_foot_lock_release_count),
            "active_frame_count": int(np.sum(exit_foot_lock_active)),
            "maximum_raw_clearance_m": float(
                np.max(exit_foot_lock_raw_clearances)
            ),
            "maximum_corrected_clearance_m": float(
                np.max(exit_foot_lock_corrected_clearances)
            ),
            "maximum_joint_correction_rad": float(
                np.max(exit_foot_lock_joint_corrections)
            ),
            "maximum_target_error_m": float(
                np.max(exit_foot_lock_target_errors)
            ),
            "maximum_joint_step_rad": float(
                np.max(exit_foot_lock_joint_steps)
            ),
            "joint_step_threshold_rad": (
                EXIT_FOOT_LOCK_MAXIMUM_JOINT_STEP_RAD
            ),
            "target_error_threshold_m": (
                EXIT_FOOT_LOCK_MAXIMUM_TARGET_ERROR_M
            ),
        },
        "support_foot_clearance": {
            "accepted": len(support_hover_indices) == 0,
            "maximum_m": float(np.max(support_foot_clearances)),
            "p95_m": float(np.percentile(support_foot_clearances, 95.0)),
            "threshold_m": MAXIMUM_SUPPORT_FOOT_CLEARANCE_M,
            "threshold_exceedance_frame_indices": support_hover_indices.tolist(),
            "per_foot_minimum_m": [
                float(np.min(per_foot_clearances[:, 0])),
                float(np.min(per_foot_clearances[:, 1])),
            ],
        },
        "kinematic_continuity": {
            "accepted": continuity_accepted,
            "maximum_joint_step_rad": maximum_joint_step,
            "maximum_joint_step_threshold_rad": 0.25,
            "maximum_root_translation_step_m": maximum_root_step,
            "maximum_root_translation_step_threshold_m": 0.08,
            "maximum_root_angular_step_rad": maximum_root_angular_step,
            "maximum_root_angular_step_threshold_rad": 0.20,
            "longest_exact_repeated_pose_run_frames": (
                longest_repeated_pose_run
            ),
            "maximum_repeated_pose_run_frames": 6,
        },
        "full_body_collision_audit": {
            "accepted": bool(collision_audit.accepted),
            "maximum_foot_penetration_m": float(
                collision_audit.maximum_foot_penetration_m
            ),
            "maximum_forbidden_body_penetration_m": float(
                collision_audit.maximum_forbidden_body_penetration_m
            ),
            "foot_threshold_exceedance_frame_indices": list(
                collision_audit.foot_threshold_exceedance_frame_indices
            ),
            "forbidden_body_threshold_exceedance_frame_indices": list(
                collision_audit.forbidden_body_threshold_exceedance_frame_indices
            ),
        },
        "duration_s": float(timestamps[-1]),
        "root_path_length_m": float(np.sum(np.linalg.norm(np.diff(actual, axis=0), axis=1))),
        "planned_route_length_m": float(
            np.sum(np.linalg.norm(np.diff(intended, axis=0), axis=1))
        ),
        "path_deviation_rmse_m": float(np.sqrt(np.mean(deviation * deviation))),
        "path_deviation_p95_m": float(np.percentile(deviation, 95.0)),
        "path_deviation_maximum_m": float(np.max(deviation)),
        "artifacts": {
            "motion": str(motion_path),
            "trace": str(destination / "rollout_trace.npz"),
            "intended_vs_actual": str(figure_path),
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
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--maximum-ticks", type=int, default=2400)
    parser.add_argument("--inter-course-dwell-s", type=float, default=0.0)
    parser.add_argument("--prefer-landing-seam-reentry", action="store_true")
    parser.add_argument("--course-index", type=int, action="append")
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args(argv)
    result = rollout(
        portal_manifest=arguments.portal_manifest,
        output_dir=arguments.output_dir,
        motionbricks_root=arguments.motionbricks_root,
        model_path=arguments.model_path,
        seed=arguments.seed,
        maximum_ticks=arguments.maximum_ticks,
        inter_course_dwell_s=arguments.inter_course_dwell_s,
        prefer_landing_seam_reentry=arguments.prefer_landing_seam_reentry,
        course_indices=(
            None
            if arguments.course_index is None
            else tuple(arguments.course_index)
        ),
        render=not arguments.no_render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
