"""Headless held-out rollout of live MotionBricks between terrain portals.

This is a globally privileged evaluator, not a deployable command interface.
It uses the robot root only to emulate a human steering toward the next known
portal.  MotionBricks produces every flat frame; exact-audited course playback
is committed only while crossing a non-flat event.
"""

from __future__ import annotations

import argparse
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
    _guard_flat_velocity,
    _submit_motionbricks,
    _terrain_surface_height,
)
from .motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    TerrainCoursePlayback,
    _slerp_wxyz,
    _yaw_wxyz,
)
from .render_stitched_motion import render_stitched_motion
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


def _submit_phase_matched_course_exit(
    full_agent: object,
    controller: object,
    *,
    context_qpos: np.ndarray,
    velocity_world_xy: np.ndarray,
    facing_yaw_world: float,
    mode_name: str,
    support_height_world: float,
) -> dict[str, object]:
    """Choose the target gait phase that least crouches from a portal exit.

    MotionBricks' ``random_seed`` indexes the target locomotion clip phase.
    A fixed seed therefore asks every authored landing pose to inbetween toward
    the same arbitrary gait phase.  Rank a small deterministic phase set by
    early root-height loss, simultaneous knee flexion, pose gap, and joint
    step, then leave the agent populated with the best continuation.
    """

    import torch

    context = np.asarray(context_qpos, dtype=np.float64)
    if context.ndim != 2 or context.shape[1] != 36:
        raise ValueError("course exit context must have shape (T,36)")
    candidates: list[dict[str, float | int]] = []
    for seed in range(0, 64, 4):
        _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=context,
            velocity_world_xy=velocity_world_xy,
            facing_yaw_world=facing_yaw_world,
            mode_name=mode_name,
            force=True,
            random_seed=seed,
            support_height_world=support_height_world,
        )
        generated_value = full_agent.frames["mujoco_qpos"]
        generated = (
            generated_value[0].detach().cpu().numpy()
            if isinstance(generated_value, torch.Tensor)
            else np.asarray(generated_value)[0]
        )
        horizon = min(len(generated), 24)
        review = np.asarray(generated[:horizon], dtype=np.float64)
        root_drop = max(0.0, float(context[-1, 2] - np.min(review[:, 2])))
        # MuJoCo joint order: left knee is joint 3 and right knee joint 9.
        double_knee = float(np.max(np.mean(review[:, (10, 16)], axis=1)))
        pose_gap = float(
            np.sqrt(np.mean((review[0, 7:] - context[-1, 7:]) ** 2))
        )
        joint_step = float(
            np.max(
                np.abs(
                    np.diff(
                        np.concatenate((context[-1:, 7:], review[:, 7:]), axis=0),
                        axis=0,
                    )
                )
            )
        )
        score = 4.0 * root_drop + 0.20 * double_knee + 0.50 * pose_gap + 0.20 * joint_step
        candidates.append(
            {
                "seed": seed,
                "score": score,
                "root_drop_m": root_drop,
                "double_knee_peak_rad": double_knee,
                "first_pose_rmse_rad": pose_gap,
                "maximum_joint_step_rad": joint_step,
            }
        )
    selected = min(candidates, key=lambda value: float(value["score"]))
    _submit_motionbricks(
        full_agent,
        controller,
        context_qpos=context,
        velocity_world_xy=velocity_world_xy,
        facing_yaw_world=facing_yaw_world,
        mode_name=mode_name,
        force=True,
        random_seed=int(selected["seed"]),
        support_height_world=support_height_world,
    )
    return {
        "selected": selected,
        "candidate_count": len(candidates),
        "score_minimum": float(selected["score"]),
        "score_median": float(np.median([value["score"] for value in candidates])),
        "score_maximum": float(max(value["score"] for value in candidates)),
    }


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
    render: bool = True,
) -> dict[str, object]:
    manifest_path = portal_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "accepted":
        raise ValueError("portal manifest must be accepted")
    courses = tuple(
        MotionBricksTerrainCourse.load(Path(value))
        for value in manifest["course_motions"]
    )
    if not courses:
        raise ValueError("portal manifest has no courses")
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
        - 1.35 * first.approach_direction_world_xy
        + 0.65 * normal
    )
    last = courses[-1]
    last_direction = last.approach_direction_world_xy
    last_normal = np.asarray((-last_direction[1], last_direction[0]))
    final_xy = last.root_position_world[-1, :2] + 0.95 * last_direction - 0.65 * last_normal

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
    current = np.asarray(full_agent.get_next_frame(), dtype=np.float64)
    timestamps = [0.0]
    qposes = [current.copy()]
    modes = ["flat"]
    requested = [desired_velocity.copy()]
    pending = 0
    playback: TerrainCoursePlayback | None = None
    active_course: MotionBricksTerrainCourse | None = None
    capture_rows: list[dict[str, object]] = []
    exit_phase_rows: list[dict[str, object]] = []
    blocked_count = 0
    completed = False

    for tick in range(int(maximum_ticks)):
        flat_target, terminal_direction = _next_flat_target(
            courses, pending, final_xy
        )
        delta = flat_target - current[:2]
        distance = float(np.linalg.norm(delta))
        target_direction = (
            delta / distance if distance > 1.0e-6 else terminal_direction
        )
        target_speed = min(0.58, max(0.04, 0.90 * distance))
        if pending >= len(courses) and distance < 0.10:
            target_speed = 0.0
        if pending < len(courses) and distance < 0.12:
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
            courses[pending].start_yaw_world
            if pending < len(courses)
            else math.atan2(float(target_direction[1]), float(target_direction[0]))
        )
        desired_facing += float(
            np.clip(_wrap(target_yaw - desired_facing), -1.4 / 30.0, 1.4 / 30.0)
        )
        desired_facing = _wrap(desired_facing)

        if playback is None and pending < len(courses):
            capture = courses[pending].portal_capture(current, desired_velocity)
            if capture.accepted:
                active_course = courses[pending]
                playback = TerrainCoursePlayback(active_course, current, blend_frames=18)
                capture_rows.append(
                    {
                        "course_index": pending,
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
            dt = 1.0 / active_course.fps
            mode = f"course_{pending}"
            if playback.done:
                finished_course_index = pending
                pending += 1
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
                exit_phase_rows.append(
                    {
                        "course_index": finished_course_index,
                        "support_height_world_m": flat_support_height,
                        **phase_result,
                    }
                )
                current = np.asarray(full_agent.get_next_frame(), dtype=np.float64)
                desired_velocity = next_velocity
                desired_facing = next_facing
                playback = None
                active_course = None
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
            _submit_motionbricks(
                full_agent,
                controller,
                context_qpos=full_agent.get_context_mujoco_qpos(),
                velocity_world_xy=guarded,
                facing_yaw_world=desired_facing,
                mode_name=mode_name,
                force=False,
                random_seed=flat_phase_seed,
                support_height_world=flat_support_height,
            )
            current = np.asarray(full_agent.get_next_frame(), dtype=np.float64)
            dt = 1.0 / 30.0
            mode = "flat"

        timestamps.append(timestamps[-1] + dt)
        qposes.append(current.copy())
        modes.append(mode)
        requested.append(desired_velocity.copy())
        if pending >= len(courses) and float(np.linalg.norm(final_xy - current[:2])) < 0.10:
            completed = True
            break

    time_array, resampled = _resample_qpos(
        np.asarray(timestamps), np.asarray(qposes), fps=50.0
    )
    motion = _motion_from_qpos(resampled, fps=50.0)
    motion_path = destination / "motion.npz"
    _save_motion(motion_path, motion)
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
        mode=np.asarray(modes, dtype=np.str_),
        resampled_timestamp_s=np.asarray(time_array, dtype=np.float32),
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
    summary = {
        "schema": "motionbricks-global-terrain-rollout/v1",
        "status": "accepted" if completed and pending == len(courses) else "incomplete",
        "portal_manifest": str(manifest_path),
        "seed": int(seed),
        "completed_all_courses": pending == len(courses),
        "reached_final_waypoint": completed,
        "portal_capture_count": len(capture_rows),
        "portal_captures": capture_rows,
        "course_exit_phase_matches": exit_phase_rows,
        "flat_frame_count": int(sum(value == "flat" for value in modes)),
        "course_frame_count": int(sum(value != "flat" for value in modes)),
        "terrain_guard_blocked_frame_count": int(blocked_count),
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
    parser.add_argument("--no-render", action="store_true")
    arguments = parser.parse_args(argv)
    result = rollout(
        portal_manifest=arguments.portal_manifest,
        output_dir=arguments.output_dir,
        motionbricks_root=arguments.motionbricks_root,
        model_path=arguments.model_path,
        seed=arguments.seed,
        maximum_ticks=arguments.maximum_ticks,
        render=not arguments.no_render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
