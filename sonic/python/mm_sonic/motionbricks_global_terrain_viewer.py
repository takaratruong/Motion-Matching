"""Browser/Switch viewer for MotionBricks flat motion plus global courses.

This is a deliberately privileged, purely kinematic ceiling.  Far from the
stair, the official MotionBricks G1 generator receives ordinary robot-local
two-stick commands.  At the best compatible prevalidated entry portal it
commits to an exact-mesh-audited flat->stair->flat course, then seeds
MotionBricks from the last four authored poses so joystick control resumes on
the landing.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import socket
import sys
import time

import numpy as np

from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .generate_motionbricks_terrain_transitions import _load_demo
from .gear_action import mujoco_to_isaaclab_joint_vector
from .motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    PortalCapture,
    TerrainCoursePlayback,
    _yaw_wxyz,
    select_terrain_portal,
)
from .privileged_terrain_matcher import DEFAULT_STAIRS_ARCHIVE
from .terrain_interactive_viewer import (
    _append_overlay_to_scene,
    _configure_terrain_browser_ui,
)
from .terrain_oracle.math3d import RigidTransform
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.render_media import _build_scene_model
from .terrain_oracle.source_grail import _load_usd_mesh
from .terrain_oracle.terrain_mesh import TerrainMeshIndex


DEFAULT_MOTIONBRICKS_ROOT = Path(
    "/move/u/bodow/Projects/reference-nvidia-groot-wbc/motionbricks"
)
DEFAULT_COURSE = Path(
    "/move/u/bodow/Projects/Motion-Matching-takara-corpus/"
    "artifacts/global_scene_terrain/"
    "motionbricks_course_target26_pos45_v1/"
    "reused_composition_v10_automatic_safe_phase/motion.npz"
)


def _catalog_course_paths(catalog_path: Path, target_clip_index: int) -> tuple[Path, ...]:
    """Return every authored and generated direction available for one mesh."""

    catalog = json.loads(catalog_path.expanduser().resolve().read_text())
    paths = []
    for route in catalog.get("routes", ()):
        if int(route["target_clip_index"]) != int(target_clip_index):
            continue
        paths.extend(
            Path(entry["motion_path"]).expanduser().resolve()
            for entry in route["entry_families"]
        )
    unique = tuple(dict.fromkeys(paths))
    if not unique:
        raise ValueError(
            f"terrain catalog has no courses for target {target_clip_index}"
        )
    return unique


def _route_manifest(
    manifest_path: Path,
) -> tuple[Path, tuple[float, float, float], tuple[float, float, float, float], tuple[Path, ...]]:
    """Load one compiled multi-event terrain scene for the browser viewer."""

    payload = json.loads(manifest_path.expanduser().resolve().read_text())
    if (
        payload.get("schema") != "generic-terrain-portal-manifest/v1"
        or payload.get("status") != "accepted"
    ):
        raise ValueError("terrain route manifest is not an accepted portal bundle")
    courses = tuple(
        Path(value).expanduser().resolve()
        for value in payload.get("course_motions", ())
    )
    if not courses:
        raise ValueError("terrain route manifest has no portal courses")
    position = tuple(float(value) for value in payload["terrain_position_world"])
    quaternion = tuple(
        float(value)
        for value in payload["terrain_quaternion_world_from_usd_wxyz"]
    )
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("terrain route manifest has an invalid scene transform")
    return (
        Path(str(payload["terrain_usd"])).expanduser().resolve(),
        position,
        quaternion,
        courses,
    )


def _rotate_xy(value: object, yaw: float) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=np.float64,
    )


def _configure_ui(browser_control: object) -> None:
    _configure_terrain_browser_ui(browser_control)
    replacements = {
        "G1 terrain motion matching · clean kinematics":
            "G1 MotionBricks · global terrain portal",
        "Waiting for first clean-kinematic G1 frame":
            "Loading MotionBricks and the audited terrain course",
        "the kinematic player stops at the next double support":
            "the committed stair route completes before joystick control resumes",
        "MM command [": "MotionBricks/terrain command [",
    }
    template = str(browser_control._HTML_TEMPLATE)
    for source, target in replacements.items():
        template = template.replace(source, target)
    browser_control._HTML_TEMPLATE = template


def _command_targets(
    command: object,
    *,
    current_yaw: float,
    desired_facing_yaw: float,
    dt_s: float,
) -> tuple[np.ndarray, float, float, str]:
    local = np.asarray(
        (float(command.travel_forward), float(command.travel_left)),
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(local))
    direction = local / magnitude if magnitude > 1.0e-6 else np.zeros(2)
    magnitude = float(np.clip(magnitude * float(command.speed_scale), 0.0, 1.0))
    if bool(command.facing_active):
        desired_facing_yaw = math.remainder(
            current_yaw
            + math.atan2(
                float(command.facing_left), float(command.facing_forward)
            ),
            2.0 * math.pi,
        )
    elif abs(float(command.yaw_left)) > 1.0e-6:
        desired_facing_yaw = math.remainder(
            desired_facing_yaw
            + float(command.yaw_left)
            * float(command.turn_scale)
            * 1.20
            * dt_s,
            2.0 * math.pi,
        )
    movement_world = _rotate_xy(direction, current_yaw)
    if magnitude <= 0.06:
        return np.zeros(2), desired_facing_yaw, 0.0, "idle"
    if magnitude < 0.48:
        speed = 0.18 + 0.42 * magnitude
        return movement_world * speed, desired_facing_yaw, speed, "slow_walk"
    speed = 0.22 + 0.68 * magnitude
    return movement_world * speed, desired_facing_yaw, speed, "walk"


def _submit_motionbricks(
    full_agent: object,
    controller: object,
    *,
    context_qpos: object,
    velocity_world_xy: object,
    facing_yaw_world: float,
    mode_name: str,
    force: bool,
    random_seed: int = 0,
    support_height_world: float = 0.0,
) -> None:
    import torch
    from motionbricks.motion_backbone.demo.clips import clip_holder_G1

    velocity = np.asarray(velocity_world_xy, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    direction = velocity / speed if speed > 1.0e-6 else np.zeros(2)
    mode_index = list(clip_holder_G1.CLIPS.keys()).index(mode_name)
    if isinstance(context_qpos, torch.Tensor):
        context = context_qpos.detach().clone().to(dtype=torch.float32)
    else:
        context = torch.as_tensor(
            np.asarray(context_qpos, dtype=np.float32), dtype=torch.float32
        )
    if context.ndim == 2:
        context = context[None]
    support_height = float(support_height_world)
    if not math.isfinite(support_height):
        raise ValueError("MotionBricks support height must be finite")
    # MotionBricks is trained with its locomotion floor at z=0.  Canonicalize
    # a raised-platform context to that floor, then restore the same support
    # height to its generated root.  Passing absolute elevated z directly
    # makes the learned inbetween collapse the pelvis by exactly the platform
    # height while it tries to return to its training floor.
    context[..., 2] -= support_height
    signals = {
        "movement_direction": torch.as_tensor(
            (direction[0], direction[1], 0.0), dtype=torch.float32
        ).view(1, 3),
        "facing_direction": torch.as_tensor(
            (
                math.cos(float(facing_yaw_world)),
                math.sin(float(facing_yaw_world)),
                0.0,
            ),
            dtype=torch.float32,
        ).view(1, 3),
        "mode": torch.tensor([[mode_index]], dtype=torch.long),
        "target_vel": torch.tensor([speed], dtype=torch.float32),
        "random_seed": torch.tensor([int(random_seed)], dtype=torch.long),
        "allowed_pred_num_tokens": controller.get_default_allowed_pred_num_tokens(
            mode_index
        ),
        "context_mujoco_qpos": context,
    }
    previous_qpos = full_agent.frames.get("mujoco_qpos")
    with torch.no_grad():
        full_agent.generate_new_frames(
            signals,
            controller.get_controller_dt() * 2.0,
            force_generation=force,
        )
        if (
            abs(support_height) > 1.0e-8
            and full_agent.frames["mujoco_qpos"] is not previous_qpos
        ):
            full_agent.frames["mujoco_qpos"][..., 2] += support_height


def _terrain_surface_height(
    terrain: TerrainMeshIndex,
    xy: object,
    *,
    ray_origin_z: float,
) -> float:
    point = np.asarray(xy, dtype=np.float64)
    hit = terrain.raycast(
        np.asarray((point[0], point[1], ray_origin_z)),
        np.asarray((0.0, 0.0, -1.0)),
    )
    return 0.0 if hit is None else float(hit.position_world[2])


def _guard_flat_velocity(
    terrain: TerrainMeshIndex,
    root_xyz: object,
    velocity_world_xy: object,
    *,
    ray_origin_z: float,
) -> tuple[np.ndarray, bool]:
    root = np.asarray(root_xyz, dtype=np.float64)
    velocity = np.asarray(velocity_world_xy, dtype=np.float64)
    if float(np.linalg.norm(velocity)) < 1.0e-6:
        return velocity, False
    current_height = _terrain_surface_height(
        terrain, root[:2], ray_origin_z=ray_origin_z
    )
    future_height = _terrain_surface_height(
        terrain,
        root[:2] + 0.40 * velocity,
        ray_origin_z=ray_origin_z,
    )
    unsafe = bool(
        future_height - current_height > 0.07
        or current_height - future_height > 0.12
    )
    return (np.zeros(2, dtype=np.float64) if unsafe else velocity), unsafe


def _project_live_root_above_support(
    qpos: np.ndarray,
    *,
    sole_adapter: _G1FootfallAdapter,
    terrain: object,
    ray_origin_z: float,
    sole_clearance_m: float = 0.003,
    maximum_lift_m: float = 0.06,
) -> tuple[np.ndarray, float]:
    """Apply the globally privileged game-style root-height projection."""

    value = np.asarray(qpos, dtype=np.float64).copy()
    soles = sole_adapter.sole_support_points_for_pose(
        root_position=value[:3],
        root_quaternion_wxyz=value[3:7],
        joints=mujoco_to_isaaclab_joint_vector(value[7:]),
    )
    required_lifts = [
        _terrain_surface_height(
            terrain, point[:2], ray_origin_z=ray_origin_z
        )
        + float(sole_clearance_m)
        - float(point[2])
        for points in soles
        for point in np.asarray(points, dtype=np.float64)
    ]
    lift = float(
        np.clip(
            max(required_lifts, default=0.0),
            0.0,
            float(maximum_lift_m),
        )
    )
    value[2] += lift
    return value, lift


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
    """Seed the least-disruptive live gait phase after an authored portal."""

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
        future_start = len(context)
        horizon = min(len(generated) - future_start, 24)
        if horizon < 2:
            raise RuntimeError("MotionBricks exit has no future frames")
        review = np.asarray(
            generated[future_start : future_start + horizon], dtype=np.float64
        )
        root_drop = max(0.0, float(context[-1, 2] - np.min(review[:, 2])))
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
        score = (
            4.0 * root_drop
            + 0.20 * double_knee
            + 0.50 * pose_gap
            + 0.20 * joint_step
        )
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
    # The decoded prefix is the conditioning history, not new motion.
    for _ in range(len(context)):
        full_agent.get_next_frame()
    return {
        "selected": selected,
        "candidate_count": len(candidates),
        "score_minimum": float(selected["score"]),
        "score_median": float(
            np.median([value["score"] for value in candidates])
        ),
        "score_maximum": float(max(value["score"] for value in candidates)),
        "discarded_conditioning_frame_count": len(context),
    }


def run(
    *,
    target_clip_index: int | None,
    course_paths: tuple[Path, ...],
    motionbricks_root: Path,
    stairs_archive: Path,
    terrain_usd: Path | None,
    terrain_position_world: tuple[float, float, float],
    terrain_quaternion_world_from_usd_wxyz: tuple[float, float, float, float],
    model_path: Path,
    host: str,
    port: int,
    token: str | None,
    wait_seconds: float,
) -> dict[str, object]:
    import mujoco
    import torch
    from diffusion_policy.inference import (
        sonic_interactive_controller as browser_control,
    )

    motionbricks_root = motionbricks_root.expanduser().resolve()
    for value in (motionbricks_root, motionbricks_root / "scripts"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    _configure_ui(browser_control)
    courses = tuple(
        MotionBricksTerrainCourse.load(path) for path in course_paths
    )
    if not courses:
        raise ValueError("viewer needs at least one terrain course")
    primary_course = courses[0]
    primary_course_qpos = primary_course.native_mujoco_qpos()

    print("[GLOBAL TERRAIN] Loading official MotionBricks checkpoints...", flush=True)
    full_agent, controller = _load_demo(motionbricks_root, seed=7)
    if terrain_usd is None:
        if target_clip_index is None:
            raise ValueError(
                "target_clip_index is required when terrain_usd is omitted"
            )
        import zarr

        archive = zarr.open_group(str(stairs_archive), mode="r")
        terrain_path = Path(
            str(archive["terrain_usd_path"][target_clip_index])
        )
        terrain_transform = RigidTransform(
            np.asarray(archive["terrain_position_env"][target_clip_index]),
            np.asarray(
                archive["terrain_rotation_env_wxyz"][target_clip_index]
            ),
        )
    else:
        terrain_path = terrain_usd.expanduser().resolve()
        terrain_transform = RigidTransform(
            np.asarray(terrain_position_world, dtype=np.float32),
            np.asarray(
                terrain_quaternion_world_from_usd_wxyz,
                dtype=np.float32,
            ),
        )
    terrain_mesh = _load_usd_mesh(
        terrain_path, source_asset_sha256="0" * 64
    )
    terrain_index = TerrainMeshIndex(terrain_mesh, terrain_transform)
    import zarr

    support_archive = zarr.open_group(str(stairs_archive), mode="r")
    sole_adapter = _G1FootfallAdapter(
        model_path,
        tuple(str(value) for value in support_archive["joint_names"][:]),
        maximum_joint_correction_rad=0.1,
    )
    spec = mujoco.MjSpec.from_file(str(model_path.expanduser().resolve()))
    model, visual_mesh_count = _build_scene_model(
        spec, terrain_mesh, terrain_transform
    )
    data = mujoco.MjData(model)
    ray_origin_z = float(np.max(terrain_index.vertices_world[:, 2]) + 2.0)

    state = browser_control.InteractiveControllerState(
        token=token,
        deadman_timeout_s=0.40,
        max_stream_fps=20.0,
        task_dim=4,
        command_profile=browser_control.DIRECT_TWO_STICK_TASK4,
        max_speed_mps=0.90,
        max_yaw_rate_rad_s=1.20,
    )
    server = browser_control.InteractiveControllerServer(
        state, host=host, port=port
    )
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 960)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), 540)
    renderer = mujoco.Renderer(model, height=540, width=960)
    camera = mujoco.MjvCamera()
    camera.distance = 3.1
    camera.azimuth = 135.0
    camera.elevation = -18.0

    current_qpos = primary_course_qpos[0].copy()
    desired_facing_yaw = primary_course.start_yaw_world
    playback: TerrainCoursePlayback | None = None
    active_course: MotionBricksTerrainCourse | None = None
    mode = "flat"
    last_capture = PortalCapture(False, math.inf, math.inf, -1.0, math.inf)
    blocked_by_terrain = False
    tick = 0
    sim_time_s = 0.0
    actual_trail: deque[np.ndarray] = deque(maxlen=200)
    selected_path = primary_course.root_position_world[
        :: max(1, primary_course.frame_count // 80)
    ].copy()
    selected_path[:, 2] += 0.10
    selected_facing = np.tile(np.asarray((1.0, 0.0)), (len(selected_path), 1))
    command_path = np.repeat(current_qpos[None, :3], 4, axis=0)
    command_facing = np.tile(np.asarray((1.0, 0.0)), (4, 1))

    def seed_motionbricks(
        course: MotionBricksTerrainCourse,
        *,
        at_end: bool,
        velocity: np.ndarray,
        facing: float,
        name: str,
    ) -> None:
        nonlocal current_qpos
        context = course.resampled_context_qpos(
            at_end=at_end, target_fps=30.0, frame_count=4
        )
        support_height = _terrain_surface_height(
            terrain_index, current_qpos[:2], ray_origin_z=ray_origin_z
        )
        if at_end:
            phase_result = _submit_phase_matched_course_exit(
                full_agent,
                controller,
                context_qpos=context,
                velocity_world_xy=velocity,
                facing_yaw_world=facing,
                mode_name=name,
                support_height_world=support_height,
            )
            print(
                "[GLOBAL TERRAIN] selected live exit phase "
                f"seed={phase_result['selected']['seed']} "
                f"score={phase_result['score_minimum']:.4f}",
                flush=True,
            )
            return
        _submit_motionbricks(
            full_agent,
            controller,
            context_qpos=context,
            velocity_world_xy=velocity,
            facing_yaw_world=facing,
            mode_name=name,
            force=True,
            support_height_world=support_height,
        )
        for _ in range(len(context)):
            full_agent.get_next_frame()
        current_qpos, _ = _project_live_root_above_support(
            np.asarray(full_agent.get_next_frame(), dtype=np.float64),
            sole_adapter=sole_adapter,
            terrain=terrain_index,
            ray_origin_z=ray_origin_z,
        )

    def reset_session() -> None:
        nonlocal current_qpos, desired_facing_yaw, playback, active_course, mode
        nonlocal tick, sim_time_s, blocked_by_terrain, last_capture
        full_agent.reset()
        current_qpos = primary_course_qpos[0].copy()
        desired_facing_yaw = primary_course.start_yaw_world
        seed_motionbricks(
            primary_course,
            at_end=False,
            velocity=np.zeros(2),
            facing=desired_facing_yaw,
            name="idle",
        )
        playback = None
        active_course = None
        mode = "flat"
        tick = 0
        sim_time_s = 0.0
        blocked_by_terrain = False
        last_capture = PortalCapture(False, math.inf, math.inf, -1.0, math.inf)
        actual_trail.clear()

    def write_qpos() -> None:
        data.qpos[:] = current_qpos
        data.time = sim_time_s
        mujoco.mj_forward(model, data)

    def publish_frame() -> None:
        camera.lookat[:] = current_qpos[:3]
        renderer.update_scene(data, camera=camera)
        _append_overlay_to_scene(
            renderer.scene,
            command_path_world=command_path,
            command_facing_local=command_facing,
            command_root_yaw=_yaw_wxyz(current_qpos[3:7]),
            selected_path_world=selected_path,
            selected_facing_local=selected_facing,
            selected_root_yaw=0.0,
            actual_trail=actual_trail,
            mujoco=mujoco,
        )
        state.publish_frame(
            frame=renderer.render(), physics_step=tick, sim_time_s=sim_time_s
        )

    reset_session()
    write_qpos()
    server.start()
    hostname = socket.gethostname()
    receipt = {
        "schema": "motionbricks_global_terrain_browser_v1",
        "hostname": hostname,
        "port": server.port,
        "token": state.token,
        "url_after_tunnel": f"http://localhost:{server.port}/?token={state.token}",
        "tunnel_command": (
            f"ssh -N -L {server.port}:{hostname}:{server.port} "
            "bodow@scdt.stanford.edu"
        ),
        "kind": "clean_kinematics_not_physics",
        "flat_controller": "official MotionBricks G1",
        "terrain_controller": "precompiled exact-mesh-audited global portal course",
        "target_clip_index": (
            None if target_clip_index is None else int(target_clip_index)
        ),
        "course_motions": [str(course.path) for course in courses],
        "course_frame_counts": [course.frame_count for course in courses],
        "course_fps": [course.fps for course in courses],
        "course_seam_indices": [
            list(course.seam_indices) for course in courses
        ],
        "entry_family_count": len(courses),
        "terrain_usd_path": str(terrain_path),
        "visual_mesh_geom_count": int(visual_mesh_count),
        "global_scene_information_used": True,
        "external_robot_odometry_used": False,
        "switch_controller": "browser Gamepad API",
        "command_frame": "left stick robot-local travel; right stick robot-local facing",
        "route_commitment": "joystick resumes after the audited course landing",
    }
    print(
        "MOTIONBRICKS_GLOBAL_TERRAIN_READY "
        + json.dumps(receipt, sort_keys=True),
        flush=True,
    )
    print(
        "[GLOBAL TERRAIN] Connect the controller, press Start, and push along "
        "the orange entry path.",
        flush=True,
    )

    next_tick = time.perf_counter()
    try:
        publish_frame()
        if not state.wait_for_start(wait_seconds):
            raise TimeoutError("browser did not request Start before timeout")
        while not state.should_end():
            if state.consume_reset_request():
                reset_session()
                write_qpos()

            command, _, _ = state.current_command()
            yaw = _yaw_wxyz(current_qpos[3:7])
            flat_dt = 1.0 / 30.0
            requested_velocity, desired_facing_yaw, target_speed, mode_name = (
                _command_targets(
                    command,
                    current_yaw=yaw,
                    desired_facing_yaw=desired_facing_yaw,
                    dt_s=flat_dt,
                )
            )
            blocked_by_terrain = False
            if playback is None:
                selection = select_terrain_portal(
                    courses, current_qpos, requested_velocity
                )
                candidate_course = courses[selection.course_index]
                last_capture = selection.capture
                selected_path = candidate_course.root_position_world[
                    :: max(1, candidate_course.frame_count // 80)
                ].copy()
                selected_path[:, 2] += 0.10
                selected_facing = np.tile(
                    np.asarray((1.0, 0.0)), (len(selected_path), 1)
                )
                if last_capture.accepted:
                    playback = TerrainCoursePlayback(
                        candidate_course, current_qpos, blend_frames=18
                    )
                    active_course = candidate_course
                    mode = "course"
                    current_qpos = playback.next_qpos()
                    current_qpos, _ = _project_live_root_above_support(
                        current_qpos,
                        sole_adapter=sole_adapter,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                    )
                    dt = 1.0 / candidate_course.fps
                    print(
                        f"frame={tick:06d} PORTAL_COMMIT "
                        f"family={selection.course_index} "
                        f"position_error={last_capture.position_error_m:.3f} "
                        f"yaw_error_deg={math.degrees(last_capture.yaw_error_rad):.1f} "
                        f"pose_rmse={last_capture.lower_body_rmse_rad:.3f}",
                        flush=True,
                    )
                else:
                    guarded_velocity, blocked_by_terrain = _guard_flat_velocity(
                        terrain_index,
                        current_qpos[:3],
                        requested_velocity,
                        ray_origin_z=ray_origin_z,
                    )
                    if blocked_by_terrain:
                        target_speed = 0.0
                        mode_name = "idle"
                    _submit_motionbricks(
                        full_agent,
                        controller,
                        context_qpos=full_agent.get_context_mujoco_qpos(),
                        velocity_world_xy=guarded_velocity,
                        facing_yaw_world=desired_facing_yaw,
                        mode_name=mode_name,
                        force=False,
                        support_height_world=_terrain_surface_height(
                            terrain_index,
                            current_qpos[:2],
                            ray_origin_z=ray_origin_z,
                        ),
                    )
                    current_qpos, _ = _project_live_root_above_support(
                        np.asarray(
                            full_agent.get_next_frame(), dtype=np.float64
                        ),
                        sole_adapter=sole_adapter,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                    )
                    mode = "flat"
                    dt = flat_dt
            else:
                if active_course is None:
                    raise RuntimeError("course playback lost its active course")
                current_qpos = playback.next_qpos()
                if playback.index <= playback.blend_frames:
                    current_qpos, _ = _project_live_root_above_support(
                        current_qpos,
                        sole_adapter=sole_adapter,
                        terrain=terrain_index,
                        ray_origin_z=ray_origin_z,
                    )
                dt = 1.0 / active_course.fps
                if playback.done:
                    seed_motionbricks(
                        active_course,
                        at_end=True,
                        velocity=requested_velocity,
                        facing=desired_facing_yaw,
                        name=mode_name,
                    )
                    playback = None
                    active_course = None
                    mode = "flat"
                    dt = flat_dt
                    print(
                        f"frame={tick:06d} COURSE_COMPLETE joystick_resumed=1",
                        flush=True,
                    )

            tick += 1
            sim_time_s += dt
            write_qpos()
            actual_trail.append(
                np.asarray(
                    (current_qpos[0], current_qpos[1], current_qpos[2] + 0.10)
                )
            )
            horizons = np.asarray((0.0, 0.3, 0.6, 0.9))
            command_path = np.empty((4, 3), dtype=np.float64)
            command_path[:, :2] = (
                current_qpos[:2] + horizons[:, None] * requested_velocity
            )
            command_path[:, 2] = current_qpos[2] + 0.10
            relative_facing = math.remainder(
                desired_facing_yaw - _yaw_wxyz(current_qpos[3:7]),
                2.0 * math.pi,
            )
            command_facing = np.tile(
                np.asarray(
                    (math.cos(relative_facing), math.sin(relative_facing))
                ),
                (4, 1),
            )
            task = np.asarray(
                (
                    requested_velocity[0],
                    requested_velocity[1],
                    math.cos(relative_facing),
                    math.sin(relative_facing),
                )
            )
            state.observe_policy(
                policy_call=tick,
                task=task,
                virtual_position_xy=command_path[-1, :2],
                actual_position_xy=current_qpos[:2],
                virtual_facing_yaw=desired_facing_yaw,
                actual_yaw=_yaw_wxyz(current_qpos[3:7]),
                target_velocity_xy=requested_velocity,
                target_facing_xy=(
                    math.cos(relative_facing), math.sin(relative_facing)
                ),
            )
            state.observe_actor_call(
                policy_call=tick,
                task=task,
                action=np.asarray(
                    (
                        float(mode == "course"),
                        float(blocked_by_terrain),
                        last_capture.position_error_m
                        if math.isfinite(last_capture.position_error_m)
                        else -1.0,
                    )
                ),
            )
            if tick % 2 == 0:
                publish_frame()
            if tick % 150 == 0:
                print(
                    f"frame={tick:06d} mode={mode} "
                    f"blocked={int(blocked_by_terrain)} "
                    f"portal_distance={last_capture.position_error_m:.3f}",
                    flush=True,
                )

            next_tick += dt
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -0.50:
                next_tick = time.perf_counter()
        return receipt
    finally:
        renderer.close()
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-clip-index", type=int)
    parser.add_argument("--course-motion", type=Path, action="append")
    parser.add_argument("--terrain-catalog", type=Path)
    parser.add_argument("--terrain-route-manifest", type=Path)
    parser.add_argument("--terrain-usd", type=Path)
    parser.add_argument(
        "--terrain-position",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--terrain-quaternion-wxyz",
        type=float,
        nargs=4,
        default=(1.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--motionbricks-root", type=Path, default=DEFAULT_MOTIONBRICKS_ROOT
    )
    parser.add_argument("--stairs-archive", type=Path, default=DEFAULT_STAIRS_ARCHIVE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_G1_MJCF)
    parser.add_argument("--browser-host", default="0.0.0.0")
    parser.add_argument("--browser-port", type=int, default=8772)
    parser.add_argument("--browser-token")
    parser.add_argument("--browser-wait-seconds", type=float, default=43200.0)
    arguments = parser.parse_args()
    target_clip_index = arguments.target_clip_index
    manifest_scene = None
    if arguments.terrain_route_manifest is not None:
        if arguments.terrain_usd is not None or arguments.terrain_catalog is not None:
            parser.error(
                "--terrain-route-manifest cannot be combined with "
                "--terrain-usd or --terrain-catalog"
            )
        manifest_scene = _route_manifest(arguments.terrain_route_manifest)
    if (
        arguments.terrain_usd is None
        and manifest_scene is None
        and target_clip_index is None
    ):
        target_clip_index = 26
    course_paths = list(arguments.course_motion or ())
    if arguments.terrain_catalog is not None:
        if target_clip_index is None:
            parser.error("--terrain-catalog requires --target-clip-index")
        course_paths.extend(
            _catalog_course_paths(
                arguments.terrain_catalog, target_clip_index
            )
        )
    if manifest_scene is not None:
        course_paths.extend(manifest_scene[3])
    course_paths = list(dict.fromkeys(course_paths))
    terrain_usd = arguments.terrain_usd
    terrain_position = tuple(arguments.terrain_position)
    terrain_quaternion = tuple(arguments.terrain_quaternion_wxyz)
    if manifest_scene is not None:
        terrain_usd, terrain_position, terrain_quaternion, _ = manifest_scene
    run(
        target_clip_index=target_clip_index,
        course_paths=tuple(course_paths or (DEFAULT_COURSE,)),
        motionbricks_root=arguments.motionbricks_root,
        stairs_archive=arguments.stairs_archive,
        terrain_usd=terrain_usd,
        terrain_position_world=terrain_position,
        terrain_quaternion_world_from_usd_wxyz=terrain_quaternion,
        model_path=arguments.model_path,
        host=arguments.browser_host,
        port=arguments.browser_port,
        token=arguments.browser_token,
        wait_seconds=arguments.browser_wait_seconds,
    )


if __name__ == "__main__":
    main()
