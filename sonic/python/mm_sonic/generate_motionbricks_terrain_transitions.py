"""Generate clean MotionBricks flat approach/exit motions for a stair asset.

This is the GPU half of the globally privileged terrain-course prototype.  It
uses the official MotionBricks G1 generator to follow a smooth, known-world
approach curve and to prepare a flat continuation after the terrain motion.
It does not stitch motions, run physics, or alter the stair asset; the CPU
composer performs phase selection, contact-aware inertialization, and exact
mesh rejection afterwards.

The generated commands are ordinary two-stick commands.  Global root position
is used only by the privileged waypoint driver in this deliberately easy
bootstrap, matching the current global-scene assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


FPS = 30.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw_wxyz(value: np.ndarray) -> float:
    w, x, y, z = (float(component) for component in value)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap(value: float) -> float:
    return math.remainder(float(value), 2.0 * math.pi)


def _rotation_xy(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float64)


def _yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))


def _multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _apply_planar_transform(
    qpos: np.ndarray,
    *,
    yaw_offset: float,
    translation: np.ndarray,
) -> np.ndarray:
    value = np.asarray(qpos, dtype=np.float64).copy()
    if value.shape != (36,) or not np.isfinite(value).all():
        raise ValueError("MotionBricks qpos must contain 36 finite values")
    shift = np.asarray(translation, dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError("planar transform translation must be finite XYZ")
    value[:2] = _rotation_xy(yaw_offset) @ value[:2] + shift[:2]
    value[2] += shift[2]
    value[3:7] = _multiply_wxyz(
        _yaw_quaternion_wxyz(yaw_offset), value[3:7]
    )
    value[3:7] /= np.linalg.norm(value[3:7])
    return value


def _bezier(
    control: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(control, dtype=np.float64)
    u = np.asarray(coordinates, dtype=np.float64)
    if points.shape != (4, 2) or u.ndim != 1:
        raise ValueError("cubic Bezier inputs are invalid")
    one = 1.0 - u
    path = (
        one[:, None] ** 3 * points[0]
        + 3.0 * one[:, None] ** 2 * u[:, None] * points[1]
        + 3.0 * one[:, None] * u[:, None] ** 2 * points[2]
        + u[:, None] ** 3 * points[3]
    )
    tangent = (
        3.0 * one[:, None] ** 2 * (points[1] - points[0])
        + 6.0
        * one[:, None]
        * u[:, None]
        * (points[2] - points[1])
        + 3.0 * u[:, None] ** 2 * (points[3] - points[2])
    )
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(norm, 1.0e-9)
    return path, tangent


def _route_facts(
    stair_motion: Path,
    coverage_summary: Path | None,
    *,
    target_clip_index: int | None,
    traversal: str | None,
) -> dict[str, object]:
    with np.load(stair_motion, allow_pickle=False) as arrays:
        root = np.asarray(arrays["root_position_world"], dtype=np.float64)
        quaternion = np.asarray(
            arrays["root_quaternion_world_wxyz"], dtype=np.float64
        )
    if root.ndim != 2 or root.shape[1:] != (3,) or len(root) < 3:
        raise ValueError("stair motion root trajectory is invalid")
    if coverage_summary is not None:
        coverage = json.loads(coverage_summary.read_text())
        route = coverage.get("target_route")
        if not isinstance(route, dict):
            raise ValueError("coverage summary has no target_route")
        start = np.asarray(route["start_xy"], dtype=np.float64)
        end = np.asarray(route["end_xy"], dtype=np.float64)
        direction = end - start
        selected_target = int(coverage["target_clip_index"])
        selected_traversal = str(coverage["traversal"])
    else:
        if target_clip_index is None or traversal not in ("up", "down"):
            raise ValueError(
                "target clip and traversal are required without coverage summary"
            )
        direction = root[-1, :2] - root[0, :2]
        selected_target = int(target_clip_index)
        selected_traversal = str(traversal)
    direction /= np.linalg.norm(direction)
    return {
        "entry_position": root[0],
        "entry_yaw": _yaw_wxyz(quaternion[0]),
        "exit_position": root[-1],
        "exit_yaw": _yaw_wxyz(quaternion[-1]),
        "route_direction": direction,
        "route_yaw": math.atan2(float(direction[1]), float(direction[0])),
        "target_clip_index": selected_target,
        "traversal": selected_traversal,
    }


def _load_demo(motionbricks_root: Path, seed: int) -> tuple[object, object]:
    scripts = motionbricks_root / "scripts"
    for value in (motionbricks_root, scripts):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))
    from browser_demo_g1 import _demo_arguments  # type: ignore
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # A few Hydra skeleton entries inside the public release remain relative
    # (``out/motionbricks_pose/...``) even though the top-level demo arguments
    # are absolute.  Match the official launcher during construction, then
    # return to the caller's project directory for artifact writes.
    previous_directory = Path.cwd()
    try:
        os.chdir(motionbricks_root)
        demo = navigation_demo(_demo_arguments(motionbricks_root))
    finally:
        os.chdir(previous_directory)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return demo.full_agent, demo.controller


def _submit_command(
    full_agent: object,
    controller: object,
    *,
    movement_direction_world_xy: np.ndarray,
    facing_yaw_world: float,
    target_speed_mps: float,
) -> None:
    from motionbricks.motion_backbone.demo.clips import clip_holder_G1

    movement = np.asarray(movement_direction_world_xy, dtype=np.float64)
    norm = float(np.linalg.norm(movement))
    if movement.shape != (2,) or norm < 1.0e-8:
        mode_name = "idle"
        movement = np.zeros(2, dtype=np.float64)
        speed = 0.0
    else:
        movement /= norm
        speed = float(target_speed_mps)
        mode_name = "slow_walk" if speed < 0.38 else "walk"
    mode_index = list(clip_holder_G1.CLIPS.keys()).index(mode_name)
    signals = {
        "movement_direction": torch.as_tensor(
            (movement[0], movement[1], 0.0), dtype=torch.float32
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
        "random_seed": torch.tensor([0], dtype=torch.long),
        "allowed_pred_num_tokens": controller.get_default_allowed_pred_num_tokens(
            mode_index
        ),
        "context_mujoco_qpos": full_agent.get_context_mujoco_qpos(),
    }
    with torch.no_grad():
        full_agent.generate_new_frames(
            signals, controller.get_controller_dt() * 2.0
        )


def _record_approach(
    full_agent: object,
    controller: object,
    facts: dict[str, object],
    *,
    angle_deg: float,
    distance_m: float,
    speed_mps: float,
    duration_s: float,
) -> dict[str, np.ndarray]:
    entry = np.asarray(facts["entry_position"], dtype=np.float64)
    route_direction = np.asarray(
        facts["route_direction"], dtype=np.float64
    )
    route_yaw = float(facts["route_yaw"])
    approach_yaw = route_yaw + math.radians(float(angle_deg))
    approach_direction = np.asarray(
        (math.cos(approach_yaw), math.sin(approach_yaw)), dtype=np.float64
    )
    start_xy = entry[:2] - float(distance_m) * approach_direction
    post_entry = entry[:2] + 0.48 * route_direction
    control = np.asarray(
        (
            start_xy,
            start_xy + 0.46 * float(distance_m) * approach_direction,
            entry[:2] - 0.34 * route_direction,
            post_entry,
        ),
        dtype=np.float64,
    )
    coordinates = np.linspace(0.0, 1.0, 1601)
    path, tangent = _bezier(control, coordinates)
    arc = np.zeros(len(path), dtype=np.float64)
    arc[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    full_agent.reset()
    first = np.asarray(full_agent.get_next_frame(), dtype=np.float64).reshape(-1)
    internal_initial_yaw = _yaw_wxyz(first[3:7])
    yaw_offset = _wrap(approach_yaw - internal_initial_yaw)
    rotated_initial = _rotation_xy(yaw_offset) @ first[:2]
    translation = np.asarray(
        (
            start_xy[0] - rotated_initial[0],
            start_xy[1] - rotated_initial[1],
            entry[2] - first[2],
        ),
        dtype=np.float64,
    )
    frame_count = int(round(float(duration_s) * FPS))
    qposes: list[np.ndarray] = []
    waypoints: list[np.ndarray] = []
    submitted_directions: list[np.ndarray] = []
    submitted_facing: list[float] = []
    nearest_index = 0
    started = time.monotonic()
    for frame in range(frame_count):
        internal = first if frame == 0 else np.asarray(
            full_agent.get_next_frame(), dtype=np.float64
        ).reshape(-1)
        world = _apply_planar_transform(
            internal, yaw_offset=yaw_offset, translation=translation
        )
        first_search = max(0, nearest_index - 12)
        distance = np.linalg.norm(path[first_search:] - world[:2], axis=1)
        nearest_index = first_search + int(np.argmin(distance))
        lookahead_arc = arc[nearest_index] + 0.34
        lookahead = min(
            len(path) - 1, int(np.searchsorted(arc, lookahead_arc))
        )
        waypoint = path[lookahead]
        delta = waypoint - world[:2]
        if float(np.linalg.norm(delta)) < 1.0e-8:
            direction_world = tangent[lookahead]
        else:
            direction_world = delta / np.linalg.norm(delta)
        facing_yaw_world = math.atan2(
            float(tangent[lookahead, 1]), float(tangent[lookahead, 0])
        )
        direction_internal = _rotation_xy(-yaw_offset) @ direction_world
        _submit_command(
            full_agent,
            controller,
            movement_direction_world_xy=direction_internal,
            facing_yaw_world=_wrap(facing_yaw_world - yaw_offset),
            target_speed_mps=float(speed_mps),
        )
        qposes.append(world.astype(np.float32))
        waypoints.append(waypoint.astype(np.float32))
        submitted_directions.append(direction_world.astype(np.float32))
        submitted_facing.append(facing_yaw_world)
        if frame % int(FPS) == 0 or frame + 1 == frame_count:
            print(
                "MOTIONBRICKS_APPROACH_PROGRESS "
                f"frame={frame + 1}/{frame_count} "
                f"elapsed_s={time.monotonic() - started:.1f} "
                f"distance_to_entry_m="
                f"{np.linalg.norm(world[:2] - entry[:2]):.3f}",
                flush=True,
            )
    return {
        "fps": np.asarray(FPS, dtype=np.float32),
        "qpos": np.asarray(qposes, dtype=np.float32),
        "root_position_world": np.asarray(qposes, dtype=np.float32)[:, :3],
        "root_quaternion_world_wxyz": np.asarray(qposes, dtype=np.float32)[:, 3:7],
        "joint_position_mjcf": np.asarray(qposes, dtype=np.float32)[:, 7:],
        "planned_curve_world_xy": path.astype(np.float32),
        "planned_curve_tangent_world_xy": tangent.astype(np.float32),
        "command_waypoint_world_xy": np.asarray(waypoints, dtype=np.float32),
        "command_movement_direction_world_xy": np.asarray(
            submitted_directions, dtype=np.float32
        ),
        "command_facing_world_yaw": np.asarray(
            submitted_facing, dtype=np.float32
        ),
        "entry_position_world": entry.astype(np.float32),
        "route_direction_world_xy": route_direction.astype(np.float32),
        "approach_angle_deg": np.asarray(angle_deg, dtype=np.float32),
    }


def _record_exit(
    full_agent: object,
    controller: object,
    facts: dict[str, object],
    *,
    speed_mps: float,
    duration_s: float,
) -> dict[str, np.ndarray]:
    exit_position = np.asarray(facts["exit_position"], dtype=np.float64)
    exit_yaw = float(facts["exit_yaw"])
    route_yaw = float(facts["route_yaw"])
    full_agent.reset()
    first = np.asarray(full_agent.get_next_frame(), dtype=np.float64).reshape(-1)
    internal_initial_yaw = _yaw_wxyz(first[3:7])
    yaw_offset = _wrap(exit_yaw - internal_initial_yaw)
    rotated_initial = _rotation_xy(yaw_offset) @ first[:2]
    translation = np.asarray(
        (
            exit_position[0] - rotated_initial[0],
            exit_position[1] - rotated_initial[1],
            exit_position[2] - first[2],
        ),
        dtype=np.float64,
    )
    movement_internal_yaw = _wrap(route_yaw - yaw_offset)
    movement_internal = np.asarray(
        (math.cos(movement_internal_yaw), math.sin(movement_internal_yaw)),
        dtype=np.float64,
    )
    frame_count = int(round(float(duration_s) * FPS))
    qposes: list[np.ndarray] = []
    submitted_facing: list[float] = []
    started = time.monotonic()
    for frame in range(frame_count):
        internal = first if frame == 0 else np.asarray(
            full_agent.get_next_frame(), dtype=np.float64
        ).reshape(-1)
        world = _apply_planar_transform(
            internal, yaw_offset=yaw_offset, translation=translation
        )
        time_s = frame / FPS
        blend = float(np.clip((time_s - 1.0) / 1.5, 0.0, 1.0))
        blend = blend * blend * (3.0 - 2.0 * blend)
        relative = _wrap(route_yaw - exit_yaw)
        facing_world = _wrap(exit_yaw + blend * relative)
        _submit_command(
            full_agent,
            controller,
            movement_direction_world_xy=movement_internal,
            facing_yaw_world=_wrap(facing_world - yaw_offset),
            target_speed_mps=float(speed_mps),
        )
        qposes.append(world.astype(np.float32))
        submitted_facing.append(facing_world)
        if frame % int(FPS) == 0 or frame + 1 == frame_count:
            print(
                "MOTIONBRICKS_EXIT_PROGRESS "
                f"frame={frame + 1}/{frame_count} "
                f"elapsed_s={time.monotonic() - started:.1f} "
                f"distance_from_landing_m="
                f"{np.linalg.norm(world[:2] - exit_position[:2]):.3f}",
                flush=True,
            )
    values = np.asarray(qposes, dtype=np.float32)
    return {
        "fps": np.asarray(FPS, dtype=np.float32),
        "qpos": values,
        "root_position_world": values[:, :3],
        "root_quaternion_world_wxyz": values[:, 3:7],
        "joint_position_mjcf": values[:, 7:],
        "command_movement_direction_world_xy": np.broadcast_to(
            np.asarray(facts["route_direction"], dtype=np.float32),
            (frame_count, 2),
        ).copy(),
        "command_facing_world_yaw": np.asarray(
            submitted_facing, dtype=np.float32
        ),
        "exit_position_world": exit_position.astype(np.float32),
        "route_direction_world_xy": np.asarray(
            facts["route_direction"], dtype=np.float32
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motionbricks-root", type=Path, required=True)
    parser.add_argument("--stair-motion", type=Path, required=True)
    parser.add_argument("--coverage-summary", type=Path)
    parser.add_argument("--target-clip-index", type=int)
    parser.add_argument("--traversal", choices=("up", "down"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approach-angle-deg", type=float, default=45.0)
    parser.add_argument("--approach-distance-m", type=float, default=2.1)
    parser.add_argument("--approach-speed-mps", type=float, default=0.42)
    parser.add_argument("--approach-duration-s", type=float, default=8.0)
    parser.add_argument("--exit-speed-mps", type=float, default=0.38)
    parser.add_argument("--exit-duration-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    for required in (
        arguments.stair_motion,
        arguments.motionbricks_root / "out/G1-clip.ckpt",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("MotionBricks generation requires a CUDA GPU")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    if (
        arguments.coverage_summary is not None
        and not arguments.coverage_summary.is_file()
    ):
        raise FileNotFoundError(arguments.coverage_summary)
    facts = _route_facts(
        arguments.stair_motion,
        arguments.coverage_summary,
        target_clip_index=arguments.target_clip_index,
        traversal=arguments.traversal,
    )
    print("MOTIONBRICKS_MODEL_LOAD_BEGIN", flush=True)
    load_started = time.monotonic()
    full_agent, controller = _load_demo(
        arguments.motionbricks_root, arguments.seed
    )
    print(
        "MOTIONBRICKS_MODEL_LOAD_DONE "
        f"elapsed_s={time.monotonic() - load_started:.1f}",
        flush=True,
    )
    approach = _record_approach(
        full_agent,
        controller,
        facts,
        angle_deg=arguments.approach_angle_deg,
        distance_m=arguments.approach_distance_m,
        speed_mps=arguments.approach_speed_mps,
        duration_s=arguments.approach_duration_s,
    )
    exit_motion = _record_exit(
        full_agent,
        controller,
        facts,
        speed_mps=arguments.exit_speed_mps,
        duration_s=arguments.exit_duration_s,
    )
    approach_path = arguments.output_dir / "approach_raw.npz"
    exit_path = arguments.output_dir / "exit_raw.npz"
    np.savez_compressed(approach_path, **approach)
    np.savez_compressed(exit_path, **exit_motion)
    receipt = {
        "schema": "motionbricks_global_terrain_transitions_v1",
        "generator": "official MotionBricks G1 checkpoints",
        "global_privilege_used_by_waypoint_driver": True,
        "external_robot_odometry_used": False,
        "motionbricks_root": str(arguments.motionbricks_root.resolve()),
        "stair_motion": str(arguments.stair_motion.resolve()),
        "stair_motion_sha256": _sha256(arguments.stair_motion),
        "coverage_summary": (
            None
            if arguments.coverage_summary is None
            else str(arguments.coverage_summary.resolve())
        ),
        "target_clip_index": facts["target_clip_index"],
        "traversal": facts["traversal"],
        "approach_angle_deg": arguments.approach_angle_deg,
        "approach_frames": int(len(approach["qpos"])),
        "exit_frames": int(len(exit_motion["qpos"])),
        "fps": FPS,
        "seed": arguments.seed,
        "approach_raw": str(approach_path.resolve()),
        "exit_raw": str(exit_path.resolve()),
    }
    (arguments.output_dir / "generation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
