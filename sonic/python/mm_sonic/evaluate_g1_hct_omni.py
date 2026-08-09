"""Headless, scheduled evaluation and state capture for the native G1 HCT policy.

The command suite is deliberately deterministic and body-relative.  Each group
of environments receives one of ten joystick programs, so every generated
terrain column is tested against steady, stop/restart, reversal, strafe, spin,
arc, and abrupt multi-axis commands in a single simulation pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Terrain-G1-HCT-Omni-v0")
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--terrain_seed", type=int, default=42)
    parser.add_argument(
        "--terrain_level",
        type=int,
        default=9,
        help="difficulty row used by every evaluation environment (0-9)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


ARGS = _parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from . import g1_hct_omni_env as hct  # noqa: E402,F401  (registers task)


PROGRAM_NAMES = (
    "steady_forward",
    "steady_backward",
    "left_strafe",
    "right_strafe",
    "diagonal_arc_left",
    "diagonal_arc_right",
    "spin_reversal",
    "forward_stop_backward",
    "abrupt_omni",
    "zigzag_yaw",
)


def _piecewise(seconds: float, pieces: tuple[tuple[float, tuple[float, float, float]], ...]) -> np.ndarray:
    for stop, value in pieces:
        if seconds < stop:
            return np.asarray(value, dtype=np.float32)
    return np.asarray(pieces[-1][1], dtype=np.float32)


def command_program(name: str, seconds: float) -> np.ndarray:
    """Return a robot-local ``(vx, vy, yaw_rate)`` joystick command."""
    if seconds < 1.0:
        return np.zeros(3, dtype=np.float32)
    if name == "steady_forward":
        return np.asarray((0.65, 0.0, 0.0), dtype=np.float32)
    if name == "steady_backward":
        return np.asarray((-0.45, 0.0, 0.0), dtype=np.float32)
    if name == "left_strafe":
        return np.asarray((0.0, 0.35, 0.0), dtype=np.float32)
    if name == "right_strafe":
        return np.asarray((0.0, -0.35, 0.0), dtype=np.float32)
    if name == "diagonal_arc_left":
        return np.asarray((0.55, 0.28, 0.55), dtype=np.float32)
    if name == "diagonal_arc_right":
        return np.asarray((0.55, -0.28, -0.55), dtype=np.float32)
    if name == "spin_reversal":
        return _piecewise(seconds, (
            (4.0, (0.0, 0.0, 0.80)),
            (7.0, (0.0, 0.0, -0.80)),
            (9.0, (0.0, 0.0, 0.0)),
            (1.0e9, (0.20, 0.0, 0.55)),
        ))
    if name == "forward_stop_backward":
        return _piecewise(seconds, (
            (4.0, (0.70, 0.0, 0.0)),
            (6.0, (0.0, 0.0, 0.0)),
            (9.0, (-0.50, 0.0, 0.0)),
            (1.0e9, (0.0, 0.0, 0.0)),
        ))
    if name == "abrupt_omni":
        return _piecewise(seconds, (
            (3.0, (0.60, 0.0, 0.0)),
            (5.0, (0.0, 0.40, 0.0)),
            (7.0, (-0.45, 0.0, 0.0)),
            (9.0, (0.0, -0.40, 0.0)),
            (1.0e9, (0.45, 0.30, 0.55)),
        ))
    if name == "zigzag_yaw":
        phase = int((seconds - 1.0) / 1.5) % 2
        return np.asarray(
            (0.55, 0.32 if phase == 0 else -0.32, -0.60 if phase == 0 else 0.60),
            dtype=np.float32,
        )
    raise ValueError(f"unknown command program: {name}")


def _terrain_column_names() -> np.ndarray:
    config = hct.HCT_OMNI_TERRAINS_CFG
    names = list(config.sub_terrains)
    proportion = np.asarray(
        [config.sub_terrains[name].proportion for name in names], dtype=np.float64
    )
    cumulative = np.cumsum(proportion / proportion.sum())
    result = []
    for column in range(config.num_cols):
        index = int(np.flatnonzero(column / config.num_cols + 0.001 < cumulative)[0])
        result.append(names[index])
    return np.asarray(result)


def _strict_agent_cfg(checkpoint: Path) -> dict:
    cfg = load_cfg_from_registry(ARGS.task, "rsl_rl_cfg_entry_point")
    cfg = cfg if isinstance(cfg, dict) else cfg.to_dict()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    keys = (state.get("model_state_dict") or state).keys()
    detected = "log" if any(key.endswith("log_std") for key in keys) else "scalar"
    cfg.setdefault("policy", {})["noise_std_type"] = detected
    return cfg


def _exact_terrain_mesh(terrain) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact simulator terrain triangles authored by TerrainImporter."""
    if len(terrain.terrain_prim_paths) != 1:
        raise RuntimeError(
            f"expected one generated terrain prim, got {terrain.terrain_prim_paths}"
        )
    stage = omni.usd.get_context().get_stage()
    mesh_path = f"{terrain.terrain_prim_paths[0]}/mesh"
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    if not mesh:
        raise RuntimeError(f"exact simulator terrain mesh is absent: {mesh_path}")
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    if not len(counts) or not np.all(counts == 3):
        raise RuntimeError("generated terrain contains non-triangular faces")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    if not len(vertices) or not len(faces):
        raise RuntimeError("generated terrain mesh is empty")
    return vertices, faces


def main() -> int:
    if ARGS.steps < 2 or ARGS.num_envs < len(PROGRAM_NAMES):
        raise SystemExit("steps must be >=2 and num_envs must cover all ten command programs")
    checkpoint = ARGS.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")

    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.episode_length_s = max(40.0, ARGS.steps * env_cfg.decimation * env_cfg.sim.dt + 2.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.scene.terrain.max_init_terrain_level = hct.HCT_OMNI_TERRAINS_CFG.num_rows - 1
    env_cfg.scene.terrain.terrain_generator.seed = ARGS.terrain_seed
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.zero_component_probability = (0.0, 0.0, 0.0)

    raw_env = gym.make(ARGS.task, cfg=env_cfg)
    raw_terrain = raw_env.unwrapped.scene.terrain
    maximum_level = hct.HCT_OMNI_TERRAINS_CFG.num_rows - 1
    if ARGS.terrain_level < 0 or ARGS.terrain_level > maximum_level:
        raise SystemExit(f"terrain_level must be in [0,{maximum_level}]")
    raw_terrain.terrain_levels.fill_(ARGS.terrain_level)
    raw_terrain.env_origins.copy_(
        raw_terrain.terrain_origins[
            raw_terrain.terrain_levels, raw_terrain.terrain_types
        ]
    )
    # Reset after moving each environment to the requested difficulty row so
    # the robot pose and local sensors are centred on that exact tile.
    raw_env.reset()
    env = RslRlVecEnvWrapper(raw_env)
    runner = OnPolicyRunner(env, _strict_agent_cfg(checkpoint), log_dir=None, device=ARGS.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=ARGS.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    scanner = unwrapped.scene["height_scanner"]
    contacts = unwrapped.scene["contact_forces"]
    command_term = unwrapped.command_manager.get_term("base_velocity")
    command_term.is_standing_env.zero_()
    if hasattr(command_term, "_zero_component"):
        command_term._zero_component.zero_()

    program_index = np.arange(ARGS.num_envs, dtype=np.int64) % len(PROGRAM_NAMES)
    program_name = np.asarray(PROGRAM_NAMES)[program_index]
    terrain = unwrapped.scene.terrain
    terrain_type = terrain.terrain_types.detach().cpu().numpy().copy()
    terrain_level_initial = terrain.terrain_levels.detach().cpu().numpy().copy()
    terrain_env_origin = terrain.env_origins.detach().cpu().numpy().astype(np.float32).copy()
    terrain_vertices_world, terrain_faces = _exact_terrain_mesh(terrain)
    column_name = _terrain_column_names()
    terrain_name = column_name[terrain_type]

    joint_names = np.asarray(robot.joint_names)
    body_names = np.asarray(robot.body_names)
    arm_joint_indices = np.asarray([
        index for index, name in enumerate(robot.joint_names)
        if "shoulder" in name or "elbow" in name
    ], dtype=np.int64)
    default_joint_position = (
        robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
    )
    foot_indices = [
        index for index, name in enumerate(robot.body_names)
        if name.endswith("ankle_roll_link")
    ]
    if len(foot_indices) != 2:
        raise RuntimeError(f"expected two ankle-roll foot bodies, got {foot_indices}")
    contact_foot_indices, contact_foot_names = contacts.find_bodies(
        ".*_ankle_roll_link", preserve_order=True
    )
    if len(contact_foot_indices) != 2:
        raise RuntimeError(
            f"expected two ankle-roll contact bodies, got {contact_foot_names}"
        )

    root_pos = []
    root_quat = []
    root_lin_vel_b = []
    root_ang_vel_b = []
    joint_pos = []
    joint_vel = []
    actions_log = []
    foot_pos = []
    foot_contact = []
    scan_xyz = []
    commands = []
    episode_length = []
    dones_log = []

    for step in range(ARGS.steps):
        seconds = step * float(unwrapped.step_dt)
        command_np = np.stack([
            command_program(name, seconds) for name in program_name
        ])
        command = torch.as_tensor(command_np, device=unwrapped.device)
        command_term.vel_command_b.copy_(command)
        command_term.is_standing_env.zero_()
        if hasattr(command_term, "_zero_component"):
            command_term._zero_component.zero_()

        # Recompute after the explicit write: the policy observation must contain
        # this frame's scheduled joystick value, not the command returned by the
        # previous environment step.
        obs = env.get_observations()
        with torch.inference_mode():
            action = policy(obs)
            _, _, dones, _ = env.step(action)

        commands.append(command_np.copy())
        actions_log.append(action.detach().cpu().numpy().copy())
        dones_log.append(dones.detach().cpu().numpy().astype(bool))
        episode_length.append(unwrapped.episode_length_buf.detach().cpu().numpy().copy())
        root_pos.append(robot.data.root_pos_w.detach().cpu().numpy().copy())
        root_quat.append(robot.data.root_quat_w.detach().cpu().numpy().copy())
        root_lin_vel_b.append(robot.data.root_lin_vel_b.detach().cpu().numpy().copy())
        root_ang_vel_b.append(robot.data.root_ang_vel_b.detach().cpu().numpy().copy())
        joint_pos.append(robot.data.joint_pos.detach().cpu().numpy().copy())
        joint_vel.append(robot.data.joint_vel.detach().cpu().numpy().copy())
        foot_pos.append(robot.data.body_pos_w[:, foot_indices].detach().cpu().numpy().copy())
        # net_forces_w is (environment, body, xyz).  The history-bearing
        # tensor is a separate field; indexing this as if it had a history
        # axis would treat the foot IDs as xyz coordinates.
        forces = contacts.data.net_forces_w[:, contact_foot_indices]
        foot_contact.append((torch.linalg.vector_norm(forces, dim=-1) > 5.0).detach().cpu().numpy())
        scan_xyz.append(scanner.data.ray_hits_w.detach().cpu().numpy().copy())

    root_pos = np.stack(root_pos).astype(np.float32)
    root_quat = np.stack(root_quat).astype(np.float32)
    root_lin_vel_b = np.stack(root_lin_vel_b).astype(np.float32)
    root_ang_vel_b = np.stack(root_ang_vel_b).astype(np.float32)
    joint_pos = np.stack(joint_pos).astype(np.float32)
    joint_vel = np.stack(joint_vel).astype(np.float32)
    foot_pos = np.stack(foot_pos).astype(np.float32)
    foot_contact = np.stack(foot_contact)
    scan_xyz = np.stack(scan_xyz).astype(np.float32)
    commands = np.stack(commands).astype(np.float32)
    episode_length = np.stack(episode_length).astype(np.int32)
    dones = np.stack(dones_log)

    xy_jump = np.linalg.norm(np.diff(root_pos[..., :2], axis=0), axis=-1)
    reset_drop = np.diff(episode_length, axis=0) < 0
    reset = dones.any(axis=0) | reset_drop.any(axis=0) | (xy_jump > 0.5).any(axis=0)
    velocity_error = root_lin_vel_b[..., :2] - commands[..., :2]
    yaw_error = root_ang_vel_b[..., 2] - commands[..., 2]
    command_rmse = np.sqrt(np.mean(np.sum(velocity_error**2, axis=-1) + yaw_error**2, axis=0))
    stance_speed = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1) / float(unwrapped.step_dt)
    stance_mask = foot_contact[1:]
    mean_stance_speed = np.asarray([
        float(stance_speed[:, env_index][stance_mask[:, env_index]].mean())
        if stance_mask[:, env_index].any() else float("inf")
        for env_index in range(ARGS.num_envs)
    ])
    arm_velocity_rms = np.sqrt(
        np.mean(joint_vel[..., arm_joint_indices] ** 2, axis=(0, 2))
    )
    arm_deviation_rms = np.sqrt(
        np.mean(
            (joint_pos[..., arm_joint_indices]
             - default_joint_position[arm_joint_indices]) ** 2,
            axis=(0, 2),
        )
    )
    maximum_joint_step = np.max(np.abs(np.diff(joint_pos, axis=0)), axis=(0, 2))
    clean = (~reset) & np.isfinite(command_rmse)

    report = {
        "checkpoint": str(checkpoint),
        "terrain_seed": ARGS.terrain_seed,
        "terrain_level": ARGS.terrain_level,
        "steps": ARGS.steps,
        "dt": float(unwrapped.step_dt),
        "num_envs": ARGS.num_envs,
        "clean_envs": int(clean.sum()),
        "reset_or_teleport_envs": int(reset.sum()),
        "maximum_xy_step_m": float(xy_jump.max(initial=0.0)),
        "command_rmse_mixed_mean": float(command_rmse[clean].mean()) if clean.any() else None,
        "mean_stance_ankle_speed_mps": float(mean_stance_speed[clean].mean()) if clean.any() else None,
        "arm_velocity_rms_rad_s": float(arm_velocity_rms[clean].mean()) if clean.any() else None,
        "arm_deviation_rms_rad": float(arm_deviation_rms[clean].mean()) if clean.any() else None,
        "maximum_joint_step_rad": float(maximum_joint_step[clean].max()) if clean.any() else None,
        "by_program": {},
        "by_terrain": {},
    }
    for key, labels in (("by_program", program_name), ("by_terrain", terrain_name)):
        for label in sorted(set(labels.tolist())):
            selected = labels == label
            passed = selected & clean
            report[key][label] = {
                "count": int(selected.sum()),
                "clean": int(passed.sum()),
                "command_rmse_mean": float(command_rmse[passed].mean()) if passed.any() else None,
            }

    output = ARGS.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        actions=np.stack(actions_log).astype(np.float32),
        foot_pos=foot_pos,
        foot_contact=foot_contact,
        scan_xyz=scan_xyz,
        commands=commands,
        episode_length=episode_length,
        dones=dones,
        clean_envs=clean,
        command_rmse=command_rmse.astype(np.float32),
        mean_stance_ankle_speed=mean_stance_speed.astype(np.float32),
        arm_velocity_rms=arm_velocity_rms.astype(np.float32),
        arm_deviation_rms=arm_deviation_rms.astype(np.float32),
        maximum_joint_step=maximum_joint_step.astype(np.float32),
        arm_joint_indices=arm_joint_indices,
        default_joint_position=default_joint_position,
        program_name=program_name,
        terrain_name=terrain_name,
        terrain_type=terrain_type,
        terrain_level_initial=terrain_level_initial,
        terrain_env_origin=terrain_env_origin,
        terrain_vertices_world=terrain_vertices_world,
        terrain_faces=terrain_faces,
        joint_names=joint_names,
        body_names=body_names,
        foot_body_indices=np.asarray(foot_indices, dtype=np.int64),
        contact_body_names=np.asarray(contacts.body_names),
        contact_foot_body_indices=np.asarray(contact_foot_indices, dtype=np.int64),
        contact_foot_body_names=np.asarray(contact_foot_names),
        checkpoint=np.asarray(str(checkpoint)),
        terrain_seed=np.int64(ARGS.terrain_seed),
        terrain_level=np.int64(ARGS.terrain_level),
        dt=np.float32(unwrapped.step_dt),
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[g1-hct-eval] wrote {output}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        SIMULATION_APP.close()
