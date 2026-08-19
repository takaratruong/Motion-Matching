"""Run one proposed Justin learner-state handoff through the production tracker.

Isaac/SONIC imports stay inside :func:`run` so proposal generation and unit
tests remain usable on CPU login nodes.  The tracker is loaded from the pinned
external repository supplied on the command line; that repository is never
modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

import numpy as np

from .justin_recovery import TRACKER_HISTORY_FRAMES


TRACKER_SCREEN_STEPS = 25
TRACKER_MPJPE_LIMIT_MM = 300.0
STABLE_TOP_TAIL_STEPS = 20
JUSTIN_PLATFORM_ENTRY_X_M = -0.87105
JUSTIN_PLATFORM_ROOT_Z_M = 1.20
JUSTIN_PLATFORM_Y_MIN_M = -4.8529
JUSTIN_PLATFORM_Y_MAX_M = -3.8529
JUSTIN_SECOND_STEP_X_MIN_M = -0.87105
JUSTIN_SECOND_STEP_X_MAX_M = -0.54085
JUSTIN_SECOND_STEP_ROOT_Z_M = 1.08


class TrackerRecoveryError(RuntimeError):
    pass


@dataclass
class AdaptiveReferenceClock:
    """Slow a reference when the robot lags without trapping it on one pose."""

    freeze_root_error_m: float = 0.20
    resume_root_error_m: float = 0.12
    max_consecutive_hold_steps: int = 1
    frozen: bool = False
    hold_steps: int = 0
    terminal_hold_steps: int = 0
    freeze_events: int = 0
    resume_events: int = 0
    forced_advance_steps: int = 0
    longest_hold_steps: int = 0
    _current_hold_steps: int = 0

    def __post_init__(self) -> None:
        if not (
            np.isfinite(self.freeze_root_error_m)
            and np.isfinite(self.resume_root_error_m)
            and 0.0 < self.resume_root_error_m < self.freeze_root_error_m
        ):
            raise TrackerRecoveryError(
                "adaptive clock needs 0 < resume error < freeze error"
            )
        if self.max_consecutive_hold_steps < 1:
            raise TrackerRecoveryError(
                "adaptive clock needs at least one consecutive hold step"
            )

    def should_advance(self, root_error_m: float, *, at_terminal: bool) -> bool:
        error = float(root_error_m)
        if not np.isfinite(error) or error < 0.0:
            raise TrackerRecoveryError("adaptive clock root error is invalid")
        if at_terminal:
            advance = False
            self.terminal_hold_steps += 1
        else:
            if self.frozen and error <= self.resume_root_error_m:
                self.frozen = False
                self.resume_events += 1
            elif not self.frozen and error >= self.freeze_root_error_m:
                self.frozen = True
                self.freeze_events += 1
            if self.frozen and (
                self._current_hold_steps >= self.max_consecutive_hold_steps
            ):
                # A tracker policy expects a moving gait command.  Continue at
                # a bounded reduced rate instead of holding one pose forever.
                advance = True
                self.forced_advance_steps += 1
            else:
                advance = not self.frozen
        if advance:
            self._current_hold_steps = 0
        else:
            self.hold_steps += 1
            self._current_hold_steps += 1
            self.longest_hold_steps = max(
                self.longest_hold_steps, self._current_hold_steps
            )
        return advance

    def receipt(self) -> dict[str, object]:
        return {
            "mode": "root_error_hysteresis",
            "freeze_root_error_m": self.freeze_root_error_m,
            "resume_root_error_m": self.resume_root_error_m,
            "max_consecutive_hold_steps": self.max_consecutive_hold_steps,
            "hold_steps": self.hold_steps,
            "terminal_hold_steps": self.terminal_hold_steps,
            "freeze_events": self.freeze_events,
            "resume_events": self.resume_events,
            "forced_advance_steps": self.forced_advance_steps,
            "longest_hold_steps": self.longest_hold_steps,
            "frozen_at_end": self.frozen,
        }


@dataclass
class SecondStageRematchTrigger:
    """Fire once after the robot is stably established on Justin's second step."""

    stable_steps: int = 5
    consecutive_steps: int = 0
    triggered: bool = False

    def __post_init__(self) -> None:
        if self.stable_steps < 1:
            raise TrackerRecoveryError(
                "second-stage rematch needs a positive stable-step count"
            )

    def observe(self, root_position_m: np.ndarray) -> bool:
        root = np.asarray(root_position_m, dtype=np.float64)
        if root.shape != (3,) or not np.isfinite(root).all():
            raise TrackerRecoveryError("second-stage rematch root position is invalid")
        on_second_step = bool(
            JUSTIN_SECOND_STEP_X_MIN_M <= root[0] <= JUSTIN_SECOND_STEP_X_MAX_M
            and JUSTIN_PLATFORM_Y_MIN_M <= root[1] <= JUSTIN_PLATFORM_Y_MAX_M
            and root[2] >= JUSTIN_SECOND_STEP_ROOT_Z_M
        )
        self.consecutive_steps = self.consecutive_steps + 1 if on_second_step else 0
        if not self.triggered and self.consecutive_steps >= self.stable_steps:
            self.triggered = True
            return True
        return False


def select_second_stage_reference_frame(
    *,
    robot_root_position_m: np.ndarray,
    robot_joint_position_rad: np.ndarray,
    candidate_frames: np.ndarray,
    reference_root_position_m: np.ndarray,
    reference_joint_position_rad: np.ndarray,
    joint_weight_m_per_rad: float,
) -> dict[str, float | int]:
    """Select a moving step-two reference pose using root and joint proximity."""

    robot_root = np.asarray(robot_root_position_m, dtype=np.float64)
    robot_joint = np.asarray(robot_joint_position_rad, dtype=np.float64)
    frames = np.asarray(candidate_frames, dtype=np.int64)
    reference_root = np.asarray(reference_root_position_m, dtype=np.float64)
    reference_joint = np.asarray(reference_joint_position_rad, dtype=np.float64)
    if robot_root.shape != (3,) or robot_joint.ndim != 1:
        raise TrackerRecoveryError("second-stage robot state has invalid shapes")
    if (
        frames.ndim != 1
        or len(frames) == 0
        or reference_root.shape != (len(frames), 3)
        or reference_joint.shape != (len(frames), len(robot_joint))
    ):
        raise TrackerRecoveryError("second-stage reference search has invalid shapes")
    if (
        not np.isfinite(robot_root).all()
        or not np.isfinite(robot_joint).all()
        or not np.isfinite(reference_root).all()
        or not np.isfinite(reference_joint).all()
        or not np.isfinite(joint_weight_m_per_rad)
        or joint_weight_m_per_rad < 0.0
    ):
        raise TrackerRecoveryError("second-stage reference search contains invalid values")
    root_error = np.linalg.norm(reference_root - robot_root[None, :], axis=1)
    joint_rmse = np.sqrt(
        np.mean((reference_joint - robot_joint[None, :]) ** 2, axis=1)
    )
    score = root_error + float(joint_weight_m_per_rad) * joint_rmse
    index = int(np.argmin(score))
    return {
        "frame": int(frames[index]),
        "score_m": float(score[index]),
        "root_position_error_m": float(root_error[index]),
        "joint_position_rmse_rad": float(joint_rmse[index]),
    }


def _kit_cache_args(cache_path: Path) -> str:
    cache = cache_path.resolve()
    if any(character.isspace() for character in str(cache)):
        raise TrackerRecoveryError("Kit cache path must not contain whitespace")
    return f"--/UJITSO/datastore/localCachePath={cache / 'DerivedDataCache'}"


def _compatible_urdf_extension_name(name: str) -> str:
    # This IsaacLab checkout asks for 2.4.31, while Justin's pinned Isaac Sim
    # environment contains the otherwise API-compatible 2.4.30 package.
    if name == "isaacsim.asset.importer.urdf-2.4.31":
        return "isaacsim.asset.importer.urdf-2.4.30"
    return name


def _patch_isaaclab_urdf_pin() -> None:
    """Adapt IsaacLab 2.3.1 to the URDF importer shipped by Isaac Sim 5.1.

    NVIDIA's 5.1 pip package ships importer 2.4.30, while this IsaacLab checkout
    requests 2.4.31 and calls one setter that 2.4.30 does not expose.  All other
    converter settings are preserved below; only the unavailable supplemental
    fixed-joint inertia setter is omitted.
    """
    from isaaclab.sim.converters import urdf_converter

    original_enable = urdf_converter.enable_extension

    def enable_installed_version(name: str):
        return original_enable(_compatible_urdf_extension_name(name))

    def get_import_config_compatible(self):
        _, import_config = urdf_converter.omni.kit.commands.execute(
            "URDFCreateImportConfig"
        )
        import_config.set_distance_scale(1.0)
        import_config.set_make_default_prim(True)
        import_config.set_create_physics_scene(False)
        import_config.set_density(self.cfg.link_density)
        import_config.set_convex_decomp(
            self.cfg.collider_type == "convex_decomposition"
        )
        import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
        import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
        import_config.set_fix_base(self.cfg.fix_base)
        import_config.set_self_collision(self.cfg.self_collision)
        import_config.set_parse_mimic(
            self.cfg.convert_mimic_joints_to_normal_joints
        )
        import_config.set_replace_cylinders_with_capsules(
            self.cfg.replace_cylinders_with_capsules
        )
        return import_config

    urdf_converter.enable_extension = enable_installed_version
    urdf_converter.UrdfConverter._get_urdf_import_config = get_import_config_compatible


def _disable_builtin_failure_terminations(env) -> tuple[str, ...]:
    """Let the explicit 25-step/300 mm recovery screen own failure decisions.

    ManagerBasedRLEnv auto-resets a terminated environment before returning its
    observation.  Leaving the training-time tracking thresholds active would
    therefore replace the injected learner state with a clean reference reset
    on the first bad step and invalidate the test.
    """
    import torch

    manager = env.env.termination_manager

    def never_terminate(raw_env, **_kwargs):
        return torch.zeros(raw_env.num_envs, dtype=torch.bool, device=raw_env.device)

    disabled: list[str] = []
    for name in manager.active_terms:
        cfg = manager.get_term_cfg(name)
        if cfg.time_out:
            continue
        cfg = copy.copy(cfg)
        cfg.func = never_terminate
        manager.set_term_cfg(name, cfg)
        disabled.append(name)
    return tuple(disabled)


@contextmanager
def _isolated_kit_cache(cache_path: Path):
    """Append per-task Kit paths without editing the production tracker.

    ``build_env`` owns AppLauncher and hard-codes its base ``kit_args``.  A
    temporary constructor wrapper is therefore the narrowest way to append the
    UJITSO and user-config paths before SimulationApp starts.  The wrapper is
    restored immediately after environment construction.
    """
    cache_path.mkdir(parents=True, exist_ok=True)
    import warp as wp

    wp.init()
    from isaaclab.app import AppLauncher

    original_init = AppLauncher.__init__
    extra = _kit_cache_args(cache_path)

    def init_with_cache(self, launcher_args=None, **kwargs):
        if isinstance(launcher_args, argparse.Namespace):
            launcher_args.kit_args = (
                f"{getattr(launcher_args, 'kit_args', '')} {extra}"
            ).strip()
        elif isinstance(launcher_args, dict):
            launcher_args["kit_args"] = (
                f"{launcher_args.get('kit_args', '')} {extra}"
            ).strip()
        else:
            kwargs["kit_args"] = f"{kwargs.get('kit_args', '')} {extra}".strip()
        return original_init(self, launcher_args, **kwargs)

    AppLauncher.__init__ = init_with_cache
    try:
        yield
    finally:
        AppLauncher.__init__ = original_init


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_proposal_candidate(
    proposal: Mapping[str, object], *, query_index: int, candidate_index: int
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if proposal.get("schema") not in {
        "justin-s13-tracker-recovery-proposal/v1",
        "justin-s13-tracker-recovery-proposal/v2",
        "justin-s13-tracker-recovery-proposal/v3",
    }:
        raise TrackerRecoveryError("unsupported recovery proposal schema")
    try:
        query = proposal["queries"][int(query_index)]  # type: ignore[index]
        candidate = query["candidates"][int(candidate_index)]  # type: ignore[index]
    except (IndexError, KeyError, TypeError) as error:
        raise TrackerRecoveryError("proposal query/candidate index is out of range") from error
    return query, candidate


def evaluate_tracker_attempt(
    *,
    mpjpe_mm: np.ndarray,
    root_position_m: np.ndarray,
    terminated_early: bool,
    screen_steps: int = TRACKER_SCREEN_STEPS,
    completed_reference: bool,
) -> dict[str, object]:
    mpjpe = np.asarray(mpjpe_mm, dtype=np.float64)
    root_position = np.asarray(root_position_m, dtype=np.float64)
    if (
        mpjpe.ndim != 1
        or root_position.shape != (len(mpjpe), 3)
        or len(mpjpe) == 0
    ):
        raise TrackerRecoveryError(
            "tracker metrics must contain MPJPE [T] and root position [T, 3]"
        )
    if not np.isfinite(mpjpe).all() or not np.isfinite(root_position).all():
        raise TrackerRecoveryError("tracker metrics contain non-finite values")
    screened = len(mpjpe) >= int(screen_steps)
    screen_peak = float(mpjpe[: min(len(mpjpe), int(screen_steps))].max())
    screen_pass = bool(
        screened and not terminated_early and screen_peak <= TRACKER_MPJPE_LIMIT_MM
    )
    tail = root_position[-min(len(root_position), STABLE_TOP_TAIL_STEPS) :]
    tail_on_platform = (
        (tail[:, 0] <= JUSTIN_PLATFORM_ENTRY_X_M)
        & (tail[:, 1] >= JUSTIN_PLATFORM_Y_MIN_M)
        & (tail[:, 1] <= JUSTIN_PLATFORM_Y_MAX_M)
        & (tail[:, 2] >= JUSTIN_PLATFORM_ROOT_Z_M)
    )
    reached_platform = bool(
        np.any(
            (root_position[:, 0] <= JUSTIN_PLATFORM_ENTRY_X_M)
            & (root_position[:, 1] >= JUSTIN_PLATFORM_Y_MIN_M)
            & (root_position[:, 1] <= JUSTIN_PLATFORM_Y_MAX_M)
            & (root_position[:, 2] >= JUSTIN_PLATFORM_ROOT_Z_M)
        )
    )
    stable_platform = bool(
        completed_reference
        and len(tail) == STABLE_TOP_TAIL_STEPS
        and bool(tail_on_platform.all())
        and float(tail[:, 2].std()) <= 0.08
    )
    full_pass = bool(
        screen_pass
        and completed_reference
        and not terminated_early
        and float(mpjpe.max()) <= TRACKER_MPJPE_LIMIT_MM
        and stable_platform
    )
    return {
        "screened": screened,
        "screen_steps": int(screen_steps),
        "screen_peak_mpjpe_mm": screen_peak,
        "screen_pass": screen_pass,
        "completed_reference": bool(completed_reference),
        "terminated_early": bool(terminated_early),
        "peak_mpjpe_mm": float(mpjpe.max()),
        "mean_mpjpe_mm": float(mpjpe.mean()),
        "reached_platform": reached_platform,
        "stable_platform": stable_platform,
        # Retain the old keys as explicit compatibility aliases.  Their
        # semantics are now the strict geometry-aware platform test.
        "stable_top": stable_platform,
        "stable_top_tail_min_root_x_m": float(tail[:, 0].min()),
        "stable_top_tail_max_root_x_m": float(tail[:, 0].max()),
        "stable_top_tail_min_root_z_m": float(tail[:, 2].min()),
        "stable_top_tail_std_root_z_m": float(tail[:, 2].std()),
        "accepted_recovery": full_pass,
    }


def evaluate_second_stage_recovery(
    *,
    mpjpe_mm: np.ndarray,
    root_position_m: np.ndarray,
    rematch_step: int,
    terminated_early: bool,
    completed_reference: bool,
) -> dict[str, object]:
    """Evaluate only the recovery suffix produced by a second-stage rematch."""

    mpjpe = np.asarray(mpjpe_mm)
    root_position = np.asarray(root_position_m)
    start = int(rematch_step)
    if start < 0 or start >= len(mpjpe):
        raise TrackerRecoveryError("second-stage evaluation rematch step is invalid")
    result = evaluate_tracker_attempt(
        mpjpe_mm=mpjpe[start:],
        root_position_m=root_position[start:],
        terminated_early=terminated_early,
        completed_reference=completed_reference,
    )
    result["rematch_step"] = start
    result["suffix_steps"] = len(mpjpe) - start
    result["accepted_second_stage_recovery"] = result["accepted_recovery"]
    return result


def _build_single_clip_bundle(
    *, tracker_repo: Path, reference_npz: Path, output: Path
) -> tuple[Path, Path, str]:
    """Build the tracker-native PKL without changing the production repository."""
    import joblib

    from sonic_port.takara.csv_to_motion_lib import (
        CSV_JOINT_ORDER,
        build_entry,
        csv_joint_permutation,
    )
    from sonic_port.takara.mjcf_spec import load_spec

    clip = reference_npz.stem
    csv_path = reference_npz.parent.parent / f"{clip}.csv"
    if not csv_path.is_file():
        raise TrackerRecoveryError(f"missing native Justin CSV: {csv_path}")
    object_usd = tracker_repo / "data/sonic_stairs/staircase13x7platform.usd"
    if not object_usd.is_file():
        raise TrackerRecoveryError(f"missing Justin tracker USD: {object_usd}")
    bundle = output / "tracker_bundle"
    robot_dir = bundle / "robot"
    object_dir = bundle / "object_usd"
    robot_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    permutation = csv_joint_permutation(CSV_JOINT_ORDER, spec.joint_names)
    csv = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    quat = csv[:, 3:7]
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    entry = build_entry(
        csv[:, :3], quat, csv[:, 7:36][:, permutation], spec.dof_axis, 30.0
    )
    motion_file = bundle / f"motion_lib_{clip}.pkl"
    joblib.dump({clip: entry}, motion_file)
    joblib.dump({clip: entry}, robot_dir / f"{clip}.pkl")
    shutil.copy2(object_usd, object_dir / f"{clip}.usd")
    shutil.copy2(object_usd, bundle / "flat_placeholder.usd")
    (bundle / "clips.json").write_text(
        json.dumps(
            [{"stem": clip, "terrain": "stairs", "n_frames": len(csv), "fps": 30}],
            indent=2,
        )
        + "\n"
    )
    return bundle, motion_file, clip


def _place_identity_object(env, motion_lib, device: str) -> None:
    import torch

    if "object" not in env.env.scene.rigid_objects:
        raise TrackerRecoveryError("tracker environment has no Justin staircase object")
    obj = env.env.scene["object"]
    origin = env.env.scene.env_origins[0].to(torch.float32)
    quat = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
    obj.write_root_pose_to_sim(torch.cat((origin, quat)).unsqueeze(0))
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    for attribute in ("_motion_object_root_pos", "_motion_object_root_quat"):
        if hasattr(motion_lib, attribute):
            buffer = getattr(motion_lib, attribute)
            value = origin if attribute.endswith("pos") else quat
            buffer[...] = value.view(*([1] * (buffer.ndim - 1)), value.shape[0]).to(buffer.dtype)


def validate_reference_blend(value: float) -> float:
    blend = float(value)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise TrackerRecoveryError("reference blend must be finite and in [0, 1]")
    return blend


_LEARNER_STATE_SHAPES = {
    "root_pose_wxyz": (7,),
    "root_velocity_world": (6,),
    "joint_pos_isaac": (29,),
    "joint_vel_isaac": (29,),
}


def validate_exact_learner_history(query, candidate):
    """Validate and materialize the chronological v3 tracker handoff history."""

    rows = query.get("learner_tracker_history")
    if not isinstance(rows, list) or len(rows) != TRACKER_HISTORY_FRAMES:
        raise TrackerRecoveryError(
            f"exact learner handoff needs {TRACKER_HISTORY_FRAMES} history frames"
        )
    query_index = int(query["trace_index"])
    expected_indices = list(
        range(query_index - TRACKER_HISTORY_FRAMES + 1, query_index + 1)
    )
    materialized = []
    for expected_index, row in zip(expected_indices, rows, strict=True):
        if not isinstance(row, Mapping) or int(row.get("trace_index", -1)) != expected_index:
            raise TrackerRecoveryError("learner tracker history is not contiguous")
        state_json = row.get("state")
        if not isinstance(state_json, Mapping) or set(state_json) != set(
            _LEARNER_STATE_SHAPES
        ):
            raise TrackerRecoveryError("learner tracker history state schema drifted")
        state = {}
        for name, shape in _LEARNER_STATE_SHAPES.items():
            value = np.asarray(state_json[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise TrackerRecoveryError(
                    f"learner tracker history {name} has invalid shape or values"
                )
            state[name] = value
        action = np.asarray(
            row.get("last_action_isaac_normalized"), dtype=np.float32
        )
        if action.shape != (29,) or not np.isfinite(action).all():
            raise TrackerRecoveryError("learner tracker history action is invalid")
        lag = query_index - expected_index
        reference_frame = int(candidate["frame"]) - lag
        if reference_frame < 0:
            raise TrackerRecoveryError("matched reference does not cover tracker history")
        materialized.append(
            {
                "trace_index": expected_index,
                "reference_frame": reference_frame,
                "state": state,
                "last_action_isaac_normalized": action,
            }
        )
    current = query.get("learner_state")
    if not isinstance(current, Mapping):
        raise TrackerRecoveryError("proposal has no current learner state")
    for name in _LEARNER_STATE_SHAPES:
        if not np.allclose(
            materialized[-1]["state"][name],
            np.asarray(current[name], dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise TrackerRecoveryError("current learner state differs from history tail")
    return tuple(materialized)


def _slerp_wxyz(start, end, amount: float):
    """Shortest-arc spherical interpolation for two scalar-first quaternions."""
    import torch

    start = start / start.norm().clamp_min(1.0e-8)
    end = end / end.norm().clamp_min(1.0e-8)
    dot = torch.dot(start, end)
    if bool((dot < 0.0).item()):
        end = -end
        dot = -dot
    dot = dot.clamp(-1.0, 1.0)
    if bool((dot > 0.9995).item()):
        result = start + float(amount) * (end - start)
        return result / result.norm().clamp_min(1.0e-8)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    result = (
        torch.sin((1.0 - float(amount)) * theta) / sin_theta * start
        + torch.sin(float(amount) * theta) / sin_theta * end
    )
    return result / result.norm().clamp_min(1.0e-8)


def _write_tracker_state(
    env,
    robot,
    motion_cmd,
    *,
    state,
    reference_frame: int,
    last_action,
    previous_action,
    device: str,
):
    """Repose one saved learner row without taking a physics step."""

    import torch

    motion_cmd.motion_ids.fill_(0)
    motion_cmd.motion_start_time_steps.fill_(int(reference_frame))
    motion_cmd.time_steps.zero_()
    root_pose = torch.as_tensor(
        state["root_pose_wxyz"], device=device, dtype=torch.float32
    )
    root_velocity = torch.as_tensor(
        state["root_velocity_world"], device=device, dtype=torch.float32
    )
    joint_pos = torch.as_tensor(
        state["joint_pos_isaac"], device=device, dtype=torch.float32
    )
    joint_vel = torch.as_tensor(
        state["joint_vel_isaac"], device=device, dtype=torch.float32
    )
    robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
    robot.write_root_velocity_to_sim(root_velocity.unsqueeze(0))
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))

    action_manager = env.env.action_manager
    action = torch.as_tensor(last_action, device=device, dtype=torch.float32).unsqueeze(0)
    previous = torch.as_tensor(
        previous_action, device=device, dtype=torch.float32
    ).unsqueeze(0)
    if action_manager.action.shape != action.shape:
        raise TrackerRecoveryError("tracker and learner action dimensions differ")
    action_manager._action.copy_(action)
    action_manager._prev_action.copy_(previous)
    env.env.scene.write_data_to_sim()
    env.env.sim.forward()
    env.env.scene.update(dt=env.env.step_dt)
    return root_pose, root_velocity, joint_pos, joint_vel


def _inject_exact_learner_handoff(
    env, robot, motion_cmd, query, candidate, device: str
):
    """Install exact learner state and the actor's ten-frame input history."""

    import torch

    history = validate_exact_learner_history(query, candidate)
    observation_manager = env.env.observation_manager
    buffers_by_group = observation_manager._group_obs_term_history_buffer
    for group_buffers in buffers_by_group.values():
        for buffer in group_buffers.values():
            buffer.reset()

    raw = None
    final_tensors = None
    for index, row in enumerate(history):
        previous_action = (
            history[index - 1]["last_action_isaac_normalized"]
            if index > 0
            else row["last_action_isaac_normalized"]
        )
        final_tensors = _write_tracker_state(
            env,
            robot,
            motion_cmd,
            state=row["state"],
            reference_frame=row["reference_frame"],
            last_action=row["last_action_isaac_normalized"],
            previous_action=previous_action,
            device=device,
        )
        raw = observation_manager.compute(update_history=True)
    assert raw is not None and final_tensors is not None

    policy_buffers = buffers_by_group.get("policy", {})
    term_lengths = {
        name: int(buffer.current_length[0].item())
        for name, buffer in policy_buffers.items()
    }
    incomplete = {
        name: length
        for name, length in term_lengths.items()
        if length != TRACKER_HISTORY_FRAMES
    }
    if incomplete:
        raise TrackerRecoveryError(f"tracker observation history is incomplete: {incomplete}")
    if "actions" not in policy_buffers:
        raise TrackerRecoveryError("tracker policy has no action history buffer")
    expected_actions = torch.as_tensor(
        np.stack(
            [row["last_action_isaac_normalized"] for row in history], axis=0
        ),
        device=device,
        dtype=torch.float32,
    )
    actual_actions = policy_buffers["actions"].buffer[0]
    action_history_max_abs_error = float(
        torch.max(torch.abs(actual_actions - expected_actions)).item()
    )
    if action_history_max_abs_error > 1.0e-6:
        raise TrackerRecoveryError("tracker action history differs from learner trace")

    root_pose, _root_velocity, joint_pos, _joint_vel = final_tensors
    reference_root_pose = torch.cat(
        (motion_cmd.body_pos_w[0, 0], motion_cmd.body_quat_w[0, 0])
    )
    reference_joint_pos = motion_cmd.joint_pos[0]
    injected_body_error = (
        (motion_cmd.body_pos_w - motion_cmd.robot_body_pos_w).norm(dim=-1) * 1000.0
    )
    quat_dot = torch.dot(root_pose[3:], reference_root_pose[3:]).abs().clamp(0.0, 1.0)
    diagnostics = {
        "reference_blend": 0.0,
        "learner_weight": 1.0,
        "injected_mpjpe_mm": float(injected_body_error.mean(dim=-1)[0].item()),
        "injected_root_position_error_m": float(
            (root_pose[:3] - reference_root_pose[:3]).norm().item()
        ),
        "injected_root_rotation_error_rad": float((2.0 * torch.acos(quat_dot)).item()),
        "injected_joint_position_rmse_rad": float(
            torch.sqrt(torch.mean((joint_pos - reference_joint_pos) ** 2)).item()
        ),
    }
    history_receipt = {
        "frames": len(history),
        "trace_indices": [row["trace_index"] for row in history],
        "reference_frames": [row["reference_frame"] for row in history],
        "policy_term_lengths": term_lengths,
        "action_history_max_abs_error": action_history_max_abs_error,
        "source": "exact learner trace",
    }
    return (
        env.process_raw_obs(raw, flatten_dict_obs=True),
        diagnostics,
        history_receipt,
    )


def _inject_recovery_state(
    env,
    robot,
    motion_cmd,
    query,
    candidate,
    device: str,
    *,
    reference_blend: float,
):
    import torch

    reference_blend = validate_reference_blend(reference_blend)
    state = query["learner_state"]
    learner_root_pose = torch.as_tensor(
        state["root_pose_wxyz"], device=device, dtype=torch.float32
    )
    learner_root_velocity = torch.as_tensor(
        state["root_velocity_world"], device=device, dtype=torch.float32
    )
    learner_joint_pos = torch.as_tensor(
        state["joint_pos_isaac"], device=device, dtype=torch.float32
    )
    learner_joint_vel = torch.as_tensor(
        state["joint_vel_isaac"], device=device, dtype=torch.float32
    )
    if tuple(learner_root_pose.shape) != (7,) or tuple(learner_root_velocity.shape) != (6,):
        raise TrackerRecoveryError("proposal learner root state has drifted")
    if tuple(learner_joint_pos.shape) != (29,) or tuple(learner_joint_vel.shape) != (29,):
        raise TrackerRecoveryError("proposal learner joint state has drifted")

    frame = int(candidate["frame"])
    motion_cmd.motion_ids.fill_(0)
    motion_cmd.motion_start_time_steps.fill_(frame)
    motion_cmd.time_steps.zero_()
    reference_root_pose = torch.cat(
        (motion_cmd.body_pos_w[0, 0], motion_cmd.body_quat_w[0, 0])
    )
    reference_root_velocity = torch.cat(
        (motion_cmd.body_lin_vel_w[0, 0], motion_cmd.body_ang_vel_w[0, 0])
    )
    reference_joint_pos = motion_cmd.joint_pos[0]
    reference_joint_vel = motion_cmd.joint_vel[0]
    if tuple(reference_joint_pos.shape) != (29,) or tuple(reference_joint_vel.shape) != (29,):
        raise TrackerRecoveryError("tracker reference joint state has drifted")

    root_pose = torch.empty_like(learner_root_pose)
    root_pose[:3] = torch.lerp(
        learner_root_pose[:3], reference_root_pose[:3], reference_blend
    )
    root_pose[3:] = _slerp_wxyz(
        learner_root_pose[3:], reference_root_pose[3:], reference_blend
    )
    root_velocity = torch.lerp(
        learner_root_velocity, reference_root_velocity, reference_blend
    )
    joint_pos = torch.lerp(learner_joint_pos, reference_joint_pos, reference_blend)
    joint_vel = torch.lerp(learner_joint_vel, reference_joint_vel, reference_blend)

    robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
    robot.write_root_velocity_to_sim(root_velocity.unsqueeze(0))
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))
    env.env.scene.write_data_to_sim()
    env.env.sim.forward()
    env.env.scene.update(dt=env.env.step_dt)
    # A fresh observation is required after the teleport.  Reusing reset()
    # observations would silently test a clean reference state instead.
    raw = env.env.observation_manager.compute()
    injected_body_error = (
        (motion_cmd.body_pos_w - motion_cmd.robot_body_pos_w).norm(dim=-1) * 1000.0
    )
    quat_dot = torch.dot(root_pose[3:], reference_root_pose[3:]).abs().clamp(0.0, 1.0)
    diagnostics = {
        "reference_blend": reference_blend,
        "learner_weight": 1.0 - reference_blend,
        "injected_mpjpe_mm": float(injected_body_error.mean(dim=-1)[0].item()),
        "injected_root_position_error_m": float(
            (root_pose[:3] - reference_root_pose[:3]).norm().item()
        ),
        "injected_root_rotation_error_rad": float((2.0 * torch.acos(quat_dot)).item()),
        "injected_joint_position_rmse_rad": float(
            torch.sqrt(torch.mean((joint_pos - reference_joint_pos) ** 2)).item()
        ),
    }
    return env.process_raw_obs(raw, flatten_dict_obs=True), diagnostics


def run(args: argparse.Namespace) -> Path:
    proposal_path = Path(args.proposal).resolve()
    proposal = json.loads(proposal_path.read_text())
    query, candidate = select_proposal_candidate(
        proposal, query_index=args.query_index, candidate_index=args.candidate_index
    )
    reference = Path(str(candidate["clip_path"])).resolve()
    if _sha256(reference) != candidate["clip_sha256"]:
        raise TrackerRecoveryError("reference clip changed after proposal generation")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tracker_repo = Path(args.tracker_repo).resolve()
    if not (tracker_repo / "sonic_port").is_dir():
        raise TrackerRecoveryError(f"not a tracker repository: {tracker_repo}")
    sys.path.insert(0, str(tracker_repo))
    bundle, motion_file, clip = _build_single_clip_bundle(
        tracker_repo=tracker_repo, reference_npz=reference, output=output
    )

    import torch

    from sonic_port.oracle.dump_obs import build_env
    from sonic_port.collect.policy import DeterministicPolicyRollout, load_actor

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    config_path = checkpoint_dir / "config.yaml"
    with _isolated_kit_cache(Path(args.kit_cache)):
        env, _sim, motion_lib, robot, _ = build_env(
            str(config_path),
            str(bundle),
            str(motion_file),
            seed=int(args.seed),
            device=args.device,
            filter_clip=clip,
            drop_future_delta=False,
            pre_env_hook=_patch_isaaclab_urdf_pin,
            config_overrides={"episode_length_s": float(args.episode_length_s)},
        )
    model = load_actor(env, str(checkpoint_dir), args.device)
    policy = DeterministicPolicyRollout(model.policy)
    env.set_is_evaluating(True)
    env.reset_all()
    disabled_terminations = _disable_builtin_failure_terminations(env)
    motion_cmd = env.env.command_manager.get_term("motion")
    _place_identity_object(env, motion_lib, args.device)
    if args.exact_learner_handoff:
        if proposal.get("schema") != "justin-s13-tracker-recovery-proposal/v3":
            raise TrackerRecoveryError(
                "exact learner handoff requires a v3 proposal with learner history"
            )
        if args.reference_blend != 0.0:
            raise TrackerRecoveryError(
                "exact learner handoff prohibits reference-state blending"
            )
        obs, injection, tracker_history = _inject_exact_learner_handoff(
            env, robot, motion_cmd, query, candidate, args.device
        )
        handoff_mode = "exact_learner_state_and_history"
    else:
        obs, injection = _inject_recovery_state(
            env,
            robot,
            motion_cmd,
            query,
            candidate,
            args.device,
            reference_blend=args.reference_blend,
        )
        tracker_history = None
        handoff_mode = "legacy_instantaneous_state"
    obs = {name: value.to(args.device) for name, value in obs.items()}

    frame = int(candidate["frame"])
    remaining = int(candidate["frames_remaining"])
    terminal_frame = frame + remaining - 1
    adaptive_clock = (
        AdaptiveReferenceClock(
            freeze_root_error_m=args.clock_freeze_root_error_m,
            resume_root_error_m=args.clock_resume_root_error_m,
            max_consecutive_hold_steps=args.max_consecutive_clock_hold_steps,
        )
        if args.adaptive_reference_clock
        else None
    )
    second_stage_trigger = (
        SecondStageRematchTrigger(stable_steps=args.second_stage_stable_steps)
        if args.second_stage_rematch
        else None
    )
    if (
        second_stage_trigger is not None
        and args.second_stage_reference_end_frame >= terminal_frame
    ):
        raise TrackerRecoveryError(
            "second-stage reference search must end before the terminal frame"
        )
    rollout_budget = remaining + (
        int(args.max_clock_hold_steps) if adaptive_clock is not None else 0
    ) + (
        int(args.second_stage_extra_steps) if second_stage_trigger is not None else 0
    )
    max_steps = min(
        rollout_budget,
        int(args.max_steps) if args.max_steps > 0 else rollout_budget,
    )
    previous_done = torch.zeros(1, dtype=torch.bool, device=args.device)
    root_pos: list[np.ndarray] = []
    root_quat: list[np.ndarray] = []
    joint_pos: list[np.ndarray] = []
    joint_vel: list[np.ndarray] = []
    mpjpe: list[float] = []
    reference_frames: list[int] = []
    reference_held: list[bool] = []
    reference_rematched: list[bool] = []
    root_tracking_error_m: list[float] = []
    second_stage_rematch_receipt: dict[str, object] | None = None
    second_stage_terminal_hold_steps = 0
    terminated_early = False
    time_out = False
    with torch.no_grad():
        for step in range(max_steps):
            rematched_this_step = False
            robot_root_before = motion_cmd.robot_body_pos_w[0, 0]
            if (
                second_stage_trigger is not None
                and second_stage_rematch_receipt is None
                and second_stage_trigger.observe(robot_root_before.detach().cpu().numpy())
            ):
                pre_rematch_frame = int(
                    (
                        motion_cmd.motion_start_time_steps + motion_cmd.time_steps
                    )[0].item()
                )
                candidate_frames_t = torch.arange(
                    args.second_stage_reference_start_frame,
                    args.second_stage_reference_end_frame + 1,
                    dtype=torch.long,
                    device=args.device,
                )
                candidate_motion_ids = motion_cmd.motion_ids[0].expand_as(
                    candidate_frames_t
                )
                reference_root = motion_cmd.motion_lib.get_root_pos_w(
                    candidate_motion_ids, candidate_frames_t
                ) + env.env.scene.env_origins[0]
                reference_joint = motion_cmd.motion_lib.get_dof_pos(
                    candidate_motion_ids, candidate_frames_t
                )
                match = select_second_stage_reference_frame(
                    robot_root_position_m=robot_root_before.detach().cpu().numpy(),
                    robot_joint_position_rad=robot.data.joint_pos[0]
                    .detach()
                    .cpu()
                    .numpy(),
                    candidate_frames=candidate_frames_t.detach().cpu().numpy(),
                    reference_root_position_m=reference_root.detach().cpu().numpy(),
                    reference_joint_position_rad=reference_joint.detach().cpu().numpy(),
                    joint_weight_m_per_rad=args.second_stage_joint_weight_m_per_rad,
                )
                selected_frame = int(match["frame"])
                if selected_frame >= pre_rematch_frame:
                    raise TrackerRecoveryError(
                        "second-stage rematch did not move back to a moving reference"
                    )
                motion_cmd.motion_start_time_steps.fill_(selected_frame)
                motion_cmd.time_steps.zero_()
                raw = env.env.observation_manager.compute(update_history=False)
                obs = env.process_raw_obs(raw, flatten_dict_obs=True)
                obs = {name: value.to(args.device) for name, value in obs.items()}
                second_stage_rematch_receipt = {
                    "rollout_step": step,
                    "stable_steps": second_stage_trigger.stable_steps,
                    "pre_rematch_reference_frame": pre_rematch_frame,
                    "selected_reference_frame": selected_frame,
                    "rewound_reference_frames": pre_rematch_frame - selected_frame,
                    "search_start_frame": args.second_stage_reference_start_frame,
                    "search_end_frame": args.second_stage_reference_end_frame,
                    "joint_weight_m_per_rad": args.second_stage_joint_weight_m_per_rad,
                    "robot_root_position_m": robot_root_before.detach().cpu().tolist(),
                    **match,
                }
                rematched_this_step = True
            current_frame_before = int(
                (motion_cmd.motion_start_time_steps + motion_cmd.time_steps)[0].item()
            )
            phase_root_error = float(
                (
                    motion_cmd.body_pos_w[0, 0]
                    - motion_cmd.robot_body_pos_w[0, 0]
                )
                .norm()
                .item()
            )
            if (
                second_stage_trigger is not None
                and current_frame_before >= terminal_frame
            ):
                advance_reference = False
                second_stage_terminal_hold_steps += 1
            elif adaptive_clock is not None:
                advance_reference = adaptive_clock.should_advance(
                    phase_root_error,
                    at_terminal=current_frame_before >= terminal_frame,
                )
            else:
                advance_reference = True
            action = policy.action(obs, previous_done)
            obs, _reward, dones, extras = env.step({"actions": action, "obs_dict": None})
            if not advance_reference:
                # TrackingCommand increments its cursor inside env.step().
                # Undo that one tick and rebuild only the current observation;
                # update_history=False avoids duplicating proprioceptive history.
                motion_cmd.time_steps.sub_(1)
                raw = env.env.observation_manager.compute(update_history=False)
                obs = env.process_raw_obs(raw, flatten_dict_obs=True)
            obs = {name: value.to(args.device) for name, value in obs.items()}
            previous_done = dones
            error = ((motion_cmd.body_pos_w - motion_cmd.robot_body_pos_w).norm(dim=-1) * 1000.0)
            mpjpe.append(float(error.mean(dim=-1)[0].item()))
            root_pos.append(robot.data.root_pos_w[0].detach().cpu().numpy().copy())
            root_quat.append(robot.data.root_quat_w[0].detach().cpu().numpy().copy())
            joint_pos.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
            joint_vel.append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
            current_frame = int(
                (motion_cmd.motion_start_time_steps + motion_cmd.time_steps)[0].item()
            )
            reference_frames.append(current_frame)
            reference_held.append(not advance_reference)
            reference_rematched.append(rematched_this_step)
            root_tracking_error_m.append(
                float(
                    (
                        motion_cmd.body_pos_w[0, 0]
                        - motion_cmd.robot_body_pos_w[0, 0]
                    )
                    .norm()
                    .item()
                )
            )
            if bool(dones[0].item()):
                time_out = bool(extras.get("time_outs", torch.zeros_like(dones))[0].item())
                terminated_early = not time_out and current_frame < terminal_frame
                break
            if step + 1 == TRACKER_SCREEN_STEPS and max(mpjpe) > TRACKER_MPJPE_LIMIT_MM:
                terminated_early = True
                break

    root_pos_a = np.asarray(root_pos, dtype=np.float32)
    mpjpe_a = np.asarray(mpjpe, dtype=np.float32)
    completed_reference = bool(
        len(reference_frames) > 0
        and reference_frames[-1] >= terminal_frame
    )
    evaluation = evaluate_tracker_attempt(
        mpjpe_mm=mpjpe_a,
        root_position_m=root_pos_a,
        terminated_early=terminated_early,
        completed_reference=completed_reference,
    )
    second_stage_evaluation = (
        evaluate_second_stage_recovery(
            mpjpe_mm=mpjpe_a,
            root_position_m=root_pos_a,
            rematch_step=int(second_stage_rematch_receipt["rollout_step"]),
            terminated_early=terminated_early,
            completed_reference=completed_reference,
        )
        if second_stage_rematch_receipt is not None
        else None
    )
    np.savez(
        output / "tracker_rollout.npz",
        root_pos=root_pos_a,
        root_quat_wxyz=np.asarray(root_quat, dtype=np.float32),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        mpjpe_mm=mpjpe_a,
        reference_frames=np.asarray(reference_frames, dtype=np.int64),
        reference_held=np.asarray(reference_held, dtype=np.bool_),
        reference_rematched=np.asarray(reference_rematched, dtype=np.bool_),
        root_tracking_error_m=np.asarray(root_tracking_error_m, dtype=np.float32),
        fps=np.asarray([50], dtype=np.int64),
    )
    receipt = {
        "schema": "justin-s13-tracker-recovery-attempt/v5",
        "proposal": str(proposal_path),
        "proposal_sha256": _sha256(proposal_path),
        "query_index": int(args.query_index),
        "candidate_index": int(args.candidate_index),
        "rewind_s": query["rewind_s"],
        "trace_index": query["trace_index"],
        "candidate": candidate,
        "tracker_repo": str(tracker_repo),
        "tracker_repo_commit": subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={tracker_repo}",
                "-C",
                str(tracker_repo),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": _sha256(checkpoint_dir / "last.pt"),
        "handoff_mode": handoff_mode,
        "exact_learner_state_injection": (
            args.exact_learner_handoff and args.reference_blend == 0.0
        ),
        "state_injection": injection,
        "tracker_history": tracker_history,
        "blend_steps": 0,
        "disabled_training_terminations": list(disabled_terminations),
        "urdf_importer_compatibility": {
            "isaaclab_requested": "2.4.31",
            "isaacsim_installed": "2.4.30",
            "omitted_unavailable_setter": "set_merge_fixed_ignore_inertia",
        },
        "fine_tuning_performed": False,
        "rollout_steps": len(mpjpe),
        "reference_clock": (
            adaptive_clock.receipt()
            if adaptive_clock is not None
            else {
                "mode": (
                    "second_stage_rematch_with_terminal_hold"
                    if second_stage_trigger is not None
                    else "fixed_rate"
                ),
                "hold_steps": second_stage_terminal_hold_steps,
                "terminal_hold_steps": second_stage_terminal_hold_steps,
            }
        ),
        "second_stage_rematch": second_stage_rematch_receipt,
        "second_stage_evaluation": second_stage_evaluation,
        "reference_terminal_frame": terminal_frame,
        "reference_final_frame": reference_frames[-1] if reference_frames else None,
        "evaluation": evaluation,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    printed_evaluation = (
        {"overall": evaluation, "second_stage": second_stage_evaluation}
        if second_stage_evaluation is not None
        else evaluation
    )
    print(json.dumps(printed_evaluation, indent=2, sort_keys=True), flush=True)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--query-index", type=int, required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tracker-repo", default="/move/u/justingu/Projects/grail-stairs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/move/u/justingu/Projects/grail-stairs/data/sonic_checkpoints/terrain_release",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length-s", type=float, default=12.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--reference-blend",
        type=validate_reference_blend,
        default=0.0,
        help="fraction of the matched tracker reference state injected at handoff",
    )
    parser.add_argument(
        "--exact-learner-handoff",
        action="store_true",
        help=(
            "use 100% learner state and reconstruct the production actor's "
            "ten-frame learner observation/action history"
        ),
    )
    parser.add_argument(
        "--adaptive-reference-clock",
        action="store_true",
        help=(
            "pause the matched reference when root tracking error exceeds the "
            "freeze threshold and resume after the robot catches up"
        ),
    )
    parser.add_argument("--clock-freeze-root-error-m", type=float, default=0.20)
    parser.add_argument("--clock-resume-root-error-m", type=float, default=0.12)
    parser.add_argument(
        "--max-consecutive-clock-hold-steps",
        type=int,
        default=1,
        help="force one reference advance after this many consecutive holds",
    )
    parser.add_argument("--max-clock-hold-steps", type=int, default=200)
    parser.add_argument(
        "--second-stage-rematch",
        action="store_true",
        help=(
            "after the robot settles on Justin's second step, rewind once to "
            "the closest moving reference pose before the platform transition"
        ),
    )
    parser.add_argument("--second-stage-stable-steps", type=int, default=5)
    parser.add_argument("--second-stage-reference-start-frame", type=int, default=230)
    parser.add_argument("--second-stage-reference-end-frame", type=int, default=270)
    parser.add_argument(
        "--second-stage-joint-weight-m-per-rad", type=float, default=0.10
    )
    parser.add_argument("--second-stage-extra-steps", type=int, default=150)
    parser.add_argument("--kit-cache", required=True)
    args = parser.parse_args(argv)
    if args.max_clock_hold_steps < 0:
        parser.error("--max-clock-hold-steps must be non-negative")
    if args.max_consecutive_clock_hold_steps < 1:
        parser.error("--max-consecutive-clock-hold-steps must be positive")
    if args.adaptive_reference_clock and args.second_stage_rematch:
        parser.error(
            "--adaptive-reference-clock and --second-stage-rematch are separate experiments"
        )
    if args.second_stage_stable_steps < 1:
        parser.error("--second-stage-stable-steps must be positive")
    if (
        args.second_stage_reference_start_frame < 0
        or args.second_stage_reference_end_frame
        < args.second_stage_reference_start_frame
    ):
        parser.error("second-stage reference frame range is invalid")
    if (
        not np.isfinite(args.second_stage_joint_weight_m_per_rad)
        or args.second_stage_joint_weight_m_per_rad < 0.0
    ):
        parser.error(
            "--second-stage-joint-weight-m-per-rad must be finite and non-negative"
        )
    if args.second_stage_extra_steps < 1:
        parser.error("--second-stage-extra-steps must be positive")
    if args.adaptive_reference_clock:
        try:
            AdaptiveReferenceClock(
                freeze_root_error_m=args.clock_freeze_root_error_m,
                resume_root_error_m=args.clock_resume_root_error_m,
                max_consecutive_hold_steps=args.max_consecutive_clock_hold_steps,
            )
        except TrackerRecoveryError as error:
            parser.error(str(error))
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
