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
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

import numpy as np


TRACKER_SCREEN_STEPS = 25
TRACKER_MPJPE_LIMIT_MM = 300.0
STABLE_TOP_ROOT_Z_M = 1.10
STABLE_TOP_TAIL_STEPS = 20


class TrackerRecoveryError(RuntimeError):
    pass


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
    if proposal.get("schema") != "justin-s13-tracker-recovery-proposal/v1":
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
    root_z_m: np.ndarray,
    terminated_early: bool,
    screen_steps: int = TRACKER_SCREEN_STEPS,
    completed_reference: bool,
) -> dict[str, object]:
    mpjpe = np.asarray(mpjpe_mm, dtype=np.float64)
    root_z = np.asarray(root_z_m, dtype=np.float64)
    if mpjpe.ndim != 1 or root_z.shape != mpjpe.shape or len(mpjpe) == 0:
        raise TrackerRecoveryError("tracker metrics must be equal non-empty vectors")
    if not np.isfinite(mpjpe).all() or not np.isfinite(root_z).all():
        raise TrackerRecoveryError("tracker metrics contain non-finite values")
    screened = len(mpjpe) >= int(screen_steps)
    screen_peak = float(mpjpe[: min(len(mpjpe), int(screen_steps))].max())
    screen_pass = bool(
        screened and not terminated_early and screen_peak <= TRACKER_MPJPE_LIMIT_MM
    )
    tail = root_z[-min(len(root_z), STABLE_TOP_TAIL_STEPS) :]
    stable_top = bool(
        completed_reference
        and len(tail) == STABLE_TOP_TAIL_STEPS
        and float(tail.min()) >= STABLE_TOP_ROOT_Z_M
        and float(tail.std()) <= 0.08
    )
    full_pass = bool(
        screen_pass
        and completed_reference
        and not terminated_early
        and float(mpjpe.max()) <= TRACKER_MPJPE_LIMIT_MM
        and stable_top
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
        "stable_top": stable_top,
        "stable_top_tail_min_root_z_m": float(tail.min()),
        "stable_top_tail_std_root_z_m": float(tail.std()),
        "accepted_recovery": full_pass,
    }


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


def _inject_learner_state(env, robot, motion_cmd, query, candidate, device: str) -> None:
    import torch

    state = query["learner_state"]
    root_pose = torch.as_tensor(state["root_pose_wxyz"], device=device, dtype=torch.float32)
    root_velocity = torch.as_tensor(
        state["root_velocity_world"], device=device, dtype=torch.float32
    )
    joint_pos = torch.as_tensor(state["joint_pos_isaac"], device=device, dtype=torch.float32)
    joint_vel = torch.as_tensor(state["joint_vel_isaac"], device=device, dtype=torch.float32)
    if tuple(root_pose.shape) != (7,) or tuple(root_velocity.shape) != (6,):
        raise TrackerRecoveryError("proposal learner root state has drifted")
    if tuple(joint_pos.shape) != (29,) or tuple(joint_vel.shape) != (29,):
        raise TrackerRecoveryError("proposal learner joint state has drifted")
    frame = int(candidate["frame"])
    motion_cmd.motion_ids.fill_(0)
    motion_cmd.motion_start_time_steps.fill_(frame)
    motion_cmd.time_steps.zero_()
    robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
    robot.write_root_velocity_to_sim(root_velocity.unsqueeze(0))
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))
    env.env.scene.write_data_to_sim()
    env.env.sim.forward()
    env.env.scene.update(dt=env.env.step_dt)
    # A fresh observation is required after the teleport.  Reusing reset()
    # observations would silently test a clean reference state instead.
    raw = env.env.observation_manager.compute()
    return env.process_raw_obs(raw, flatten_dict_obs=True)


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
    obs = _inject_learner_state(
        env, robot, motion_cmd, query, candidate, args.device
    )
    obs = {name: value.to(args.device) for name, value in obs.items()}

    frame = int(candidate["frame"])
    remaining = int(candidate["frames_remaining"])
    max_steps = min(remaining, int(args.max_steps) if args.max_steps > 0 else remaining)
    previous_done = torch.zeros(1, dtype=torch.bool, device=args.device)
    root_pos: list[np.ndarray] = []
    root_quat: list[np.ndarray] = []
    joint_pos: list[np.ndarray] = []
    joint_vel: list[np.ndarray] = []
    mpjpe: list[float] = []
    reference_frames: list[int] = []
    terminated_early = False
    time_out = False
    with torch.no_grad():
        for step in range(max_steps):
            action = policy.action(obs, previous_done)
            obs, _reward, dones, extras = env.step({"actions": action, "obs_dict": None})
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
            if bool(dones[0].item()):
                time_out = bool(extras.get("time_outs", torch.zeros_like(dones))[0].item())
                terminated_early = not time_out and current_frame < frame + remaining - 1
                break
            if step + 1 == TRACKER_SCREEN_STEPS and max(mpjpe) > TRACKER_MPJPE_LIMIT_MM:
                terminated_early = True
                break

    root_pos_a = np.asarray(root_pos, dtype=np.float32)
    mpjpe_a = np.asarray(mpjpe, dtype=np.float32)
    completed_reference = bool(
        len(reference_frames) > 0
        and (reference_frames[-1] >= frame + remaining - 1 or time_out)
    )
    evaluation = evaluate_tracker_attempt(
        mpjpe_mm=mpjpe_a,
        root_z_m=root_pos_a[:, 2],
        terminated_early=terminated_early,
        completed_reference=completed_reference,
    )
    np.savez(
        output / "tracker_rollout.npz",
        root_pos=root_pos_a,
        root_quat_wxyz=np.asarray(root_quat, dtype=np.float32),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        joint_vel=np.asarray(joint_vel, dtype=np.float32),
        mpjpe_mm=mpjpe_a,
        reference_frames=np.asarray(reference_frames, dtype=np.int64),
        fps=np.asarray([50], dtype=np.int64),
    )
    receipt = {
        "schema": "justin-s13-tracker-recovery-attempt/v1",
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
        "exact_state_injection": True,
        "blend_steps": 0,
        "disabled_training_terminations": list(disabled_terminations),
        "urdf_importer_compatibility": {
            "isaaclab_requested": "2.4.31",
            "isaacsim_installed": "2.4.30",
            "omitted_unavailable_setter": "set_merge_fixed_ignore_inertia",
        },
        "fine_tuning_performed": False,
        "rollout_steps": len(mpjpe),
        "evaluation": evaluation,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True), flush=True)
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
    parser.add_argument("--kit-cache", required=True)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
