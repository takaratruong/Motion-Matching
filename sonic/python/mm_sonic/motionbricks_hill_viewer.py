"""Controllable kinematic MotionBricks probe on one smooth hill."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Iterator

import numpy as np

from .motionbricks_hill import GentleHillProfile
from .motionbricks_hill_conditioning import MotionBricksHillConditioner
from .motionbricks_hill_ik import (
    HillFootIKDiagnostics,
    MotionBricksHillFootIK,
)


DEFAULT_MOTIONBRICKS_ROOT = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motionbricks-root",
        type=Path,
        default=DEFAULT_MOTIONBRICKS_ROOT,
        help="official pinned MotionBricks checkout containing out/ checkpoints",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="maximum interactive playback frames",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="bounded frame count for an automatic forward smoke run",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="run the bounded automatic smoke command without a window",
    )
    parser.add_argument(
        "--no-ik",
        action="store_true",
        help="render raw MotionBricks qpos without stance-aware foot IK",
    )
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--mesh-samples", type=int, default=151)
    parser.add_argument(
        "--trace-every",
        type=int,
        default=15,
        help="print root/terrain/conditioned target heights every N frames",
    )
    return parser


def _checkpoint_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "out/G1-clip.ckpt",
        root
        / "out/motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt",
        root
        / "out/motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt",
        root
        / "out/motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt",
    )


def _validate_runtime(arguments: argparse.Namespace) -> Path:
    root = arguments.motionbricks_root.expanduser().resolve()
    required = (
        root / "motionbricks/motion_backbone/demo/utils.py",
        root / "assets/skeletons/g1/g1.xml",
        root / "assets/skeletons/g1/scene_29dof.xml",
        *_checkpoint_paths(root),
    )
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "MotionBricks runtime is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    pointer_files = tuple(
        path for path in _checkpoint_paths(root) if path.stat().st_size < 1024
    )
    if pointer_files:
        raise RuntimeError(
            "MotionBricks checkpoints are still Git-LFS pointers. Run "
            "`git -c lfs.fetchexclude= lfs pull "
            "--include='motionbricks/out/**' --exclude=''` from the parent "
            "GR00T checkout. Pointers: "
            + ", ".join(str(path) for path in pointer_files)
        )
    if arguments.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if arguments.smoke_steps < 0:
        raise ValueError("--smoke-steps cannot be negative")
    if arguments.no_viewer and arguments.smoke_steps < 1:
        raise ValueError("--no-viewer requires --smoke-steps")
    if arguments.mesh_samples < 2:
        raise ValueError("--mesh-samples must be at least two")
    if arguments.trace_every < 1:
        raise ValueError("--trace-every must be positive")
    return root


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _motionbricks_arguments(root: Path, arguments: argparse.Namespace) -> object:
    return SimpleNamespace(
        humanoid_xml=str(root / "assets/skeletons/g1/scene_29dof.xml"),
        humanoid_scene_xml=str(root / "assets/skeletons/g1/scene_29dof.xml"),
        skeleton_xml=str(root / "assets/skeletons/g1/g1.xml"),
        result_dir=str(root / "out"),
        data_root=str(root / "datasets"),
        explicit_dataset_folder=str(root / "datasets/motionbricks-G1"),
        clips_ckpt=str(root / "out/G1-clip.ckpt"),
        reprocess_clips=0,
        controller="wasd",
        lookat_movement_direction=1,
        has_viewer=not arguments.no_viewer,
        pre_filter_qpos=1,
        source_root_realignment=1,
        target_root_realignment=1,
        force_canonicalization=1,
        skip_ending_target_cond=0,
        random_speed_scale=0,
        speed_scale=[1.0, 1.0],
        generate_dt=2.0,
        max_steps=arguments.max_steps,
        random_seed=arguments.random_seed,
        num_runs=1,
        use_qpos=1,
        planner="default",
        allowed_mode=None,
        clips="G1",
        return_model_configs=True,
        return_dataloader=True,
        recording_dir=None,
        EXP="default",
    )


def _build_hill_model(
    humanoid_scene_xml: Path,
    profile: GentleHillProfile,
    *,
    sample_count: int,
    timestep: float,
) -> tuple[object, object]:
    import mujoco

    vertices, faces = profile.mesh(sample_count)
    spec = mujoco.MjSpec.from_file(str(humanoid_scene_xml))
    spec.add_mesh(
        name="motionbricks_gentle_hill",
        uservert=vertices.ravel(),
        userface=faces.ravel(),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="motionbricks_gentle_hill",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="motionbricks_gentle_hill",
        contype=0,
        conaffinity=0,
        rgba=(0.18, 0.42, 0.22, 1.0),
    )
    model = spec.compile()
    model.opt.timestep = float(timestep)
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_id >= 0:
        model.geom_rgba[floor_id, 3] = 0.0
    return model, mujoco.MjData(model)


def _automatic_keys() -> dict[str, bool]:
    keys = {
        name: False
        for name in (
            "w",
            "a",
            "s",
            "d",
            "left",
            "right",
            "up",
            "down",
            "shift",
            "ctrl",
            "enter",
            "x",
            "z",
            "c",
            "v",
            "b",
            "r",
            "t",
            "f",
            "g",
            "q",
            "e",
        )
    }
    keys["w"] = True
    return keys


def _dummy_viewer() -> object:
    return SimpleNamespace(
        cam=SimpleNamespace(
            lookat=np.zeros(3, dtype=np.float64),
            distance=4.0,
            azimuth=0.0,
            elevation=-20.0,
        )
    )


def _trace_line(
    step: int,
    qpos: np.ndarray,
    profile: GentleHillProfile,
    conditioner: MotionBricksHillConditioner,
    ik_diagnostics: HillFootIKDiagnostics | None,
) -> str:
    terrain_height = profile.height(qpos[:2])
    trace = conditioner.latest_trace
    target_text = "waiting"
    if trace is not None:
        target_text = ",".join(
            f"{value:.3f}" for value in trace.target_heights_world
        )
    ik_text = "ik=off"
    if ik_diagnostics is not None:
        phases = "/".join(
            phase.value for phase in ik_diagnostics.phases
        )
        ik_text = (
            f"ik={phases} "
            f"penetration={ik_diagnostics.raw_penetration_m:.3f}"
            f"->{ik_diagnostics.corrected_penetration_m:.3f} "
            f"residual={ik_diagnostics.maximum_target_residual_m:.3f} "
            f"correction={ik_diagnostics.maximum_joint_correction_rad:.3f} "
            f"accepted={int(ik_diagnostics.accepted)} "
            f"reason={ik_diagnostics.reason.replace(' ', '_')}"
        )
    return (
        f"step={step:05d} root_z={float(qpos[2]):.3f} "
        f"terrain_z={terrain_height:.3f} "
        f"target_terrain_z=[{target_text}] {ik_text}"
    )


def _step_agent(
    demo: object,
    profile: GentleHillProfile,
    *,
    viewer: object,
    random_seed: int,
    automatic: bool,
) -> np.ndarray:
    import torch

    qpos = np.asarray(demo.full_agent.get_next_frame(), dtype=np.float64)
    context = demo.full_agent.get_context_mujoco_qpos()
    demo.mj_data.qpos[:] = qpos
    control_info = (
        {"force_idle": False, "allowed_mode": None, "key_pressed": _automatic_keys()}
        if automatic
        else {"force_idle": False, "allowed_mode": None}
    )
    signals = demo.controller.generate_control_signals(
        viewer,
        demo.mj_model,
        demo.mj_data,
        visualize=not automatic,
        control_info=control_info,
    )
    support_height = profile.height(
        context[0, 0, :2].detach().cpu().numpy()
    )
    normalized_context = context.detach().clone()
    normalized_context[..., 2] -= support_height
    signals["context_mujoco_qpos"] = normalized_context
    signals["random_seed"] = torch.tensor(
        (int(random_seed),), dtype=torch.long
    )

    previous_frames = demo.full_agent.frames.get("mujoco_qpos")
    with torch.no_grad():
        demo.full_agent.generate_new_frames(
            signals,
            demo.controller.get_controller_dt() * 2.0,
        )
        if demo.full_agent.frames.get("mujoco_qpos") is not previous_frames:
            demo.full_agent.frames["mujoco_qpos"][..., 2] += support_height
    return qpos


def _run(arguments: argparse.Namespace) -> int:
    root = _validate_runtime(arguments)
    if arguments.no_viewer and not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    sys.path.insert(0, str(root))
    try:
        import mujoco
        import mujoco.viewer
        import torch
        from motionbricks.motion_backbone.demo.utils import navigation_demo

        np.random.seed(arguments.random_seed)
        torch.manual_seed(arguments.random_seed)
        torch.cuda.manual_seed_all(arguments.random_seed)
        with _working_directory(root):
            demo = navigation_demo(_motionbricks_arguments(root, arguments))
        demo.full_agent.reset()

        profile = GentleHillProfile()
        model, data = _build_hill_model(
            root / "assets/skeletons/g1/scene_29dof.xml",
            profile,
            sample_count=arguments.mesh_samples,
            timestep=float(demo.mj_model.opt.timestep),
        )
        demo.mj_model = model
        demo.mj_data = data
        conditioner = MotionBricksHillConditioner(profile.height)
        conditioner.install(demo.full_agent)
        foot_ik = (
            None
            if arguments.no_ik
            else MotionBricksHillFootIK(model, profile.height)
        )

        print(
            "MotionBricks gentle hill: "
            f"max grade {profile.max_slope_degrees:.2f} degrees; "
            + (
                "stance-aware display IK"
                if foot_ik is not None
                else "raw display, IK disabled"
            )
            + "; no terrain-normal root rotation."
        )
        print("Controls: W/A/S/D move relative to camera; rotate camera to steer; Esc closes.")

        if arguments.no_viewer:
            viewer = _dummy_viewer()
            qpos = np.asarray(demo.full_agent.get_next_frame())
            ik_diagnostics = None
            display_qpos = qpos
            for step in range(arguments.smoke_steps):
                qpos = _step_agent(
                    demo,
                    profile,
                    viewer=viewer,
                    random_seed=arguments.random_seed,
                    automatic=True,
                )
                ik_result = (
                    None
                    if foot_ik is None
                    else foot_ik.apply(
                        qpos, float(model.opt.timestep)
                    )
                )
                display_qpos = (
                    qpos if ik_result is None else ik_result.qpos
                )
                ik_diagnostics = (
                    None if ik_result is None else ik_result.diagnostics
                )
                data.qpos[:] = display_qpos
                mujoco.mj_forward(model, data)
                if step % arguments.trace_every == 0:
                    print(
                        _trace_line(
                            step,
                            qpos,
                            profile,
                            conditioner,
                            ik_diagnostics,
                        )
                    )
            print(
                _trace_line(
                    arguments.smoke_steps,
                    qpos,
                    profile,
                    conditioner,
                    ik_diagnostics,
                )
            )
            if not np.isfinite(demo.full_agent.frames["mujoco_qpos"].detach().cpu().numpy()).all():
                raise RuntimeError("MotionBricks generated non-finite qpos")
            if not np.isfinite(display_qpos).all():
                raise RuntimeError("foot IK produced non-finite qpos")
            return 0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 0.0
            viewer.cam.elevation = -20.0
            viewer.cam.distance = 4.0
            step = 0
            while viewer.is_running() and step < arguments.max_steps:
                started = time.monotonic()
                qpos = _step_agent(
                    demo,
                    profile,
                    viewer=viewer,
                    random_seed=arguments.random_seed,
                    automatic=False,
                )
                ik_result = (
                    None
                    if foot_ik is None
                    else foot_ik.apply(
                        qpos, float(model.opt.timestep)
                    )
                )
                display_qpos = (
                    qpos if ik_result is None else ik_result.qpos
                )
                ik_diagnostics = (
                    None if ik_result is None else ik_result.diagnostics
                )
                data.qpos[:] = display_qpos
                step += 1
                mujoco.mj_forward(model, data)
                viewer.cam.lookat[:] = qpos[:3]
                viewer.sync()
                if step % arguments.trace_every == 0:
                    print(
                        _trace_line(
                            step,
                            qpos,
                            profile,
                            conditioner,
                            ik_diagnostics,
                        )
                    )
                remaining = float(model.opt.timestep) - (
                    time.monotonic() - started
                )
                if remaining > 0.0:
                    time.sleep(remaining)
        return 0
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return _run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
