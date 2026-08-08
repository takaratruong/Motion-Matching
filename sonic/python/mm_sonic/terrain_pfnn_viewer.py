"""Controllable native-G1 viewer for the terrain-conditioned PFNN."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .gear_action import isaaclab_to_mujoco_joint_vector
from .terrain_pfnn.hill_map import TerrainPFNNHillMap
from .terrain_pfnn.runtime import FPS, TerrainPFNNRuntime, TerrainSample


DEFAULT_MODEL = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/"
    "assets/skeletons/g1/g1_29dof.xml"
)
DEFAULT_SCENE = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/motionbricks/"
    "assets/skeletons/g1/scene_29dof.xml"
)
DEFAULT_DATASET = Path(
    "sonic/runs/terrain-pfnn-v1/canary-dataset-recurrence-v2/manifest.json"
)
DEFAULT_CHECKPOINT = Path(
    "sonic/runs/terrain-pfnn-v1/pipeline-overfit-physical-envelope-v3/best.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--trace-every", type=int, default=30)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument(
        "--strict-envelope",
        action="store_true",
        help="freeze on envelope violations instead of previewing raw PFNN output",
    )
    return parser


def _command_from_pressed(pressed: Iterable[str], speed: float) -> np.ndarray:
    keys = set(pressed)
    command = np.asarray(
        (
            float("w" in keys) - float("s" in keys),
            float("a" in keys) - float("d" in keys),
        ),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(command))
    if norm > 0.0:
        command *= float(speed) / norm
    return command


class _HillTerrainCallback:
    def __init__(self, terrain: TerrainPFNNHillMap) -> None:
        self.terrain = terrain

    def __call__(self, xy: object) -> TerrainSample | None:
        height = self.terrain.height_at(xy)
        if height is None:
            return None
        grade = self.terrain.grade_degrees_at(xy)
        if grade is None:
            return None
        return TerrainSample(
            float(height),
            np.asarray((math.tan(math.radians(grade)), 0.0), dtype=np.float64),
        )

    def collision_heights_at(self, xy: object) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("collision XY must have finite shape [N,2]")
        heights = [self.terrain.height_at(point) for point in points]
        if any(value is None for value in heights):
            raise ValueError("collision query left the rendered hill map")
        return np.asarray(heights, dtype=np.float64)


def _validate(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        Path(value).expanduser().resolve()
        for value in (
            arguments.checkpoint,
            arguments.dataset,
            arguments.model_path,
            arguments.scene_xml,
        )
    )
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("missing PFNN viewer input: " + ", ".join(map(str, missing)))
    if not math.isfinite(arguments.speed) or arguments.speed <= 0.0:
        raise ValueError("--speed must be finite and positive")
    if arguments.max_steps < 1 or arguments.trace_every < 1:
        raise ValueError("--max-steps and --trace-every must be positive")
    if arguments.smoke_steps < 0 or (arguments.no_viewer and arguments.smoke_steps < 1):
        raise ValueError("--no-viewer requires positive --smoke-steps")
    return paths  # type: ignore[return-value]


def _build_scene(scene_xml: Path, terrain: TerrainPFNNHillMap) -> tuple[object, object]:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.add_mesh(
        name="terrain_pfnn_three_hills",
        uservert=terrain.vertices.ravel(),
        userface=terrain.faces.ravel(),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="terrain_pfnn_three_hills",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="terrain_pfnn_three_hills",
        contype=0,
        conaffinity=0,
        rgba=(0.16, 0.42, 0.20, 1.0),
    )
    model = spec.compile()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_rgba[floor, 3] = 0.0
    return model, mujoco.MjData(model)


def _apply_frame(model: object, data: object, frame: object) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = frame.root_position_world
    data.qpos[3:7] = frame.root_quaternion_world_wxyz
    data.qpos[7:36] = isaaclab_to_mujoco_joint_vector(
        frame.joint_position_isaaclab
    )


def _trace(step: int, frame: object, terrain: _HillTerrainCallback) -> str:
    sample = terrain(frame.root_position_world[:2])
    grade = float("nan") if sample is None else sample.absolute_grade_degrees
    diagnostics = frame.diagnostics
    contacts = "".join(
        "1" if value >= 0.5 else "0" for value in frame.contact_probability
    )
    return (
        f"step={step:05d} xy=({frame.root_position_world[0]:.3f},"
        f"{frame.root_position_world[1]:.3f}) z={frame.root_position_world[2]:.3f} "
        f"grade={grade:.2f} phase={frame.phase:.3f} "
        f"advance={float(diagnostics.get('phase_advance', 0.0)):.3f} "
        f"speed={float(diagnostics.get('desired_speed_m_s', 0.0)):.3f} "
        f"contacts={contacts} hold={diagnostics.get('hold_reason', '-')} "
        f"preview={','.join(diagnostics.get('preview_envelope_violations', ())) or '-'}"
    )


def _load_runtime(
    arguments: argparse.Namespace,
    checkpoint: Path,
    dataset: Path,
    model_path: Path,
    terrain: _HillTerrainCallback,
) -> TerrainPFNNRuntime:
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    digest = manifest.get("dataset_digest_sha256")
    if type(digest) is not str or len(digest) != 64:
        raise ValueError("dataset manifest digest is invalid")
    return TerrainPFNNRuntime.from_checkpoint(
        checkpoint,
        dataset_digest=digest,
        model_path=model_path,
        height_and_grade_at=terrain,
        device=arguments.device,
        enforce_motion_envelope=arguments.strict_envelope,
    )


def _run(arguments: argparse.Namespace) -> int:
    checkpoint, dataset, model_path, scene_xml = _validate(arguments)
    terrain_map = TerrainPFNNHillMap(half_width_m=4.0)
    terrain = _HillTerrainCallback(terrain_map)
    runtime = _load_runtime(
        arguments, checkpoint, dataset, model_path, terrain
    )
    model, data = _build_scene(scene_xml, terrain_map)
    import mujoco

    def advance(step: int, command: np.ndarray) -> object:
        frame = runtime.step(command, camera_yaw=0.0)
        _apply_frame(model, data, frame)
        mujoco.mj_forward(model, data)
        if step % arguments.trace_every == 0 or "hold_reason" in frame.diagnostics:
            print(_trace(step, frame, terrain), flush=True)
        return frame

    if arguments.no_viewer:
        first_hold: dict[str, object] | None = None
        for step in range(arguments.smoke_steps):
            frame = advance(step, np.asarray((arguments.speed, 0.0)))
            if first_hold is None and "hold_reason" in frame.diagnostics:
                first_hold = dict(frame.diagnostics)
                break
        if first_hold is not None:
            print(json.dumps({"accepted": False, "first_hold": first_hold}, sort_keys=True))
            return 2
        print(json.dumps({"accepted": True, "steps": arguments.smoke_steps}, sort_keys=True))
        return 0

    import mujoco.viewer
    from pynput import keyboard

    pressed: set[str] = set()
    stopped = [False]

    def on_press(key: object) -> bool | None:
        if key == keyboard.Key.esc:
            stopped[0] = True
            return False
        character = getattr(key, "char", None)
        if isinstance(character, str) and character.lower() in "wasd":
            pressed.add(character.lower())
        return None

    def on_release(key: object) -> None:
        character = getattr(key, "char", None)
        if isinstance(character, str):
            pressed.discard(character.lower())

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.azimuth = 180.0
            viewer.cam.elevation = -18.0
            viewer.cam.distance = 4.0
            step = 0
            while viewer.is_running() and not stopped[0] and step < arguments.max_steps:
                started = time.monotonic()
                frame = advance(
                    step, _command_from_pressed(pressed, arguments.speed)
                )
                viewer.cam.lookat[:] = frame.root_position_world
                viewer.sync()
                step += 1
                remaining = 1.0 / FPS - (time.monotonic() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        listener.stop()
        listener.join(timeout=1.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
