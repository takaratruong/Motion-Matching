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
import torch

from .gear_action import isaaclab_to_mujoco_joint_vector
from .train_classic_g1_pfnn import load_classic_checkpoint
from .terrain_pfnn.hill_map import TerrainPFNNHillMap
from .terrain_pfnn.kinematics import TorchG1ForwardKinematics
from .terrain_pfnn.pfnn_surface import (
    PlacedPFNNSurface,
    load_placed_pfnn_surface,
)
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
    "sonic/runs/native-g1-pfnn/expanded/mixed-corpus-filtered/manifest.json"
)
DEFAULT_CHECKPOINT = Path(
    "sonic/runs/native-g1-pfnn/expanded/"
    "model-mixed-filtered-rollout16-final/best.pt"
)
DEFAULT_TERRAIN_FIT = Path(
    "sonic/runs/native-g1-pfnn/expanded/vertical-corpus/terrain/"
    "WalkingUpSteps08_000__01550_01675.npz"
)
DEFAULT_IDLE_CLIPS = Path(
    "/home/ubuntu/projects/gear-sonic-pinned-60de0df/"
    "motionbricks/out/G1-clip.ckpt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--idle-clips", type=Path, default=DEFAULT_IDLE_CLIPS)
    parser.add_argument("--terrain-fit", type=Path, default=DEFAULT_TERRAIN_FIT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--trace-every", type=int, default=30)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
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


class _PFNNTerrainCallback:
    """One exact placed PFNN height function shared by runtime and mesh."""

    def __init__(
        self,
        surface: PlacedPFNNSurface,
        *,
        x_samples: np.ndarray,
        y_samples: np.ndarray,
    ) -> None:
        if not isinstance(surface, PlacedPFNNSurface):
            raise TypeError("surface must be a PlacedPFNNSurface")
        x = np.asarray(x_samples, dtype=np.float64)
        y = np.asarray(y_samples, dtype=np.float64)
        if (
            x.ndim != 1
            or y.ndim != 1
            or min(len(x), len(y)) < 2
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0.0)
            or np.any(np.diff(y) <= 0.0)
        ):
            raise ValueError("PFNN terrain mesh axes must be finite and increasing")
        self.surface = surface
        self.x_min, self.x_max = float(x[0]), float(x[-1])
        self.y_min, self.y_max = float(y[0]), float(y[-1])
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
        xy = np.stack((grid_x, grid_y), axis=-1)
        self.vertices = np.column_stack(
            (xy.reshape(-1, 2), surface.height_at(xy).reshape(-1))
        )
        faces: list[tuple[int, int, int]] = []
        columns = len(x)
        for row in range(len(y) - 1):
            for column in range(columns - 1):
                lower_left = row * columns + column
                lower_right = lower_left + 1
                upper_left = lower_left + columns
                upper_right = upper_left + 1
                faces.append((lower_left, lower_right, upper_right))
                faces.append((lower_left, upper_right, upper_left))
        self.faces = np.asarray(faces, dtype=np.int32)

    def _supported(self, points: np.ndarray) -> bool:
        return bool(
            np.all(points[..., 0] >= self.x_min)
            and np.all(points[..., 0] <= self.x_max)
            and np.all(points[..., 1] >= self.y_min)
            and np.all(points[..., 1] <= self.y_max)
        )

    def __call__(self, xy: object) -> TerrainSample | None:
        point = np.asarray(xy, dtype=np.float64)
        if (
            point.shape != (2,)
            or not np.isfinite(point).all()
            or not self._supported(point)
        ):
            return None
        height = float(self.surface.height_at(point[None, :])[0])
        gradient = self.surface.gradient_at(point[None, :])[0]
        return TerrainSample(height, gradient)

    def collision_heights_at(self, xy: object) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or not self._supported(points)
        ):
            raise ValueError("collision query left the rendered PFNN surface")
        return self.surface.height_at(points)


class _PFNNCourseCallback:
    """Flat run-up followed by the unchanged scaled released-PFNN surface."""

    _BLEND_START_X_M = 0.75
    _BLEND_STOP_X_M = 1.25

    def __init__(
        self,
        surface: PlacedPFNNSurface,
        *,
        x_samples: np.ndarray,
        y_samples: np.ndarray,
    ) -> None:
        if not isinstance(surface, PlacedPFNNSurface):
            raise TypeError("surface must be a PlacedPFNNSurface")
        x = np.asarray(x_samples, dtype=np.float64)
        y = np.asarray(y_samples, dtype=np.float64)
        if (
            x.ndim != 1
            or y.ndim != 1
            or min(len(x), len(y)) < 2
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0.0)
            or np.any(np.diff(y) <= 0.0)
        ):
            raise ValueError("PFNN terrain mesh axes must be finite and increasing")
        self.surface = surface
        # PFNN stores horizontal X/Z in centimetres.  Start half a metre before
        # the fitted contact centre so the original surface unfolds in +world X.
        self.source_anchor_xy = np.asarray(
            (
                surface.fit.contact_center_xz[0] * surface.scale / 100.0 - 0.5,
                -surface.fit.contact_center_xz[1] * surface.scale / 100.0,
            ),
            dtype=np.float64,
        )
        self.source_height_m = float(
            surface.height_at(self.source_anchor_xy[None, :])[0]
        )
        self.x_min, self.x_max = float(x[0]), float(x[-1])
        self.y_min, self.y_max = float(y[0]), float(y[-1])
        grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
        xy = np.stack((grid_x, grid_y), axis=-1)
        self.vertices = np.column_stack(
            (xy.reshape(-1, 2), self._heights_and_gradients(xy.reshape(-1, 2))[0])
        )
        columns = len(x)
        self.faces = np.asarray(
            [
                triangle
                for row in range(len(y) - 1)
                for column in range(columns - 1)
                for triangle in (
                    (
                        row * columns + column,
                        row * columns + column + 1,
                        (row + 1) * columns + column + 1,
                    ),
                    (
                        row * columns + column,
                        (row + 1) * columns + column + 1,
                        (row + 1) * columns + column,
                    ),
                )
            ],
            dtype=np.int32,
        )

    def _supported(self, points: np.ndarray) -> bool:
        return bool(
            np.all(points[..., 0] >= self.x_min)
            and np.all(points[..., 0] <= self.x_max)
            and np.all(points[..., 1] >= self.y_min)
            and np.all(points[..., 1] <= self.y_max)
        )

    def _heights_and_gradients(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(points, dtype=np.float64)
        source = np.empty_like(query)
        source[:, 0] = (
            self.source_anchor_xy[0]
            + query[:, 0]
            - self._BLEND_START_X_M
        )
        source[:, 1] = self.source_anchor_xy[1] + query[:, 1]
        raw_height = self.surface.height_at(source) - self.source_height_m
        raw_gradient = self.surface.gradient_at(source)
        u = np.clip(
            (query[:, 0] - self._BLEND_START_X_M)
            / (self._BLEND_STOP_X_M - self._BLEND_START_X_M),
            0.0,
            1.0,
        )
        weight = u * u * (3.0 - 2.0 * u)
        weight_derivative = (
            6.0 * u * (1.0 - u)
            / (self._BLEND_STOP_X_M - self._BLEND_START_X_M)
        )
        height = weight * raw_height
        gradient = weight[:, None] * raw_gradient
        gradient[:, 0] += weight_derivative * raw_height
        return height, gradient

    def __call__(self, xy: object) -> TerrainSample | None:
        point = np.asarray(xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all() or not self._supported(point):
            return None
        height, gradient = self._heights_and_gradients(point[None, :])
        return TerrainSample(float(height[0]), gradient[0])

    def collision_heights_at(self, xy: object) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.isfinite(points).all()
            or not self._supported(points)
        ):
            raise ValueError("collision query left the rendered PFNN course")
        return self._heights_and_gradients(points)[0]


def _viewer_terrain_map() -> TerrainPFNNHillMap:
    """Build a wide extrusion so steering mistakes remain on queried terrain."""

    return TerrainPFNNHillMap(grid_spacing_m=0.075, half_width_m=12.0)


def _configure_camera(viewer: object) -> None:
    """Use a side view so longitudinal hill grades are visibly legible."""

    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -18.0
    viewer.cam.distance = 4.0


def _upright_yaw_quaternion(quaternion_wxyz: object) -> np.ndarray:
    """Remove roll/pitch while preserving the predicted world yaw."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("root quaternion must be finite wxyz")
    w, x, y, z = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray(
        (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
        dtype=np.float64,
    )


def _validate(arguments: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    paths = tuple(
        Path(value).expanduser().resolve()
        for value in (
            arguments.checkpoint,
            arguments.dataset,
            arguments.model_path,
            arguments.scene_xml,
            arguments.idle_clips,
        )
    )
    missing = tuple(path for path in paths if not path.is_file())
    if arguments.terrain_fit is not None:
        terrain_fit = Path(arguments.terrain_fit).expanduser().resolve()
        if not terrain_fit.is_file():
            missing = (*missing, terrain_fit)
    if missing:
        raise FileNotFoundError("missing PFNN viewer input: " + ", ".join(map(str, missing)))
    if not math.isfinite(arguments.speed) or arguments.speed <= 0.0:
        raise ValueError("--speed must be finite and positive")
    if arguments.max_steps < 1 or arguments.trace_every < 1:
        raise ValueError("--max-steps and --trace-every must be positive")
    if arguments.smoke_steps < 0 or (arguments.no_viewer and arguments.smoke_steps < 1):
        raise ValueError("--no-viewer requires positive --smoke-steps")
    return paths  # type: ignore[return-value]


def _motionbricks_idle_mujoco_qpos(path: Path) -> np.ndarray:
    """Load MotionBricks' native G1 idle keyframe without executable pickle."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("MotionBricks G1 idle clips cannot be loaded safely") from error
    if type(payload) is not dict:
        raise ValueError("MotionBricks G1 idle clips are invalid")
    qpos = payload.get("mujoco_qpos")
    counts = payload.get("num_frames_per_clip")
    if (
        not isinstance(qpos, torch.Tensor)
        or qpos.ndim != 3
        or qpos.shape[0] < 1
        or qpos.shape[1] < 1
        or qpos.shape[2] != 36
        or not isinstance(counts, torch.Tensor)
        or counts.shape != (qpos.shape[0],)
        or int(counts[0]) < 1
        or not bool(torch.isfinite(qpos[0, 0]).all())
    ):
        raise ValueError("MotionBricks G1 idle clips are invalid")
    result = np.asarray(qpos[0, 0].to(dtype=torch.float64), dtype=np.float64).copy()
    if result[2] <= 0.0 or not np.isclose(np.linalg.norm(result[3:7]), 1.0, atol=1.0e-4):
        raise ValueError("MotionBricks G1 idle keyframe is invalid")
    return result


def _build_scene(
    scene_xml: Path, terrain: object, idle_clips: Path
) -> tuple[object, object]:
    import mujoco

    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.add_mesh(
        name="terrain_pfnn_three_hills",
        uservert=terrain.vertices.ravel(),
        userface=terrain.faces.ravel(),
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
        smoothnormal=1,
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
    model.qpos0[7:36] = _motionbricks_idle_mujoco_qpos(idle_clips)[7:36]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor >= 0:
        model.geom_rgba[floor, 3] = 0.0
    return model, mujoco.MjData(model)


def _apply_frame(model: object, data: object, frame: object) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = frame.root_position_world
    data.qpos[3:7] = _upright_yaw_quaternion(
        frame.root_quaternion_world_wxyz
    )
    if not bool(frame.diagnostics.get("initial_idle_pose_held", False)):
        data.qpos[7:36] = isaaclab_to_mujoco_joint_vector(
            frame.joint_position_isaaclab
        )


def _trace(step: int, frame: object, terrain: object) -> str:
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
        f"requested={float(diagnostics.get('desired_speed_m_s', 0.0)):.3f} "
        f"realized={float(diagnostics.get('realized_speed_m_s', 0.0)):.3f} "
        f"contacts={contacts} hold={diagnostics.get('hold_reason', '-')}"
    )


def _load_runtime(
    arguments: argparse.Namespace,
    checkpoint: Path,
    dataset: Path,
    model_path: Path,
    terrain: object,
    *,
    strict: bool = False,
) -> TerrainPFNNRuntime:
    manifest = json.loads(dataset.read_text(encoding="utf-8"))
    digest = manifest.get("dataset_sha256", manifest.get("dataset_digest_sha256"))
    if type(digest) is not str or len(digest) != 64:
        raise ValueError("dataset manifest digest is invalid")
    loaded = load_classic_checkpoint(checkpoint)
    if loaded.dataset_digest != digest:
        raise ValueError("classic PFNN checkpoint dataset digest mismatch")
    released = getattr(loaded, "source_kind", None) == "released_pfnn"
    if released and (
        manifest.get("selection_sha256")
        != loaded.vertical_slice_receipt_sha256
        or manifest.get("terrain_receipt_set_sha256")
        != loaded.terrain_receipt_set_sha256
    ):
        raise ValueError("classic PFNN source or terrain receipt mismatch")
    kinematics = TorchG1ForwardKinematics.from_mjcf(model_path)
    return TerrainPFNNRuntime(
        checkpoint=loaded,
        kinematics=kinematics,
        height_and_grade_at=terrain,
        device=arguments.device,
        enforce_motion_envelope=strict,
        command_driven_root=False,
        hold_idle_pose=True,
        maximum_grade_degrees=89.0 if released else 20.0,
    )


def _run(arguments: argparse.Namespace) -> int:
    checkpoint, dataset, model_path, scene_xml, idle_clips = _validate(arguments)
    if arguments.terrain_fit is None:
        terrain_map = _viewer_terrain_map()
        terrain = _HillTerrainCallback(terrain_map)
        rendered_terrain = terrain_map
    else:
        surface = load_placed_pfnn_surface(arguments.terrain_fit)
        terrain = _PFNNCourseCallback(
            surface,
            x_samples=np.linspace(-2.0, 6.0, 321),
            y_samples=np.linspace(-4.0, 4.0, 161),
        )
        rendered_terrain = terrain
    runtime = _load_runtime(
        arguments, checkpoint, dataset, model_path, terrain
    )
    model, data = _build_scene(scene_xml, rendered_terrain, idle_clips)
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
            _configure_camera(viewer)
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
