"""Authenticated read-only playback for saved Torch terrain rollouts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .joints import ContractError, PINNED_TARGET_TO_SOURCE_PERMUTATION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ContractError(f"cannot load {label}: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


@dataclass(frozen=True)
class SavedTerrainRollout:
    run_root: Path
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping
    resolved_config: Mapping
    frame_count: int
    terrain_origin_scene_xy: np.ndarray
    terrain_cell_size_m: float
    terrain_height_z: np.ndarray
    terrain_sha256: str
    translation_scene_xy: np.ndarray
    yaw_scene_from_matcher: float
    clip_paths: Sequence[str]

    def scene_to_matcher_xy(self, scene_xy: np.ndarray) -> np.ndarray:
        relative = np.asarray(scene_xy) - self.translation_scene_xy
        cosine = math.cos(self.yaw_scene_from_matcher)
        sine = math.sin(self.yaw_scene_from_matcher)
        x = relative[..., 0]
        y = relative[..., 1]
        return np.stack(
            (cosine * x + sine * y, -sine * x + cosine * y), axis=-1
        )


def _require_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
    *,
    finite: bool = True,
) -> np.ndarray:
    if name not in arrays:
        raise ContractError(f"saved rollout array is missing: {name}")
    value = arrays[name]
    if value.shape != shape:
        raise ContractError(
            f"saved rollout {name} shape {value.shape} != {shape}"
        )
    if finite and not np.isfinite(value).all():
        raise ContractError(f"saved rollout {name} contains non-finite values")
    return value


def load_saved_rollout(run_root: str | Path) -> SavedTerrainRollout:
    run_root = Path(run_root).resolve()
    if not run_root.is_dir() or run_root.is_symlink():
        raise ContractError(f"rollout run is not a real directory: {run_root}")
    metrics = _read_json(run_root / "metrics.json", "rollout metrics")
    resolved = _read_json(
        run_root / "resolved_config.json", "resolved rollout config"
    )
    archive_path = run_root / "rollout.npz"
    expected_archive_hash = metrics.get("rollout_npz_sha256")
    if (
        not isinstance(expected_archive_hash, str)
        or len(expected_archive_hash) != 64
        or not archive_path.is_file()
        or _sha256(archive_path) != expected_archive_hash
    ):
        raise ContractError("saved rollout archive hash mismatch")
    if (
        metrics.get("resolved_config_sha256")
        != _canonical_json_sha256(resolved)
    ):
        raise ContractError("saved rollout resolved-config hash mismatch")
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except Exception as error:
        raise ContractError("saved rollout archive cannot be loaded") from error
    if "time_s" not in arrays or arrays["time_s"].ndim != 1:
        raise ContractError("saved rollout time_s must be one-dimensional")
    frames = len(arrays["time_s"])
    if frames <= 0:
        raise ContractError("saved rollout must contain at least one frame")
    required_shapes = {
        "selected_clip_index": (frames,),
        "selected_frame": (frames,),
        "motion_feature_cost": (frames,),
        "terrain_feature_cost": (frames,),
        "total_feature_cost": (frames,),
        "selected_total_cost": (frames,),
        "step_time_ns": (frames,),
        "joint_position": (frames, 29),
        "root_position_world": (frames, 3),
        "root_orientation_world_wxyz": (frames, 4),
        "feature_body_position_world": (frames, 3, 3),
        "terrain_patch_position_world": (frames, 91, 3),
        "foot_clearance_m": (frames, 2),
        "progress_m": (frames,),
    }
    for name, shape in required_shapes.items():
        _require_array(arrays, name, shape)
    selected_clip = arrays["selected_clip_index"]
    selected_frame = arrays["selected_frame"]
    if selected_clip.dtype.kind not in "iu" or selected_frame.dtype.kind not in "iu":
        raise ContractError("saved rollout selection arrays must use integers")

    dataset_root = Path(resolved.get("dataset_root", "")).resolve()
    manifest_path = dataset_root / "manifest.json"
    if (
        not manifest_path.is_file()
        or _sha256(manifest_path) != resolved.get("dataset_manifest_sha256")
    ):
        raise ContractError("saved rollout dataset manifest hash mismatch")
    manifest = _read_json(manifest_path, "terrain dataset manifest")
    descriptors = manifest.get("clips")
    if not isinstance(descriptors, list) or len(descriptors) != 5:
        raise ContractError("terrain dataset clip manifest is invalid")
    clip_paths = tuple(
        descriptor.get("relative_motion_path", "")
        for descriptor in descriptors
        if isinstance(descriptor, dict)
    )
    if len(clip_paths) != 5 or any(not path for path in clip_paths):
        raise ContractError("terrain dataset clip paths are invalid")
    if (
        np.any(selected_clip < 0)
        or np.any(selected_clip >= len(clip_paths))
    ):
        raise ContractError("saved rollout selected clip index is invalid")
    query_scene = resolved.get("query_scene")
    matches = [
        descriptor
        for descriptor in descriptors
        if descriptor.get("relative_motion_path") == query_scene
    ]
    if len(matches) != 1:
        raise ContractError("saved rollout query scene mapping is invalid")
    descriptor = matches[0]
    motion_path = dataset_root / descriptor["relative_motion_path"]
    if (
        not motion_path.is_file()
        or _sha256(motion_path) != descriptor.get("motion_sha256")
    ):
        raise ContractError("saved rollout query motion hash mismatch")
    terrain = descriptor.get("terrain")
    if not isinstance(terrain, dict) or terrain.get("kind") != "heightgrid":
        raise ContractError("saved rollout query terrain is invalid")
    terrain_path = dataset_root / terrain.get("path", "")
    terrain_hash = terrain.get("sha256")
    if (
        not isinstance(terrain_hash, str)
        or not terrain_path.is_file()
        or _sha256(terrain_path) != terrain_hash
    ):
        raise ContractError("saved rollout terrain grid hash mismatch")
    try:
        with np.load(terrain_path, allow_pickle=False) as grid:
            if set(grid.files) != {"origin_xy", "cell_size_m", "height_z"}:
                raise ValueError("grid fields")
            origin = np.asarray(grid["origin_xy"], np.float32)
            cell_array = np.asarray(grid["cell_size_m"])
            height = np.asarray(grid["height_z"], np.float32)
        with np.load(motion_path, allow_pickle=False) as motion:
            root_position = np.asarray(motion["body_pos_w"][0, 0], np.float32)
            root_quaternion = np.asarray(motion["body_quat_w"][0, 0], np.float32)
    except Exception as error:
        raise ContractError("saved rollout terrain dependencies cannot be loaded") from error
    if (
        origin.shape != (2,)
        or cell_array.size != 1
        or height.ndim != 2
        or min(height.shape) < 2
        or not np.isfinite(height).all()
    ):
        raise ContractError("saved rollout terrain grid arrays are invalid")
    cell = float(cell_array.reshape(-1)[0])
    if not math.isfinite(cell) or cell <= 0.0:
        raise ContractError("saved rollout terrain cell size is invalid")
    for value in arrays.values():
        value.setflags(write=False)
    origin.setflags(write=False)
    height.setflags(write=False)
    return SavedTerrainRollout(
        run_root=run_root,
        arrays=MappingProxyType(arrays),
        metrics=MappingProxyType(metrics),
        resolved_config=MappingProxyType(resolved),
        frame_count=frames,
        terrain_origin_scene_xy=origin,
        terrain_cell_size_m=cell,
        terrain_height_z=height,
        terrain_sha256=terrain_hash,
        translation_scene_xy=root_position[:2].copy(),
        yaw_scene_from_matcher=_yaw_from_wxyz(root_quaternion),
        clip_paths=clip_paths,
    )


@dataclass
class PlaybackController:
    frame_count: int
    frame_index: int = 0
    paused: bool = False
    closed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.frame_count, int) or self.frame_count <= 0:
            raise ContractError("playback frame_count must be positive")

    def handle_key(self, key: str) -> None:
        normalized = str(key).lower()
        if normalized == "escape":
            self.closed = True
        elif normalized in (" ", "space"):
            self.paused = not self.paused
        elif normalized == "left":
            self.frame_index = max(0, self.frame_index - 1)
        elif normalized == "right":
            self.frame_index = min(self.frame_count - 1, self.frame_index + 1)
        elif normalized == "r":
            self.frame_index = 0


def _mujoco_positions(
    saved: SavedTerrainRollout,
    g1_xml: Path,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import mujoco
    except ImportError as error:
        raise ContractError("mujoco is required to reconstruct the saved skeleton") from error
    model = mujoco.MjModel.from_xml_path(str(g1_xml))
    if model.nq != 36:
        raise ContractError(f"G1 model nq must equal 36, got {model.nq}")
    data = mujoco.MjData(model)
    qpos = np.zeros(36, np.float64)
    qpos[:3] = saved.arrays["root_position_world"][frame_index]
    qpos[3:7] = saved.arrays["root_orientation_world_wxyz"][frame_index]
    target = saved.arrays["joint_position"][frame_index]
    source = np.empty(29, np.float64)
    source[np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)] = target
    qpos[7:] = source
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.xpos.copy(), np.asarray(model.body_parentid, np.int32)


def _draw_saved_frame(
    axes,
    saved: SavedTerrainRollout,
    *,
    g1_xml: Path,
    frame_index: int,
) -> None:
    axes.clear()
    height = saved.terrain_height_z
    ny, nx = height.shape
    stride = max(1, int(max(nx, ny) / 70))
    x_scene = (
        float(saved.terrain_origin_scene_xy[0])
        + np.arange(nx, dtype=np.float64) * saved.terrain_cell_size_m
    )
    y_scene = (
        float(saved.terrain_origin_scene_xy[1])
        + np.arange(ny, dtype=np.float64) * saved.terrain_cell_size_m
    )
    scene_x, scene_y = np.meshgrid(
        x_scene[::stride], y_scene[::stride], indexing="xy"
    )
    matcher_xy = saved.scene_to_matcher_xy(
        np.stack((scene_x, scene_y), axis=-1)
    )
    terrain_z = height[::stride, ::stride]
    axes.plot_surface(
        matcher_xy[..., 0],
        matcher_xy[..., 1],
        terrain_z,
        color="#9a866e",
        alpha=0.55,
        linewidth=0,
        antialiased=False,
    )

    body_position, parents = _mujoco_positions(
        saved, g1_xml, frame_index
    )
    for body in range(1, len(body_position)):
        parent = int(parents[body])
        if parent <= 0:
            continue
        segment = body_position[[parent, body]]
        axes.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color="#20242a",
            linewidth=1.8,
        )
    axes.scatter(
        body_position[1:, 0],
        body_position[1:, 1],
        body_position[1:, 2],
        color="#d9e2ec",
        edgecolor="#20242a",
        s=10,
        depthshade=False,
    )
    diagnostic = saved.arrays["feature_body_position_world"][frame_index]
    axes.scatter(
        diagnostic[:, 0],
        diagnostic[:, 1],
        diagnostic[:, 2],
        color=("#d62728", "#2ca02c", "#1f77b4"),
        s=28,
        depthshade=False,
    )
    patch = saved.arrays["terrain_patch_position_world"][frame_index]
    axes.scatter(
        patch[:, 0],
        patch[:, 1],
        patch[:, 2] + 0.006,
        color="#ffbf00",
        s=7,
        alpha=0.85,
        depthshade=False,
    )
    trajectory = saved.arrays["root_position_world"][: frame_index + 1]
    axes.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="#d62728",
        linewidth=2.0,
    )

    selected_clip = int(saved.arrays["selected_clip_index"][frame_index])
    selected_frame = int(saved.arrays["selected_frame"][frame_index])
    clearance = saved.arrays["foot_clearance_m"][frame_index]
    latency_ms = float(saved.arrays["step_time_ns"][frame_index]) / 1e6
    title = (
        f"{saved.metrics['condition']}  frame {frame_index + 1}/{saved.frame_count}\n"
        f"{saved.clip_paths[selected_clip]}:{selected_frame}  "
        f"motion={saved.arrays['motion_feature_cost'][frame_index]:.2f}  "
        f"terrain={saved.arrays['terrain_feature_cost'][frame_index]:.2f}  "
        f"total={saved.arrays['selected_total_cost'][frame_index]:.2f}\n"
        f"clearance L/R={clearance[0]:+.3f}/{clearance[1]:+.3f} m  "
        f"step={latency_ms:.2f} ms"
    )
    axes.set_title(title, fontsize=9)
    axes.set_xlabel("matcher X (m)")
    axes.set_ylabel("matcher Y (m)")
    axes.set_zlabel("Z (m)")
    focus = saved.arrays["root_position_world"][frame_index]
    axes.set_xlim(focus[0] - 1.2, focus[0] + 1.8)
    axes.set_ylim(focus[1] - 1.2, focus[1] + 1.2)
    axes.set_zlim(-0.05, max(1.8, float(focus[2]) + 0.5))
    axes.set_box_aspect((3.0, 2.4, 1.85))
    axes.view_init(elev=24, azim=-62)


def render_saved_frame(
    saved: SavedTerrainRollout,
    *,
    g1_xml: str | Path,
    frame_index: int,
    output_png: str | Path,
) -> None:
    if not 0 <= int(frame_index) < saved.frame_count:
        raise ContractError("viewer frame index is outside the saved rollout")
    g1_xml = Path(g1_xml).resolve()
    if not g1_xml.is_file():
        raise ContractError(f"G1 XML is missing: {g1_xml}")
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(10, 7), dpi=120)
    axes = figure.add_subplot(111, projection="3d")
    _draw_saved_frame(
        axes, saved, g1_xml=g1_xml, frame_index=int(frame_index)
    )
    figure.tight_layout()
    output_png = Path(output_png).resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png)
    plt.close(figure)


def show_saved_rollout(
    saved: SavedTerrainRollout, *, g1_xml: str | Path
) -> None:
    from matplotlib import pyplot as plt

    g1_xml = Path(g1_xml).resolve()
    controller = PlaybackController(saved.frame_count)
    figure = plt.figure(figsize=(11, 8))
    axes = figure.add_subplot(111, projection="3d")

    def redraw() -> None:
        _draw_saved_frame(
            axes,
            saved,
            g1_xml=g1_xml,
            frame_index=controller.frame_index,
        )
        figure.canvas.draw_idle()

    def key_press(event) -> None:
        controller.handle_key(event.key)
        if controller.closed:
            plt.close(figure)
        else:
            redraw()

    def timer_tick() -> None:
        if (
            not controller.closed
            and not controller.paused
            and controller.frame_index < controller.frame_count - 1
        ):
            controller.frame_index += 1
            redraw()

    figure.canvas.mpl_connect("key_press_event", key_press)
    timer = figure.canvas.new_timer(interval=20)
    timer.add_callback(timer_tick)
    timer.start()
    redraw()
    plt.show()


def save_saved_video(
    saved: SavedTerrainRollout,
    *,
    g1_xml: str | Path,
    output_mp4: str | Path,
    fps: int = 50,
) -> None:
    if not isinstance(fps, int) or fps <= 0:
        raise ContractError("video fps must be a positive integer")
    from matplotlib import animation, pyplot as plt

    g1_xml = Path(g1_xml).resolve()
    output_mp4 = Path(output_mp4).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11, 8), dpi=120)
    axes = figure.add_subplot(111, projection="3d")
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5000,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    with writer.saving(figure, str(output_mp4), dpi=120):
        for frame_index in range(saved.frame_count):
            _draw_saved_frame(
                axes,
                saved,
                g1_xml=g1_xml,
                frame_index=frame_index,
            )
            figure.tight_layout()
            writer.grab_frame()
    plt.close(figure)


def build_viewer_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View an authenticated saved Torch terrain rollout."
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output-png")
    parser.add_argument("--output-mp4")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_viewer_argument_parser().parse_args(argv)
    if args.output_png and args.output_mp4:
        raise ContractError("choose only one of --output-png or --output-mp4")
    saved = load_saved_rollout(args.run)
    if args.output_mp4:
        save_saved_video(
            saved,
            g1_xml=args.g1_xml,
            output_mp4=args.output_mp4,
        )
    elif args.output_png:
        render_saved_frame(
            saved,
            g1_xml=args.g1_xml,
            frame_index=args.frame,
            output_png=args.output_png,
        )
    else:
        show_saved_rollout(saved, g1_xml=args.g1_xml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
