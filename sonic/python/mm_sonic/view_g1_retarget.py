"""Interactively inspect a validated native G1 retarget artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

from mm_sonic.retarget_pfnn_bvh_g1 import G1_JOINT_NAMES, validate_g1_motion


@dataclass
class PlaybackState:
    frame_count: int
    frame_index: int = 0
    paused: bool = False

    def __post_init__(self) -> None:
        if type(self.frame_count) is not int or self.frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if type(self.frame_index) is not int:
            raise ValueError("frame_index must be an integer")
        self.frame_index %= self.frame_count

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def seek(self, delta: int) -> None:
        if type(delta) is not int:
            raise ValueError("seek delta must be an integer")
        self.frame_index = (self.frame_index + delta) % self.frame_count

    def home(self) -> None:
        self.frame_index = 0

    def advance(self) -> None:
        if not self.paused:
            self.seek(1)


@dataclass(frozen=True)
class TerrainPlatform:
    center_x: float
    center_y: float
    yaw: float
    top_height: float
    half_length: float
    half_width: float = 0.4


def derive_step_platforms(
    root_position: np.ndarray,
    support_height: np.ndarray,
    *,
    group_frames: int = 15,
) -> tuple[TerrainPlatform, ...]:
    """Build a visible monotone stair path from retargeted support heights."""

    root = np.asarray(root_position, dtype=np.float64)
    support = np.asarray(support_height, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3 or len(root) < 2:
        raise ValueError("root_position must have finite shape [T, 3] with T >= 2")
    if support.shape != (len(root),):
        raise ValueError("support_height must have shape [T]")
    if not np.isfinite(root).all() or not np.isfinite(support).all():
        raise ValueError("terrain inputs must be finite")
    if type(group_frames) is not int or group_frames < 1:
        raise ValueError("group_frames must be a positive integer")

    raw_heights: list[float] = []
    chunks: list[tuple[int, int]] = []
    for start in range(0, len(root), group_frames):
        stop = min(start + group_frames, len(root))
        chunks.append((start, stop))
        raw_heights.append(float(np.median(support[start:stop])))
    heights = np.maximum.accumulate(np.asarray(raw_heights, dtype=np.float64))
    heights -= min(0.0, float(heights[0]))

    platforms: list[TerrainPlatform] = []
    for index, (start, stop) in enumerate(chunks):
        points = root[start:stop, :2]
        center = np.mean(points, axis=0)
        before = root[max(0, start - 1), :2]
        after = root[min(len(root) - 1, stop), :2]
        direction = after - before
        if np.linalg.norm(direction) < 1.0e-8:
            yaw = platforms[-1].yaw if platforms else 0.0
        else:
            yaw = float(np.arctan2(direction[1], direction[0]))
        along = np.array([np.cos(yaw), np.sin(yaw)])
        extent = float(np.ptp(points @ along)) if len(points) > 1 else 0.0
        platforms.append(
            TerrainPlatform(
                center_x=float(center[0]),
                center_y=float(center[1]),
                yaw=yaw,
                top_height=max(0.0, float(heights[index])),
                half_length=max(0.25, 0.5 * extent + 0.18),
            )
        )
    return tuple(platforms)


def load_motion(path: Path) -> dict[str, object]:
    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "root_pos",
            "root_quat",
            "dof",
            "fps",
            "engine",
            "joint_names",
            "joint_limits",
        }
        if set(archive.files) != required:
            raise ValueError(
                f"motion artifact fields must be exactly {sorted(required)}, got {sorted(archive.files)}"
            )
        motion: dict[str, object] = {
            "root_pos": np.asarray(archive["root_pos"], dtype=np.float64).copy(),
            "root_quat": np.asarray(archive["root_quat"], dtype=np.float64).copy(),
            "dof": np.asarray(archive["dof"], dtype=np.float64).copy(),
            "fps": float(np.asarray(archive["fps"]).item()),
            "engine": str(np.asarray(archive["engine"]).item()),
            "joint_names": tuple(str(value) for value in archive["joint_names"].tolist()),
            "joint_limits": np.asarray(archive["joint_limits"], dtype=np.float64).copy(),
        }
    if motion["engine"] != "gmr":
        raise ValueError("approval artifact must be produced by GMR")
    if motion["joint_names"] != G1_JOINT_NAMES:
        raise ValueError("approval artifact joint order does not match the native G1 ABI")
    frames = len(np.asarray(motion["root_pos"]))
    validate_g1_motion(
        motion,
        expected_frames=frames,
        expected_fps=120.0,
        joint_limits=np.asarray(motion["joint_limits"]),
    )
    return motion


def _support_heights(motion: dict[str, object], model: object) -> np.ndarray:
    import mujoco

    root_pos = np.asarray(motion["root_pos"])
    root_quat = np.asarray(motion["root_quat"])
    dof = np.asarray(motion["dof"])
    data = mujoco.MjData(model)
    heel = np.array([-0.066, 0.0, -0.034], dtype=np.float64)
    toe = np.array([0.12, 0.0, -0.034], dtype=np.float64)
    body_ids = tuple(
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{side}_ankle_roll_link",
        )
        for side in ("left", "right")
    )
    support = np.empty(len(root_pos), dtype=np.float64)
    for frame in range(len(root_pos)):
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame, [3, 0, 1, 2]]
        data.qpos[7:] = dof[frame]
        mujoco.mj_forward(model, data)
        heights = []
        for body_id in body_ids:
            rotation = data.xmat[body_id].reshape(3, 3)
            position = data.xpos[body_id]
            heights.extend(((rotation @ heel + position)[2], (rotation @ toe + position)[2]))
        support[frame] = min(heights)
    return support


def _terrain_model_xml(gmr_root: Path, platforms: tuple[TerrainPlatform, ...]) -> str:
    source = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("GMR G1 model is missing its compiler element")
    compiler.set("meshdir", str((source.parent / "meshes").resolve(strict=True)))
    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError("GMR G1 model is missing worldbody")
    world = worldbodies[-1]
    for index, platform in enumerate(platforms):
        base_height = -0.05
        half_height = max(0.025, 0.5 * (platform.top_height - base_height))
        center_z = platform.top_height - half_height
        half_yaw = 0.5 * platform.yaw
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"retarget_step_{index:03d}",
                "type": "box",
                "pos": f"{platform.center_x:.9g} {platform.center_y:.9g} {center_z:.9g}",
                "size": f"{platform.half_length:.9g} {platform.half_width:.9g} {half_height:.9g}",
                "quat": f"{np.cos(half_yaw):.9g} 0 0 {np.sin(half_yaw):.9g}",
                "rgba": "0.28 0.42 0.22 1",
                "contype": "1",
                "conaffinity": "1",
            },
        )
    return ET.tostring(root, encoding="unicode")


class _TerrainMotionViewer:
    def __init__(self, *, gmr_root: Path, motion: dict[str, object], keyboard_callback) -> None:
        import mujoco
        import mujoco.viewer
        from loop_rate_limiters import RateLimiter

        base_model = mujoco.MjModel.from_xml_path(
            str(gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")
        )
        support = _support_heights(motion, base_model)
        self.platforms = derive_step_platforms(
            np.asarray(motion["root_pos"]), support, group_frames=15
        )
        self.model = mujoco.MjModel.from_xml_string(
            _terrain_model_xml(gmr_root, self.platforms)
        )
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=keyboard_callback,
        )
        self.rate_limiter = RateLimiter(frequency=float(motion["fps"]), warn=False)
        self.pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

    def step(self, *, root_pos, root_rot, dof_pos, **_ignored) -> None:
        import mujoco

        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot
        self.data.qpos[7:] = dof_pos
        mujoco.mj_forward(self.model, self.data)
        self.viewer.cam.lookat = self.data.xpos[self.pelvis_id]
        self.viewer.cam.distance = 2.3
        self.viewer.cam.elevation = -14
        self.viewer.sync()
        self.rate_limiter.sleep()

    def close(self) -> None:
        self.viewer.close()


def view_motion(*, motion_path: Path, gmr_root: Path, terrain: str = "none") -> None:
    motion = load_motion(motion_path)
    gmr_root = Path(gmr_root).resolve(strict=True)
    sys.path.insert(0, str(gmr_root))
    try:
        from general_motion_retargeting import RobotMotionViewer
        from mujoco.glfw import glfw

        root_pos = np.asarray(motion["root_pos"])
        root_quat_xyzw = np.asarray(motion["root_quat"])
        dof = np.asarray(motion["dof"])
        fps = float(motion["fps"])
        state = PlaybackState(frame_count=len(root_pos))
        holder: dict[str, object] = {}

        def keyboard_callback(key: int) -> None:
            if key == glfw.KEY_SPACE:
                state.toggle_pause()
            elif key == glfw.KEY_RIGHT:
                state.seek(1)
            elif key == glfw.KEY_LEFT:
                state.seek(-1)
            elif key == glfw.KEY_HOME:
                state.home()
            elif key == glfw.KEY_ESCAPE and "viewer" in holder:
                holder["viewer"].close()

        if terrain == "auto-steps":
            viewer = _TerrainMotionViewer(
                gmr_root=gmr_root,
                motion=motion,
                keyboard_callback=keyboard_callback,
            )
            print(f"Terrain: {len(viewer.platforms)} support-height platforms", flush=True)
        elif terrain == "none":
            viewer = RobotMotionViewer(
                robot_type="unitree_g1",
                motion_fps=int(fps),
                camera_follow=True,
                keyboard_callback=keyboard_callback,
            )
        else:
            raise ValueError(f"unsupported terrain mode {terrain!r}")
        holder["viewer"] = viewer
        print(
            "Controls: Space pause/resume | Left/Right seek | Home restart | Escape close",
            flush=True,
        )
        last_status = 0.0
        try:
            while viewer.viewer.is_running():
                index = state.frame_index
                quaternion_wxyz = root_quat_xyzw[index, [3, 0, 1, 2]]
                viewer.step(
                    root_pos=root_pos[index],
                    root_rot=quaternion_wxyz,
                    dof_pos=dof[index],
                    rate_limit=True,
                    follow_camera=True,
                )
                now = time.monotonic()
                if now - last_status >= 1.0:
                    print(
                        f"frame {index + 1}/{state.frame_count} | "
                        f"time {index / fps:.3f}s | {fps:.0f} Hz | "
                        f"{'paused' if state.paused else 'playing'}",
                        flush=True,
                    )
                    last_status = now
                state.advance()
        finally:
            if viewer.viewer.is_running():
                viewer.close()
    finally:
        if sys.path and sys.path[0] == str(gmr_root):
            sys.path.pop(0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--terrain", choices=("none", "auto-steps"), default="none")
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            {
                "motion": str(args.motion.resolve()),
                "gmr_root": str(args.gmr_root.resolve()),
                "terrain": args.terrain,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    view_motion(motion_path=args.motion, gmr_root=args.gmr_root, terrain=args.terrain)


if __name__ == "__main__":
    main()
