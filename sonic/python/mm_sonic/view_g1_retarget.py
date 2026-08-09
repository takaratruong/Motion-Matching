"""Interactively inspect a validated native G1 retarget artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

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


def view_motion(*, motion_path: Path, gmr_root: Path) -> None:
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

        viewer = RobotMotionViewer(
            robot_type="unitree_g1",
            motion_fps=int(fps),
            camera_follow=True,
            keyboard_callback=keyboard_callback,
        )
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    print(
        json.dumps(
            {
                "motion": str(args.motion.resolve()),
                "gmr_root": str(args.gmr_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    view_motion(motion_path=args.motion, gmr_root=args.gmr_root)


if __name__ == "__main__":
    main()
