from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from mm_sonic.hands import (
    LEFT_HAND_JOINT_ORDER,
    RIGHT_HAND_JOINT_ORDER,
)
from mm_sonic.joints import ContractError, TARGET_JOINT_ORDER
from mm_sonic.manual_replay import _count_state_rows, render_manual_replay
from mm_sonic.scene import normalize_run_local_actuators


def _synthetic_scene(scene_dir: Path) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    joint_order = list(TARGET_JOINT_ORDER) + list(LEFT_HAND_JOINT_ORDER) + list(
        RIGHT_HAND_JOINT_ORDER
    )
    bodies = []
    for index, joint in enumerate(joint_order):
        bodies.append(
            f'<body name="{joint}_link" pos="{0.03 * index:.4g} 0 1.0">'
            f'<joint name="{joint}" type="hinge" axis="0 1 0" range="-2 2"/>'
            f'<geom name="{joint}_geom" type="sphere" size="0.01" density="100"/>'
            "</body>"
        )
    motors = "".join(
        f'<motor name="{joint}_motor" joint="{joint}" gear="1"/>'
        for joint in reversed(joint_order)
    )
    robot_xml = (
        '<mujoco model="replay_synthetic">'
        '<compiler angle="radian"/>'
        '<worldbody><body name="pelvis" pos="0 0 0.2">'
        '<freejoint name="floating_base_joint"/>'
        '<geom name="pelvis_geom" type="sphere" size="0.1" density="100"/>'
        + "".join(bodies)
        + "</body></worldbody>"
        + f"<actuator>{motors}</actuator>"
        + "</mujoco>\n"
    ).encode("utf-8")
    robot_bytes, _, _ = normalize_run_local_actuators(
        robot_xml, label="run-local robot XML"
    )
    robot_path = scene_dir / "gear_robot.xml"
    robot_path.write_bytes(robot_bytes)
    scene_bytes = (
        '<mujoco model="replay_scene">'
        f'<include file="{robot_path}"/>'
        '<worldbody><geom name="floor" type="plane" size="0 0 .05"/>'
        "</worldbody></mujoco>\n"
    ).encode("utf-8")
    (scene_dir / "gear_scene.xml").write_bytes(scene_bytes)


def _write_states(state_log: Path, rows: int, nq: int, nv: int) -> None:
    state_log.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(rows):
        qpos = [0.0] * nq
        qpos[2] = 0.9
        qpos[3] = 1.0
        lines.append(
            json.dumps(
                {
                    "step": 4 * (index + 1),
                    "sim_time_s": 0.02 * (index + 1),
                    "state": {"qpos": qpos, "qvel": [0.0] * nv},
                },
                separators=(",", ":"),
            )
        )
    state_log.write_text("\n".join(lines) + "\n", encoding="utf-8")


class StateRowCountTests(unittest.TestCase):
    def test_counts_short_and_long_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "state.jsonl"
            _write_states(log, 5, nq=50, nv=49)
            self.assertEqual(_count_state_rows(log), 5)
            _write_states(log, 600, nq=50, nv=49)
            self.assertEqual(_count_state_rows(log), 600)

    def test_rejects_empty_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "state.jsonl"
            log.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "at least one row"):
                _count_state_rows(log)


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") is not None
    and shutil.which("ffmpeg") is not None,
    "mujoco and ffmpeg are required for the guarded renderer smoke test",
)
class RenderManualReplayTests(unittest.TestCase):
    def test_short_stream_emits_video_montage_and_closeup(self) -> None:
        import mujoco

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            _synthetic_scene(root / "scene")
            model = mujoco.MjModel.from_xml_path(str(root / "scene/gear_scene.xml"))
            _write_states(
                root / "scored-sim-logs/state.jsonl",
                rows=5,
                nq=model.nq,
                nv=model.nv,
            )
            video, montage, closeup = render_manual_replay(
                root, Path(temporary) / "out"
            )
            for path in (video, montage, closeup):
                self.assertTrue(path.exists())
            self.assertEqual(closeup.name, "g1-sonic-hand-closeup.png")
            self.assertGreater(closeup.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
