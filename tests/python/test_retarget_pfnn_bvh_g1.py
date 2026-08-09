import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.retarget_pfnn_bvh_g1 import (
    prepare_pfnn_bvh,
    validate_g1_motion,
)


def _bvh_text(
    *,
    frames: int = 2,
    frame_time: str = "0.008333",
    spine: str = "Spine1",
    include_left_toe: bool = True,
    include_right_toe: bool = True,
    motion_rows: int | None = None,
) -> str:
    left_toe = "\n        JOINT LeftToeBase\n        {\n          OFFSET 0 0 1\n          CHANNELS 3 Zrotation Xrotation Yrotation\n        }" if include_left_toe else ""
    right_toe = "\n        JOINT RightToeBase\n        {\n          OFFSET 0 0 1\n          CHANNELS 3 Zrotation Xrotation Yrotation\n        }" if include_right_toe else ""
    rows = frames if motion_rows is None else motion_rows
    values = " ".join("0" for _ in range(24))
    motion = "\n".join(values for _ in range(rows))
    return f"""HIERARCHY
ROOT Hips
{{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT {spine}
  {{
    OFFSET 0 1 0
    CHANNELS 3 Zrotation Xrotation Yrotation
  }}
  JOINT LeftFoot
  {{
    OFFSET 0 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation{left_toe}
  }}
  JOINT RightFoot
  {{
    OFFSET 0 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation{right_toe}
  }}
}}
MOTION
Frames: {frames}
Frame Time: {frame_time}
{motion}
"""


class PFNNBVHRetargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _prepare(self, text: str):
        source = self.root / "source.bvh"
        prepared = self.root / "prepared.bvh"
        source.write_text(text)
        return prepare_pfnn_bvh(source, prepared), prepared

    def test_prepares_native_120_hz_source(self) -> None:
        receipt, prepared = self._prepare(_bvh_text())
        contents = prepared.read_text()
        self.assertEqual(receipt.frame_count, 2)
        self.assertEqual(receipt.fps, 120.0)
        self.assertIn("JOINT Spine2", contents)
        self.assertNotIn("JOINT Spine1", contents)
        self.assertEqual(receipt.aliases, (("Spine1", "Spine2"),))
        self.assertEqual(len(receipt.source_sha256), 64)
        self.assertEqual(len(receipt.prepared_sha256), 64)

        invalid = {
            "missing Spine1": _bvh_text(spine="Spine"),
            "existing Spine2": _bvh_text(spine="Spine2"),
            "frame mismatch": _bvh_text(motion_rows=1),
            "not 120 Hz": _bvh_text(frame_time="0.016667"),
            "missing left toe": _bvh_text(include_left_toe=False),
            "missing right toe": _bvh_text(include_right_toe=False),
        }
        for label, text in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self._prepare(text)

    def test_validates_native_g1_artifact(self) -> None:
        frames = 4
        limits = np.repeat(np.array([[-1.0, 1.0]]), 29, axis=0)
        motion = {
            "root_pos": np.zeros((frames, 3), dtype=np.float64),
            "root_quat": np.tile(
                np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
                (frames, 1),
            ),
            "dof": np.zeros((frames, 29), dtype=np.float64),
            "fps": 120.0,
        }
        validate_g1_motion(
            motion,
            expected_frames=frames,
            expected_fps=120.0,
            joint_limits=limits,
        )

        invalid = {}
        invalid["root shape"] = {**motion, "root_pos": np.zeros((frames, 2))}
        invalid["quaternion shape"] = {
            **motion,
            "root_quat": np.zeros((frames, 3)),
        }
        invalid["joint shape"] = {**motion, "dof": np.zeros((frames, 28))}
        invalid["nonfinite"] = {
            **motion,
            "root_pos": np.full((frames, 3), np.nan),
        }
        invalid["quaternion norm"] = {
            **motion,
            "root_quat": np.zeros((frames, 4)),
        }
        beyond = motion["dof"].copy()
        beyond[1, 7] = 1.00001
        invalid["joint limit"] = {**motion, "dof": beyond}
        invalid["wrong fps"] = {**motion, "fps": 60.0}

        for label, value in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                validate_g1_motion(
                    value,
                    expected_frames=frames,
                    expected_fps=120.0,
                    joint_limits=limits,
                )

        with self.assertRaises(ValueError):
            validate_g1_motion(
                motion,
                expected_frames=frames + 1,
                expected_fps=120.0,
                joint_limits=limits,
            )

    def test_viewer_playback_controls(self) -> None:
        from mm_sonic.view_g1_retarget import PlaybackState

        state = PlaybackState(frame_count=3)
        self.assertEqual(state.frame_index, 0)
        self.assertFalse(state.paused)

        state.toggle_pause()
        self.assertTrue(state.paused)
        state.advance()
        self.assertEqual(state.frame_index, 0)
        state.seek(+1)
        self.assertEqual(state.frame_index, 1)
        state.seek(-2)
        self.assertEqual(state.frame_index, 2)
        state.home()
        self.assertEqual(state.frame_index, 0)

        state.toggle_pause()
        state.advance()
        self.assertEqual(state.frame_index, 1)
        state.seek(+2)
        self.assertEqual(state.frame_index, 0)

        with self.assertRaises(ValueError):
            PlaybackState(frame_count=0)


if __name__ == "__main__":
    unittest.main()
