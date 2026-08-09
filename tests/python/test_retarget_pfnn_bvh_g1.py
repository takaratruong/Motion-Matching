import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.retarget_pfnn_bvh_g1 import (
    frame_slice,
    postprocess_motion,
    prepare_pfnn_bvh,
    retarget_slices,
    scale_pfnn_frames,
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
        self.assertIn("JOINT LeftToeBase", contents)
        self.assertIn("JOINT RightToeBase", contents)
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

    def test_derives_monotone_step_platforms_from_support_heights(self) -> None:
        from mm_sonic.view_g1_retarget import derive_step_platforms

        root = np.zeros((8, 3), dtype=np.float64)
        root[:, 0] = np.arange(8) * 0.1
        support = np.array([0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15])
        platforms = derive_step_platforms(root, support, group_frames=2)
        self.assertEqual(len(platforms), 4)
        np.testing.assert_allclose(
            [platform.top_height for platform in platforms],
            [0.0, 0.1, 0.2, 0.2],
            atol=1.0e-12,
        )
        self.assertTrue(all(platform.half_length >= 0.25 for platform in platforms))
        self.assertTrue(all(platform.half_width == 0.4 for platform in platforms))
        np.testing.assert_allclose([platform.yaw for platform in platforms], 0.0)

        with self.assertRaises(ValueError):
            derive_step_platforms(root, support[:-1], group_frames=2)

    def test_selects_bounded_debug_slice(self) -> None:
        selected = frame_slice(total_frames=8171, start_frame=240, frame_count=120)
        self.assertEqual(selected, slice(240, 360))
        self.assertEqual(
            frame_slice(total_frames=8171, start_frame=0, frame_count=None),
            slice(0, 8171),
        )
        for values in (
            {"total_frames": 0, "start_frame": 0, "frame_count": 1},
            {"total_frames": 10, "start_frame": -1, "frame_count": 1},
            {"total_frames": 10, "start_frame": 10, "frame_count": 1},
            {"total_frames": 10, "start_frame": 0, "frame_count": 0},
            {"total_frames": 10, "start_frame": 5, "frame_count": 6},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                frame_slice(**values)

    def test_scales_pfnn_positions_without_changing_orientations(self) -> None:
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        frames = [
            {
                "Hips": [np.array([1.0, 2.0, 3.0]), orientation.copy()],
                "LeftFoot": [np.array([-1.0, 0.0, 4.0]), orientation.copy()],
            }
        ]
        scaled = scale_pfnn_frames(frames)
        np.testing.assert_allclose(scaled[0]["Hips"][0], [5.6444, 11.2888, 16.9332])
        np.testing.assert_allclose(scaled[0]["LeftFoot"][0], [-5.6444, 0.0, 22.5776])
        np.testing.assert_array_equal(scaled[0]["Hips"][1], orientation)
        np.testing.assert_array_equal(frames[0]["Hips"][0], [1.0, 2.0, 3.0])

    def test_selects_hidden_warmup_before_display_slice(self) -> None:
        warmup, displayed = retarget_slices(
            total_frames=8171,
            start_frame=7700,
            frame_count=120,
            warmup_frames=120,
        )
        self.assertEqual(warmup, slice(7580, 7700))
        self.assertEqual(displayed, slice(7700, 7820))
        self.assertEqual(
            retarget_slices(
                total_frames=100,
                start_frame=20,
                frame_count=10,
                warmup_frames=30,
            ),
            (slice(0, 20), slice(20, 30)),
        )
        with self.assertRaises(ValueError):
            retarget_slices(
                total_frames=100,
                start_frame=20,
                frame_count=10,
                warmup_frames=-1,
            )

    def test_source_grounding_preserves_the_gmr_world_pose(self) -> None:
        class Motion:
            def __init__(self) -> None:
                self.root_pos = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.5]])

        class Project:
            @staticmethod
            def postprocess(motion):
                motion.root_pos = motion.root_pos.copy()
                motion.root_pos[:, 2] -= 0.5
                return motion

        source = Motion()
        preserved, offset = postprocess_motion(
            source, project_driver=Project(), grounding="source"
        )
        np.testing.assert_array_equal(
            preserved.root_pos, [[0.0, 0.0, 1.0], [1.0, 2.0, 1.5]]
        )
        self.assertEqual(offset, 0.0)

        flattened, offset = postprocess_motion(
            Motion(), project_driver=Project(), grounding="flat"
        )
        np.testing.assert_array_equal(
            flattened.root_pos, [[0.0, 0.0, 0.5], [1.0, 2.0, 1.0]]
        )
        self.assertEqual(offset, 0.5)


if __name__ == "__main__":
    unittest.main()
