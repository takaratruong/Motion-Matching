from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np


GMR_ROOT = Path("/home/ubuntu/.cache/native-g1-pfnn/GMR")
WORKTREE = Path("/home/ubuntu/worktrees/motion-matching-full-walking-integration")
PFNN_BVH = Path(
    "/home/ubuntu/datasets/pfnn/pfnn/data/animations/LocomotionFlat01_000.bvh"
)
sys.path.insert(0, str(GMR_ROOT))
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "sonic/python"))


class PFNNG1BridgeTests(unittest.TestCase):
    def test_gmr_initialization_does_not_corrupt_json_stdout(self) -> None:
        from mm_sonic.published_pfnn_g1_bridge import G1RetargetBridge

        captured = io.StringIO()
        with redirect_stdout(captured):
            G1RetargetBridge(GMR_ROOT)
        self.assertEqual(captured.getvalue(), "")

    def test_parses_fixed_little_endian_export_record(self) -> None:
        from mm_sonic.published_pfnn_g1_bridge import read_header, read_record

        header = struct.pack(
            "<8sIIIIfI", b"PFNNXFM\0", 1, 31, 912, 1, 60.0, 0
        )
        prefix = struct.pack(
            "<QIIff3f2f", 17, 4, 3, 1.25, 42.5, 10.0, 20.0, 30.0, 0.6, 0.8
        )
        pose = np.arange(31 * 7, dtype="<f4").tobytes()
        stream = io.BytesIO(header + prefix + pose)

        metadata = read_header(stream)
        record = read_record(stream, metadata)

        self.assertEqual(metadata.joint_count, 31)
        self.assertEqual(metadata.fps, 60.0)
        self.assertEqual(record.frame, 17)
        self.assertEqual(record.world, 4)
        self.assertEqual(record.flags, 3)
        np.testing.assert_array_equal(
            record.positions[0], np.asarray((0.0, 1.0, 2.0), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            record.quaternions_wxyz[-1],
            np.asarray((213.0, 214.0, 215.0, 216.0), dtype=np.float32),
        )

    def test_source_pose_conversion_matches_existing_offline_gmr_contract(self) -> None:
        from general_motion_retargeting.utils.lafan1 import load_bvh_file
        from general_motion_retargeting.utils.lafan_vendor import utils
        from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
        from mm_sonic.retarget_pfnn_bvh_g1 import (
            prepare_pfnn_bvh,
            scale_pfnn_frames,
        )
        from mm_sonic.published_pfnn_g1_bridge import (
            SOURCE_JOINT_NAMES,
            source_pose_to_gmr_frame,
        )

        source = read_bvh(str(PFNN_BVH))
        global_quaternion, global_position = utils.quat_fk(
            source.quats, source.pos, source.parents
        )
        self.assertEqual(tuple(source.bones), SOURCE_JOINT_NAMES)
        frame_index = 123

        actual, source_root = source_pose_to_gmr_frame(
            global_position[frame_index] * 5.6444,
            global_quaternion[frame_index],
        )

        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared.bvh"
            prepare_pfnn_bvh(PFNN_BVH, prepared)
            expected_frames, _ = load_bvh_file(str(prepared), format="nokov")
            expected = scale_pfnn_frames([expected_frames[frame_index]])[0]

        self.assertEqual(set(actual), set(expected))
        for name in expected:
            np.testing.assert_allclose(actual[name][0], expected[name][0], atol=1e-9)
            self.assertAlmostEqual(
                abs(float(np.dot(actual[name][1], expected[name][1]))),
                1.0,
                places=7,
            )
        np.testing.assert_allclose(source_root, expected["Hips"][0], atol=1e-9)

    def test_released_runtime_centimeters_are_not_scaled_twice(self) -> None:
        from mm_sonic.published_pfnn_g1_bridge import source_pose_to_gmr_frame

        positions = np.zeros((31, 3), dtype=np.float64)
        positions[0] = (100.0, 200.0, 300.0)
        quaternions = np.zeros((31, 4), dtype=np.float64)
        quaternions[:, 0] = 1.0

        _, root = source_pose_to_gmr_frame(positions, quaternions)

        np.testing.assert_allclose(root, (1.0, -3.0, 2.0), atol=1e-12)

    def test_bridge_preserves_gmr_root_without_viewer_correction(self) -> None:
        from mm_sonic.published_pfnn_g1_bridge import ExportRecord, G1RetargetBridge

        expected = np.linspace(-0.5, 0.5, 36, dtype=np.float64)

        class FakeRetargeter:
            def retarget(self, frame):
                self.frame = frame
                return expected.copy()

        bridge = G1RetargetBridge.__new__(G1RetargetBridge)
        bridge._retargeter = FakeRetargeter()
        bridge._world = None
        bridge._previous_frame = None
        bridge._source_root_anchor = None
        bridge._gmr_root_anchor = None
        positions = np.zeros((31, 3), dtype=np.float64)
        positions[0] = (100.0, 200.0, 300.0)
        quaternions = np.zeros((31, 4), dtype=np.float64)
        quaternions[:, 0] = 1.0
        record = ExportRecord(
            frame=0,
            world=2,
            flags=1,
            phase=0.0,
            sampled_root_height=0.0,
            trajectory_root_xyz=np.zeros(3),
            trajectory_forward_xz=np.asarray((0.0, 1.0)),
            positions=positions,
            quaternions_wxyz=quaternions,
        )

        output = bridge.retarget(record)

        np.testing.assert_allclose(output["qpos_wxyz"], expected, atol=0.0)
        np.testing.assert_allclose(
            output["source_root_pos_zup_m"], expected[:3], atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
