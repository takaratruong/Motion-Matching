import tempfile
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.torch_motion_matcher import TorchMotionMatcher
from mm_sonic.torch_motion_sonic import (
    SonicReferenceAdapter,
    canonical_target_buffer_from_match,
    initial_qpos_from_match,
)
from tests.python.torch_motion_test_utils import (
    build_varying_takara_arrays,
    write_takara_arrays,
)


class _Publisher:
    def __init__(self, events):
        self.events = events

    def prepare(self, buffer, **_kwargs):
        self.events.append("prepare")
        return buffer

    def send_prepared(self, prepared):
        self.events.append("send")
        return {"count": prepared.count}


class _Gate:
    def __init__(self, events):
        self.events = events

    def prepare_release(self, *, expected_stream_frame_end):
        self.events.append(f"arm:{expected_stream_frame_end}")
        return ("release", expected_stream_frame_end)


class TorchMotionSonicTests(unittest.TestCase):
    def _matcher(self, root):
        arrays = build_varying_takara_arrays(frames=100)
        write_takara_arrays(root / "walk", arrays)
        matcher = TorchMotionMatcher.from_folder(root, device="cpu")
        return matcher, matcher.reset()

    def test_dense_buffer_and_initial_qpos_are_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            matcher, reset = self._matcher(Path(tmp))
            buffer = canonical_target_buffer_from_match(
                reset, global_frame_start=7
            )
            self.assertEqual(buffer.count, 46)
            np.testing.assert_array_equal(buffer.frame_index, np.arange(7, 53))
            np.testing.assert_array_equal(
                buffer.joint_position,
                reset.dense_joint_position_window.numpy(),
            )
            qpos = initial_qpos_from_match(reset)
            self.assertEqual(qpos.shape, (36,))
            np.testing.assert_allclose(qpos[:3], reset.root_position_world.numpy())
            np.testing.assert_allclose(
                qpos[3:7], reset.root_orientation_world_wxyz.numpy()
            )
            self.assertEqual(matcher.motion_inventory_sha256.__len__(), 64)

    def test_prepare_send_then_arm_advances_ack_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            matcher, _reset = self._matcher(Path(tmp))
            result = matcher.step((0.5, 0.0), 0.0)
            events = []
            adapter = SonicReferenceAdapter(
                _Publisher(events),
                _Gate(events),
                initial_acked_sequence=0,
                initial_global_frame_end=45,
            )
            prepared = adapter.prepare(result, global_frame_start=1)
            self.assertEqual(events, ["prepare"])
            ack = adapter.send_and_wait(prepared)
            self.assertEqual(events, ["prepare", "send", "arm:46"])
            self.assertEqual(ack.publication.global_frame_end, 46)
            with self.assertRaises(Exception):
                adapter.send_and_wait(prepared)


if __name__ == "__main__":
    unittest.main()
