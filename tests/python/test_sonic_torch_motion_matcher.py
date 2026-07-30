import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from mm_sonic.torch_motion_matcher import (
    TorchMotionMatcher,
    decay_spring_offsets,
)
from mm_sonic.torch_motion_data import MotionFolder
from tests.python.torch_motion_test_utils import (
    build_varying_takara_arrays,
    write_takara_arrays,
)


class _MatcherExtension:
    name = "matcher_test"
    dimension = 2
    weight = 1.0

    def __init__(self):
        self.query_calls = 0

    def database_rows(self, folder, device):
        output = []
        for clip in folder.clips:
            frame = torch.arange(
                clip.valid_frame_stop, dtype=torch.float32, device=device
            )
            output.append(torch.stack((frame, frame.square() + frame), dim=1))
        return tuple(output)

    def query_row(self, state, trajectory):
        self.query_calls += 1
        value = state.root_position_world[0]
        return torch.stack((value, value.square() + value))


class TorchMotionMatcherTests(unittest.TestCase):
    def test_spring_matches_independent_equation(self):
        position = torch.tensor([0.4, -0.2], dtype=torch.float32)
        velocity = torch.tensor([-0.1, 0.3], dtype=torch.float32)
        time = torch.tensor(0.17, dtype=torch.float32)
        actual_p, actual_v = decay_spring_offsets(
            position, velocity, halflife_s=0.1, time_s=time
        )
        y = (4.0 * math.log(2.0) / (0.1 + 1.0e-5)) / 2.0
        j1 = velocity.numpy() + position.numpy() * y
        decay = math.exp(-y * 0.17)
        expected_p = decay * (position.numpy() + j1 * 0.17)
        expected_v = decay * (velocity.numpy() - j1 * y * 0.17)
        np.testing.assert_allclose(actual_p.numpy(), expected_p, atol=1e-6)
        np.testing.assert_allclose(actual_v.numpy(), expected_v, atol=1e-6)

    def test_reset_prepare_commit_and_dense_windows(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            reset = matcher.reset()
            self.assertEqual(reset.diagnostics.sequence, 0)
            self.assertEqual(reset.dense_joint_position_window.shape, (46, 29))
            self.assertEqual(
                reset.dense_feature_body_position_window.shape, (46, 3, 3)
            )
            self.assertEqual(
                reset.dense_feature_body_velocity_window.shape, (46, 3, 3)
            )
            np.testing.assert_array_equal(
                reset.joint_position_window.numpy(),
                reset.dense_joint_position_window[::5].numpy(),
            )
            prepared = matcher.prepare_step((0.5, 0.0), 0.0)
            self.assertEqual(reset.diagnostics.sequence, 0)
            result = matcher.commit(prepared)
            self.assertEqual(result.diagnostics.sequence, 1)
            with self.assertRaises(Exception):
                matcher.commit(prepared)
            for value in (
                result.dense_joint_position_window,
                result.dense_root_position_window,
                result.dense_root_orientation_window_wxyz,
                result.dense_feature_body_position_window,
                result.dense_feature_body_velocity_window,
            ):
                self.assertTrue(torch.isfinite(value).all())
            norms = torch.linalg.vector_norm(
                result.dense_root_orientation_window_wxyz, dim=-1
            )
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

            isolated = result.dense_feature_body_position_window.clone()
            isolated += 100.0
            next_result = matcher.step((0.5, 0.0), 0.0)
            self.assertFalse(
                torch.allclose(
                    isolated, next_result.dense_feature_body_position_window
                )
            )

    def test_extension_query_and_split_costs_are_used_by_matcher(self):
        arrays = build_varying_takara_arrays(frames=100)
        extension = _MatcherExtension()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root, device="cpu", extension=extension
            )
            matcher.reset()
            result = matcher.step((0.5, 0.0), 0.0)

        self.assertEqual(extension.query_calls, 1)
        self.assertGreaterEqual(result.diagnostics.motion_feature_cost, 0.0)
        self.assertGreaterEqual(result.diagnostics.extension_feature_cost, 0.0)
        self.assertAlmostEqual(
            result.diagnostics.motion_feature_cost
            + result.diagnostics.extension_feature_cost,
            result.diagnostics.selected_feature_cost,
            places=4,
        )

    def test_explicit_none_preserves_flat_matcher_for_100_commands(self):
        arrays = build_varying_takara_arrays(frames=160)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            omitted = TorchMotionMatcher.from_folder(root, device="cpu")
            explicit = TorchMotionMatcher.from_folder(
                root, device="cpu", extension=None
            )
            omitted.reset()
            explicit.reset()
            for step in range(100):
                velocity = (
                    0.4 * math.cos(0.03 * step),
                    0.3 * math.sin(0.03 * step),
                )
                heading = 0.01 * step
                left = omitted.step(velocity, heading)
                right = explicit.step(velocity, heading)
                self.assertEqual(
                    replace(
                        left.diagnostics, search_time_ns=None, step_time_ns=0
                    ),
                    replace(
                        right.diagnostics, search_time_ns=None, step_time_ns=0
                    ),
                )
                torch.testing.assert_close(
                    left.dense_joint_position_window,
                    right.dense_joint_position_window,
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    left.dense_root_position_window,
                    right.dense_root_position_window,
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    left.dense_feature_body_position_window,
                    right.dense_feature_body_position_window,
                    rtol=0.0,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
