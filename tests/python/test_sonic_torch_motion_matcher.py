import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mm_sonic.torch_motion_matcher import (
    ForcedMotionAlignment,
    TorchMotionMatcher,
    decay_spring_offsets,
)
from mm_sonic.torch_motion_data import MotionFolder
from tests.python.torch_motion_test_utils import (
    build_varying_takara_arrays,
    write_takara_arrays,
)


class TorchMotionMatcherTests(unittest.TestCase):
    def test_forced_alignment_preserves_full_terrain_transform(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            row = matcher.database.row_for_source(0, 10)
            successor = matcher.database.row_for_source(0, 11)
            self.assertIsNotNone(row)
            self.assertIsNotNone(successor)
            alignment = ForcedMotionAlignment(
                yaw_offset_rad=0.35,
                translation_world_xyz=(1.2, -0.7, 1.1),
            )
            reset = matcher.reset_to_row(int(row), alignment=alignment)

            source = arrays["body_pos_w"][10, 0]
            cosine, sine = math.cos(0.35), math.sin(0.35)
            expected = np.asarray(
                (
                    cosine * source[0] - sine * source[1] + 1.2,
                    sine * source[0] + cosine * source[1] - 0.7,
                    source[2] + 1.1,
                )
            )
            np.testing.assert_allclose(
                reset.root_position_world.numpy(), expected, atol=1e-6
            )

            prepared = matcher.prepare_step(
                (0.0, 0.0),
                0.0,
                forced_row=int(successor),
                forced_alignment=alignment,
            )
            stepped = matcher.commit(prepared)
            self.assertAlmostEqual(
                float(stepped.root_position_world[2]),
                float(arrays["body_pos_w"][11, 0, 2] + 1.1),
                places=5,
            )

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
            np.testing.assert_array_equal(
                reset.joint_position_window.numpy(),
                reset.dense_joint_position_window[::5].numpy(),
            )
            prepared = matcher.prepare_step((0.5, 0.0), 0.0)
            self.assertEqual(reset.diagnostics.sequence, 0)
            result = matcher.commit(prepared)
            self.assertEqual(result.diagnostics.sequence, 1)
            self.assertEqual(
                result.dense_body_position_window.shape, (46, 30, 3)
            )
            with self.assertRaises(Exception):
                matcher.commit(prepared)
            for value in (
                result.dense_joint_position_window,
                result.dense_root_position_window,
                result.dense_root_orientation_window_wxyz,
                result.dense_body_position_window,
            ):
                self.assertTrue(torch.isfinite(value).all())
            norms = torch.linalg.vector_norm(
                result.dense_root_orientation_window_wxyz, dim=-1
            )
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_ranked_preparations_offer_distinct_candidates_from_one_state(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()

            prepare_steps = getattr(matcher, "prepare_steps", None)
            self.assertTrue(callable(prepare_steps))
            candidates = tuple(
                prepare_steps(
                    (0.5, 0.0),
                    0.0,
                    maximum_candidates=3,
                )
            )

            self.assertEqual(len(candidates), 3)
            self.assertEqual(
                len({candidate.selected_row for candidate in candidates}),
                3,
            )
            committed = matcher.commit(candidates[1])
            self.assertEqual(committed.diagnostics.sequence, 1)
            with self.assertRaises(Exception):
                matcher.commit(candidates[0])

    def test_prepare_row_stages_one_explicit_source_phase_transactionally(self):
        arrays = build_varying_takara_arrays(frames=100)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(root, device="cpu")
            matcher.reset()
            target_row = matcher.database.row_for_source(0, 20)
            self.assertIsNotNone(target_row)

            prepared = matcher.prepare_row(
                int(target_row),
                (0.2, 0.0),
                0.0,
            )

            self.assertEqual(prepared.selected_row, target_row)
            result = matcher.commit(prepared)
            self.assertEqual(result.diagnostics.selected_frame, 20)
            self.assertTrue(result.diagnostics.transitioned)

    def test_supported_reset_selects_and_resets_to_matching_source_phase(self):
        arrays = build_varying_takara_arrays(frames=100)
        target_frame = 20
        root = arrays["body_pos_w"][target_frame, 0]
        feet = arrays["body_pos_w"][target_frame, [18, 19]]
        root_wxyz = arrays["body_quat_w"][target_frame, 0]
        w, x, y, z = (float(value) for value in root_wxyz)
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        c, s = math.cos(yaw), math.sin(yaw)
        relative = feet - root
        feet_root_local = np.stack(
            (
                c * relative[:, 0] + s * relative[:, 1],
                -s * relative[:, 0] + c * relative[:, 1],
                relative[:, 2],
            ),
            axis=1,
        )
        root_velocity = arrays["body_lin_vel_w"][target_frame, 0]
        root_velocity_local_xy = np.asarray(
            (
                c * root_velocity[0] + s * root_velocity[1],
                -s * root_velocity[0] + c * root_velocity[1],
            ),
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root_path = Path(tmp)
            write_takara_arrays(root_path / "walk", arrays)
            matcher = TorchMotionMatcher.from_folder(
                root_path, device="cpu"
            )

            row = matcher.select_supported_reset_row(
                joint_position=arrays["joint_pos"][target_frame],
                feet_position_root_local=feet_root_local,
                support_contact=np.asarray((True, True)),
                root_velocity_local_xy=root_velocity_local_xy,
            )
            lead = matcher.source_lead_row(row, lead_frames=15)
            lead_rows = matcher.source_lead_rows(row, lead_frames=15)
            result = matcher.reset_to_row(row)

            self.assertEqual(
                matcher.database.row_for_source(0, target_frame), row
            )
            self.assertEqual(
                matcher.database.row_for_source(0, target_frame - 15),
                lead,
            )
            self.assertEqual(lead_rows[0], lead)
            self.assertEqual(lead_rows[-1], row)
            self.assertEqual(len(lead_rows), 16)
            self.assertEqual(
                result.diagnostics.selected_frame, target_frame
            )


if __name__ == "__main__":
    unittest.main()
