import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_path_phase_composite import compose_path_phases


class PathPhaseCompositeTests(unittest.TestCase):
    @staticmethod
    def connector(start, stop, frames, joint_offset=0.0):
        progress = np.linspace(start, stop, frames)
        joints = np.zeros((frames, 29), dtype=np.float64)
        joints[:, 0] = 0.2 * progress + joint_offset
        joints[:, 1] = 0.1 * np.sin(progress)
        return {
            "joint_position": joints,
            "root_position_world": np.column_stack(
                (
                    progress,
                    np.zeros(frames),
                    np.full(frames, 0.8),
                )
            ),
            "root_orientation_world_wxyz": np.tile(
                (1.0, 0.0, 0.0, 0.0), (frames, 1)
            ),
        }

    def test_composes_overlapping_phases_with_exact_continuous_boundaries(self):
        mount = self.connector(0.0, 1.0, 30)
        interior = self.connector(0.65, 1.35, 30)
        dismount = self.connector(1.45, 2.4, 35, joint_offset=0.02)
        supports = (
            np.ones((len(mount["joint_position"]), 2), dtype=np.bool_),
            np.ones((len(interior["joint_position"]), 2), dtype=np.bool_),
            np.ones((len(dismount["joint_position"]), 2), dtype=np.bool_),
        )

        arrays, support, metrics = compose_path_phases(
            mount=mount,
            interior=interior,
            dismount=dismount,
            mount_support=supports[0],
            interior_support=supports[1],
            dismount_support=supports[2],
            blend_frames=10,
        )

        self.assertEqual(arrays["joint_position"].shape[1], 29)
        self.assertTrue(support.any(axis=1).all())
        self.assertLessEqual(metrics["mount_interior_root_gap_m"], 0.12)
        self.assertLess(
            metrics["maximum_joint_step_rad"],
            0.08,
        )
        self.assertLess(
            metrics["maximum_root_step_m"],
            0.08,
        )
        np.testing.assert_allclose(
            np.linalg.norm(
                arrays["root_orientation_world_wxyz"], axis=1
            ),
            1.0,
            atol=1.0e-12,
        )
        for boundary in metrics["splice_output_frames"]:
            np.testing.assert_allclose(
                arrays["joint_position"][boundary],
                arrays["joint_position"][boundary - 1],
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                arrays["root_position_world"][boundary],
                arrays["root_position_world"][boundary - 1],
                atol=1.0e-12,
            )

    def test_rejects_when_mount_and_shifted_interior_do_not_overlap(self):
        mount = self.connector(0.0, 0.2, 15)
        interior = self.connector(1.0, 1.3, 20)
        dismount = self.connector(1.4, 2.0, 20)
        support = np.ones((20, 2), dtype=np.bool_)

        with self.assertRaisesRegex(ContractError, "overlap"):
            compose_path_phases(
                mount=mount,
                interior=interior,
                dismount=dismount,
                mount_support=support[:15],
                interior_support=support,
                dismount_support=support,
                blend_frames=10,
            )

    def test_preserves_unsupported_source_frames_for_later_certification(self):
        mount = self.connector(0.0, 1.0, 30)
        interior = self.connector(0.65, 1.35, 30)
        dismount = self.connector(1.45, 2.4, 35)
        mount_support = np.ones((30, 2), dtype=np.bool_)
        mount_support[4] = False

        _, support, _ = compose_path_phases(
            mount=mount,
            interior=interior,
            dismount=dismount,
            mount_support=mount_support,
            interior_support=np.ones((30, 2), dtype=np.bool_),
            dismount_support=np.ones((35, 2), dtype=np.bool_),
            blend_frames=10,
        )

        self.assertFalse(support[4].any())

    def test_uses_existing_second_overlap_without_translating_interior(self):
        mount = self.connector(0.0, 1.0, 40)
        interior = self.connector(0.65, 1.8, 50)
        dismount = self.connector(1.3, 2.4, 45)

        _, _, metrics = compose_path_phases(
            mount=mount,
            interior=interior,
            dismount=dismount,
            mount_support=np.ones((40, 2), dtype=np.bool_),
            interior_support=np.ones((50, 2), dtype=np.bool_),
            dismount_support=np.ones((45, 2), dtype=np.bool_),
            blend_frames=10,
        )

        np.testing.assert_allclose(
            metrics["interior_translation_matcher_xyz"],
            (0.0, 0.0, 0.0),
            atol=0.0,
        )
        self.assertLessEqual(
            metrics["interior_dismount_root_gap_m"],
            0.12,
        )
        self.assertLess(
            metrics["interior_source_start_frame"],
            metrics["interior_source_stop_frame"],
        )


if __name__ == "__main__":
    unittest.main()
