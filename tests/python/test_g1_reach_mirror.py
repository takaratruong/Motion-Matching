import unittest

import numpy as np

from resources import quat as holden_quat
from resources.g1_reach_builder.mirror import MIRROR_BONES, mirror_reach
from resources.g1_reach_builder.motions import (
    ReachAugmentation,
    ReachHand,
    build_captured_reach,
)
from tests.python.test_g1_reach_motions import (
    accepted_annotation,
    motion_corpus,
)


class ReachMirrorTests(unittest.TestCase):
    def test_mirrors_complete_pose_endpoint_approach_and_provenance(self):
        left = build_captured_reach(
            motion_corpus(), accepted_annotation(departure=20, grab=80)
        )

        right = mirror_reach(left)

        self.assertEqual(len(MIRROR_BONES), 31)
        self.assertEqual(right.active_hand, ReachHand.RIGHT)
        self.assertEqual(right.augmentation, ReachAugmentation.MIRRORED)
        self.assertEqual(right.original_reach_id, left.reach_id)
        self.assertEqual(right.source_frames.tolist(), left.source_frames.tolist())
        self.assertEqual(right.contact_index, left.contact_index)
        self.assertEqual(right.return_available, left.return_available)
        np.testing.assert_allclose(
            right.endpoint_position_root,
            np.asarray(left.endpoint_position_root) * [1, 1, -1],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            right.approach_direction_root,
            np.asarray(left.approach_direction_root) * [1, 1, -1],
            atol=1e-5,
        )
        matrices = holden_quat.to_xform(right.rotations)
        np.testing.assert_allclose(
            np.linalg.det(matrices), 1.0, atol=1e-4
        )
        np.testing.assert_array_equal(
            right.foot_contacts[:, ::-1], left.foot_contacts
        )

    def test_double_mirror_reconstructs_original_pose(self):
        left = build_captured_reach(
            motion_corpus(), accepted_annotation(departure=20, grab=80)
        )
        right = mirror_reach(left)

        reconstructed = mirror_reach(right, allow_mirrored=True)

        np.testing.assert_allclose(
            reconstructed.positions, left.positions, atol=1e-5
        )
        dots = np.abs(np.sum(
            reconstructed.rotations * left.rotations, axis=-1
        ))
        np.testing.assert_allclose(dots, 1.0, atol=1e-4)
        np.testing.assert_allclose(
            reconstructed.endpoint_position_root,
            left.endpoint_position_root,
            atol=1e-5,
        )

    def test_normal_build_refuses_to_mirror_a_mirror(self):
        left = build_captured_reach(
            motion_corpus(), accepted_annotation(departure=20, grab=80)
        )
        with self.assertRaisesRegex(ValueError, "already mirrored"):
            mirror_reach(mirror_reach(left))


if __name__ == "__main__":
    unittest.main()
