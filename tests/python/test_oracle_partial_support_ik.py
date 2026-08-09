from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES
from mm_sonic.terrain_oracle.reference_stitch import _G1FootfallAdapter


_G1_MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") is not None and _G1_MODEL.is_file(),
    "G1 model and MuJoCo are required",
)
class PartialSupportIKIntegrationTest(unittest.TestCase):
    def test_mask_does_not_pull_unsupported_sole_corner(self) -> None:
        adapter = _G1FootfallAdapter(
            _G1_MODEL,
            ISAACLAB_JOINT_NAMES,
            maximum_joint_correction_rad=0.35,
        )
        root = np.asarray((0.0, 0.0, 0.8), dtype=np.float64)
        quaternion = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        joints = np.zeros(len(ISAACLAB_JOINT_NAMES), dtype=np.float64)
        targets = [
            value.copy()
            for value in adapter.sole_positions_for_pose(
                root_position=root,
                root_quaternion_wxyz=quaternion,
                joints=joints,
            )
        ]
        # A terrain trough can leave one rigid corner well above the surface.
        # It remains collision geometry, but must not drag the foot toward an
        # impossible four-point target.
        targets[0][-1, 2] -= 0.20
        targets[1][-1, 2] -= 0.20
        mask = np.asarray((True, True, True, False), dtype=bool)

        adapted, correction, error = adapter.adapt_to_targets(
            root_position=root,
            root_quaternion_wxyz=quaternion,
            authored_joints=joints,
            sole_targets_world=targets,
            sole_target_masks=(mask, mask),
        )

        np.testing.assert_allclose(adapted, joints, atol=1.0e-8)
        self.assertLess(correction, 1.0e-8)
        self.assertLess(error, 1.0e-8)


if __name__ == "__main__":
    unittest.main()
