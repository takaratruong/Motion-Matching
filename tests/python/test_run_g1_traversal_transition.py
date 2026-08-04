import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_traversal_transition",
    ROOT / "resources" / "run_g1_traversal_transition.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TraversalTransitionTests(unittest.TestCase):
    def test_slerp_uses_the_short_quaternion_arc(self):
        output = MODULE._quaternion_slerp(
            np.array((1.0, 0.0, 0.0, 0.0)),
            np.array((-1.0, 0.0, 0.0, 0.0)),
            np.array((0.0, 0.5, 1.0)),
        )

        np.testing.assert_allclose(
            output,
            np.array(
                (
                    (1.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                )
            ),
            atol=1e-12,
        )

    def test_junction_requires_a_shared_supported_foot(self):
        incoming = {
            "joint_position": np.zeros((2, 29)),
            "root_position_world": np.zeros((2, 3)),
            "source_support_mask": np.array(
                ((True, False), (False, True))
            ),
        }
        outgoing = {
            "joint_position": np.zeros((2, 29)),
            "root_position_world": np.zeros((2, 3)),
            "source_support_mask": np.array(
                ((False, True), (True, False))
            ),
        }
        incoming_feet = np.zeros((2, 2, 3))
        outgoing_feet = np.zeros((2, 2, 3))
        incoming_feet[1, 1, :2] = (0.2, 0.1)
        outgoing_feet[0, 1, :2] = (0.1, 0.1)
        outgoing_feet[1, 0, :2] = (1.0, 1.0)

        incoming_frame, outgoing_frame, foot, shift = (
            MODULE._select_junction(
                incoming=incoming,
                outgoing=outgoing,
                incoming_feet=incoming_feet,
                outgoing_feet=outgoing_feet,
                yaw_scene_from_matcher=0.0,
                tail_frames=2,
            )
        )

        self.assertEqual((incoming_frame, outgoing_frame, foot), (1, 0, 1))
        np.testing.assert_allclose(shift, (0.1, 0.0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
