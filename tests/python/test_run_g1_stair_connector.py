import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_stair_connector",
    ROOT / "resources" / "run_g1_stair_connector.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunG1StairConnectorTest(unittest.TestCase):
    def test_boundary_uses_incoming_backward_and_outgoing_forward_derivative(self):
        frames = 5
        arrays = {
            "joint_position": np.repeat(
                np.arange(frames, dtype=np.float64)[:, None],
                29,
                axis=1,
            ),
            "root_position_world": np.stack(
                (
                    np.arange(frames, dtype=np.float64),
                    np.zeros(frames),
                    np.ones(frames),
                ),
                axis=1,
            ),
            "qpos": np.tile(
                np.asarray((0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)),
                (frames, 1),
            ),
            "foot_position_world": np.zeros((frames, 2, 3)),
        }

        incoming = MODULE._boundary_from_artifact(
            arrays,
            frame=2,
            side="incoming",
            dt_s=0.5,
        )
        outgoing = MODULE._boundary_from_artifact(
            arrays,
            frame=2,
            side="outgoing",
            dt_s=0.5,
        )

        np.testing.assert_allclose(incoming.joint_velocity, 2.0)
        np.testing.assert_allclose(outgoing.joint_velocity, 2.0)
        np.testing.assert_allclose(
            incoming.root_velocity_world,
            (2.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(
            outgoing.root_velocity_world,
            (2.0, 0.0, 0.0),
        )

    def test_minimum_sole_clearance_reduces_all_feet_and_samples(self):
        sole = np.zeros((2, 2, 3, 3), dtype=np.float64)
        sole[..., 2] = 0.10
        surface = np.zeros((2, 2, 3), dtype=np.float64)
        surface[0, 1, 2] = 0.20
        surface[1, 0, 0] = 0.08

        result = MODULE._minimum_sole_clearance_by_frame(sole, surface)

        np.testing.assert_allclose(result, (-0.10, 0.02))


if __name__ == "__main__":
    unittest.main()
