import dataclasses
import unittest

import numpy as np

from resources.build_g1_episode_motion_pack import (
    EpisodeMotion,
    mirror_episode_motion,
    validate_episode_motion,
)
from resources.g1_reach_builder.mirror import MIRROR_BONES


def synthetic_episode_motion() -> EpisodeMotion:
    frames = 4
    positions = np.zeros((frames, 31, 3), np.float32)
    positions[:, 0, 0] = np.linspace(0.0, 0.3, frames)
    positions[:, 0, 2] = np.linspace(0.1, 0.4, frames)
    for bone in range(1, 31):
        positions[:, bone, 0] = 0.01 * bone
        positions[:, bone, 1] = 0.02 * bone
        positions[:, bone, 2] = -0.005 * bone
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    return EpisodeMotion(
        positions=positions,
        velocities=np.zeros_like(positions),
        rotations=rotations,
        angular_velocities=np.zeros_like(positions),
        foot_contacts=np.array(
            [[1, 0], [1, 1], [0, 1], [1, 0]], np.uint8
        ),
        source_frames=np.arange(frames, dtype=np.int32),
    )


class EpisodeMotionPackTests(unittest.TestCase):
    def test_mirror_swaps_limbs_contacts_and_reflects_root_z(self):
        source = synthetic_episode_motion()

        mirrored = mirror_episode_motion(source)

        self.assertEqual(mirrored.positions.shape, source.positions.shape)
        np.testing.assert_allclose(
            mirrored.positions[:, 0, 2],
            -source.positions[:, 0, 2],
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(
            mirrored.foot_contacts,
            source.foot_contacts[:, ::-1],
        )
        double_mirrored = mirror_episode_motion(mirrored)
        np.testing.assert_allclose(
            double_mirrored.positions,
            source.positions,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            double_mirrored.rotations,
            source.rotations,
            atol=2.0e-5,
        )
        self.assertEqual(len(MIRROR_BONES), 31)

    def test_validate_rejects_wrong_skeleton_width(self):
        source = synthetic_episode_motion()
        malformed = dataclasses.replace(
            source,
            positions=source.positions[:, :-1],
        )

        with self.assertRaisesRegex(ValueError, "31 bones"):
            validate_episode_motion(malformed)

    def test_validate_rejects_nonbinary_contacts(self):
        source = synthetic_episode_motion()
        contacts = source.foot_contacts.copy()
        contacts[0, 0] = 2
        malformed = dataclasses.replace(source, foot_contacts=contacts)

        with self.assertRaisesRegex(ValueError, "foot contacts"):
            validate_episode_motion(malformed)


if __name__ == "__main__":
    unittest.main()
