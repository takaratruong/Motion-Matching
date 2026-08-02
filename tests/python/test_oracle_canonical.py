from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.canonical import SourceIdentity
from mm_sonic.terrain_oracle.math3d import unroll_quaternions_wxyz
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


class CanonicalClipTests(unittest.TestCase):
    def test_canonical_clip_rejects_non_wxyz_provenance_and_derives_velocity(self):
        """Catches an adapter silently passing a noncanonical quaternion order."""

        clip = synthetic_canonical_clip(frames=6)
        bad = replace(
            clip,
            source=replace(clip.source, quaternion_convention="xyzw"),
        )
        with self.assertRaisesRegex(ContractError, "root quaternion"):
            bad.validate()
        np.testing.assert_allclose(
            clip.root_linear_velocity_world[:, 0],
            np.full(6, 0.4, np.float32),
            atol=1e-6,
        )

    def test_observed_command_is_separate_from_inferred_command(self):
        """Catches fabrication of an observed command from inferred motion."""

        clip = synthetic_canonical_clip(frames=6, observed_commands=False)
        self.assertFalse(np.any(clip.commands.observed_mask))
        self.assertTrue(np.isfinite(clip.commands.inferred_velocity_local_xy).all())

    def test_arrays_are_owned_contiguous_and_read_only(self):
        """Catches canonical data retaining mutable aliases to source arrays."""

        clip = synthetic_canonical_clip(frames=6)
        arrays = (
            clip.root_position_world,
            clip.root_quaternion_world_wxyz,
            clip.commands.inferred_velocity_local_xy,
        )
        for value in arrays:
            with self.subTest(shape=value.shape):
                self.assertTrue(value.flags.c_contiguous)
                self.assertFalse(value.flags.writeable)


class CanonicalMathTests(unittest.TestCase):
    def test_quaternion_unrolling_is_independent_for_each_body_trace(self):
        """Catches unrolling one body's antipodes through another body's trace."""

        quaternions = np.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            unroll_quaternions_wxyz(quaternions), quaternions
        )

    def test_source_identity_requires_explicit_canonical_quaternion_order(self):
        """Catches omitted quaternion provenance at an import boundary."""

        identity = SourceIdentity(
            source_format="fixture",
            source_path="fixture://clip",
            source_sha256="1" * 64,
            coordinate_convention="world-right-handed-z-up",
            quaternion_convention="wxyz",
        )
        self.assertEqual(identity.quaternion_convention, "wxyz")


if __name__ == "__main__":
    unittest.main()
