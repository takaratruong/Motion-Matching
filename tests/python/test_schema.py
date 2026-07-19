import unittest
import numpy as np

from resources.g1_terrain_builder.schema import (
    ArtifactSet, HoldenClip, SkeletonSpec, SourceClip,
)


class SchemaTests(unittest.TestCase):
    def test_source_clip_rejects_bad_qpos_width(self):
        clip = SourceClip(
            name="bad", fps=25.0, qpos=np.zeros((2, 35), np.float32),
            source_frames=np.arange(2), terrain_id="flat",
        )
        with self.assertRaisesRegex(ValueError, "qpos shape"):
            clip.validate()

    def test_skeleton_signature_changes_with_parent_order(self):
        a = SkeletonSpec(("Simulation", "Hips"), np.array([-1, 0], np.int32))
        b = SkeletonSpec(("Simulation", "Hips"), np.array([-1, -1], np.int32))
        self.assertNotEqual(a.signature(), b.signature())

    def test_holden_clip_empty_factory_and_validation_require_twelve_columns(self):
        clip = HoldenClip.empty(frames=3, bones=2)
        self.assertEqual(clip.terrain_features.shape, (3, 12))
        self.assertEqual(clip.terrain_features.dtype, np.dtype(np.float32))
        clip.terrain_features = np.zeros((3, 11), np.float32)
        with self.assertRaisesRegex(ValueError, "terrain feature shape"):
            clip.validate()

    def test_holden_clip_requires_three_finite_support_columns(self):
        clip = HoldenClip.empty(frames=3, bones=2)
        clip.terrain_support = np.zeros((3, 2), np.float32)
        with self.assertRaisesRegex(ValueError, "support shape"):
            clip.validate()
        clip.terrain_support = np.full((3, 3), np.nan, np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            clip.validate()

    def test_artifact_set_requires_three_finite_support_columns(self):
        artifacts = ArtifactSet.empty(frames=3, bones=2)
        artifacts.terrain_support = np.zeros((3, 2), np.float32)
        with self.assertRaisesRegex(ValueError, "support shape"):
            artifacts.validate()
        artifacts.terrain_support = np.full((3, 3), np.nan, np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            artifacts.validate()

    def test_artifact_set_empty_factory_and_validation_require_twelve_columns(self):
        artifacts = ArtifactSet.empty(frames=3, bones=2)
        self.assertEqual(artifacts.terrain_features.shape, (3, 12))
        self.assertEqual(artifacts.terrain_features.dtype, np.dtype(np.float32))
        artifacts.terrain_features = np.zeros((3, 11), np.float32)
        with self.assertRaisesRegex(ValueError, "terrain feature shape"):
            artifacts.validate()

    def test_artifact_set_rejects_range_overlap(self):
        artifacts = ArtifactSet.empty(frames=4, bones=2)
        artifacts.range_starts = np.array([0, 2], np.int32)
        artifacts.range_stops = np.array([3, 4], np.int32)
        with self.assertRaisesRegex(ValueError, "ranges overlap"):
            artifacts.validate()


if __name__ == "__main__":
    unittest.main()
