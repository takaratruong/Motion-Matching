import unittest
from dataclasses import replace
import numpy as np

from resources.g1_terrain_builder.schema import (
    ArtifactSet, HoldenClip, SkeletonSpec, SourceClip, SourceFrameRange,
    TERRAIN_FAMILIES, TerrainBank, TerrainBankIndex,
)


class SchemaTests(unittest.TestCase):
    def _terrain_bank_index(self):
        ranges = tuple(
            SourceFrameRange(
                source_name=name,
                source_start=0,
                source_stop=2,
                source_frame_count=2,
                global_start=2 * index,
                global_stop=2 * index + 2,
            )
            for index, name in enumerate(("flat-a", "curb-a", "slope-a", "stair-a"))
        )
        banks = tuple(
            TerrainBank(family=family, range_indices=(index,))
            for index, family in enumerate(TERRAIN_FAMILIES)
        )
        return TerrainBankIndex(frame_count=8, ranges=ranges, banks=banks)

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

    def test_terrain_bank_schema_has_exact_order_and_references_ranges(self):
        self.assertEqual(TERRAIN_FAMILIES, ("flat", "curb", "slope", "stair"))
        index = self._terrain_bank_index()

        index.validate()

        self.assertEqual(
            index.ranges_for("flat"),
            (index.ranges[0],),
        )
        self.assertEqual(
            index.ranges_for("stair"),
            (index.ranges[3],),
        )
        self.assertEqual(
            (
                index.ranges[2].source_start,
                index.ranges[2].source_stop,
                index.ranges[2].global_start,
                index.ranges[2].global_stop,
            ),
            (0, 2, 4, 6),
        )

    def test_terrain_bank_schema_rejects_unknown_duplicate_or_missing_ownership(self):
        valid = self._terrain_bank_index()
        cases = (
            (
                "unknown-family",
                replace(
                    valid,
                    banks=valid.banks[:-1] + (TerrainBank("mud", (3,)),),
                ),
                "terrain families",
            ),
            (
                "wrong-family-order",
                replace(
                    valid,
                    banks=(valid.banks[1], valid.banks[0], *valid.banks[2:]),
                ),
                "terrain families",
            ),
            (
                "duplicate-range-owner",
                replace(
                    valid,
                    banks=(
                        TerrainBank("flat", (0, 1)),
                        TerrainBank("curb", (1,)),
                        *valid.banks[2:],
                    ),
                ),
                "duplicate.*ownership",
            ),
            (
                "missing-range-owner",
                replace(
                    valid,
                    banks=(valid.banks[0], TerrainBank("curb", ()), *valid.banks[2:]),
                ),
                "incomplete.*ownership",
            ),
            (
                "out-of-bound-reference",
                replace(
                    valid,
                    banks=valid.banks[:-1] + (TerrainBank("stair", (4,)),),
                ),
                "range index",
            ),
            (
                "duplicate-source",
                replace(
                    valid,
                    ranges=(
                        valid.ranges[0],
                        replace(valid.ranges[1], source_name="flat-a"),
                        *valid.ranges[2:],
                    ),
                ),
                "duplicate source ownership",
            ),
        )
        for name, candidate, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    candidate.validate()

    def test_terrain_bank_schema_rejects_gaps_overlaps_and_incomplete_coverage(self):
        valid = self._terrain_bank_index()
        cases = (
            (
                "gap",
                replace(
                    valid,
                    frame_count=9,
                    ranges=(
                        valid.ranges[0],
                        replace(valid.ranges[1], global_start=3, global_stop=5),
                        replace(valid.ranges[2], global_start=5, global_stop=7),
                        replace(valid.ranges[3], global_start=7, global_stop=9),
                    ),
                ),
                "gap",
            ),
            (
                "overlap",
                replace(
                    valid,
                    ranges=(
                        valid.ranges[0],
                        replace(valid.ranges[1], global_start=1, global_stop=3),
                        *valid.ranges[2:],
                    ),
                ),
                "overlap",
            ),
            (
                "incomplete-tail",
                replace(valid, frame_count=9),
                "incomplete global-frame coverage",
            ),
        )
        for name, candidate, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    candidate.validate()

    def test_terrain_bank_schema_rejects_invalid_or_out_of_bound_ranges(self):
        valid = self._terrain_bank_index()
        cases = (
            (
                "empty-source",
                replace(valid.ranges[1], source_stop=0),
                "empty or reversed source range",
            ),
            (
                "reversed-source",
                replace(valid.ranges[1], source_start=2, source_stop=1),
                "empty or reversed source range",
            ),
            (
                "source-before-zero",
                replace(valid.ranges[1], source_start=-1, source_stop=1),
                "source range out of bounds",
            ),
            (
                "source-after-end",
                replace(valid.ranges[1], source_stop=3),
                "source range out of bounds",
            ),
            (
                "empty-global",
                replace(valid.ranges[1], global_stop=2),
                "empty or reversed global range",
            ),
            (
                "reversed-global",
                replace(valid.ranges[1], global_start=3, global_stop=2),
                "empty or reversed global range",
            ),
            (
                "global-before-zero",
                replace(valid.ranges[0], global_start=-1, global_stop=1),
                "global range out of bounds",
            ),
            (
                "global-after-end",
                replace(valid.ranges[3], global_stop=9),
                "global range out of bounds",
            ),
            (
                "different-source-global-lengths",
                replace(valid.ranges[1], source_stop=1),
                "range lengths differ",
            ),
        )
        for name, changed_range, message in cases:
            ranges = list(valid.ranges)
            changed_index = {
                "global-before-zero": 0,
                "global-after-end": 3,
            }.get(name, 1)
            ranges[changed_index] = changed_range
            candidate = replace(valid, ranges=tuple(ranges))
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    candidate.validate()


if __name__ == "__main__":
    unittest.main()
