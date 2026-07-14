import unittest

from resources.g1_interaction_builder import splits as split_module
from resources.g1_interaction_builder.schema import EvaluationSplit
from resources.g1_interaction_builder.splits import (
    split_objects,
    validate_split,
)
from tests.python.interaction_fixture import labeled_clips_for_objects


class InteractionSplitTests(unittest.TestCase):
    def test_split_is_deterministic_order_stable_and_object_disjoint(self):
        clips = labeled_clips_for_objects(["a", "a", "b", "c", "d"])

        first = split_objects(clips, heldout_count=2, seed=7)
        second = split_objects(clips, heldout_count=2, seed=7)
        reordered = split_objects(
            list(reversed(clips)), heldout_count=2, seed=7
        )

        self.assertEqual(first, second)
        self.assertEqual(first, reordered)
        self.assertEqual(first.seed, 7)
        self.assertEqual(first.database_objects, ("b", "d"))
        self.assertEqual(first.heldout_objects, ("a", "c"))
        self.assertFalse(
            set(first.database_objects) & set(first.heldout_objects)
        )

    def test_seed_changes_the_exact_sorted_heldout_selection(self):
        clips = labeled_clips_for_objects(["d", "c", "b", "a"])

        split = split_objects(clips, heldout_count=2, seed=11)

        self.assertEqual(split.database_objects, ("a", "c"))
        self.assertEqual(split.heldout_objects, ("b", "d"))

    def test_split_rejects_too_few_objects(self):
        with self.assertRaisesRegex(
            ValueError,
            "heldout_count 2 must leave at least one database object; "
            "got 2 unique objects",
        ):
            split_objects(
                labeled_clips_for_objects(["a", "b"]),
                heldout_count=2,
            )

    def test_split_rejects_nonpositive_and_all_heldout_counts(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        cases = (
            (0, "heldout_count must be positive, got 0"),
            (-1, "heldout_count must be positive, got -1"),
            (
                3,
                "heldout_count 3 must leave at least one database "
                "object; got 3 unique objects",
            ),
            (
                4,
                "heldout_count 4 must leave at least one database "
                "object; got 3 unique objects",
            ),
        )
        for heldout_count, message in cases:
            with self.subTest(heldout_count=heldout_count):
                with self.assertRaisesRegex(ValueError, message):
                    split_objects(clips, heldout_count=heldout_count)

    def test_validate_split_rejects_object_identity_leak(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        leaking = EvaluationSplit(
            seed=7,
            database_objects=("a", "b"),
            heldout_objects=("b", "c"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "overlapping object IDs in database and heldout partitions.*b",
        ):
            validate_split(clips, leaking)

    def test_validate_split_rejects_missing_or_unknown_objects(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        cases = (
            (
                EvaluationSplit(7, ("a",), ("b",)),
                "missing source object IDs from split.*c",
            ),
            (
                EvaluationSplit(7, ("a", "unknown"), ("b", "c")),
                "unknown split object IDs not present in clips.*unknown",
            ),
        )

        for split, message in cases:
            with self.subTest(split=split):
                with self.assertRaisesRegex(
                    ValueError, message
                ):
                    validate_split(clips, split)

    def test_validate_split_rejects_duplicate_ids_before_set_comparison(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        cases = (
            (
                EvaluationSplit(7, ("a", "a", "b"), ("c",)),
                "duplicate database object IDs.*a",
            ),
            (
                EvaluationSplit(7, ("a",), ("b", "b", "c")),
                "duplicate heldout object IDs.*b",
            ),
        )

        for split, message in cases:
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, message):
                    validate_split(clips, split)

    def test_validate_split_rejects_unsorted_serialized_tuples(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        cases = (
            (
                EvaluationSplit(7, ("b", "a"), ("c",)),
                "database object IDs must be sorted.*b.*a",
            ),
            (
                EvaluationSplit(7, ("a",), ("c", "b")),
                "heldout object IDs must be sorted.*c.*b",
            ),
        )

        for split, message in cases:
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, message):
                    validate_split(clips, split)

    def test_validate_split_rejects_empty_partitions(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        cases = (
            (
                EvaluationSplit(7, (), ("a", "b", "c")),
                "database partition is empty",
            ),
            (
                EvaluationSplit(7, ("a", "b", "c"), ()),
                "heldout partition is empty",
            ),
        )

        for split, message in cases:
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, message):
                    validate_split(clips, split)

    def test_partition_clips_preserves_input_order_and_exact_coverage(self):
        clips = labeled_clips_for_objects(["held", "db", "db", "held"])
        split = EvaluationSplit(7, ("db",), ("held",))

        database, heldout = split_module.partition_clips(clips, split)

        self.assertEqual(
            [clip.motion.sequence_id for clip in database],
            [clips[1].motion.sequence_id, clips[2].motion.sequence_id],
        )
        self.assertEqual(
            [clip.motion.sequence_id for clip in heldout],
            [clips[0].motion.sequence_id, clips[3].motion.sequence_id],
        )
        self.assertEqual(len(database) + len(heldout), len(clips))

    def test_validate_split_accepts_duplicate_clips_for_one_object(self):
        clips = labeled_clips_for_objects(["a", "a", "b", "c"])
        split = EvaluationSplit(7, ("a", "c"), ("b",))

        self.assertIsNone(validate_split(clips, split))


if __name__ == "__main__":
    unittest.main()
