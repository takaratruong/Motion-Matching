import unittest

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
            ValueError, "need at least 3 unique objects, got 2"
        ):
            split_objects(
                labeled_clips_for_objects(["a", "b"]),
                heldout_count=2,
            )

    def test_split_rejects_nonpositive_and_all_heldout_counts(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        for heldout_count in (0, -1, 3, 4):
            with self.subTest(heldout_count=heldout_count):
                with self.assertRaises(ValueError):
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
            "object identity leak between database and heldout split",
        ):
            validate_split(clips, leaking)

    def test_validate_split_rejects_missing_or_unknown_objects(self):
        clips = labeled_clips_for_objects(["a", "b", "c"])
        cases = (
            EvaluationSplit(7, ("a",), ("b",)),
            EvaluationSplit(7, ("a", "unknown"), ("b", "c")),
        )

        for split in cases:
            with self.subTest(split=split):
                with self.assertRaisesRegex(
                    ValueError,
                    "split does not cover every source object exactly once",
                ):
                    validate_split(clips, split)

    def test_validate_split_accepts_duplicate_clips_for_one_object(self):
        clips = labeled_clips_for_objects(["a", "a", "b", "c"])
        split = EvaluationSplit(7, ("a", "c"), ("b",))

        self.assertIsNone(validate_split(clips, split))


if __name__ == "__main__":
    unittest.main()
