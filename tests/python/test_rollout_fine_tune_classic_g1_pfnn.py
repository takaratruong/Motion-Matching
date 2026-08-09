from __future__ import annotations

import unittest

from mm_sonic.build_g1_pfnn_vertical_dataset import build_vertical_dataset
from mm_sonic.rollout_fine_tune_classic_g1_pfnn import vertical_rollout_sequences
from mm_sonic.train_classic_g1_pfnn import _VerticalTrainingView
from tests.python.test_build_g1_pfnn_vertical_dataset import _source


class RolloutFineTuneClassicG1PFNNTest(unittest.TestCase):
    def test_sequences_are_same_clip_lane_consecutive_and_predecessor_qualified(self) -> None:
        dataset = build_vertical_dataset(
            (_source("train", "released_train"), _source("validation", "released_val"))
        )
        view = _VerticalTrainingView(dataset, "train")

        sequences = vertical_rollout_sequences(view, 3)

        self.assertGreater(len(sequences), 0)
        for sequence in sequences:
            rows = [view[index] for index in sequence.indices]
            predecessor = view[sequence.predecessor_index]
            self.assertEqual(predecessor["clip_id"], rows[0]["clip_id"])
            self.assertEqual(predecessor["sequence_lane"], rows[0]["sequence_lane"])
            self.assertEqual(predecessor["center_frame"] + 1, rows[0]["center_frame"])
            self.assertEqual(
                [row["center_frame"] for row in rows],
                list(range(rows[0]["center_frame"], rows[0]["center_frame"] + 3)),
            )


if __name__ == "__main__":
    unittest.main()
