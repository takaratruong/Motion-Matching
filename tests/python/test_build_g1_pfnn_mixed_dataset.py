from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.build_g1_pfnn_mixed_dataset import combine_vertical_and_grail
from mm_sonic.build_g1_pfnn_vertical_dataset import build_vertical_dataset
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from tests.python.test_build_g1_pfnn_vertical_dataset import _source


class _GrailRows:
    split = "train"
    x_mean = np.zeros(INPUT_LAYOUT.size, dtype=np.float32)
    x_std = np.ones(INPUT_LAYOUT.size, dtype=np.float32)
    y_mean = np.zeros(OUTPUT_LAYOUT.size, dtype=np.float32)
    y_std = np.ones(OUTPUT_LAYOUT.size, dtype=np.float32)

    def __init__(self) -> None:
        self.rows = []
        for family in range(2):
            for variant in range(2):
                clip = f"terrain_slopes__slope_{family:03d}__{variant:03d}"
                for center in range(2):
                    x = np.full(INPUT_LAYOUT.size, family + 0.1 * variant)
                    y = np.full(OUTPUT_LAYOUT.size, 0.2 * center)
                    y[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 0.0, 1.0)
                    self.rows.append(
                        {
                            "x": x.astype(np.float32),
                            "y": y.astype(np.float32),
                            "phase": np.float32(0.2 * center),
                            "clip_id": clip,
                            "split": "train",
                            "sequence_lane": "motion",
                            "terrain_class": "ascent",
                            "center_frame": center,
                            "terrain_sha256": ("a" if family == 0 else "b") * 64,
                        }
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class BuildG1PFNNMixedDatasetTest(unittest.TestCase):
    def test_combines_all_grail_families_with_variant_holdouts_and_mirrors(self) -> None:
        vertical = build_vertical_dataset(
            (_source("train", "released_train"), _source("validation", "released_val"))
        )

        mixed = combine_vertical_and_grail(
            vertical,
            _GrailRows(),
            grail_dataset_sha256="9" * 64,
        )

        self.assertEqual(
            len(mixed.splits["train"].phase),
            len(vertical.splits["train"].phase) + 8,
        )
        self.assertEqual(
            len(mixed.splits["validation"].phase),
            len(vertical.splits["validation"].phase) + 8,
        )
        train_clips = set(mixed.splits["train"].clip_id.tolist())
        validation_clips = set(mixed.splits["validation"].clip_id.tolist())
        self.assertIn("terrain_slopes__slope_000__000", train_clips)
        self.assertIn("terrain_slopes__slope_000__000__mirror", train_clips)
        self.assertIn("terrain_slopes__slope_000__001", validation_clips)
        self.assertFalse(train_clips & validation_clips)
        self.assertEqual(mixed.selection_sha256, vertical.selection_sha256)
        self.assertNotEqual(mixed.dataset_sha256, vertical.dataset_sha256)

    def test_drops_infeasible_retarget_joint_transition_before_normalizing(self) -> None:
        vertical = build_vertical_dataset(
            (_source("train", "released_train"), _source("validation", "released_val"))
        )
        grail = _GrailRows()
        for row in grail.rows:
            if (
                row["clip_id"] == "terrain_slopes__slope_000__000"
                and row["center_frame"] == 1
            ):
                row["y"][OUTPUT_LAYOUT["joint_position"].start] = 1.0

        mixed = combine_vertical_and_grail(
            vertical, grail, grail_dataset_sha256="9" * 64
        )

        train_clips = mixed.splits["train"].clip_id
        self.assertEqual(
            int(np.count_nonzero(train_clips == "terrain_slopes__slope_000__000")),
            1,
        )
        self.assertEqual(
            int(
                np.count_nonzero(
                    train_clips == "terrain_slopes__slope_000__000__mirror"
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
