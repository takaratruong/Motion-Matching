from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.build_g1_pfnn_mixed_dataset import combine_vertical_and_grail
from mm_sonic.build_g1_pfnn_vertical_dataset import build_vertical_dataset
from mm_sonic.terrain_pfnn.dataset import normalize_pfnn_input
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from tests.python.test_build_g1_pfnn_vertical_dataset import _source


class _GrailRows:
    split = "train"
    x_mean = np.linspace(-1.0, 1.0, INPUT_LAYOUT.size, dtype=np.float32)
    x_std = np.linspace(0.5, 2.0, INPUT_LAYOUT.size, dtype=np.float32)
    y_mean = np.zeros(OUTPUT_LAYOUT.size, dtype=np.float32)
    y_std = np.ones(OUTPUT_LAYOUT.size, dtype=np.float32)

    def __init__(self) -> None:
        self.rows = []
        self.physical_x_by_key = {}
        self.physical_y_by_key = {}
        for family in range(2):
            for variant in range(2):
                clip = f"terrain_slopes__slope_{family:03d}__{variant:03d}"
                for center in range(2):
                    x = np.linspace(
                        family + 0.1 * variant,
                        family + 1.0 + 0.1 * variant,
                        INPUT_LAYOUT.size,
                        dtype=np.float32,
                    )
                    y = np.full(OUTPUT_LAYOUT.size, 0.2 * center)
                    y[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 0.0, 1.0)
                    key = (clip, "motion", center)
                    self.physical_x_by_key[key] = x.copy()
                    self.physical_y_by_key[key] = y.copy()
                    self.rows.append(
                        {
                            "x": normalize_pfnn_input(x, self.x_mean, self.x_std),
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

    def test_grail_merge_restores_every_physical_input_and_target_field(self):
        grail = _GrailRows()
        expected_x = grail.physical_x_by_key
        expected_y = grail.physical_y_by_key
        mixed = combine_vertical_and_grail(
            build_vertical_dataset((
                _source("train", "released_train"),
                _source("validation", "released_val"),
            )),
            grail,
            grail_dataset_sha256="9" * 64,
        )
        for split in ("train", "validation"):
            arrays = mixed.splits[split]
            for index, clip in enumerate(arrays.clip_id):
                if not str(clip).startswith("terrain_slopes__") or arrays.mirrored[index]:
                    continue
                key = (str(clip), str(arrays.sequence_lane[index]),
                       int(arrays.center_frame_120hz[index] // 4))
                np.testing.assert_allclose(arrays.x[index], expected_x[key], rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(arrays.y[index], expected_y[key], rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
