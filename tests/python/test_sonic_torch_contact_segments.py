from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from mm_sonic.joints import ContractError

try:
    from mm_sonic.torch_contact_segments import (
        ContactSegment,
        ContactSegmentIndex,
        segments_from_support_mask,
    )
except ImportError:
    @dataclass(frozen=True, order=True)
    class ContactSegment:
        clip_index: int
        start_frame: int
        end_frame: int
        entering_foot: int

    class ContactSegmentIndex:
        @classmethod
        def from_support_masks(cls, *_args, **_kwargs):
            raise AssertionError("contact segment index is missing")

        @classmethod
        def from_dataset(cls, *_args, **_kwargs):
            raise AssertionError("contact segment index is missing")

    def segments_from_support_mask(*_args, **_kwargs):
        raise AssertionError("contact segmentation is missing")


class ContactSegmentTests(unittest.TestCase):
    def test_segments_run_from_onset_to_next_opposite_onset(self):
        support = torch.zeros((80, 2), dtype=torch.bool)
        support[10:28, 0] = True
        support[30:49, 1] = True
        support[51:70, 0] = True

        segments = segments_from_support_mask(
            clip_index=3,
            support_mask=support,
            minimum_frames=5,
            maximum_frames=60,
        )

        self.assertEqual(
            segments,
            (
                ContactSegment(3, 10, 30, 0),
                ContactSegment(3, 30, 51, 1),
            ),
        )

    def test_segment_bounds_reject_startup_and_anomalous_gaps(self):
        support = torch.zeros((180, 2), dtype=torch.bool)
        support[2:3, 1] = True
        support[4:5, 0] = True
        support[54:86, 0] = True
        support[87:167, 1] = True
        support[168:, 0] = True

        self.assertEqual(
            segments_from_support_mask(
                clip_index=2,
                support_mask=support,
                minimum_frames=5,
                maximum_frames=60,
            ),
            (ContactSegment(2, 54, 87, 0),),
        )

    def test_rejects_malformed_support_and_bounds(self):
        valid = torch.zeros((20, 2), dtype=torch.bool)
        cases = (
            (torch.zeros((20, 2)), 5, 60, "boolean"),
            (torch.zeros(20, dtype=torch.bool), 5, 60, "shape"),
            (valid, True, 60, "minimum"),
            (valid, 0, 60, "minimum"),
            (valid, 10, 9, "maximum"),
        )
        for support, minimum, maximum, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ContractError, message):
                    segments_from_support_mask(
                        0, support, minimum, maximum
                    )

    def test_index_returns_owned_support_and_exact_entries(self):
        flat = torch.zeros((80, 2), dtype=torch.bool)
        terrain = torch.zeros((80, 2), dtype=torch.bool)
        terrain[10:29, 0] = True
        terrain[30:49, 1] = True
        terrain[50:70, 0] = True
        index = ContactSegmentIndex.from_support_masks(
            (flat, terrain),
            terrain_clip_indices=(1,),
            minimum_frames=5,
            maximum_frames=60,
        )

        self.assertIsNone(index.entry(1, 11))
        self.assertEqual(index.entry(1, 10), ContactSegment(1, 10, 30, 0))
        returned = index.support_mask(1)
        returned[10, 0] = False
        self.assertTrue(bool(index.support_mask(1)[10, 0]))

    def test_entry_eligibility_keeps_flat_and_only_terrain_onsets(self):
        flat = torch.zeros((80, 2), dtype=torch.bool)
        terrain = torch.zeros((80, 2), dtype=torch.bool)
        terrain[10:29, 0] = True
        terrain[30:49, 1] = True
        terrain[50:70, 0] = True
        index = ContactSegmentIndex.from_support_masks(
            (flat, terrain),
            terrain_clip_indices=(1,),
            minimum_frames=5,
            maximum_frames=60,
        )
        database = SimpleNamespace(
            _search_clip_index=torch.tensor([0, 0, 1, 1, 1]),
            _search_frame_index=torch.tensor([2, 7, 9, 10, 30]),
            device=torch.device("cpu"),
        )

        eligible = index.terrain_entry_eligibility(database)

        self.assertEqual(eligible.dtype, torch.bool)
        self.assertEqual(eligible.tolist(), [True, True, False, True, True])


@unittest.skipUnless(
    os.environ.get("MM_GRAIL_CONTACT_TEST") == "1",
    "set MM_GRAIL_CONTACT_TEST=1 for authenticated real-data oracle",
)
class ContactSegmentRealDataTests(unittest.TestCase):
    def test_authenticated_stair_has_frozen_segment_inventory(self):
        from mm_sonic.torch_terrain_features import TerrainDataset

        dataset_root = Path(
            os.environ.get(
                "MM_GRAIL_CONTACT_DATASET",
                "build/torch-grail-contact-segment",
            )
        )
        dataset = TerrainDataset.load(dataset_root, device="cpu")
        target = (
            "terrain/grail/"
            "grail-stair_p1-d01be55953dba78b90e9/motion.npz"
        )
        clip_index = next(
            index
            for index, clip in enumerate(dataset.folder.clips)
            if clip.relative_path == target
        )
        index = ContactSegmentIndex.from_dataset(
            dataset, minimum_frames=5, maximum_frames=60
        )

        self.assertEqual(
            tuple(
                segment.end_frame - segment.start_frame
                for segment in index.segments
                if segment.clip_index == clip_index
            ),
            (33, 40, 41, 54, 35, 36, 36, 47, 19, 22),
        )


if __name__ == "__main__":
    unittest.main()
