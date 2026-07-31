from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.joints import ContractError

try:
    from mm_sonic.torch_contact_segments import (
        ContactSegment,
        ContactSegmentIndex,
        SegmentPlacement,
        SegmentValidation,
        TerrainContactSegmentPolicy,
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

    SegmentPlacement = SegmentValidation = TerrainContactSegmentPolicy = None


class _ConstantGrid:
    def __init__(self, height):
        self.height = float(height)

    def sample_xy(self, points):
        return torch.full(
            points.shape[:-1], self.height, dtype=points.dtype,
            device=points.device,
        )


class _IdentityAlignment:
    def __init__(self):
        self.translation_scene_xy = torch.zeros(2)
        self.yaw_scene_from_matcher = torch.zeros(())

    def matcher_to_scene_xy(self, points):
        return points


class _FakeFootKinematics:
    def __init__(self, feet):
        self.feet = np.asarray(feet, np.float64)

    def foot_positions(self, joints, roots, quaternions):
        count = len(joints)
        return self.feet[:count].copy()


class _OutOfDomainGrid(_ConstantGrid):
    def __init__(self, height):
        super().__init__(height)
        self.calls = 0

    def sample_xy(self, points):
        self.calls += 1
        if self.calls == 1:
            return super().sample_xy(points)
        raise ContractError("outside the authoritative terrain domain")


def _placement_policy(*, foot_kinematics=None, query_grid=None):
    flat = torch.zeros((80, 2), dtype=torch.bool)
    terrain = torch.zeros((80, 2), dtype=torch.bool)
    terrain[10:29, 0] = True
    terrain[30:49, 1] = True
    terrain[50:70, 0] = True
    index = ContactSegmentIndex.from_support_masks(
        (flat, terrain), terrain_clip_indices=(1,),
        minimum_frames=5, maximum_frames=60,
    )
    body = torch.zeros((80, 3, 3), dtype=torch.float32)
    body[:, 1, :2] = torch.tensor([0.25, 0.10])
    body[:, 2, :2] = torch.tensor([0.25, -0.10])
    body[:, 1:, 2] = 0.135
    clip = SimpleNamespace(body_position_world=body.numpy())
    layout = SimpleNamespace(
        left_foot_body_index=1, right_foot_body_index=2,
    )
    dataset = SimpleNamespace(
        folder=SimpleNamespace(clips=(SimpleNamespace(), clip), layout=layout),
        clip_grids=(None, _ConstantGrid(0.10)),
        clip_alignments=(None, _IdentityAlignment()),
        device=torch.device("cpu"),
    )
    extension = SimpleNamespace(
        dataset=dataset,
        query_grid=(
            _ConstantGrid(0.50) if query_grid is None else query_grid
        ),
        alignment=_IdentityAlignment(),
    )
    return TerrainContactSegmentPolicy(
        index=index,
        extension=extension,
        foot_kinematics=foot_kinematics,
    )


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

    def test_policy_places_whole_segment_by_entering_support_surface(self):
        policy = _placement_policy()

        placement = policy.resolve_entry(
            clip_index=1,
            frame_index=10,
            yaw_offset=torch.tensor(0.0),
            translation_xy=torch.tensor([2.0, 0.0]),
            current_support_mask=torch.tensor([False, False]),
        )

        self.assertIsInstance(placement, SegmentPlacement)
        self.assertEqual(placement.segment, ContactSegment(1, 10, 30, 0))
        self.assertAlmostEqual(placement.vertical_offset_m, 0.40, places=6)
        self.assertEqual(tuple(placement.source_support_mask.shape), (20, 2))
        self.assertTrue(bool(placement.source_support_mask[0, 0]))

    def test_policy_defers_support_switch_to_emitted_fk_validation(self):
        policy = _placement_policy()
        arguments = dict(
            clip_index=1,
            frame_index=10,
            yaw_offset=torch.tensor(0.0),
            translation_xy=torch.zeros(2),
        )

        self.assertIsNotNone(
            policy.resolve_entry(
                **arguments,
                current_support_mask=torch.tensor([False, True]),
            )
        )
        self.assertIsNotNone(
            policy.resolve_entry(
                **arguments,
                current_support_mask=torch.tensor([True, False]),
            )
        )
        self.assertIsNotNone(
            policy.resolve_entry(
                **arguments,
                current_support_mask=torch.tensor([False, False]),
            )
        )

    def test_policy_rejects_entry_far_from_authenticated_scene_phase(self):
        base = _placement_policy()
        policy = TerrainContactSegmentPolicy(
            index=base.index,
            extension=base.extension,
            maximum_scene_xy_mismatch_m=0.10,
        )
        common = dict(
            clip_index=1,
            frame_index=10,
            yaw_offset=torch.tensor(0.0),
            current_support_mask=torch.tensor([False, False]),
        )

        self.assertIsNone(
            policy.resolve_entry(
                **common, translation_xy=torch.tensor([0.11, 0.0])
            )
        )
        self.assertIsNotNone(
            policy.resolve_entry(
                **common, translation_xy=torch.tensor([0.10, 0.0])
            )
        )

    def test_policy_uses_authenticated_source_to_query_scene_yaw(self):
        policy = _placement_policy()
        policy.dataset.clip_alignments[1].yaw_scene_from_matcher.fill_(0.4)
        policy.extension.alignment.yaw_scene_from_matcher.fill_(-0.2)

        yaw = policy.registered_yaw_offset(1)

        self.assertAlmostEqual(float(yaw), 0.6, places=6)

    def test_policy_requires_positive_terrain_entry_inertialization(self):
        base = _placement_policy()
        with self.assertRaisesRegex(ContractError, "inertialization"):
            TerrainContactSegmentPolicy(
                index=base.index,
                extension=base.extension,
                entry_inertialization_halflife_s=0.0,
            )

    def test_policy_samples_current_query_support_with_frozen_thresholds(self):
        policy = _placement_policy()
        bodies = torch.tensor(
            [[0.0, 0.0, 0.8], [0.0, 0.1, 0.535], [0.0, -0.1, 0.60]]
        )
        velocities = torch.zeros((3, 3))
        velocities[2, 2] = 0.2

        support = policy.query_support_mask(bodies, velocities)

        self.assertEqual(support.tolist(), [True, False])

    def test_full_segment_validation_accepts_authoritative_fk_contact(self):
        feet = np.zeros((20, 2, 3), np.float64)
        feet[..., 2] = 0.60
        feet[:, 0, 2] = 0.535
        policy = _placement_policy(
            foot_kinematics=_FakeFootKinematics(feet)
        )
        placement = policy.resolve_entry(
            clip_index=1, frame_index=10,
            yaw_offset=torch.tensor(0.0), translation_xy=torch.zeros(2),
            current_support_mask=torch.tensor([False, False]),
        )

        validation = policy.validate_emitted(
            placement,
            joint_position=torch.zeros((20, 29)),
            root_position=torch.zeros((20, 3)),
            root_orientation_wxyz=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]] * 20
            ),
        )

        self.assertIsInstance(validation, SegmentValidation)
        self.assertTrue(validation.accepted)
        self.assertIsNone(validation.reason)
        self.assertEqual(validation.foot_position_world.shape, (20, 2, 3))
        self.assertFalse(validation.foot_position_world.flags.writeable)
        self.assertEqual(validation.longest_unsupported_frames, 0)

    def test_full_segment_validation_reports_first_exact_failure(self):
        base = np.zeros((20, 2, 3), np.float64)
        base[..., 2] = 0.60
        base[:, 0, 2] = 0.535
        cases = []
        penetration = base.copy()
        penetration[5, 1, 2] = 0.46
        cases.append((penetration, None, "penetration", None))
        entering = base.copy()
        entering[0, 0, 2] = 0.60
        cases.append((entering, None, "entering_support", None))
        unsupported = base.copy()
        unsupported[3:13, :, 2] = 0.60
        cases.append((unsupported, None, "unsupported_run", 11))
        lost = base.copy()
        lost[5:7, :, 2] = 0.60
        cases.append((lost, None, "source_support_lost", None))
        cases.append((base, _OutOfDomainGrid(0.5), "out_of_domain", None))

        for feet, grid, reason, longest in cases:
            with self.subTest(reason=reason):
                policy = _placement_policy(
                    foot_kinematics=_FakeFootKinematics(feet),
                    query_grid=grid,
                )
                placement = policy.resolve_entry(
                    clip_index=1, frame_index=10,
                    yaw_offset=torch.tensor(0.0),
                    translation_xy=torch.zeros(2),
                    current_support_mask=torch.tensor([False, False]),
                )
                validation = policy.validate_emitted(
                    placement,
                    joint_position=torch.zeros((20, 29)),
                    root_position=torch.zeros((20, 3)),
                    root_orientation_wxyz=torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0]] * 20
                    ),
                )
                self.assertFalse(validation.accepted)
                self.assertEqual(validation.reason, reason)
                if longest is not None:
                    self.assertEqual(
                        validation.longest_unsupported_frames, longest
                    )


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
