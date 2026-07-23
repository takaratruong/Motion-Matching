from pathlib import Path
import unittest

import numpy as np

from resources.g1_reach_builder.annotations import ReachAnnotation
from resources.g1_reach_builder.motions import (
    ReachAugmentation,
    ReachHand,
    build_captured_reach,
)
from resources.g1_reach_builder.annotations import load_annotations
from resources.g1_reach_builder.review import read_review_corpus
from resources.g1_reach_builder.schema import ReviewCorpus
from resources.g1_interaction_builder.schema import G1_SKELETON


def motion_corpus(frames: int = 180) -> ReviewCorpus:
    positions = np.zeros((frames, 31, 3), np.float32)
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    positions[:, 0] = np.array([1.0, 0.0, 2.0], np.float32)
    positions[:, 1, 1] = 0.8
    peak = min(120, frames - 30)
    return_start = peak + 1
    return_stop = min(frames - 6, return_start + 23)
    positions[:return_start, 23, 0] = np.linspace(0.0, 0.6, return_start)
    positions[return_start:return_stop, 23, 0] = np.linspace(
        0.6, 0.0, return_stop - return_start, endpoint=False
    )
    positions[:, 23, 2] = 0.15
    positions[:, 7, 1] = -0.8
    positions[:, 13, 1] = -0.8
    return ReviewCorpus(
        fps=25.0,
        positions=positions,
        rotations=rotations,
        range_starts=np.array([0], np.int32),
        range_stops=np.array([frames], np.int32),
        source_frames=np.arange(frames, dtype=np.int32) + 1000,
        sequence_ids=("pickup_north_0",),
        archive_members=("g1_retargeted_motions/gmr_pkl/pickup_north_0.pkl",),
        archive_paths=("/trusted/motions.zip",),
        archive_sha256=("a" * 64,),
        source_fps=np.array([30.0]),
        fps_overridden=(False,),
        conversion_reports=({},),
        skeleton_signature=G1_SKELETON.signature(),
    )


def accepted_annotation(departure: int = 10, grab: int = 120) -> ReachAnnotation:
    return ReachAnnotation(
        proposal_id="pickup_north_0:abc",
        sequence_id="pickup_north_0",
        active_hand="left",
        departure_frame=departure,
        grab_frame=grab,
        source_departure_frame=1000 + departure,
        source_grab_frame=1000 + grab,
        status="accepted",
        note="",
        endpoint_position_root=(0.0, 0.0, 0.0),
        endpoint_rotation_root_wxyz=(1.0, 0.0, 0.0, 0.0),
        approach_direction_root=(1.0, 0.0, 0.0),
    )


class CapturedReachTests(unittest.TestCase):
    def test_retains_paired_return_and_normalizes_start_root(self):
        reach = build_captured_reach(
            motion_corpus(), accepted_annotation()
        )

        self.assertEqual(reach.frame_count, 120)
        self.assertEqual(reach.source_frames[0], 30)
        self.assertEqual(reach.source_frames[-1], 149)
        self.assertEqual(reach.contact_index, 90)
        self.assertEqual(reach.source_frames[reach.contact_index], 120)
        self.assertTrue(reach.return_available)
        np.testing.assert_allclose(reach.positions[0, 0], [0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(
            reach.rotations[0, 0], [1, 0, 0, 0], atol=1e-6
        )
        self.assertEqual(reach.active_hand, ReachHand.LEFT)
        self.assertEqual(reach.augmentation, ReachAugmentation.CAPTURED)
        self.assertIsNone(reach.original_reach_id)
        self.assertEqual(reach.proposal_id, "pickup_north_0:abc")
        self.assertEqual(reach.fps, 25.0)
        self.assertEqual(reach.velocities.shape, reach.positions.shape)
        self.assertEqual(reach.angular_velocities.shape, reach.positions.shape)
        self.assertEqual(reach.foot_contacts.shape, (120, 2))
        np.testing.assert_allclose(
            np.linalg.norm(reach.rotations, axis=-1), 1.0, atol=1e-4
        )

    def test_requires_accepted_left_hand_annotation(self):
        corpus = motion_corpus()
        pending = accepted_annotation()
        pending = ReachAnnotation(**{**pending.__dict__, "status": "pending"})
        with self.assertRaisesRegex(ValueError, "accepted"):
            build_captured_reach(corpus, pending)
        right = accepted_annotation()
        right = ReachAnnotation(**{**right.__dict__, "active_hand": "right"})
        with self.assertRaisesRegex(ValueError, "left"):
            build_captured_reach(corpus, right)

    def test_short_reach_uses_available_approach_window(self):
        reach = build_captured_reach(
            motion_corpus(), accepted_annotation(departure=10, grab=13)
        )
        self.assertEqual(reach.contact_index, 3)
        self.assertLess(reach.contact_index, reach.frame_count - 1)
        self.assertAlmostEqual(
            float(np.linalg.norm(reach.approach_direction_root)), 1.0, places=5
        )

    def test_grab_pause_uses_recent_outbound_approach_evidence(self):
        corpus = motion_corpus(80)
        corpus.positions[:, 23, 0] = 0.0
        corpus.positions[10:31, 23, 0] = np.linspace(0.0, 0.40, 21)
        corpus.positions[31:51, 23, 0] = 0.40
        corpus.positions[51:74, 23, 0] = np.linspace(
            0.40, 0.0, 23, endpoint=False
        )

        reach = build_captured_reach(
            corpus, accepted_annotation(departure=10, grab=50)
        )

        np.testing.assert_allclose(
            reach.approach_direction_root, [1.0, 0.0, 0.0], atol=1e-6
        )


class CapturedReviewCorpusTests(unittest.TestCase):
    def test_all_accepted_reaches_have_paired_returns(self):
        root = Path(__file__).resolve().parents[2]
        review = root / "build/g1-reaches/review-v2"
        annotations = root / "build/g1-reaches/annotations-v2.json"
        corpus = read_review_corpus(review)
        document = load_annotations(annotations, review)

        captured = [
            build_captured_reach(corpus, annotation)
            for annotation in document.annotations
            if annotation.status == "accepted"
        ]

        self.assertEqual(len(captured), 192)
        self.assertEqual(sum(r.return_available for r in captured), 191)
        self.assertEqual(sum(not r.return_available for r in captured), 1)
        self.assertEqual(
            [r.proposal_id for r in captured if not r.return_available],
            ["pickup_north_1:cfdcdaa4db704be8"],
        )
        self.assertTrue(all(
            (r.contact_index + 1 < r.frame_count) == r.return_available
            for r in captured
        ))
        self.assertTrue(all(
            np.array_equal(
                np.diff(r.source_frames), np.ones(r.frame_count - 1)
            )
            for r in captured
        ))


if __name__ == "__main__":
    unittest.main()
