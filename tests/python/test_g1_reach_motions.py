import unittest

import numpy as np

from resources.g1_reach_builder.annotations import ReachAnnotation
from resources.g1_reach_builder.motions import (
    ReachAugmentation,
    ReachHand,
    build_captured_reach,
)
from resources.g1_reach_builder.schema import ReviewCorpus
from resources.g1_interaction_builder.schema import G1_SKELETON


def motion_corpus(frames: int = 130) -> ReviewCorpus:
    positions = np.zeros((frames, 31, 3), np.float32)
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    positions[:, 0] = np.array([1.0, 0.0, 2.0], np.float32)
    positions[:, 1, 1] = 0.8
    positions[:, 23, 0] = np.linspace(0.0, 0.6, frames)
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
    def test_clips_outbound_motion_through_grab_and_normalizes_start_root(self):
        reach = build_captured_reach(
            motion_corpus(), accepted_annotation()
        )

        self.assertEqual(reach.frame_count, 91)
        self.assertEqual(reach.source_frames[0], 1030)
        self.assertEqual(reach.source_frames[-1], 1120)
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
        self.assertEqual(reach.foot_contacts.shape, (91, 2))
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
        self.assertEqual(reach.frame_count, 4)
        self.assertAlmostEqual(
            float(np.linalg.norm(reach.approach_direction_root)), 1.0, places=5
        )

    def test_grab_pause_uses_recent_outbound_approach_evidence(self):
        corpus = motion_corpus(80)
        corpus.positions[:, 23, 0] = 0.0
        corpus.positions[10:31, 23, 0] = np.linspace(0.0, 0.40, 21)
        corpus.positions[31:51, 23, 0] = 0.40

        reach = build_captured_reach(
            corpus, accepted_annotation(departure=10, grab=50)
        )

        np.testing.assert_allclose(
            reach.approach_direction_root, [1.0, 0.0, 0.0], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
