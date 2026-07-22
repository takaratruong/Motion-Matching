import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from resources.g1_reach_builder.review import (
    read_review_corpus,
    read_reach_proposals,
    write_review_corpus,
)
from resources.g1_reach_builder.schema import ReviewCorpus
from resources.g1_reach_builder.segmentation import (
    SegmentationConfig,
    propose_reaches,
    propose_wrist_trace,
)
from resources.g1_interaction_builder.schema import G1_SKELETON


def synthetic_trace() -> np.ndarray:
    trace = np.zeros((160, 3), np.float64)
    trace[20:51, 0] = np.linspace(0.0, 0.40, 31)
    trace[51:76, 0] = np.linspace(0.38, 0.0, 25)
    trace[90:126, 1] = np.linspace(0.0, 0.35, 36)
    trace[126:151, 1] = np.linspace(0.33, 0.0, 25)
    trace[:, 2] = 0.015 * np.sin(np.arange(160) * 0.4)
    return trace


def drifting_neutral_trace() -> np.ndarray:
    keyframes = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.40, 0.00, 0.00],
            [0.00, 0.20, 0.00],
            [0.00, 0.60, 0.00],
            [-0.20, 0.20, 0.00],
            [-0.60, 0.20, 0.00],
            [-0.20, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    frames = [0, 30, 60, 90, 120, 150, 180]
    trace = np.empty((181, 3), np.float64)
    for left in range(6):
        trace[frames[left] : frames[left + 1] + 1] = np.linspace(
            keyframes[left], keyframes[left + 1], 31
        )
    return trace


def review_corpus_for_trace(trace: np.ndarray) -> ReviewCorpus:
    frames = len(trace)
    positions = np.zeros((frames, 31, 3), np.float32)
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    positions[:, 23] = trace.astype(np.float32)
    return ReviewCorpus(
        fps=25.0,
        positions=positions,
        rotations=rotations,
        range_starts=np.array([0], np.int32),
        range_stops=np.array([frames], np.int32),
        source_frames=np.arange(frames, dtype=np.int32) + 100,
        sequence_ids=("pickup_north_0",),
        archive_members=("g1_retargeted_motions/gmr_pkl/pickup_north_0.pkl",),
        archive_paths=("/trusted/motions.zip",),
        archive_sha256=("a" * 64,),
        source_fps=np.array([30.0]),
        fps_overridden=(False,),
        conversion_reports=({},),
        skeleton_signature=G1_SKELETON.signature(),
    )


class ReachSegmentationTests(unittest.TestCase):
    def test_keeps_terminal_nonreturning_outside_run(self):
        trace = np.zeros((80, 3), np.float64)
        trace[20:61, 0] = np.linspace(0.0, 0.40, 41)
        trace[61:, 0] = 0.40

        proposals = propose_wrist_trace(
            trace, np.arange(len(trace)), "pickup_east_1"
        )

        self.assertEqual(len(proposals), 1)
        self.assertGreaterEqual(proposals[0].grab_frame, 55)

    def test_splits_prominent_endpoints_inside_one_global_outside_run(self):
        trace = drifting_neutral_trace()

        proposals = propose_wrist_trace(
            trace,
            np.arange(len(trace), dtype=np.int32),
            "pickup_north_2",
        )

        self.assertEqual(len(proposals), 3)
        np.testing.assert_allclose(
            [trace[proposal.grab_frame] for proposal in proposals],
            [trace[30], trace[90], trace[180]],
            atol=0.02,
        )

    def test_proposes_maximum_excursion_and_preceding_neutral_departure(self):
        trace = synthetic_trace()

        proposals = propose_wrist_trace(
            trace,
            np.arange(len(trace), dtype=np.int32) + 100,
            "pickup_north_0",
            SegmentationConfig(),
        )

        self.assertEqual([proposal.grab_frame for proposal in proposals], [50, 125])
        self.assertEqual([proposal.departure_frame for proposal in proposals], [25, 97])
        self.assertEqual(
            [proposal.source_grab_frame for proposal in proposals], [150, 225]
        )
        self.assertTrue(all(proposal.status == "pending" for proposal in proposals))
        self.assertTrue(all(proposal.grab_frame - proposal.departure_frame <= 90 for proposal in proposals))
        self.assertTrue(all(proposal.excursion_m >= 0.18 for proposal in proposals))
        self.assertTrue(all(proposal.approach_displacement_m >= 0.01 for proposal in proposals))

    def test_merges_nearby_peaks_and_keeps_larger_excursion(self):
        trace = np.zeros((80, 3), np.float64)
        trace[10:31, 0] = np.linspace(0.0, 0.25, 21)
        trace[31:36, 0] = np.linspace(0.23, 0.14, 5)
        trace[36:46, 0] = np.linspace(0.16, 0.32, 10)
        trace[46:61, 0] = np.linspace(0.30, 0.0, 15)

        proposals = propose_wrist_trace(
            trace, np.arange(80), "drawers", SegmentationConfig()
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].grab_frame, 45)

    def test_review_publication_contains_checksum_bound_pending_proposals(self):
        corpus = review_corpus_for_trace(synthetic_trace())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review"

            write_review_corpus(output, corpus)
            loaded = read_review_corpus(output)
            proposals = read_reach_proposals(output)

            self.assertEqual(len(proposals), 2)
            self.assertEqual(proposals, propose_reaches(loaded))
            self.assertTrue(all(proposal.status == "pending" for proposal in proposals))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["proposal_count"], 2)
            self.assertEqual(len(manifest["proposals_sha256"]), 64)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"manifest.json", "review_motions.npz", "proposals.json"},
            )


if __name__ == "__main__":
    unittest.main()
