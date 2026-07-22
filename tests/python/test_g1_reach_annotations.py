from pathlib import Path
import tempfile
import unittest

import numpy as np

from resources.g1_reach_builder.annotations import (
    create_annotation_document,
    load_annotations,
    update_annotation,
    write_annotations_atomic,
)
from resources.g1_reach_builder.review import (
    read_reach_proposals,
    write_review_corpus,
)
from resources.g1_reach_builder.schema import ReviewCorpus
from resources.g1_interaction_builder.schema import G1_SKELETON


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "annotate_g1_reaches.py"


def corpus_fixture() -> ReviewCorpus:
    frames = 80
    positions = np.zeros((frames, 31, 3), np.float32)
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    positions[10:41, 23, 0] = np.linspace(0.0, 0.40, 31)
    positions[41:66, 23, 0] = np.linspace(0.38, 0.0, 25)
    return ReviewCorpus(
        fps=25.0,
        positions=positions,
        rotations=rotations,
        range_starts=np.array([0], np.int32),
        range_stops=np.array([frames], np.int32),
        source_frames=np.arange(frames, dtype=np.int32) + 300,
        sequence_ids=("pickup_north_0",),
        archive_members=("g1_retargeted_motions/gmr_pkl/pickup_north_0.pkl",),
        archive_paths=("/trusted/motions.zip",),
        archive_sha256=("a" * 64,),
        source_fps=np.array([30.0]),
        fps_overridden=(False,),
        conversion_reports=({},),
        skeleton_signature=G1_SKELETON.signature(),
    )


class ReachAnnotationTests(unittest.TestCase):
    def make_review(self, directory: str):
        review = Path(directory) / "review"
        write_review_corpus(review, corpus_fixture())
        return review, read_reach_proposals(review)[0]

    def test_creates_pending_document_bound_to_review_and_proposals(self):
        with tempfile.TemporaryDirectory() as directory:
            review, proposal = self.make_review(directory)

            document = create_annotation_document(review)

            self.assertEqual(len(document.annotations), 1)
            annotation = document.annotations[0]
            self.assertEqual(annotation.proposal_id, proposal.proposal_id)
            self.assertEqual(annotation.status, "pending")
            self.assertEqual(annotation.active_hand, "left")
            self.assertEqual(len(document.review_motions_sha256), 64)
            self.assertEqual(len(document.proposals_sha256), 64)

    def test_accept_recomputes_confirmed_endpoint_and_source_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            review, proposal = self.make_review(directory)
            document = create_annotation_document(review)
            annotation = update_annotation(
                review,
                document.annotations[0],
                departure_frame=10,
                grab_frame=40,
                status="accepted",
                note="confirmed",
            )

            self.assertEqual(annotation.frame_count, 31)
            self.assertEqual(annotation.source_departure_frame, 310)
            self.assertEqual(annotation.source_grab_frame, 340)
            self.assertEqual(annotation.status, "accepted")
            self.assertEqual(annotation.note, "confirmed")
            np.testing.assert_allclose(
                annotation.endpoint_position_root, [0.40, 0.0, 0.0], atol=1e-6
            )
            np.testing.assert_allclose(
                annotation.endpoint_rotation_root_wxyz,
                [1.0, 0.0, 0.0, 0.0],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                annotation.approach_direction_root, [1.0, 0.0, 0.0], atol=1e-6
            )
            self.assertEqual(annotation.proposal_id, proposal.proposal_id)

    def test_rejects_degenerate_accepted_approach_but_allows_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _ = self.make_review(directory)
            annotation = create_annotation_document(review).annotations[0]

            with self.assertRaisesRegex(ValueError, "approach displacement"):
                update_annotation(
                    review,
                    annotation,
                    departure_frame=0,
                    grab_frame=5,
                    status="accepted",
                    note="",
                )
            rejected = update_annotation(
                review,
                annotation,
                departure_frame=0,
                grab_frame=5,
                status="rejected",
                note="not a reach",
            )
            self.assertEqual(rejected.status, "rejected")

    def test_atomic_document_round_trip_preserves_existing_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            review, _ = self.make_review(directory)
            path = Path(directory) / "annotations.json"
            document = create_annotation_document(review)
            write_annotations_atomic(path, document)
            before = path.read_bytes()

            loaded = load_annotations(path, review)
            self.assertEqual(loaded, document)
            invalid = type(document)(
                version=document.version,
                review_motions_sha256=document.review_motions_sha256,
                proposals_sha256=document.proposals_sha256,
                annotations=document.annotations + (document.annotations[0],),
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                write_annotations_atomic(path, invalid)
            self.assertEqual(path.read_bytes(), before)

    def test_flat_tool_exposes_complete_keyboard_contract(self):
        source = TOOL.read_text(encoding="utf-8")
        for contract in (
            "sys.path.insert",
            "toggle_play", "scrub(-1)", "scrub(1)", "scrub(-10)",
            "scrub(10)", "previous_proposal", "next_proposal",
            "set_departure", "set_grab", "accept_current",
            "reject_current", "save", "escape", "--smoke-test",
            "G1_SKELETON.parents", "import tkinter", "Canvas(",
        ):
            self.assertIn(contract, source)
        self.assertNotIn("matplotlib", source)
        self.assertNotIn("g1_mesh_renderer", source)
        self.assertNotIn("terrain_runtime", source)


if __name__ == "__main__":
    unittest.main()
