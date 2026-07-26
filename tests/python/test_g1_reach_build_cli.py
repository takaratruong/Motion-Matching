from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from resources import build_g1_reach_database as build_cli
from resources.g1_reach_builder.annotations import (
    create_annotation_document,
    update_annotation,
    write_annotations_atomic,
)
from resources.g1_reach_builder.artifacts import read_reach_pack
from resources.g1_reach_builder.review import write_review_corpus
from tests.python.test_g1_reach_annotations import corpus_fixture


def arguments(review: Path, annotations: Path, output: Path, allow_pending=False):
    return Namespace(
        review=review,
        annotations=annotations,
        output=output,
        allow_pending=allow_pending,
    )


def write_accepted_corpus(review: Path, annotations: Path, sequence_id: str):
    corpus = replace(corpus_fixture(), sequence_ids=(sequence_id,))
    write_review_corpus(review, corpus)
    document = create_annotation_document(review)
    accepted = update_annotation(
        review,
        document.annotations[0],
        departure_frame=10,
        grab_frame=40,
        status="accepted",
        note="",
    )
    write_annotations_atomic(
        annotations,
        replace(document, annotations=(accepted,)),
    )


class ReachBuildCLITests(unittest.TestCase):
    def test_pending_annotations_block_normal_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review"
            annotations = root / "annotations.json"
            output = root / "pack"
            write_review_corpus(review, corpus_fixture())
            write_annotations_atomic(
                annotations, create_annotation_document(review)
            )

            result = build_cli.run(arguments(review, annotations, output))

            self.assertEqual(result, 1)
            self.assertFalse(output.exists())

    def test_accepted_annotation_emits_captured_and_mirrored_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review"
            annotations = root / "annotations.json"
            output = root / "pack"
            write_review_corpus(review, corpus_fixture())
            document = create_annotation_document(review)
            accepted = update_annotation(
                review,
                document.annotations[0],
                departure_frame=10,
                grab_frame=40,
                status="accepted",
                note="",
            )
            write_annotations_atomic(
                annotations,
                replace(document, annotations=(accepted,)),
            )

            result = build_cli.run(arguments(review, annotations, output))
            artifact, _, manifest = read_reach_pack(output)

            self.assertEqual(result, 0)
            self.assertEqual(artifact.active_hands.tolist(), [0, 1])
            self.assertEqual(artifact.augmentations.tolist(), [0, 1])
            self.assertEqual(manifest["pending_annotations"], 0)
            self.assertEqual(manifest["rejected_annotations"], 0)
            self.assertEqual(manifest["captured_reaches"], 1)
            self.assertEqual(manifest["mirrored_reaches"], 1)
            self.assertEqual(manifest["paired_returns"], 2)
            self.assertEqual(manifest["unavailable_returns"], 0)
            self.assertEqual(
                manifest["return_frame_count"],
                int((artifact.range_stops - artifact.contact_frames - 1).sum()),
            )
            self.assertEqual(
                manifest["minimum_return_frames"],
                int((artifact.range_stops - artifact.contact_frames - 1).min()),
            )
            self.assertEqual(
                manifest["maximum_return_frames"],
                int((artifact.range_stops - artifact.contact_frames - 1).max()),
            )
            self.assertEqual(
                [record["contact_index"] for record in manifest["reach_records"]],
                [30, 30],
            )
            self.assertEqual(
                [record["return_available"] for record in manifest["reach_records"]],
                [True, True],
            )


    def test_combines_two_review_corpora_before_mirroring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_a = root / "review_a"
            review_b = root / "review_b"
            annotations_a = root / "annotations_a.json"
            annotations_b = root / "annotations_b.json"
            output = root / "pack"
            write_accepted_corpus(review_a, annotations_a, "pickup_north_0")
            write_accepted_corpus(
                review_b, annotations_b, "tabletop_soma/height0_top"
            )

            result = build_cli.run(Namespace(
                review=[review_a, review_b],
                annotations=[annotations_a, annotations_b],
                output=output,
                allow_pending=False,
            ))
            artifact, _, manifest = read_reach_pack(output)

            self.assertEqual(result, 0)
            self.assertEqual(manifest["captured_reaches"], 2)
            self.assertEqual(manifest["mirrored_reaches"], 2)
            self.assertEqual(len(manifest["corpora"]), 2)
            self.assertEqual(
                [record["captured_reaches"] for record in manifest["corpora"]],
                [1, 1],
            )
            self.assertEqual(
                [record["sequence_ids"] for record in manifest["corpora"]],
                [["pickup_north_0"], ["tabletop_soma/height0_top"]],
            )
            self.assertEqual(artifact.active_hands.tolist(), [0, 1, 0, 1])

    def test_rejects_mismatched_or_duplicate_corpus_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_a = root / "review_a"
            review_b = root / "review_b"
            annotations_a = root / "annotations_a.json"
            annotations_b = root / "annotations_b.json"
            output = root / "pack"
            write_accepted_corpus(review_a, annotations_a, "pickup_north_0")
            write_accepted_corpus(review_b, annotations_b, "pickup_north_0")

            mismatched = build_cli.run(Namespace(
                review=[review_a, review_b],
                annotations=[annotations_a],
                output=output,
                allow_pending=False,
            ))
            self.assertEqual(mismatched, 1)
            self.assertFalse(output.exists())

            duplicate = build_cli.run(Namespace(
                review=[review_a, review_b],
                annotations=[annotations_a, annotations_b],
                output=output,
                allow_pending=False,
            ))
            self.assertEqual(duplicate, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
