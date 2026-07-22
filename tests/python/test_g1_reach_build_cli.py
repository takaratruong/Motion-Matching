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


if __name__ == "__main__":
    unittest.main()
