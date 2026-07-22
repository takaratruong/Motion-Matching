import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from resources.confirm_g1_reach_defaults import main as confirm_main
from resources.g1_reach_builder.annotations import load_annotations
from resources.g1_reach_builder.review import write_review_corpus
from tests.python.test_g1_reach_segmentation import (
    review_corpus_for_trace,
    synthetic_trace,
)


def add_degenerate_proposal(review: Path) -> None:
    proposals_path = review / "proposals.json"
    document = json.loads(proposals_path.read_text(encoding="utf-8"))
    invalid = dict(document["proposals"][0])
    invalid.update(
        proposal_id="pickup_north_0:degenerate",
        departure_frame=0,
        grab_frame=1,
        source_departure_frame=100,
        source_grab_frame=101,
        excursion_m=0.0,
        approach_displacement_m=0.0,
        confidence=0.0,
        status="pending",
    )
    document["proposals"].append(invalid)
    proposals_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = review / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["proposals_sha256"] = hashlib.sha256(
        proposals_path.read_bytes()
    ).hexdigest()
    manifest["proposal_count"] = len(document["proposals"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ConfirmG1ReachDefaultsTests(unittest.TestCase):
    def test_confirms_valid_defaults_and_rejects_structural_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "review"
            annotations = root / "annotations.json"
            write_review_corpus(
                review, review_corpus_for_trace(synthetic_trace())
            )
            add_degenerate_proposal(review)

            result = confirm_main([
                "--review", str(review), "--output", str(annotations),
            ])

            self.assertEqual(result, 0)
            document = load_annotations(annotations, review)
            self.assertEqual(
                [annotation.status for annotation in document.annotations],
                ["accepted", "accepted", "rejected"],
            )
            self.assertIn("structural:", document.annotations[-1].note)
            before = annotations.read_bytes()

            self.assertEqual(confirm_main([
                "--review", str(review), "--output", str(annotations),
            ]), 0)
            self.assertEqual(annotations.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
