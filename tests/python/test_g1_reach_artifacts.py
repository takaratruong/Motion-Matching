import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from resources.g1_reach_builder.artifacts import (
    DATABASE_MAGIC,
    FEATURE_HEADER,
    FEATURE_MAGIC,
    assemble_reach_pack,
    read_reach_pack,
    write_reach_pack,
)
from resources.g1_reach_builder.mirror import mirror_reach
from resources.g1_reach_builder.motions import build_captured_reach
from tests.python.test_g1_reach_motions import (
    accepted_annotation,
    motion_corpus,
)


def reach_pair():
    captured = build_captured_reach(
        motion_corpus(), accepted_annotation(departure=20, grab=80)
    )
    return [captured, mirror_reach(captured)]


def manifest_fixture(artifact, features):
    return {
        "schema": "g1-reach-pack",
        "version": 2,
        "captured_reaches": 1,
        "mirrored_reaches": 1,
        "total_reaches": 2,
        "frame_count": len(artifact.positions),
        "feature_dimension": features.values.shape[1],
        "pending_annotations": 0,
        "rejected_annotations": 0,
    }


class ReachArtifactTests(unittest.TestCase):
    def test_round_trips_database_features_and_provenance(self):
        reaches = reach_pair()
        artifact, features = assemble_reach_pack(reaches)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"

            write_reach_pack(
                output, artifact, features, manifest_fixture(artifact, features)
            )
            loaded, loaded_features, manifest = read_reach_pack(output)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"reach_database.bin", "reach_features.bin", "manifest.json"},
            )
            self.assertEqual(
                (output / "reach_database.bin").read_bytes()[:8], DATABASE_MAGIC
            )
            self.assertEqual(
                (output / "reach_features.bin").read_bytes()[:8], FEATURE_MAGIC
            )
            np.testing.assert_array_equal(loaded.positions, artifact.positions)
            np.testing.assert_array_equal(
                loaded.contact_frames, artifact.contact_frames
            )
            np.testing.assert_array_equal(
                loaded_features.values, features.values
            )
            self.assertEqual(loaded.source_names, ("pickup_north_0",))
            self.assertEqual(loaded.active_hands.tolist(), [0, 1])
            self.assertEqual(loaded.augmentations.tolist(), [0, 1])
            self.assertEqual(loaded.original_indices.tolist(), [-1, 0])
            self.assertEqual(manifest["captured_reaches"], 1)
            self.assertEqual(manifest["mirrored_reaches"], 1)
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["paired_returns"], 2)
            self.assertEqual(manifest["unavailable_returns"], 0)
            self.assertEqual(manifest["return_frame_count"], 138)
            self.assertEqual(manifest["minimum_return_frames"], 69)
            self.assertEqual(manifest["maximum_return_frames"], 69)
            self.assertEqual(len(manifest["reach_database_sha256"]), 64)

    def test_rejects_contact_frames_outside_their_clip_ranges(self):
        artifact, _ = assemble_reach_pack(reach_pair())
        artifact.contact_frames = artifact.range_stops.copy()

        with self.assertRaisesRegex(ValueError, "contact frames"):
            artifact.validate()

    def test_rejects_trailing_database_bytes_and_manifest_checksum_mutation(self):
        artifact, features = assemble_reach_pack(reach_pair())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            write_reach_pack(
                output, artifact, features, manifest_fixture(artifact, features)
            )
            database = output / "reach_database.bin"
            database.write_bytes(database.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "checksum|trailing"):
                read_reach_pack(output)

            write_reach_pack(
                output, artifact, features, manifest_fixture(artifact, features)
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reach_features_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_reach_pack(output)

    def test_rejects_features_that_differ_from_endpoints_only_by_zero_sign(self):
        artifact, features = assemble_reach_pack(reach_pair())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            write_reach_pack(
                output, artifact, features, manifest_fixture(artifact, features)
            )
            feature_path = output / "reach_features.bin"
            data = bytearray(feature_path.read_bytes())
            words = np.frombuffer(data, dtype="<u4", offset=FEATURE_HEADER.size)
            zero = np.flatnonzero(words & 0x7FFFFFFF == 0)[0]
            words[zero] ^= np.uint32(0x80000000)
            feature_path.write_bytes(data)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["reach_features_sha256"] = hashlib.sha256(data).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "feature values"):
                read_reach_pack(output)

    def test_publication_is_byte_deterministic(self):
        artifact, features = assemble_reach_pack(reach_pair())
        manifest = manifest_fixture(artifact, features)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_reach_pack(first, artifact, features, manifest)
            write_reach_pack(second, artifact, features, manifest)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )

    def test_rejects_manifest_augmentation_count_mismatch(self):
        artifact, features = assemble_reach_pack(reach_pair())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            write_reach_pack(
                output, artifact, features, manifest_fixture(artifact, features)
            )
            path = output / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["captured_reaches"] = 2
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "captured reach count"):
                read_reach_pack(output)


if __name__ == "__main__":
    unittest.main()
