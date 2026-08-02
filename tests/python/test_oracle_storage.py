from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh
from mm_sonic.terrain_oracle.storage import (
    load_corpus,
    publish_corpus,
    read_clip,
    read_mesh,
    write_clip,
    write_mesh,
)
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


ROOT = Path(__file__).resolve().parents[2]


class OracleStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_clip_round_trip_is_allow_pickle_false_and_content_addressed(self):
        """Catches nondeterministic, pickle-backed, or incorrectly named clips."""

        clip = synthetic_canonical_clip(frames=8)
        record = write_clip(self.root / "clips", clip)

        self.assertEqual(record.relative_path, f"clips/{record.sha256}.npz")
        artifact = self.root / record.relative_path
        self.assertEqual(record.sha256, hashlib.sha256(artifact.read_bytes()).hexdigest())
        with np.load(artifact, allow_pickle=False) as archive:
            self.assertTrue(archive.files)
            self.assertTrue(all(archive[name].dtype != object for name in archive.files))
        restored = read_clip(artifact)
        np.testing.assert_array_equal(restored.joint_position, clip.joint_position)
        self.assertEqual(restored.source, clip.source)

        second_root = self.root / "second"
        second = write_clip(second_root / "clips", clip)
        self.assertEqual(second.sha256, record.sha256)
        self.assertEqual(
            (second_root / second.relative_path).read_bytes(), artifact.read_bytes()
        )
        with self.assertRaises(FileExistsError):
            write_clip(self.root / "clips", clip)

    def test_clip_reader_rejects_an_artifact_with_a_mismatched_filename_digest(self):
        """Catches accepting NPZ bytes under a forged content-addressed name."""

        record = write_clip(self.root / "clips", synthetic_canonical_clip(frames=8))
        artifact = self.root / record.relative_path
        forged = artifact.with_name(f"{'0' * 64}.npz")
        artifact.rename(forged)

        with self.assertRaisesRegex(ContractError, "digest"):
            read_clip(forged)

    def test_mesh_round_trip_is_content_addressed(self):
        """Catches lossy mesh storage or a digest unrelated to stored bytes."""

        mesh = CanonicalTerrainMesh(
            vertices_local=np.array(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                dtype=np.float32,
            ),
            faces=np.array(((0, 1, 2),), dtype=np.int32),
            valid_faces=np.array((True,), dtype=np.bool_),
            source_asset_sha256="a" * 64,
        )
        record = write_mesh(self.root / "meshes", mesh)
        artifact = self.root / record.relative_path

        self.assertEqual(record.relative_path, f"meshes/{record.sha256}.npz")
        self.assertEqual(record.sha256, hashlib.sha256(artifact.read_bytes()).hexdigest())
        restored = read_mesh(artifact)
        np.testing.assert_array_equal(restored.vertices_local, mesh.vertices_local)
        np.testing.assert_array_equal(restored.faces, mesh.faces)
        np.testing.assert_array_equal(restored.valid_faces, mesh.valid_faces)

    def test_manifest_is_schema_valid_and_published_without_replacement(self):
        """Catches a mutable corpus manifest or drift from the canonical contract."""

        record = write_clip(self.root / "clips", synthetic_canonical_clip(frames=8))
        corpus_path = publish_corpus(
            self.root / "corpus", [record], {"name": "synthetic-corpus"}
        )
        manifest = load_corpus(corpus_path)
        document = json.loads((corpus_path / "manifest.json").read_text("ascii"))
        schema = json.loads(
            (ROOT / "sonic/schemas/terrain_oracle_corpus_v1.schema.json").read_text(
                "utf-8"
            )
        )

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema"]["const"], document["schema"])
        self.assertEqual(manifest.schema, "terrain-oracle-corpus/v1")
        self.assertEqual(manifest.coordinate_frame, "z-up-right-handed")
        self.assertEqual(manifest.quaternion_convention, "wxyz")
        self.assertEqual(manifest.fps, 50.0)
        self.assertEqual(manifest.clips, (record,))
        self.assertEqual(manifest.metadata, {"name": "synthetic-corpus"})
        with self.assertRaises(FileExistsError):
            publish_corpus(self.root / "corpus", [record], {"name": "again"})


if __name__ == "__main__":
    unittest.main()
