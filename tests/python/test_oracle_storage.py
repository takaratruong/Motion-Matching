from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.canonical import CanonicalTerrainMesh, TerrainBinding
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.storage import (
    MeshRecord,
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

    def test_clip_round_trip_preserves_complete_motion_and_terrain_provenance(self):
        """Catches dropping source or terrain size/license/transform evidence."""

        clip = synthetic_canonical_clip(frames=8)
        terrain = TerrainBinding(
            asset_path="/readonly/terrain.obj",
            asset_size_bytes=4096,
            asset_sha256="b" * 64,
            asset_license_id="CC-BY-4.0",
            mesh_sha256="c" * 64,
            world_from_terrain=RigidTransform(
                translation_world=np.array((1.0, 2.0, 3.0), dtype=np.float32),
                quaternion_world_from_local_wxyz=np.array(
                    (1.0, 0.0, 0.0, 0.0), dtype=np.float32
                ),
            ),
            validity_mask_path=None,
        )
        clip = replace(
            clip,
            source=replace(
                clip.source,
                source_size_bytes=8192,
                source_license_id="CC-BY-4.0",
            ),
            terrain=terrain,
        )

        record = write_clip(self.root / "clips", clip)
        restored = read_clip(self.root / record.relative_path)

        self.assertEqual(restored.source, clip.source)
        self.assertIsNotNone(restored.terrain)
        self.assertEqual(restored.terrain.asset_path, terrain.asset_path)
        self.assertEqual(restored.terrain.asset_size_bytes, terrain.asset_size_bytes)
        self.assertEqual(restored.terrain.asset_sha256, terrain.asset_sha256)
        self.assertEqual(restored.terrain.asset_license_id, terrain.asset_license_id)
        self.assertEqual(restored.terrain.mesh_sha256, terrain.mesh_sha256)
        np.testing.assert_array_equal(
            restored.terrain.world_from_terrain.translation_world,
            terrain.world_from_terrain.translation_world,
        )
        np.testing.assert_array_equal(
            restored.terrain.world_from_terrain.quaternion_world_from_local_wxyz,
            terrain.world_from_terrain.quaternion_world_from_local_wxyz,
        )

    def test_load_corpus_rejects_record_paths_that_do_not_match_their_digests(self):
        """Catches a manifest redirecting a declared digest to another path."""

        clip_record = write_clip(
            self.root / "clips", synthetic_canonical_clip(frames=8)
        )
        mesh_record = write_mesh(
            self.root / "meshes",
            CanonicalTerrainMesh(
                vertices_local=np.array(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=np.float32,
                ),
                faces=np.array(((0, 1, 2),), dtype=np.int32),
                valid_faces=np.array((True,), dtype=np.bool_),
                source_asset_sha256="a" * 64,
            ),
        )
        corpus_path = publish_corpus(
            self.root / "corpus",
            [clip_record],
            {"mesh_records": [mesh_record.to_dict()]},
        )
        manifest_path = corpus_path / "manifest.json"
        original = json.loads(manifest_path.read_text("ascii"))
        for collection, path in (
            ("clips", f"clips/{'0' * 64}.npz"),
            ("clips", f"../clips/{clip_record.sha256}.npz"),
            ("meshes", f"/meshes/{mesh_record.sha256}.npz"),
            ("meshes", f"clips/{mesh_record.sha256}.npz"),
        ):
            with self.subTest(collection=collection, path=path):
                forged = json.loads(json.dumps(original))
                forged[collection][0]["relative_path"] = path
                manifest_path.write_text(json.dumps(forged), "ascii")
                with self.assertRaisesRegex(ContractError, "relative_path"):
                    load_corpus(corpus_path)
        manifest_path.write_text(json.dumps(original), "ascii")

    def test_publish_corpus_validates_direct_mesh_records_before_creating_output(self):
        """Catches direct records bypassing the manifest record-path contract."""

        clip_record = write_clip(
            self.root / "clips", synthetic_canonical_clip(frames=8)
        )
        invalid = MeshRecord(
            relative_path=f"../meshes/{'a' * 64}.npz",
            sha256="a" * 64,
            vertex_count=3,
            face_count=1,
        )
        rejected_destination = self.root / "rejected-corpus"

        with self.assertRaisesRegex(ContractError, "relative_path"):
            publish_corpus(
                rejected_destination,
                [clip_record],
                {"mesh_records": [invalid]},
            )
        self.assertFalse(rejected_destination.exists())

        valid = write_mesh(
            self.root / "meshes",
            CanonicalTerrainMesh(
                vertices_local=np.array(
                    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    dtype=np.float32,
                ),
                faces=np.array(((0, 1, 2),), dtype=np.int32),
                valid_faces=np.array((True,), dtype=np.bool_),
                source_asset_sha256="a" * 64,
            ),
        )
        published = publish_corpus(
            self.root / "accepted-corpus",
            [clip_record],
            {"mesh_records": [valid]},
        )
        self.assertEqual(load_corpus(published).meshes, (valid,))

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
        self.assertEqual(
            schema["properties"]["joint_order"]["const"],
            list(manifest.joint_order),
        )
        self.assertEqual(
            schema["properties"]["body_order"]["const"],
            list(manifest.body_order),
        )
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            pass
        else:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(document)
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
