import json
import os
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from resources.g1_terrain_builder.artifacts import (
    publish_artifacts,
    read_terrain_sidecar,
    write_terrain_sidecar,
)
from resources.g1_terrain_builder.schema import ArtifactSet


class ArtifactTests(unittest.TestCase):
    @staticmethod
    def _terrain_writer(staging):
        with open(os.path.join(staging, "terrain.bin"), "wb") as stream:
            stream.write(b"terrain")
        with open(os.path.join(staging, "terrain.obj"), "w") as stream:
            stream.write("v 0 0 0\n")

    def _assert_no_publication_scratch(self, parent, output):
        self.assertFalse(os.path.exists(output + ".previous"))
        self.assertFalse(
            any(name.startswith(".g1_terrain-") for name in os.listdir(parent))
        )

    def test_terrain_sidecar_round_trip_is_exact_little_endian(self):
        features = np.arange(20, dtype=np.float32).reshape(5, 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_features.bin")

            write_terrain_sidecar(path, features)
            output = read_terrain_sidecar(path)
            with open(path, "rb") as stream:
                payload = stream.read()

        np.testing.assert_array_equal(output, features)
        self.assertEqual(
            payload[:16], struct.pack("<4sIII", b"G1TF", 1, 5, 4)
        )
        self.assertEqual(payload[16:], features.astype("<f4").tobytes())
        self.assertEqual(output.dtype, np.dtype("<f4"))

    def test_terrain_sidecar_writer_rejects_invalid_values_and_shapes(self):
        cases = (
            (np.zeros((0, 4), np.float32), "finite.*\\(N, 4\\)"),
            (np.zeros((2, 3), np.float32), "finite.*\\(N, 4\\)"),
            (np.zeros((2, 4, 1), np.float32), "finite.*\\(N, 4\\)"),
            (np.full((2, 4), np.nan), "finite"),
            (np.full((2, 4), np.inf), "finite"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_features.bin")
            for features, message in cases:
                with self.subTest(shape=features.shape, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        write_terrain_sidecar(path, features)

    def test_terrain_sidecar_reader_rejects_corrupt_schema_and_payload(self):
        valid = (
            struct.pack("<4sIII", b"G1TF", 1, 2, 4)
            + np.zeros((2, 4), "<f4").tobytes()
        )
        corruptions = (
            ("header", valid[:8], "truncated"),
            ("magic", b"BAD!" + valid[4:], "schema"),
            (
                "version",
                struct.pack("<4sIII", b"G1TF", 2, 2, 4) + valid[16:],
                "schema",
            ),
            (
                "dims",
                struct.pack("<4sIII", b"G1TF", 1, 2, 3) + valid[16:],
                "schema",
            ),
            ("payload", valid[:-1], "truncated"),
            ("trailing", valid + b"x", "trailing"),
            ("empty", struct.pack("<4sIII", b"G1TF", 1, 0, 4), "invalid"),
            (
                "nan",
                struct.pack("<4sIII", b"G1TF", 1, 1, 4)
                + np.full((1, 4), np.nan, "<f4").tobytes(),
                "finite",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, payload, message in corruptions:
                path = os.path.join(temporary, name + ".bin")
                with open(path, "wb") as stream:
                    stream.write(payload)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, message):
                        read_terrain_sidecar(path)

    def test_publish_replaces_directory_with_complete_validated_set(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.terrain_features[:] = np.arange(16).reshape(4, 4)
        manifest = {"schema": "test", "validation": {"ok": True}}
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            with open(os.path.join(output, "old"), "w") as stream:
                stream.write("sentinel")

            publish_artifacts(
                output, artifacts, manifest, self._terrain_writer
            )

            self.assertEqual(
                set(os.listdir(output)),
                {
                    "database.bin",
                    "terrain_features.bin",
                    "terrain.bin",
                    "terrain.obj",
                    "manifest.json",
                    "validation.json",
                },
            )
            with open(os.path.join(output, "manifest.json")) as stream:
                self.assertEqual(json.load(stream), manifest)
            with open(os.path.join(output, "validation.json")) as stream:
                self.assertEqual(json.load(stream), manifest["validation"])
            np.testing.assert_array_equal(
                read_terrain_sidecar(
                    os.path.join(output, "terrain_features.bin")
                ),
                artifacts.terrain_features,
            )
            self._assert_no_publication_scratch(temporary, output)

    def test_publish_failure_preserves_previous_directory(self):
        artifacts = ArtifactSet.empty(4, 2)
        manifest = {"schema": "test", "validation": {"ok": True}}
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            sentinel = os.path.join(output, "old")
            with open(sentinel, "w") as stream:
                stream.write("sentinel")

            def failed_writer(staging):
                with open(os.path.join(staging, "partial"), "w") as stream:
                    stream.write("partial")
                raise RuntimeError("terrain export failed")

            with self.assertRaisesRegex(RuntimeError, "terrain export"):
                publish_artifacts(output, artifacts, manifest, failed_writer)

            with open(sentinel) as stream:
                self.assertEqual(stream.read(), "sentinel")
            self.assertEqual(os.listdir(output), ["old"])
            self._assert_no_publication_scratch(temporary, output)

    def test_publish_recovers_crash_backup_before_new_staging_failure(self):
        artifacts = ArtifactSet.empty(4, 2)
        manifest = {"schema": "test", "validation": {"ok": True}}
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            backup = output + ".previous"
            os.mkdir(backup)
            with open(os.path.join(backup, "old"), "w") as stream:
                stream.write("last-good")

            def failed_writer(staging):
                raise RuntimeError("injected terrain export failure")

            with self.assertRaisesRegex(RuntimeError, "injected"):
                publish_artifacts(output, artifacts, manifest, failed_writer)

            self.assertTrue(
                os.path.isdir(output),
                "crash backup was not recovered as the authoritative output",
            )
            with open(os.path.join(output, "old")) as stream:
                self.assertEqual(stream.read(), "last-good")
            self.assertEqual(os.listdir(output), ["old"])
            self._assert_no_publication_scratch(temporary, output)

    def test_publish_requires_both_terrain_outputs_before_replacing(self):
        artifacts = ArtifactSet.empty(4, 2)
        manifest = {"schema": "test", "validation": {"ok": True}}
        for missing in ("terrain.bin", "terrain.obj"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = os.path.join(temporary, "published")
                os.mkdir(output)
                sentinel = os.path.join(output, "old")
                with open(sentinel, "w") as stream:
                    stream.write("sentinel")

                def incomplete_writer(staging):
                    for name in ("terrain.bin", "terrain.obj"):
                        if name != missing:
                            with open(os.path.join(staging, name), "w") as stream:
                                stream.write("terrain")

                with self.assertRaisesRegex(ValueError, missing):
                    publish_artifacts(
                        output, artifacts, manifest, incomplete_writer
                    )

                with open(sentinel) as stream:
                    self.assertEqual(stream.read(), "sentinel")
                self.assertEqual(os.listdir(output), ["old"])
                self._assert_no_publication_scratch(temporary, output)

    def test_publish_validates_every_serialized_staged_artifact(self):
        artifacts = ArtifactSet.empty(4, 2)
        manifest = {"schema": "test", "validation": {"ok": True}}
        corruptions = (
            ("database.bin", "truncated"),
            ("terrain_features.bin", "truncated"),
            ("manifest.json", "manifest.json"),
            ("validation.json", "validation.json"),
        )
        for corrupt_name, message in corruptions:
            with (
                self.subTest(corrupt_name=corrupt_name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = os.path.join(temporary, "published")
                os.mkdir(output)
                sentinel = os.path.join(output, "old")
                with open(sentinel, "w") as stream:
                    stream.write("sentinel")

                def corrupting_writer(staging):
                    self._terrain_writer(staging)
                    with open(
                        os.path.join(staging, corrupt_name), "wb"
                    ) as stream:
                        stream.write(b"corrupt")

                with self.assertRaisesRegex(ValueError, message):
                    publish_artifacts(
                        output, artifacts, manifest, corrupting_writer
                    )

                with open(sentinel) as stream:
                    self.assertEqual(stream.read(), "sentinel")
                self.assertEqual(os.listdir(output), ["old"])
                self._assert_no_publication_scratch(temporary, output)

    def test_publish_rejects_invalid_manifest_without_touching_output(self):
        artifacts = ArtifactSet.empty(4, 2)
        manifests = (
            (None, "manifest"),
            ({"schema": "test"}, "validation"),
            ({"validation": True}, "validation"),
            ({"validation": {"score": np.nan}}, "JSON"),
        )
        for manifest, message in manifests:
            with (
                self.subTest(manifest=manifest),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = os.path.join(temporary, "published")
                os.mkdir(output)
                sentinel = os.path.join(output, "old")
                with open(sentinel, "w") as stream:
                    stream.write("sentinel")

                with self.assertRaisesRegex((TypeError, ValueError), message):
                    publish_artifacts(
                        output, artifacts, manifest, self._terrain_writer
                    )

                with open(sentinel) as stream:
                    self.assertEqual(stream.read(), "sentinel")
                self.assertEqual(os.listdir(output), ["old"])
                self._assert_no_publication_scratch(temporary, output)

    def test_publish_rename_failure_rolls_back_previous_directory(self):
        artifacts = ArtifactSet.empty(4, 2)
        manifest = {"schema": "test", "validation": {"ok": True}}
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            sentinel = os.path.join(output, "old")
            with open(sentinel, "w") as stream:
                stream.write("sentinel")
            real_replace = os.replace
            failed_once = False

            def fail_staging_replace(source, destination):
                nonlocal failed_once
                if (
                    destination == output
                    and ".g1_terrain-" in source
                    and not failed_once
                ):
                    failed_once = True
                    raise OSError("injected publish rename failure")
                return real_replace(source, destination)

            with mock.patch(
                "resources.g1_terrain_builder.artifacts.os.replace",
                side_effect=fail_staging_replace,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    publish_artifacts(
                        output, artifacts, manifest, self._terrain_writer
                    )

            with open(sentinel) as stream:
                self.assertEqual(stream.read(), "sentinel")
            self.assertEqual(os.listdir(output), ["old"])
            self._assert_no_publication_scratch(temporary, output)


if __name__ == "__main__":
    unittest.main()
