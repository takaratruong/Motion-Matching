import json
import os
import struct
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

from resources.g1_terrain_builder.artifacts import (
    SUPPORT_COLUMNS,
    publish_artifacts,
    read_support_sidecar,
    read_terrain_sidecar,
    read_walkability,
    support_sidecar_bytes,
    terrain_sidecar_bytes,
    walkability_bytes,
    write_support_sidecar,
    write_terrain_sidecar,
    write_walkability,
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
        expected = (
            struct.pack("<4sIII", b"G1TF", 1, 5, 4)
            + features.astype("<f4").tobytes()
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_features.bin")

            write_terrain_sidecar(path, features)
            output = read_terrain_sidecar(path)
            with open(path, "rb") as stream:
                payload = stream.read()

        np.testing.assert_array_equal(output, features)
        self.assertEqual(terrain_sidecar_bytes(features), expected)
        self.assertEqual(payload, expected)
        self.assertEqual(output.dtype, np.dtype("<f4"))

    def test_terrain_sidecar_writer_rejects_invalid_values_and_shapes(self):
        cases = (
            (np.zeros((0, 4), np.float32), "finite.*\\(N, 4\\)"),
            (np.zeros((2, 3), np.float32), "finite.*\\(N, 4\\)"),
            (np.zeros((2, 4, 1), np.float32), "finite.*\\(N, 4\\)"),
            (np.full((2, 4), np.nan), "finite"),
            (np.full((2, 4), np.inf), "finite"),
            (np.ones((2, 4), np.float16), "float32 or float64"),
            (np.ones((2, 4), np.int32), "float32 or float64"),
            (np.ones((2, 4), np.bool_), "float32 or float64"),
            (np.full((2, 4), "1.0", dtype="<U3"), "float32 or float64"),
            (
                np.ones((2, 4), np.complex64) * (1 + 1j),
                "float32 or float64",
            ),
            (np.ones((2, 4), dtype=object), "float32 or float64"),
            (np.full((2, 4), 1.0e300, np.float64), "float32"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_features.bin")
            for features, message in cases:
                with self.subTest(shape=features.shape, message=message):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with self.assertRaisesRegex(ValueError, message):
                            write_terrain_sidecar(path, features)

    def test_float_matrix_codec_allows_finite_float64_rounding(self):
        features = np.array(
            [
                [1.0 / 3.0, -1.0 / 7.0, 1.0e-20, -1.0e20],
                [np.pi, -np.e, 0.0, np.finfo(np.float32).max],
            ],
            dtype=np.float64,
        )
        expected = (
            struct.pack("<4sIII", b"G1TF", 1, 2, 4)
            + features.astype("<f4").tobytes(order="C")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(terrain_sidecar_bytes(features), expected)

    def test_float_matrix_codec_normalizes_array_conversion_overflow(self):
        class OverflowingArray:
            def __array__(self, dtype=None):
                raise OverflowError("injected conversion overflow")

        with self.assertRaisesRegex(ValueError, "real floating matrix"):
            terrain_sidecar_bytes(OverflowingArray())

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
            (
                "inf",
                struct.pack("<4sIII", b"G1TF", 1, 1, 4)
                + np.full((1, 4), np.inf, "<f4").tobytes(),
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

    def test_support_sidecar_round_trip_is_exact_little_endian(self):
        support = np.array(
            [[0.5, -0.25, 1.25], [2.0, 3.5, -4.0]], dtype=np.float32
        )
        expected = (
            struct.pack("<4sIII", b"G1SP", 1, 2, 3)
            + support.astype("<f4").tobytes(order="C")
        )
        self.assertEqual(
            SUPPORT_COLUMNS,
            (
                "source_root_height_m",
                "source_left_toe_height_m",
                "source_right_toe_height_m",
            ),
        )
        self.assertEqual(support_sidecar_bytes(support), expected)

        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_support.bin")
            write_support_sidecar(path, support)
            output = read_support_sidecar(path)
            with open(path, "rb") as stream:
                payload = stream.read()

        self.assertEqual(payload, expected)
        np.testing.assert_array_equal(output, support)
        self.assertEqual(output.dtype, np.dtype("<f4"))

    def test_support_sidecar_writer_rejects_invalid_values_and_shapes(self):
        cases = (
            (np.zeros((0, 3), np.float32), "finite.*\\(N, 3\\)"),
            (np.zeros((2, 2), np.float32), "finite.*\\(N, 3\\)"),
            (np.zeros((2, 3, 1), np.float32), "finite.*\\(N, 3\\)"),
            (np.full((2, 3), np.nan), "finite"),
            (np.full((2, 3), np.inf), "finite"),
            (np.ones((2, 3), np.float16), "float32 or float64"),
            (np.ones((2, 3), np.int32), "float32 or float64"),
            (np.ones((2, 3), np.bool_), "float32 or float64"),
            (np.full((2, 3), "1.0", dtype="<U3"), "float32 or float64"),
            (
                np.ones((2, 3), np.complex64) * (1 + 1j),
                "float32 or float64",
            ),
            (np.ones((2, 3), dtype=object), "float32 or float64"),
            (np.full((2, 3), 1.0e300, np.float64), "float32"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_support.bin")
            for support, message in cases:
                with self.subTest(shape=support.shape, message=message):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with self.assertRaisesRegex(ValueError, message):
                            write_support_sidecar(path, support)

    def test_support_sidecar_reader_rejects_corrupt_schema_and_payload(self):
        valid = (
            struct.pack("<4sIII", b"G1SP", 1, 2, 3)
            + np.zeros((2, 3), "<f4").tobytes()
        )
        corruptions = (
            ("header", valid[:8], "truncated"),
            ("magic", b"BAD!" + valid[4:], "schema"),
            (
                "version",
                struct.pack("<4sIII", b"G1SP", 2, 2, 3) + valid[16:],
                "schema",
            ),
            (
                "dims",
                struct.pack("<4sIII", b"G1SP", 1, 2, 2) + valid[16:],
                "schema",
            ),
            ("payload", valid[:-1], "truncated"),
            ("trailing", valid + b"x", "trailing"),
            ("empty", struct.pack("<4sIII", b"G1SP", 1, 0, 3), "invalid"),
            (
                "nan",
                struct.pack("<4sIII", b"G1SP", 1, 1, 3)
                + np.full((1, 3), np.nan, "<f4").tobytes(),
                "finite",
            ),
            (
                "inf",
                struct.pack("<4sIII", b"G1SP", 1, 1, 3)
                + np.full((1, 3), np.inf, "<f4").tobytes(),
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
                        read_support_sidecar(path)

    def test_float_matrix_reader_rejects_platform_index_overflow(self):
        payload = (
            struct.pack("<4sIII", b"G1SP", 1, 2, 3)
            + np.zeros((2, 3), "<f4").tobytes()
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "terrain_support.bin")
            with open(path, "wb") as stream:
                stream.write(payload)
            simulated_iinfo = mock.Mock(max=16)
            with (
                mock.patch(
                    "resources.g1_terrain_builder.artifacts.np.iinfo",
                    return_value=simulated_iinfo,
                ),
                self.assertRaisesRegex(ValueError, "platform index limit"),
            ):
                read_support_sidecar(path)

    def test_walkability_round_trip_is_exact_little_endian_row_major(self):
        walkability = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
        expected = (
            struct.pack("<4sIII", b"G1WM", 1, 3, 2)
            + bytes((0, 1, 2, 2, 1, 0))
        )
        self.assertEqual(walkability_bytes(walkability), expected)

        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "walkability.bin")
            write_walkability(path, walkability)
            output = read_walkability(path)
            with open(path, "rb") as stream:
                payload = stream.read()

        self.assertEqual(payload, expected)
        np.testing.assert_array_equal(output, walkability)
        self.assertEqual(output.dtype, np.dtype("uint8"))

    def test_walkability_writer_rejects_invalid_shapes_dtypes_and_classes(self):
        cases = (
            (np.zeros((1, 2), np.uint8), "at least 2x2"),
            (np.zeros((2, 1), np.uint8), "at least 2x2"),
            (np.zeros(4, np.uint8), "2-D"),
            (np.zeros((2, 2, 1), np.uint8), "2-D"),
            (np.zeros((2, 2), np.float32), "integer.*classes"),
            (np.full((2, 2), np.nan), "integer.*classes"),
            (np.zeros((2, 2), np.bool_), "integer.*classes"),
            (np.zeros((2, 2), dtype=object), "integer.*classes"),
            (np.zeros((2, 2), np.complex64), "integer.*classes"),
            (np.array([[-1, 0], [1, 2]], np.int8), "classes 0, 1, or 2"),
            (np.array([[0, 1], [2, 3]], np.uint8), "classes 0, 1, or 2"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "walkability.bin")
            for walkability, message in cases:
                with self.subTest(
                    shape=walkability.shape,
                    dtype=walkability.dtype,
                    message=message,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        write_walkability(path, walkability)

    def test_walkability_writer_rejects_platform_index_overflow_before_open(self):
        sentinel = b"last-good-walkability"
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "walkability.bin")
            with open(path, "wb") as stream:
                stream.write(sentinel)
            simulated_iinfo = mock.Mock(max=3)
            with (
                mock.patch(
                    "resources.g1_terrain_builder.artifacts.np.iinfo",
                    return_value=simulated_iinfo,
                ),
                self.assertRaisesRegex(ValueError, "platform index limit"),
            ):
                write_walkability(path, np.zeros((2, 2), np.uint8))
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), sentinel)

    def test_walkability_reader_rejects_corrupt_schema_and_payload(self):
        valid = struct.pack("<4sIII", b"G1WM", 1, 3, 2) + bytes(
            (0, 1, 2, 2, 1, 0)
        )
        corruptions = (
            ("header", valid[:8], "truncated"),
            ("magic", b"BAD!" + valid[4:], "schema"),
            (
                "version",
                struct.pack("<4sIII", b"G1WM", 2, 3, 2) + valid[16:],
                "schema",
            ),
            (
                "width",
                struct.pack("<4sIII", b"G1WM", 1, 1, 2) + bytes((0, 1)),
                "dimensions",
            ),
            (
                "height",
                struct.pack("<4sIII", b"G1WM", 1, 3, 1) + bytes((0, 1, 2)),
                "dimensions",
            ),
            ("payload", valid[:-1], "truncated"),
            ("trailing", valid + b"x", "trailing"),
            ("class", valid[:-1] + bytes((3,)), "classes 0, 1, or 2"),
            (
                "huge",
                struct.pack("<4sIII", b"G1WM", 1, 0xFFFFFFFF, 2),
                "truncated",
            ),
            (
                "huge_product",
                struct.pack(
                    "<4sIII", b"G1WM", 1, 0xFFFFFFFF, 0xFFFFFFFF
                ),
                "platform index limit",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, payload, message in corruptions:
                path = os.path.join(temporary, name + ".bin")
                with open(path, "wb") as stream:
                    stream.write(payload)
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, message):
                        read_walkability(path)

    def test_invalid_serializers_do_not_truncate_existing_files(self):
        sentinel = b"last-good-sidecar"
        cases = (
            (write_terrain_sidecar, np.full((2, 4), np.nan)),
            (
                write_support_sidecar,
                np.ones((2, 3), np.complex64) * (1 + 1j),
            ),
            (write_walkability, np.zeros((2, 2), np.float32)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "sidecar.bin")
            for writer, values in cases:
                with self.subTest(writer=writer.__name__):
                    with open(path, "wb") as stream:
                        stream.write(sentinel)
                    with self.assertRaises(ValueError):
                        writer(path, values)
                    with open(path, "rb") as stream:
                        self.assertEqual(stream.read(), sentinel)

    def test_publish_replaces_directory_with_complete_validated_set(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.terrain_features[:] = np.arange(16).reshape(4, 4)
        artifacts.terrain_support[:] = np.arange(12).reshape(4, 3) / 10.0
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
                    "terrain_support.bin",
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
            np.testing.assert_array_equal(
                read_support_sidecar(
                    os.path.join(output, "terrain_support.bin")
                ),
                artifacts.terrain_support,
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
            ("terrain_support.bin", "truncated"),
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

    def test_publish_rejects_valid_but_mismatched_support_sidecar(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.terrain_support[:] = np.arange(12).reshape(4, 3)
        manifest = {"schema": "test", "validation": {"ok": True}}
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            os.mkdir(output)
            sentinel = os.path.join(output, "old")
            with open(sentinel, "w") as stream:
                stream.write("sentinel")

            def mismatching_writer(staging):
                self._terrain_writer(staging)
                write_support_sidecar(
                    os.path.join(staging, "terrain_support.bin"),
                    np.zeros_like(artifacts.terrain_support),
                )

            with self.assertRaisesRegex(ValueError, "terrain_support"):
                publish_artifacts(
                    output, artifacts, manifest, mismatching_writer
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
