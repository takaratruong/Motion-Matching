import copy
import hashlib
import inspect
import json
import os
import struct
import tempfile
import threading
import time
import unittest
import warnings
from functools import lru_cache
from unittest import mock

import numpy as np

from resources.g1_terrain_builder import artifacts as artifacts_module
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
from resources.g1_terrain_builder.scenes import (
    REQUIRED_SCENE_IDS,
    BuiltScene,
    SceneDefinition,
    ScenePack,
    SceneRoute,
    build_scene_pack,
    canonical_json_bytes as scene_json_bytes,
)
from resources.g1_terrain_builder.schema import ArtifactSet
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    surface_semantics,
    surface_semantics_signature,
)


@lru_cache(maxsize=1)
def tiny_scene_pack():
    definitions = []
    for scene_id in REQUIRED_SCENE_IDS:
        definitions.append(SceneDefinition(
            scene_id=scene_id,
            label=scene_id,
            provenance={
                "kind": "procedural", "source_ids": [],
                "parameters": {"fixture": True},
            },
            surface=FlatTerrain(),
            heightfield_bounds_xz=(-0.02, 0.02, -0.02, 0.02),
            playable_bounds_xz=(-0.01, 0.01, -0.01, 0.01),
            lookahead_bounds_xz=(-0.02, 0.02, -0.02, 0.02),
            spawn_position=(0.0, 0.0, 0.0),
            spawn_yaw_radians=0.0,
            regions={
                "certified": ({
                    "id": "fixture",
                    "bounds_xz": [-0.01, 0.01, -0.01, 0.01],
                },),
                "stress": (), "blocked": (),
            },
            routes=(SceneRoute(
                "fixture", ((0.0, 0.0), (0.0, 0.01)),
                "traverse", 1, 0.0,
            ),),
            walkability=lambda x, z: 1,
        ))
    return build_scene_pack(definitions)


def tiny_manifest_base(artifacts, *, diagnostic_mode=True):
    frames = len(artifacts.positions)
    return {
        "schema": "g1-terrain-artifacts/v2",
        "output_fps": 25.0,
        "feature_dimensions": 31,
        "terrain_dimensions": 4,
        "support_dimensions": 3,
        "terrain_feature_distances_m": [0.25, 0.5, 0.75, 1.0],
        "total_clips": 1,
        "grail_clips": 0,
        "skipped_clips": 0,
        "database_frames": frames,
        "diagnostic_mode": diagnostic_mode,
        "sources": [{
            "name": "fixture", "terrain_id": "flat", "source_fps": 25.0,
            "source_frames": frames, "output_frames": frames,
            "range_start": 0, "range_stop": frames,
            "source_frame_map": list(range(frames)),
        }],
        "skeleton": {
            "names": ["Simulation", "Hips"], "parents": [-1, 0],
            "signature": "0" * 64,
        },
        "contact": {
            "speed_threshold": 0.15, "height_threshold": 0.06,
            "median_filter_frames": 3,
        },
        "surface": {
            "semantics": surface_semantics(),
            "signature": surface_semantics_signature(),
        },
        "validation": {
            "schema": "g1-terrain-validation/v1",
            "duration_error_s": [0.0],
            "fk_max_error_m": [0.0],
            "quaternion_norm_max_error": [0.0],
        },
    }


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def repack_with_scene_metadata(pack, mutation):
    scene = pack.scenes[0]
    metadata = copy.deepcopy(scene.metadata)
    mutation(metadata)
    scene_json = scene_json_bytes(metadata)
    changed_scene = BuiltScene(
        scene.scene_id, scene_json, scene.terrain_bin,
        scene.terrain_obj, scene.walkability_bin,
    )
    index = copy.deepcopy(pack.index)
    index["scenes"][0]["sha256"] = hashlib.sha256(scene_json).hexdigest()
    return ScenePack(
        scene_json_bytes(index), (changed_scene, *pack.scenes[1:]))


class ArtifactTests(unittest.TestCase):

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

class ArtifactPublicationV2Tests(unittest.TestCase):
    def _scratch(self, parent, output):
        prefix = f".{os.path.basename(output)}.staging-"
        return sorted(
            os.path.join(parent, name)
            for name in os.listdir(parent)
            if name.startswith(prefix)
        )

    def _old_output(self, temporary):
        output = os.path.join(temporary, "published")
        os.mkdir(output)
        with open(os.path.join(output, "old"), "wb") as stream:
            stream.write(b"last-good")
        return output

    def test_v2_publish_api_has_explicit_candidate_validator(self):
        parameters = inspect.signature(publish_artifacts).parameters
        self.assertEqual(tuple(parameters), (
            "output_dir", "artifacts", "manifest_base", "scene_pack",
            "validate_candidate",
        ))
        self.assertIs(
            parameters["validate_candidate"].default,
            inspect.Parameter.empty,
        )
        with self.assertRaises(TypeError):
            publish_artifacts(None, None, None, None)

    def test_publish_writes_exact_complete_tree_and_final_hash_manifest(self):
        artifacts = ArtifactSet.empty(4, 2)
        artifacts.terrain_features[:] = np.arange(16).reshape(4, 4)
        artifacts.terrain_support[:] = np.arange(12).reshape(4, 3)
        pack = tiny_scene_pack()
        validated = []
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            manifest = publish_artifacts(
                output, artifacts, tiny_manifest_base(artifacts), pack,
                lambda path: validated.append(path),
            )

            self.assertEqual(len(validated), 1)
            self.assertNotEqual(validated[0], output)
            self.assertEqual(set(os.listdir(output)), {
                "database.bin", "terrain_features.bin", "terrain_support.bin",
                "manifest.json", "validation.json", "scenes",
            })
            scenes_root = os.path.join(output, "scenes")
            self.assertEqual(
                sorted(os.listdir(scenes_root)),
                sorted(["index.json", *REQUIRED_SCENE_IDS]),
            )
            for scene_id in REQUIRED_SCENE_IDS:
                self.assertEqual(set(os.listdir(
                    os.path.join(scenes_root, scene_id))), {
                    "scene.json", "terrain.bin", "terrain.obj",
                    "walkability.bin",
                })
            self.assertEqual(
                manifest["database"]["sha256"],
                file_sha256(os.path.join(output, "database.bin")),
            )
            self.assertEqual(
                manifest["sidecars"]["terrain_features"]["sha256"],
                file_sha256(os.path.join(output, "terrain_features.bin")),
            )
            self.assertEqual(
                manifest["sidecars"]["terrain_support"]["sha256"],
                file_sha256(os.path.join(output, "terrain_support.bin")),
            )
            self.assertEqual(
                manifest["scene_index"]["sha256"],
                file_sha256(os.path.join(scenes_root, "index.json")),
            )
            self.assertEqual(
                manifest["validation_file"]["sha256"],
                file_sha256(os.path.join(output, "validation.json")),
            )
            with open(os.path.join(output, "manifest.json"), "rb") as stream:
                self.assertEqual(stream.read(), scene_json_bytes(manifest))
            with open(os.path.join(scenes_root, "index.json"), "rb") as stream:
                self.assertEqual(stream.read(), pack.index_json)
            self.assertEqual(self._scratch(temporary, output), [])

    def test_post_callback_byte_mutation_preserves_previous_output(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def mutate(staging):
                path = os.path.join(
                    staging, "scenes", REQUIRED_SCENE_IDS[0], "terrain.bin")
                with open(path, "ab") as stream:
                    stream.write(b"post-callback-corruption")

            with self.assertRaisesRegex(ValueError, "scene|staged|bytes"):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts), pack,
                    mutate,
                )

            with open(os.path.join(output, "old"), "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")
            self.assertEqual(os.listdir(output), ["old"])
            self.assertEqual(self._scratch(temporary, output), [])

    def test_post_callback_extra_directory_and_symlinks_are_rejected(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()

        def extra_directory(staging):
            os.mkdir(os.path.join(staging, "extra-empty"))

        def file_symlink(staging):
            path = os.path.join(staging, "terrain_features.bin")
            os.unlink(path)
            os.symlink("database.bin", path)

        def directory_symlink(staging):
            os.symlink(
                os.path.join("scenes", REQUIRED_SCENE_IDS[0]),
                os.path.join(staging, "scene-alias"),
                target_is_directory=True,
            )

        for name, mutation in (
            ("extra_directory", extra_directory),
            ("file_symlink", file_symlink),
            ("directory_symlink", directory_symlink),
        ):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = self._old_output(temporary)
                with self.assertRaisesRegex(ValueError, "tree|symlink|node"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts), pack,
                        mutation,
                    )
                with open(os.path.join(output, "old"), "rb") as stream:
                    self.assertEqual(stream.read(), b"last-good")
                self.assertEqual(self._scratch(temporary, output), [])

    def test_exchange_failure_preserves_previous_output(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)
            with mock.patch.object(
                artifacts_module, "_rename_exchange", create=True,
                side_effect=OSError("injected exchange failure"),
            ):
                with self.assertRaisesRegex(OSError, "exchange failure"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts), pack,
                        lambda path: None,
                    )
            with open(os.path.join(output, "old"), "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")
            self.assertEqual(self._scratch(temporary, output), [])

    def test_manifest_base_is_exact_and_validated_before_staging(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        base = tiny_manifest_base(artifacts)
        cases = []
        extra = copy.deepcopy(base)
        extra["extra"] = True
        cases.append(("extra_key", extra, "manifest base keys"))
        missing = copy.deepcopy(base)
        missing.pop("sources")
        cases.append(("missing_key", missing, "manifest base keys"))
        schema = copy.deepcopy(base)
        schema["schema"] = "g1-terrain-artifacts/v1"
        cases.append(("schema", schema, "manifest schema"))
        validation_key = copy.deepcopy(base)
        validation_key["validation"]["extra"] = []
        cases.append(("validation_key", validation_key, "validation"))
        validation_schema = copy.deepcopy(base)
        validation_schema["validation"]["schema"] = "wrong"
        cases.append(("validation_schema", validation_schema, "validation"))
        nonfinite = copy.deepcopy(base)
        nonfinite["validation"]["duration_error_s"] = [float("nan")]
        cases.append(("nonfinite", nonfinite, "finite"))
        surface = copy.deepcopy(base)
        surface["surface"]["semantics"]["cell_size_m"] = 0.04
        cases.append(("surface_signature", surface, "surface signature"))

        for name, candidate, message in cases:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = self._old_output(temporary)
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    publish_artifacts(
                        output, artifacts, candidate, pack,
                        lambda path: None,
                    )
                self.assertEqual(os.listdir(output), ["old"])
                self.assertEqual(self._scratch(temporary, output), [])

    def test_candidate_failure_preserves_previous_output_and_cleans_owned_scratch(self):
        artifacts = ArtifactSet.empty(4, 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)
            with self.assertRaisesRegex(ValueError, "injected candidate"):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(),
                    lambda path: (_ for _ in ()).throw(
                        ValueError("injected candidate failure")),
                )
            with open(os.path.join(output, "old"), "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")
            self.assertEqual(self._scratch(temporary, output), [])

    def test_post_callback_fifo_is_rejected_without_opening_it(self):
        artifacts = ArtifactSet.empty(4, 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def add_fifo(staging):
                os.mkfifo(os.path.join(staging, "blocking-fifo"))

            with self.assertRaisesRegex(ValueError, "non-regular node"):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(), add_fifo,
                )
            self.assertEqual(os.listdir(output), ["old"])
            self.assertEqual(self._scratch(temporary, output), [])

    def test_pack_requires_exact_ids_index_descriptors_and_asset_descriptors(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        variants = []
        variants.append((
            "reordered_scenes",
            ScenePack(pack.index_json, tuple(reversed(pack.scenes))),
            "required scene order",
        ))
        variants.append((
            "subset_scenes",
            ScenePack(pack.index_json, pack.scenes[:-1]),
            "required scene order",
        ))
        index = copy.deepcopy(pack.index)
        index["scene_ids"] = list(reversed(index["scene_ids"]))
        variants.append((
            "reordered_index", ScenePack(scene_json_bytes(index), pack.scenes),
            "required scene IDs",
        ))
        index = copy.deepcopy(pack.index)
        index["scenes"][0]["sha256"] = "0" * 64
        variants.append((
            "forged_index_sha",
            ScenePack(scene_json_bytes(index), pack.scenes),
            "descriptor SHA-256",
        ))
        index = copy.deepcopy(pack.index)
        index["scenes"][0]["extra"] = True
        variants.append((
            "extra_index_descriptor_key",
            ScenePack(scene_json_bytes(index), pack.scenes),
            "descriptor SHA-256",
        ))
        variants.append((
            "asset_sha",
            repack_with_scene_metadata(
                pack,
                lambda metadata: metadata["heightfield"].__setitem__(
                    "sha256", "0" * 64)),
            "heightfield descriptor",
        ))
        variants.append((
            "asset_path",
            repack_with_scene_metadata(
                pack,
                lambda metadata: metadata["mesh"].__setitem__(
                    "path", "../terrain.obj")),
            "mesh descriptor",
        ))
        variants.append((
            "asset_extra_key",
            repack_with_scene_metadata(
                pack,
                lambda metadata: metadata["walkability"].__setitem__(
                    "unexpected", True)),
            "walkability descriptor",
        ))
        variants.append((
            "scene_id",
            repack_with_scene_metadata(
                pack, lambda metadata: metadata.__setitem__("id", "forged")),
            "scene JSON ID",
        ))
        variants.append((
            "scene_surface",
            repack_with_scene_metadata(
                pack, lambda metadata: metadata.__setitem__(
                    "surface_signature", "0" * 64)),
            "surface signature",
        ))

        for name, candidate_pack, message in variants:
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = os.path.join(temporary, "published")
                with self.assertRaisesRegex(ValueError, message):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts),
                        candidate_pack, lambda path: None,
                    )
                self.assertFalse(os.path.lexists(output))
                self.assertEqual(self._scratch(temporary, output), [])

    def test_absent_output_uses_atomic_replace_not_exchange(self):
        artifacts = ArtifactSet.empty(4, 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")
            with mock.patch.object(
                artifacts_module, "_rename_exchange",
                side_effect=AssertionError("exchange must not be used"),
            ):
                manifest = publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(), lambda path: None,
                )
            self.assertTrue(os.path.isdir(output))
            with open(os.path.join(output, "manifest.json"), "rb") as stream:
                self.assertEqual(stream.read(), scene_json_bytes(manifest))
            self.assertEqual(self._scratch(temporary, output), [])

    def test_existing_output_symlink_and_non_directory_are_rejected(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        for kind in ("symlink", "file"):
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = os.path.join(temporary, "published")
                if kind == "symlink":
                    target = os.path.join(temporary, "target")
                    os.mkdir(target)
                    with open(os.path.join(target, "old"), "wb") as stream:
                        stream.write(b"last-good")
                    os.symlink(target, output, target_is_directory=True)
                else:
                    with open(output, "wb") as stream:
                        stream.write(b"last-good")
                with self.assertRaisesRegex(ValueError, "real directory"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts), pack,
                        lambda path: None,
                    )
                self.assertEqual(self._scratch(temporary, output), [])
                if kind == "symlink":
                    with open(os.path.join(target, "old"), "rb") as stream:
                        self.assertEqual(stream.read(), b"last-good")
                else:
                    with open(output, "rb") as stream:
                        self.assertEqual(stream.read(), b"last-good")

    def test_parent_fsync_failure_rolls_back_existing_and_absent_output(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        real_fsync_parent = artifacts_module._fsync_parent
        for existed in (True, False):
            with (
                self.subTest(existed=existed),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = self._old_output(temporary) if existed else \
                    os.path.join(temporary, "published")
                calls = 0

                def fail_first_parent_fsync(descriptor):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise OSError("injected parent fsync failure")
                    return real_fsync_parent(descriptor)

                with mock.patch.object(
                    artifacts_module, "_fsync_parent",
                    side_effect=fail_first_parent_fsync,
                ):
                    with self.assertRaisesRegex(OSError, "parent fsync"):
                        publish_artifacts(
                            output, artifacts, tiny_manifest_base(artifacts),
                            pack, lambda path: None,
                        )
                self.assertEqual(calls, 2)
                if existed:
                    self.assertEqual(os.listdir(output), ["old"])
                    with open(os.path.join(output, "old"), "rb") as stream:
                        self.assertEqual(stream.read(), b"last-good")
                else:
                    self.assertFalse(os.path.lexists(output))
                self.assertEqual(self._scratch(temporary, output), [])

    def test_failed_rollback_preserves_both_paths(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        real_exchange = artifacts_module._rename_exchange
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)
            exchange_calls = 0

            def fail_rollback_exchange(source, destination):
                nonlocal exchange_calls
                exchange_calls += 1
                if exchange_calls == 2:
                    raise OSError("injected rollback exchange failure")
                return real_exchange(source, destination)

            with (
                mock.patch.object(
                    artifacts_module, "_rename_exchange",
                    side_effect=fail_rollback_exchange,
                ),
                mock.patch.object(
                    artifacts_module, "_fsync_parent",
                    side_effect=OSError("injected durability failure"),
                ),
                self.assertRaises(artifacts_module.PublicationRollbackError)
                as raised,
            ):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts), pack,
                    lambda path: None,
                )
            self.assertFalse(raised.exception.committed)
            self.assertTrue(os.path.isdir(output))
            self.assertTrue(os.path.isdir(raised.exception.scratch_path))
            with open(os.path.join(
                    raised.exception.scratch_path, "old"), "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")
            self.assertTrue(os.path.isfile(os.path.join(output, "manifest.json")))

    def test_post_commit_cleanup_failure_keeps_new_output_and_scratch(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)
            with (
                mock.patch.object(
                    artifacts_module, "_remove_owned_scratch",
                    side_effect=OSError("injected cleanup failure"),
                ),
                self.assertRaises(artifacts_module.PublicationCommittedError)
                as raised,
            ):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts), pack,
                    lambda path: None,
                )
            self.assertTrue(raised.exception.committed)
            self.assertEqual(raised.exception.output_dir, output)
            self.assertTrue(os.path.isfile(os.path.join(output, "manifest.json")))
            self.assertTrue(os.path.isdir(raised.exception.scratch_path))
            with open(os.path.join(
                    raised.exception.scratch_path, "old"), "rb") as stream:
                self.assertEqual(stream.read(), b"last-good")

    def test_success_does_not_turn_post_delete_fsync_into_cleanup_failure(self):
        artifacts = ArtifactSet.empty(4, 2)
        real_fsync_parent = artifacts_module._fsync_parent
        calls = 0
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def fail_second_parent_fsync(descriptor):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise OSError("post-delete fsync cannot preserve scratch")
                return real_fsync_parent(descriptor)

            with mock.patch.object(
                artifacts_module, "_fsync_parent",
                side_effect=fail_second_parent_fsync,
            ):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(), lambda path: None,
                )
            self.assertEqual(calls, 1)
            self.assertTrue(os.path.isfile(os.path.join(output, "manifest.json")))
            self.assertEqual(self._scratch(temporary, output), [])

    def test_error_cleanup_removes_only_this_calls_scratch(self):
        artifacts = ArtifactSet.empty(4, 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)
            unknown = os.path.join(temporary, ".published.staging-unknown")
            os.mkdir(unknown)
            with open(os.path.join(unknown, "sentinel"), "wb") as stream:
                stream.write(b"unrelated")
            with self.assertRaisesRegex(ValueError, "candidate"):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(),
                    lambda path: (_ for _ in ()).throw(
                        ValueError("candidate failed")),
                )
            self.assertEqual(self._scratch(temporary, output), [unknown])
            with open(os.path.join(unknown, "sentinel"), "rb") as stream:
                self.assertEqual(stream.read(), b"unrelated")

    def test_same_name_foreign_staging_replacement_is_never_deleted(self):
        artifacts = ArtifactSet.empty(4, 2)
        moved_owned = []
        foreign_marker = []
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def replace_staging_with_foreign_directory(staging):
                moved = staging + ".callback-moved-owned"
                os.rename(staging, moved)
                os.mkdir(staging)
                marker = os.path.join(staging, "foreign-sentinel")
                with open(marker, "wb") as stream:
                    stream.write(b"foreign-do-not-delete")
                moved_owned.append(moved)
                foreign_marker.append(marker)

            error = None
            try:
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(), replace_staging_with_foreign_directory,
                )
            except Exception as caught:
                error = caught

            self.assertIsNotNone(error)
            self.assertRegex(str(error), "identity")
            self.assertTrue(os.path.isdir(moved_owned[0]))
            self.assertTrue(os.path.isfile(foreign_marker[0]))
            with open(foreign_marker[0], "rb") as stream:
                self.assertEqual(stream.read(), b"foreign-do-not-delete")
            self.assertEqual(os.listdir(output), ["old"])

    def test_staged_tree_is_revalidated_under_parent_lock(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        real_validate = artifacts_module._validate_staged_v2
        calls = 0
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def mutate_on_locked_validation(staging, *arguments):
                nonlocal calls
                calls += 1
                if calls == 3:
                    with open(os.path.join(staging, "late-extra"), "wb") \
                            as stream:
                        stream.write(b"late")
                return real_validate(staging, *arguments)

            with mock.patch.object(
                artifacts_module, "_validate_staged_v2",
                side_effect=mutate_on_locked_validation,
            ):
                with self.assertRaisesRegex(ValueError, "tree"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts), pack,
                        lambda path: None,
                    )
            self.assertEqual(calls, 3)
            self.assertEqual(os.listdir(output), ["old"])
            self.assertEqual(self._scratch(temporary, output), [])

    def test_database_hashing_and_staging_do_not_read_whole_database_bytes(self):
        artifacts = ArtifactSet.empty(4, 2)
        seen = []
        real_read = artifacts_module._read_regular_bytes
        with tempfile.TemporaryDirectory() as temporary:
            output = os.path.join(temporary, "published")

            def reject_database_reads(path):
                seen.append(path)
                if os.path.basename(path) in ("database.bin", "database.bin.building"):
                    raise AssertionError("whole database read")
                return real_read(path)

            with mock.patch.object(
                artifacts_module, "_read_regular_bytes",
                side_effect=reject_database_reads,
            ):
                publish_artifacts(
                    output, artifacts, tiny_manifest_base(artifacts),
                    tiny_scene_pack(), lambda path: None,
                )
        self.assertEqual(seen, [])

    def test_staged_float_arrays_are_compared_by_encoded_bits(self):
        artifacts = ArtifactSet.empty(4, 2)
        real_reader = artifacts_module.read_holden_database
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def signed_zero_reader(path):
                loaded = real_reader(path)
                loaded.positions[0, 0, 0] = np.float32(-0.0)
                return loaded

            with mock.patch.object(
                artifacts_module, "read_holden_database",
                side_effect=signed_zero_reader,
            ):
                with self.assertRaisesRegex(ValueError, "positions"):
                    publish_artifacts(
                        output, artifacts, tiny_manifest_base(artifacts),
                        tiny_scene_pack(), lambda path: None,
                    )
            self.assertEqual(os.listdir(output), ["old"])
            self.assertEqual(self._scratch(temporary, output), [])

    def test_sha256_file_streams_without_whole_file_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "payload")
            payload = b"stream-me" * 1000
            with open(path, "wb") as stream:
                stream.write(payload)
            with mock.patch.object(
                artifacts_module, "_read_regular_bytes",
                side_effect=AssertionError("whole-file read"),
            ):
                self.assertEqual(
                    artifacts_module.sha256_file(path),
                    hashlib.sha256(payload).hexdigest(),
                )

    def test_two_thread_publishers_are_serialized_without_cross_rollback(self):
        artifacts = ArtifactSet.empty(4, 2)
        pack = tiny_scene_pack()
        real_exchange = artifacts_module._rename_exchange
        active = 0
        maximum_active = 0
        activity_lock = threading.Lock()
        barrier = threading.Barrier(2)
        results = []
        errors = []
        with tempfile.TemporaryDirectory() as temporary:
            output = self._old_output(temporary)

            def observed_exchange(source, destination):
                nonlocal active, maximum_active
                self.assertTrue(os.path.isdir(source))
                self.assertTrue(os.path.isdir(destination))
                with activity_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    time.sleep(0.05)
                    return real_exchange(source, destination)
                finally:
                    with activity_lock:
                        active -= 1

            def worker(marker):
                try:
                    results.append(publish_artifacts(
                        output, artifacts,
                        tiny_manifest_base(
                            artifacts, diagnostic_mode=marker),
                        pack, lambda path: barrier.wait(timeout=5.0),
                    ))
                except BaseException as error:
                    errors.append(error)

            with mock.patch.object(
                artifacts_module, "_rename_exchange",
                side_effect=observed_exchange,
            ):
                threads = [
                    threading.Thread(target=worker, args=(marker,))
                    for marker in (False, True)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10.0)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(maximum_active, 1)
            with open(os.path.join(output, "manifest.json"), encoding="utf-8") \
                    as stream:
                final_manifest = json.load(stream)
            self.assertIn(final_manifest["diagnostic_mode"], (False, True))
            self.assertEqual(self._scratch(temporary, output), [])


if __name__ == "__main__":
    unittest.main()
