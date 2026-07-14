import copy
import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
import unittest
from functools import lru_cache
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np

from resources import validate_g1_terrain_database as validator_module
from resources.g1_terrain_builder.artifacts import (
    canonical_json_bytes,
    publish_artifacts,
)
from resources.g1_terrain_builder.scenes import (
    GRAIL_DEFAULT_BASE,
    all_scene_definitions,
    build_scene_pack,
)
from resources.g1_terrain_builder.schema import (
    ArtifactSet,
    HoldenClip,
    SkeletonSpec,
)
from resources.g1_terrain_builder.terrain import (
    HeightGrid,
    surface_semantics,
    surface_semantics_signature,
)
from resources.validate_g1_terrain_database import validate_artifact_directory


LOCKED_GRAIL_HEIGHTS = {
    GRAIL_DEFAULT_BASE: 0.2921024334377573,
    "terrain_curbs__curb_186__004": 0.12238701526200782,
    "terrain_curbs__curb_022__001": 0.24007104328948528,
    "terrain_curbs__curb_165__006": 0.3599740964554129,
}


def _fake_grail_clip(base):
    clip = HoldenClip.empty(frames=5, bones=31)
    clip.name = base
    clip.terrain_id = base
    clip.positions[:, 0, 0] = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4], np.float32)
    clip.positions[:, 0, 2] = np.array(
        [0.0, 0.4, 0.8, 1.2, 1.6], np.float32)
    return clip


@lru_cache(maxsize=1)
def _canonical_scene_pack():
    clips = {
        base: _fake_grail_clip(base)
        for base in LOCKED_GRAIL_HEIGHTS
    }
    return build_scene_pack(all_scene_definitions(
        LOCKED_GRAIL_HEIGHTS, clips))


def _fixture_artifacts():
    artifacts = ArtifactSet.empty(frames=3, bones=31)
    artifacts.parents[:] = np.array([
        -1, 0,
        1, 2, 3, 4, 5, 6,
        1, 8, 9, 10, 11, 12,
        1, 14, 15,
        16, 17, 18, 19, 20, 21, 22,
        16, 24, 25, 26, 27, 28, 29,
    ], np.int32)
    artifacts.positions[:, 0, 2] = np.array(
        [0.0, 0.02, 0.04], np.float32)
    artifacts.contacts[:] = np.array([[1, 1], [1, 0], [0, 0]], np.uint8)
    artifacts.terrain_features[:] = np.array(
        [0.0, 0.01, 0.02, 0.03], np.float32)
    artifacts.terrain_support[:] = np.array(
        [0.0, 0.04, 0.04], np.float32)
    return artifacts


def _fixture_manifest(artifacts):
    names = (
        "Simulation", "Hips",
        "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee",
        "LeftAnkle", "LeftToe",
        "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
        "RightAnkle", "RightToe",
        "Spine", "Spine1", "Spine2",
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
        "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
        "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist",
    )
    skeleton = SkeletonSpec(names, artifacts.parents.copy())
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
        "diagnostic_mode": True,
        "sources": [{
            "name": "takara_walk_50hz",
            "terrain_id": "flat",
            "source_fps": 25.0,
            "source_frames": frames,
            "output_frames": frames,
            "range_start": 0,
            "range_stop": frames,
            "source_frame_map": list(range(frames)),
        }],
        "skeleton": {
            "names": list(skeleton.names),
            "parents": skeleton.parents.tolist(),
            "signature": skeleton.signature(),
        },
        "contact": {
            "speed_threshold": 0.15,
            "height_threshold": 0.06,
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


def _publish_fixture(output):
    artifacts = _fixture_artifacts()
    manifest = publish_artifacts(
        output,
        artifacts,
        _fixture_manifest(artifacts),
        _canonical_scene_pack(),
        lambda candidate: None,
    )
    return manifest


def _sha256(path):
    with open(path, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _load_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    with open(path, "wb") as stream:
        stream.write(canonical_json_bytes(value))


def _test_f32(value):
    return float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def _test_f32_add(left, right):
    return _test_f32(np.float32(np.float32(left) + np.float32(right)))


def _test_f32_sub(left, right):
    return _test_f32(np.float32(np.float32(left) - np.float32(right)))


def _test_f32_mul(left, right):
    return _test_f32(np.float32(np.float32(left) * np.float32(right)))


def _test_f32_div(left, right):
    return _test_f32(np.float32(np.float32(left) / np.float32(right)))


def _test_grid_index(grid, x, z):
    gx = _test_f32_div(
        _test_f32_sub(_test_f32(x), _test_f32(grid.origin_x)),
        _test_f32(grid.cell_size))
    gz = _test_f32_div(
        _test_f32_sub(_test_f32(z), _test_f32(grid.origin_z)),
        _test_f32(grid.cell_size))
    ix = min(int(np.floor(np.float32(_test_f32_add(gx, 0.5)))), grid.nx - 1)
    iz = min(int(np.floor(np.float32(_test_f32_add(gz, 0.5)))), grid.nz - 1)
    return ix, iz


def _test_route_samples(grid, points):
    output = [points[0]]
    step = _test_f32_div(grid.cell_size, 2.0)
    for start, stop in zip(points, points[1:]):
        dx = _test_f32_sub(stop[0], start[0])
        dz = _test_f32_sub(stop[1], start[1])
        squared = _test_f32_add(
            _test_f32_mul(dx, dx), _test_f32_mul(dz, dz))
        distance = _test_f32(np.float32(np.sqrt(np.float32(squared))))
        count = max(1, int(np.ceil(np.float32(
            _test_f32_div(distance, step)))))
        for index in range(1, count + 1):
            if index == count:
                output.append(stop)
            else:
                alpha = _test_f32_div(index, count)
                output.append((
                    _test_f32_add(start[0], _test_f32_mul(alpha, dx)),
                    _test_f32_add(start[1], _test_f32_mul(alpha, dz)),
                ))
    return output


def _test_route_covers(grid, points):
    samples = _test_route_samples(grid, points)
    covers = []
    for start, stop in zip(samples, samples[1:]):
        start_ix, start_iz = _test_grid_index(grid, *start)
        stop_ix, stop_iz = _test_grid_index(grid, *stop)
        covers.append({
            (iz, ix)
            for iz in range(min(start_iz, stop_iz), max(start_iz, stop_iz) + 1)
            for ix in range(min(start_ix, stop_ix), max(start_ix, stop_ix) + 1)
        })
    return covers


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._template_temporary = tempfile.TemporaryDirectory()
        cls._template = os.path.join(
            cls._template_temporary.name, "canonical-artifact")
        _publish_fixture(cls._template)

    @classmethod
    def tearDownClass(cls):
        cls._template_temporary.cleanup()

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.output = os.path.join(self._temporary.name, "artifact")
        shutil.copytree(self._template, self.output)

    def tearDown(self):
        self._temporary.cleanup()

    @staticmethod
    def _expected_summary():
        return {
            "frames": 3,
            "clips": 1,
            "bones": 31,
            "scenes": 14,
            "source_rows": 0,
        }

    def _path(self, relative):
        return os.path.join(self.output, *relative.split("/"))

    def _rewrite_json(self, preserve, relative, mutation):
        path = preserve(relative)
        value = _load_json(path)
        mutation(value)
        _write_json(path, value)

    def _resign_scene(self, preserve, scene_id):
        scene_relative = f"scenes/{scene_id}/scene.json"
        scene_path = self._path(scene_relative)
        index_relative = "scenes/index.json"
        index_path = preserve(index_relative)
        index = _load_json(index_path)
        descriptor = next(
            value for value in index["scenes"] if value["id"] == scene_id)
        descriptor["sha256"] = _sha256(scene_path)
        _write_json(index_path, index)
        manifest_path = preserve("manifest.json")
        manifest = _load_json(manifest_path)
        manifest["scene_index"]["sha256"] = _sha256(index_path)
        _write_json(manifest_path, manifest)

    def _resign_asset(self, preserve, scene_id, descriptor_name, filename):
        scene_relative = f"scenes/{scene_id}/scene.json"
        scene_path = preserve(scene_relative)
        scene = _load_json(scene_path)
        scene[descriptor_name]["sha256"] = _sha256(
            self._path(f"scenes/{scene_id}/{filename}"))
        _write_json(scene_path, scene)
        self._resign_scene(preserve, scene_id)

    def _resign_root_payload(self, preserve, descriptor_path):
        manifest_path = preserve("manifest.json")
        manifest = _load_json(manifest_path)
        descriptor = manifest
        for key in descriptor_path:
            descriptor = descriptor[key]
        descriptor["sha256"] = _sha256(descriptor["path"]
            if os.path.isabs(descriptor["path"])
            else self._path(descriptor["path"]))
        _write_json(manifest_path, manifest)

    def _decode_scene(self, scene_id):
        scene = _load_json(self._path(f"scenes/{scene_id}/scene.json"))
        with open(self._path(f"scenes/{scene_id}/terrain.bin"), "rb") as stream:
            terrain = stream.read()
        magic, version, nx, nz, ox, oz, cell, exterior = struct.unpack_from(
            "<4sIII4f", terrain)
        self.assertEqual((magic, version), (b"G1HF", 2))
        heights = np.frombuffer(
            terrain, "<f4", nx * nz, 32).reshape(nz, nx).copy()
        grid = HeightGrid(
            heights, float(ox), float(oz), float(cell), float(exterior))
        with open(
            self._path(f"scenes/{scene_id}/walkability.bin"), "rb"
        ) as stream:
            payload = stream.read()
        wm_magic, wm_version, wm_nx, wm_nz = struct.unpack_from(
            "<4sIII", payload)
        self.assertEqual(
            (wm_magic, wm_version, wm_nx, wm_nz),
            (b"G1WM", 1, nx, nz))
        values = np.frombuffer(
            payload, np.uint8, nx * nz, 16).reshape(nz, nx).copy()
        return scene, grid, values

    def _write_walkability(self, preserve, scene_id, values):
        relative = f"scenes/{scene_id}/walkability.bin"
        path = preserve(relative)
        with open(path, "rb") as stream:
            header = stream.read(16)
        with open(path, "wb") as stream:
            stream.write(header)
            stream.write(np.ascontiguousarray(values, np.uint8).tobytes())
        self._resign_asset(
            preserve, scene_id, "walkability", "walkability.bin")

    @staticmethod
    def _endpoint_indices(grid, x, z):
        ox = np.float32(grid.origin_x)
        oz = np.float32(grid.origin_z)
        cell = np.float32(grid.cell_size)
        xs = np.asarray([
            np.float32(ox + np.float32(np.float32(ix) * cell))
            for ix in range(grid.nx)
        ], np.float32)
        zs = np.asarray([
            np.float32(oz + np.float32(np.float32(iz) * cell))
            for iz in range(grid.nz)
        ], np.float32)
        dx = np.float32(xs[np.newaxis, :] - np.float32(x))
        dz = np.float32(zs[:, np.newaxis] - np.float32(z))
        radius = np.float32(0.20)
        limit = np.float32(
            np.float32(radius * radius) + np.float32(1e-8))
        mask = np.float32(
            np.float32(dx * dx) + np.float32(dz * dz)) <= limit
        return set(zip(*np.nonzero(mask)))

    def _reject(self, label, message, mutation):
        originals = {}

        def preserve(relative):
            path = self._path(relative)
            if relative not in originals:
                if os.path.lexists(path) and not os.path.isdir(path):
                    if os.path.islink(path):
                        originals[relative] = ("link", os.readlink(path))
                    else:
                        with open(path, "rb") as stream:
                            originals[relative] = ("file", stream.read())
                elif os.path.isdir(path):
                    originals[relative] = ("directory", None)
                else:
                    originals[relative] = ("missing", None)
            return path

        try:
            mutation(preserve)
            with self.subTest(case=label):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    validate_artifact_directory(self.output)
        finally:
            for relative, (kind, payload) in reversed(tuple(originals.items())):
                path = self._path(relative)
                if os.path.lexists(path):
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path)
                    else:
                        os.unlink(path)
                if kind == "file":
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as stream:
                        stream.write(payload)
                elif kind == "link":
                    os.symlink(payload, path)
                elif kind == "directory":
                    os.makedirs(path)
            with self.subTest(case=label + "-restored-baseline"):
                self.assertEqual(
                    validate_artifact_directory(self.output),
                    self._expected_summary())

    def test_direct_canonical_v2_fixture_validates_normally(self):
        self.assertEqual(
            validate_artifact_directory(self.output), self._expected_summary())

    def test_full_source_true_fails_as_explicitly_unimplemented(self):
        with self.assertRaisesRegex(ValueError, "not implemented.*Task 12B"):
            validate_artifact_directory(
                self.output, full_source_validation=True)

    def test_v2_cli_reports_the_single_dispatched_schema_and_dimensions(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = validator_module.main([self.output])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            "VALID g1-terrain-artifacts/v2 frames=3 clips=1 bones=31 "
            "terrain_dims=4 support_dims=3 scenes=14 source_rows=0\n")

    def test_json_typing_and_canonical_encoding_are_strict(self):
        def output_fps_is_int(preserve):
            self._rewrite_json(
                preserve, "manifest.json",
                lambda value: value.__setitem__("output_fps", 25))

        self._reject("int-instead-of-float", "output_fps", output_fps_is_int)

        def contact_filter_is_bool(preserve):
            self._rewrite_json(
                preserve, "manifest.json",
                lambda value: value["contact"].__setitem__(
                    "median_filter_frames", True))

        self._reject("bool-instead-of-int", "contact", contact_filter_is_bool)

        def validation_metric_is_int(preserve):
            manifest_path = preserve("manifest.json")
            validation_path = preserve("validation.json")
            manifest = _load_json(manifest_path)
            manifest["validation"]["fk_max_error_m"][0] = 0
            validation = copy.deepcopy(manifest["validation"])
            _write_json(validation_path, validation)
            manifest["validation_file"]["sha256"] = _sha256(validation_path)
            _write_json(manifest_path, manifest)

        self._reject(
            "metric-int-instead-of-float", "finite JSON float",
            validation_metric_is_int)

        def duplicate_manifest_key(preserve):
            with open(preserve("manifest.json"), "wb") as stream:
                stream.write(b'{"schema":"x","schema":"y"}\n')

        self._reject(
            "duplicate-json-key", "duplicate JSON key", duplicate_manifest_key)

        def nonfinite_manifest(preserve):
            path = preserve("manifest.json")
            with open(path, "rb") as stream:
                payload = stream.read()
            payload = payload.replace(
                b'"output_fps": 25.0', b'"output_fps": NaN')
            with open(path, "wb") as stream:
                stream.write(payload)

        self._reject("nonfinite-json", "non-finite JSON", nonfinite_manifest)

        def noncanonical_manifest(preserve):
            path = preserve("manifest.json")
            with open(path, "ab") as stream:
                stream.write(b" ")

        self._reject(
            "noncanonical-json", "not canonical", noncanonical_manifest)

    def test_source_frames_are_bounded_before_float_or_numpy_conversion(self):
        def huge_source_frames(preserve):
            self._rewrite_json(
                preserve, "manifest.json",
                lambda value: value["sources"][0].__setitem__(
                    "source_frames", 10 ** 300))

        self._reject(
            "huge-source-frames", "source_frames exceeds.*input bound",
            huge_source_frames)

    def test_paths_tree_hashes_database_and_sidecars_are_authenticated(self):
        def unsafe_scene_path(preserve):
            scene_id = "stairs-shallow"
            relative = f"scenes/{scene_id}/scene.json"
            self._rewrite_json(
                preserve, relative,
                lambda value: value["heightfield"].__setitem__(
                    "path", "../terrain.bin"))
            self._resign_scene(preserve, scene_id)

        self._reject("unsafe-relative-path", "path is unsafe", unsafe_scene_path)

        def database_symlink(preserve):
            path = preserve("database.bin")
            os.unlink(path)
            os.symlink("terrain_features.bin", path)

        self._reject("symlink", "symlink", database_symlink)

        def extra_file(preserve):
            path = preserve("unexpected.bin")
            with open(path, "wb") as stream:
                stream.write(b"x")

        self._reject("extra-tree-node", "unexpected file", extra_file)

        def extra_directory(preserve):
            os.mkdir(preserve("unexpected-directory"))

        self._reject(
            "extra-directory", "unexpected directory", extra_directory)

        def fifo_node(preserve):
            os.mkfifo(preserve("unexpected-fifo"))

        self._reject(
            "fifo-node", "non-regular node", fifo_node)

        def database_hash_corruption(preserve):
            path = preserve("database.bin")
            with open(path, "r+b") as stream:
                stream.seek(100)
                byte = stream.read(1)
                stream.seek(100)
                stream.write(bytes([byte[0] ^ 1]))

        self._reject(
            "database-hash", "database SHA-256 mismatch",
            database_hash_corruption)

        def coordinated_support_nan(preserve):
            path = preserve("terrain_support.bin")
            with open(path, "r+b") as stream:
                stream.seek(-4, os.SEEK_END)
                stream.write(struct.pack("<f", float("nan")))
            manifest_path = preserve("manifest.json")
            manifest = _load_json(manifest_path)
            manifest["sidecars"]["terrain_support"]["sha256"] = _sha256(path)
            _write_json(manifest_path, manifest)

        self._reject(
            "coordinated-support", "finite", coordinated_support_nan)

        def coordinated_contact_value(preserve):
            path = preserve("database.bin")
            with open(path, "r+b") as stream:
                stream.seek(-1, os.SEEK_END)
                stream.write(b"\x02")
            manifest_path = preserve("manifest.json")
            manifest = _load_json(manifest_path)
            manifest["database"]["sha256"] = _sha256(path)
            _write_json(manifest_path, manifest)

        self._reject(
            "coordinated-contact", "contacts.*binary|values must be 0 or 1",
            coordinated_contact_value)

        def attacker_database_header(preserve):
            path = preserve("database.bin")
            with open(path, "r+b") as stream:
                stream.write(struct.pack("<I", 0xffffffff))
            manifest_path = preserve("manifest.json")
            manifest = _load_json(manifest_path)
            manifest["database"]["sha256"] = _sha256(path)
            _write_json(manifest_path, manifest)

        self._reject(
            "attacker-database-header", "database positions dimensions",
            attacker_database_header)

    def test_heightfield_and_walkability_headers_and_payloads_are_bounded(self):
        scene_id = "stairs-shallow"

        def corrupt_heightfield_magic(preserve):
            path = preserve(f"scenes/{scene_id}/terrain.bin")
            with open(path, "r+b") as stream:
                stream.write(b"BAD!")
            self._resign_asset(
                preserve, scene_id, "heightfield", "terrain.bin")

        self._reject(
            "G1HF-magic", "G1HF/v2", corrupt_heightfield_magic)

        def attacker_heightfield_dimensions(preserve):
            path = preserve(f"scenes/{scene_id}/terrain.bin")
            with open(path, "r+b") as stream:
                stream.seek(8)
                stream.write(struct.pack("<I", 0xffffffff))
            self._resign_asset(
                preserve, scene_id, "heightfield", "terrain.bin")

        self._reject(
            "G1HF-huge-dimension", "axis exceeds|overflowing|payload",
            attacker_heightfield_dimensions)

        def corrupt_heightfield_payload(preserve):
            path = preserve(f"scenes/{scene_id}/terrain.bin")
            with open(path, "r+b") as stream:
                stream.seek(32)
                stream.write(struct.pack("<f", float("nan")))
            self._resign_asset(
                preserve, scene_id, "heightfield", "terrain.bin")

        self._reject(
            "G1HF-nonfinite-payload", "invalid binary32",
            corrupt_heightfield_payload)

        def corrupt_walkability_header(preserve):
            path = preserve(f"scenes/{scene_id}/walkability.bin")
            with open(path, "r+b") as stream:
                stream.seek(8)
                stream.write(struct.pack("<I", 0xffffffff))
            self._resign_asset(
                preserve, scene_id, "walkability", "walkability.bin")

        self._reject(
            "G1WM-huge-dimension", "axis exceeds|overflowing|payload",
            corrupt_walkability_header)

        def corrupt_walkability_class(preserve):
            path = preserve(f"scenes/{scene_id}/walkability.bin")
            with open(path, "r+b") as stream:
                stream.seek(16)
                stream.write(b"\x03")
            self._resign_asset(
                preserve, scene_id, "walkability", "walkability.bin")

        self._reject(
            "G1WM-invalid-class", "invalid G1WM class",
            corrupt_walkability_class)

        def semantically_harmless_obj_change(preserve):
            path = preserve(f"scenes/{scene_id}/terrain.obj")
            with open(path, "rb") as stream:
                payload = stream.read()
            with open(path, "wb") as stream:
                stream.write(b"# harmless but noncanonical\n" + payload)
            self._resign_asset(preserve, scene_id, "mesh", "terrain.obj")

        self._reject(
            "exact-OBJ-bytes", "exact fixed-diagonal heightfield mesh",
            semantically_harmless_obj_change)

    def test_endpoint_footprint_uses_exact_float32_radius_allowance(self):
        def from_bits(bits):
            return struct.unpack("<f", struct.pack("<I", bits))[0]

        def bits(value):
            return struct.unpack("<I", struct.pack("<f", value))[0]

        radius = from_bits(0x3e4ccccd)
        dx = from_bits(0x3e4cccce)
        radius_squared = np.float32(np.float32(radius) * np.float32(radius))
        distance_squared = np.float32(np.float32(dx) * np.float32(dx))
        limit = np.float32(radius_squared + np.float32(1e-8))
        self.assertEqual(bits(radius_squared), 0x3d23d70b)
        self.assertEqual(bits(distance_squared), 0x3d23d70c)
        self.assertEqual(bits(limit), 0x3d23d70e)
        self.assertGreater(distance_squared, radius_squared)
        self.assertLessEqual(distance_squared, limit)

        grid = HeightGrid(
            np.zeros((3, 4), np.float32),
            -float(dx), -float(dx), float(dx), 0.0)
        classes = np.ones((3, 4), np.uint8)
        classes[1, 2] = 2
        observed = validator_module._endpoint_footprint_classes(
            grid, classes, 0.0, 0.0)
        self.assertEqual(observed, {1, 2})

    def test_region_route_endpoint_and_grail_classifier_oracles_are_exact(self):
        def region_offset_alias(preserve):
            scene_id = "stairs-shallow"
            relative = f"scenes/{scene_id}/scene.json"

            def mutate(scene):
                scene["regions"]["certified"][0]["bounds_xz"][1] = \
                    float(np.float32(0.851))

            self._rewrite_json(preserve, relative, mutate)
            self._resign_scene(preserve, scene_id)

        self._reject(
            "region-offset-alias", "region disagrees with G1WM",
            region_offset_alias)

        def diagonal_corner_graze(preserve):
            scene_id = "grail-curb-low"
            scene, grid, values = self._decode_scene(scene_id)
            start = tuple(scene["routes"][0]["waypoints_xz"][0])
            start_ix, start_iz = _test_grid_index(grid, *start)
            diagonal_stop = (
                float(np.float32(
                    grid.origin_x + (start_ix + 1) * grid.cell_size)),
                float(np.float32(
                    grid.origin_z + (start_iz + 1) * grid.cell_size)),
            )
            scene["routes"][0]["waypoints_xz"][1] = list(diagonal_stop)
            points = tuple(tuple(point) for point in
                           scene["routes"][0]["waypoints_xz"])
            covers = _test_route_covers(grid, points)
            samples = _test_route_samples(grid, points)
            sampled = {
                (iz, ix)
                for ix, iz in (
                    _test_grid_index(grid, *point)
                    for point in samples)
            }
            endpoints = self._endpoint_indices(grid, *points[0]) \
                | self._endpoint_indices(grid, *points[-1])
            candidates = sorted(
                set().union(*(set(cover) for cover in covers))
                - sampled - endpoints)
            self.assertTrue(candidates)
            iz, ix = next(
                value for value in candidates if values[value] == 1)
            values[iz, ix] = 0
            playable_xmax = scene["bounds"]["playable_max_xz"][0]
            playable_zmax = scene["bounds"]["playable_max_xz"][1]
            scene["regions"]["certified"][0]["bounds_xz"] = [
                float(np.float32(playable_xmax - 0.10)),
                float(np.float32(playable_xmax - 0.08)),
                float(np.float32(playable_zmax - 0.10)),
                float(np.float32(playable_zmax - 0.08)),
            ]
            scene_path = preserve(f"scenes/{scene_id}/scene.json")
            _write_json(scene_path, scene)
            self._write_walkability(preserve, scene_id, values)

        self._reject(
            "diagonal-corner-graze", "expected route enters wrong",
            diagonal_corner_graze)

        def endpoint_only_halo_cell(preserve):
            scene_id = "grail-curb-low"
            scene, grid, values = self._decode_scene(scene_id)
            points = tuple(tuple(point) for point in
                           scene["routes"][0]["waypoints_xz"])
            route_cells = set().union(*(
                set(cover) for cover in
                _test_route_covers(grid, points)))
            endpoint_cells = self._endpoint_indices(grid, *points[0])
            candidates = sorted(endpoint_cells - route_cells)
            iz, ix = next(
                value for value in candidates
                if values[value] == 1
                and (grid.origin_x + value[1] * grid.cell_size) < -0.08)
            values[iz, ix] = 0
            scene_relative = f"scenes/{scene_id}/scene.json"
            scene_path = preserve(scene_relative)
            scene["regions"]["certified"][0]["bounds_xz"] = [
                float(np.float32(-0.05)), float(np.float32(0.45)),
                float(np.float32(-0.05)), float(np.float32(1.65)),
            ]
            _write_json(scene_path, scene)
            self._write_walkability(preserve, scene_id, values)

        self._reject(
            "endpoint-only-halo-cell", "endpoint footprint enters wrong",
            endpoint_only_halo_cell)

        def unused_grail_classifier_cell(preserve):
            scene_id = "grail-curb-low"
            scene, grid, values = self._decode_scene(scene_id)
            playable_xmax = scene["bounds"]["playable_max_xz"][0]
            playable_zmin = scene["bounds"]["playable_min_xz"][1]
            playable_zmax = scene["bounds"]["playable_max_xz"][1]
            target_x = float(np.float32(playable_xmax + np.float32(0.20)))
            target_z = float(np.float32(
                0.5 * (playable_zmin + playable_zmax)))
            ix, iz = _test_grid_index(grid, target_x, target_z)
            self.assertEqual(values[iz, ix], 1)
            values[iz, ix] = 0
            self._write_walkability(preserve, scene_id, values)

        self._reject(
            "unused-GRAIL-cell", "complete GRAIL classifier",
            unused_grail_classifier_cell)

        def coordinated_grail_heightfield_parity(preserve):
            scene_id = "grail-curb-low"
            _, grid, _ = self._decode_scene(scene_id)
            heights = grid.heights.copy()
            minimum = np.float32(heights.min())
            maximum = np.float32(heights.max())
            margin = np.float32(0.002)
            candidates = np.argwhere(
                (heights > np.float32(minimum + margin))
                & (heights < np.float32(maximum - margin)))
            self.assertTrue(len(candidates))
            iz, ix = candidates[len(candidates) // 2]
            original = np.float32(heights[iz, ix])
            heights[iz, ix] = np.float32(original + np.float32(0.001))
            self.assertGreater(
                abs(float(heights[iz, ix]) - float(original)), 1e-6)
            self.assertEqual(np.float32(heights.min()), minimum)
            self.assertEqual(np.float32(heights.max()), maximum)
            changed = HeightGrid(
                heights, grid.origin_x, grid.origin_z,
                grid.cell_size, grid.exterior_height)
            terrain_path = preserve(f"scenes/{scene_id}/terrain.bin")
            obj_path = preserve(f"scenes/{scene_id}/terrain.obj")
            with open(terrain_path, "wb") as stream:
                stream.write(changed.g1hf_bytes())
            with open(obj_path, "wb") as stream:
                stream.write(changed.obj_bytes())
            self._resign_asset(
                preserve, scene_id, "heightfield", "terrain.bin")
            self._resign_asset(preserve, scene_id, "mesh", "terrain.obj")

        self._reject(
            "GRAIL-heightfield-parity", "node surface parity failed",
            coordinated_grail_heightfield_parity)

    def test_procedural_traverse_waypoints_keep_one_meter_grid_margin(self):
        def route_lacks_margin(preserve):
            scene_id = "stairs-shallow"
            scene, grid, _ = self._decode_scene(scene_id)
            scene["routes"][0]["waypoints_xz"][1][0] = float(np.float32(
                np.float32(grid.origin_x) + np.float32(0.99)))
            _write_json(
                preserve(f"scenes/{scene_id}/scene.json"), scene)
            self._resign_scene(preserve, scene_id)

        self._reject(
            "procedural-route-margin",
            "certified route lacks 1 m lookahead margin",
            route_lacks_margin)

    def test_grail_source_height_lock_is_independent_of_loaded_source(self):
        canonical_from_base = validator_module.GrailTerrain.from_base

        def drifted_from_base(base):
            terrain = canonical_from_base(base)
            footprint = dict(terrain.footprint())
            footprint["height"] = float(footprint["height"]) + 0.001

            class DriftedTerrain:
                @staticmethod
                def footprint():
                    return footprint

            return DriftedTerrain()

        with mock.patch.object(
            validator_module.GrailTerrain, "from_base",
            side_effect=drifted_from_base,
        ):
            with self.assertRaisesRegex(
                ValueError, "canonical GRAIL source height changed",
            ):
                validate_artifact_directory(self.output)
        self.assertEqual(
            validate_artifact_directory(self.output), self._expected_summary())

    def test_safe_stop_grail_provenance_and_procedural_bytes_are_locked(self):
        def safe_stop_reentry(preserve):
            scene_id = "blocked-course"
            scene, grid, values = self._decode_scene(scene_id)
            route = scene["routes"][0]
            points = tuple(tuple(point) for point in route["waypoints_xz"])
            covers = _test_route_covers(grid, points)
            class_sets = [
                {int(values[index]) for index in cover}
                for cover in covers
            ]
            transition = class_sets.index({0, 1})
            candidate = next(
                index for index in covers[transition + 5]
                if values[index] == 0)
            values[candidate] = 1
            # Retain three generic regions but move blocked witnesses away from
            # the changed route cell so the route oracle owns this rejection.
            blocked = np.argwhere(values == 0)
            scene_path = preserve(f"scenes/{scene_id}/scene.json")
            for entry, (iz, ix) in zip(
                    scene["regions"]["blocked"], blocked[:2]):
                x = float(np.float32(grid.origin_x + ix * grid.cell_size))
                z = float(np.float32(grid.origin_z + iz * grid.cell_size))
                entry["bounds_xz"] = [
                    x, float(np.nextafter(np.float32(x), np.float32(np.inf))),
                    z, float(np.nextafter(np.float32(z), np.float32(np.inf))),
                ]
            _write_json(scene_path, scene)
            self._write_walkability(preserve, scene_id, values)

        self._reject(
            "safe-stop-reentry", "safe-stop route", safe_stop_reentry)

        def safe_stop_transition_removed(preserve):
            scene_id = "blocked-course"
            scene, grid, values = self._decode_scene(scene_id)
            points = tuple(tuple(point) for point in
                           scene["routes"][0]["waypoints_xz"])
            covers = _test_route_covers(grid, points)
            class_sets = [{int(values[index]) for index in cover}
                          for cover in covers]
            transition = class_sets.index({0, 1})
            extra_transition = next(
                cover for cover in covers[5:transition - 5]
                if len(cover) >= 2
                and all(values[index] == 1 for index in cover))
            values[next(iter(extra_transition))] = 0
            scene_path = preserve(f"scenes/{scene_id}/scene.json")
            certified = np.argwhere(values == 1)[len(np.argwhere(values == 1)) // 2]
            blocked = np.argwhere(values == 0)
            witnesses = [certified, blocked[0], blocked[len(blocked) // 2]]
            entries = [
                scene["regions"]["certified"][0],
                *scene["regions"]["blocked"],
            ]
            for entry, (iz, ix) in zip(entries, witnesses):
                x = float(np.float32(grid.origin_x + ix * grid.cell_size))
                z = float(np.float32(grid.origin_z + iz * grid.cell_size))
                entry["bounds_xz"] = [
                    x, float(np.nextafter(np.float32(x), np.float32(np.inf))),
                    z, float(np.nextafter(np.float32(z), np.float32(np.inf))),
                ]
            _write_json(scene_path, scene)
            self._write_walkability(preserve, scene_id, values)

        self._reject(
            "safe-stop-transition", "safe-stop route",
            safe_stop_transition_removed)

        def wrong_grail_provenance(preserve):
            scene_id = "grail-curb-low"
            relative = f"scenes/{scene_id}/scene.json"
            self._rewrite_json(
                preserve, relative,
                lambda scene: scene["provenance"]["source_ids"].__setitem__(
                    1, "wrong-source"))
            self._resign_scene(preserve, scene_id)

        self._reject(
            "wrong-GRAIL-provenance", "GRAIL provenance",
            wrong_grail_provenance)

        def wrong_grail_parameter(preserve):
            scene_id = "grail-curb-low"
            relative = f"scenes/{scene_id}/scene.json"
            self._rewrite_json(
                preserve, relative,
                lambda scene: scene["provenance"]["parameters"].__setitem__(
                    "target_height_m", 0.13))
            self._resign_scene(preserve, scene_id)

        self._reject(
            "wrong-GRAIL-parameter", "GRAIL provenance",
            wrong_grail_parameter)

        def wrong_grail_class(preserve):
            scene_id = "grail-curb-low"
            relative = f"scenes/{scene_id}/scene.json"

            def mutate(scene):
                scene["routes"][0]["expected_outcome"] = \
                    "traverse-or-safe-stop"
                scene["routes"][0]["walkability_class"] = 2

            self._rewrite_json(preserve, relative, mutate)
            self._resign_scene(preserve, scene_id)

        self._reject(
            "wrong-GRAIL-class", "expected route enters wrong",
            wrong_grail_class)

        def procedural_byte_mismatch(preserve):
            scene_id = "stairs-shallow"
            relative = f"scenes/{scene_id}/scene.json"
            self._rewrite_json(
                preserve, relative,
                lambda scene: scene.__setitem__("label", "Changed Label"))
            self._resign_scene(preserve, scene_id)

        self._reject(
            "procedural-byte-mismatch", "deterministic procedural bytes",
            procedural_byte_mismatch)


if __name__ == "__main__":
    unittest.main()
