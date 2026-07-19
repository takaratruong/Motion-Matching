import copy
import gc
import hashlib
import io
import inspect
import json
import os
import shutil
import struct
import tempfile
import types
import unittest
import weakref
from functools import lru_cache
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np

from resources import validate_g1_terrain_database as validator_module
from resources.g1_terrain_builder.artifacts import (
    canonical_json_bytes,
    publish_artifacts,
)
from resources.g1_terrain_builder.motion_index import (
    DIRECTION_FORWARD,
    DIRECTION_IDLE,
    MotionIndex,
    SPEED_LOW,
    SPEED_MOVING,
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
    SourceFrameRange,
    TERRAIN_FAMILIES,
    TerrainBank,
    TerrainBankIndex,
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
LOCKED_GRAIL_ROOT_PATH_X_BOUNDS = {
    GRAIL_DEFAULT_BASE: (
        -0.16539472341537476, -0.009255850687623024),
    "terrain_curbs__curb_186__004": (
        -0.28973305225372314, -0.02736859768629074),
    "terrain_curbs__curb_022__001": (
        -0.39980870485305786, -0.06907640397548676),
    "terrain_curbs__curb_165__006": (
        -0.29937341809272766, -0.030186962336301804),
}
SMALL_FULL_GRAIL_BASES = tuple(sorted(LOCKED_GRAIL_HEIGHTS))
SMALL_EXTRA_GRAIL_BASES = (
    "terrain_curbs__curb_199__006",
    "terrain_curbs__curb_199__007",
)
SMALL_STREAM_GRAIL_BASES = tuple(sorted(
    SMALL_FULL_GRAIL_BASES + SMALL_EXTRA_GRAIL_BASES))


def _fake_grail_clip(base):
    clip = HoldenClip.empty(frames=5, bones=31)
    clip.name = base
    clip.terrain_id = base
    clip.positions[:, 0, 0] = np.linspace(
        *LOCKED_GRAIL_ROOT_PATH_X_BOUNDS[base], 5, dtype=np.float32)
    clip.positions[:, 0, 2] = np.array(
        [0.0, 0.4, 0.8, 1.2, 1.6], np.float32)
    return clip


def _small_full_source_case(grail_bases=SMALL_FULL_GRAIL_BASES):
    frames_per_clip = 3
    total_clips = 1 + len(grail_bases)
    total_frames = total_clips * frames_per_clip
    database = ArtifactSet.empty(frames=total_frames, bones=31)
    starts = np.arange(
        0, total_frames, frames_per_clip, dtype=np.int32)
    database.range_starts = starts
    database.range_stops = starts + np.int32(frames_per_clip)
    sources = [{
        "name": "takara_walk_50hz",
        "terrain_id": "flat",
        "terrain_family": "flat",
        "source_fps": 50.0,
        "source_frames": 5,
        "output_frames": frames_per_clip,
        "range_start": 0,
        "range_stop": frames_per_clip,
        "source_frame_map": [0, 2, 4],
    }]
    for offset, base in enumerate(grail_bases, 1):
        start = offset * frames_per_clip
        sources.append({
            "name": base,
            "terrain_id": base,
            "terrain_family": "curb",
            "source_fps": 25.0,
            "source_frames": 3,
            "output_frames": frames_per_clip,
            "range_start": start,
            "range_stop": start + frames_per_clip,
            "source_frame_map": [0, 1, 2],
        })
    manifest = {
        "diagnostic_mode": False,
        "total_clips": total_clips,
        "grail_clips": len(grail_bases),
        "database_frames": total_frames,
        "sources": sources,
        "skeleton": {
            "names": list(validator_module.G1_SKELETON_NAMES),
            "parents": list(validator_module.G1_SKELETON_PARENTS),
        },
        "validation": {
            "fk_max_error_m": [0.0] * total_clips,
            "duration_error_s": [0.0] * total_clips,
            "quaternion_norm_max_error": [0.0] * total_clips,
        },
    }
    return manifest, database


def _small_full_constant_patches(grail_bases=SMALL_FULL_GRAIL_BASES):
    grail_rows = len(grail_bases) * 3
    return {
        "FULL_SOURCE_GRAIL_CLIPS": len(grail_bases),
        "FULL_SOURCE_TOTAL_CLIPS": 1 + len(grail_bases),
        "FULL_SOURCE_ROWS": 3 + grail_rows,
        "FULL_SOURCE_GRAIL_ROWS": grail_rows,
        "FULL_SOURCE_TAKARA_SOURCE_FRAMES": 5,
        "FULL_SOURCE_TAKARA_OUTPUT_FRAMES": 3,
        "FULL_SOURCE_GRAIL_SOURCE_FRAMES": 3,
        "FULL_SOURCE_GRAIL_OUTPUT_FRAMES": 3,
    }


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
    artifacts.terrain_features[:] = np.arange(12, dtype=np.float32) * 0.01
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
        "schema": "g1-terrain-artifacts/v3",
        "output_fps": 25.0,
        "feature_dimensions": 39,
        "terrain_dimensions": 12,
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
            "terrain_family": "flat",
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


def _fixture_indexes(artifacts, manifest):
    ranges = tuple(
        SourceFrameRange(
            source["name"], 0, source["output_frames"],
            source["output_frames"], source["range_start"],
            source["range_stop"],
        )
        for source in manifest["sources"]
    )
    family_indices = {family: [] for family in TERRAIN_FAMILIES}
    for index, source in enumerate(manifest["sources"]):
        family_indices[source["terrain_family"]].append(index)
    banks = tuple(
        TerrainBank(family, tuple(family_indices[family]))
        for family in TERRAIN_FAMILIES
    )
    frames = len(artifacts.positions)
    directions = np.full(frames, DIRECTION_IDLE, np.uint16)
    speeds = np.full(frames, SPEED_LOW, np.uint8)
    if frames > 1:
        directions[1] = DIRECTION_FORWARD
        speeds[1] = SPEED_MOVING
    return (
        MotionIndex(directions, speeds, np.zeros(frames, np.int8)),
        TerrainBankIndex(frames, ranges, banks),
    )


def _publish_fixture(output):
    artifacts = _fixture_artifacts()
    manifest_base = _fixture_manifest(artifacts)
    motion_index, terrain_banks = _fixture_indexes(
        artifacts, manifest_base)
    manifest = publish_artifacts(
        output,
        artifacts,
        manifest_base,
        _canonical_scene_pack(),
        motion_index,
        terrain_banks,
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


@unittest.skip(
    "Task 7 must replace the legacy single-curb full-source reconstruction"
)
class FullSourceSeamTests(unittest.TestCase):
    def test_source_options_merge_defaults_and_reject_bad_keys_and_values(self):
        expected = {
            "grail_glob": "/home/ubuntu/datasets/GRAIL/data/curb/robot/*.pkl",
            "g1_xml": (
                "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/"
                "g1_29dof.xml"),
            "takara": (
                "/home/ubuntu/Downloads/takara_walk_50hz.npz_v0/"
                "motion.npz"),
            "remap": "/home/ubuntu/projects/g1_mm/isaac_to_mj.npy",
        }
        self.assertEqual(
            validator_module._validate_full_source_options(None), expected)
        replacement = "/tmp/alternate-g1.xml"
        merged = validator_module._validate_full_source_options({
            "g1_xml": replacement,
        })
        self.assertEqual(
            merged, {**expected, "g1_xml": replacement})

        bad_options = (
            [],
            {"unknown": "value"},
            {"g1_xml": True},
            {"g1_xml": 1},
            {"g1_xml": b"path"},
            {"g1_xml": None},
            {"g1_xml": ""},
        )
        for value in bad_options:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    "source_options|unknown full-source option|non-empty string",
                ):
                    validator_module._validate_full_source_options(value)

    def test_float32_and_uint8_row_comparators_are_encoded_exact(self):
        published = np.array([[0.0, 1.0], [-2.0, 3.0]], np.float32)
        validator_module._compare_f32_rows(
            "positions", published, published.copy())

        signed_zero = published.copy()
        signed_zero[0, 0] = np.float32(-0.0)
        with self.assertRaisesRegex(ValueError, "positions.*float32 bits differ"):
            validator_module._compare_f32_rows(
                "positions", published, signed_zero)

        one_ulp = published.copy()
        one_ulp.view(np.uint32)[0, 1] += np.uint32(1)
        with self.assertRaisesRegex(ValueError, "positions.*float32 bits differ"):
            validator_module._compare_f32_rows("positions", published, one_ulp)

        with self.assertRaisesRegex(ValueError, "positions.*float32 dtype"):
            validator_module._compare_f32_rows(
                "positions", published, published.astype(np.float64))
        with self.assertRaisesRegex(ValueError, "positions.*shape"):
            validator_module._compare_f32_rows(
                "positions", published, published[:1])
        with self.assertRaisesRegex(ValueError, "positions.*C-contiguous"):
            validator_module._compare_f32_rows(
                "positions", published[:, ::-1], published[:, ::-1])

        contacts = np.array([[0, 1], [1, 0]], np.uint8)
        validator_module._compare_u8_rows(
            "contacts", contacts, contacts.copy())
        changed = contacts.copy()
        changed[1, 1] = 1
        with self.assertRaisesRegex(ValueError, "contacts.*uint8 rows differ"):
            validator_module._compare_u8_rows("contacts", contacts, changed)
        with self.assertRaisesRegex(ValueError, "contacts.*uint8 dtype"):
            validator_module._compare_u8_rows(
                "contacts", contacts, contacts.astype(np.int64))

    def test_recompute_clip_rebuilds_every_derived_channel(self):
        frames = 3
        bones = 31
        names = tuple(validator_module.G1_SKELETON_NAMES)
        parents = np.asarray(
            validator_module.G1_SKELETON_PARENTS, np.int32)
        skeleton = SkeletonSpec(names, parents)
        clip = HoldenClip.empty(frames, bones)
        report = {
            "fk_max_error_m": 0.0,
            "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }
        global_positions = np.zeros((frames, bones, 3), np.float32)
        global_rotations = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], np.float32),
            (frames, bones, 1))
        velocities = np.full_like(clip.positions, np.float32(1.25))
        angular = np.full_like(clip.positions, np.float32(-0.5))
        contacts = np.array([[1, 0], [1, 1], [0, 1]], np.uint8)
        support = np.array([
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ], np.float32)
        features = [
            np.full(4, np.float32(frame + 1), np.float32)
            for frame in range(frames)
        ]
        source = types.SimpleNamespace(name="clip")
        terrain = object()
        kinematics = object()
        contact_config = object()
        convert = mock.Mock(return_value=(clip, skeleton, report))
        forward = mock.Mock(return_value=(
            global_positions, global_rotations))
        derive = mock.Mock(return_value=(velocities, angular))
        derive_contact = mock.Mock(return_value=contacts)
        sample_support = mock.Mock(return_value=support)
        build_centerline = mock.Mock(
            return_value=np.array([[0.0, 0.0], [0.0, 1.0]]))
        sample_features = mock.Mock(side_effect=features)
        mul_vec = mock.Mock(return_value=np.tile(
            np.array([0.0, 0.0, 1.0]), (frames, 1)))

        with mock.patch.multiple(
            validator_module,
            convert_source_clip=convert,
            forward_kinematics_arrays=forward,
            derive_velocities=derive,
            derive_contacts=derive_contact,
            sample_terrain_support=sample_support,
            build_facing_centerline=build_centerline,
            sample_terrain_features=sample_features,
            ContactConfig=mock.Mock(return_value=contact_config),
            holden_quat=types.SimpleNamespace(mul_vec=mul_vec),
            create=True,
        ):
            rebuilt, observed_report = validator_module._recompute_clip(
                source, terrain, kinematics, names, parents)

        self.assertIs(rebuilt, clip)
        self.assertIs(observed_report, report)
        np.testing.assert_array_equal(rebuilt.velocities, velocities)
        np.testing.assert_array_equal(rebuilt.angular_velocities, angular)
        np.testing.assert_array_equal(rebuilt.contacts, contacts)
        np.testing.assert_array_equal(rebuilt.terrain_support, support)
        np.testing.assert_array_equal(
            rebuilt.terrain_features, np.asarray(features, np.float32))
        convert.assert_called_once_with(source, kinematics, 25.0)
        derive_contact.assert_called_once_with(
            global_positions, terrain, 7, 13, 25.0, contact_config)
        sample_support.assert_called_once_with(
            global_positions, terrain, 0, 7, 13)
        self.assertEqual(sample_features.call_count, frames)

    def test_recompute_clip_checks_skeleton_names_and_parents_independently(self):
        names = tuple(validator_module.G1_SKELETON_NAMES)
        parents = np.asarray(
            validator_module.G1_SKELETON_PARENTS, np.int32)
        clip = HoldenClip.empty(3, 31)
        report = {}
        cases = (
            (SkeletonSpec(("Wrong",) + names[1:], parents),
             "skeleton names"),
            (SkeletonSpec(names, np.arange(-1, 30, dtype=np.int32)),
             "skeleton parents"),
        )
        for skeleton, message in cases:
            with self.subTest(message=message), mock.patch.object(
                validator_module, "convert_source_clip",
                return_value=(clip, skeleton, report), create=True,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    validator_module._recompute_clip(
                        types.SimpleNamespace(name="clip"),
                        object(), object(), names, parents)

    def test_validator_has_no_finalize_clip_dependency(self):
        source = inspect.getsource(validator_module)
        self.assertNotIn("finalize_clip", source)
        self.assertNotIn("resources.build_g1_terrain_database", source)

    def test_rebuilt_clip_row_comparison_covers_every_persisted_channel(self):
        database = ArtifactSet.empty(frames=3, bones=31)
        clip = HoldenClip.empty(frames=3, bones=31)
        validator_module._compare_rebuilt_clip_rows(
            database, 0, 3, clip, "sources[0]")

        float_fields = (
            "positions", "velocities", "rotations", "angular_velocities",
            "terrain_features",
        )
        for name in float_fields:
            changed = getattr(clip, name).copy()
            changed.view(np.uint32).flat[0] += np.uint32(1)
            original = getattr(clip, name)
            setattr(clip, name, changed)
            try:
                with self.subTest(field=name), self.assertRaisesRegex(
                    ValueError, f"{name}.*float32 bits differ",
                ):
                    validator_module._compare_rebuilt_clip_rows(
                        database, 0, 3, clip, "sources[0]")
            finally:
                setattr(clip, name, original)

        for column in range(3):
            changed = clip.terrain_support.copy()
            changed.view(np.uint32)[0, column] += np.uint32(1)
            original = clip.terrain_support
            clip.terrain_support = changed
            try:
                with self.subTest(support_column=column), \
                        self.assertRaisesRegex(
                            ValueError,
                            "terrain_support.*float32 bits differ"):
                    validator_module._compare_rebuilt_clip_rows(
                        database, 0, 3, clip, "sources[0]")
            finally:
                clip.terrain_support = original

        clip.contacts[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "contacts.*uint8 rows differ"):
            validator_module._compare_rebuilt_clip_rows(
                database, 0, 3, clip, "sources[0]")
        clip.contacts[0, 0] = 0
        original_contacts = clip.contacts
        clip.contacts = clip.contacts.astype(bool)
        try:
            with self.assertRaisesRegex(ValueError, "contacts.*uint8 dtype"):
                validator_module._compare_rebuilt_clip_rows(
                    database, 0, 3, clip, "sources[0]")
        finally:
            clip.contacts = original_contacts

    def test_rebuilt_clip_metadata_locks_identity_map_and_report_tolerance(self):
        source = types.SimpleNamespace(
            name="takara_walk_50hz",
            terrain_id="flat",
            fps=50.0,
            qpos=np.zeros((5, 36), np.float32),
            source_frames=np.arange(5, dtype=np.int64),
        )
        clip = HoldenClip.empty(frames=3, bones=31)
        clip.name = source.name
        clip.terrain_id = source.terrain_id
        clip.source_frames = np.array([0, 2, 4], np.int64)
        entry = {
            "name": source.name,
            "terrain_id": source.terrain_id,
            "source_fps": 50.0,
            "source_frames": 5,
            "output_frames": 3,
            "range_start": 0,
            "range_stop": 3,
            "source_frame_map": [0, 2, 4],
        }
        validation = {
            "fk_max_error_m": [0.0],
            "duration_error_s": [0.0],
            "quaternion_norm_max_error": [0.0],
        }
        report = {
            "fk_max_error_m": validator_module.SOURCE_REPORT_ATOL,
            "duration_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        }
        validator_module._validate_rebuilt_clip_metadata(
            source, clip, entry, report, validation, 0)

        too_large = dict(report)
        too_large["fk_max_error_m"] = float(np.nextafter(
            validator_module.SOURCE_REPORT_ATOL, np.inf))
        with self.assertRaisesRegex(ValueError, "fk_max_error_m.*differs"):
            validator_module._validate_rebuilt_clip_metadata(
                source, clip, entry, too_large, validation, 0)

        missing = dict(report)
        del missing["duration_error_s"]
        with self.assertRaisesRegex(ValueError, "report keys"):
            validator_module._validate_rebuilt_clip_metadata(
                source, clip, entry, missing, validation, 0)

        mutations = (
            ("source-name", source, "name", "wrong", "source name"),
            ("source-terrain", source, "terrain_id", "wrong", "terrain"),
            ("source-fps", source, "fps", 25.0, "source_fps"),
            ("clip-name", clip, "name", "wrong", "rebuilt clip name"),
            ("clip-terrain", clip, "terrain_id", "wrong", "rebuilt terrain"),
            ("source-map", clip, "source_frames",
             np.array([0, 1, 4], np.int64), "source frame map"),
        )
        for case, owner, attribute, value, message in mutations:
            original = getattr(owner, attribute)
            setattr(owner, attribute, value)
            try:
                with self.subTest(case=case), self.assertRaisesRegex(
                    ValueError, message,
                ):
                    validator_module._validate_rebuilt_clip_metadata(
                        source, clip, entry, report, validation, 0)
            finally:
                setattr(owner, attribute, original)

    def test_full_manifest_contract_and_source_discovery_are_exact(self):
        manifest, database = _small_full_source_case()
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("g1.xml", "takara.npz", "remap.npy"):
                with open(os.path.join(temporary, name), "wb") as stream:
                    stream.write(b"x")
            for base in SMALL_FULL_GRAIL_BASES:
                with open(
                    os.path.join(temporary, base + ".pkl"), "wb",
                ) as stream:
                    stream.write(b"x")
            options = {
                "grail_glob": os.path.join(temporary, "*.pkl"),
                "g1_xml": os.path.join(temporary, "g1.xml"),
                "takara": os.path.join(temporary, "takara.npz"),
                "remap": os.path.join(temporary, "remap.npy"),
            }
            with mock.patch.multiple(
                validator_module, **_small_full_constant_patches(),
                create=True,
            ):
                validator_module._validate_full_source_manifest_contract(
                    manifest, database)
                bases, path_by_base = \
                    validator_module._discover_full_source_corpus(
                        manifest, options)
            self.assertEqual(bases, SMALL_FULL_GRAIL_BASES)
            self.assertEqual(tuple(path_by_base), SMALL_FULL_GRAIL_BASES)

            wrong_order = copy.deepcopy(manifest)
            wrong_order["sources"][1], wrong_order["sources"][2] = (
                wrong_order["sources"][2], wrong_order["sources"][1])
            with mock.patch.multiple(
                validator_module, **_small_full_constant_patches(),
                create=True,
            ), self.assertRaisesRegex(ValueError, "basename.*manifest order"):
                validator_module._discover_full_source_corpus(
                    wrong_order, options)

            os.rename(
                os.path.join(temporary, SMALL_FULL_GRAIL_BASES[-1] + ".pkl"),
                os.path.join(temporary, "unexpected.pkl"))
            with mock.patch.multiple(
                validator_module, **_small_full_constant_patches(),
                create=True,
            ), self.assertRaisesRegex(ValueError, "basename.*manifest"):
                validator_module._discover_full_source_corpus(
                    manifest, options)

        mutations = (
            ("diagnostic_mode", True, "non-diagnostic"),
            ("total_clips", 4, "total clip count"),
            ("grail_clips", 3, "GRAIL clip count"),
            ("database_frames", 14, "row count"),
        )
        for key, value, message in mutations:
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.subTest(key=key), mock.patch.multiple(
                validator_module, **_small_full_constant_patches(),
                create=True,
            ), self.assertRaisesRegex(ValueError, message):
                validator_module._validate_full_source_manifest_contract(
                    changed, database)

        changed = copy.deepcopy(manifest)
        changed["sources"][0]["source_frames"] = 6
        with mock.patch.multiple(
            validator_module, **_small_full_constant_patches(), create=True,
        ), self.assertRaisesRegex(ValueError, "Takara source frame count"):
            validator_module._validate_full_source_manifest_contract(
                changed, database)
        changed = copy.deepcopy(manifest)
        changed["sources"][1]["output_frames"] = 2
        with mock.patch.multiple(
            validator_module, **_small_full_constant_patches(), create=True,
        ), self.assertRaisesRegex(ValueError, "GRAIL.*frame count"):
            validator_module._validate_full_source_manifest_contract(
                changed, database)

    def test_source_discovery_stops_after_one_match_beyond_the_limit(self):
        manifest, _ = _small_full_source_case()
        limit = len(SMALL_FULL_GRAIL_BASES)

        class HostileMatches:
            def __init__(self):
                self.consumed = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.consumed += 1
                if self.consumed > limit + 1:
                    raise AssertionError(
                        "source discovery exhausted a hostile iterator")
                return f"/virtual/grail_{self.consumed:04d}.pkl"

        matches = HostileMatches()
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("g1.xml", "takara.npz", "remap.npy"):
                with open(os.path.join(temporary, name), "wb") as stream:
                    stream.write(b"x")
            options = {
                "grail_glob": "/virtual/*.pkl",
                "g1_xml": os.path.join(temporary, "g1.xml"),
                "takara": os.path.join(temporary, "takara.npz"),
                "remap": os.path.join(temporary, "remap.npy"),
            }
            with mock.patch.multiple(
                validator_module, **_small_full_constant_patches(),
                create=True,
            ), mock.patch.object(
                validator_module.glob, "iglob", return_value=matches,
            ), mock.patch.object(
                validator_module.glob, "glob",
                side_effect=lambda pattern: list(matches),
            ), self.assertRaisesRegex(
                ValueError, "exactly 4 clips",
            ):
                validator_module._discover_full_source_corpus(
                    manifest, options)

        self.assertEqual(matches.consumed, limit + 1)

    def test_grail_surfaces_are_premeasured_and_literal_selection_is_locked(self):
        events = []
        heights = dict(LOCKED_GRAIL_HEIGHTS)

        class Terrain:
            def __init__(self, base):
                self.base = base

            def footprint(self):
                events.append(("measure", self.base))
                return {"height": heights[self.base]}

        with mock.patch.object(
            validator_module.GrailTerrain, "from_base",
            side_effect=lambda base: Terrain(base),
        ):
            measured, selected = validator_module._premeasure_grail_surfaces(
                SMALL_FULL_GRAIL_BASES)
        self.assertEqual(
            events,
            [("measure", base) for base in SMALL_FULL_GRAIL_BASES])
        self.assertEqual(measured, heights)
        self.assertEqual(selected, validator_module.GRAIL_EXPECTED_BASES)

        heights[GRAIL_DEFAULT_BASE] += 0.001
        with mock.patch.object(
            validator_module.GrailTerrain, "from_base",
            side_effect=lambda base: Terrain(base),
        ), self.assertRaisesRegex(ValueError, "source height changed"):
            validator_module._premeasure_grail_surfaces(
                SMALL_FULL_GRAIL_BASES)

    def test_full_source_streams_in_order_and_retains_only_selected_clips(self):
        manifest, database = _small_full_source_case(
            SMALL_STREAM_GRAIL_BASES)
        events = []
        source_refs = {}
        clip_refs = {}
        heights = {
            **LOCKED_GRAIL_HEIGHTS,
            SMALL_EXTRA_GRAIL_BASES[0]: 10.0,
            SMALL_EXTRA_GRAIL_BASES[1]: 20.0,
        }

        class Terrain:
            def __init__(self, base):
                self.base = base
                events.append(("terrain", base))

            def footprint(self):
                events.append(("measure", self.base))
                return {"height": heights[self.base]}

        class Kinematics:
            def __init__(self, path):
                events.append(("kinematics", path))

        class Source:
            pass

        def make_source(name, terrain_id, fps, source_frames):
            source = Source()
            source.name = name
            source.terrain_id = terrain_id
            source.fps = float(fps)
            source.qpos = np.zeros((source_frames, 36), np.float32)
            source.source_frames = np.arange(
                source_frames, dtype=np.int64)
            source_refs[name] = weakref.ref(source)
            return source

        def load_takara(path, remap):
            events.append(("load", "takara_walk_50hz"))
            return make_source("takara_walk_50hz", "flat", 50.0, 5)

        def load_grail(path):
            base = os.path.splitext(os.path.basename(path))[0]
            events.append(("load", base))
            return make_source(base, base, 25.0, 3)

        def recompute(source, terrain, kinematics, names, parents):
            events.append(("recompute", source.name))
            self.assertIs(names, manifest["skeleton"]["names"])
            self.assertIs(parents, manifest["skeleton"]["parents"])
            clip = HoldenClip.empty(frames=3, bones=31)
            clip.name = source.name
            clip.terrain_id = source.terrain_id
            clip.source_frames = (
                np.array([0, 2, 4], np.int64)
                if source.terrain_id == "flat"
                else np.arange(3, dtype=np.int64))
            clip_refs[source.name] = weakref.ref(clip)
            return clip, {
                "fk_max_error_m": 0.0,
                "duration_error_s": 0.0,
                "quaternion_norm_max_error": 0.0,
            }

        def validate_selected(scenes, selected_clips, selected):
            gc.collect()
            events.append(("selected-scenes", tuple(sorted(selected_clips))))
            self.assertEqual(selected, validator_module.GRAIL_EXPECTED_BASES)
            self.assertEqual(
                set(selected_clips),
                set(validator_module.GRAIL_EXPECTED_BASES.values()))
            self.assertTrue(all(
                source_ref() is None for source_ref in source_refs.values()))
            self.assertTrue(all(
                clip_refs[base]() is None for base in SMALL_EXTRA_GRAIL_BASES))

        with tempfile.TemporaryDirectory() as temporary:
            for name in ("g1.xml", "takara.npz", "remap.npy"):
                with open(os.path.join(temporary, name), "wb") as stream:
                    stream.write(b"x")
            for base in SMALL_STREAM_GRAIL_BASES:
                with open(
                    os.path.join(temporary, base + ".pkl"), "wb",
                ) as stream:
                    stream.write(b"x")
            options = {
                "grail_glob": os.path.join(temporary, "*.pkl"),
                "g1_xml": os.path.join(temporary, "g1.xml"),
                "takara": os.path.join(temporary, "takara.npz"),
                "remap": os.path.join(temporary, "remap.npy"),
            }
            with mock.patch.multiple(
                validator_module,
                **_small_full_constant_patches(SMALL_STREAM_GRAIL_BASES),
                G1Kinematics=Kinematics,
                FlatTerrain=lambda: Terrain("flat"),
                load_takara=load_takara,
                load_grail=load_grail,
                _recompute_clip=recompute,
                _validate_selected_scene_reconstruction=validate_selected,
                create=True,
            ), mock.patch.object(
                validator_module.GrailTerrain, "from_base",
                side_effect=lambda base: Terrain(base),
            ):
                rows = validator_module._validate_all_source_rows(
                    manifest, database, {}, options,
                    progress=lambda *values: events.append(values))
        self.assertEqual(rows, 21)
        load_names = [
            value[1] for value in events
            if len(value) == 2 and value[0] == "load"
        ]
        self.assertEqual(
            load_names, ["takara_walk_50hz", *SMALL_STREAM_GRAIL_BASES])
        first_load = next(
            index for index, value in enumerate(events) if value[0] == "load")
        measured_surface_indices = [
            index for index, value in enumerate(events)
            if len(value) == 2 and value[0] == "measure"]
        self.assertEqual(
            len(measured_surface_indices), len(SMALL_STREAM_GRAIL_BASES))
        self.assertLess(max(measured_surface_indices), first_load)
        progress_measure_indices = [
            index for index, value in enumerate(events)
            if len(value) == 4 and value[0] == "measure"]
        self.assertEqual(
            len(progress_measure_indices), len(SMALL_STREAM_GRAIL_BASES))
        self.assertLess(max(progress_measure_indices), first_load)
        self.assertEqual(
            sum(value[0] == "kinematics" for value in events), 1)

    def test_takara_support_requires_positive_zero_bits_in_all_columns(self):
        support = np.zeros((3, 3), np.float32)
        validator_module._validate_takara_support_zero_bits(support)
        for column in range(3):
            changed = support.copy()
            changed[1, column] = np.float32(-0.0)
            with self.subTest(column=column), self.assertRaisesRegex(
                ValueError, "Takara support.*positive-zero",
            ):
                validator_module._validate_takara_support_zero_bits(changed)

    def test_selected_scene_reconstruction_is_exact_for_all_four_sources(self):
        scenes = {
            built.scene_id: (json.loads(built.scene_json), None, None)
            for built in _canonical_scene_pack().scenes[:4]
        }
        clips = {
            base: _fake_grail_clip(base) for base in LOCKED_GRAIL_HEIGHTS
        }
        validator_module._validate_selected_scene_reconstruction(
            scenes, clips, validator_module.GRAIL_EXPECTED_BASES)

        changed = copy.deepcopy(scenes)
        changed["grail-curb-low"][0]["routes"][0]["waypoints_xz"][1][0] = \
            float(np.float32(9.0))
        with self.assertRaisesRegex(
            ValueError, "deterministic converted-source scene changed",
        ):
            validator_module._validate_selected_scene_reconstruction(
                changed, clips, validator_module.GRAIL_EXPECTED_BASES)

    def test_full_source_cli_forwards_paths_and_reports_rebuilt_rows(self):
        summary = {
            "frames": 6_000_003,
            "clips": 20_003,
            "bones": 31,
            "scenes": 14,
            "source_rows": 6_000_003,
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        arguments = [
            "/tmp/full-artifact",
            "--full-source-validation",
            "--grail-glob", "/tmp/grail/*.pkl",
            "--g1-xml", "/tmp/g1.xml",
            "--takara", "/tmp/takara.npz",
            "--remap", "/tmp/remap.npy",
        ]
        with mock.patch.object(
            validator_module, "validate_artifact_directory",
            return_value=summary,
        ) as validate, redirect_stdout(stdout), redirect_stderr(stderr):
            status = validator_module.main(arguments)

        self.assertEqual(status, 0)
        validate.assert_called_once_with(
            "/tmp/full-artifact", True, {
                "grail_glob": "/tmp/grail/*.pkl",
                "g1_xml": "/tmp/g1.xml",
                "takara": "/tmp/takara.npz",
                "remap": "/tmp/remap.npy",
            })
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            "VALID g1-terrain-artifacts/v3 frames=6000003 clips=20003 "
            "bones=31 terrain_dims=12 support_dims=3 scenes=14 "
            "source_rows=6000003\n")


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

    def _reject(
        self, label, message, mutation, *, verify_restored_baseline=True,
    ):
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
            if verify_restored_baseline:
                with self.subTest(case=label + "-restored-baseline"):
                    self.assertEqual(
                        validate_artifact_directory(self.output),
                        self._expected_summary())

    def test_direct_canonical_v3_fixture_validates_normally(self):
        class UninspectableSourceOptions:
            def __iter__(self):
                raise AssertionError("normal mode inspected source_options")

        expected_region_ids = {
            "grail-curb-default": {
                "certified": ("left-apron", "right-apron"),
                "stress": ("curb-route",), "blocked": (),
            },
            "grail-curb-low": {
                "certified": ("curb-route",),
                "stress": (), "blocked": (),
            },
            "grail-curb-medium": {
                "certified": ("left-apron", "right-apron"),
                "stress": ("curb-route",), "blocked": (),
            },
            "grail-curb-high": {
                "certified": ("left-apron", "right-apron"),
                "stress": ("curb-route",), "blocked": (),
            },
            "ramp-15-stress": {
                "certified": ("left-apron", "right-apron"),
                "stress": ("course",), "blocked": (),
            },
            "blocked-course": {
                "certified": (
                    "approach", "left-bypass", "right-bypass"),
                "stress": (), "blocked": ("wall", "gap", "ramp"),
            },
        }
        for scene_id, expected in expected_region_ids.items():
            scene = _load_json(self._path(
                f"scenes/{scene_id}/scene.json"))
            observed = {
                class_name: tuple(
                    region["id"] for region in scene["regions"][class_name])
                for class_name in ("certified", "stress", "blocked")
            }
            with self.subTest(scene=scene_id):
                self.assertEqual(observed, expected)

        maximum_grid_cells = max(
            np.prod(struct.unpack_from("<II", built.terrain_bin, 8))
            for built in _canonical_scene_pack().scenes
        )
        self.assertEqual(maximum_grid_cells, 306726)
        self.assertEqual(
            validator_module._MAX_GRID_CELLS, maximum_grid_cells)
        maximum_obj_bytes = max(
            len(built.terrain_obj)
            for built in _canonical_scene_pack().scenes
        )
        self.assertEqual(maximum_obj_bytes, 21762970)
        self.assertEqual(
            validator_module._MAX_OBJ_BYTES, maximum_obj_bytes)

        self.assertEqual(
            validate_artifact_directory(
                self.output,
                source_options=UninspectableSourceOptions()),
            self._expected_summary())

    def test_v3_manifest_motion_and_bank_descriptors_are_exact(self):
        manifest = _load_json(self._path("manifest.json"))
        self.assertEqual(set(manifest), {
            "schema", "output_fps", "feature_dimensions",
            "terrain_dimensions", "support_dimensions",
            "terrain_feature_distances_m", "total_clips", "grail_clips",
            "skipped_clips", "database_frames", "diagnostic_mode",
            "sources", "skeleton", "contact", "surface", "database",
            "sidecars", "motion_index", "motion_banks", "scene_index",
            "validation_file", "validation",
        })
        self.assertEqual(
            (manifest["schema"], manifest["feature_dimensions"],
             manifest["terrain_dimensions"], manifest["support_dimensions"]),
            ("g1-terrain-artifacts/v3", 39, 12, 3),
        )
        self.assertEqual(set(manifest["sources"][0]), {
            "name", "terrain_id", "terrain_family", "source_fps",
            "source_frames", "output_frames", "range_start", "range_stop",
            "source_frame_map",
        })
        self.assertEqual(manifest["sources"][0]["terrain_family"], "flat")
        self.assertEqual(manifest["motion_index"], {
            "path": "motion_index.bin", "schema": "G1MI/v1",
            "version": 1, "frame_count": 3, "row_width": 4,
            "sha256": _sha256(self._path("motion_index.bin")),
        })
        banks = manifest["motion_banks"]
        self.assertEqual(set(banks), {
            "schema", "frame_count", "ranges", "banks", "sha256",
        })
        self.assertEqual(
            [bank["family"] for bank in banks["banks"]],
            ["flat", "curb", "slope", "stair"],
        )
        bank_payload = dict(banks)
        bank_digest = bank_payload.pop("sha256")
        self.assertEqual(
            bank_digest,
            hashlib.sha256(canonical_json_bytes(bank_payload)).hexdigest(),
        )
        with open(self._path("motion_index.bin"), "rb") as stream:
            payload = stream.read()
        self.assertEqual(
            struct.unpack_from("<4sIII", payload), (b"G1MI", 1, 3, 4))

    def test_motion_index_tree_hash_size_rows_and_values_are_authenticated(self):
        def resign_motion(preserve):
            self._resign_root_payload(preserve, ["motion_index"])

        def missing(preserve):
            os.unlink(preserve("motion_index.bin"))

        def symlink(preserve):
            path = preserve("motion_index.bin")
            os.unlink(path)
            os.symlink("terrain_support.bin", path)

        def wrong_hash(preserve):
            self._rewrite_json(
                preserve, "manifest.json",
                lambda value: value["motion_index"].__setitem__(
                    "sha256", "0" * 64))

        def wrong_size(preserve):
            path = preserve("motion_index.bin")
            with open(path, "r+b") as stream:
                stream.truncate(os.path.getsize(path) - 1)
            resign_motion(preserve)

        def wrong_rows(preserve):
            path = preserve("motion_index.bin")
            with open(path, "r+b") as stream:
                stream.seek(8)
                stream.write(struct.pack("<I", 2))
            resign_motion(preserve)

        def zero_direction(preserve):
            path = preserve("motion_index.bin")
            with open(path, "r+b") as stream:
                stream.seek(16)
                stream.write(b"\x00\x00")
            resign_motion(preserve)

        def bad_elevation(preserve):
            path = preserve("motion_index.bin")
            with open(path, "r+b") as stream:
                stream.seek(19)
                stream.write(b"\x02")
            resign_motion(preserve)

        for name, message, mutation in (
            ("missing", "missing|tree", missing),
            ("symlink", "symlink", symlink),
            ("hash", "motion index SHA-256", wrong_hash),
            ("size", "motion index size", wrong_size),
            ("rows", "motion index header|frame count", wrong_rows),
            ("direction", "direction mask", zero_direction),
            ("elevation", "elevation", bad_elevation),
        ):
            self._reject(
                name, message, mutation, verify_restored_baseline=False)

    def test_sources_and_motion_banks_require_complete_unique_family_ownership(self):
        def rewrite_banks(preserve, mutation):
            path = preserve("manifest.json")
            manifest = _load_json(path)
            mutation(manifest)
            payload = dict(manifest["motion_banks"])
            payload.pop("sha256")
            manifest["motion_banks"]["sha256"] = hashlib.sha256(
                canonical_json_bytes(payload)).hexdigest()
            _write_json(path, manifest)

        def unknown_source_family(preserve):
            rewrite_banks(
                preserve,
                lambda value: value["sources"][0].__setitem__(
                    "terrain_family", "mud"),
            )

        def unknown_bank_family(preserve):
            rewrite_banks(
                preserve,
                lambda value: value["motion_banks"]["banks"][0].__setitem__(
                    "family", "mud"),
            )

        def missing_owner(preserve):
            rewrite_banks(
                preserve,
                lambda value: value["motion_banks"]["banks"][0].__setitem__(
                    "range_indices", []),
            )

        def mismatched_range(preserve):
            rewrite_banks(
                preserve,
                lambda value: value["motion_banks"]["ranges"][0].__setitem__(
                    "global_stop", 2),
            )

        for name, message, mutation in (
            ("source-family", "terrain family", unknown_source_family),
            ("bank-family", "terrain families", unknown_bank_family),
            ("missing-owner", "ownership", missing_owner),
            ("range", "range|coverage", mismatched_range),
        ):
            self._reject(
                name, message, mutation, verify_restored_baseline=False)

    def test_large_corpus_counts_are_validated_from_metadata_without_fixed_totals(self):
        manifest = _load_json(self._path("manifest.json"))
        manifest.update({
            "diagnostic_mode": False,
            "total_clips": 20_003,
            "grail_clips": 20_002,
            "database_frames": 6_000_003,
        })
        validator_module._validate_manifest_header(manifest)
        self.assertEqual(
            validator_module._motion_payload_sizes(6_000_003, 20_003),
            (
                176 + 8 * 20_003 + 1614 * 6_000_003,
                16 + 48 * 6_000_003,
                16 + 12 * 6_000_003,
                16 + 4 * 6_000_003,
            ),
        )

    def test_v1_artifact_set_manifest_is_rejected_without_dispatch(self):
        manifest_path = self._path("manifest.json")
        manifest = _load_json(manifest_path)
        manifest["schema"] = "g1-terrain-artifacts/v1"
        _write_json(manifest_path, manifest)

        with self.assertRaisesRegex(
            ValueError,
            r"^schema must be g1-terrain-artifacts/v3$",
        ):
            validate_artifact_directory(self.output)

    def test_full_source_runs_after_normal_gates_with_loaded_v3_objects(self):
        options = {"g1_xml": "/tmp/test-g1.xml"}
        observed = {}

        def validate_sources(
            manifest, database, scenes, source_options, progress=None,
        ):
            observed["manifest"] = manifest
            observed["database"] = database
            observed["scenes"] = scenes
            observed["source_options"] = source_options
            self.assertIsNotNone(progress)
            progress("recompute", 1, 1, "fixture-source")
            return 3

        stderr = io.StringIO()
        with mock.patch.object(
            validator_module, "_validate_all_source_rows",
            side_effect=validate_sources,
        ) as full_validator, redirect_stderr(stderr):
            summary = validate_artifact_directory(
                self.output, full_source_validation=True,
                source_options=options)

            unexpected = os.path.join(self.output, "unexpected.bin")
            with open(unexpected, "wb") as stream:
                stream.write(b"x")
            try:
                with self.assertRaisesRegex(ValueError, "unexpected file"):
                    validate_artifact_directory(
                        self.output, full_source_validation=True,
                        source_options=options)
            finally:
                os.unlink(unexpected)

        self.assertEqual(summary, {
            **self._expected_summary(), "source_rows": 3,
        })
        self.assertEqual(full_validator.call_count, 1)
        self.assertIs(observed["source_options"], options)
        self.assertEqual(observed["manifest"]["schema"], validator_module.SCHEMA)
        self.assertEqual(len(observed["database"].positions), 3)
        self.assertEqual(
            tuple(observed["scenes"]), validator_module.LOCKED_SCENE_IDS)
        self.assertEqual(
            stderr.getvalue(),
            "FULL-SOURCE recompute 1/1 fixture-source\n")

    def test_v3_cli_reports_the_fixed_schema_and_dimensions(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = validator_module.main([self.output])
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(
            stdout.getvalue(),
            "VALID g1-terrain-artifacts/v3 frames=3 clips=1 bones=31 "
            "terrain_dims=12 support_dims=3 scenes=14 source_rows=0\n")

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
                playable_xmax = np.float32(
                    scene["bounds"]["playable_max_xz"][0])
                scene["regions"]["certified"][0]["bounds_xz"][1] = \
                    float(np.float32(
                        playable_xmax + np.float32(0.251)))

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
