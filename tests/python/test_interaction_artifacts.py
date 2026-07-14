import copy
import dataclasses
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from resources.g1_interaction_builder.artifacts import (
    DB_MAGIC,
    ENDIAN_MARKER,
    FEATURE_MAGIC,
    VERSION,
    _write_array,
    assemble_database,
    read_artifact_set,
    write_artifact_set,
)
from resources.g1_interaction_builder.features import (
    FEATURE_GROUPS,
    build_database_features,
)
from resources.g1_interaction_builder.schema import (
    EvaluationSplit,
    FeatureGroup,
    G1_SKELETON,
    InteractionArtifact,
    InteractionValidationError,
)
from resources.g1_interaction_builder.splits import partition_clips
from resources.g1_terrain_builder.schema import SkeletonSpec
from tests.python.interaction_fixture import labeled_clips_for_objects


DB_HEADER = struct.Struct("<8s8I")
FEATURE_HEADER = struct.Struct("<8s5I")


def artifact_fixture():
    clips = labeled_clips_for_objects(
        ["database_fixture_object", "heldout_fixture_object"]
    )
    split = EvaluationSplit(
        seed=20260714,
        database_objects=("database_fixture_object",),
        heldout_objects=("heldout_fixture_object",),
    )
    database, _ = partition_clips(clips, split)
    labeled = database[0]
    artifact = assemble_database(database, G1_SKELETON)
    # This production boundary intentionally receives every labeled clip. It
    # owns filtering so held-out objects cannot affect persisted statistics.
    features = build_database_features(clips, split, G1_SKELETON)
    manifest = {
        "schema_version": 1,
        "target_fps": 25.0,
        "skeleton_signature": G1_SKELETON.signature(),
        "clips": [
            {
                "sequence_id": labeled.motion.sequence_id,
                "object_id": labeled.motion.object_id,
                "range_start": 0,
                "range_stop": len(labeled.motion.positions),
            }
        ],
    }
    report = {
        "schema_version": 1,
        "source_clips": 2,
        "included_clips": 2,
        "rejected_clips": 0,
        "included_frames": len(labeled.motion.positions),
        "rejections_by_code": {},
        "rejections": [],
        "numeric_bounds": {
            "fk_max_error_m": 0.0,
            "fk_rotation_max_error_degrees": 0.0,
            "duration_max_error_s": 0.0,
            "quaternion_norm_max_error": 0.0,
        },
    }
    return artifact, features, split, manifest, report


def assert_artifact_equal(
    test: unittest.TestCase,
    expected: InteractionArtifact,
    actual: InteractionArtifact,
) -> None:
    for field in dataclasses.fields(InteractionArtifact):
        left = getattr(expected, field.name)
        right = getattr(actual, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
            test.assertEqual(left.dtype, right.dtype, field.name)
        else:
            test.assertEqual(left, right, field.name)


def mutate_bad_magic(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[:8] = b"BADMAGIC"
    path.write_bytes(data)


def mutate_truncated(path: Path) -> None:
    data = path.read_bytes()
    path.write_bytes(data[:-1])


def mutate_trailing(path: Path) -> None:
    with path.open("ab") as stream:
        stream.write(b"x")


def binary_mutations():
    return (
        (mutate_bad_magic, "database magic"),
        (mutate_truncated, "truncated source_frames"),
        (mutate_trailing, "trailing bytes"),
    )


def replace_u32(path: Path, offset: int, value: int) -> None:
    data = bytearray(path.read_bytes())
    data[offset:offset + 4] = struct.pack("<I", value)
    path.write_bytes(data)


def rewrite_json(path: Path, update) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def snapshot(output: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.is_file()
    }


def private_publish_paths(output: Path) -> list[Path]:
    return sorted(output.parent.glob(f".{output.name}.*-{os.getpid()}"))


def database_offsets(frame_count: int, clip_count: int) -> dict[str, int]:
    offset = DB_HEADER.size
    fields = (
        ("parents", 31 * 4),
        ("range_starts", clip_count * 4),
        ("range_stops", clip_count * 4),
        ("positions", frame_count * 31 * 3 * 4),
        ("velocities", frame_count * 31 * 3 * 4),
        ("rotations", frame_count * 31 * 4 * 4),
        ("angular_velocities", frame_count * 31 * 3 * 4),
        ("foot_contacts", frame_count * 2),
        ("hand_contacts", frame_count * 2),
        ("hand_dof", frame_count * 14 * 4),
        ("hand_dof_velocities", frame_count * 14 * 4),
        ("phases", frame_count),
        ("active_hands", clip_count),
        ("time_to_contact", frame_count * 4),
        ("object_positions", frame_count * 3 * 4),
        ("object_rotations", frame_count * 4 * 4),
        ("object_velocities", frame_count * 3 * 4),
        ("object_angular_velocities", frame_count * 3 * 4),
        ("table_positions", clip_count * 3 * 4),
        ("table_rotations", clip_count * 4 * 4),
        ("table_sizes", clip_count * 3 * 4),
        ("object_dimensions", clip_count * 3 * 4),
        ("grasp_positions_object", clip_count * 3 * 4),
        ("grasp_rotations_object", clip_count * 4 * 4),
        ("approach_directions_object", clip_count * 3 * 4),
        ("source_frames", frame_count * 4),
    )
    offsets = {}
    for name, size in fields:
        offsets[name] = offset
        offset += size
    offsets["end"] = offset
    return offsets


def expected_feature_size(frame_count: int) -> int:
    return FEATURE_HEADER.size + 2 * 5 * 4 + 2 * 71 * 4 + (
        frame_count * 71 * 4
    )


class InteractionArtifactAssemblyTests(unittest.TestCase):
    def test_assembly_sorts_clips_and_preserves_every_source_field(self):
        clips = labeled_clips_for_objects(["z_object", "a_object"])
        clips[0].motion.table_position[:] = [9.0, 8.0, 7.0]
        clips[1].motion.table_position[:] = [1.0, 2.0, 3.0]
        clips[0].motion.source_frames += 100
        clips[1].motion.source_frames += 200
        expected = sorted(clips, key=lambda clip: clip.motion.sequence_id)

        artifact = assemble_database(clips, G1_SKELETON)

        frames = len(expected[0].motion.positions)
        np.testing.assert_array_equal(artifact.range_starts, [0, frames])
        np.testing.assert_array_equal(
            artifact.range_stops, [frames, 2 * frames]
        )
        frame_fields = (
            "positions",
            "velocities",
            "rotations",
            "angular_velocities",
            "foot_contacts",
            "hand_contacts",
            "hand_dof",
            "hand_dof_velocities",
            "object_positions",
            "object_rotations",
            "object_velocities",
            "object_angular_velocities",
            "source_frames",
        )
        for name in frame_fields:
            np.testing.assert_array_equal(
                getattr(artifact, name),
                np.concatenate(
                    [getattr(clip.motion, name) for clip in expected]
                ),
                err_msg=name,
            )
        np.testing.assert_array_equal(
            artifact.phases,
            np.concatenate([clip.phases for clip in expected]),
        )
        np.testing.assert_array_equal(
            artifact.time_to_contact,
            np.concatenate([clip.time_to_contact for clip in expected]),
        )
        np.testing.assert_array_equal(
            artifact.active_hands,
            [int(clip.active_hand) for clip in expected],
        )
        clip_fields = (
            ("table_positions", "table_position"),
            ("table_rotations", "table_rotation"),
            ("table_sizes", "table_size"),
            ("object_dimensions", "object_dimensions"),
        )
        for artifact_name, motion_name in clip_fields:
            np.testing.assert_array_equal(
                getattr(artifact, artifact_name),
                np.stack(
                    [getattr(clip.motion, motion_name) for clip in expected]
                ),
                err_msg=artifact_name,
            )
        label_fields = (
            ("grasp_positions_object", "grasp_position_object"),
            ("grasp_rotations_object", "grasp_rotation_object"),
            (
                "approach_directions_object",
                "approach_direction_object",
            ),
        )
        for artifact_name, label_name in label_fields:
            np.testing.assert_array_equal(
                getattr(artifact, artifact_name),
                np.stack(
                    [getattr(clip, label_name) for clip in expected]
                ),
                err_msg=artifact_name,
            )
        for field in dataclasses.fields(InteractionArtifact):
            value = getattr(artifact, field.name)
            if isinstance(value, np.ndarray):
                self.assertTrue(value.flags.c_contiguous, field.name)

    def test_assembly_rejects_empty_input_and_wrong_skeleton(self):
        with self.assertRaisesRegex(
            ValueError, "cannot assemble an empty interaction database"
        ):
            assemble_database([], G1_SKELETON)

        wrong_names = list(G1_SKELETON.names)
        wrong_names[-1] = "WrongWrist"
        wrong = SkeletonSpec(tuple(wrong_names), G1_SKELETON.parents.copy())
        with self.assertRaisesRegex(
            InteractionValidationError, "skeleton_mismatch"
        ):
            assemble_database(
                labeled_clips_for_objects(["database"]), wrong
            )

    def test_validate_enforces_exact_shapes_and_dtypes(self):
        artifact, _, _, _, _ = artifact_fixture()
        cases = (
            ("fps", lambda value: setattr(value, "fps", 24), "fps.*25"),
            (
                "parents shape",
                lambda value: setattr(value, "parents", value.parents[:-1]),
                "parents shape.*31",
            ),
            (
                "position shape",
                lambda value: setattr(
                    value, "positions", value.positions[:, :-1]
                ),
                "positions shape",
            ),
            (
                "position dtype",
                lambda value: setattr(
                    value, "positions", value.positions.astype(np.float64)
                ),
                "positions dtype.*float32",
            ),
            (
                "contacts dtype",
                lambda value: setattr(
                    value,
                    "hand_contacts",
                    value.hand_contacts.astype(np.int32),
                ),
                "hand_contacts dtype.*uint8",
            ),
            (
                "source dtype",
                lambda value: setattr(
                    value,
                    "source_frames",
                    value.source_frames.astype(np.int64),
                ),
                "source_frames dtype.*int32",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(label=label):
                invalid = copy.deepcopy(artifact)
                mutation(invalid)
                with self.assertRaisesRegex(
                    InteractionValidationError, message
                ):
                    invalid.validate()

    def test_validate_rejects_wrong_parents_and_invalid_ranges(self):
        artifact, _, _, _, _ = artifact_fixture()
        two_ranges = copy.deepcopy(artifact)
        frame_count = len(two_ranges.positions)
        split = frame_count // 2
        two_ranges.range_starts = np.array([0, split], np.int32)
        two_ranges.range_stops = np.array([split, frame_count], np.int32)
        two_ranges.active_hands = np.repeat(
            two_ranges.active_hands, 2
        ).astype(np.uint8)
        for name in (
            "table_positions",
            "table_rotations",
            "table_sizes",
            "object_dimensions",
            "grasp_positions_object",
            "grasp_rotations_object",
            "approach_directions_object",
        ):
            setattr(
                two_ranges,
                name,
                np.repeat(getattr(two_ranges, name), 2, axis=0),
            )
        # Both synthetic ranges still contain contact, lift, and hold.
        two_ranges.phases[:split] = np.minimum(
            two_ranges.phases[:split], 4
        )
        two_ranges.phases[0] = 2
        two_ranges.phases[1] = 3
        two_ranges.phases[2:split] = 4
        two_ranges.phases[split] = 2
        two_ranges.phases[split + 1] = 3
        two_ranges.phases[split + 2:] = 4
        two_ranges.validate()

        cases = (
            (
                "parents",
                lambda value: value.parents.__setitem__(1, -1),
                "parents.*exact G1",
            ),
            (
                "empty",
                lambda value: value.range_stops.__setitem__(0, 0),
                "empty range",
            ),
            (
                "leading gap",
                lambda value: value.range_starts.__setitem__(0, 1),
                "range coverage.*start at 0",
            ),
            (
                "interior gap",
                lambda value: value.range_starts.__setitem__(1, split + 1),
                "range coverage.*gap",
            ),
            (
                "overlap",
                lambda value: value.range_starts.__setitem__(1, split - 1),
                "range coverage.*overlap",
            ),
            (
                "wrong end",
                lambda value: value.range_stops.__setitem__(1, frame_count - 1),
                "range coverage.*frame count",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(label=label):
                invalid = copy.deepcopy(two_ranges)
                mutation(invalid)
                with self.assertRaisesRegex(
                    InteractionValidationError, message
                ):
                    invalid.validate()

    def test_validate_rejects_invalid_semantics_and_nonfinite_values(self):
        artifact, _, _, _, _ = artifact_fixture()
        cases = (
            (
                "phase order",
                lambda value: value.phases.__setitem__(1, 4),
                "phases.*monotonic",
            ),
            (
                "phase enum",
                lambda value: value.phases.__setitem__(0, 5),
                "phases.*0.*4",
            ),
            (
                "missing contact",
                lambda value: value.phases.__setitem__(
                    value.phases == 2, 1
                ),
                "range 0.*contact phase",
            ),
            (
                "missing lift",
                lambda value: value.phases.__setitem__(
                    value.phases == 3, 2
                ),
                "range 0.*lift phase",
            ),
            (
                "missing hold",
                lambda value: value.phases.__setitem__(
                    value.phases == 4, 3
                ),
                "range 0.*hold phase",
            ),
            (
                "contact value",
                lambda value: value.hand_contacts.__setitem__((0, 0), 2),
                "hand_contacts.*0.*1",
            ),
            (
                "active hand",
                lambda value: value.active_hands.__setitem__(0, 2),
                "active_hands.*0.*1",
            ),
            (
                "body quaternion",
                lambda value: value.rotations.__setitem__((0, 0, 0), 2.0),
                "rotations.*unit quaternion",
            ),
            (
                "object quaternion",
                lambda value: value.object_rotations.__setitem__((0, 0), 0.0),
                "object_rotations.*unit quaternion",
            ),
            (
                "table quaternion",
                lambda value: value.table_rotations.__setitem__((0, 0), 0.0),
                "table_rotations.*unit quaternion",
            ),
            (
                "grasp quaternion",
                lambda value: value.grasp_rotations_object.__setitem__(
                    (0, 0), 0.0
                ),
                "grasp_rotations_object.*unit quaternion",
            ),
            (
                "table dimension",
                lambda value: value.table_sizes.__setitem__((0, 0), 0.0),
                "table_sizes.*positive",
            ),
            (
                "object dimension",
                lambda value: value.object_dimensions.__setitem__(
                    (0, 0), -1.0
                ),
                "object_dimensions.*positive",
            ),
            (
                "nonfinite",
                lambda value: value.positions.__setitem__((0, 0, 0), np.nan),
                "positions.*non-finite",
            ),
            (
                "source order",
                lambda value: value.source_frames.__setitem__(
                    1, value.source_frames[0] - 1
                ),
                "source_frames.*nonnegative and nondecreasing",
            ),
        )
        for label, mutation, message in cases:
            with self.subTest(label=label):
                invalid = copy.deepcopy(artifact)
                mutation(invalid)
                with self.assertRaisesRegex(
                    InteractionValidationError, message
                ):
                    invalid.validate()


class InteractionArtifactSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_every_array_and_header(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            loaded_artifact, loaded_features, *metadata = (
                read_artifact_set(output)
            )

        assert_artifact_equal(self, artifact, loaded_artifact)
        np.testing.assert_array_equal(features.values, loaded_features.values)
        np.testing.assert_array_equal(
            features.offsets, loaded_features.offsets
        )
        np.testing.assert_array_equal(features.scales, loaded_features.scales)
        self.assertEqual(features.groups, loaded_features.groups)
        self.assertEqual(metadata[0], manifest)
        self.assertEqual(
            metadata[1],
            {
                "seed": split.seed,
                "database_objects": list(split.database_objects),
                "heldout_objects": list(split.heldout_objects),
            },
        )
        self.assertEqual(metadata[2], report)
        self.assertEqual(metadata[0]["schema_version"], 1)
        self.assertEqual(
            metadata[0]["skeleton_signature"], G1_SKELETON.signature()
        )

    def test_binary_headers_sizes_and_little_endian_bytes_are_exact(self):
        artifact, features, split, manifest, report = artifact_fixture()
        frame_count = len(artifact.positions)
        clip_count = len(artifact.range_starts)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            database = (output / "interaction_database.bin").read_bytes()
            feature_bytes = (
                output / "interaction_features.bin"
            ).read_bytes()

        self.assertEqual(DB_HEADER.size, 40)
        self.assertEqual(FEATURE_HEADER.size, 28)
        self.assertEqual(
            DB_HEADER.unpack_from(database),
            (
                DB_MAGIC,
                VERSION,
                ENDIAN_MARKER,
                25,
                1,
                frame_count,
                31,
                clip_count,
                14,
            ),
        )
        self.assertEqual(
            FEATURE_HEADER.unpack_from(feature_bytes),
            (
                FEATURE_MAGIC,
                VERSION,
                ENDIAN_MARKER,
                frame_count,
                71,
                5,
            ),
        )
        self.assertEqual(database[12:16], b"\x04\x03\x02\x01")
        self.assertEqual(feature_bytes[12:16], b"\x04\x03\x02\x01")
        offsets = database_offsets(frame_count, clip_count)
        self.assertEqual(len(database), offsets["end"])
        self.assertEqual(len(feature_bytes), expected_feature_size(frame_count))
        self.assertEqual(database[DB_HEADER.size:DB_HEADER.size + 4], b"\xff" * 4)
        self.assertEqual(
            feature_bytes[FEATURE_HEADER.size:FEATURE_HEADER.size + 20],
            struct.pack("<5I", 0, 33, 45, 57, 65),
        )
        self.assertEqual(
            feature_bytes[
                FEATURE_HEADER.size + 20:FEATURE_HEADER.size + 40
            ],
            struct.pack("<5I", 33, 45, 57, 65, 71),
        )

    def test_write_array_rejects_noncontiguous_before_endian_conversion(self):
        noncontiguous = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        self.assertFalse(noncontiguous.flags.c_contiguous)

        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            _write_array(io.BytesIO(), noncontiguous, "f4")

    def test_write_array_converts_contiguous_big_endian_input_to_little_endian(self):
        value = np.array([0x01020304], dtype=">u4")
        self.assertTrue(value.flags.c_contiguous)
        stream = io.BytesIO()

        _write_array(stream, value, "u4")

        self.assertEqual(stream.getvalue(), b"\x04\x03\x02\x01")

    def test_json_is_deterministic_portable_and_newline_terminated(self):
        artifact, features, split, manifest, report = artifact_fixture()
        manifest = {"z_extra": "portable", **manifest}
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            write_artifact_set(
                first, artifact, features, split, manifest, report
            )
            write_artifact_set(
                second, artifact, features, split, manifest, report
            )
            for name in (
                "manifest.json",
                "evaluation_split.json",
                "validation_report.json",
            ):
                left = (first / name).read_bytes()
                right = (second / name).read_bytes()
                self.assertEqual(left, right, name)
                self.assertTrue(left.endswith(b"\n"), name)
                self.assertEqual(left, left.decode("utf-8").encode("utf-8"))
            keys = list(
                json.loads((first / "manifest.json").read_text()).keys()
            )
            self.assertEqual(keys, sorted(keys))

    def test_rejects_truncation_trailing_bytes_and_bad_magic(self):
        artifact, features, split, manifest, report = artifact_fixture()
        for mutation, message in binary_mutations():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                mutation(output / "interaction_database.bin")
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_rejects_feature_truncation_trailing_bytes_and_bad_magic(self):
        artifact, features, split, manifest, report = artifact_fixture()
        cases = (
            (mutate_bad_magic, "feature magic"),
            (mutate_truncated, "truncated feature values"),
            (mutate_trailing, "trailing bytes"),
        )
        for mutation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                mutation(output / "interaction_features.bin")
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_rejects_invalid_database_header_fields_and_counts(self):
        artifact, features, split, manifest, report = artifact_fixture()
        cases = (
            (8, 2, "database version"),
            (12, 0, "database endian marker"),
            (16, 24, "database fps"),
            (20, 2, "database fps"),
            (24, 0, "database frame count"),
            (24, 0xFFFFFFFF, "database frame count.*file size"),
            (28, 30, "database bone count"),
            (32, 0, "database clip count"),
            (32, 0xFFFFFFFF, "database clip count.*file size"),
            (36, 13, "database hand dof count"),
        )
        for offset, value, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                replace_u32(
                    output / "interaction_database.bin", offset, value
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_rejects_invalid_feature_header_fields_counts_and_groups(self):
        artifact, features, split, manifest, report = artifact_fixture()
        cases = (
            (8, 2, "feature version"),
            (12, 0, "feature endian marker"),
            (16, 0, "feature frame count"),
            (16, 0xFFFFFFFF, "feature frame count.*file size"),
            (20, 70, "feature dimension"),
            (24, 4, "feature group count"),
            (28, 1, "feature group ranges"),
            (28 + 5 * 4, 32, "feature group ranges"),
        )
        for offset, value, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                replace_u32(
                    output / "interaction_features.bin", offset, value
                )
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_reader_revalidates_binary_ranges_and_nonfinite_arrays(self):
        artifact, features, split, manifest, report = artifact_fixture()
        frame_count = len(artifact.positions)
        clip_count = len(artifact.range_starts)
        offsets = database_offsets(frame_count, clip_count)
        cases = (
            (
                "range start",
                "interaction_database.bin",
                offsets["range_starts"],
                struct.pack("<i", 1),
                "range coverage.*start at 0",
            ),
            (
                "database nan",
                "interaction_database.bin",
                offsets["positions"],
                struct.pack("<f", np.nan),
                "positions.*non-finite",
            ),
            (
                "feature nan",
                "interaction_features.bin",
                FEATURE_HEADER.size + 40,
                struct.pack("<f", np.nan),
                "feature offsets.*non-finite",
            ),
        )
        for label, filename, offset, replacement, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                path = output / filename
                data = bytearray(path.read_bytes())
                data[offset:offset + len(replacement)] = replacement
                path.write_bytes(data)
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_reader_rejects_cross_file_frame_and_manifest_range_mismatch(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            path = output / "interaction_features.bin"
            data = bytearray(path.read_bytes())
            frame_count = len(features.values) - 1
            data[16:20] = struct.pack("<I", frame_count)
            del data[-71 * 4:]
            path.write_bytes(data)
            with self.assertRaisesRegex(
                ValueError, "feature/database frame count mismatch"
            ):
                read_artifact_set(output)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            rewrite_json(
                output / "manifest.json",
                lambda value: value["clips"][0].__setitem__(
                    "range_stop", value["clips"][0]["range_stop"] - 1
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "manifest clip ranges.*database ranges"
            ):
                read_artifact_set(output)

    def test_reader_rejects_malformed_json_schema_skeleton_and_split(self):
        artifact, features, split, manifest, report = artifact_fixture()

        def invalid_json(path: Path) -> None:
            path.write_text("{not json}\n", encoding="utf-8")

        def wrong_manifest_version(path: Path) -> None:
            rewrite_json(
                path,
                lambda value: value.__setitem__("schema_version", 2),
            )

        def wrong_manifest_fps(path: Path) -> None:
            rewrite_json(
                path, lambda value: value.__setitem__("target_fps", 50.0)
            )

        def wrong_skeleton(path: Path) -> None:
            rewrite_json(
                path,
                lambda value: value.__setitem__(
                    "skeleton_signature", "wrong"
                ),
            )

        def wrong_report_version(path: Path) -> None:
            rewrite_json(
                path,
                lambda value: value.__setitem__("schema_version", 2),
            )

        def wrong_report_frames(path: Path) -> None:
            rewrite_json(
                path,
                lambda value: value.__setitem__("included_frames", 0),
            )

        def heldout_database_object(path: Path) -> None:
            rewrite_json(
                path,
                lambda value: (
                    value.__setitem__("database_objects", ["other"]),
                    value.__setitem__(
                        "heldout_objects", ["database_fixture_object"]
                    ),
                ),
            )

        def nan_report(path: Path) -> None:
            value = json.loads(path.read_text(encoding="utf-8"))
            value["numeric_bounds"]["fk_max_error_m"] = float("nan")
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")

        cases = (
            (
                "manifest.json",
                invalid_json,
                "invalid manifest JSON",
            ),
            (
                "manifest.json",
                wrong_manifest_version,
                "manifest schema_version.*1",
            ),
            (
                "manifest.json",
                wrong_manifest_fps,
                "manifest target_fps.*25",
            ),
            (
                "manifest.json",
                wrong_skeleton,
                "manifest skeleton_signature.*G1",
            ),
            (
                "validation_report.json",
                wrong_report_version,
                "validation report schema_version.*1",
            ),
            (
                "validation_report.json",
                wrong_report_frames,
                "validation report included_frames.*database frame count",
            ),
            (
                "evaluation_split.json",
                heldout_database_object,
                "manifest object.*database split",
            ),
            (
                "validation_report.json",
                nan_report,
                "validation report JSON.*non-finite",
            ),
        )
        for filename, mutation, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                mutation(output / filename)
                with self.assertRaisesRegex(ValueError, message):
                    read_artifact_set(output)

    def test_writer_rejects_invalid_features_and_numpy_json_values(self):
        artifact, features, split, manifest, report = artifact_fixture()
        cases = []

        wrong_dimension = copy.deepcopy(features)
        wrong_dimension.values = wrong_dimension.values[:, :-1]
        cases.append((wrong_dimension, manifest, "feature dimension.*71"))

        wrong_groups = copy.deepcopy(features)
        wrong_groups.groups = (
            FeatureGroup("pose", 0, 32),
            *FEATURE_GROUPS[1:],
        )
        cases.append((wrong_groups, manifest, "feature groups"))

        nonpositive_scale = copy.deepcopy(features)
        nonpositive_scale.scales[0] = 0.0
        cases.append((nonpositive_scale, manifest, "feature scales.*positive"))

        nonfinite = copy.deepcopy(features)
        nonfinite.values[0, 0] = np.nan
        cases.append((nonfinite, manifest, "feature values.*non-finite"))

        numpy_manifest = copy.deepcopy(manifest)
        numpy_manifest["schema_version"] = np.int64(1)
        cases.append((features, numpy_manifest, "JSON serializable"))

        for value, metadata, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    write_artifact_set(
                        output,
                        artifact,
                        value,
                        split,
                        metadata,
                        report,
                    )
                self.assertFalse(output.exists())
                self.assertEqual(private_publish_paths(output), [])

    def test_failed_publish_preserves_previous_output_and_cleans_private_paths(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            before = snapshot(output)
            features.values[0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )

            self.assertEqual(before, snapshot(output))
            self.assertEqual(private_publish_paths(output), [])

    def test_nonfinite_json_publish_preserves_previous_output(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            before = snapshot(output)
            invalid_report = copy.deepcopy(report)
            invalid_report["numeric_bounds"]["fk_max_error_m"] = np.nan

            with self.assertRaisesRegex(
                ValueError, "non-finite JSON value"
            ):
                write_artifact_set(
                    output,
                    artifact,
                    features,
                    split,
                    manifest,
                    invalid_report,
                )

            self.assertEqual(before, snapshot(output))
            self.assertEqual(private_publish_paths(output), [])

    def test_failed_final_rename_restores_previous_complete_directory(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            before = snapshot(output)
            changed_manifest = copy.deepcopy(manifest)
            changed_manifest["generation"] = 2
            real_replace = os.replace

            def fail_new_directory_replace(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path.name.startswith(".pack.tmp-")
                    and destination_path == output
                ):
                    raise OSError("simulated final rename failure")
                return real_replace(source, destination)

            with patch(
                "resources.g1_interaction_builder.artifacts.os.replace",
                side_effect=fail_new_directory_replace,
            ):
                with self.assertRaisesRegex(
                    OSError, "simulated final rename failure"
                ):
                    write_artifact_set(
                        output,
                        artifact,
                        features,
                        split,
                        changed_manifest,
                        report,
                    )

            self.assertEqual(before, snapshot(output))
            self.assertEqual(private_publish_paths(output), [])
            read_artifact_set(output)


if __name__ == "__main__":
    unittest.main()
