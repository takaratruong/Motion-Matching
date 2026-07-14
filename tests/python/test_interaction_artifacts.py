import copy
from collections import Counter
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from resources.g1_interaction_builder import artifacts as artifact_io
from resources.g1_interaction_builder.artifacts import (
    DB_MAGIC,
    FEATURE_MAGIC,
    _read_database,
    _read_features,
    _write_array,
    _write_database,
    _write_features,
    assemble_database,
    read_artifact_set,
    write_artifact_set,
)
from resources.g1_interaction_builder.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    build_features,
    serialized_feature_groups,
)
from resources.g1_interaction_builder.metadata import (
    DEPENDENCY_VERSION_KEYS,
    GRAIL_DATASET_ID,
)
from resources.g1_interaction_builder.schema import (
    EvaluationSplit,
    FeatureGroup,
    FeatureSet,
    G1_SKELETON,
    InteractionArtifact,
    InteractionPhase,
    InteractionValidationError,
    PhaseConfig,
)
from resources.g1_terrain_builder.schema import SkeletonSpec
from tests.python.interaction_fixture import labeled_clips_for_objects


LITERAL_DB_HEADER_BYTES = 40
LITERAL_FEATURE_HEADER_BYTES = 28


def required_manifest_metadata(split: EvaluationSplit) -> dict:
    return {
        "database_magic": DB_MAGIC.decode("ascii"),
        "feature_magic": FEATURE_MAGIC.decode("ascii"),
        "skeleton_names": list(G1_SKELETON.names),
        "skeleton_parents": G1_SKELETON.parents.astype(int).tolist(),
        "source_root": "/synthetic/grail/data/pickup_table",
        "dataset_id": GRAIL_DATASET_ID,
        "phase_config": dataclasses.asdict(PhaseConfig()),
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": serialized_feature_groups(),
        "split": {
            "seed": split.seed,
            "database_object_count": len(split.database_objects),
            "heldout_object_count": len(split.heldout_objects),
        },
        "dependency_versions": {
            name: "test-version" for name in DEPENDENCY_VERSION_KEYS
        },
        "git_commit": "0" * 40,
        "diagnostic_limit": None,
        "source_date_epoch": None,
    }


def artifact_fixture():
    clips = labeled_clips_for_objects(
        ["database_fixture_object", "heldout_fixture_object"]
    )
    split = EvaluationSplit(
        seed=20260714,
        database_objects=("database_fixture_object",),
        heldout_objects=("heldout_fixture_object",),
    )
    ordered, artifact, features = artifact_io.prepare_artifacts(
        clips, split, G1_SKELETON
    )
    labeled = ordered[0]
    manifest = {
        **required_manifest_metadata(split),
        "schema_version": 1,
        "source_clips": 2,
        "included_clips": 2,
        "rejected_clips": 0,
        "target_fps": 25.0,
        "skeleton_signature": G1_SKELETON.signature(),
        "clips": [
            {
                "sequence_id": labeled.motion.sequence_id,
                "object_id": labeled.motion.object_id,
                "active_hand": int(labeled.active_hand),
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
    return sorted(output.parent.glob(f".{output.name}.tmp-*")) + sorted(
        output.parent.glob(f".{output.name}.previous-*")
    )


def database_offsets(frame_count: int, clip_count: int) -> dict[str, int]:
    offset = LITERAL_DB_HEADER_BYTES
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
    return LITERAL_FEATURE_HEADER_BYTES + 2 * 5 * 4 + 2 * 71 * 4 + (
        frame_count * 71 * 4
    )


def _literal_array_bytes(value: np.ndarray, dtype: str) -> bytes:
    array = np.asarray(
        value,
        dtype=np.dtype(dtype).newbyteorder("<"),
        order="C",
    )
    return array.tobytes(order="C")


def literal_database_wire(value: InteractionArtifact) -> bytes:
    frame_count, bone_count = value.positions.shape[:2]
    clip_count = len(value.range_starts)
    pieces = [
        b"G1INTDB1",
        struct.pack(
            "<8I",
            1,
            0x01020304,
            25,
            1,
            frame_count,
            bone_count,
            clip_count,
            14,
        ),
    ]
    for array, dtype in (
        (value.parents, "i4"),
        (value.range_starts, "i4"),
        (value.range_stops, "i4"),
        (value.positions, "f4"),
        (value.velocities, "f4"),
        (value.rotations, "f4"),
        (value.angular_velocities, "f4"),
        (value.foot_contacts, "u1"),
        (value.hand_contacts, "u1"),
        (value.hand_dof, "f4"),
        (value.hand_dof_velocities, "f4"),
        (value.phases, "u1"),
        (value.active_hands, "u1"),
        (value.time_to_contact, "f4"),
        (value.object_positions, "f4"),
        (value.object_rotations, "f4"),
        (value.object_velocities, "f4"),
        (value.object_angular_velocities, "f4"),
        (value.table_positions, "f4"),
        (value.table_rotations, "f4"),
        (value.table_sizes, "f4"),
        (value.object_dimensions, "f4"),
        (value.grasp_positions_object, "f4"),
        (value.grasp_rotations_object, "f4"),
        (value.approach_directions_object, "f4"),
        (value.source_frames, "i4"),
    ):
        pieces.append(_literal_array_bytes(array, dtype))
    return b"".join(pieces)


def literal_feature_wire(value: FeatureSet) -> bytes:
    pieces = [
        b"G1INTFT1",
        struct.pack(
            "<5I",
            1,
            0x01020304,
            len(value.values),
            71,
            5,
        ),
        struct.pack("<5I", 0, 33, 45, 57, 65),
        struct.pack("<5I", 33, 45, 57, 65, 71),
        _literal_array_bytes(value.offsets, "f4"),
        _literal_array_bytes(value.scales, "f4"),
        _literal_array_bytes(value.values, "f4"),
    ]
    return b"".join(pieces)


def wire_oracle_fixture() -> tuple[InteractionArtifact, FeatureSet]:
    artifact, _, _, _, _ = artifact_fixture()
    artifact = copy.deepcopy(artifact)
    artifact.positions.fill(np.float32(1.25))
    artifact.velocities.fill(np.float32(2.25))
    artifact.rotations[:] = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    artifact.angular_velocities.fill(np.float32(3.25))
    artifact.foot_contacts.fill(0)
    artifact.hand_dof.fill(np.float32(4.25))
    artifact.hand_dof_velocities.fill(np.float32(5.25))
    artifact.object_positions.fill(np.float32(6.25))
    artifact.object_rotations[:] = np.array(
        [0.0, 1.0, 0.0, 0.0], np.float32
    )
    artifact.object_velocities.fill(np.float32(7.25))
    artifact.object_angular_velocities.fill(np.float32(8.25))
    artifact.table_positions.fill(np.float32(9.25))
    artifact.table_rotations[:] = np.array(
        [0.0, 0.0, 1.0, 0.0], np.float32
    )
    artifact.table_sizes.fill(np.float32(10.25))
    artifact.object_dimensions.fill(np.float32(11.25))
    artifact.grasp_positions_object.fill(np.float32(12.25))
    artifact.grasp_rotations_object[:] = np.array(
        [0.0, 0.0, 0.0, 1.0], np.float32
    )
    artifact.approach_directions_object[:] = np.array(
        [0.6, 0.0, 0.8], np.float32
    )
    artifact.source_frames[:] = np.arange(
        37, 37 + len(artifact.source_frames), dtype=np.int32
    )
    artifact.validate()

    groups = (
        FeatureGroup("pose", 0, 33),
        FeatureGroup("trajectory", 33, 45),
        FeatureGroup("grasp", 45, 57),
        FeatureGroup("root_target", 57, 65),
        FeatureGroup("context", 65, 71),
    )
    offsets = (np.arange(71, dtype=np.float32) + np.float32(20.25))
    scales = (np.arange(71, dtype=np.float32) + np.float32(100.5))
    values = (
        np.arange(3 * 71, dtype=np.float32).reshape(3, 71)
        + np.float32(1000.75)
    )
    return artifact, FeatureSet(values, offsets, scales, groups)


class GuardedReadStream(io.BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.requests: list[int] = []

    def read(self, count: int = -1) -> bytes:
        if count >= 0:
            self.requests.append(count)
            remaining = len(self.getbuffer()) - self.tell()
            if count > remaining:
                raise AssertionError(
                    f"attempted amplified read {count} with {remaining} remaining"
                )
        return super().read(count)


class GuardedBinaryPath:
    def __init__(self, value: bytes):
        self.value = value
        self.stream = GuardedReadStream(value)

    def stat(self) -> SimpleNamespace:
        return SimpleNamespace(st_size=len(self.value))

    def open(self, mode: str) -> GuardedReadStream:
        if mode != "rb":
            raise AssertionError(f"unexpected mode {mode}")
        return self.stream


def multi_artifact_fixture():
    clips = labeled_clips_for_objects(
        ["z_database", "a_database", "heldout_fixture_object"]
    )
    split = EvaluationSplit(
        seed=20260714,
        database_objects=("a_database", "z_database"),
        heldout_objects=("heldout_fixture_object",),
    )
    ordered, artifact, features = artifact_io.prepare_artifacts(
        clips, split, G1_SKELETON
    )
    manifest_clips = []
    start = 0
    for clip in ordered:
        stop = start + len(clip.motion.positions)
        manifest_clips.append(
            {
                "sequence_id": clip.motion.sequence_id,
                "object_id": clip.motion.object_id,
                "active_hand": int(clip.active_hand),
                "range_start": start,
                "range_stop": stop,
            }
        )
        start = stop
    manifest = {
        **required_manifest_metadata(split),
        "schema_version": 1,
        "source_clips": 3,
        "included_clips": 3,
        "rejected_clips": 0,
        "target_fps": 25.0,
        "skeleton_signature": G1_SKELETON.signature(),
        "clips": manifest_clips,
    }
    report = {
        "schema_version": 1,
        "source_clips": 3,
        "included_clips": 3,
        "rejected_clips": 0,
        "included_frames": len(artifact.positions),
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


def rejection_record(
    sequence_id: str,
    code: str,
    stage: str = "source",
) -> dict[str, str]:
    return {
        "sequence_id": sequence_id,
        "object_id": "rejected_object",
        "stage": stage,
        "code": code,
        "message": f"{code}: actionable detail",
    }


class InteractionArtifactAssemblyTests(unittest.TestCase):
    def test_assembly_makes_every_persisted_array_c_contiguous(self):
        labeled = labeled_clips_for_objects(["database"])[0]
        labeled.motion.foot_contacts = np.asfortranarray(
            labeled.motion.foot_contacts
        )
        self.assertFalse(labeled.motion.foot_contacts.flags.c_contiguous)

        artifact = assemble_database([labeled], G1_SKELETON)

        for field in dataclasses.fields(InteractionArtifact):
            value = getattr(artifact, field.name)
            if isinstance(value, np.ndarray):
                self.assertTrue(
                    value.flags.c_contiguous,
                    msg=f"non-C-contiguous artifact field {field.name}",
                )

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

    def test_prepare_artifacts_filters_and_aligns_one_sorted_database_tuple(
        self,
    ):
        clips = labeled_clips_for_objects(
            ["z_database", "heldout", "a_database"]
        )
        clips[0].motion.positions[:, 0, 0] = 9.0
        clips[0].motion.object_dimensions[:] = [0.9, 0.8, 0.7]
        clips[2].motion.positions[:, 0, 0] = 1.0
        clips[2].motion.object_dimensions[:] = [0.1, 0.2, 0.3]
        split = EvaluationSplit(
            seed=20260714,
            database_objects=("a_database", "z_database"),
            heldout_objects=("heldout",),
        )

        ordered, artifact, features = artifact_io.prepare_artifacts(
            clips, split, G1_SKELETON
        )

        self.assertEqual(
            [clip.motion.sequence_id for clip in ordered],
            sorted(
                clip.motion.sequence_id
                for clip in clips
                if clip.motion.object_id != "heldout"
            ),
        )
        frames = len(ordered[0].motion.positions)
        np.testing.assert_array_equal(
            artifact.positions[:frames], ordered[0].motion.positions
        )
        np.testing.assert_array_equal(
            artifact.positions[frames:], ordered[1].motion.positions
        )
        expected_features = build_features(ordered, G1_SKELETON)
        np.testing.assert_array_equal(features.values, expected_features.values)
        np.testing.assert_array_equal(
            features.offsets, expected_features.offsets
        )
        np.testing.assert_array_equal(features.scales, expected_features.scales)

    def test_prepare_artifacts_rejects_duplicate_sequence_ids(self):
        clips = labeled_clips_for_objects(["database_a", "database_b", "heldout"])
        clips[1].motion.sequence_id = clips[0].motion.sequence_id
        split = EvaluationSplit(
            seed=20260714,
            database_objects=("database_a", "database_b"),
            heldout_objects=("heldout",),
        )

        with self.assertRaisesRegex(
            InteractionValidationError, "duplicate sequence_id"
        ):
            artifact_io.prepare_artifacts(clips, split, G1_SKELETON)

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
        two_ranges.time_to_contact[:] = 0.0
        two_ranges.hand_contacts[:, int(two_ranges.active_hands[0])] = 1
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

    def test_validate_rejects_invalid_derived_time_and_approach_semantics(self):
        artifact, _, _, _, _ = artifact_fixture()
        contact = int(
            np.flatnonzero(artifact.phases == InteractionPhase.CONTACT)[0]
        )
        cases = (
            (
                "negative time",
                lambda value: value.time_to_contact.__setitem__(0, -0.01),
                "time_to_contact.*nonnegative.*range 0",
            ),
            (
                "increasing time",
                lambda value: value.time_to_contact.__setitem__(
                    1, value.time_to_contact[0] + 0.01
                ),
                "time_to_contact.*nonincreasing.*range 0",
            ),
            (
                "nonzero after contact",
                lambda value: value.time_to_contact.__setitem__(contact, 0.01),
                "time_to_contact.*zero from CONTACT.*range 0",
            ),
            (
                "nonfinite approach",
                lambda value: value.approach_directions_object.__setitem__(
                    (0, 0), np.nan
                ),
                "approach_directions_object.*non-finite",
            ),
            (
                "vertical approach",
                lambda value: value.approach_directions_object.__setitem__(
                    (0, 1), 0.01
                ),
                "approach direction.*horizontal.*range 0",
            ),
            (
                "nonunit approach",
                lambda value: value.approach_directions_object.__setitem__(
                    0, value.approach_directions_object[0] * 2.0
                ),
                "approach direction.*unit.*range 0",
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

    def test_validate_rejects_missing_active_contact_through_hold_window(self):
        artifact, _, _, _, _ = artifact_fixture()
        active = int(artifact.active_hands[0])
        contact = np.flatnonzero(artifact.phases == InteractionPhase.CONTACT)
        lift = np.flatnonzero(artifact.phases == InteractionPhase.LIFT)
        hold = np.flatnonzero(artifact.phases == InteractionPhase.HOLD)
        cases = (
            ("CONTACT", int(contact[0]), "CONTACT.*active-hand contact.*range 0"),
            ("LIFT", int(lift[0]), "LIFT.*active-hand contact.*range 0"),
            (
                "HOLD",
                int(hold[4]),
                "first 5 HOLD samples.*active-hand contact.*range 0",
            ),
        )
        for label, frame, message in cases:
            with self.subTest(label=label):
                invalid = copy.deepcopy(artifact)
                invalid.hand_contacts[frame, active] = 0
                with self.assertRaisesRegex(
                    InteractionValidationError, message
                ):
                    invalid.validate()

        contact_may_end_later = copy.deepcopy(artifact)
        contact_may_end_later.hand_contacts[int(hold[5]), active] = 0
        contact_may_end_later.validate()

    def test_validate_accepts_documented_float_tolerance_noise(self):
        artifact, _, _, _, _ = artifact_fixture()
        contact_or_later = artifact.phases >= InteractionPhase.CONTACT
        artifact.time_to_contact[contact_or_later] = np.float32(5e-5)
        artifact.approach_directions_object[0] = np.array(
            [np.sqrt(1.0 - 5e-10), 5e-5, 0.0], np.float32
        )

        artifact.validate()

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

    def test_writer_matches_independent_literal_wire_oracle(self):
        artifact, features = wire_oracle_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "database.bin"
            feature_path = Path(tmp) / "features.bin"
            _write_database(database_path, artifact)
            _write_features(feature_path, features)

            database = database_path.read_bytes()
            feature_bytes = feature_path.read_bytes()

        expected_database = literal_database_wire(artifact)
        expected_features = literal_feature_wire(features)
        self.assertEqual(LITERAL_DB_HEADER_BYTES, 40)
        self.assertEqual(LITERAL_FEATURE_HEADER_BYTES, 28)
        self.assertEqual(database, expected_database)
        self.assertEqual(feature_bytes, expected_features)
        offsets = database_offsets(
            len(artifact.positions), len(artifact.range_starts)
        )
        self.assertEqual(offsets["end"], len(expected_database))
        self.assertEqual(
            len(expected_features), expected_feature_size(len(features.values))
        )
        self.assertEqual(expected_database[:8], b"G1INTDB1")
        self.assertEqual(expected_features[:8], b"G1INTFT1")
        self.assertEqual(expected_database[12:16], b"\x04\x03\x02\x01")
        self.assertEqual(expected_features[12:16], b"\x04\x03\x02\x01")

    def test_reader_accepts_independent_literal_wire_fixture(self):
        artifact, features = wire_oracle_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            database_path = Path(tmp) / "database.bin"
            feature_path = Path(tmp) / "features.bin"
            database_path.write_bytes(literal_database_wire(artifact))
            feature_path.write_bytes(literal_feature_wire(features))

            loaded_artifact = _read_database(database_path)
            loaded_features = _read_features(feature_path)

        assert_artifact_equal(self, artifact, loaded_artifact)
        np.testing.assert_array_equal(features.values, loaded_features.values)
        np.testing.assert_array_equal(
            features.offsets, loaded_features.offsets
        )
        np.testing.assert_array_equal(features.scales, loaded_features.scales)
        self.assertEqual(features.groups, loaded_features.groups)

    def test_database_hostile_count_never_attempts_amplified_read(self):
        file_size = 1 << 20
        frame_count = file_size
        amplified_request = frame_count * 31 * 3 * 4
        self.assertEqual(amplified_request, 390_070_272)
        prefix = b"G1INTDB1" + struct.pack(
            "<8I",
            1,
            0x01020304,
            25,
            1,
            frame_count,
            31,
            1,
            14,
        )
        path = GuardedBinaryPath(prefix + bytes(file_size - len(prefix)))

        with self.assertRaisesRegex(ValueError, "truncated positions"):
            _read_database(path)

        self.assertLessEqual(max(path.stream.requests), 31 * 4)
        self.assertNotIn(amplified_request, path.stream.requests)

    def test_feature_hostile_count_never_attempts_amplified_read(self):
        file_size = 1 << 20
        frame_count = file_size
        amplified_request = frame_count * 71 * 4
        self.assertEqual(amplified_request, 297_795_584)
        prefix = b"".join(
            (
                b"G1INTFT1",
                struct.pack(
                    "<5I", 1, 0x01020304, frame_count, 71, 5
                ),
                struct.pack("<5I", 0, 33, 45, 57, 65),
                struct.pack("<5I", 33, 45, 57, 65, 71),
                struct.pack("<71f", *([0.0] * 71)),
                struct.pack("<71f", *([1.0] * 71)),
            )
        )
        path = GuardedBinaryPath(prefix + bytes(file_size - len(prefix)))

        with self.assertRaisesRegex(ValueError, "truncated feature values"):
            _read_features(path)

        self.assertLessEqual(max(path.stream.requests), 71 * 4)
        self.assertNotIn(amplified_request, path.stream.requests)

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
                LITERAL_FEATURE_HEADER_BYTES + 40,
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

    def test_manifest_requires_sorted_ids_and_exact_database_objects(self):
        artifact, features, split, manifest, report = multi_artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )

            def reverse_sequence_ids(value: dict) -> None:
                sequence_ids = [
                    clip["sequence_id"] for clip in value["clips"]
                ][::-1]
                for clip, sequence_id in zip(
                    value["clips"], sequence_ids
                ):
                    clip["sequence_id"] = sequence_id

            rewrite_json(
                output / "manifest.json",
                reverse_sequence_ids,
            )
            with self.assertRaisesRegex(
                ValueError, "manifest sequence_ids.*lexicographically sorted"
            ):
                read_artifact_set(output)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            rewrite_json(
                output / "evaluation_split.json",
                lambda value: value.__setitem__(
                    "database_objects",
                    ["a_database", "unrepresented", "z_database"],
                ),
            )
            rewrite_json(
                output / "manifest.json",
                lambda value: value["split"].__setitem__(
                    "database_object_count", 3
                ),
            )
            with self.assertRaisesRegex(
                ValueError,
                "manifest database object identities.*exactly match",
            ):
                read_artifact_set(output)

    def test_manifest_and_report_counts_are_cross_file_consistent(self):
        artifact, features, split, manifest, report = artifact_fixture()

        def count_values(source: int, included: int, rejected: int) -> dict:
            return {
                "source_clips": source,
                "included_clips": included,
                "rejected_clips": rejected,
            }

        cases = []
        inconsistent_manifest = copy.deepcopy(manifest)
        inconsistent_manifest.update(count_values(99, 2, 0))
        cases.append(
            (
                "manifest arithmetic",
                inconsistent_manifest,
                report,
                "manifest source_clips.*included_clips plus rejected_clips",
            )
        )

        different_manifest = copy.deepcopy(manifest)
        different_manifest.update(count_values(2, 1, 1))
        cases.append(
            (
                "cross-file mismatch",
                different_manifest,
                report,
                "manifest/report.*included_clips.*mismatch",
            )
        )

        too_few_included_manifest = copy.deepcopy(manifest)
        too_few_included_report = copy.deepcopy(report)
        too_few_included_manifest.update(count_values(0, 0, 0))
        too_few_included_report.update(count_values(0, 0, 0))
        cases.append(
            (
                "artifact clips exceed included",
                too_few_included_manifest,
                too_few_included_report,
                "database artifact clip count.*included_clips",
            )
        )

        for label, manifest_value, report_value, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                with self.assertRaisesRegex(ValueError, message):
                    write_artifact_set(
                        output,
                        artifact,
                        features,
                        split,
                        manifest_value,
                        report_value,
                    )

    def test_report_rejections_require_exact_shape_histogram_and_order(self):
        artifact, features, split, manifest, report = artifact_fixture()

        def values_with(
            records: list[dict],
            histogram: dict[str, int],
            rejected: int,
        ) -> tuple[dict, dict]:
            manifest_value = copy.deepcopy(manifest)
            report_value = copy.deepcopy(report)
            counts = {
                "source_clips": 2 + rejected,
                "included_clips": 2,
                "rejected_clips": rejected,
            }
            manifest_value.update(counts)
            report_value.update(counts)
            report_value["rejections"] = records
            report_value["rejections_by_code"] = histogram
            return manifest_value, report_value

        one = rejection_record("pickup_table__bad__001", "missing_field")
        missing_message = dict(one)
        del missing_message["message"]
        wrong_type = dict(one)
        wrong_type["stage"] = 3
        unsorted = [
            rejection_record("pickup_table__z__001", "missing_field"),
            rejection_record("pickup_table__a__001", "non_finite"),
        ]
        cases = (
            (
                "list length",
                *values_with([], {"missing_field": 1}, 1),
                "rejections length.*rejected_clips",
            ),
            (
                "histogram",
                *values_with([one], {"non_finite": 1}, 1),
                "rejections_by_code.*exact.*histogram",
            ),
            (
                "shape",
                *values_with([missing_message], {"missing_field": 1}, 1),
                "rejection 0 fields.*message",
            ),
            (
                "type",
                *values_with([wrong_type], {"missing_field": 1}, 1),
                "rejection 0 stage.*nonempty string",
            ),
            (
                "order",
                *values_with(
                    unsorted,
                    dict(Counter(record["code"] for record in unsorted)),
                    2,
                ),
                "rejections must be sorted.*sequence_id.*stage.*code",
            ),
        )
        for label, manifest_value, report_value, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                with self.assertRaisesRegex(ValueError, message):
                    write_artifact_set(
                        output,
                        artifact,
                        features,
                        split,
                        manifest_value,
                        report_value,
                    )

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
                "manifest database object identities.*exactly match",
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

    def test_stale_previous_is_restored_before_invalid_new_inputs(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            before = snapshot(output)
            previous = output.parent / ".pack.previous-424242"
            os.replace(output, previous)
            invalid_features = copy.deepcopy(features)
            invalid_features.values[0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_artifact_set(
                    output,
                    artifact,
                    invalid_features,
                    split,
                    manifest,
                    report,
                )

            self.assertEqual(before, snapshot(output))
            self.assertFalse(previous.exists())
            read_artifact_set(output)

    def test_valid_previous_replaces_invalid_public_pack_before_input_failure(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            before = snapshot(output)
            previous = output.parent / ".pack.previous-424243"
            shutil.copytree(output, previous)
            mutate_bad_magic(output / "interaction_database.bin")
            invalid_features = copy.deepcopy(features)
            invalid_features.values[0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_artifact_set(
                    output,
                    artifact,
                    invalid_features,
                    split,
                    manifest,
                    report,
                )

            self.assertEqual(before, snapshot(output))
            self.assertFalse(previous.exists())
            read_artifact_set(output)

    def test_valid_public_pack_wins_over_stale_previous_before_input_failure(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )
            before = snapshot(output)
            previous = output.parent / ".pack.previous-424244"
            shutil.copytree(output, previous)
            rewrite_json(
                previous / "manifest.json",
                lambda value: value.__setitem__("generation", "older"),
            )
            invalid_features = copy.deepcopy(features)
            invalid_features.values[0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_artifact_set(
                    output,
                    artifact,
                    invalid_features,
                    split,
                    manifest,
                    report,
                )

            self.assertEqual(before, snapshot(output))
            self.assertFalse(previous.exists())
            read_artifact_set(output)

    def test_invalid_sole_previous_is_never_deleted_unvalidated(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            previous = output.parent / f".pack.previous-{os.getpid()}"
            previous.mkdir()
            (previous / "only-copy.txt").write_text(
                "must survive", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, "cannot recover.*previous artifact"
            ):
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )

            self.assertFalse(output.exists())
            self.assertTrue(previous.is_dir())
            self.assertEqual(
                (previous / "only-copy.txt").read_text(encoding="utf-8"),
                "must survive",
            )

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


class CrossLanguageProbeTests(unittest.TestCase):
    def test_probe_replays_the_published_python_artifacts(self):
        artifact, features, split, manifest, report = artifact_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pack"
            write_artifact_set(
                output, artifact, features, split, manifest, report
            )

            completed = subprocess.run(
                ["./interaction_probe", str(output), "--json"],
                check=True,
                capture_output=True,
                text=True,
            )
            actual = json.loads(completed.stdout)

            database_path = output / "interaction_database.bin"
            feature_path = output / "interaction_features.bin"
            quaternion_arrays = (
                artifact.rotations,
                artifact.object_rotations,
                artifact.table_rotations,
                artifact.grasp_rotations_object,
            )
            expected_quaternion_error = max(
                float(
                    np.max(
                        np.abs(
                            np.linalg.norm(
                                values.astype(np.float64), axis=-1
                            )
                            - 1.0
                        )
                    )
                )
                for values in quaternion_arrays
            )
            expected = {
                "bone_count": artifact.positions.shape[1],
                "clip_count": len(artifact.range_starts),
                "database_sha256": hashlib.sha256(
                    database_path.read_bytes()
                ).hexdigest(),
                "feature_count": features.values.shape[1],
                "feature_sha256": hashlib.sha256(
                    feature_path.read_bytes()
                ).hexdigest(),
                "first_source_frame": int(artifact.source_frames[0]),
                "frame_count": len(artifact.positions),
                "last_source_frame": int(artifact.source_frames[-1]),
                "phase_counts": [
                    int(np.count_nonzero(artifact.phases == phase))
                    for phase in range(5)
                ],
            }

            self.assertEqual(list(actual), sorted(actual))
            self.assertTrue(completed.stdout.endswith("\n"))
            self.assertNotIn(" ", completed.stdout.rstrip("\n"))
            self.assertEqual(completed.stderr, "")
            actual_quaternion_error = actual.pop(
                "max_quaternion_norm_error"
            )
            self.assertEqual(actual, expected)
            self.assertAlmostEqual(
                actual_quaternion_error,
                expected_quaternion_error,
                places=8,
            )

    def test_probe_requires_all_three_json_artifacts(self):
        artifact, features, split, manifest, report = artifact_fixture()
        for filename in (
            "manifest.json",
            "evaluation_split.json",
            "validation_report.json",
        ):
            with self.subTest(
                filename=filename
            ), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "pack"
                write_artifact_set(
                    output, artifact, features, split, manifest, report
                )
                (output / filename).unlink()

                completed = subprocess.run(
                    ["./interaction_probe", str(output), "--json"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(filename, completed.stderr)


if __name__ == "__main__":
    unittest.main()
