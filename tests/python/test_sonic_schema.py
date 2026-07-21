import json
from pathlib import Path
from types import MappingProxyType
import unittest

import numpy as np

from mm_sonic.joints import load_joint_contract
from mm_sonic.schema import (
    ContractError,
    InitialBoundary,
    JointFeasibilityIdentity,
    SourceChunk,
    loads_exact,
    parse_initial_boundary,
    parse_joint_feasibility_identity,
    parse_source_chunk,
)


ROOT = Path(__file__).resolve().parents[2]
JOINT_CONTRACT = ROOT / "sonic" / "configs" / "g1_joint_contract.json"
SOURCE_SCHEMA = ROOT / "sonic" / "schemas" / "mm_chunk_v1.schema.json"
TARGET_SCHEMA = ROOT / "sonic" / "schemas" / "target_chunk_v1.schema.json"


def f32(value):
    return float(np.float32(value))


def joint_feasibility_identity():
    violations = [0] * 29
    for index, count in {
        4: 20,
        5: 938,
        9: 4,
        10: 14,
        11: 52,
        14: 12,
        18: 13,
        25: 10,
    }.items():
        violations[index] = count
    return {
        "schema": "g1-joint-feasibility-certificate/v1",
        "frame_count": 459682,
        "raw_safe_count": 458619,
        "raw_unsafe_count": 1063,
        "search_safe_count": 458000,
        "joint_limit_violation_count": violations,
        "mask_sha256": "a" * 64,
    }


class JointFeasibilityIdentityTests(unittest.TestCase):
    def test_valid_real_shaped_identity_is_frozen_and_owned(self):
        source = joint_feasibility_identity()
        parsed = parse_joint_feasibility_identity(source)
        self.assertEqual(
            parsed,
            JointFeasibilityIdentity(
                schema="g1-joint-feasibility-certificate/v1",
                frame_count=459682,
                raw_safe_count=458619,
                raw_unsafe_count=1063,
                search_safe_count=458000,
                joint_limit_violation_count=tuple(
                    source["joint_limit_violation_count"]
                ),
                mask_sha256="a" * 64,
            ),
        )
        source["joint_limit_violation_count"][5] = 0
        self.assertEqual(parsed.joint_limit_violation_count[5], 938)
        with self.assertRaises((AttributeError, TypeError)):
            parsed.raw_safe_count = 0

    def test_identity_requires_a_plain_dict_with_exact_keys_and_schema(self):
        valid = joint_feasibility_identity()
        cases = []
        missing = dict(valid)
        missing.pop("mask_sha256")
        cases.append(missing)
        extra = dict(valid)
        extra["unknown"] = 1
        cases.append(extra)
        schema = dict(valid)
        schema["schema"] = "g1-joint-feasibility-certificate/v2"
        cases.append(schema)
        cases.extend((MappingProxyType(valid), [], None))
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ContractError):
                parse_joint_feasibility_identity(value)

    def test_identity_rejects_wrong_count_types_bools_and_negative_counts(self):
        count_fields = (
            "frame_count",
            "raw_safe_count",
            "raw_unsafe_count",
            "search_safe_count",
        )
        for field in count_fields:
            for value in (True, False, 1.0, "1", None, -1):
                candidate = joint_feasibility_identity()
                candidate[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(
                    ContractError
                ):
                    parse_joint_feasibility_identity(candidate)

    def test_identity_reconciles_frame_and_search_counts(self):
        cases = (
            ("frame_count", 459681),
            ("raw_safe_count", 458618),
            ("raw_unsafe_count", 1062),
            ("search_safe_count", 0),
            ("search_safe_count", 458620),
        )
        for field, value in cases:
            candidate = joint_feasibility_identity()
            candidate[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(
                ContractError
            ):
                parse_joint_feasibility_identity(candidate)

    def test_identity_requires_exact_nonnegative_29_count_vector_and_sum(self):
        valid = joint_feasibility_identity()
        vectors = (
            tuple(valid["joint_limit_violation_count"]),
            valid["joint_limit_violation_count"][:-1],
            valid["joint_limit_violation_count"] + [0],
            [*valid["joint_limit_violation_count"][:5], -1, *([0] * 23)],
            [*valid["joint_limit_violation_count"][:5], True, *([0] * 23)],
            [*valid["joint_limit_violation_count"][:5], 1.0, *([0] * 23)],
            [0] * 29,
        )
        for vector in vectors:
            candidate = joint_feasibility_identity()
            candidate["joint_limit_violation_count"] = vector
            with self.subTest(vector=vector), self.assertRaises(ContractError):
                parse_joint_feasibility_identity(candidate)

    def test_identity_requires_a_lowercase_full_sha256_digest(self):
        for value in (
            "A" * 64,
            "a" * 63,
            "a" * 65,
            "g" * 64,
            "a" * 63 + "\n",
            7,
        ):
            candidate = joint_feasibility_identity()
            candidate["mask_sha256"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                parse_joint_feasibility_identity(candidate)


class SourceFixture:
    def __init__(self):
        self.contract = load_joint_contract(JOINT_CONTRACT)
        self.source_names = tuple(row.source_joint for row in self.contract.rows)
        self.target_names = tuple(
            next(
                row.target_name
                for row in self.contract.rows
                if row.target_index == target_index
            )
            for target_index in range(29)
        )

    def initial(self):
        return {
            "session_id": "session-0",
            "source_joint_names": list(self.source_names),
            "joint_position_source": [f32(index) for index in range(29)],
            "joint_velocity_source": [f32(-index) for index in range(29)],
            "physical_pelvis_position_holden": [f32(1.0), f32(2.0), f32(3.0)],
            "physical_pelvis_orientation_holden": [f32(1.0), 0.0, 0.0, 0.0],
            "virtual_root_position_holden": [f32(4.0), f32(5.0), f32(6.0)],
            "virtual_root_orientation_holden": [f32(1.0), 0.0, 0.0, 0.0],
        }

    def chunk(self, source_intervals=10):
        boundaries = range(source_intervals + 1)
        steps = range(source_intervals)
        return {
            "schema": "mm-chunk/v1",
            "session_id": "session-0",
            "candidate_id": "candidate-0",
            "predecessor_id": None,
            "source_rate_hz": 25,
            "source_intervals": source_intervals,
            "timestamps_s": [f32(index / 25.0) for index in boundaries],
            "source_joint_names": list(self.source_names),
            "target_joint_names": list(self.target_names),
            "joint_position_source": [
                [f32(boundary * 32 + joint) for joint in range(29)]
                for boundary in boundaries
            ],
            "joint_velocity_source": [
                [f32(-boundary * 32 - joint) for joint in range(29)]
                for boundary in boundaries
            ],
            "physical_pelvis_position_holden": [
                [f32(boundary), f32(2 * boundary), f32(-boundary)]
                for boundary in boundaries
            ],
            "physical_pelvis_orientation_holden": [
                [1.0, 0.0, 0.0, 0.0] for _ in boundaries
            ],
            "virtual_root_position_holden": [
                [f32(boundary), 0.0, f32(boundary + 1)]
                for boundary in boundaries
            ],
            "virtual_root_orientation_holden": [
                [1.0, 0.0, 0.0, 0.0] for _ in boundaries
            ],
            "selected_database_frame": [100 + step for step in steps],
            "candidate_preview_count": [2 if step == 2 else 0 for step in steps],
            "candidate_limit_rejection_count": [
                1 if step == 2 else 0 for step in steps
            ],
            "first_rejected_database_frame": [
                927 if step == 2 else -1 for step in steps
            ],
            "first_rejected_joint_index": [
                5 if step == 2 else -1 for step in steps
            ],
            "first_rejected_joint_position": [
                -0.27224052 if step == 2 else 0.0 for step in steps
            ],
            "searched": [bool(step % 2) for step in steps],
            "transitioned": [False for _ in steps],
            "terrain_cost": [f32(step / 8.0) for step in steps],
            "terrain_values": [
                [f32(step + sample) for sample in range(4)] for step in steps
            ],
            "terrain_points_holden": [
                [
                    [f32(step), f32(sample), f32(step + sample)]
                    for sample in range(4)
                ]
                for step in steps
            ],
            "support_height": [f32(step / 16.0) for step in steps],
            "support_target": [f32(step / 32.0) for step in steps],
            "scene": {
                "scene_id": "grail-curb-low",
                "route_id": "curb-forward",
                "terrain_weight": f32(4.0),
                "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
                "heightfield_sha256": "1" * 64,
                "mesh_sha256": "2" * 64,
                "walkability_sha256": "3" * 64,
            },
            "command": {
                "requested_velocity_holden": [0.0, 0.0, f32(0.5)],
                "desired_heading_holden_wxyz": [1.0, 0.0, 0.0, 0.0],
                "applied_velocity_holden": [
                    [0.0, 0.0, f32(step / 16.0)] for step in steps
                ],
            },
            "artifacts": {
                "build_commit": "77ae68c",
                "joint_contract_sha256": "4" * 64,
                "motion_manifest_sha256": "5" * 64,
                "database_sha256": "6" * 64,
                "terrain_features_sha256": "7" * 64,
                "terrain_support_sha256": "8" * 64,
                "scene_index_sha256": "9" * 64,
                "skeleton_signature": "a" * 64,
                "coordinate_signature": "holden-y-up-right-handed-forward-plus-z",
            },
        }

    def chunk_v2(self, source_intervals=5):
        payload = self.chunk(source_intervals=source_intervals)
        payload["schema"] = "mm-chunk/v2"
        payload["command"]["applied_heading_holden_wxyz"] = [
            [1.0, 0.0, 0.0, 0.0] for _ in range(source_intervals)
        ]
        return payload


class ExactJsonTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected_at_every_object_depth(self):
        texts = (
            '{"schema":"mm-chunk/v1","schema":"mm-chunk/v1"}',
            '{"scene":{"scene_id":"a","scene_id":"b"}}',
            '{"command":{"nested":{"x":1,"x":2}}}',
            '{"artifacts":{"hashes":{"x":1,"x":2}}}',
        )
        for text in texts:
            with self.subTest(text=text), self.assertRaisesRegex(
                ContractError, "duplicate JSON key"
            ):
                loads_exact(text)

    def test_nonstandard_json_constants_are_rejected(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ContractError, "invalid JSON constant"
            ):
                loads_exact('{"value":' + value + "}")

    def test_source_and_target_schema_documents_are_closed_and_versioned(self):
        source = loads_exact(SOURCE_SCHEMA.read_text(encoding="utf-8"))
        target = loads_exact(TARGET_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(source["$id"], "mm-chunk/v1")
        self.assertFalse(source["additionalProperties"])
        self.assertEqual(target["$id"], "target-chunk/v1")
        self.assertFalse(target["additionalProperties"])
        self.assertEqual(
            set(target["required"]),
            {
                "schema",
                "session_id",
                "accepted_chunk_id",
                "source_candidate_id",
                "frame_index",
                "timestamps_s",
                "joint_position",
                "joint_velocity",
                "body_quat_w",
                "physical_pelvis_position",
                "virtual_root_position",
                "virtual_root_quat_w",
                "scene",
                "command",
                "hashes",
            },
        )


class InitialBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SourceFixture()

    def test_initial_boundary_is_exact_owned_and_immutable(self):
        payload = self.fixture.initial()
        boundary = parse_initial_boundary(payload, self.fixture.contract)
        self.assertIsInstance(boundary, InitialBoundary)
        self.assertEqual(boundary.session_id, "session-0")
        self.assertEqual(boundary.source_joint_names, self.fixture.source_names)
        for field in (
            "joint_position_source",
            "joint_velocity_source",
            "physical_pelvis_position_holden",
            "physical_pelvis_orientation_holden",
            "virtual_root_position_holden",
            "virtual_root_orientation_holden",
        ):
            value = getattr(boundary, field)
            self.assertEqual(value.dtype, np.dtype(np.float32), field)
            self.assertTrue(value.flags.c_contiguous, field)
            self.assertTrue(value.flags.owndata, field)
            self.assertFalse(value.flags.writeable, field)
            with self.assertRaises(ValueError):
                value.flat[0] = 123.0
        original = boundary.joint_position_source.copy()
        payload["joint_position_source"][0] = 999.0
        np.testing.assert_array_equal(boundary.joint_position_source, original)

    def test_initial_boundary_rejects_extra_missing_and_wrong_shapes(self):
        cases = []
        extra = self.fixture.initial()
        extra["extra"] = 1
        cases.append(extra)
        missing = self.fixture.initial()
        del missing["session_id"]
        cases.append(missing)
        short_joints = self.fixture.initial()
        short_joints["joint_position_source"] = short_joints["joint_position_source"][:-1]
        cases.append(short_joints)
        short_vector = self.fixture.initial()
        short_vector["physical_pelvis_position_holden"] = [0.0, 0.0]
        cases.append(short_vector)
        short_quaternion = self.fixture.initial()
        short_quaternion["virtual_root_orientation_holden"] = [1.0, 0.0, 0.0]
        cases.append(short_quaternion)
        for payload in cases:
            with self.subTest(keys=tuple(payload)), self.assertRaises(ContractError):
                parse_initial_boundary(payload, self.fixture.contract)

    def test_initial_boundary_rejects_duplicate_missing_or_unknown_joints(self):
        for names in (
            list(self.fixture.source_names[:-1]) + [self.fixture.source_names[0]],
            list(self.fixture.source_names[:-1]),
            list(self.fixture.source_names[:-1]) + ["unregistered_joint"],
        ):
            payload = self.fixture.initial()
            payload["source_joint_names"] = names
            with self.subTest(names=names[-2:]), self.assertRaises(ContractError):
                parse_initial_boundary(payload, self.fixture.contract)


class SourceChunkTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SourceFixture()

    def test_valid_chunk_owns_all_arrays_and_nested_values(self):
        payload = self.fixture.chunk()
        chunk = parse_source_chunk(payload, self.fixture.contract)
        self.assertIsInstance(chunk, SourceChunk)
        self.assertEqual(chunk.source_joint_names, self.fixture.source_names)
        self.assertEqual(chunk.target_joint_names, self.fixture.target_names)
        float_fields = (
            "timestamps_s",
            "joint_position_source",
            "joint_velocity_source",
            "physical_pelvis_position_holden",
            "physical_pelvis_orientation_holden",
            "virtual_root_position_holden",
            "virtual_root_orientation_holden",
            "terrain_cost",
            "terrain_values",
            "terrain_points_holden",
            "support_height",
            "support_target",
        )
        for field in float_fields:
            value = getattr(chunk, field)
            self.assertEqual(value.dtype, np.dtype(np.float32), field)
            self.assertTrue(value.flags.owndata, field)
            self.assertTrue(value.flags.c_contiguous, field)
            self.assertFalse(value.flags.writeable, field)
        rejected_position = chunk.first_rejected_joint_position
        self.assertEqual(rejected_position.dtype, np.dtype(np.float64))
        self.assertTrue(rejected_position.flags.owndata)
        self.assertTrue(rejected_position.flags.c_contiguous)
        self.assertFalse(rejected_position.flags.writeable)
        integer_fields = (
            "selected_database_frame",
            "candidate_preview_count",
            "candidate_limit_rejection_count",
            "first_rejected_database_frame",
            "first_rejected_joint_index",
        )
        for field in integer_fields:
            self.assertEqual(getattr(chunk, field).dtype, np.dtype(np.int64))
        self.assertEqual(chunk.searched.dtype, np.dtype(np.bool_))
        self.assertEqual(chunk.transitioned.dtype, np.dtype(np.bool_))
        for field in (*integer_fields, "searched", "transitioned"):
            value = getattr(chunk, field)
            self.assertTrue(value.flags.owndata, field)
            self.assertFalse(value.flags.writeable, field)
        self.assertEqual(chunk.candidate_limit_rejection_count[2], 1)
        self.assertEqual(chunk.first_rejected_database_frame[2], 927)
        self.assertEqual(chunk.first_rejected_joint_index[2], 5)
        self.assertEqual(
            int(chunk.first_rejected_joint_position[2].view(np.uint64)),
            int(np.float64(-0.27224052).view(np.uint64)),
        )
        self.assertIsInstance(chunk.scene, MappingProxyType)
        self.assertIsInstance(chunk.command, MappingProxyType)
        self.assertIsInstance(chunk.artifacts, MappingProxyType)
        for field in (
            "requested_velocity_holden",
            "desired_heading_holden_wxyz",
            "applied_velocity_holden",
        ):
            value = chunk.command[field]
            self.assertTrue(value.flags.owndata, field)
            self.assertFalse(value.flags.writeable, field)
        with self.assertRaises(TypeError):
            chunk.scene["scene_id"] = "changed"
        with self.assertRaises(TypeError):
            chunk.artifacts["build_commit"] = "changed"
        original_position = chunk.joint_position_source.copy()
        original_command = chunk.command["applied_velocity_holden"].copy()
        payload["joint_position_source"][0][0] = 999.0
        payload["command"]["applied_velocity_holden"][0][0] = 999.0
        np.testing.assert_array_equal(chunk.joint_position_source, original_position)
        np.testing.assert_array_equal(
            chunk.command["applied_velocity_holden"], original_command
        )

    def test_responsive_five_interval_chunk_derives_all_shapes(self):
        payload = self.fixture.chunk(source_intervals=5)
        chunk = parse_source_chunk(payload, self.fixture.contract)
        self.assertEqual(chunk.source_intervals, 5)
        self.assertEqual(chunk.timestamps_s.shape, (6,))
        self.assertEqual(chunk.joint_position_source.shape[0], 6)
        self.assertEqual(chunk.joint_velocity_source.shape[0], 6)
        self.assertEqual(chunk.physical_pelvis_position_holden.shape[0], 6)
        self.assertEqual(chunk.virtual_root_orientation_holden.shape[0], 6)
        for field in (
            "selected_database_frame",
            "candidate_preview_count",
            "candidate_limit_rejection_count",
            "first_rejected_database_frame",
            "first_rejected_joint_index",
            "first_rejected_joint_position",
            "searched",
            "transitioned",
            "terrain_cost",
            "support_height",
            "support_target",
        ):
            self.assertEqual(getattr(chunk, field).shape[0], 5, field)
        self.assertEqual(chunk.terrain_values.shape, (5, 4))
        self.assertEqual(chunk.terrain_points_holden.shape, (5, 4, 3))
        self.assertEqual(chunk.command["applied_velocity_holden"].shape, (5, 3))
        expected_timestamps = np.array(
            [np.float32(index) / np.float32(25) for index in range(6)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(chunk.timestamps_s, expected_timestamps)

    def test_ten_interval_chunk_keeps_twenty_step_shapes(self):
        chunk = parse_source_chunk(self.fixture.chunk(), self.fixture.contract)
        self.assertEqual(chunk.source_intervals, 10)
        self.assertEqual(chunk.timestamps_s.shape, (11,))
        self.assertEqual(chunk.selected_database_frame.shape[0], 10)

    def test_chunk_rejects_unsupported_source_interval_counts(self):
        for bad in (0, 4, 6, 7, 9, 11, 20):
            payload = self.fixture.chunk()
            payload["source_intervals"] = bad
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_extra_or_missing_keys_at_every_object_level(self):
        cases = []
        for path in ((), ("scene",), ("command",), ("artifacts",)):
            extra = self.fixture.chunk()
            target = extra
            for component in path:
                target = target[component]
            target["extra"] = 1
            cases.append(("extra/" + "/".join(path), extra))
            missing = self.fixture.chunk()
            target = missing
            for component in path:
                target = target[component]
            del target[next(iter(target))]
            cases.append(("missing/" + "/".join(path), missing))
        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_wrong_schema_and_fixed_version_fields(self):
        mutations = {
            "schema": "mm-chunk/v2",
            "source_rate_hz": 50,
            "source_intervals": 11,
        }
        for field, value in mutations.items():
            payload = self.fixture.chunk()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_wrong_scene_coordinate_signature(self):
        payload = self.fixture.chunk()
        payload["scene"]["coordinate_signature"] = "wrong-scene-basis"
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_wrong_artifact_coordinate_signature(self):
        payload = self.fixture.chunk()
        payload["artifacts"]["coordinate_signature"] = "wrong-artifact-basis"
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_mismatched_coordinate_signatures(self):
        payload = self.fixture.chunk()
        payload["scene"]["coordinate_signature"] = "wrong-scene-basis"
        payload["artifacts"]["coordinate_signature"] = "wrong-artifact-basis"
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_matching_noncanonical_coordinate_signatures(self):
        payload = self.fixture.chunk()
        payload["scene"]["coordinate_signature"] = "wrong-shared-basis"
        payload["artifacts"]["coordinate_signature"] = "wrong-shared-basis"
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_every_wrong_length_or_shape_family(self):
        mutations = {
            "timestamps_s": lambda value: value[:-1],
            "joint_position_source": lambda value: value[:-1],
            "joint_velocity_source": lambda value: [value[0][:-1], *value[1:]],
            "physical_pelvis_position_holden": lambda value: [value[0][:-1], *value[1:]],
            "physical_pelvis_orientation_holden": lambda value: [value[0][:-1], *value[1:]],
            "virtual_root_position_holden": lambda value: value[:-1],
            "virtual_root_orientation_holden": lambda value: value[:-1],
            "selected_database_frame": lambda value: value[:-1],
            "candidate_preview_count": lambda value: value[:-1],
            "candidate_limit_rejection_count": lambda value: value[:-1],
            "first_rejected_database_frame": lambda value: value[:-1],
            "first_rejected_joint_index": lambda value: value[:-1],
            "first_rejected_joint_position": lambda value: value[:-1],
            "searched": lambda value: value[:-1],
            "transitioned": lambda value: value[:-1],
            "terrain_cost": lambda value: value[:-1],
            "terrain_values": lambda value: [value[0][:-1], *value[1:]],
            "terrain_points_holden": lambda value: [[value[0][0][:-1], *value[0][1:]], *value[1:]],
            "support_height": lambda value: value[:-1],
            "support_target": lambda value: value[:-1],
        }
        for field, mutate in mutations.items():
            payload = self.fixture.chunk()
            payload[field] = mutate(payload[field])
            with self.subTest(field=field), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)
        for field in (
            "requested_velocity_holden",
            "desired_heading_holden_wxyz",
            "applied_velocity_holden",
        ):
            payload = self.fixture.chunk()
            payload["command"][field] = payload["command"][field][:-1]
            with self.subTest(field="command." + field), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_duplicate_missing_and_unregistered_joint_names(self):
        cases = []
        for field, canonical in (
            ("source_joint_names", self.fixture.source_names),
            ("target_joint_names", self.fixture.target_names),
        ):
            duplicate = self.fixture.chunk()
            duplicate[field][-1] = duplicate[field][0]
            cases.append((field + "/duplicate", duplicate))
            missing = self.fixture.chunk()
            missing[field] = missing[field][:-1]
            cases.append((field + "/missing", missing))
            unknown = self.fixture.chunk()
            unknown[field][-1] = "unregistered_joint"
            cases.append((field + "/unknown", unknown))
            if field == "target_joint_names":
                reordered = self.fixture.chunk()
                reordered[field][0], reordered[field][1] = (
                    reordered[field][1], reordered[field][0]
                )
                cases.append((field + "/wrong-order", reordered))
        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_source_joint_order_may_change_when_names_and_columns_agree(self):
        payload = self.fixture.chunk()
        permutation = list(reversed(range(29)))
        payload["source_joint_names"] = [
            payload["source_joint_names"][index] for index in permutation
        ]
        for field in ("joint_position_source", "joint_velocity_source"):
            payload[field] = [
                [row[index] for index in permutation] for row in payload[field]
            ]
        chunk = parse_source_chunk(payload, self.fixture.contract)
        self.assertEqual(chunk.source_joint_names, tuple(reversed(self.fixture.source_names)))

    def test_chunk_rejects_nonfinite_or_non_binary32_source_numbers(self):
        cases = []
        for value in (float("nan"), float("inf"), float("-inf"), 0.1):
            payload = self.fixture.chunk()
            payload["terrain_cost"][0] = value
            cases.append((repr(value), payload))
        payload = self.fixture.chunk()
        payload["selected_database_frame"][0] = True
        cases.append(("boolean integer", payload))
        payload = self.fixture.chunk()
        payload["candidate_preview_count"][0] = True
        cases.append(("boolean preview count", payload))
        payload = self.fixture.chunk()
        payload["first_rejected_joint_position"][2] = float("nan")
        cases.append(("non-finite rejected position", payload))
        payload = self.fixture.chunk()
        payload["first_rejected_joint_position"][2] = True
        cases.append(("boolean rejected position", payload))
        payload = self.fixture.chunk()
        payload["searched"][0] = 1
        cases.append(("integer boolean", payload))
        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_enforces_candidate_preview_count_and_sentinel_invariants(self):
        mutations = []
        for field in (
            "candidate_preview_count",
            "candidate_limit_rejection_count",
        ):
            payload = self.fixture.chunk()
            payload[field][0] = -1
            mutations.append((field + "/negative", payload))
        payload = self.fixture.chunk()
        payload["candidate_limit_rejection_count"][0] = 1
        mutations.append(("rejections-exceed-previews", payload))
        for field, value in (
            ("first_rejected_database_frame", 8),
            ("first_rejected_joint_index", 5),
            ("first_rejected_joint_position", -0.3),
        ):
            payload = self.fixture.chunk()
            payload[field][0] = value
            mutations.append(("zero-rejection/" + field, payload))
        for field, value in (
            ("first_rejected_database_frame", -1),
            ("first_rejected_joint_index", -1),
            ("first_rejected_joint_index", 29),
            ("first_rejected_joint_position", 0.0),
        ):
            payload = self.fixture.chunk()
            payload[field][2] = value
            mutations.append(("positive-rejection/" + field + "/" + str(value), payload))
        payload = self.fixture.chunk()
        payload["first_rejected_database_frame"][2] = \
            payload["selected_database_frame"][2]
        mutations.append(("selected-equals-rejected", payload))
        for label, payload in mutations:
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_retains_sub_binary32_rejection_witness_exactly(self):
        payload = self.fixture.chunk()
        row_index = 2
        lower = self.fixture.contract.rows[row_index].lower
        witness = float(np.nextafter(lower, -np.inf))
        self.assertLess(witness, lower)
        self.assertEqual(np.float32(witness), np.float32(lower))
        payload["first_rejected_joint_index"][2] = row_index
        payload["first_rejected_joint_position"][2] = witness

        chunk = parse_source_chunk(payload, self.fixture.contract)

        self.assertEqual(
            int(chunk.first_rejected_joint_position[2].view(np.uint64)),
            int(np.float64(witness).view(np.uint64)),
        )

    def test_cpp_nine_digit_decimal_recovers_the_original_binary32_bits(self):
        original = np.float32(1.0 / 3.0)
        cpp_decimal = float(format(float(original), ".9g"))
        self.assertNotEqual(cpp_decimal, float(original))
        payload = self.fixture.chunk()
        payload["terrain_cost"][0] = cpp_decimal
        chunk = parse_source_chunk(payload, self.fixture.contract)
        self.assertEqual(
            int(chunk.terrain_cost[0].view(np.uint32)),
            int(original.view(np.uint32)),
        )

    def test_chunk_rejects_json_integers_that_lose_binary32_precision(self):
        for value in (2**60 + 1, -(2**60 + 1)):
            payload = self.fixture.chunk()
            payload["terrain_cost"][0] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_accepts_genuinely_binary32_exact_json_integers(self):
        for value in (2**60, -(2**60)):
            payload = self.fixture.chunk()
            payload["terrain_cost"][0] = value
            chunk = parse_source_chunk(payload, self.fixture.contract)
            with self.subTest(value=value):
                self.assertEqual(int(chunk.terrain_cost[0]), value)

    def test_chunk_rejects_nonunit_quaternions(self):
        paths = (
            ("physical_pelvis_orientation_holden", 0),
            ("virtual_root_orientation_holden", 0),
            ("command", "desired_heading_holden_wxyz"),
        )
        for path in paths:
            payload = self.fixture.chunk()
            target = payload
            for component in path[:-1]:
                target = target[component]
            target[path[-1]] = [2.0, 0.0, 0.0, 0.0]
            with self.subTest(path=path), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_chunk_rejects_nonmonotonic_or_noncanonical_timestamps(self):
        nonmonotonic = self.fixture.chunk()
        nonmonotonic["timestamps_s"][4] = nonmonotonic["timestamps_s"][3]
        off_grid = self.fixture.chunk()
        off_grid["timestamps_s"][4] = f32(0.16015625)
        for label, payload in (("nonmonotonic", nonmonotonic), ("off-grid", off_grid)):
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

    def test_json_round_trip_preserves_the_valid_binary32_fixture(self):
        text = json.dumps(self.fixture.chunk(), separators=(",", ":"))
        chunk = parse_source_chunk(loads_exact(text), self.fixture.contract)
        expected = np.asarray(self.fixture.chunk()["joint_position_source"], np.float32)
        np.testing.assert_array_equal(
            chunk.joint_position_source.view(np.uint32), expected.view(np.uint32)
        )


class SourceChunkV2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = SourceFixture()

    def test_v2_document_is_closed_versioned_and_dual_interval(self):
        document = loads_exact(
            (ROOT / "sonic" / "schemas" / "mm_chunk_v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(document["$id"], "mm-chunk/v2")
        self.assertFalse(document["additionalProperties"])
        self.assertIn("applied_heading_holden_wxyz", document["properties"][
            "command"]["required"])

    def test_v1_fixture_still_parses_unchanged(self):
        chunk = parse_source_chunk(self.fixture.chunk(), self.fixture.contract)
        self.assertNotIn("applied_heading_holden_wxyz", chunk.command)

    def test_v2_chunk_parses_owned_immutable_applied_headings(self):
        requested_heading = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        chunk = parse_source_chunk(
            self.fixture.chunk_v2(source_intervals=5), self.fixture.contract
        )
        self.assertEqual(
            chunk.command["applied_heading_holden_wxyz"].shape, (5, 4)
        )
        self.assertFalse(
            chunk.command["applied_heading_holden_wxyz"].flags.writeable
        )
        np.testing.assert_array_equal(
            chunk.command["desired_heading_holden_wxyz"], requested_heading
        )

    def test_v2_rejects_missing_applied_heading_field(self):
        payload = self.fixture.chunk_v2(source_intervals=5)
        del payload["command"]["applied_heading_holden_wxyz"]
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_v1_with_applied_heading_field_is_rejected(self):
        payload = self.fixture.chunk()
        payload["command"]["applied_heading_holden_wxyz"] = [
            [1.0, 0.0, 0.0, 0.0] for _ in range(10)
        ]
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_v2_rejects_wrong_row_count(self):
        payload = self.fixture.chunk_v2(source_intervals=5)
        payload["command"]["applied_heading_holden_wxyz"] = payload[
            "command"]["applied_heading_holden_wxyz"][:-1]
        with self.assertRaises(ContractError):
            parse_source_chunk(payload, self.fixture.contract)

    def test_v2_rejects_non_unit_and_nan_applied_headings(self):
        non_unit = self.fixture.chunk_v2(source_intervals=5)
        non_unit["command"]["applied_heading_holden_wxyz"][0] = [
            2.0, 0.0, 0.0, 0.0
        ]
        with self.assertRaises(ContractError):
            parse_source_chunk(non_unit, self.fixture.contract)
        nan_rows = self.fixture.chunk_v2(source_intervals=5)
        nan_rows["command"]["applied_heading_holden_wxyz"][0] = [
            float("nan"), 0.0, 0.0, 0.0
        ]
        with self.assertRaises(ContractError):
            parse_source_chunk(nan_rows, self.fixture.contract)

    def test_v2_rejects_duplicate_command_keys(self):
        text = (
            '{"command":{"applied_heading_holden_wxyz":[],'
            '"applied_heading_holden_wxyz":[]}}'
        )
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            loads_exact(text)


if __name__ == "__main__":
    unittest.main()
