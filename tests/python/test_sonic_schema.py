import json
from pathlib import Path
from types import MappingProxyType
import unittest

import numpy as np

from mm_sonic.joints import load_joint_contract
from mm_sonic.schema import (
    ContractError,
    InitialBoundary,
    SourceChunk,
    loads_exact,
    parse_initial_boundary,
    parse_source_chunk,
)


ROOT = Path(__file__).resolve().parents[2]
JOINT_CONTRACT = ROOT / "sonic" / "configs" / "g1_joint_contract.json"
SOURCE_SCHEMA = ROOT / "sonic" / "schemas" / "mm_chunk_v1.schema.json"
TARGET_SCHEMA = ROOT / "sonic" / "schemas" / "target_chunk_v1.schema.json"


def f32(value):
    return float(np.float32(value))


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

    def chunk(self):
        boundaries = range(11)
        steps = range(10)
        return {
            "schema": "mm-chunk/v1",
            "session_id": "session-0",
            "candidate_id": "candidate-0",
            "predecessor_id": None,
            "source_rate_hz": 25,
            "source_intervals": 10,
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
        self.assertEqual(chunk.selected_database_frame.dtype, np.dtype(np.int64))
        self.assertEqual(chunk.searched.dtype, np.dtype(np.bool_))
        self.assertEqual(chunk.transitioned.dtype, np.dtype(np.bool_))
        for field in ("selected_database_frame", "searched", "transitioned"):
            value = getattr(chunk, field)
            self.assertTrue(value.flags.owndata, field)
            self.assertFalse(value.flags.writeable, field)
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
        payload["searched"][0] = 1
        cases.append(("integer boolean", payload))
        for label, payload in cases:
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_source_chunk(payload, self.fixture.contract)

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


if __name__ == "__main__":
    unittest.main()
