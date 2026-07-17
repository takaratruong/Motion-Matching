import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import unittest

import numpy as np

from mm_sonic.joints import load_joint_contract
from mm_sonic.schema import (
    loads_exact,
    parse_initial_boundary,
    parse_source_chunk,
)


ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(os.environ.get(
    "SONIC_MM_SERVER", ROOT / "sonic" / "build" / "mm_chunk_server"))
SCHEMA = ROOT / "sonic" / "schemas" / "mm_chunk_v1.schema.json"
JOINT_CONTRACT = ROOT / "sonic" / "configs" / "g1_joint_contract.json"
MIN_BINARY32_SUBNORMAL = struct.unpack("<f", bytes.fromhex("01000000"))[0]
MAX_BINARY32_SUBNORMAL = struct.unpack("<f", bytes.fromhex("ffff7f00"))[0]
MAX_BINARY32 = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
PROJECTED_BOUNDARY_FIELDS = (
    "joint_position_source",
    "joint_velocity_source",
    "physical_pelvis_position_holden",
    "physical_pelvis_orientation_holden",
    "virtual_root_position_holden",
    "virtual_root_orientation_holden",
)
QUATERNION_BOUNDARY_FIELDS = (
    "physical_pelvis_orientation_holden",
    "virtual_root_orientation_holden",
)
JOINT_FEASIBILITY_SCHEMA = "g1-joint-feasibility-certificate/v1"
JOINT_FEASIBILITY_KEYS = (
    "schema",
    "frame_count",
    "raw_safe_count",
    "raw_unsafe_count",
    "search_safe_count",
    "mask_sha256",
    "joint_limit_violation_count",
)


def expected_fake_joint_feasibility():
    frame_count = 1
    raw_safe = bytes([1])
    search_safe = bytes([1])
    mask_sha256 = hashlib.sha256(
        JOINT_FEASIBILITY_SCHEMA.encode("ascii")
        + struct.pack("<Q", frame_count)
        + raw_safe
        + search_safe
    ).hexdigest()
    return {
        "schema": JOINT_FEASIBILITY_SCHEMA,
        "frame_count": frame_count,
        "raw_safe_count": frame_count,
        "raw_unsafe_count": 0,
        "search_safe_count": frame_count,
        "mask_sha256": mask_sha256,
        "joint_limit_violation_count": [0] * 29,
    }


def hello(request_id="r0"):
    return {"v": 1, "op": "hello", "request_id": request_id}


def reset(request_id="r1", session_id="s1"):
    return {
        "v": 1,
        "op": "reset",
        "request_id": request_id,
        "session_id": session_id,
        "scene_id": "grail-curb-low",
        "route_id": "curb-forward",
        "terrain_weight": 4.0,
    }


def generate(
    request_id="r2",
    candidate_id="c000000",
    predecessor_id=None,
):
    return {
        "v": 1,
        "op": "generate",
        "request_id": request_id,
        "session_id": "s1",
        "candidate_id": candidate_id,
        "predecessor_id": predecessor_id,
        "source_intervals": 10,
        "requested_velocity_holden": [0.0, 0.0, 0.5],
        "desired_heading_holden_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def finish(op, request_id, candidate_id):
    return {
        "v": 1,
        "op": op,
        "request_id": request_id,
        "session_id": "s1",
        "candidate_id": candidate_id,
    }


class ChunkServer:
    def __init__(self, *, real=False):
        if not SERVER.is_file():
            raise FileNotFoundError(f"MM chunk server is missing: {SERVER}")
        env = os.environ.copy()
        if real:
            env.pop("SONIC_MM_TEST_ADAPTER", None)
            env["SONIC_TERRAIN_DIR"] = os.environ["SONIC_TERRAIN_DIR"]
        else:
            env["SONIC_MM_TEST_ADAPTER"] = "1"
            env.pop("SONIC_TERRAIN_DIR", None)
        self.process = subprocess.Popen(
            [str(SERVER)],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def request(self, value):
        self.write_bytes((json.dumps(value, separators=(",", ":")) + "\n").encode())
        return self.response()

    def write_bytes(self, value):
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def response(self):
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read().decode("utf-8", "replace")
            raise AssertionError(f"server exited without a response: {stderr}")
        decoded = loads_exact(line.decode("utf-8"))
        self.assert_envelope(decoded)
        return decoded

    @staticmethod
    def assert_envelope(value):
        expected = {"v", "ok", "op", "request_id", "data" if value.get("ok") else "error"}
        if set(value) != expected:
            raise AssertionError(f"response envelope keys differ: {value}")
        if value["v"] != 1 or not isinstance(value["ok"], bool):
            raise AssertionError(f"response envelope identity differs: {value}")

    def close_stdin(self):
        self.process.stdin.close()
        self.process.stdin = None

    def wait(self, timeout=30):
        stdout, stderr = self.process.communicate(timeout=timeout)
        return self.process.returncode, stdout, stderr

    def terminate(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.communicate()


class ChunkServerProtocolTest(unittest.TestCase):
    def setUp(self):
        self.server = ChunkServer()

    def tearDown(self):
        self.server.terminate()

    def test_stdout_is_jsonl_pure_and_candidate_lifecycle_is_transactional(self):
        hello_response = self.server.request(hello())
        self.assertTrue(hello_response["ok"])
        identity = hello_response["data"]
        self.assertEqual(identity["protocol_version"], 1)
        self.assertEqual(identity["source_rate_hz"], 25)
        self.assertEqual(identity["supported_source_intervals"], [10])
        self.assertEqual(len(identity["source_joint_names"]), 29)
        self.assertEqual(len(identity["target_joint_names"]), 29)
        for field in (
            "skeleton_signature",
            "joint_contract_sha256",
            "motion_manifest_sha256",
            "database_sha256",
            "terrain_features_sha256",
            "terrain_support_sha256",
            "scene_index_sha256",
        ):
            self.assertRegex(identity[field], r"^[0-9a-f]{64}$")
        self.assertEqual(
            tuple(identity["joint_feasibility"]),
            JOINT_FEASIBILITY_KEYS,
        )
        self.assertEqual(
            identity["joint_feasibility"],
            expected_fake_joint_feasibility(),
        )

        reset_response = self.server.request(reset())
        self.assertTrue(reset_response["ok"])
        initial = reset_response["data"]["initial_boundary"]
        self.assertEqual(initial["session_id"], "s1")
        self.assertEqual(len(initial["joint_position_source"]), 29)

        generated = self.server.request(generate())
        self.assertTrue(generated["ok"])
        candidate = generated["data"]
        self.assertEqual(candidate["schema"], "mm-chunk/v1")
        self.assertEqual(len(candidate["timestamps_s"]), 11)
        self.assertEqual(
            [struct.pack("<f", value) for value in candidate["timestamps_s"]],
            [struct.pack("<f", index / 25) for index in range(11)],
        )
        self.assertEqual(len(candidate["joint_position_source"]), 11)
        self.assertTrue(all(len(row) == 29 for row in candidate["joint_position_source"]))
        self.assertEqual(len(candidate["selected_database_frame"]), 10)
        for field in (
            "candidate_preview_count",
            "candidate_limit_rejection_count",
            "first_rejected_database_frame",
            "first_rejected_joint_index",
            "first_rejected_joint_position",
        ):
            self.assertEqual(len(candidate[field]), 10, field)
        self.assertEqual(candidate["candidate_preview_count"], [0] * 10)
        self.assertEqual(candidate["candidate_limit_rejection_count"], [0] * 10)
        self.assertEqual(candidate["first_rejected_database_frame"], [-1] * 10)
        self.assertEqual(candidate["first_rejected_joint_index"], [-1] * 10)
        self.assertEqual(candidate["first_rejected_joint_position"], [0.0] * 10)
        self.assertEqual(len(candidate["terrain_values"]), 10)
        self.assertTrue(all(len(row) == 4 for row in candidate["terrain_values"]))
        self.assertEqual(set(candidate["command"]), {
            "requested_velocity_holden",
            "desired_heading_holden_wxyz",
            "applied_velocity_holden",
        })
        self.assertEqual(len(candidate["command"]["applied_velocity_holden"]), 10)
        self.assertTrue(all(
            len(row) == 3
            for row in candidate["command"]["applied_velocity_holden"]
        ))

        outstanding = self.server.request(generate("r-out", "c000001"))
        self.assertFalse(outstanding["ok"])
        self.assertEqual(outstanding["error"], {
            "code": "candidate_outstanding",
            "message": "candidate c000000 must be committed or aborted",
        })

        self.assertTrue(self.server.request(finish("abort", "r3", "c000000"))["ok"])
        regenerated = self.server.request(generate("r4"))
        self.assertEqual(regenerated["data"], candidate)
        committed = self.server.request(finish("commit", "r5", "c000000"))
        self.assertEqual(committed["data"], {
            "session_id": "s1",
            "candidate_id": "c000000",
            "active_candidate_id": "c000000",
        })

        successor = self.server.request(generate("r6", "c000001", "c000000"))
        self.assertTrue(successor["ok"])
        successor_data = successor["data"]
        for field in (
            "joint_position_source",
            "joint_velocity_source",
            "physical_pelvis_position_holden",
            "physical_pelvis_orientation_holden",
            "virtual_root_position_holden",
            "virtual_root_orientation_holden",
        ):
            self.assertEqual(successor_data[field][0], candidate[field][-1])
        self.assertTrue(self.server.request(finish("abort", "r7", "c000001"))["ok"])
        closed = self.server.request({"v": 1, "op": "close", "request_id": "r8"})
        self.assertTrue(closed["ok"])
        return_code, stdout_tail, stderr = self.server.wait()
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout_tail, b"")
        self.assertIn(b"test adapter", stderr)
        self.assertNotIn(b"test adapter", json.dumps(closed).encode())

    def test_fake_joint_feasibility_identity_is_stable_across_fresh_launches(self):
        first = self.server.request(hello("first"))["data"]["joint_feasibility"]
        second_server = ChunkServer()
        try:
            second = second_server.request(hello("second"))["data"][
                "joint_feasibility"
            ]
        finally:
            second_server.terminate()

        expected = expected_fake_joint_feasibility()
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertEqual(tuple(first), JOINT_FEASIBILITY_KEYS)
        self.assertEqual(tuple(second), JOINT_FEASIBILITY_KEYS)
        for field in (
            "frame_count",
            "raw_safe_count",
            "raw_unsafe_count",
            "search_safe_count",
        ):
            self.assertIs(type(first[field]), int)
            self.assertGreaterEqual(first[field], 0)
        self.assertEqual(len(first["joint_limit_violation_count"]), 29)
        self.assertTrue(all(
            type(value) is int and value >= 0
            for value in first["joint_limit_violation_count"]
        ))

    def test_strict_json_exact_keys_and_binary32_numbers(self):
        cases = (
            (b'{"v":1,"op":"hello","request_id":"r","extra":0}\n', "invalid_request"),
            (b'{"v":1,"v":1,"op":"hello","request_id":"r"}\n', "invalid_json"),
            (b'{"v":2,"op":"hello","request_id":"r"}\n', "unsupported_version"),
            (b'{"v":1.5,"op":"hello","request_id":"r"}\n', "invalid_request"),
            (b'{"v":NaN,"op":"hello","request_id":"r"}\n', "invalid_json"),
            (b'{"v":1,"op":"hello","request_id":"r"} trailing\n', "invalid_json"),
            (b'{"v":1,"op":"hel\xfflo","request_id":"r"}\n', "invalid_json"),
        )
        for encoded, code in cases:
            with self.subTest(encoded=encoded):
                self.server.write_bytes(encoded)
                response = self.server.response()
                self.assertFalse(response["ok"])
                self.assertEqual(response["error"]["code"], code)

        self.assertTrue(self.server.request(hello("after-errors"))["ok"])
        non_binary32 = reset("bad-float")
        non_binary32["terrain_weight"] = 0.1
        response = self.server.request(non_binary32)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")
        self.assertTrue(self.server.request(reset("good-reset"))["ok"])

        fractional = generate("fractional")
        fractional["source_intervals"] = 10.5
        response = self.server.request(fractional)
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["code"], "invalid_request")

    def test_exact_binary32_boundaries_have_no_unstated_weight_range(self):
        self.assertTrue(self.server.request(hello())["ok"])
        for index, weight in enumerate((
            MIN_BINARY32_SUBNORMAL,
            -MIN_BINARY32_SUBNORMAL,
            MAX_BINARY32_SUBNORMAL,
            -MAX_BINARY32_SUBNORMAL,
            -4.0,
            16.0,
            MAX_BINARY32,
        )):
            with self.subTest(weight=weight):
                request = reset(f"weight-{index}")
                request["terrain_weight"] = weight
                response = self.server.request(request)
                self.assertTrue(response["ok"], response)
                self.assertEqual(
                    struct.pack("<f", response["data"]["scene"]["terrain_weight"]),
                    struct.pack("<f", weight),
                )

        request = generate("subnormal-vector")
        request["requested_velocity_holden"] = [
            MIN_BINARY32_SUBNORMAL,
            -MAX_BINARY32_SUBNORMAL,
            0.5,
        ]
        candidate = self.server.request(request)
        self.assertTrue(candidate["ok"], candidate)
        for applied in candidate["data"]["command"]["applied_velocity_holden"]:
            self.assertEqual(
                [struct.pack("<f", value) for value in applied],
                [
                    struct.pack("<f", MIN_BINARY32_SUBNORMAL),
                    struct.pack("<f", -MAX_BINARY32_SUBNORMAL),
                    struct.pack("<f", 0.5),
                ],
            )

    def test_identifiers_have_no_byte_or_control_character_ceiling(self):
        request_id = "request\x00\n🙂"
        response = self.server.request(hello(request_id))
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["request_id"], request_id)

        long_identifier = "会" * 257
        request = reset("long-identifiers", long_identifier)
        request["scene_id"] = "scene\x01" + long_identifier
        request["route_id"] = "route\t" + long_identifier
        response = self.server.request(request)
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["data"]["session_id"], long_identifier)
        self.assertEqual(response["data"]["scene"]["scene_id"], request["scene_id"])
        self.assertEqual(response["data"]["scene"]["route_id"], request["route_id"])

    def test_request_line_has_no_unstated_one_mib_ceiling(self):
        request_id = "r" * (1024 * 1024 + 32)
        response = self.server.request(hello(request_id))
        self.assertTrue(response["ok"], response)
        self.assertEqual(response["request_id"], request_id)

    def test_schema_identity_matches_generated_candidate(self):
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], "mm-chunk/v1")
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue(self.server.request(hello())["ok"])
        self.assertTrue(self.server.request(reset())["ok"])
        candidate = self.server.request(generate())["data"]
        self.assertEqual(candidate["schema"], schema["$id"])
        self.assertEqual(set(candidate), set(schema["required"]))

    def test_clean_eof_and_explicit_close(self):
        self.assertTrue(self.server.request(hello())["ok"])
        self.server.close_stdin()
        return_code, stdout_tail, stderr = self.server.wait()
        self.assertEqual(return_code, 0)
        self.assertEqual(stdout_tail, b"")
        self.assertIn(b"test adapter", stderr)

    def test_failed_reset_stages_preserve_prior_session_and_scene(self):
        self.assertTrue(self.server.request(hello())["ok"])
        self.assertTrue(self.server.request(reset())["ok"])
        baseline = self.server.request(generate())["data"]
        self.assertTrue(self.server.request(finish("abort", "r3", "c000000"))["ok"])

        failures = (
            ("fail-scene", "curb-forward"),
            ("grail-curb-low", "fail-route"),
            ("fail-reset", "curb-forward"),
            ("fail-observe", "curb-forward"),
        )
        for index, (scene_id, route_id) in enumerate(failures):
            with self.subTest(scene_id=scene_id, route_id=route_id):
                request = reset(f"failed-reset-{index}", "replacement")
                request["scene_id"] = scene_id
                request["route_id"] = route_id
                request["terrain_weight"] = 7.0
                failed = self.server.request(request)
                self.assertFalse(failed["ok"], failed)
                self.assertEqual(failed["error"]["code"], "reset_failed")

                regenerated = self.server.request(generate(
                    f"regenerate-{index}", "c000000", None
                ))
                self.assertTrue(regenerated["ok"], regenerated)
                self.assertEqual(regenerated["data"], baseline)
                self.assertTrue(self.server.request(finish(
                    "abort", f"abort-{index}", "c000000"
                ))["ok"])

    def test_serialization_failure_never_publishes_reset_or_candidate(self):
        self.assertTrue(self.server.request(hello())["ok"])
        self.assertTrue(self.server.request(reset())["ok"])
        baseline = self.server.request(generate())["data"]
        self.assertEqual(baseline["terrain_cost"][0], 4.0)
        self.assertTrue(self.server.request(finish("abort", "r3", "c000000"))["ok"])

        bad_reset = reset("nonfinite-reset", "replacement")
        bad_reset["scene_id"] = "nonfinite-reset"
        bad_reset["terrain_weight"] = 7.0
        response = self.server.request(bad_reset)
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error"]["code"], "serialization_failed")

        after_reset_failure = self.server.request(generate("after-bad-reset"))
        self.assertTrue(after_reset_failure["ok"], after_reset_failure)
        self.assertEqual(after_reset_failure["data"], baseline)
        self.assertTrue(self.server.request(finish(
            "abort", "after-bad-reset-abort", "c000000"
        ))["ok"])

        response = self.server.request(generate(
            "nonfinite-generate", "nonfinite-generate"
        ))
        self.assertFalse(response["ok"], response)
        self.assertEqual(response["error"]["code"], "serialization_failed")

        after_generate_failure = self.server.request(generate("after-bad-generate"))
        self.assertTrue(after_generate_failure["ok"], after_generate_failure)
        self.assertEqual(after_generate_failure["data"], baseline)
        self.assertTrue(self.server.request(finish(
            "abort", "after-bad-generate-abort", "c000000"
        ))["ok"])

    def test_candidate_preview_serialization_failures_never_publish(self):
        self.assertTrue(self.server.request(hello())["ok"])
        self.assertTrue(self.server.request(reset())["ok"])
        cases = (
            "negative-preview-count",
            "negative-rejection-count",
            "rejections-exceed-previews",
            "bad-zero-rejected-frame",
            "bad-zero-rejected-joint",
            "bad-zero-rejected-position",
            "bad-positive-rejected-frame",
            "bad-positive-rejected-joint",
            "bad-positive-rejected-position",
            "in-range-rejected-position",
            "selected-equals-rejected-frame",
        )
        for index, candidate_id in enumerate(cases):
            with self.subTest(candidate_id=candidate_id):
                response = self.server.request(generate(
                    f"invalid-preview-{index}", candidate_id
                ))
                self.assertFalse(response["ok"], response)
                self.assertEqual(
                    response["error"]["code"], "serialization_failed"
                )
                control = self.server.request(generate(
                    f"control-{index}", "c000000"
                ))
                self.assertTrue(control["ok"], control)
                self.assertEqual(control["data"]["candidate_preview_count"], [0] * 10)
                self.assertTrue(self.server.request(finish(
                    "abort", f"control-abort-{index}", "c000000"
                ))["ok"])


class ChunkServerSourceOwnershipTest(unittest.TestCase):
    def test_real_certificate_build_order_ownership_and_runtime_binding(self):
        source = (ROOT / "sonic" / "cpp" / "mm_chunk_server.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('#include "sonic/cpp/g1_joint_feasibility.h"', source)

        adapter = source.index("class mm_real_adapter")
        adapter_stop = source.index("\nstruct mm_fake_state", adapter)
        adapter_source = source[adapter:adapter_stop]
        initialize_start = adapter_source.index("    bool initialize(\n")
        initialize_stop = adapter_source.index(
            "\n    const mm_server_identity& identity() const",
            initialize_start,
        )
        initialize_body = adapter_source[initialize_start:initialize_stop]
        self.assertLess(
            initialize_body.index("sonic_joint_contract_load("),
            initialize_body.index(
                "joint contract source/target order is not fixed"
            ),
        )
        self.assertLess(
            initialize_body.index(
                "joint contract source/target order is not fixed"
            ),
            initialize_body.index("g1_database_validate("),
        )
        self.assertLess(
            initialize_body.index("g1_database_validate("),
            initialize_body.index(
                "sonic_build_joint_feasibility_certificate("
            ),
        )
        self.assertIn(
            "sonic_joint_feasibility_certificate joint_feasibility_;",
            adapter_source,
        )
        classifier_start = source.index(
            "static g1_runtime_joint_preview_verdict "
            "mm_real_classify_joint_preview("
        )
        classifier_stop = source.index(
            "\nclass mm_real_adapter",
            classifier_start,
        )
        classifier_source = source[classifier_start:classifier_stop]
        self.assertIn("sonic_project_joint_state(", classifier_source)
        self.assertIn("SonicJointProjectionLimit", classifier_source)
        self.assertIn("projection.row", classifier_source)
        self.assertIn("std::isfinite(projection.position)", classifier_source)
        self.assertIn("projection.lower == contract[row].lower", classifier_source)
        self.assertIn("projection.upper == contract[row].upper", classifier_source)

        advance_start = adapter_source.index("    bool advance(\n")
        advance_stop = adapter_source.index("\nprivate:", advance_start)
        advance_body = adapter_source[advance_start:advance_stop]
        self.assertEqual(advance_body.count("g1_runtime_step("), 1)
        self.assertIn("g1_runtime_frame_feasibility", advance_body)
        self.assertIn("joint_feasibility_.raw_safe.data", advance_body)
        self.assertIn("joint_feasibility_.search_safe.data", advance_body)
        self.assertIn("joint_feasibility_.frame_count", advance_body)
        self.assertIn(
            "g1_runtime_joint_preview_validator preview_validator",
            advance_body,
        )
        self.assertIn("preview_validator.context = this", advance_body)
        self.assertIn(
            "preview_validator.evaluate = validate_joint_preview",
            advance_body,
        )
        runtime_call = advance_body[advance_body.index("g1_runtime_step("):]
        self.assertIn("runtime_feasibility", runtime_call)
        self.assertIn("preview_validator", runtime_call)

    def test_fake_certificate_uses_shared_digest_and_projection_gate_remains(self):
        source = (ROOT / "sonic" / "cpp" / "mm_chunk_server.cpp").read_text(
            encoding="utf-8"
        )
        fake_start = source.index("class mm_fake_adapter")
        fake_stop = source.index(
            "\nstatic void mm_json_write_string_array", fake_start
        )
        fake_source = source[fake_start:fake_stop]
        self.assertIn("sonic_joint_feasibility_digest(", fake_source)
        self.assertIn(
            "sonic_joint_feasibility_certificate joint_feasibility_;",
            fake_source,
        )

        adapter = source.index("class mm_real_adapter")
        observe_start = source.index("    bool observe_boundary(\n", adapter)
        observe_stop = source.index("\n    std::string terrain_root_;", observe_start)
        observe_body = source[observe_start:observe_stop]
        self.assertEqual(observe_body.count("sonic_project_pose("), 1)
        self.assertLess(
            observe_body.index("sonic_project_pose("),
            observe_body.index("output = candidate;"),
        )

    def test_real_reset_builds_candidate_features_before_controller_reset(self):
        source = (ROOT / "sonic" / "cpp" / "mm_chunk_server.cpp").read_text(
            encoding="utf-8"
        )
        adapter = source.index("class mm_real_adapter")
        reset_start = source.index("    bool prepare_reset(\n", adapter)
        reset_stop = source.index("\n    bool clone(\n", reset_start)
        reset_body = source[reset_start:reset_stop]
        self.assertLess(
            reset_body.index("rebuild_matching_features("),
            reset_body.index("g1_controller_state_reset("),
            "requested-weight features must be validated before controller reset",
        )

    def test_artifact_helpers_have_one_odr_safe_shared_owner(self):
        controller = (ROOT / "controller.cpp").read_text(encoding="utf-8")
        server = (ROOT / "sonic" / "cpp" / "mm_chunk_server.cpp").read_text(
            encoding="utf-8"
        )
        projector = (
            ROOT / "sonic" / "cpp" / "g1_project_pose_cli.cpp"
        ).read_text(encoding="utf-8")
        database_header = (
            ROOT / "sonic" / "cpp" / "g1_database_validation.h"
        ).read_text(encoding="utf-8")
        contract_header = (
            ROOT / "sonic" / "cpp" / "g1_joint_contract_io.h"
        ).read_text(encoding="utf-8")

        self.assertIn('#include "sonic/cpp/g1_database_validation.h"', controller)
        self.assertIn('#include "sonic/cpp/g1_database_validation.h"', server)
        self.assertIn('#include "sonic/cpp/g1_joint_contract_io.h"', server)
        self.assertIn('#include "sonic/cpp/g1_joint_contract_io.h"', projector)
        self.assertIn("static inline bool g1_database_validate(", database_header)
        self.assertIn(
            "static inline bool g1_matching_features_validate(",
            database_header,
        )
        self.assertIn(
            "static inline bool sonic_joint_contract_load(",
            contract_header,
        )

        combined = controller + server + projector
        for duplicate in (
            "static bool g1_database_validate(",
            "static bool g1_matching_features_validate(",
            "static bool mm_server_database_is_valid(",
            "static bool mm_server_matching_features_are_valid(",
            "static int mm_server_bone_index(",
            "static bool mm_server_load_joint_contract(",
            "static int cli_bone_index(",
            "static bool cli_load_contract(",
        ):
            self.assertNotIn(duplicate, combined)


@unittest.skipUnless(
    os.environ.get("SONIC_TERRAIN_DIR"),
    "SONIC_TERRAIN_DIR is required for the guarded real-artifact test",
)
class RealArtifactChunkServerTest(unittest.TestCase):
    def test_embedded_nul_scene_and_route_cannot_alias_real_ids(self):
        aliases = (
            ("scene_id", "grail-curb-low\x00ignored-suffix"),
            ("route_id", "curb-forward\x00ignored-suffix"),
        )
        for field, alias in aliases:
            with self.subTest(field=field):
                server = ChunkServer(real=True)
                try:
                    self.assertTrue(server.request(hello())["ok"])
                    self.assertTrue(server.request(reset())["ok"])
                    baseline = server.request(generate())["data"]
                    self.assertTrue(server.request(finish(
                        "abort", "nul-baseline-abort", "c000000"
                    ))["ok"])

                    aliased_reset = reset(f"nul-{field}", "replacement")
                    aliased_reset[field] = alias
                    aliased_reset["terrain_weight"] = 0.0
                    failed = server.request(aliased_reset)
                    self.assertFalse(failed["ok"], failed)
                    self.assertEqual(failed["error"]["code"], "reset_failed")

                    regenerated = server.request(generate(
                        f"nul-{field}-regenerate"
                    ))["data"]
                    self.assertEqual(regenerated["session_id"], "s1")
                    self.assertEqual(regenerated["scene"], baseline["scene"])
                    self.assertEqual(
                        regenerated["terrain_cost"], baseline["terrain_cost"]
                    )
                    self.assertEqual(regenerated, baseline)
                    self.assertTrue(server.request(finish(
                        "abort", f"nul-{field}-abort", "c000000"
                    ))["ok"])
                    self.assertTrue(server.request({
                        "v": 1,
                        "op": "close",
                        "request_id": f"nul-{field}-close",
                    })["ok"])
                    return_code, stdout_tail, stderr = server.wait(timeout=240)
                    self.assertEqual(
                        return_code, 0, stderr.decode("utf-8", "replace")
                    )
                    self.assertEqual(stdout_tail, b"")
                finally:
                    server.terminate()

    def test_real_generate_emits_11_boundaries_and_abort_regenerates_exactly(self):
        server = ChunkServer(real=True)
        try:
            contract = load_joint_contract(JOINT_CONTRACT)
            source_names = tuple(row.source_joint for row in contract.rows)
            target_names = tuple(
                next(
                    row.target_name
                    for row in contract.rows
                    if row.target_index == index
                )
                for index in range(29)
            )
            identity = server.request(hello())["data"]
            self.assertNotEqual(identity["build_commit"], "test-adapter")
            initial = server.request(reset())["data"]["initial_boundary"]
            first = server.request(generate())["data"]
            initial_model = parse_initial_boundary(initial, contract)
            first_model = parse_source_chunk(first, contract)
            self.assertEqual(initial_model.source_joint_names, source_names)
            self.assertEqual(first_model.source_joint_names, source_names)
            self.assertEqual(first_model.target_joint_names, target_names)
            self.assertEqual(
                tuple(initial["source_joint_names"]),
                initial_model.source_joint_names,
            )
            self.assertEqual(
                tuple(first["source_joint_names"]),
                first_model.source_joint_names,
            )
            self.assertEqual(
                initial_model.source_joint_names,
                first_model.source_joint_names,
            )
            self.assertEqual(len(first["joint_position_source"]), 11)
            self.assertEqual(len(first["joint_velocity_source"]), 11)
            for field in PROJECTED_BOUNDARY_FIELDS:
                raw_initial = np.asarray(initial[field], np.float32)
                raw_candidate = np.asarray(first[field][0], np.float32)
                decoded_initial = getattr(initial_model, field)
                decoded_candidate = getattr(first_model, field)[0]
                for actual, expected in (
                    (decoded_initial, raw_initial),
                    (decoded_candidate, raw_candidate),
                    (decoded_candidate, decoded_initial),
                ):
                    np.testing.assert_array_equal(
                        actual.view(np.uint32), expected.view(np.uint32)
                    )
            maximum_quaternion_norm_error = 0.0
            for field in QUATERNION_BOUNDARY_FIELDS:
                raw = np.asarray(first[field], np.float32)
                decoded = getattr(first_model, field)
                np.testing.assert_array_equal(
                    decoded.view(np.uint32), raw.view(np.uint32)
                )
                maximum_quaternion_norm_error = max(
                    maximum_quaternion_norm_error,
                    float(
                        np.max(
                            np.abs(
                                np.linalg.norm(raw.astype(np.float64), axis=1)
                                - 1.0
                            )
                        )
                    ),
                )
            self.assertTrue(server.request(finish("abort", "r3", "c000000"))["ok"])
            second = server.request(generate("r4"))["data"]
            self.assertEqual(second, first)
            canonical = json.dumps(
                first,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            print(
                "real artifact evidence: "
                f"sha256={hashlib.sha256(canonical).hexdigest()} "
                f"boundaries={len(first['timestamps_s'])} "
                f"boundary_zero_fields={len(PROJECTED_BOUNDARY_FIELDS)} "
                f"quaternion_norm_error={maximum_quaternion_norm_error:.9g} "
                "names_exact=true decode_repair=false "
                "abort_regenerate_equal=true"
            )
            self.assertTrue(server.request(finish("abort", "r5", "c000000"))["ok"])

            failed_reset = reset("real-failed-rebuild", "replacement")
            failed_reset["terrain_weight"] = -4.0
            failed = server.request(failed_reset)
            self.assertFalse(failed["ok"], failed)
            self.assertEqual(failed["error"]["code"], "reset_failed")
            after_failure = server.request(generate("real-after-failure"))["data"]
            self.assertEqual(after_failure, first)
            self.assertTrue(server.request(finish(
                "abort", "real-after-failure-abort", "c000000"
            ))["ok"])

            route_failure = reset("real-failed-route", "replacement")
            route_failure["route_id"] = "missing-route"
            failed = server.request(route_failure)
            self.assertFalse(failed["ok"], failed)
            self.assertEqual(failed["error"]["code"], "reset_failed")
            after_route_failure = server.request(generate("real-after-route"))["data"]
            self.assertEqual(after_route_failure, first)
            self.assertTrue(server.request(finish(
                "abort", "real-after-route-abort", "c000000"
            ))["ok"])

            self.assertTrue(server.request({"v": 1, "op": "close", "request_id": "r6"})["ok"])
            return_code, stdout_tail, stderr = server.wait(timeout=240)
            self.assertEqual(return_code, 0, stderr.decode("utf-8", "replace"))
            self.assertEqual(stdout_tail, b"")
        finally:
            server.terminate()


if __name__ == "__main__":
    unittest.main()
