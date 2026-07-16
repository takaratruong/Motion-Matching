import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import unittest


ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(os.environ.get(
    "SONIC_MM_SERVER", ROOT / "sonic" / "build" / "mm_chunk_server"))
SCHEMA = ROOT / "sonic" / "schemas" / "mm_chunk_v1.schema.json"


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
        decoded = json.loads(line)
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


@unittest.skipUnless(
    os.environ.get("SONIC_TERRAIN_DIR"),
    "SONIC_TERRAIN_DIR is required for the guarded real-artifact test",
)
class RealArtifactChunkServerTest(unittest.TestCase):
    def test_real_generate_emits_11_boundaries_and_abort_regenerates_exactly(self):
        server = ChunkServer(real=True)
        try:
            identity = server.request(hello())["data"]
            self.assertNotEqual(identity["build_commit"], "test-adapter")
            initial = server.request(reset())["data"]["initial_boundary"]
            first = server.request(generate())["data"]
            self.assertEqual(len(first["joint_position_source"]), 11)
            self.assertEqual(len(first["joint_velocity_source"]), 11)
            self.assertEqual(first["joint_position_source"][0], initial["joint_position_source"])
            self.assertEqual(first["joint_velocity_source"][0], initial["joint_velocity_source"])
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
                "abort_regenerate_equal=true"
            )
            self.assertTrue(server.request(finish("abort", "r5", "c000000"))["ok"])
            self.assertTrue(server.request({"v": 1, "op": "close", "request_id": "r6"})["ok"])
            return_code, stdout_tail, stderr = server.wait(timeout=240)
            self.assertEqual(return_code, 0, stderr.decode("utf-8", "replace"))
            self.assertEqual(stdout_tail, b"")
        finally:
            server.terminate()


if __name__ == "__main__":
    unittest.main()
