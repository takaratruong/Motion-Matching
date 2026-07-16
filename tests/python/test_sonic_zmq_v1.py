from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from mm_sonic.artifacts import RunBundle
from mm_sonic.joints import ContractError
from mm_sonic.timeline import CanonicalTargetBuffer
from mm_sonic.zmq_v1 import (
    HEADER_BYTES,
    TOPIC,
    DecodedPoseV1,
    PosePublisher,
    decode_pose_v1,
    encode_pose_v1,
    pose_header,
    verify_pose_v1_parity,
)


_DTYPES = {
    "f32": np.dtype("<f4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
}


def _bit_bytes(value: np.ndarray, dtype: str) -> bytes:
    return np.asarray(value).astype(dtype, copy=False).tobytes(order="C")


def _assert_array_bits_equal(
    case: unittest.TestCase,
    actual: np.ndarray,
    expected: np.ndarray,
    dtype: str,
) -> None:
    case.assertEqual(actual.shape, expected.shape)
    case.assertEqual(actual.dtype, np.dtype(dtype))
    case.assertEqual(_bit_bytes(actual, dtype), _bit_bytes(expected, dtype))


def make_buffer(count: int, first_frame: int) -> CanonicalTargetBuffer:
    joint_position = np.linspace(
        np.float32(-3.25),
        np.float32(6.5),
        num=count * 29,
        dtype=np.float32,
    ).reshape(count, 29)
    joint_velocity = np.linspace(
        np.float32(9.75),
        np.float32(-7.5),
        num=count * 29,
        dtype=np.float32,
    ).reshape(count, 29)
    # Preserve distinctive low-order bits and signed zero in the golden fixture.
    joint_position[0, 0] = np.array([0x3F800001], dtype="<u4").view("<f4")[0]
    joint_position[0, 1] = np.float32(-0.0)
    joint_velocity[0, 0] = np.array([0xC0200001], dtype="<u4").view("<f4")[0]
    body_quat_w = np.empty((count, 4), dtype=np.float32)
    for index in range(count):
        if index % 2 == 0:
            body_quat_w[index] = np.array(
                [1.0, -0.0, 0.0, 0.0], dtype=np.float32
            )
        else:
            body_quat_w[index] = np.array(
                [0.5, -0.5, 0.5, -0.5], dtype=np.float32
            )
    frame_index = np.arange(
        first_frame, first_frame + count, dtype=np.int64
    )
    return CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=body_quat_w,
        frame_index=frame_index,
    )


def manual_message(buffer: CanonicalTargetBuffer) -> bytes:
    header_json = json.dumps(
        pose_header(buffer.count),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return b"".join(
        (
            TOPIC,
            header_json,
            bytes(HEADER_BYTES - len(header_json)),
            _bit_bytes(buffer.joint_position, "<f4"),
            _bit_bytes(buffer.joint_velocity, "<f4"),
            _bit_bytes(buffer.body_quat_w, "<f4"),
            _bit_bytes(buffer.frame_index, "<i8"),
            bytes((0,)),
        )
    )


def replace_header(message: bytes, header: object) -> bytes:
    header_json = json.dumps(
        header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if len(header_json) >= HEADER_BYTES:
        raise AssertionError("test header must fit the fixed header block")
    return b"".join(
        (
            message[: len(TOPIC)],
            header_json,
            bytes(HEADER_BYTES - len(header_json)),
            message[len(TOPIC) + HEADER_BYTES :],
        )
    )


def payload_offsets(header: dict[str, object]) -> dict[str, tuple[int, int]]:
    offset = len(TOPIC) + HEADER_BYTES
    output: dict[str, tuple[int, int]] = {}
    for field in header["fields"]:
        assert isinstance(field, dict)
        shape = field["shape"]
        assert isinstance(shape, list)
        size = int(np.prod(shape, dtype=np.int64)) * _DTYPES[field["dtype"]].itemsize
        output[field["name"]] = (offset, offset + size)
        offset += size
    return output


class PoseCodecGoldenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = make_buffer(2, 0x0102030405060708)
        self.expected_header = {
            "v": 1,
            "endian": "le",
            "count": 2,
            "fields": [
                {"name": "joint_pos", "dtype": "f32", "shape": [2, 29]},
                {"name": "joint_vel", "dtype": "f32", "shape": [2, 29]},
                {"name": "body_quat_w", "dtype": "f32", "shape": [2, 4]},
                {"name": "frame_index", "dtype": "i64", "shape": [2]},
                {"name": "catch_up", "dtype": "u8", "shape": [1]},
            ],
        }

    def test_exact_header_builder_and_golden_message_bytes(self) -> None:
        self.assertEqual(pose_header(2), self.expected_header)

        message = encode_pose_v1(self.buffer)
        expected = manual_message(self.buffer)
        self.assertEqual(message, expected)
        self.assertEqual(message[:4], b"pose")

        header_block = message[4 : 4 + HEADER_BYTES]
        terminator = header_block.index(0)
        header_json = header_block[:terminator]
        self.assertLess(len(header_json), HEADER_BYTES)
        self.assertEqual(
            header_json,
            json.dumps(
                self.expected_header,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8"),
        )
        self.assertEqual(header_block[terminator:], bytes(HEADER_BYTES - terminator))
        self.assertEqual(json.loads(header_json), self.expected_header)

        offsets = payload_offsets(self.expected_header)
        self.assertEqual(offsets["joint_pos"][0], 4 + HEADER_BYTES)
        self.assertEqual(
            message[slice(*offsets["joint_pos"])],
            _bit_bytes(self.buffer.joint_position, "<f4"),
        )
        self.assertEqual(
            message[slice(*offsets["joint_vel"])],
            _bit_bytes(self.buffer.joint_velocity, "<f4"),
        )
        self.assertEqual(
            message[slice(*offsets["body_quat_w"])],
            _bit_bytes(self.buffer.body_quat_w, "<f4"),
        )
        self.assertEqual(
            message[slice(*offsets["frame_index"])],
            _bit_bytes(self.buffer.frame_index, "<i8"),
        )
        self.assertEqual(message[slice(*offsets["catch_up"])], b"\0")
        self.assertEqual(len(message), offsets["catch_up"][1])
        self.assertEqual(message[4 + HEADER_BYTES : 8 + HEADER_BYTES], b"\x01\x00\x80?")

    def test_decoder_uses_the_hand_built_header_and_preserves_all_bits(self) -> None:
        message = manual_message(self.buffer)
        decoded = decode_pose_v1(message)

        self.assertIsInstance(decoded, DecodedPoseV1)
        self.assertEqual(decoded.header, self.expected_header)
        self.assertEqual(decoded.catch_up, 0)
        _assert_array_bits_equal(
            self, decoded.joint_position, self.buffer.joint_position, "<f4"
        )
        _assert_array_bits_equal(
            self, decoded.joint_velocity, self.buffer.joint_velocity, "<f4"
        )
        _assert_array_bits_equal(
            self, decoded.body_quat_w, self.buffer.body_quat_w, "<f4"
        )
        _assert_array_bits_equal(
            self, decoded.frame_index, self.buffer.frame_index, "<i8"
        )
        verified = verify_pose_v1_parity(message, self.buffer)
        self.assertEqual(verified.catch_up, 0)

    def test_codec_import_and_pure_round_trip_do_not_import_pyzmq(self) -> None:
        code = r'''
import sys
import numpy as np
from mm_sonic.timeline import CanonicalTargetBuffer
from mm_sonic.zmq_v1 import decode_pose_v1, encode_pose_v1, pose_header
assert "zmq" not in sys.modules
buffer = CanonicalTargetBuffer(
    joint_position=np.zeros((1, 29), dtype=np.float32),
    joint_velocity=np.zeros((1, 29), dtype=np.float32),
    body_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    frame_index=np.array([0], dtype=np.int64),
)
decoded = decode_pose_v1(encode_pose_v1(buffer))
assert decoded.frame_index.tolist() == [0]
assert pose_header(1)["count"] == 1
assert "zmq" not in sys.modules
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class PoseCodecRejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = make_buffer(2, 41)
        self.message = encode_pose_v1(self.buffer)
        self.header = pose_header(2)
        self.offsets = payload_offsets(self.header)

    def test_encoder_rejects_corrupted_canonical_shape_dtype_and_layout(self) -> None:
        cases = (
            (
                "shape",
                "joint_position",
                np.zeros((2, 28), dtype=np.float32),
            ),
            (
                "dtype",
                "joint_velocity",
                np.zeros((2, 29), dtype=np.float64),
            ),
            (
                "C-contiguous",
                "joint_position",
                np.asfortranarray(np.zeros((2, 29), dtype=np.float32)),
            ),
        )
        for expected, field, value in cases:
            with self.subTest(field=field, expected=expected):
                corrupted = copy.copy(self.buffer)
                object.__setattr__(corrupted, field, value)
                with self.assertRaisesRegex(ContractError, expected):
                    encode_pose_v1(corrupted)

    def test_encoder_rejects_index_gaps_and_nonfinite_values(self) -> None:
        gap = copy.copy(self.buffer)
        object.__setattr__(gap, "frame_index", np.array([41, 43], dtype=np.int64))
        with self.assertRaisesRegex(ContractError, "contiguous"):
            encode_pose_v1(gap)

        nonfinite = copy.copy(self.buffer)
        values = self.buffer.joint_position.copy()
        values[0, 0] = np.inf
        object.__setattr__(nonfinite, "joint_position", values)
        with self.assertRaisesRegex(ContractError, "finite"):
            encode_pose_v1(nonfinite)

    def test_encoder_rejects_oversized_header(self) -> None:
        with mock.patch(
            "mm_sonic.zmq_v1.pose_header",
            return_value={"padding": "x" * (HEADER_BYTES + 1)},
        ):
            with self.assertRaisesRegex(ContractError, "exceeds 1280"):
                encode_pose_v1(self.buffer)

    def test_decoder_rejects_wrong_topic_and_payload_length(self) -> None:
        with self.assertRaisesRegex(ContractError, "topic"):
            decode_pose_v1(b"post" + self.message[4:])
        with self.assertRaisesRegex(ContractError, "payload length"):
            decode_pose_v1(self.message + b"extra")
        with self.assertRaisesRegex(ContractError, "payload length"):
            decode_pose_v1(self.message[:-1])

    def test_decoder_rejects_wrong_field_shape_and_dtype(self) -> None:
        wrong_shape = copy.deepcopy(self.header)
        wrong_shape["fields"][0]["shape"] = [2, 28]
        with self.assertRaisesRegex(ContractError, "header contract"):
            decode_pose_v1(replace_header(self.message, wrong_shape))

        wrong_dtype = copy.deepcopy(self.header)
        wrong_dtype["fields"][1]["dtype"] = "f64"
        with self.assertRaisesRegex(ContractError, "header contract"):
            decode_pose_v1(replace_header(self.message, wrong_dtype))

    def test_decoder_rejects_nonzero_padding_and_noncanonical_json(self) -> None:
        malformed = bytearray(self.message)
        header_block = malformed[4 : 4 + HEADER_BYTES]
        terminator = header_block.index(0)
        malformed[4 + terminator + 1] = ord("x")
        with self.assertRaisesRegex(ContractError, "padding"):
            decode_pose_v1(bytes(malformed))

        spaced = json.dumps(self.header, ensure_ascii=True).encode("utf-8")
        noncanonical = b"".join(
            (
                TOPIC,
                spaced,
                bytes(HEADER_BYTES - len(spaced)),
                self.message[4 + HEADER_BYTES :],
            )
        )
        with self.assertRaisesRegex(ContractError, "compact"):
            decode_pose_v1(noncanonical)

    def test_decoder_rejects_index_gap_nonfinite_and_baseline_catch_up(self) -> None:
        gap = bytearray(self.message)
        begin, end = self.offsets["frame_index"]
        gap[begin:end] = np.array([41, 43], dtype="<i8").tobytes()
        with self.assertRaisesRegex(ContractError, "contiguous"):
            decode_pose_v1(bytes(gap))

        nonfinite = bytearray(self.message)
        begin, _ = self.offsets["joint_pos"]
        nonfinite[begin : begin + 4] = np.array([np.inf], dtype="<f4").tobytes()
        with self.assertRaisesRegex(ContractError, "finite"):
            decode_pose_v1(bytes(nonfinite))

        catch_up = bytearray(self.message)
        begin, end = self.offsets["catch_up"]
        self.assertEqual(end - begin, 1)
        catch_up[begin] = 1
        with self.assertRaisesRegex(ContractError, "catch_up.*zero"):
            decode_pose_v1(bytes(catch_up))

    def test_parity_check_compares_float_bits_not_numeric_equality(self) -> None:
        changed = bytearray(self.message)
        begin, _ = self.offsets["joint_pos"]
        changed[begin : begin + 4] = np.float32(1.0).tobytes()
        with self.assertRaisesRegex(ContractError, "joint_position.*bit parity"):
            verify_pose_v1_parity(bytes(changed), self.buffer)


class TransmissionArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = RunBundle.create(self.root, "stage-a", "stream")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_archive_writes_exact_bytes_digest_and_rejects_overwrite(self) -> None:
        message = manual_message(make_buffer(20, 1))
        digest = hashlib.sha256(message).hexdigest()
        record = self.bundle.archive_transmission(
            message,
            first_frame_index=1,
            last_frame_index=20,
        )

        binary = self.bundle.path / "transmitted" / "000001-000020.bin"
        sidecar = self.bundle.path / "transmitted" / "000001-000020.sha256"
        self.assertEqual(binary.read_bytes(), message)
        self.assertEqual(sidecar.read_text("ascii"), f"{digest}  {binary.name}\n")
        self.assertEqual(record["sha256"], digest)
        self.assertEqual(record["size"], len(message))
        self.assertEqual(record["message_path"], "transmitted/000001-000020.bin")
        self.assertEqual(record["digest_path"], "transmitted/000001-000020.sha256")

        with self.assertRaisesRegex(ContractError, "already exists"):
            self.bundle.archive_transmission(
                b"replacement",
                first_frame_index=1,
                last_frame_index=20,
            )
        self.assertEqual(binary.read_bytes(), message)

    def test_archive_rejects_invalid_frame_range(self) -> None:
        for first, last in ((-1, 0), (2, 1), (True, 1)):
            with self.subTest(first=first, last=last):
                with self.assertRaisesRegex(ContractError, "frame range"):
                    self.bundle.archive_transmission(
                        b"message",
                        first_frame_index=first,
                        last_frame_index=last,
                    )


class DependencyContractTests(unittest.TestCase):
    def test_pyzmq_is_declared_only_by_the_integration_extra(self) -> None:
        root = Path(__file__).resolve().parents[2]
        pyproject = (root / "sonic" / "pyproject.toml").read_text("utf-8")

        def strings_in_array(name: str) -> list[str]:
            match = re.search(
                rf"(?m)^{re.escape(name)}\s*=\s*\[(.*?)^\]$",
                pyproject,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match, f"missing TOML array: {name}")
            assert match is not None
            return re.findall(r'"([^"]+)"', match.group(1))

        core = strings_in_array("dependencies")
        integration = strings_in_array("integration")
        self.assertNotIn("pyzmq", core)
        self.assertEqual(set(integration), {"mujoco", "pyzmq"})


@unittest.skipUnless(
    importlib.util.find_spec("zmq") is not None,
    "pyzmq is installed only in the isolated Task 8 integration environment",
)
class PosePublisherLoopbackTests(unittest.TestCase):
    def test_pub_bind_sub_connect_uses_explicit_barrier(self) -> None:
        import zmq

        temporary = tempfile.TemporaryDirectory()
        context = zmq.Context()
        synchronization = context.socket(zmq.REP)
        synchronization.setsockopt(zmq.LINGER, 0)
        synchronization.setsockopt(zmq.RCVTIMEO, 5000)
        synchronization.setsockopt(zmq.SNDTIMEO, 5000)
        synchronization_port = synchronization.bind_to_random_port(
            "tcp://127.0.0.1"
        )
        synchronization_endpoint = f"tcp://127.0.0.1:{synchronization_port}"
        bundle = RunBundle.create(Path(temporary.name), "stage-a", "loopback")
        publisher = PosePublisher(
            "tcp://127.0.0.1:*", bundle=bundle, context=context
        )
        results: queue.Queue[object] = queue.Queue()

        def receive() -> None:
            subscriber = context.socket(zmq.SUB)
            barrier = context.socket(zmq.REQ)
            try:
                subscriber.setsockopt(zmq.LINGER, 0)
                subscriber.setsockopt(zmq.CONFLATE, 0)
                subscriber.setsockopt(zmq.SUBSCRIBE, TOPIC)
                subscriber.setsockopt(zmq.RCVTIMEO, 5000)
                subscriber.connect(publisher.endpoint)
                barrier.setsockopt(zmq.LINGER, 0)
                barrier.setsockopt(zmq.RCVTIMEO, 5000)
                barrier.setsockopt(zmq.SNDTIMEO, 5000)
                barrier.connect(synchronization_endpoint)
                barrier.send(b"subscriber-ready")
                if barrier.recv() != b"publish":
                    raise AssertionError("invalid synchronization reply")
                results.put((subscriber.recv(), subscriber.recv()))
            except BaseException as error:
                results.put(error)
            finally:
                barrier.close()
                subscriber.close()

        receiver = threading.Thread(target=receive, name="zmq-v1-loopback")
        receiver.start()
        try:
            self.assertEqual(synchronization.recv(), b"subscriber-ready")
            synchronization.send(b"publish")

            initial = make_buffer(1, 0)
            chunk = make_buffer(20, 1)
            initial_record = publisher.send(initial)
            chunk_record = publisher.send(chunk)

            receiver.join(timeout=6.0)
            self.assertFalse(receiver.is_alive(), "loopback receiver did not finish")
            received = results.get_nowait()
            if isinstance(received, BaseException):
                raise received
            initial_message, chunk_message = received
            self.assertEqual(initial_message, encode_pose_v1(initial))
            self.assertEqual(chunk_message, encode_pose_v1(chunk))
            verify_pose_v1_parity(initial_message, initial)
            verify_pose_v1_parity(chunk_message, chunk)

            for record, first, last, message in (
                (initial_record, 0, 0, initial_message),
                (chunk_record, 1, 20, chunk_message),
            ):
                stem = f"{first:06d}-{last:06d}"
                binary = bundle.path / "transmitted" / f"{stem}.bin"
                sidecar = bundle.path / "transmitted" / f"{stem}.sha256"
                digest = hashlib.sha256(message).hexdigest()
                self.assertEqual(binary.read_bytes(), message)
                self.assertEqual(
                    sidecar.read_text("ascii"), f"{digest}  {binary.name}\n"
                )
                self.assertEqual(record["sha256"], digest)

            publisher.close()
            with self.assertRaisesRegex(ContractError, "closed"):
                publisher.send(make_buffer(1, 21))
            publisher.close()
        finally:
            if receiver.is_alive():
                receiver.join(timeout=6.0)
            publisher.close()
            synchronization.close()
            context.term()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
