"""Pure GEAR packed-pose v1 codec plus a lazily imported ZMQ publisher."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .artifacts import RunBundle
from .joints import ContractError
from .timeline import CanonicalTargetBuffer


TOPIC = b"pose"
HEADER_BYTES = 1280

_FIELD_DTYPES = {
    "f32": np.dtype("<f4"),
    "i64": np.dtype("<i8"),
    "u8": np.dtype("u1"),
}


def _tag_failure(error: BaseException, site: str) -> None:
    if getattr(error, "failure_site", None) is None:
        try:
            error.failure_site = site
        except BaseException:
            pass


def pose_header(count: int) -> dict[str, object]:
    return {
        "v": 1,
        "endian": "le",
        "count": count,
        "fields": [
            {"name": "joint_pos", "dtype": "f32", "shape": [count, 29]},
            {"name": "joint_vel", "dtype": "f32", "shape": [count, 29]},
            {"name": "body_quat_w", "dtype": "f32", "shape": [count, 4]},
            {"name": "frame_index", "dtype": "i64", "shape": [count]},
            {"name": "catch_up", "dtype": "u8", "shape": [1]},
        ],
    }


def _validate_canonical_buffer(buffer: CanonicalTargetBuffer) -> None:
    if not isinstance(buffer, CanonicalTargetBuffer):
        raise ContractError("ZMQ v1 encoding requires a CanonicalTargetBuffer")
    raw_frame = np.asarray(buffer.frame_index)
    if raw_frame.ndim != 1 or raw_frame.shape[0] <= 0:
        raise ContractError("canonical frame_index has an invalid shape")
    count = int(raw_frame.shape[0])
    fields = (
        ("joint_position", buffer.joint_position, np.dtype("<f4"), (count, 29)),
        ("joint_velocity", buffer.joint_velocity, np.dtype("<f4"), (count, 29)),
        ("body_quat_w", buffer.body_quat_w, np.dtype("<f4"), (count, 4)),
        ("frame_index", buffer.frame_index, np.dtype("<i8"), (count,)),
    )
    for name, value, dtype, shape in fields:
        array = np.asarray(value)
        if array.shape != shape:
            raise ContractError(
                f"canonical {name} must have shape {shape}, got {array.shape}"
            )
        if array.dtype != dtype:
            raise ContractError(
                f"canonical {name} has invalid dtype {array.dtype}; expected {dtype}"
            )
        if not array.flags.c_contiguous:
            raise ContractError(f"canonical {name} must be C-contiguous")
    if not np.all(np.isfinite(buffer.joint_position)):
        raise ContractError("canonical joint_position must contain finite values")
    if not np.all(np.isfinite(buffer.joint_velocity)):
        raise ContractError("canonical joint_velocity must contain finite values")
    if not np.all(np.isfinite(buffer.body_quat_w)):
        raise ContractError("canonical body_quat_w must contain finite values")
    frame_index = np.asarray(buffer.frame_index)
    if np.any(frame_index < 0) or not np.all(np.diff(frame_index) == 1):
        raise ContractError(
            "canonical frame_index must be nonnegative and contiguous"
        )
    norms = np.linalg.norm(buffer.body_quat_w.astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1.0) > 1.0e-5):
        raise ContractError("canonical body_quat_w must contain unit quaternions")


def encode_pose_v1(buffer: CanonicalTargetBuffer) -> bytes:
    _validate_canonical_buffer(buffer)
    header_json = json.dumps(
        pose_header(buffer.count),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if len(header_json) > 1280:
        raise ContractError("ZMQ v1 header exceeds 1280 bytes")
    header = header_json + bytes(1280 - len(header_json))
    payload = b"".join((
        buffer.joint_position.astype("<f4", copy=False).tobytes(order="C"),
        buffer.joint_velocity.astype("<f4", copy=False).tobytes(order="C"),
        buffer.body_quat_w.astype("<f4", copy=False).tobytes(order="C"),
        buffer.frame_index.astype("<i8", copy=False).tobytes(order="C"),
        bytes((0,)),
    ))
    return b"pose" + header + payload


@dataclass(frozen=True)
class DecodedPoseV1:
    header: Mapping[str, object]
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_quat_w: np.ndarray
    frame_index: np.ndarray
    catch_up: int

    @property
    def count(self) -> int:
        return int(self.frame_index.shape[0])

    @property
    def buffer(self) -> CanonicalTargetBuffer:
        return CanonicalTargetBuffer(
            joint_position=self.joint_position,
            joint_velocity=self.joint_velocity,
            body_quat_w=self.body_quat_w,
            frame_index=self.frame_index,
        )


@dataclass(frozen=True)
class PreparedPosePublication:
    """One exact archived message awaiting a single local socket send."""

    message: bytes
    archive: Mapping[str, object]
    phase: str
    attempt: int | None
    _publisher_token: object
    _sequence: int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ContractError(f"ZMQ v1 header contains duplicate key: {key}")
        output[key] = value
    return output


def _decode_header(message: bytes) -> tuple[dict[str, object], bytes]:
    if len(message) < len(TOPIC) or message[: len(TOPIC)] != TOPIC:
        raise ContractError("ZMQ v1 message has the wrong topic")
    header_start = len(TOPIC)
    header_end = header_start + HEADER_BYTES
    if len(message) < header_end:
        raise ContractError("ZMQ v1 message is shorter than its fixed header")
    block = message[header_start:header_end]
    terminator = block.find(b"\0")
    if terminator < 0:
        raise ContractError("ZMQ v1 header has no NUL padding")
    if any(block[terminator:]):
        raise ContractError("ZMQ v1 header padding must contain only NUL bytes")
    encoded = block[:terminator]
    if not encoded:
        raise ContractError("ZMQ v1 header JSON must not be empty")
    try:
        text = encoded.decode("utf-8")
        header = json.loads(text, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("ZMQ v1 header is not valid UTF-8 JSON") from error
    if type(header) is not dict:
        raise ContractError("ZMQ v1 header must be a JSON object")
    compact = json.dumps(
        header, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    if compact != encoded:
        raise ContractError("ZMQ v1 header JSON must use compact encoding")
    if set(header) != {"v", "endian", "count", "fields"}:
        raise ContractError("ZMQ v1 header contract has invalid keys")
    count = header["count"]
    if type(count) is not int or count <= 0:
        raise ContractError("ZMQ v1 header contract has an invalid count")
    if header != pose_header(count):
        raise ContractError("ZMQ v1 header contract does not match pose v1")
    return header, message[header_end:]


def _field_layout(
    header: Mapping[str, object],
) -> tuple[tuple[str, np.dtype, tuple[int, ...], int, int], ...]:
    raw_fields = header["fields"]
    if type(raw_fields) is not list:
        raise ContractError("ZMQ v1 header contract fields must be an array")
    offset = 0
    layout: list[tuple[str, np.dtype, tuple[int, ...], int, int]] = []
    for field in raw_fields:
        if type(field) is not dict or set(field) != {"name", "dtype", "shape"}:
            raise ContractError("ZMQ v1 header contract contains an invalid field")
        name = field["name"]
        dtype_name = field["dtype"]
        shape_value = field["shape"]
        if type(name) is not str or dtype_name not in _FIELD_DTYPES:
            raise ContractError("ZMQ v1 header contract contains an invalid dtype")
        if (
            type(shape_value) is not list
            or not shape_value
            or any(type(size) is not int or size <= 0 for size in shape_value)
        ):
            raise ContractError("ZMQ v1 header contract contains an invalid shape")
        shape = tuple(shape_value)
        dtype = _FIELD_DTYPES[dtype_name]
        item_count = math.prod(shape)
        size = item_count * dtype.itemsize
        layout.append((name, dtype, shape, offset, offset + size))
        offset += size
    return tuple(layout)


def decode_pose_v1(
    message: bytes | bytearray | memoryview,
    *,
    baseline_mode: bool = True,
) -> DecodedPoseV1:
    if not isinstance(message, (bytes, bytearray, memoryview)):
        raise ContractError("ZMQ v1 message must be bytes")
    if type(baseline_mode) is not bool:
        raise ContractError("baseline_mode must be a boolean")
    raw_message = bytes(message)
    header, payload = _decode_header(raw_message)
    layout = _field_layout(header)
    expected_size = layout[-1][4] if layout else 0
    if len(payload) != expected_size:
        raise ContractError(
            "ZMQ v1 payload length does not match the header: "
            f"expected {expected_size}, got {len(payload)}"
        )
    decoded: dict[str, np.ndarray] = {}
    for name, dtype, shape, begin, end in layout:
        field = np.frombuffer(payload[begin:end], dtype=dtype)
        decoded[name] = field.reshape(shape).copy(order="C")
    catch_up = int(decoded["catch_up"][0])
    if baseline_mode and catch_up != 0:
        raise ContractError("baseline ZMQ v1 catch_up must remain zero")
    canonical = CanonicalTargetBuffer(
        joint_position=decoded["joint_pos"],
        joint_velocity=decoded["joint_vel"],
        body_quat_w=decoded["body_quat_w"],
        frame_index=decoded["frame_index"],
    )
    _validate_canonical_buffer(canonical)
    return DecodedPoseV1(
        header=MappingProxyType(header),
        joint_position=canonical.joint_position,
        joint_velocity=canonical.joint_velocity,
        body_quat_w=canonical.body_quat_w,
        frame_index=canonical.frame_index,
        catch_up=catch_up,
    )


def _little_endian_bytes(value: np.ndarray, dtype: str) -> bytes:
    return np.asarray(value).astype(dtype, copy=False).tobytes(order="C")


def verify_pose_v1_parity(
    message: bytes | bytearray | memoryview,
    buffer: CanonicalTargetBuffer,
) -> DecodedPoseV1:
    _validate_canonical_buffer(buffer)
    decoded = decode_pose_v1(message, baseline_mode=True)
    comparisons = (
        (
            "joint_position",
            decoded.joint_position,
            buffer.joint_position,
            "<f4",
        ),
        (
            "joint_velocity",
            decoded.joint_velocity,
            buffer.joint_velocity,
            "<f4",
        ),
        ("body_quat_w", decoded.body_quat_w, buffer.body_quat_w, "<f4"),
        ("frame_index", decoded.frame_index, buffer.frame_index, "<i8"),
    )
    for name, actual, expected, dtype in comparisons:
        if (
            actual.shape != expected.shape
            or _little_endian_bytes(actual, dtype)
            != _little_endian_bytes(expected, dtype)
        ):
            raise ContractError(
                f"decoded ZMQ v1 {name} does not have exact bit parity"
            )
    return decoded


class PosePublisher:
    """Own one production PUB-bind socket and archive every exact send."""

    def __init__(
        self,
        endpoint: str,
        *,
        bundle: RunBundle,
        context: object | None = None,
    ) -> None:
        if type(endpoint) is not str or not endpoint:
            raise ContractError("PosePublisher endpoint must be a nonempty string")
        if not isinstance(bundle, RunBundle):
            raise ContractError("PosePublisher requires a RunBundle")
        try:
            import zmq
        except ImportError as error:
            raise ContractError(
                "PosePublisher requires the 'integration' extra with pyzmq"
            ) from error

        owns_context = context is None
        zmq_context = zmq.Context() if owns_context else context
        socket = None
        try:
            socket = zmq_context.socket(zmq.PUB)
            socket.setsockopt(zmq.CONFLATE, 0)
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(endpoint)
            bound = socket.getsockopt(zmq.LAST_ENDPOINT)
            if isinstance(bound, bytes):
                bound = bound.decode("ascii")
            if type(bound) is not str or not bound:
                raise ContractError(
                    "PosePublisher could not resolve its bound endpoint"
                )
        except BaseException:
            if socket is not None:
                socket.close(linger=0)
            if owns_context:
                zmq_context.term()
            raise

        self._bundle = bundle
        self._context = zmq_context
        self._owns_context = owns_context
        self._socket = socket
        self._endpoint = bound
        self._closed = False
        self._publication_token = object()
        self._next_publication = 0
        self._prepared_publications: dict[int, PreparedPosePublication] = {}
        self._sent_publications: set[int] = set()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def prepare(
        self,
        buffer: CanonicalTargetBuffer,
        *,
        phase: str = "timeline",
        attempt: int | None = None,
    ) -> PreparedPosePublication:
        if self._closed:
            raise ContractError("PosePublisher is closed")
        try:
            message = encode_pose_v1(buffer)
            decoded = verify_pose_v1_parity(message, buffer)
        except BaseException as error:
            _tag_failure(error, "zmq_encoding")
            raise
        first_frame_index = int(decoded.frame_index[0])
        last_frame_index = int(decoded.frame_index[-1])
        try:
            archived = self._bundle.archive_transmission(
                message,
                first_frame_index=first_frame_index,
                last_frame_index=last_frame_index,
                phase=phase,
                attempt=attempt,
            )
        except BaseException as error:
            _tag_failure(error, "artifact_enqueue")
            raise
        sequence = self._next_publication
        self._next_publication += 1
        publication = PreparedPosePublication(
            message=message,
            archive=archived,
            phase=phase,
            attempt=attempt,
            _publisher_token=self._publication_token,
            _sequence=sequence,
        )
        self._prepared_publications[sequence] = publication
        return publication

    def send_prepared(
        self,
        publication: PreparedPosePublication,
    ) -> Mapping[str, object]:
        if self._closed:
            raise ContractError("PosePublisher is closed")
        if not isinstance(publication, PreparedPosePublication):
            raise ContractError("local send requires a prepared pose publication")
        if (
            publication._publisher_token is not self._publication_token
            or self._prepared_publications.get(publication._sequence)
            is not publication
        ):
            raise ContractError(
                "local send requires the exact prepared object archived by this publisher"
            )
        if publication._sequence in self._sent_publications:
            raise ContractError("prepared pose publication was already sent")
        self._socket.send(publication.message)
        self._sent_publications.add(publication._sequence)
        result = dict(publication.archive)
        result["local_send_completed"] = True
        return MappingProxyType(result)

    def send(
        self,
        buffer: CanonicalTargetBuffer,
        *,
        phase: str = "timeline",
        attempt: int | None = None,
    ) -> Mapping[str, object]:
        publication = self.prepare(buffer, phase=phase, attempt=attempt)
        return self.send_prepared(publication)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._socket.close(linger=0)
        if self._owns_context:
            self._context.term()

    def __enter__(self) -> "PosePublisher":
        if self._closed:
            raise ContractError("PosePublisher is closed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
