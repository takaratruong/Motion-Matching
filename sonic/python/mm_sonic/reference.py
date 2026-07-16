"""Official SONIC reference-directory export from one canonical session buffer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import stat

import numpy as np

from .artifacts import RunBundle, verify_run_inventory
from .joints import ContractError
from .timeline import CanonicalTargetBuffer


JOINT_POS_HEADER = tuple(f"joint_{index}" for index in range(29))
JOINT_VEL_HEADER = tuple(f"joint_vel_{index}" for index in range(29))
BODY_QUAT_HEADER = ("body_0_w", "body_0_x", "body_0_y", "body_0_z")
BODY_POS_HEADER = ("body_0_x", "body_0_y", "body_0_z")
ROOT_DIAGNOSTIC_HEADER = (
    "frame_index",
    "physical_pelvis_x",
    "physical_pelvis_y",
    "physical_pelvis_z",
    "virtual_root_x",
    "virtual_root_y",
    "virtual_root_z",
    "virtual_root_qw",
    "virtual_root_qx",
    "virtual_root_qy",
    "virtual_root_qz",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def format_f32(value: np.float32) -> str:
    scalar = np.float32(value)
    if not np.isfinite(scalar):
        raise ContractError("cannot serialize non-finite float32")
    return format(float(scalar), ".9g")


def _owned_f32(value: object, columns: int, label: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] != columns:
        raise ContractError(f"{label} must have shape [N,{columns}]")
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.ascontiguousarray(source, dtype=np.float32).copy(order="C")
    if not np.all(np.isfinite(output)):
        raise ContractError(f"{label} must contain finite float32 values")
    output.flags.writeable = False
    return output


@dataclass(frozen=True)
class ReferenceDiagnostics:
    physical_pelvis_position: np.ndarray
    virtual_root_position: np.ndarray
    virtual_root_quat_w: np.ndarray

    def __post_init__(self) -> None:
        physical = _owned_f32(
            self.physical_pelvis_position, 3, "physical pelvis diagnostic"
        )
        virtual = _owned_f32(self.virtual_root_position, 3, "virtual root diagnostic")
        quaternion = _owned_f32(
            self.virtual_root_quat_w, 4, "virtual root quaternion diagnostic"
        )
        if not (physical.shape[0] == virtual.shape[0] == quaternion.shape[0]):
            raise ContractError("diagnostic arrays must have the same frame count")
        norms = np.linalg.norm(quaternion.astype(np.float64), axis=1)
        if np.any(np.abs(norms - 1.0) > 1.0e-5):
            raise ContractError("virtual root diagnostics must contain unit quaternions")
        object.__setattr__(self, "physical_pelvis_position", physical)
        object.__setattr__(self, "virtual_root_position", virtual)
        object.__setattr__(self, "virtual_root_quat_w", quaternion)

    @property
    def count(self) -> int:
        return int(self.physical_pelvis_position.shape[0])


@dataclass(frozen=True)
class WrittenReference:
    reference_directory: Path
    canonical_target_sha256: str
    diagnostic_sha256: str
    frame_count: int


def _canonical_hash(buffer: CanonicalTargetBuffer) -> str:
    digest = hashlib.sha256()
    for value, dtype in (
        (buffer.joint_position, "<f4"),
        (buffer.joint_velocity, "<f4"),
        (buffer.body_quat_w, "<f4"),
        (buffer.frame_index, "<i8"),
    ):
        digest.update(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def _diagnostic_hash(diagnostics: ReferenceDiagnostics) -> str:
    digest = hashlib.sha256()
    for value in (
        diagnostics.physical_pelvis_position,
        diagnostics.virtual_root_position,
        diagnostics.virtual_root_quat_w,
    ):
        digest.update(np.ascontiguousarray(value, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def _csv_bytes(header: tuple[str, ...], values: np.ndarray) -> bytes:
    rows = [",".join(header)]
    rows.extend(
        ",".join(format_f32(value) for value in row)
        for row in np.asarray(values, dtype=np.float32)
    )
    return ("\n".join(rows) + "\n").encode("ascii")


def _diagnostic_csv_bytes(
    frame_index: np.ndarray, diagnostics: ReferenceDiagnostics
) -> bytes:
    values = np.concatenate(
        (
            diagnostics.physical_pelvis_position,
            diagnostics.virtual_root_position,
            diagnostics.virtual_root_quat_w,
        ),
        axis=1,
    )
    rows = [",".join(ROOT_DIAGNOSTIC_HEADER)]
    for index, row in zip(frame_index, values, strict=True):
        rows.append(
            ",".join((str(int(index)), *(format_f32(value) for value in row)))
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def _npz_bytes(buffer: CanonicalTargetBuffer) -> bytes:
    output = io.BytesIO()
    np.savez(
        output,
        joint_position=np.ascontiguousarray(buffer.joint_position, dtype="<f4"),
        joint_velocity=np.ascontiguousarray(buffer.joint_velocity, dtype="<f4"),
        body_quat_w=np.ascontiguousarray(buffer.body_quat_w, dtype="<f4"),
        frame_index=np.ascontiguousarray(buffer.frame_index, dtype="<i8"),
    )
    return output.getvalue()


def write_reference_bundle(
    bundle: RunBundle,
    canonical: CanonicalTargetBuffer,
    diagnostics: ReferenceDiagnostics,
    *,
    source_sha256: str,
    scene_id: str,
    route_id: str | None,
) -> WrittenReference:
    if not isinstance(bundle, RunBundle):
        raise ContractError("reference output requires a RunBundle")
    if not isinstance(canonical, CanonicalTargetBuffer):
        raise ContractError("reference output requires a CanonicalTargetBuffer")
    if not isinstance(diagnostics, ReferenceDiagnostics):
        raise ContractError("reference output requires ReferenceDiagnostics")
    expected_indices = np.arange(canonical.count, dtype=np.int64)
    if not np.array_equal(canonical.frame_index, expected_indices):
        raise ContractError("official reference must include frame zero and all rows")
    if (canonical.count - 1) % 20 != 0:
        raise ContractError(
            "official reference must contain frame zero plus complete 20-row chunks"
        )
    if diagnostics.count != canonical.count:
        raise ContractError("diagnostic frame count does not match the canonical buffer")
    if type(source_sha256) is not str or _SHA256.fullmatch(source_sha256) is None:
        raise ContractError("source_sha256 must be a lowercase SHA-256 digest")
    if type(scene_id) is not str or not scene_id:
        raise ContractError("scene_id must be a nonempty string")
    if route_id is not None and (type(route_id) is not str or not route_id):
        raise ContractError("route_id must be null or a nonempty string")

    canonical_sha256 = _canonical_hash(canonical)
    diagnostic_sha256 = _diagnostic_hash(diagnostics)
    prefix = "reference/mm_sonic"
    body_position = np.zeros((canonical.count, 3), dtype=np.float32)
    outputs = {
        f"{prefix}/joint_pos.csv": _csv_bytes(
            JOINT_POS_HEADER, canonical.joint_position
        ),
        f"{prefix}/joint_vel.csv": _csv_bytes(
            JOINT_VEL_HEADER, canonical.joint_velocity
        ),
        f"{prefix}/body_quat.csv": _csv_bytes(
            BODY_QUAT_HEADER, canonical.body_quat_w
        ),
        f"{prefix}/body_pos.csv": _csv_bytes(BODY_POS_HEADER, body_position),
    }
    metadata = (
        "Metadata for: mm_sonic\n"
        "==============================\n\n"
        "Body part indexes:\n"
        "[0]\n\n"
        f"Total timesteps: {canonical.count}\n"
    )
    route = "null" if route_id is None else route_id
    info = (
        "MM-to-SONIC official reference\n"
        "source_rate_hz: 25\n"
        "target_rate_hz: 50\n"
        f"frame_count: {canonical.count}\n"
        f"joint_position_shape: [{canonical.count}, 29]\n"
        f"joint_velocity_shape: [{canonical.count}, 29]\n"
        f"body_quaternion_shape: [{canonical.count}, 4]\n"
        f"body_position_shape: [{canonical.count}, 3]\n"
        f"source_sha256: {source_sha256}\n"
        f"canonical_target_sha256: {canonical_sha256}\n"
        f"diagnostic_sha256: {diagnostic_sha256}\n"
        f"scene_id: {scene_id}\n"
        f"route_id: {route}\n"
        "body_position: zero and untracked\n"
    )
    outputs[f"{prefix}/metadata.txt"] = metadata.encode("utf-8")
    outputs[f"{prefix}/info.txt"] = info.encode("utf-8")
    outputs["mm_root_diagnostic.csv"] = _diagnostic_csv_bytes(
        canonical.frame_index, diagnostics
    )
    outputs["canonical_target.npz"] = _npz_bytes(canonical)
    for relative, data in outputs.items():
        bundle.write_bytes(relative, data)

    validate_reference_bundle(bundle.path, canonical, diagnostics)
    return WrittenReference(
        reference_directory=bundle.path / "reference" / "mm_sonic",
        canonical_target_sha256=canonical_sha256,
        diagnostic_sha256=diagnostic_sha256,
        frame_count=canonical.count,
    )


def _read_regular(path: Path) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"reference file is a symlink: {path.name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"reference path is not a regular file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_csv(path: Path, header: tuple[str, ...]) -> np.ndarray:
    try:
        raw = _read_regular(path)
        text = raw.decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError(f"cannot decode reference CSV: {path.name}") from error
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError(f"reference CSV lacks an exact final newline: {path.name}")
    lines = text.splitlines()
    if not lines or lines[0] != ",".join(header):
        raise ContractError(f"reference CSV header mismatch: {path.name}")
    if len(lines) <= 1:
        raise ContractError(f"reference CSV has no rows: {path.name}")
    output = np.empty((len(lines) - 1, len(header)), dtype=np.float32)
    for row_index, line in enumerate(lines[1:]):
        values = line.split(",")
        if len(values) != len(header):
            raise ContractError(f"reference CSV shape mismatch: {path.name}")
        for column_index, token in enumerate(values):
            try:
                value = np.float32(token)
            except ValueError as error:
                raise ContractError(f"invalid float32 in {path.name}") from error
            if not np.isfinite(value) or token != format_f32(value):
                raise ContractError(f"non-canonical float32 in {path.name}")
            output[row_index, column_index] = value
    return output


def read_body_indexes(path: str | os.PathLike[str]) -> tuple[int, ...]:
    file_path = Path(path)
    try:
        text = _read_regular(file_path).decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractError("metadata.txt is unreadable") from error
    match = re.search(r"(?m)^Body part indexes:\n\[([^\]]*)\]$", text)
    if match is None:
        raise ContractError("metadata.txt lacks Body part indexes")
    content = match.group(1).strip()
    if not content:
        return ()
    tokens = content.replace(",", " ").split()
    try:
        return tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ContractError("metadata.txt has invalid body indexes") from error


def load_reference_directory(
    path: str | os.PathLike[str],
) -> CanonicalTargetBuffer:
    root = Path(path).expanduser().resolve(strict=True)
    joint_position = _decode_csv(root / "joint_pos.csv", JOINT_POS_HEADER)
    joint_velocity = _decode_csv(root / "joint_vel.csv", JOINT_VEL_HEADER)
    body_quat = _decode_csv(root / "body_quat.csv", BODY_QUAT_HEADER)
    body_position = _decode_csv(root / "body_pos.csv", BODY_POS_HEADER)
    count = joint_position.shape[0]
    if (
        joint_velocity.shape != (count, 29)
        or body_quat.shape != (count, 4)
        or body_position.shape != (count, 3)
    ):
        raise ContractError("official reference CSV frame counts or shapes differ")
    if np.any(body_position.view(np.uint32) != np.uint32(0)):
        raise ContractError("official body_pos.csv must contain positive float32 zeros")
    if read_body_indexes(root / "metadata.txt") != (0,):
        raise ContractError("official metadata body indexes must equal [0]")
    return CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=body_quat,
        frame_index=np.arange(count, dtype=np.int64),
    )


def _require_bits_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.shape != expected.shape or not np.array_equal(
        np.ascontiguousarray(actual, dtype=np.float32).view(np.uint32),
        np.ascontiguousarray(expected, dtype=np.float32).view(np.uint32),
    ):
        raise ContractError(f"{label} is not bit-equal to the canonical buffer")


def _decode_diagnostics(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    raw = _read_regular(path)
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise ContractError("mm_root_diagnostic.csv lacks an exact final newline")
    lines = raw.decode("ascii").splitlines()
    if not lines or lines[0] != ",".join(ROOT_DIAGNOSTIC_HEADER):
        raise ContractError("mm_root_diagnostic.csv header mismatch")
    if len(lines) != count + 1:
        raise ContractError("mm_root_diagnostic.csv frame count mismatch")
    indices = np.empty(count, dtype=np.int64)
    values = np.empty((count, 10), dtype=np.float32)
    for row_index, line in enumerate(lines[1:]):
        tokens = line.split(",")
        if len(tokens) != 11:
            raise ContractError("mm_root_diagnostic.csv shape mismatch")
        try:
            indices[row_index] = int(tokens[0])
        except ValueError as error:
            raise ContractError("invalid diagnostic frame index") from error
        if tokens[0] != str(int(indices[row_index])):
            raise ContractError("non-canonical diagnostic frame index")
        for column_index, token in enumerate(tokens[1:]):
            value = np.float32(token)
            if not np.isfinite(value) or token != format_f32(value):
                raise ContractError("non-canonical diagnostic float32")
            values[row_index, column_index] = value
    return indices, values


def validate_reference_bundle(
    run_path: str | os.PathLike[str],
    canonical: CanonicalTargetBuffer,
    diagnostics: ReferenceDiagnostics,
) -> bool:
    root = Path(run_path).expanduser().resolve(strict=True)
    reference = root / "reference" / "mm_sonic"
    required = {
        "joint_pos.csv",
        "joint_vel.csv",
        "body_quat.csv",
        "body_pos.csv",
        "metadata.txt",
        "info.txt",
    }
    actual = {candidate.name for candidate in reference.iterdir()}
    missing = sorted(required - actual)
    if missing:
        raise ContractError(f"required reference file is missing: {missing[0]}")
    extra = sorted(actual - required)
    if extra:
        raise ContractError(f"unexpected reference file: {extra[0]}")
    loaded = load_reference_directory(reference)
    _require_bits_equal(loaded.joint_position, canonical.joint_position, "joint_pos.csv")
    _require_bits_equal(loaded.joint_velocity, canonical.joint_velocity, "joint_vel.csv")
    _require_bits_equal(loaded.body_quat_w, canonical.body_quat_w, "body_quat.csv")
    if not np.array_equal(loaded.frame_index, canonical.frame_index):
        raise ContractError("reference frame indexes are not bit-equal to canonical")

    archive_bytes = _read_regular(root / "canonical_target.npz")
    try:
        with np.load(io.BytesIO(archive_bytes), allow_pickle=False) as archive:
            if set(archive.files) != {
                "joint_position",
                "joint_velocity",
                "body_quat_w",
                "frame_index",
            }:
                raise ContractError("canonical_target.npz has unexpected arrays")
            for name in ("joint_position", "joint_velocity", "body_quat_w"):
                _require_bits_equal(archive[name], getattr(canonical, name), f"npz {name}")
            if not np.array_equal(archive["frame_index"], canonical.frame_index):
                raise ContractError("npz frame_index is not bit-equal to canonical")
    except (OSError, ValueError) as error:
        raise ContractError("canonical_target.npz is invalid") from error

    indices, values = _decode_diagnostics(root / "mm_root_diagnostic.csv", canonical.count)
    if not np.array_equal(indices, canonical.frame_index):
        raise ContractError("diagnostic frame indexes are not bit-equal to canonical")
    _require_bits_equal(values[:, 0:3], diagnostics.physical_pelvis_position, "physical pelvis diagnostics")
    _require_bits_equal(values[:, 3:6], diagnostics.virtual_root_position, "virtual root diagnostics")
    _require_bits_equal(values[:, 6:10], diagnostics.virtual_root_quat_w, "virtual root quaternion diagnostics")
    if (root / "inventory.json").exists():
        verify_run_inventory(root)
    return True
