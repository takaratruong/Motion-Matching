"""Strict reader for the Holden/G1 walking database binary format."""

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from resources.g1_interaction_builder.schema import G1_SKELETON


BONE_COUNT = 31
CONTACT_COUNT = 2


@dataclass(frozen=True)
class HoldenDatabase:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contacts: np.ndarray


def _read_exact(data: memoryview, offset: int, size: int) -> tuple[memoryview, int]:
    end = offset + size
    if size < 0 or end > len(data):
        raise ValueError("truncated Holden database")
    return data[offset:end], end


def _read_u32(data: memoryview, offset: int) -> tuple[int, int]:
    raw, offset = _read_exact(data, offset, 4)
    return struct.unpack("<I", raw)[0], offset


def _read_shape_header(
    data: memoryview, offset: int, label: str, frames: int | None
) -> tuple[int, int]:
    row_count, offset = _read_u32(data, offset)
    bones, offset = _read_u32(data, offset)
    if frames is not None and row_count != frames:
        raise ValueError(f"{label} frame count does not match positions")
    if bones != BONE_COUNT:
        raise ValueError(f"{label} must contain exactly {BONE_COUNT} bones")
    return row_count, offset


def _read_array(
    data: memoryview, offset: int, dtype: str, shape: tuple[int, ...]
) -> tuple[np.ndarray, int]:
    count = int(np.prod(shape, dtype=np.int64))
    raw, offset = _read_exact(data, offset, count * np.dtype(dtype).itemsize)
    return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape).copy(), offset


def _require_finite(*arrays: np.ndarray) -> None:
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Holden database contains non-finite float values")


def _validate_ranges(
    starts: np.ndarray, stops: np.ndarray, frames: int
) -> None:
    if len(starts) == 0 or starts.shape != stops.shape:
        raise ValueError("Holden database ranges are invalid")
    if (
        starts[0] != 0
        or stops[-1] != frames
        or np.any(starts < 0)
        or np.any(stops > frames)
        or np.any(starts >= stops)
        or not np.array_equal(starts[1:], stops[:-1])
    ):
        raise ValueError("Holden database ranges must be contiguous and cover frames")


def read_holden_database(path: Path) -> HoldenDatabase:
    """Read exactly the eight arrays written by ``generate_database_g1.py``."""
    data = memoryview(Path(path).read_bytes())
    offset = 0

    frames, offset = _read_shape_header(data, offset, "positions", None)
    if frames == 0:
        raise ValueError("Holden database must contain at least one frame")
    positions, offset = _read_array(data, offset, "<f4", (frames, BONE_COUNT, 3))

    _, offset = _read_shape_header(data, offset, "velocities", frames)
    velocities, offset = _read_array(data, offset, "<f4", (frames, BONE_COUNT, 3))

    _, offset = _read_shape_header(data, offset, "rotations", frames)
    rotations, offset = _read_array(data, offset, "<f4", (frames, BONE_COUNT, 4))

    _, offset = _read_shape_header(data, offset, "angular velocities", frames)
    angular_velocities, offset = _read_array(
        data, offset, "<f4", (frames, BONE_COUNT, 3)
    )

    parent_count, offset = _read_u32(data, offset)
    if parent_count != BONE_COUNT:
        raise ValueError(f"parents must contain exactly {BONE_COUNT} bones")
    parents, offset = _read_array(data, offset, "<i4", (BONE_COUNT,))

    range_count, offset = _read_u32(data, offset)
    starts, offset = _read_array(data, offset, "<i4", (range_count,))
    stop_count, offset = _read_u32(data, offset)
    stops, offset = _read_array(data, offset, "<i4", (stop_count,))

    contact_frames, offset = _read_u32(data, offset)
    contact_count, offset = _read_u32(data, offset)
    if contact_frames != frames:
        raise ValueError("contacts frame count does not match positions")
    if contact_count != CONTACT_COUNT:
        raise ValueError("Holden database must contain exactly two contacts")
    contacts, offset = _read_array(data, offset, "u1", (frames, CONTACT_COUNT))

    if offset != len(data):
        raise ValueError("trailing bytes in Holden database")
    if not np.array_equal(parents, G1_SKELETON.parents):
        raise ValueError("parents must match the exact canonical G1 hierarchy")
    _require_finite(positions, velocities, rotations, angular_velocities)
    _validate_ranges(starts, stops, frames)
    return HoldenDatabase(
        positions=positions,
        velocities=velocities,
        rotations=rotations,
        angular_velocities=angular_velocities,
        parents=parents,
        range_starts=starts,
        range_stops=stops,
        contacts=contacts,
    )
