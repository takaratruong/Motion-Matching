"""Pure 25 Hz terrain-bank motion-index derivation and G1MI/v1 codec."""

import math
import os
import struct
from dataclasses import dataclass

import numpy as np

from .schema import TerrainBankIndex


DIRECTION_IDLE = 0x001
DIRECTION_FORWARD = 0x002
DIRECTION_FORWARD_RIGHT = 0x004
DIRECTION_RIGHT = 0x008
DIRECTION_BACK_RIGHT = 0x010
DIRECTION_BACKWARD = 0x020
DIRECTION_BACK_LEFT = 0x040
DIRECTION_LEFT = 0x080
DIRECTION_FORWARD_LEFT = 0x100

SPEED_LOW = 0x01
SPEED_MOVING = 0x02

DIRECTION_OVERLAP_HALF_WIDTH_DEGREES = 5.0
MAX_LATERAL_HEADING_CHANGE_DEGREES = 15.0
MOVING_MIN_DISPLACEMENT_METRES = 0.08
LOW_SPEED_MAX_DISPLACEMENT_METRES = 0.12
ELEVATION_CHANGE_THRESHOLD_METRES = 0.03
ONE_SECOND_FRAMES = 25

G1MI_MAGIC = b"G1MI"
G1MI_VERSION = 1
G1MI_ROW_FORMAT = "<HBb"

_HEADER = struct.Struct("<4sIII")
_ROW = struct.Struct(G1MI_ROW_FORMAT)
G1MI_ROW_WIDTH = _ROW.size
_ROW_DTYPE = np.dtype([
    ("direction_mask", "<u2"),
    ("speed_mask", "u1"),
    ("elevation_mode", "i1"),
])
_UINT32_MAX = (1 << 32) - 1
_DIRECTION_MOVING_BITS = (
    DIRECTION_FORWARD,
    DIRECTION_FORWARD_RIGHT,
    DIRECTION_RIGHT,
    DIRECTION_BACK_RIGHT,
    DIRECTION_BACKWARD,
    DIRECTION_BACK_LEFT,
    DIRECTION_LEFT,
    DIRECTION_FORWARD_LEFT,
)
_VALID_DIRECTION_MASKS = np.array(
    (
        DIRECTION_IDLE,
        *_DIRECTION_MOVING_BITS,
        *(
            bit | _DIRECTION_MOVING_BITS[(index + 1) % 8]
            for index, bit in enumerate(_DIRECTION_MOVING_BITS)
        ),
    ),
    dtype=np.uint16,
)
_VALID_SPEED_MASKS = np.array(
    (SPEED_LOW, SPEED_MOVING, SPEED_LOW | SPEED_MOVING),
    dtype=np.uint8,
)
_SECTOR_CENTERS = tuple(
    (math.radians(45.0 * index), bit)
    for index, bit in enumerate(_DIRECTION_MOVING_BITS)
)
_SECTOR_HALF_WIDTH_RADIANS = math.radians(
    22.5 + DIRECTION_OVERLAP_HALF_WIDTH_DEGREES
)
_SECTOR_COMPARISON_TOLERANCE_RADIANS = 8 * math.ulp(
    _SECTOR_HALF_WIDTH_RADIANS
)
_MAX_LATERAL_HEADING_CHANGE_RADIANS = math.radians(
    MAX_LATERAL_HEADING_CHANGE_DEGREES
)


def _integer_vector(values, label: str) -> np.ndarray:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a one-dimensional integer array") from error
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError(f"{label} must be a one-dimensional integer array")
    return source


def _checked_sizes(frame_count: int) -> tuple[int, int]:
    if frame_count < 1 or frame_count > _UINT32_MAX:
        raise ValueError("invalid or overflowing G1MI frame count")
    platform_limit = np.iinfo(np.intp).max
    if frame_count > (platform_limit - _HEADER.size) // G1MI_ROW_WIDTH:
        raise ValueError("G1MI payload exceeds platform index limit")
    payload_size = frame_count * G1MI_ROW_WIDTH
    return payload_size, _HEADER.size + payload_size


def _validated_arrays(
    direction_masks,
    speed_masks,
    elevation_modes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = _integer_vector(direction_masks, "direction masks")
    speeds = _integer_vector(speed_masks, "speed masks")
    elevations = _integer_vector(elevation_modes, "elevation modes")
    frame_count = len(directions)
    if len(speeds) != frame_count or len(elevations) != frame_count:
        raise ValueError("motion index arrays must have the same frame count")
    _checked_sizes(frame_count)

    if directions.dtype.kind == "i" and np.any(directions < 0):
        raise ValueError("invalid direction mask")
    if np.any(directions > np.iinfo(np.uint16).max):
        raise ValueError("invalid direction mask")
    normalized_directions = np.array(
        directions, dtype=np.uint16, copy=True, order="C"
    )
    if not np.isin(normalized_directions, _VALID_DIRECTION_MASKS).all():
        raise ValueError("invalid direction mask")

    if speeds.dtype.kind == "i" and np.any(speeds < 0):
        raise ValueError("invalid speed mask")
    if np.any(speeds > np.iinfo(np.uint8).max):
        raise ValueError("invalid speed mask")
    normalized_speeds = np.array(
        speeds, dtype=np.uint8, copy=True, order="C"
    )
    if not np.isin(normalized_speeds, _VALID_SPEED_MASKS).all():
        raise ValueError("invalid speed mask")

    if np.any(elevations < -1) or np.any(elevations > 1):
        raise ValueError("invalid elevation mode")
    normalized_elevations = np.array(
        elevations, dtype=np.int8, copy=True, order="C"
    )
    return normalized_directions, normalized_speeds, normalized_elevations


@dataclass(frozen=True)
class MotionIndex:
    direction_masks: np.ndarray
    speed_masks: np.ndarray
    elevation_modes: np.ndarray

    def __post_init__(self) -> None:
        directions, speeds, elevations = _validated_arrays(
            self.direction_masks,
            self.speed_masks,
            self.elevation_modes,
        )
        for values in (directions, speeds, elevations):
            values.setflags(write=False)
        object.__setattr__(self, "direction_masks", directions)
        object.__setattr__(self, "speed_masks", speeds)
        object.__setattr__(self, "elevation_modes", elevations)

    def __len__(self) -> int:
        return len(self.direction_masks)


def _real_derivation_array(values, shape, label: str) -> np.ndarray:
    try:
        source = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} has invalid shape or dtype") from error
    if source.shape != shape or source.dtype.kind != "f":
        raise ValueError(f"{label} has invalid shape or dtype")
    if not np.isfinite(source).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.array(source, dtype=np.float64, copy=True, order="C")


def _direction_mask(
    dx: float,
    dz: float,
    heading: float,
    heading_change: float,
    distance: float,
) -> int:
    if distance < MOVING_MIN_DISPLACEMENT_METRES:
        return DIRECTION_IDLE

    sine = math.sin(heading)
    cosine = math.cos(heading)
    local_right = dx * cosine - dz * sine
    local_forward = dx * sine + dz * cosine
    travel_angle = math.atan2(local_right, local_forward)
    mask = 0
    for center, bit in _SECTOR_CENTERS:
        sector_distance = abs(math.remainder(
            travel_angle - center, math.tau
        ))
        if sector_distance <= (
            _SECTOR_HALF_WIDTH_RADIANS
            + _SECTOR_COMPARISON_TOLERANCE_RADIANS
        ):
            mask |= bit

    if heading_change > _MAX_LATERAL_HEADING_CHANGE_RADIANS:
        mask &= DIRECTION_FORWARD | DIRECTION_BACKWARD
    return mask or DIRECTION_IDLE


def _speed_mask(distance: float) -> int:
    mask = 0
    if distance <= LOW_SPEED_MAX_DISPLACEMENT_METRES:
        mask |= SPEED_LOW
    if distance >= MOVING_MIN_DISPLACEMENT_METRES:
        mask |= SPEED_MOVING
    return mask


def _elevation_mode(change: float) -> int:
    if change > ELEVATION_CHANGE_THRESHOLD_METRES:
        return 1
    if change < -ELEVATION_CHANGE_THRESHOLD_METRES:
        return -1
    return 0


def derive_motion_index(
    root_positions,
    root_headings,
    terrain_banks: TerrainBankIndex,
) -> MotionIndex:
    """Derive per-frame masks without crossing source-global ranges.

    Positions use simulation ``(x, y, z)`` coordinates. Heading is radians,
    with zero along ``+Z`` and positive rotation toward ``+X``; it is observed
    independently and is never changed to match travel.
    """
    try:
        frame_count = len(root_positions)
    except TypeError as error:
        raise ValueError("root positions have invalid shape or dtype") from error
    positions = _real_derivation_array(
        root_positions, (frame_count, 3), "root positions"
    )
    headings = _real_derivation_array(
        root_headings, (frame_count,), "root headings"
    )
    if not isinstance(terrain_banks, TerrainBankIndex):
        raise ValueError("terrain banks must be a TerrainBankIndex")
    terrain_banks.validate()
    if terrain_banks.frame_count != frame_count:
        raise ValueError("terrain bank frame count does not match root positions")

    directions = np.zeros(frame_count, dtype=np.uint16)
    speeds = np.zeros(frame_count, dtype=np.uint8)
    elevations = np.zeros(frame_count, dtype=np.int8)
    for source_range in terrain_banks.ranges:
        for frame in range(source_range.global_start, source_range.global_stop):
            horizon = min(
                frame + ONE_SECOND_FRAMES,
                source_range.global_stop - 1,
            )
            displacement = positions[horizon] - positions[frame]
            dx = float(displacement[0])
            dz = float(displacement[2])
            distance = math.hypot(dx, dz)
            heading_change = abs(math.remainder(
                float(headings[horizon] - headings[frame]), math.tau
            ))
            directions[frame] = _direction_mask(
                dx,
                dz,
                float(headings[frame]),
                heading_change,
                distance,
            )
            speeds[frame] = _speed_mask(distance)
            elevations[frame] = _elevation_mode(float(displacement[1]))

    return MotionIndex(directions, speeds, elevations)


def motion_index_bytes(
    direction_masks,
    speed_masks,
    elevation_modes,
) -> bytes:
    """Return a deterministic, fully validated G1MI/v1 payload."""
    directions, speeds, elevations = _validated_arrays(
        direction_masks,
        speed_masks,
        elevation_modes,
    )
    rows = np.empty(len(directions), dtype=_ROW_DTYPE)
    rows["direction_mask"] = directions
    rows["speed_mask"] = speeds
    rows["elevation_mode"] = elevations
    return (
        _HEADER.pack(
            G1MI_MAGIC,
            G1MI_VERSION,
            len(directions),
            G1MI_ROW_WIDTH,
        )
        + rows.tobytes(order="C")
    )


def write_motion_index(
    path: os.PathLike | str,
    direction_masks,
    speed_masks,
    elevation_modes,
) -> None:
    """Validate and encode completely before opening ``path`` for writing."""
    payload = motion_index_bytes(
        direction_masks,
        speed_masks,
        elevation_modes,
    )
    with open(path, "wb") as stream:
        stream.write(payload)


def _parse_header(header: bytes) -> tuple[int, int, int]:
    if len(header) < _HEADER.size:
        raise ValueError("truncated G1MI header")
    magic, version, frame_count, row_width = _HEADER.unpack_from(header)
    if magic != G1MI_MAGIC:
        raise ValueError("invalid G1MI magic")
    if version != G1MI_VERSION:
        raise ValueError("unsupported G1MI version")
    if row_width != G1MI_ROW_WIDTH:
        raise ValueError("invalid G1MI row width")
    payload_size, total_size = _checked_sizes(frame_count)
    return frame_count, payload_size, total_size


def _decode_rows(payload: bytes, frame_count: int) -> MotionIndex:
    rows = np.frombuffer(payload, dtype=_ROW_DTYPE, count=frame_count)
    return MotionIndex(
        rows["direction_mask"],
        rows["speed_mask"],
        rows["elevation_mode"],
    )


def motion_index_from_bytes(payload: bytes) -> MotionIndex:
    """Decode exact G1MI/v1 bytes into owned, read-only arrays."""
    if type(payload) is not bytes:
        raise TypeError("G1MI payload must be exact bytes")
    frame_count, payload_size, total_size = _parse_header(payload[:_HEADER.size])
    if len(payload) < total_size:
        raise ValueError("truncated G1MI payload")
    if len(payload) > total_size:
        raise ValueError("trailing G1MI bytes")
    return _decode_rows(payload[_HEADER.size:_HEADER.size + payload_size], frame_count)


def read_motion_index(path: os.PathLike | str) -> MotionIndex:
    """Read G1MI/v1 after checking header arithmetic and exact file size."""
    with open(path, "rb") as stream:
        header = stream.read(_HEADER.size)
        frame_count, payload_size, total_size = _parse_header(header)
        actual_size = os.fstat(stream.fileno()).st_size
        if actual_size < total_size:
            raise ValueError(f"{path}: truncated G1MI payload")
        if actual_size > total_size:
            raise ValueError(f"{path}: trailing G1MI bytes")
        payload = stream.read(payload_size)
        if len(payload) != payload_size:
            raise ValueError(f"{path}: truncated G1MI payload")
    return _decode_rows(payload, frame_count)
