"""Validated dimensions and range-safe temporal sampling for G1 LMM."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from resources import quat


@dataclass(frozen=True)
class G1LmmDimensions:
    features: int = 31
    latent: int = 32
    bones: int = 31
    contacts: int = 2

    def __post_init__(self) -> None:
        for name in ("features", "latent", "bones", "contacts"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bones < 2:
            raise ValueError("bones must include a root and at least one child")

    @property
    def state(self) -> int:
        return self.features + self.latent

    @property
    def compressor_input(self) -> int:
        # Local and character-space position (3), rotation-6D (6), linear
        # velocity (3), and angular velocity (3), plus root velocities/contact.
        return 2 * 15 * (self.bones - 1) + 6 + self.contacts

    @property
    def decompressor_output(self) -> int:
        return 15 * (self.bones - 1) + 6 + self.contacts


@dataclass(frozen=True)
class TrainingBundle:
    root: Path
    manifest: dict
    manifest_sha256: str
    dimensions: G1LmmDimensions
    positions: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contacts: np.ndarray
    features: np.ndarray
    feature_offset: np.ndarray
    feature_scale: np.ndarray

    @property
    def frames(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class TrainingArrays:
    compressor_input: np.ndarray
    decompressor_target: np.ndarray
    local_positions: np.ndarray
    local_rotation_xy: np.ndarray
    local_velocities: np.ndarray
    local_angular_velocities: np.ndarray
    character_positions: np.ndarray
    character_transforms: np.ndarray
    character_rotation_xy: np.ndarray
    character_velocities: np.ndarray
    character_angular_velocities: np.ndarray
    root_velocity: np.ndarray
    root_angular_velocity: np.ndarray


class _BinaryCursor:
    def __init__(self, payload: bytes, label: str) -> None:
        self.payload = payload
        self.label = label
        self.offset = 0

    def unpack(self, fmt: str, description: str) -> tuple[int, ...]:
        size = struct.calcsize(fmt)
        if self.offset + size > len(self.payload):
            raise ValueError(f"{self.label}: truncated {description}")
        result = struct.unpack_from(fmt, self.payload, self.offset)
        self.offset += size
        return result

    def array(self, shape: tuple[int, ...], dtype: str, description: str) -> np.ndarray:
        count = int(np.prod(shape, dtype=np.int64))
        itemsize = np.dtype(dtype).itemsize
        if count < 0 or self.offset + count * itemsize > len(self.payload):
            raise ValueError(f"{self.label}: truncated {description}")
        result = np.frombuffer(
            self.payload, dtype=dtype, count=count, offset=self.offset
        ).reshape(shape).copy()
        self.offset += count * itemsize
        return result

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise ValueError(f"{self.label}: trailing bytes")


def _read_array2(
    cursor: _BinaryCursor,
    dtype: str,
    components: tuple[int, ...],
    description: str,
) -> np.ndarray:
    rows, columns = cursor.unpack("<II", f"{description} header")
    if rows <= 0 or columns <= 0:
        raise ValueError(f"{cursor.label}: invalid {description} dimensions")
    return cursor.array((rows, columns, *components), dtype, description)


def _read_array1(cursor: _BinaryCursor, dtype: str, description: str) -> np.ndarray:
    (size,) = cursor.unpack("<I", f"{description} header")
    if size <= 0:
        raise ValueError(f"{cursor.label}: invalid {description} dimension")
    return cursor.array((size,), dtype, description)


def _artifact_payload(root: Path, manifest: dict, name: str) -> bytes:
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict or type(artifacts.get(name)) is not dict:
        raise ValueError(f"manifest does not bind {name}")
    descriptor = artifacts[name]
    if descriptor.get("path") != name:
        raise ValueError(f"manifest path for {name} is invalid")
    path = root / name
    payload = path.read_bytes()
    if descriptor.get("size_bytes") != len(payload):
        raise ValueError(f"{name} size does not match manifest")
    digest = hashlib.sha256(payload).hexdigest()
    if descriptor.get("sha256") != digest:
        raise ValueError(f"{name} SHA-256 does not match manifest")
    return payload


def load_training_bundle(data_directory: str | Path) -> TrainingBundle:
    """Load and fail-closed validate an accepted Task 1 flat data bundle."""

    root = Path(data_directory).resolve()
    manifest_payload = (root / "manifest.json").read_bytes()
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid data manifest JSON") from error
    if type(manifest) is not dict:
        raise ValueError("data manifest must be an object")
    if manifest.get("schema") != "g1-lmm-flat-data/v2":
        raise ValueError("data manifest must use continuity-safe g1-lmm-flat-data/v2")
    if manifest.get("status") != "accepted":
        raise ValueError("data manifest is not an accepted G1 LMM flat bundle")
    if manifest.get("output_fps") != 60.0:
        raise ValueError("data manifest must bind exactly 60 Hz")
    if manifest.get("trajectory_horizons") != [20, 40, 60]:
        raise ValueError("data manifest has stale trajectory horizons")
    declared = manifest.get("dimensions")
    if type(declared) is not dict:
        raise ValueError("data manifest dimensions are missing")
    dimensions = G1LmmDimensions(
        features=declared.get("features", -1),
        latent=32,
        bones=declared.get("bones", -1),
        contacts=declared.get("contacts", -1),
    )
    if dimensions != G1LmmDimensions() or manifest.get("feature_dimensions") != 31:
        raise ValueError("data manifest does not have the exact G1 LMM dimensions")

    database = _BinaryCursor(_artifact_payload(root, manifest, "database.bin"), "database.bin")
    positions = _read_array2(database, "<f4", (3,), "positions")
    velocities = _read_array2(database, "<f4", (3,), "velocities")
    rotations = _read_array2(database, "<f4", (4,), "rotations")
    angular_velocities = _read_array2(database, "<f4", (3,), "angular velocities")
    parents = _read_array1(database, "<i4", "parents")
    range_starts = _read_array1(database, "<i4", "range starts")
    range_stops = _read_array1(database, "<i4", "range stops")
    contacts = _read_array2(database, "u1", (), "contacts")
    database.finish()

    frames = len(positions)
    expected_bones = dimensions.bones
    if not (
        positions.shape == velocities.shape == angular_velocities.shape == (frames, expected_bones, 3)
        and rotations.shape == (frames, expected_bones, 4)
        and parents.shape == (expected_bones,)
        and contacts.shape == (frames, dimensions.contacts)
    ):
        raise ValueError("database arrays do not have the exact G1 dimensions")
    if manifest.get("database_frames") != frames:
        raise ValueError("database frame count does not match manifest")
    if not all(
        np.isfinite(values).all()
        for values in (positions, velocities, rotations, angular_velocities)
    ):
        raise ValueError("database contains non-finite motion values")
    quaternion_norm = np.linalg.norm(rotations.astype(np.float64), axis=2)
    if np.max(np.abs(quaternion_norm - 1.0)) > 1.0e-5:
        raise ValueError("database rotations are not normalized")
    if parents[0] != -1 or np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, expected_bones)):
        raise ValueError("database parent hierarchy is invalid")
    if np.any(contacts > 1):
        raise ValueError("database contacts must be binary")
    starts, stops = _validated_ranges(range_starts, range_stops, row_count=frames)
    manifest_ranges = manifest.get("ranges")
    if type(manifest_ranges) is not list or len(manifest_ranges) != len(starts):
        raise ValueError("manifest ranges do not match database")
    for index, (start, stop) in enumerate(zip(starts, stops)):
        entry = manifest_ranges[index]
        if type(entry) is not dict or entry.get("start") != int(start) or entry.get("stop") != int(stop):
            raise ValueError("manifest range boundaries do not match database")

    continuity = manifest.get("continuity")
    continuity_keys = {
        "schema",
        "threshold_rad_per_frame",
        "minimum_range_frames",
        "source_native_rejected_edge_count",
        "database_local_rejected_edge_count",
        "union_rejected_edge_count",
        "dropped_fragment_count",
        "dropped_frame_count",
        "published_range_count",
        "published_frame_count",
        "maximum_admitted_native_step_rad",
        "maximum_admitted_local_rotation_step_rad",
        "range_digest_sha256",
    }
    if type(continuity) is not dict or set(continuity) != continuity_keys:
        raise ValueError("v2 data manifest has no exact continuity receipt")
    if continuity["schema"] != "g1-lmm-continuity/v1":
        raise ValueError("continuity receipt schema is unsupported")
    if continuity["threshold_rad_per_frame"] != 0.25:
        raise ValueError("continuity receipt must bind the 0.25 rad/frame ceiling")
    if continuity["minimum_range_frames"] != 61:
        raise ValueError("continuity receipt must bind 61-frame minimum ranges")
    exact_counts = {
        "source_native_rejected_edge_count": 31,
        "database_local_rejected_edge_count": 32,
        "union_rejected_edge_count": 32,
        "dropped_fragment_count": 20,
        "dropped_frame_count": 233,
        "published_range_count": 13,
        "published_frame_count": 3853,
    }
    for key, expected in exact_counts.items():
        if type(continuity[key]) is not int or continuity[key] != expected:
            raise ValueError(f"continuity receipt {key} must equal {expected}")
    for key in (
        "maximum_admitted_native_step_rad",
        "maximum_admitted_local_rotation_step_rad",
    ):
        value = continuity[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0.0:
            raise ValueError(f"continuity receipt {key} is invalid")
        if value > 0.25:
            raise ValueError(f"continuity receipt {key} exceeds 0.25 rad/frame")
    if np.any(stops - starts < continuity["minimum_range_frames"]):
        raise ValueError("database contains a range shorter than continuity minimum")
    range_digest_rows = []
    for entry in manifest_ranges:
        source_first = entry.get("source_first_frame")
        source_last = entry.get("source_last_frame")
        if type(source_first) is not int or type(source_last) is not int \
                or source_first < 0 or source_last < source_first:
            raise ValueError("manifest range source provenance is invalid")
        range_digest_rows.append(
            [entry["start"], entry["stop"], source_first, source_last]
        )
    recomputed_range_digest = hashlib.sha256(
        np.asarray(range_digest_rows, dtype="<i4").tobytes(order="C")
    ).hexdigest()
    if continuity["range_digest_sha256"] != recomputed_range_digest:
        raise ValueError("continuity range digest does not match manifest ranges")
    recomputed_max = 0.0
    for start, stop in zip(starts, stops):
        left = rotations[int(start) : int(stop) - 1].astype(np.float64)
        right = rotations[int(start) + 1 : int(stop)].astype(np.float64)
        dot = np.clip(np.abs(np.sum(left * right, axis=2)), 0.0, 1.0)
        recomputed_max = max(recomputed_max, float(np.max(2.0 * np.arccos(dot))))
    if recomputed_max > 0.25 + 1.0e-7:
        raise ValueError(
            f"recomputed admitted local-rotation step {recomputed_max:.9f} exceeds 0.25 rad/frame"
        )
    if abs(
        recomputed_max - float(continuity["maximum_admitted_local_rotation_step_rad"])
    ) > 1.0e-6:
        raise ValueError("continuity receipt does not match recomputed local-rotation maximum")

    features_cursor = _BinaryCursor(_artifact_payload(root, manifest, "features.bin"), "features.bin")
    features = _read_array2(features_cursor, "<f4", (), "features")
    feature_offset = _read_array1(features_cursor, "<f4", "feature offset")
    feature_scale = _read_array1(features_cursor, "<f4", "feature scale")
    features_cursor.finish()
    if features.shape != (frames, dimensions.features):
        raise ValueError("feature rows do not align with the database")
    if feature_offset.shape != (dimensions.features,) or feature_scale.shape != (dimensions.features,):
        raise ValueError("feature normalization dimensions are invalid")
    if not all(np.isfinite(values).all() for values in (features, feature_offset, feature_scale)):
        raise ValueError("features contain non-finite values")
    if np.any(feature_scale <= 0.0):
        raise ValueError("feature scales must be strictly positive")
    if "feature_offset" in manifest and not np.array_equal(
        feature_offset, np.asarray(manifest["feature_offset"], dtype=np.float32)
    ):
        raise ValueError("feature offset does not match manifest")
    if "feature_scale" in manifest and not np.array_equal(
        feature_scale, np.asarray(manifest["feature_scale"], dtype=np.float32)
    ):
        raise ValueError("feature scale does not match manifest")

    return TrainingBundle(
        root=root,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        dimensions=dimensions,
        positions=positions,
        rotations=rotations,
        velocities=velocities,
        angular_velocities=angular_velocities,
        parents=parents,
        range_starts=range_starts,
        range_stops=range_stops,
        contacts=contacts,
        features=features,
        feature_offset=feature_offset,
        feature_scale=feature_scale,
    )


def build_training_arrays(bundle: TrainingBundle) -> TrainingArrays:
    """Construct the exact legacy compressor and decompressor row ordering."""

    positions = bundle.positions.astype(np.float32, copy=False)
    rotations = bundle.rotations.astype(np.float32, copy=False)
    velocities = bundle.velocities.astype(np.float32, copy=False)
    angular = bundle.angular_velocities.astype(np.float32, copy=False)
    rotation_xy = quat.to_xform_xy(rotations).astype(np.float32)
    global_rotation, global_position, global_velocity, global_angular = quat.fk_vel(
        rotations, positions, velocities, angular, bundle.parents
    )
    character_rotation = quat.inv_mul(global_rotation[:, 0:1], global_rotation)
    character_position = quat.inv_mul_vec(
        global_rotation[:, 0:1], global_position - global_position[:, 0:1]
    )
    character_velocity = quat.inv_mul_vec(global_rotation[:, 0:1], global_velocity)
    character_angular = quat.inv_mul_vec(global_rotation[:, 0:1], global_angular)
    character_transform = quat.to_xform(character_rotation).astype(np.float32)
    character_rotation_xy = quat.to_xform_xy(character_rotation).astype(np.float32)
    root_velocity = quat.inv_mul_vec(rotations[:, 0], velocities[:, 0]).astype(np.float32)
    root_angular = quat.inv_mul_vec(rotations[:, 0], angular[:, 0]).astype(np.float32)
    frames = bundle.frames

    compressor_input = np.concatenate(
        [
            positions[:, 1:].reshape(frames, -1),
            rotation_xy[:, 1:].reshape(frames, -1),
            velocities[:, 1:].reshape(frames, -1),
            angular[:, 1:].reshape(frames, -1),
            character_position[:, 1:].reshape(frames, -1),
            character_rotation_xy[:, 1:].reshape(frames, -1),
            character_velocity[:, 1:].reshape(frames, -1),
            character_angular[:, 1:].reshape(frames, -1),
            root_velocity,
            root_angular,
            bundle.contacts.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    decompressor_target = np.concatenate(
        [
            positions[:, 1:].reshape(frames, -1),
            rotation_xy[:, 1:].reshape(frames, -1),
            velocities[:, 1:].reshape(frames, -1),
            angular[:, 1:].reshape(frames, -1),
            root_velocity,
            root_angular,
            bundle.contacts.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if compressor_input.shape != (frames, bundle.dimensions.compressor_input):
        raise AssertionError("compressor input assembly violated the G1 dimension contract")
    if decompressor_target.shape != (frames, bundle.dimensions.decompressor_output):
        raise AssertionError("decompressor output assembly violated the G1 dimension contract")
    if not np.isfinite(compressor_input).all() or not np.isfinite(decompressor_target).all():
        raise ValueError("derived training arrays contain non-finite values")
    return TrainingArrays(
        compressor_input=np.ascontiguousarray(compressor_input),
        decompressor_target=np.ascontiguousarray(decompressor_target),
        local_positions=positions,
        local_rotation_xy=rotation_xy,
        local_velocities=velocities,
        local_angular_velocities=angular,
        character_positions=character_position.astype(np.float32),
        character_transforms=character_transform,
        character_rotation_xy=character_rotation_xy,
        character_velocities=character_velocity.astype(np.float32),
        character_angular_velocities=character_angular.astype(np.float32),
        root_velocity=root_velocity,
        root_angular_velocity=root_angular,
    )


def _validated_ranges(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    *,
    row_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(range_starts, dtype=np.int64)
    stops = np.asarray(range_stops, dtype=np.int64)
    if starts.ndim != 1 or stops.ndim != 1 or starts.shape != stops.shape:
        raise ValueError("range starts/stops must be equal-length one-dimensional arrays")
    if len(starts) == 0:
        raise ValueError("at least one range is required")
    if np.any(starts < 0) or np.any(stops <= starts):
        raise ValueError("each range must be a non-empty [start, stop) interval")
    if np.any(starts[1:] < stops[:-1]):
        raise ValueError("ranges must be ordered and non-overlapping")
    if row_count is not None and np.any(stops > row_count):
        raise ValueError("range exceeds available rows")
    return starts, stops


def range_safe_windows(
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    window: int,
    *,
    excluded_starts: np.ndarray | None = None,
    excluded_stops: np.ndarray | None = None,
    exclusion_halo: int = 0,
) -> np.ndarray:
    """Return every adjacent window wholly contained in one admitted range.

    Optional excluded intervals are expanded by ``exclusion_halo`` on both
    sides. Any window touching the expanded interval is removed, preventing a
    withheld frame from leaking through derivatives or recurrent context.
    """

    starts, stops = _validated_ranges(range_starts, range_stops)
    if not isinstance(window, (int, np.integer)) or int(window) <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(exclusion_halo, (int, np.integer)) or int(exclusion_halo) < 0:
        raise ValueError("exclusion_halo must be a non-negative integer")
    window = int(window)
    exclusion_halo = int(exclusion_halo)

    excluded: list[tuple[int, int]] = []
    if excluded_starts is not None or excluded_stops is not None:
        if excluded_starts is None or excluded_stops is None:
            raise ValueError("excluded starts and stops must be supplied together")
        exc_starts, exc_stops = _validated_ranges(excluded_starts, excluded_stops)
        excluded = [
            (max(0, int(start) - exclusion_halo), int(stop) + exclusion_halo)
            for start, stop in zip(exc_starts, exc_stops)
        ]

    windows: list[np.ndarray] = []
    offsets = np.arange(window, dtype=np.int64)
    for start, stop in zip(starts, stops):
        for first in range(int(start), int(stop) - window + 1):
            last_exclusive = first + window
            if any(first < exc_stop and exc_start < last_exclusive for exc_start, exc_stop in excluded):
                continue
            windows.append(first + offsets)
    if not windows:
        return np.empty((0, window), dtype=np.int64)
    return np.stack(windows).astype(np.int64, copy=False)


def range_safe_deltas(
    values: np.ndarray,
    range_starts: np.ndarray,
    range_stops: np.ndarray,
    *,
    dt: float,
) -> np.ndarray:
    """Differentiate adjacent values separately within every admitted range."""

    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim < 1:
        raise ValueError("values must have a row dimension")
    starts, stops = _validated_ranges(range_starts, range_stops, row_count=len(rows))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(rows).all():
        raise ValueError("values contain non-finite data")

    inverse_dt = np.float32(1.0 / dt)
    pieces = [
        (rows[int(start) + 1 : int(stop)] - rows[int(start) : int(stop) - 1]) * inverse_dt
        for start, stop in zip(starts, stops)
        if stop - start >= 2
    ]
    if not pieces:
        return np.empty((0, *rows.shape[1:]), dtype=np.float32)
    return np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
