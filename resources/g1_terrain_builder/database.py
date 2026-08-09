from dataclasses import dataclass
from numbers import Integral
import os
import struct
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, "/home/ubuntu/projects/motion-matching/resources")
import quat as holden_quat

from .schema import ArtifactSet, HoldenClip, SkeletonSpec


LMM_CONTACT_SPEED_THRESHOLD = 0.15
LMM_CONTACT_MEDIAN_FILTER_FRAMES = 6
LMM_CONTACT_FILTER_MODE = "nearest"


@dataclass(frozen=True)
class ContactConfig:
    speed_threshold: float = 0.15
    height_threshold: float = 0.06
    median_filter_frames: int = 3

    def __post_init__(self) -> None:
        try:
            speed_is_valid = (
                np.ndim(self.speed_threshold) == 0
                and np.isfinite(self.speed_threshold)
                and self.speed_threshold > 0.0
            )
        except TypeError:
            speed_is_valid = False
        if not speed_is_valid:
            raise ValueError("contact speed threshold must be positive and finite")

        try:
            height_is_valid = (
                np.ndim(self.height_threshold) == 0
                and np.isfinite(self.height_threshold)
                and self.height_threshold > 0.0
            )
        except TypeError:
            height_is_valid = False
        if not height_is_valid:
            raise ValueError("contact height threshold must be positive and finite")

        size = self.median_filter_frames
        if (
            not isinstance(size, Integral)
            or isinstance(size, (bool, np.bool_))
            or size < 1
            or size % 2 == 0
        ):
            raise ValueError("contact median filter size must be a positive odd integer")


def _validate_parent_hierarchy(parents: np.ndarray, bones: int) -> None:
    parents = np.asarray(parents)
    if parents.shape != (bones,):
        raise ValueError(
            f"parents shape must be ({bones},), got {parents.shape}")
    if not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("parents must contain integer indices")
    if bones == 0 or int(parents[0]) != -1:
        raise ValueError("parents must begin with one root index of -1")
    for bone in range(1, bones):
        parent = int(parents[bone])
        if parent < 0 or parent >= bone:
            raise ValueError(
                "parents must be topologically ordered with exactly one root; "
                f"bone {bone} has parent {parent}")


def _validate_skeleton(skeleton: SkeletonSpec) -> None:
    if not isinstance(skeleton, SkeletonSpec):
        raise TypeError("skeleton must be a SkeletonSpec")
    if len(skeleton.names) != len(skeleton.parents):
        raise ValueError(
            "skeleton names and parents must have the same bone count")
    if len(set(skeleton.names)) != len(skeleton.names):
        raise ValueError("skeleton names must be unique")
    _validate_parent_hierarchy(skeleton.parents, len(skeleton.names))


def _validate_binary_values(name: str, array: np.ndarray) -> None:
    array = np.asarray(array)
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(f"{name} must contain binary integer values")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} values must be 0 or 1")


def _validate_clip(clip: HoldenClip, bones: int) -> None:
    if not isinstance(clip, HoldenClip):
        raise TypeError("clips must contain HoldenClip instances")
    clip.validate()
    frames = len(clip.positions)
    if frames == 0:
        raise ValueError(f"{clip.name}: clip must contain at least one frame")
    if clip.positions.shape[1] != bones:
        raise ValueError(
            f"{clip.name}: bone count {clip.positions.shape[1]} "
            f"does not match skeleton bone count {bones}")
    if np.asarray(clip.source_frames).shape != (frames,):
        raise ValueError(
            f"{clip.name}: source frame shape must be ({frames},)")
    _validate_binary_values(f"{clip.name}: contacts", clip.contacts)


def _validate_artifacts(artifacts: ArtifactSet) -> None:
    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("artifacts must be an ArtifactSet")

    positions = np.asarray(artifacts.positions)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions shape must be (frames, bones, 3), got {positions.shape}")
    frames, bones, _ = positions.shape
    if frames == 0 or bones == 0:
        raise ValueError("positions must contain at least one frame and one bone")

    expected = {
        "velocities": (frames, bones, 3),
        "rotations": (frames, bones, 4),
        "angular_velocities": (frames, bones, 3),
        "contacts": (frames, 2),
        "terrain_features": (frames, 4),
        "terrain_support": (frames, 3),
    }
    for name, shape in expected.items():
        actual = np.asarray(getattr(artifacts, name)).shape
        if actual != shape:
            raise ValueError(f"{name} shape must be {shape}, got {actual}")

    for name in (
        "positions", "velocities", "rotations", "angular_velocities",
        "terrain_features", "terrain_support",
    ):
        array = np.asarray(getattr(artifacts, name))
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must be numeric")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")

    _validate_parent_hierarchy(artifacts.parents, bones)
    _validate_binary_values("contacts", artifacts.contacts)

    starts = np.asarray(artifacts.range_starts)
    stops = np.asarray(artifacts.range_stops)
    if starts.ndim != 1 or stops.ndim != 1 or starts.shape != stops.shape:
        raise ValueError("range starts and stops must be aligned one-dimensional arrays")
    if not (
        np.issubdtype(starts.dtype, np.integer)
        and np.issubdtype(stops.dtype, np.integer)
    ):
        raise ValueError("range starts and stops must contain integer indices")
    if len(starts) == 0:
        raise ValueError("at least one animation range is required")
    if int(starts[0]) != 0 or int(stops[-1]) != frames:
        raise ValueError("animation ranges must cover all frames")
    if np.any(starts >= stops):
        raise ValueError("animation ranges must be non-empty")
    if len(starts) > 1 and np.any(starts[1:] != stops[:-1]):
        raise ValueError("animation ranges must be contiguous and non-overlapping")

    artifacts.validate()


def combine_clips(
    clips: list[HoldenClip], skeleton: SkeletonSpec,
) -> ArtifactSet:
    if not clips:
        raise ValueError("at least one clip is required")
    _validate_skeleton(skeleton)
    bones = len(skeleton.parents)
    for clip in clips:
        _validate_clip(clip, bones)

    lengths64 = np.asarray([len(clip.positions) for clip in clips], np.int64)
    total_frames = int(lengths64.sum())
    if total_frames > np.iinfo(np.int32).max:
        raise ValueError("combined clip frame count exceeds Holden int32 range")
    stops = np.cumsum(lengths64, dtype=np.int64).astype(np.int32)
    starts = np.concatenate((np.array([0], np.int32), stops[:-1]))
    artifacts = ArtifactSet(
        np.concatenate([clip.positions for clip in clips]).astype(
            np.float32, copy=False),
        np.concatenate([clip.velocities for clip in clips]).astype(
            np.float32, copy=False),
        np.concatenate([clip.rotations for clip in clips]).astype(
            np.float32, copy=False),
        np.concatenate([clip.angular_velocities for clip in clips]).astype(
            np.float32, copy=False),
        np.asarray(skeleton.parents, np.int32).copy(),
        starts,
        stops,
        np.concatenate([clip.contacts for clip in clips]).astype(
            np.uint8, copy=False),
        np.concatenate([clip.terrain_features for clip in clips]).astype(
            np.float32, copy=False),
        np.concatenate([clip.terrain_support for clip in clips]).astype(
            np.float32, copy=False),
    )
    _validate_artifacts(artifacts)
    return artifacts


def _write_array2(stream, array: np.ndarray, dtype: str) -> None:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    if values.ndim < 2:
        raise ValueError("array2 values must have at least two dimensions")
    rows, columns = values.shape[:2]
    if rows > 0xFFFFFFFF or columns > 0xFFFFFFFF:
        raise ValueError("array2 dimensions exceed the Holden uint32 header")
    stream.write(struct.pack("<II", rows, columns))
    stream.write(values.tobytes())


def _write_array1(stream, array: np.ndarray, dtype: str) -> None:
    values = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    if values.ndim != 1:
        raise ValueError("array1 values must be one-dimensional")
    if len(values) > 0xFFFFFFFF:
        raise ValueError("array1 length exceeds the Holden uint32 header")
    stream.write(struct.pack("<I", len(values)))
    stream.write(values.tobytes())


def write_holden_database(path: os.PathLike | str, artifacts: ArtifactSet) -> None:
    _validate_artifacts(artifacts)
    with open(path, "wb") as stream:
        _write_array2(stream, artifacts.positions, "<f4")
        _write_array2(stream, artifacts.velocities, "<f4")
        _write_array2(stream, artifacts.rotations, "<f4")
        _write_array2(stream, artifacts.angular_velocities, "<f4")
        _write_array1(stream, artifacts.parents, "<i4")
        _write_array1(stream, artifacts.range_starts, "<i4")
        _write_array1(stream, artifacts.range_stops, "<i4")
        _write_array2(stream, artifacts.contacts, "u1")


def _read_exact(stream, size: int, description: str) -> bytes:
    remaining = os.fstat(stream.fileno()).st_size - stream.tell()
    if size > remaining:
        raise ValueError(f"truncated {description}")
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"truncated {description}")
    return data


def _read_array1(stream, dtype: str) -> np.ndarray:
    header = _read_exact(stream, 4, "array1 header")
    rows, = struct.unpack("<I", header)
    itemsize = np.dtype(dtype).itemsize
    raw = _read_exact(stream, rows * itemsize, "array payload")
    return np.frombuffer(raw, dtype=dtype, count=rows).copy()


def _read_array2(
    stream, dtype: str, components: tuple[int, ...] = (),
) -> np.ndarray:
    header = _read_exact(stream, 8, "array2 header")
    rows, columns = struct.unpack("<II", header)
    component_count = 1
    for component in components:
        component_count *= component
    count = rows * columns * component_count
    raw = _read_exact(
        stream, count * np.dtype(dtype).itemsize, "array payload")
    return np.frombuffer(raw, dtype=dtype, count=count).reshape(
        (rows, columns) + components).copy()


def read_holden_database(path: os.PathLike | str) -> ArtifactSet:
    with open(path, "rb") as stream:
        positions = _read_array2(stream, "<f4", (3,))
        velocities = _read_array2(stream, "<f4", (3,))
        rotations = _read_array2(stream, "<f4", (4,))
        angular_velocities = _read_array2(stream, "<f4", (3,))
        parents = _read_array1(stream, "<i4")
        range_starts = _read_array1(stream, "<i4")
        range_stops = _read_array1(stream, "<i4")
        contacts = _read_array2(stream, "u1")
        if stream.read(1):
            raise ValueError("trailing database bytes")

    artifacts = ArtifactSet(
        positions,
        velocities,
        rotations,
        angular_velocities,
        parents,
        range_starts,
        range_stops,
        contacts,
        np.zeros((len(positions), 4), np.float32),
        np.zeros((len(positions), 3), np.float32),
    )
    _validate_artifacts(artifacts)
    return artifacts


def _validated_motion_arrays(
    positions: np.ndarray, rotations: np.ndarray, fps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    positions = np.asarray(positions, np.float64)
    rotations = np.asarray(rotations, np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"position shape must be (frames, bones, 3), got {positions.shape}")
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        raise ValueError(
            f"rotation shape must be (frames, bones, 4), got {rotations.shape}")
    if positions.shape[:2] != rotations.shape[:2]:
        raise ValueError("position and rotation shapes must align")
    if len(positions) < 3:
        raise ValueError("derivatives require at least three aligned bone frames")
    fps = _validated_fps(fps, "derivative")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise ValueError("derivative inputs must contain only finite values")
    norms = np.linalg.norm(rotations, axis=-1)
    if np.any(norms < 1e-8) or np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError("rotation inputs must contain normalized quaternions")
    return positions, rotations, fps


def _validated_fps(fps: float, context: str) -> float:
    try:
        value = float(fps)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} fps must be positive and finite") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context} fps must be positive and finite")
    return value


def derive_velocities(
    positions: np.ndarray, rotations: np.ndarray, fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions, rotations, fps = _validated_motion_arrays(
        positions, rotations, fps)
    velocity = np.gradient(positions, axis=0, edge_order=2) * fps
    frame_delta = holden_quat.to_scaled_angle_axis(holden_quat.abs(
        holden_quat.mul_inv(rotations[1:], rotations[:-1]))) * fps
    angular_velocity = np.empty(positions.shape, np.float64)
    angular_velocity[0] = frame_delta[0]
    angular_velocity[-1] = frame_delta[-1]
    angular_velocity[1:-1] = 0.5 * (
        frame_delta[:-1] + frame_delta[1:])
    return (
        velocity.astype(np.float32),
        angular_velocity.astype(np.float32),
    )


def sample_terrain_support(
    global_positions: np.ndarray,
    terrain,
    root: int,
    left_toe: int,
    right_toe: int,
) -> np.ndarray:
    positions = np.asarray(global_positions, np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3 or not len(positions):
        raise ValueError("support positions must have shape (frames, bones, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("support positions must be finite")
    indices = (root, left_toe, right_toe)
    if any(
        not isinstance(index, Integral)
        or isinstance(index, (bool, np.bool_)) or index < 0
        or index >= positions.shape[1] for index in indices
    ) or len(set(indices)) != 3:
        raise ValueError("support bone indices must be distinct and in range")
    if not callable(getattr(terrain, "height", None)):
        raise TypeError("support terrain must provide height(x, z)")
    support = np.empty((len(positions), 3), np.float32)
    for frame in range(len(positions)):
        for column, bone in enumerate(indices):
            value = float(terrain.height(
                float(positions[frame, bone, 0]),
                float(positions[frame, bone, 2])))
            if not np.isfinite(value):
                raise ValueError("support terrain heights must be finite")
            with np.errstate(over="ignore", invalid="ignore"):
                encoded = np.float32(value)
            if not np.isfinite(encoded):
                raise ValueError("support terrain heights must be finite float32")
            support[frame, column] = encoded
    return support


def derive_contacts(
    global_positions: np.ndarray,
    terrain,
    left: int,
    right: int,
    fps: float,
    config: ContactConfig = ContactConfig(),
) -> np.ndarray:
    positions = np.asarray(global_positions, np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "contact positions must have shape (frames, bones, 3)")
    if len(positions) < 3:
        raise ValueError("contacts require at least three global-position frames")
    if not np.all(np.isfinite(positions)):
        raise ValueError("contact positions must contain only finite values")
    fps = _validated_fps(fps, "contact")
    if not isinstance(config, ContactConfig):
        raise TypeError("config must be a ContactConfig")
    if not isinstance(left, Integral) or not isinstance(right, Integral):
        raise ValueError("contact foot indices must be integers")
    bones = positions.shape[1]
    if left < 0 or right < 0 or left >= bones or right >= bones:
        raise ValueError("contact foot indices are out of range")
    if left == right:
        raise ValueError("left and right contact foot indices must be distinct")
    if not callable(getattr(terrain, "height", None)):
        raise TypeError("terrain must provide a height(x, z) method")

    feet = positions[:, [left, right]]
    velocity = np.gradient(feet, axis=0, edge_order=2) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    ground = np.empty((len(feet), 2), np.float64)
    for frame in range(len(feet)):
        for side in range(2):
            try:
                height = terrain.height(
                    float(feet[frame, side, 0]),
                    float(feet[frame, side, 2]),
                )
                ground[frame, side] = float(height)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "terrain height queries must return finite scalar values") \
                    from error
    if not np.all(np.isfinite(ground)):
        raise ValueError("terrain height queries must return finite values")

    relative_height = feet[:, :, 1] - ground
    contacts = (
        (speed < float(config.speed_threshold))
        & (np.abs(relative_height) < float(config.height_threshold))
    )
    for side in range(2):
        contacts[:, side] = ndimage.median_filter(
            contacts[:, side],
            size=int(config.median_filter_frames),
            mode="nearest",
        )
    return contacts.astype(np.uint8)


def derive_lmm_contacts(
    local_positions: np.ndarray,
    local_rotations: np.ndarray,
    parents: np.ndarray,
    left: int,
    right: int,
    fps: float,
) -> np.ndarray:
    """Reproduce the bundled Orange Duck 60 Hz contact-label rule."""

    positions, rotations, fps = _validated_motion_arrays(
        local_positions, local_rotations, fps)
    if fps != 60.0:
        raise ValueError("LMM contacts require exact 60 Hz motion")
    frames, bones = positions.shape[:2]
    if frames < 4:
        raise ValueError("LMM contacts require at least four motion frames")
    _validate_parent_hierarchy(parents, bones)
    if any(
        not isinstance(index, Integral)
        or isinstance(index, (bool, np.bool_))
        or index < 0 or index >= bones
        for index in (left, right)
    ):
        raise ValueError("LMM contact toe indices are out of range")
    if left == right:
        raise ValueError("LMM contact toe indices must be distinct")

    velocities = np.empty_like(positions)
    velocities[1:-1] = (
        0.5 * (positions[2:] - positions[1:-1]) * fps
        + 0.5 * (positions[1:-1] - positions[:-2]) * fps
    )
    velocities[0] = velocities[1] - (velocities[3] - velocities[2])
    velocities[-1] = velocities[-2] + (
        velocities[-2] - velocities[-3])

    angular_velocities = np.zeros_like(positions)
    with np.errstate(divide="ignore", invalid="ignore"):
        angular_velocities[1:-1] = (
            0.5 * holden_quat.to_scaled_angle_axis(holden_quat.abs(
                holden_quat.mul_inv(
                    rotations[2:], rotations[1:-1]))) * fps
            + 0.5 * holden_quat.to_scaled_angle_axis(holden_quat.abs(
                holden_quat.mul_inv(
                    rotations[1:-1], rotations[:-2]))) * fps
        )
    angular_velocities[0] = angular_velocities[1] - (
        angular_velocities[3] - angular_velocities[2])
    angular_velocities[-1] = angular_velocities[-2] + (
        angular_velocities[-2] - angular_velocities[-3])
    _global_rotations, _global_positions, global_velocities, \
        _global_angular_velocities = holden_quat.fk_vel(
            rotations, positions, velocities, angular_velocities, parents)
    if not np.isfinite(global_velocities).all():
        raise ValueError("LMM global velocities must be finite")

    toe_velocities = global_velocities[:, np.asarray([left, right])]
    speed = np.sqrt(np.sum(toe_velocities**2, axis=-1))
    contacts = speed < LMM_CONTACT_SPEED_THRESHOLD
    for side in range(2):
        contacts[:, side] = ndimage.median_filter(
            contacts[:, side],
            size=LMM_CONTACT_MEDIAN_FILTER_FRAMES,
            mode=LMM_CONTACT_FILTER_MODE,
        )
    return contacts.astype(np.uint8)


def forward_kinematics_arrays(
    positions: np.ndarray, rotations: np.ndarray, parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions)
    rotations = np.asarray(rotations)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions shape must be (frames, bones, 3), got {positions.shape}")
    if rotations.shape != positions.shape[:2] + (4,):
        raise ValueError(
            "rotations shape must align with positions as (frames, bones, 4)")
    if len(positions) == 0:
        raise ValueError("forward kinematics requires at least one frame")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise ValueError("forward kinematics inputs must contain only finite values")
    rotation_norms = np.linalg.norm(rotations, axis=-1)
    if (
        np.any(rotation_norms < 1e-8)
        or np.max(np.abs(rotation_norms - 1.0)) > 1e-3
    ):
        raise ValueError(
            "forward kinematics rotations must be normalized quaternions")
    _validate_parent_hierarchy(parents, positions.shape[1])

    global_positions = np.empty_like(positions)
    global_rotations = np.empty_like(rotations)
    for bone, parent in enumerate(np.asarray(parents)):
        if parent < 0:
            global_positions[:, bone] = positions[:, bone]
            global_rotations[:, bone] = rotations[:, bone]
        else:
            global_rotations[:, bone] = holden_quat.mul(
                global_rotations[:, parent], rotations[:, bone])
            global_positions[:, bone] = (
                global_positions[:, parent]
                + holden_quat.mul_vec(
                    global_rotations[:, parent], positions[:, bone])
            )
    return global_positions, global_rotations
