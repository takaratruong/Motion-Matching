from dataclasses import dataclass
import hashlib
import json
import numpy as np


G1_SKELETON_NAMES = (
    "Simulation", "Hips",
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee",
    "LeftAnkle", "LeftToe",
    "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
    "RightAnkle", "RightToe",
    "Spine", "Spine1", "Spine2",
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
    "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
    "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist",
)
G1_SKELETON_PARENTS = (
    -1, 0,
    1, 2, 3, 4, 5, 6,
    1, 8, 9, 10, 11, 12,
    1, 14, 15,
    16, 17, 18, 19, 20, 21, 22,
    16, 24, 25, 26, 27, 28, 29,
)
G1_SKELETON_SIGNATURE = (
    "6138d9364b6f4178c25e2c1ac7039f3ce5fedf6b11a0b8375dea712633abd2e7"
)


@dataclass
class SourceClip:
    name: str
    fps: float
    qpos: np.ndarray
    source_frames: np.ndarray
    terrain_id: str
    provenance: dict | None = None

    def validate(self) -> None:
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 36:
            raise ValueError(f"qpos shape must be (T, 36), got {self.qpos.shape}")
        if self.fps <= 0 or not np.isfinite(self.fps):
            raise ValueError(f"invalid fps {self.fps}")
        if self.source_frames.shape != (len(self.qpos),):
            raise ValueError("source frame shape does not match qpos")
        if not np.isfinite(self.qpos).all():
            raise ValueError("qpos contains non-finite values")


@dataclass(frozen=True)
class SkeletonSpec:
    names: tuple[str, ...]
    parents: np.ndarray

    def signature(self) -> str:
        payload = json.dumps(
            {"names": self.names, "parents": self.parents.tolist()},
            separators=(",", ":"), sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def require_canonical_g1_skeleton(
    skeleton: SkeletonSpec, label: str = "G1 skeleton",
) -> None:
    parents = np.asarray(skeleton.parents)
    if tuple(skeleton.names) != G1_SKELETON_NAMES \
            or parents.shape != (len(G1_SKELETON_PARENTS),) \
            or tuple(parents.tolist()) != G1_SKELETON_PARENTS \
            or skeleton.signature() != G1_SKELETON_SIGNATURE:
        raise ValueError(
            f"{label} does not match the canonical G1 skeleton")


@dataclass
class HoldenClip:
    name: str
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    contacts: np.ndarray
    terrain_features: np.ndarray
    terrain_support: np.ndarray
    source_frames: np.ndarray
    terrain_id: str
    source_left_indices: np.ndarray
    source_right_indices: np.ndarray
    source_alpha: np.ndarray

    @classmethod
    def empty(cls, frames: int, bones: int) -> "HoldenClip":
        return cls(
            "empty",
            np.zeros((frames, bones, 3), np.float32),
            np.zeros((frames, bones, 3), np.float32),
            np.tile(np.array([1, 0, 0, 0], np.float32), (frames, bones, 1)),
            np.zeros((frames, bones, 3), np.float32),
            np.zeros((frames, 2), np.uint8),
            np.zeros((frames, 4), np.float32),
            np.zeros((frames, 3), np.float32),
            np.arange(frames),
            "flat",
            np.arange(frames, dtype=np.int32),
            np.arange(frames, dtype=np.int32),
            np.zeros(frames, np.float32),
        )

    def validate(self) -> None:
        frames, bones, xyz = self.positions.shape
        if xyz != 3 or self.velocities.shape != (frames, bones, 3):
            raise ValueError("position/velocity shape mismatch")
        if self.rotations.shape != (frames, bones, 4):
            raise ValueError("rotation shape mismatch")
        if self.angular_velocities.shape != (frames, bones, 3):
            raise ValueError("angular velocity shape mismatch")
        if self.contacts.shape != (frames, 2):
            raise ValueError("contact shape mismatch")
        if self.terrain_features.shape != (frames, 4):
            raise ValueError("terrain feature shape must be (T, 4)")
        if self.terrain_support.shape != (frames, 3):
            raise ValueError("terrain support shape must be (T, 3)")
        for name, values, dtype in (
            ("source left", self.source_left_indices, np.int32),
            ("source right", self.source_right_indices, np.int32),
            ("source alpha", self.source_alpha, np.float32),
        ):
            if values.shape != (frames,) or values.dtype != np.dtype(dtype):
                raise ValueError(
                    f"{name} provenance must have shape (T,) and dtype "
                    f"{np.dtype(dtype)}")
        if np.any(self.source_left_indices > self.source_right_indices) \
                or np.any(self.source_alpha < 0.0) \
                or np.any(self.source_alpha > 1.0):
            raise ValueError("source interpolation provenance is invalid")
        arrays = (
            self.positions, self.velocities, self.rotations,
            self.angular_velocities, self.terrain_features,
            self.terrain_support,
        )
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("converted clip contains non-finite values")


@dataclass(frozen=True)
class FeatureSet:
    values: np.ndarray
    offset: np.ndarray
    scale: np.ndarray

    def validate(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != 31 \
                or self.values.shape[0] < 1:
            raise ValueError("matching features must have shape (T, 31)")
        if self.offset.shape != (31,) or self.scale.shape != (31,):
            raise ValueError("feature offset and scale must have shape (31,)")
        for name, values in (
            ("features", self.values), ("offset", self.offset),
            ("scale", self.scale),
        ):
            if values.dtype != np.dtype(np.float32):
                raise ValueError(f"{name} must use float32")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values")
        if np.any(self.scale <= 0.0):
            raise ValueError("feature scale must be strictly positive")


@dataclass
class ArtifactSet:
    positions: np.ndarray
    velocities: np.ndarray
    rotations: np.ndarray
    angular_velocities: np.ndarray
    parents: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    contacts: np.ndarray
    terrain_features: np.ndarray
    terrain_support: np.ndarray

    @classmethod
    def empty(cls, frames: int, bones: int) -> "ArtifactSet":
        clip = HoldenClip.empty(frames, bones)
        return cls(
            clip.positions, clip.velocities, clip.rotations,
            clip.angular_velocities, np.arange(-1, bones - 1, dtype=np.int32),
            np.array([0], np.int32), np.array([frames], np.int32),
            clip.contacts, clip.terrain_features, clip.terrain_support,
        )

    def validate(self) -> None:
        frames = len(self.positions)
        if self.range_starts.shape != self.range_stops.shape:
            raise ValueError("range arrays differ")
        if len(self.range_starts) == 0:
            raise ValueError("no animation ranges")
        if self.range_starts[0] != 0 or self.range_stops[-1] != frames:
            raise ValueError("ranges do not cover all frames")
        if np.any(self.range_starts[1:] < self.range_stops[:-1]):
            raise ValueError("ranges overlap")
        if np.any(self.range_starts >= self.range_stops):
            raise ValueError("empty animation range")
        if self.terrain_features.shape != (frames, 4):
            raise ValueError("terrain feature shape must be (N, 4)")
        if self.terrain_support.shape != (frames, 3):
            raise ValueError("terrain support shape must be (N, 3)")
        if not np.isfinite(self.terrain_support).all():
            raise ValueError("terrain support must contain only finite values")
