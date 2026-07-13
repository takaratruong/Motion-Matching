from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass
class SourceClip:
    name: str
    fps: float
    qpos: np.ndarray
    source_frames: np.ndarray
    terrain_id: str

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
        arrays = (
            self.positions, self.velocities, self.rotations,
            self.angular_velocities, self.terrain_features,
            self.terrain_support,
        )
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("converted clip contains non-finite values")


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
