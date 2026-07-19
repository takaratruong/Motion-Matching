from dataclasses import dataclass
import hashlib
import json
import numpy as np


TERRAIN_FAMILIES = ("flat", "curb", "slope", "stair")


def _is_integer(value) -> bool:
    return type(value) is int or (
        isinstance(value, np.integer) and not isinstance(value, np.bool_)
    )


@dataclass(frozen=True)
class SourceFrameRange:
    source_name: str
    source_start: int
    source_stop: int
    source_frame_count: int
    global_start: int
    global_stop: int

    def validate(self, frame_count: int) -> None:
        if type(self.source_name) is not str or not self.source_name:
            raise ValueError("source name must be a non-empty string")
        integer_fields = (
            self.source_start,
            self.source_stop,
            self.source_frame_count,
            self.global_start,
            self.global_stop,
        )
        if not all(_is_integer(value) for value in integer_fields):
            raise ValueError("source/global range values must be integers")
        if self.source_start >= self.source_stop:
            raise ValueError("empty or reversed source range")
        if (
            self.source_frame_count < 1
            or self.source_start < 0
            or self.source_stop > self.source_frame_count
        ):
            raise ValueError("source range out of bounds")
        if self.global_start >= self.global_stop:
            raise ValueError("empty or reversed global range")
        if self.global_start < 0 or self.global_stop > frame_count:
            raise ValueError("global range out of bounds")
        if (
            self.source_stop - self.source_start
            != self.global_stop - self.global_start
        ):
            raise ValueError("source/global range lengths differ")


@dataclass(frozen=True)
class TerrainBank:
    family: str
    range_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "range_indices", tuple(self.range_indices))


@dataclass(frozen=True)
class TerrainBankIndex:
    frame_count: int
    ranges: tuple[SourceFrameRange, ...]
    banks: tuple[TerrainBank, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranges", tuple(self.ranges))
        object.__setattr__(self, "banks", tuple(self.banks))

    def validate(self) -> None:
        if not _is_integer(self.frame_count) or self.frame_count < 1:
            raise ValueError("terrain bank frame count must be a positive integer")
        if not self.ranges:
            raise ValueError("terrain bank ranges must not be empty")
        if len(self.banks) != len(TERRAIN_FAMILIES) or not all(
            isinstance(bank, TerrainBank) for bank in self.banks
        ):
            raise ValueError("terrain banks must contain exact terrain families")
        if tuple(bank.family for bank in self.banks) != TERRAIN_FAMILIES:
            raise ValueError(
                f"terrain families must be exactly {TERRAIN_FAMILIES}"
            )

        source_names = set()
        expected_start = 0
        for source_range in self.ranges:
            if not isinstance(source_range, SourceFrameRange):
                raise ValueError("terrain bank ranges must be SourceFrameRange records")
            source_range.validate(self.frame_count)
            if source_range.source_name in source_names:
                raise ValueError(
                    f"duplicate source ownership for {source_range.source_name}"
                )
            source_names.add(source_range.source_name)
            if source_range.global_start < expected_start:
                raise ValueError("global-frame ranges overlap")
            if source_range.global_start > expected_start:
                raise ValueError("gap in global-frame range coverage")
            expected_start = source_range.global_stop
        if expected_start != self.frame_count:
            raise ValueError("incomplete global-frame coverage")

        ownership = [0] * len(self.ranges)
        for bank in self.banks:
            for range_index in bank.range_indices:
                if not _is_integer(range_index) or not (
                    0 <= range_index < len(self.ranges)
                ):
                    raise ValueError("terrain bank range index out of bounds")
                ownership[int(range_index)] += 1
        if any(count > 1 for count in ownership):
            raise ValueError("duplicate terrain-bank range ownership")
        if any(count == 0 for count in ownership):
            raise ValueError("incomplete terrain-bank range ownership")

    def ranges_for(self, family: str) -> tuple[SourceFrameRange, ...]:
        self.validate()
        if family not in TERRAIN_FAMILIES:
            raise ValueError(f"unknown terrain family {family!r}")
        bank = self.banks[TERRAIN_FAMILIES.index(family)]
        return tuple(self.ranges[index] for index in bank.range_indices)


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
            np.zeros((frames, 12), np.float32),
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
        if self.terrain_features.shape != (frames, 12):
            raise ValueError("terrain feature shape must be (T, 12)")
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
        if self.terrain_features.shape != (frames, 12):
            raise ValueError("terrain feature shape must be (N, 12)")
        if self.terrain_support.shape != (frames, 3):
            raise ValueError("terrain support shape must be (N, 3)")
        if not np.isfinite(self.terrain_support).all():
            raise ValueError("terrain support must contain only finite values")
