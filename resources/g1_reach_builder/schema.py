from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GMRSource:
    sequence_id: str
    archive_member: str
    archive_path: Path
    archive_sha256: str
    fps: float
    fps_overridden: bool
    qpos: np.ndarray
    source_frames: np.ndarray
    disposition: str = "included"

    def validate(self) -> None:
        if not self.sequence_id:
            raise ValueError("GMR source has empty sequence id")
        if self.disposition not in {"included", "excluded"}:
            raise ValueError(
                f"{self.sequence_id}: invalid disposition {self.disposition}"
            )
        qpos = np.asarray(self.qpos)
        if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) == 0:
            raise ValueError(
                f"{self.sequence_id}: qpos shape must be (T, 36), got {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError(f"{self.sequence_id}: qpos contains non-finite values")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"{self.sequence_id}: invalid fps {self.fps}")
        frames = np.asarray(self.source_frames)
        if frames.shape != (len(qpos),):
            raise ValueError(
                f"{self.sequence_id}: source frame shape {frames.shape}"
            )
        if not np.array_equal(frames, np.arange(len(qpos))):
            raise ValueError(
                f"{self.sequence_id}: source frames must be contiguous from zero"
            )


@dataclass
class ReviewCorpus:
    fps: float
    positions: np.ndarray
    rotations: np.ndarray
    range_starts: np.ndarray
    range_stops: np.ndarray
    source_frames: np.ndarray
    sequence_ids: tuple[str, ...]
    archive_members: tuple[str, ...]
    archive_paths: tuple[str, ...]
    archive_sha256: tuple[str, ...]
    source_fps: np.ndarray
    fps_overridden: tuple[bool, ...]
    conversion_reports: tuple[dict[str, float], ...]
    skeleton_signature: str
    excluded_sequence_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.fps != 25.0:
            raise ValueError(f"review corpus requires 25 Hz, got {self.fps}")
        positions = np.asarray(self.positions)
        rotations = np.asarray(self.rotations)
        if positions.ndim != 3 or positions.shape[1:] != (31, 3):
            raise ValueError(
                f"review position shape must be (T, 31, 3), got {positions.shape}"
            )
        if rotations.shape != (len(positions), 31, 4):
            raise ValueError(
                "review rotation shape must match positions and 31 bones"
            )
        if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
            raise ValueError("review transforms contain non-finite values")
        norm_error = np.max(np.abs(np.linalg.norm(rotations, axis=-1) - 1.0))
        if norm_error > 1e-4:
            raise ValueError(f"review quaternion norm error {norm_error}")
        starts = np.asarray(self.range_starts)
        stops = np.asarray(self.range_stops)
        clips = len(self.sequence_ids)
        if starts.shape != (clips,) or stops.shape != (clips,):
            raise ValueError("review range count does not match source count")
        if clips == 0:
            raise ValueError("review corpus has no included sources")
        if starts[0] != 0 or stops[-1] != len(positions):
            raise ValueError("review ranges do not cover all frames")
        if np.any(starts >= stops) or np.any(starts[1:] != stops[:-1]):
            raise ValueError("review ranges must be nonempty and contiguous")
        if np.asarray(self.source_frames).shape != (len(positions),):
            raise ValueError("review source frame count mismatch")
        metadata = (
            self.archive_members,
            self.archive_paths,
            self.archive_sha256,
            self.fps_overridden,
            self.conversion_reports,
        )
        if any(len(values) != clips for values in metadata):
            raise ValueError("review source metadata count mismatch")
        if np.asarray(self.source_fps).shape != (clips,):
            raise ValueError("review source fps count mismatch")

