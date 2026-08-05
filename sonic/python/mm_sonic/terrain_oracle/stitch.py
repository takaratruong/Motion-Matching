"""Practical cross-clip reconstruction for canonical terrain-oracle fragments.

Selections are half-open source-frame ranges.  Each new fragment is placed by
matching its entry root position and yaw to the previous output tail; that
entry sample is then omitted, so source time continues at the following 50 Hz
sample.  Joint and root-tilt entry offsets decay over the new fragment to
avoid a visible pose pop without changing its root translation or yaw path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .math3d import (
    quaternion_inverse_wxyz,
    quaternion_multiply_wxyz,
    slerp_wxyz,
)


@dataclass(frozen=True)
class FragmentSelection:
    """One half-open range in a clip selected by its manifest-array index."""

    archive_clip_index: int
    source_start_frame: int
    source_stop_frame: int


@dataclass(frozen=True)
class FrameProvenance:
    """The original corpus clip and frame that supplied one output sample."""

    archive_clip_index: int
    source_frame: int
    clip_id: str


@dataclass(frozen=True)
class StitchedMotion:
    """A 50 Hz stitched pose trace plus source provenance and seam locations."""

    fps: float
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray
    provenance: tuple[FrameProvenance, ...]
    seam_indices: tuple[int, ...]


@dataclass(frozen=True)
class _ArchiveClip:
    """The root and joint arrays required from one Zarr clip span."""

    clip_id: str
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)), dtype=np.float64
    )


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    return value / np.linalg.norm(value)


def _root_tilt(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    """Return the root orientation with its vertical-axis yaw factored out."""

    return _normalize_quaternion(
        quaternion_multiply_wxyz(
            quaternion_inverse_wxyz(_yaw_quaternion(yaw)), quaternion
        )
    )


def _rotate_xy(vectors: np.ndarray, yaw: float) -> np.ndarray:
    result = np.array(vectors, dtype=np.float64, copy=True)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x = np.asarray(vectors, dtype=np.float64)[..., 0]
    y = np.asarray(vectors, dtype=np.float64)[..., 1]
    result[..., 0] = cosine * x - sine * y
    result[..., 1] = sine * x + cosine * y
    return result


def _decay(frame_offset: np.ndarray, decay_frames: float) -> np.ndarray:
    return np.exp(-np.asarray(frame_offset, dtype=np.float64) / decay_frames)


def _load_selected_clips(
    archive_path: Path,
    selections: Sequence[FragmentSelection],
) -> tuple[tuple[_ArchiveClip, ...], tuple[FragmentSelection, ...], float]:
    import zarr

    archive = zarr.open_group(str(archive_path), mode="r")
    selection_tuple = tuple(selections)
    indices = {selection.archive_clip_index for selection in selection_tuple}
    clips: dict[int, _ArchiveClip] = {}
    for index in indices:
        start = int(archive["clip_start_idx"][index])
        stop = int(archive["clip_end_idx"][index])
        quaternions_xyzw = np.asarray(archive["body_quat_w"][start:stop, 0], dtype=np.float64)
        clips[index] = _ArchiveClip(
            clip_id=str(archive["clip_names"][index]),
            root_position_world=np.asarray(
                archive["body_pos_w"][start:stop, 0], dtype=np.float64
            ),
            root_quaternion_world_wxyz=quaternions_xyzw[..., (3, 0, 1, 2)],
            joint_position=np.asarray(archive["joint_pos"][start:stop], dtype=np.float64),
        )
    return (
        tuple(clips[selection.archive_clip_index] for selection in selection_tuple),
        selection_tuple,
        float(archive["fps"][0]),
    )


def stitch_archive(
    archive_path: Path | str,
    selections: Sequence[FragmentSelection],
    *,
    decay_frames: float = 8.0,
) -> StitchedMotion:
    """Stitch ordered canonical fragment selections from one corpus archive.

    The first selection is copied in source coordinates.  Each following
    selection aligns its *entry* root pose to the previous result, drops that
    duplicate entry sample, and emits the remaining source frames immediately.
    """

    clips, selection_tuple, fps = _load_selected_clips(Path(archive_path), selections)
    if not selection_tuple:
        return StitchedMotion(
            fps=fps,
            root_position_world=np.empty((0, 3), dtype=np.float32),
            root_quaternion_world_wxyz=np.empty((0, 4), dtype=np.float32),
            joint_position=np.empty((0, 29), dtype=np.float32),
            provenance=(),
            seam_indices=(),
        )

    first_selection = selection_tuple[0]
    first_clip = clips[0]
    first_frames = np.arange(
        first_selection.source_start_frame, first_selection.source_stop_frame
    )
    root_positions = [
        np.asarray(first_clip.root_position_world[first_frames], dtype=np.float64).copy()
    ]
    root_quaternions = [
        np.asarray(
            first_clip.root_quaternion_world_wxyz[first_frames], dtype=np.float64
        ).copy()
    ]
    joint_positions = [
        np.asarray(first_clip.joint_position[first_frames], dtype=np.float64).copy()
    ]
    provenance: list[FrameProvenance] = [
        FrameProvenance(first_selection.archive_clip_index, int(frame), first_clip.clip_id)
        for frame in first_frames
    ]
    seam_indices: list[int] = []

    for clip, selection in zip(clips[1:], selection_tuple[1:], strict=True):
        source_frames = np.arange(selection.source_start_frame, selection.source_stop_frame)
        entry_frame = int(source_frames[0])
        emitted_frames = source_frames[1:]
        if not len(emitted_frames):
            continue

        previous_position = root_positions[-1][-1]
        previous_quaternion = _normalize_quaternion(root_quaternions[-1][-1])
        previous_yaw = _yaw_wxyz(previous_quaternion)
        entry_position = np.asarray(clip.root_position_world[entry_frame], dtype=np.float64)
        entry_quaternion = np.asarray(
            clip.root_quaternion_world_wxyz[entry_frame], dtype=np.float64
        )
        entry_yaw = _yaw_wxyz(entry_quaternion)
        yaw_alignment = previous_yaw - entry_yaw

        source_positions = np.asarray(
            clip.root_position_world[emitted_frames], dtype=np.float64
        )
        aligned_positions = previous_position + _rotate_xy(
            source_positions - entry_position, yaw_alignment
        )

        source_quaternions = np.asarray(
            clip.root_quaternion_world_wxyz[emitted_frames], dtype=np.float64
        )
        source_yaws = np.array([_yaw_wxyz(value) for value in source_quaternions])
        source_tilts = np.asarray(
            [_root_tilt(value, yaw) for value, yaw in zip(source_quaternions, source_yaws)],
            dtype=np.float64,
        )
        previous_tilt = _root_tilt(previous_quaternion, previous_yaw)
        entry_tilt = _root_tilt(entry_quaternion, entry_yaw)
        tilt_offset = _normalize_quaternion(
            quaternion_multiply_wxyz(previous_tilt, quaternion_inverse_wxyz(entry_tilt))
        )
        source_offsets = emitted_frames - entry_frame
        decay = _decay(source_offsets, decay_frames)
        tilt_corrections = slerp_wxyz(
            np.broadcast_to(tilt_offset, (len(emitted_frames), 4)),
            np.broadcast_to(np.array((1.0, 0.0, 0.0, 0.0)), (len(emitted_frames), 4)),
            1.0 - decay,
        )
        corrected_tilts = []
        for correction, tilt in zip(tilt_corrections, source_tilts, strict=True):
            candidate = _normalize_quaternion(quaternion_multiply_wxyz(correction, tilt))
            corrected_tilts.append(_root_tilt(candidate, _yaw_wxyz(candidate)))
        aligned_quaternions = np.asarray(
            [
                _normalize_quaternion(
                    quaternion_multiply_wxyz(
                        _yaw_quaternion(yaw + yaw_alignment),
                        tilt,
                    )
                )
                for yaw, tilt in zip(source_yaws, corrected_tilts, strict=True)
            ],
            dtype=np.float64,
        )
        for frame, quaternion in enumerate(aligned_quaternions):
            reference = previous_quaternion if frame == 0 else aligned_quaternions[frame - 1]
            if np.dot(reference, quaternion) < 0.0:
                aligned_quaternions[frame] *= -1.0

        entry_joints = np.asarray(clip.joint_position[entry_frame], dtype=np.float64)
        joint_offset = joint_positions[-1][-1] - entry_joints
        aligned_joints = np.asarray(clip.joint_position[emitted_frames], dtype=np.float64)
        aligned_joints = aligned_joints + decay[:, None] * joint_offset

        seam_indices.append(sum(len(values) for values in root_positions))
        root_positions.append(aligned_positions)
        root_quaternions.append(aligned_quaternions)
        joint_positions.append(aligned_joints)
        provenance.extend(
            FrameProvenance(selection.archive_clip_index, int(frame), clip.clip_id)
            for frame in emitted_frames
        )

    return StitchedMotion(
        fps=fps,
        root_position_world=np.concatenate(root_positions).astype(np.float32),
        root_quaternion_world_wxyz=np.concatenate(root_quaternions).astype(np.float32),
        joint_position=np.concatenate(joint_positions).astype(np.float32),
        provenance=tuple(provenance),
        seam_indices=tuple(seam_indices),
    )
