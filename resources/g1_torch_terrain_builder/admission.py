"""Common FK, contact-alignment, and duplicate admission for terrain clips."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np

from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from resources.g1_torch_stair_builder.conversion import NativeMotionArrays

from .registry import ResolvedSource
from .terrain import TerrainEvidence


_SOLE_M = 0.035
_CONTACT_ERROR_CANDIDATE_M = 0.08
_CONTACT_VERTICAL_SPEED_MPS = 0.12
_MINIMUM_CONTACT_SAMPLES = 50
_MINIMUM_ELEVATED_CONTACT_SAMPLES = 25
_ELEVATED_SURFACE_MINIMUM_M = 0.03
_CONTACT_P95_ERROR_M = 0.035
_FK_P95_ERROR_M = 0.005
_FK_MAXIMUM_ERROR_M = 0.02
_LOCOMOTION_BODIES = np.asarray(
    (0, 1, 2, 4, 5, 7, 8, 10, 11, 14, 15, 18, 19), np.int64
)


def _empty_distribution() -> Mapping[str, float]:
    return MappingProxyType(
        {
            "minimum": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "maximum": math.nan,
        }
    )


def _distribution(value: np.ndarray) -> Mapping[str, float]:
    array = np.asarray(value, np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        return _empty_distribution()
    return MappingProxyType(
        {
            "minimum": float(array.min()),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "maximum": float(array.max()),
        }
    )


@dataclass(frozen=True)
class AdmissionReport:
    logical_name: str
    accepted: bool
    reason: str | None
    output_frames: int
    contact_sample_count: int
    elevated_contact_sample_count: int
    contact_height_error_m: Mapping[str, float]
    fk_position_error_m: Mapping[str, float]
    fk_full_body_position_error_m: Mapping[str, float]
    duplicate_of: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.logical_name, str) or not self.logical_name:
            raise ValueError("admission logical name must be non-empty")
        if type(self.accepted) is not bool:
            raise ValueError("admission accepted must be boolean")
        if self.accepted != (self.reason is None):
            raise ValueError("accepted admission must have no rejection reason")
        if type(self.output_frames) is not int or self.output_frames < 0:
            raise ValueError("admission output_frames must be non-negative")
        if (
            type(self.contact_sample_count) is not int
            or self.contact_sample_count < 0
        ):
            raise ValueError("contact sample count must be non-negative")
        if (
            type(self.elevated_contact_sample_count) is not int
            or self.elevated_contact_sample_count < 0
        ):
            raise ValueError(
                "elevated contact sample count must be non-negative"
            )


def _report(
    source: ResolvedSource,
    motion: NativeMotionArrays,
    *,
    reason: str | None,
    contact_count: int = 0,
    elevated_contact_count: int = 0,
    contact_error: Mapping[str, float] | None = None,
    fk_error: Mapping[str, float] | None = None,
    fk_full_error: Mapping[str, float] | None = None,
    duplicate_of: str | None = None,
) -> AdmissionReport:
    return AdmissionReport(
        logical_name=source.spec.logical_name,
        accepted=reason is None,
        reason=reason,
        output_frames=len(motion.joint_position),
        contact_sample_count=contact_count,
        elevated_contact_sample_count=elevated_contact_count,
        contact_height_error_m=(
            _empty_distribution() if contact_error is None else contact_error
        ),
        fk_position_error_m=(
            _empty_distribution() if fk_error is None else fk_error
        ),
        fk_full_body_position_error_m=(
            _empty_distribution()
            if fk_full_error is None
            else fk_full_error
        ),
        duplicate_of=duplicate_of,
    )


def _native_qpos(motion: NativeMotionArrays) -> np.ndarray:
    frames = len(motion.joint_position)
    source_joint = np.empty((frames, 29), np.float64)
    source_joint[
        :, np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION, np.int64)
    ] = motion.joint_position
    return np.ascontiguousarray(
        np.concatenate(
            (
                motion.body_position_world[:, 0],
                motion.body_quaternion_world_wxyz[:, 0],
                source_joint,
            ),
            axis=1,
        ),
        np.float64,
    )


def _terrain_xy(
    points_xy: np.ndarray, transform: tuple[float, float, float]
) -> np.ndarray:
    tx, ty, yaw = (float(value) for value in transform)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    return np.stack(
        (
            cosine * x - sine * y + tx,
            sine * x + cosine * y + ty,
        ),
        axis=-1,
    ).astype(np.float32)


def admit_candidate(
    source: ResolvedSource,
    motion: NativeMotionArrays,
    terrain: TerrainEvidence | None,
    *,
    fk: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    accepted_motion_hashes: Mapping[str, str],
) -> AdmissionReport:
    if not isinstance(source, ResolvedSource):
        raise TypeError("source must be a ResolvedSource")
    if not isinstance(motion, NativeMotionArrays):
        raise TypeError("motion must be NativeMotionArrays")
    motion.validate()
    if not callable(fk):
        raise TypeError("fk must be callable")
    if not isinstance(accepted_motion_hashes, Mapping):
        raise TypeError("accepted_motion_hashes must be a mapping")
    duplicate = accepted_motion_hashes.get(source.spec.motion_sha256)
    if duplicate is not None:
        if not isinstance(duplicate, str) or not duplicate:
            raise ValueError("duplicate owner must be a non-empty string")
        return _report(
            source, motion, reason="exact_duplicate", duplicate_of=duplicate
        )
    if terrain is None:
        return _report(
            source, motion, reason="missing_authoritative_terrain"
        )
    if not isinstance(terrain, TerrainEvidence):
        raise TypeError("terrain must be TerrainEvidence or None")

    qpos = _native_qpos(motion)
    try:
        fk_position, fk_quaternion = fk(qpos)
    except Exception as error:
        raise ValueError("candidate FK evaluation failed") from error
    fk_position = np.asarray(fk_position, np.float64)
    fk_quaternion = np.asarray(fk_quaternion, np.float64)
    expected_position = motion.body_position_world.shape
    expected_quaternion = motion.body_quaternion_world_wxyz.shape
    if (
        fk_position.shape != expected_position
        or fk_quaternion.shape != expected_quaternion
        or not np.isfinite(fk_position).all()
        or not np.isfinite(fk_quaternion).all()
    ):
        raise ValueError("candidate FK returned invalid body arrays")
    quaternion_norm = np.linalg.norm(fk_quaternion, axis=-1)
    if np.max(np.abs(quaternion_norm - 1.0)) > 1e-4:
        raise ValueError("candidate FK returned invalid quaternions")
    full_fk_distance = np.linalg.norm(
        fk_position - motion.body_position_world.astype(np.float64), axis=-1
    )
    fk_distance = full_fk_distance[:, _LOCOMOTION_BODIES]
    fk_error = _distribution(fk_distance)
    full_fk_error = _distribution(full_fk_distance)
    if (
        fk_error["p95"] > _FK_P95_ERROR_M
        or fk_error["maximum"] > _FK_MAXIMUM_ERROR_M
    ):
        return _report(
            source,
            motion,
            reason="fk_mismatch",
            fk_error=fk_error,
            fk_full_error=full_fk_error,
        )

    feet = motion.body_position_world[:, (18, 19)].astype(np.float64)
    scene_xy = _terrain_xy(
        feet[..., :2], terrain.motion_to_terrain_xy_yaw
    )
    surface = terrain.grid.sample_xy(scene_xy).astype(np.float64)
    clearance = feet[..., 2] - surface
    vertical_speed = np.abs(
        motion.body_linear_velocity_world[:, (18, 19), 2].astype(np.float64)
    )
    error = np.abs(clearance - _SOLE_M)
    contact = (
        (error <= _CONTACT_ERROR_CANDIDATE_M)
        & (vertical_speed <= _CONTACT_VERTICAL_SPEED_MPS)
    )
    contact_error = _distribution(error[contact])
    contact_count = int(contact.sum())
    terrain_minimum = float(np.min(terrain.grid.height_z))
    has_elevated_terrain = (
        float(np.max(terrain.grid.height_z)) - terrain_minimum
        > _ELEVATED_SURFACE_MINIMUM_M
    )
    elevated_contact = contact & (
        surface > terrain_minimum + _ELEVATED_SURFACE_MINIMUM_M
    )
    elevated_contact_count = int(elevated_contact.sum())
    if contact_count < _MINIMUM_CONTACT_SAMPLES:
        return _report(
            source,
            motion,
            reason="insufficient_contact_samples",
            contact_count=contact_count,
            elevated_contact_count=elevated_contact_count,
            contact_error=contact_error,
            fk_error=fk_error,
            fk_full_error=full_fk_error,
        )
    if contact_error["p95"] > _CONTACT_P95_ERROR_M:
        return _report(
            source,
            motion,
            reason="contact_alignment_failed",
            contact_count=contact_count,
            elevated_contact_count=elevated_contact_count,
            contact_error=contact_error,
            fk_error=fk_error,
            fk_full_error=full_fk_error,
        )
    if (
        has_elevated_terrain
        and elevated_contact_count < _MINIMUM_ELEVATED_CONTACT_SAMPLES
    ):
        return _report(
            source,
            motion,
            reason="insufficient_elevated_contact_samples",
            contact_count=contact_count,
            elevated_contact_count=elevated_contact_count,
            contact_error=contact_error,
            fk_error=fk_error,
            fk_full_error=full_fk_error,
        )
    return _report(
        source,
        motion,
        reason=None,
        contact_count=contact_count,
        elevated_contact_count=elevated_contact_count,
        contact_error=contact_error,
        fk_error=fk_error,
        fk_full_error=full_fk_error,
    )
