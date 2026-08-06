"""Bounded display-only foot IK for the MotionBricks hill prototype."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


HeightQuery = Callable[[object], float]


class FootPhase(str, Enum):
    SWING = "swing"
    STANCE = "stance"
    RELEASE = "release"


@dataclass(frozen=True)
class HillFootIKDiagnostics:
    phases: tuple[FootPhase, FootPhase]
    raw_penetration_m: float
    corrected_penetration_m: float
    maximum_target_residual_m: float
    maximum_joint_correction_rad: float
    iterations: int
    accepted: bool
    reason: str


@dataclass(frozen=True)
class HillFootIKResult:
    qpos: np.ndarray
    diagnostics: HillFootIKDiagnostics


def _next_foot_phase(
    previous: FootPhase,
    minimum_clearance_m: float,
    speed_mps: float,
) -> FootPhase:
    if previous is FootPhase.STANCE:
        if minimum_clearance_m > 0.075 or speed_mps > 0.75:
            return FootPhase.RELEASE
        return FootPhase.STANCE
    if previous is FootPhase.RELEASE:
        return FootPhase.SWING
    if minimum_clearance_m <= 0.035 and speed_mps <= 0.35:
        return FootPhase.STANCE
    return FootPhase.SWING


def _project_stance_targets(
    centers_world: np.ndarray,
    radii: np.ndarray,
    height_query: HeightQuery,
) -> np.ndarray:
    centers = np.asarray(centers_world, dtype=np.float64)
    sphere_radii = np.asarray(radii, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("sole centers must have shape (probes, 3)")
    if sphere_radii.shape != (len(centers),):
        raise ValueError("sole radii must match probe count")
    targets = centers.copy()
    targets[:, 2] = np.asarray(
        [float(height_query(point[:2])) for point in centers]
    ) + sphere_radii
    if not np.isfinite(targets).all():
        raise ValueError("stance targets must be finite")
    return targets


def _required_swing_lift(
    centers_world: np.ndarray,
    radii: np.ndarray,
    height_query: HeightQuery,
    minimum_clearance_m: float = 0.015,
) -> float:
    centers = np.asarray(centers_world, dtype=np.float64)
    sphere_radii = np.asarray(radii, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("sole centers must have shape (probes, 3)")
    if sphere_radii.shape != (len(centers),):
        raise ValueError("sole radii must match probe count")
    deficits = [
        float(height_query(point[:2]))
        + float(radius)
        + float(minimum_clearance_m)
        - float(point[2])
        for point, radius in zip(centers, sphere_radii, strict=True)
    ]
    lift = max(0.0, max(deficits, default=0.0))
    if not np.isfinite(lift):
        raise ValueError("swing lift must be finite")
    return lift
