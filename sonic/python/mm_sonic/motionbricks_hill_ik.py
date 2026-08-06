"""Bounded display-only foot IK for the MotionBricks hill prototype."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


HeightQuery = Callable[[object], float]

_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_LEG_JOINT_NAMES = (
    (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
)


class FootPhase(str, Enum):
    SWING = "swing"
    STANCE = "stance"
    RELEASE = "release"


@dataclass(frozen=True)
class HillFootIKDiagnostics:
    phases: tuple[FootPhase, FootPhase]
    authored_stance: tuple[bool, bool]
    locked: tuple[bool, bool]
    root_height_correction_m: float
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


def _bounded_root_height_correction(
    current_m: float,
    *,
    required_support_shift_m: float,
    has_support: bool,
) -> float:
    current = float(current_m)
    required = float(required_support_shift_m)
    if (
        not np.isfinite(current)
        or not np.isfinite(required)
        or type(has_support) is not bool
    ):
        raise ValueError("root-height inputs must be finite")
    if has_support:
        desired = current + required
        step = 0.025
    else:
        desired = 0.0
        step = 0.010
    rate_bounded = float(
        np.clip(desired, current - step, current + step)
    )
    return float(np.clip(rate_bounded, -0.20, 0.20))


def _bounded_leg_correction(
    *,
    raw: np.ndarray,
    desired: np.ndarray,
    previous: np.ndarray,
    limits: np.ndarray,
) -> tuple[np.ndarray, bool]:
    raw_values = np.asarray(raw, dtype=np.float64)
    desired_values = np.asarray(desired, dtype=np.float64)
    previous_values = np.asarray(previous, dtype=np.float64)
    joint_limits = np.asarray(limits, dtype=np.float64)
    if (
        raw_values.shape != desired_values.shape
        or raw_values.shape != previous_values.shape
        or joint_limits.shape != (len(raw_values), 2)
    ):
        raise ValueError("leg correction arrays have incompatible shapes")
    desired_values = np.clip(desired_values, -0.35, 0.35)
    rate_bounded = np.clip(
        desired_values,
        previous_values - 0.06,
        previous_values + 0.06,
    )
    limit_bounded = (
        np.clip(
            raw_values + rate_bounded,
            joint_limits[:, 0] + 1.0e-6,
            joint_limits[:, 1] - 1.0e-6,
        )
        - raw_values
    )
    limit_forced = bool(
        np.any(np.abs(limit_bounded - rate_bounded) > 1.0e-12)
    )
    return limit_bounded, limit_forced


def _descendants(model: object, root_body: int) -> frozenset[int]:
    result: set[int] = set()
    for body_id in range(1, int(model.nbody)):
        cursor = body_id
        while cursor > 0:
            if cursor == root_body:
                result.add(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return frozenset(result)


class MotionBricksHillFootIK:
    """Apply bounded leg-only IK to a copy of a native G1 qpos."""

    def __init__(
        self,
        model: object,
        height_query: HeightQuery,
        *,
        maximum_iterations: int = 24,
        damping: float = 0.012,
        posture_weight: float = 0.00005,
    ) -> None:
        import mujoco

        if maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if damping <= 0.0 or posture_weight < 0.0:
            raise ValueError("solver weights must be non-negative")
        self._mujoco = mujoco
        self.model = model
        self.height_query = height_query
        self.maximum_iterations = int(maximum_iterations)
        self.damping = float(damping)
        self.posture_weight = float(posture_weight)
        self._discover_g1_layout()
        self._data = mujoco.MjData(model)
        self.reset()

    @property
    def non_leg_qpos_addresses(self) -> np.ndarray:
        return self._non_leg_qpos.copy()

    def reset(self) -> None:
        self._previous_contact = (False, False)
        self._root_height_correction_m = 0.0
        self._phases = (FootPhase.SWING, FootPhase.SWING)
        self._targets: tuple[np.ndarray | None, np.ndarray | None] = (
            None,
            None,
        )
        self._previous_raw_centers: (
            tuple[np.ndarray, np.ndarray] | None
        ) = None
        self._corrections = (
            np.zeros(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        )

    def snapshot_state(
        self,
    ) -> tuple[
        tuple[bool, bool],
        tuple[np.ndarray | None, np.ndarray | None],
        float,
        tuple[np.ndarray, np.ndarray],
    ]:
        targets = tuple(
            None if target is None else target.copy()
            for target in self._targets
        )
        corrections = tuple(
            value.copy() for value in self._corrections
        )
        return (
            self._previous_contact,
            targets,
            float(self._root_height_correction_m),
            corrections,
        )

    def _discover_g1_layout(self) -> None:
        mujoco = self._mujoco
        free = np.flatnonzero(
            np.asarray(self.model.jnt_type)
            == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        if free.size != 1:
            raise ValueError("G1 model must contain exactly one free joint")
        root_address = int(self.model.jnt_qposadr[int(free[0])])
        if root_address != 0:
            raise ValueError("G1 free root must occupy qpos[0:7]")

        leg_qpos: list[np.ndarray] = []
        leg_dofs: list[np.ndarray] = []
        leg_limits: list[np.ndarray] = []
        for names in _LEG_JOINT_NAMES:
            joint_ids = np.asarray(
                [
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                    for name in names
                ],
                dtype=np.int32,
            )
            if np.any(joint_ids < 0):
                raise ValueError("G1 model is missing a leg joint")
            leg_qpos.append(
                np.asarray(
                    self.model.jnt_qposadr[joint_ids], dtype=np.int32
                )
            )
            leg_dofs.append(
                np.asarray(
                    self.model.jnt_dofadr[joint_ids], dtype=np.int32
                )
            )
            leg_limits.append(
                np.asarray(self.model.jnt_range[joint_ids], dtype=np.float64)
            )
        self._leg_qpos = (leg_qpos[0], leg_qpos[1])
        self._leg_dofs = (leg_dofs[0], leg_dofs[1])
        self._leg_limits = (leg_limits[0], leg_limits[1])

        sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        sphere_geoms: list[tuple[int, ...]] = []
        sphere_radii: list[np.ndarray] = []
        for body_name in _FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"G1 model is missing body {body_name}")
            descendants = _descendants(self.model, int(body_id))
            spheres = tuple(
                geom_id
                for geom_id in range(int(self.model.ngeom))
                if (
                    int(self.model.geom_bodyid[geom_id]) in descendants
                    and int(self.model.geom_type[geom_id]) == sphere_type
                )
            )
            if len(spheres) != 4:
                raise ValueError(
                    f"G1 model must have four sole spheres for {body_name}"
                )
            sphere_geoms.append(spheres)
            sphere_radii.append(
                np.asarray(
                    [self.model.geom_size[index, 0] for index in spheres],
                    dtype=np.float64,
                )
            )
        self._sphere_geoms = (sphere_geoms[0], sphere_geoms[1])
        self._sphere_radii = (sphere_radii[0], sphere_radii[1])

        leg_addresses = np.concatenate(self._leg_qpos)
        all_addresses = np.arange(int(self.model.nq), dtype=np.int32)
        self._non_leg_qpos = all_addresses[
            ~np.isin(all_addresses, leg_addresses)
        ]

    def _set_pose(self, qpos: np.ndarray) -> None:
        self._data.qpos[:] = qpos
        self._mujoco.mj_forward(self.model, self._data)

    def _sole_centers(self, foot: int) -> np.ndarray:
        return np.asarray(
            [
                self._data.geom_xpos[index]
                for index in self._sphere_geoms[foot]
            ],
            dtype=np.float64,
        )

    def _clearances(
        self, centers: np.ndarray, radii: np.ndarray
    ) -> np.ndarray:
        terrain = np.asarray(
            [float(self.height_query(point[:2])) for point in centers],
            dtype=np.float64,
        )
        clearance = centers[:, 2] - radii - terrain
        if not np.isfinite(clearance).all():
            raise ValueError("terrain clearances must be finite")
        return clearance

    def _solve_foot(
        self,
        foot: int,
        target: np.ndarray,
        reference_qpos: np.ndarray,
        phase: FootPhase,
    ) -> int:
        addresses = self._leg_qpos[foot]
        dofs = self._leg_dofs[foot]
        limits = self._leg_limits[foot]
        reference = reference_qpos[addresses]
        identity = np.eye(6, dtype=np.float64)
        jacobian_position = np.zeros(
            (3, int(self.model.nv)), dtype=np.float64
        )
        iterations = 0
        for _ in range(self.maximum_iterations):
            iterations += 1
            self._mujoco.mj_forward(self.model, self._data)
            current = self._sole_centers(foot)
            jacobians: list[np.ndarray] = []
            for geom_id in self._sphere_geoms[foot]:
                jacobian_position.fill(0.0)
                self._mujoco.mj_jac(
                    self.model,
                    self._data,
                    jacobian_position,
                    None,
                    self._data.geom_xpos[geom_id],
                    int(self.model.geom_bodyid[geom_id]),
                )
                jacobians.append(jacobian_position[:, dofs].copy())
            vertical_error = target[:, 2] - current[:, 2]
            rows = [value[2:3] for value in jacobians]
            error = vertical_error
            residual = float(np.max(np.abs(vertical_error)))
            if phase is FootPhase.STANCE:
                centroid_error = (
                    np.mean(target[:, :2], axis=0)
                    - np.mean(current[:, :2], axis=0)
                )
                centroid_jacobian = np.mean(
                    np.asarray(jacobians), axis=0
                )
                rows.extend(
                    (
                        centroid_jacobian[0:1],
                        centroid_jacobian[1:2],
                    )
                )
                error = np.concatenate((vertical_error, centroid_error))
                residual = max(
                    residual, float(np.linalg.norm(centroid_error))
                )
            if residual <= 2.5e-4:
                break
            jacobian = np.vstack(rows)
            transpose = jacobian.T
            lhs = (
                transpose @ jacobian
                + (self.damping**2 + self.posture_weight) * identity
            )
            rhs = transpose @ error + self.posture_weight * (
                reference - self._data.qpos[addresses]
            )
            delta = np.clip(np.linalg.solve(lhs, rhs), -0.04, 0.04)
            self._data.qpos[addresses] = np.clip(
                self._data.qpos[addresses] + delta,
                limits[:, 0] + 1.0e-6,
                limits[:, 1] - 1.0e-6,
            )
        return iterations

    def _fallback(
        self,
        raw_qpos: np.ndarray,
        reason: str,
        *,
        raw_penetration_m: float = 0.0,
    ) -> HillFootIKResult:
        diagnostics = HillFootIKDiagnostics(
            phases=self._phases,
            authored_stance=self._previous_contact,
            locked=tuple(
                target is not None for target in self._targets
            ),
            root_height_correction_m=self._root_height_correction_m,
            raw_penetration_m=float(raw_penetration_m),
            corrected_penetration_m=float(raw_penetration_m),
            maximum_target_residual_m=0.0,
            maximum_joint_correction_rad=0.0,
            iterations=0,
            accepted=False,
            reason=reason,
        )
        return HillFootIKResult(raw_qpos.copy(), diagnostics)

    def apply(
        self, raw_qpos: object, dt_s: float
    ) -> HillFootIKResult:
        raw = np.asarray(raw_qpos, dtype=np.float64)
        if raw.shape != (int(self.model.nq),):
            raise ValueError("raw qpos must have shape (model.nq,)")
        if not np.isfinite(raw).all():
            raise ValueError("raw qpos must be finite")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        raw = raw.copy()
        raw_penetration = 0.0

        try:
            self._set_pose(raw)
            raw_centers = (
                self._sole_centers(0).copy(),
                self._sole_centers(1).copy(),
            )
            raw_clearances = (
                self._clearances(
                    raw_centers[0], self._sphere_radii[0]
                ),
                self._clearances(
                    raw_centers[1], self._sphere_radii[1]
                ),
            )
            raw_penetration = max(
                0.0,
                -min(
                    float(np.min(raw_clearances[0])),
                    float(np.min(raw_clearances[1])),
                ),
            )
            if self._previous_raw_centers is None:
                speeds = (float("inf"), float("inf"))
            else:
                speeds = tuple(
                    float(
                        np.linalg.norm(
                            np.mean(raw_centers[foot], axis=0)
                            - np.mean(
                                self._previous_raw_centers[foot], axis=0
                            )
                        )
                        / float(dt_s)
                    )
                    for foot in range(2)
                )
            phases = tuple(
                _next_foot_phase(
                    self._phases[foot],
                    float(np.min(raw_clearances[foot])),
                    speeds[foot],
                )
                for foot in range(2)
            )

            proposed_targets: list[np.ndarray | None] = []
            solve_targets: list[np.ndarray | None] = []
            for foot in range(2):
                if phases[foot] is FootPhase.STANCE:
                    if (
                        self._phases[foot] is FootPhase.STANCE
                        and self._targets[foot] is not None
                    ):
                        target = self._targets[foot].copy()
                    else:
                        target = _project_stance_targets(
                            raw_centers[foot],
                            self._sphere_radii[foot],
                            self.height_query,
                        )
                    proposed_targets.append(target.copy())
                    solve_targets.append(target)
                elif phases[foot] is FootPhase.SWING:
                    target = None
                    if self._previous_raw_centers is not None:
                        lift = _required_swing_lift(
                            raw_centers[foot],
                            self._sphere_radii[foot],
                            self.height_query,
                        )
                        if lift > 0.0:
                            target = raw_centers[foot].copy()
                            target[:, 2] += lift
                    proposed_targets.append(None)
                    solve_targets.append(target)
                else:
                    proposed_targets.append(None)
                    solve_targets.append(None)

            self._set_pose(raw)
            iterations = 0
            for foot, target in enumerate(solve_targets):
                if target is not None:
                    iterations += self._solve_foot(
                        foot, target, raw, phases[foot]
                    )

            corrections: list[np.ndarray] = []
            limit_forced: list[bool] = []
            for foot in range(2):
                addresses = self._leg_qpos[foot]
                bounded, forced = _bounded_leg_correction(
                    raw=raw[addresses],
                    desired=self._data.qpos[addresses] - raw[addresses],
                    previous=self._corrections[foot],
                    limits=self._leg_limits[foot],
                )
                corrections.append(bounded)
                limit_forced.append(forced)

            candidate = raw.copy()
            for foot in range(2):
                candidate[self._leg_qpos[foot]] += corrections[foot]
            candidate[self._non_leg_qpos] = raw[self._non_leg_qpos]
            if not np.isfinite(candidate).all():
                raise ValueError("IK candidate must be finite")
            self._set_pose(candidate)
            corrected_centers = (
                self._sole_centers(0).copy(),
                self._sole_centers(1).copy(),
            )
            corrected_clearances = (
                self._clearances(
                    corrected_centers[0], self._sphere_radii[0]
                ),
                self._clearances(
                    corrected_centers[1], self._sphere_radii[1]
                ),
            )
            corrected_penetration = max(
                0.0,
                -min(
                    float(np.min(corrected_clearances[0])),
                    float(np.min(corrected_clearances[1])),
                ),
            )

            residuals = [
                float(
                    np.max(
                        np.abs(
                            target[:, 2] - corrected_centers[foot][:, 2]
                        )
                    )
                )
                for foot, target in enumerate(solve_targets)
                if target is not None
            ]
            maximum_residual = max(residuals, default=0.0)
            stance_residuals = [
                float(
                    np.max(
                        np.abs(
                            proposed_targets[foot][:, 2]
                            - corrected_centers[foot][:, 2]
                        )
                    )
                )
                for foot in range(2)
                if proposed_targets[foot] is not None
            ]
            maximum_stance_residual = max(
                stance_residuals, default=0.0
            )
            maximum_correction = max(
                float(np.max(np.abs(value)))
                for value in corrections
            )
            maximum_change = max(
                (
                    float(
                        np.max(
                            np.abs(value - self._corrections[foot])
                        )
                    )
                    for foot, value in enumerate(corrections)
                    if not limit_forced[foot]
                ),
                default=0.0,
            )
            if maximum_correction > 0.35 + 1.0e-9:
                raise ValueError("joint correction exceeded absolute bound")
            if maximum_change > 0.06 + 1.0e-9:
                raise ValueError("joint correction exceeded frame bound")
            acquiring_stance = maximum_stance_residual > 0.015
            if (
                acquiring_stance
                and corrected_penetration
                >= raw_penetration - 1.0e-6
            ):
                return self._fallback(
                    raw,
                    "stance acquisition did not reduce penetration",
                    raw_penetration_m=raw_penetration,
                )

            committed_targets = (
                (
                    None
                    if proposed_targets[0] is None
                    else proposed_targets[0].copy()
                ),
                (
                    None
                    if proposed_targets[1] is None
                    else proposed_targets[1].copy()
                ),
            )
            committed_centers = (
                raw_centers[0].copy(),
                raw_centers[1].copy(),
            )
            committed_corrections = (
                corrections[0].copy(),
                corrections[1].copy(),
            )
            self._phases = (phases[0], phases[1])
            self._targets = committed_targets
            self._previous_raw_centers = committed_centers
            self._corrections = committed_corrections

            diagnostics = HillFootIKDiagnostics(
                phases=self._phases,
                authored_stance=self._previous_contact,
                locked=tuple(
                    target is not None for target in self._targets
                ),
                root_height_correction_m=self._root_height_correction_m,
                raw_penetration_m=raw_penetration,
                corrected_penetration_m=corrected_penetration,
                maximum_target_residual_m=maximum_residual,
                maximum_joint_correction_rad=maximum_correction,
                iterations=iterations,
                accepted=True,
                reason=(
                    "acquiring stance" if acquiring_stance else "ok"
                ),
            )
            return HillFootIKResult(candidate, diagnostics)
        except Exception as error:
            return self._fallback(
                raw,
                str(error),
                raw_penetration_m=raw_penetration,
            )
