"""Bounded per-foot terrain retargeting for offline horizontal routes."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .joints import ContractError, PINNED_TARGET_TO_SOURCE_PERMUTATION
from .torch_g1_fk import MujocoG1FootKinematics, target_state_qpos


def retarget_foot_targets(
    *,
    foot_position_world: object,
    support_mask: object,
    target_surface_at_ankle_m: object,
    minimum_sole_clearance_by_foot_m: object,
    ankle_origin_sole_m: float,
    swing_clearance_margin_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build foot-origin constraints for support and colliding swing feet."""

    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    surface = np.asarray(target_surface_at_ankle_m, dtype=np.float64)
    clearance = np.asarray(
        minimum_sole_clearance_by_foot_m, dtype=np.float64
    )
    if (
        feet.shape != (2, 3)
        or support.dtype != np.bool_
        or support.shape != (2,)
        or surface.shape != (2,)
        or clearance.shape != (2,)
        or not all(
            np.isfinite(value).all()
            for value in (feet, surface, clearance)
        )
        or not math.isfinite(float(ankle_origin_sole_m))
        or ankle_origin_sole_m <= 0.0
        or not math.isfinite(float(swing_clearance_margin_m))
        or swing_clearance_margin_m < 0.0
    ):
        raise ContractError("horizontal terrain retarget targets are invalid")
    targets = feet.copy()
    targets[support, 2] = (
        surface[support] + float(ankle_origin_sole_m)
    )
    colliding_swing = (~support) & (
        clearance < float(swing_clearance_margin_m)
    )
    targets[colliding_swing, 2] += (
        float(swing_clearance_margin_m) - clearance[colliding_swing]
    )
    return support | colliding_swing, targets


class WideBoundG1TerrainRetargeter:
    """Solve selected G1 feet with bounds suitable for adjacent stair treads."""

    def __init__(
        self,
        g1_xml: object,
        *,
        maximum_joint_deviation_rad: float = 1.4,
        maximum_root_height_deviation_m: float = 0.20,
    ) -> None:
        if (
            not math.isfinite(float(maximum_joint_deviation_rad))
            or maximum_joint_deviation_rad <= 0.0
            or not math.isfinite(float(maximum_root_height_deviation_m))
            or maximum_root_height_deviation_m <= 0.0
        ):
            raise ContractError("horizontal terrain retarget bounds are invalid")
        self._kinematics = MujocoG1FootKinematics(g1_xml)
        self._joint_deviation = float(maximum_joint_deviation_rad)
        self._root_deviation = float(maximum_root_height_deviation_m)

    @property
    def kinematics(self) -> MujocoG1FootKinematics:
        return self._kinematics

    def solve_frame(
        self,
        *,
        joint_position: object,
        root_position_world: object,
        root_orientation_world_wxyz: object,
        solve_feet: object,
        target_foot_position_world: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joint_position, dtype=np.float64)
        root = np.asarray(root_position_world, dtype=np.float64)
        quaternion = np.asarray(
            root_orientation_world_wxyz, dtype=np.float64
        )
        mask = np.asarray(solve_feet)
        targets = np.asarray(target_foot_position_world, dtype=np.float64)
        if (
            joints.shape != (29,)
            or root.shape != (3,)
            or quaternion.shape != (4,)
            or mask.dtype != np.bool_
            or mask.shape != (2,)
            or not bool(mask.any())
            or targets.shape != (2, 3)
            or not all(
                np.isfinite(value).all()
                for value in (joints, root, quaternion, targets)
            )
        ):
            raise ContractError("horizontal terrain retarget frame is invalid")
        kinematics = self._kinematics
        data = kinematics._data
        model = kinematics._model
        mujoco = kinematics._mujoco
        data.qpos[:] = target_state_qpos(joints, root, quaternion)
        selected_feet = np.flatnonzero(mask)
        selected_qpos = np.concatenate(
            [
                kinematics._leg_qpos_addresses[index]
                for index in selected_feet
            ]
        )
        selected_dofs = np.concatenate(
            [
                kinematics._leg_dof_addresses[index]
                for index in selected_feet
            ]
        )
        selected_ranges = np.concatenate(
            [kinematics._leg_ranges[index] for index in selected_feet]
        )
        seed_joints = data.qpos[selected_qpos].copy()
        seed = np.concatenate((seed_joints, [root[2]]))
        lower = np.concatenate(
            (
                np.maximum(
                    selected_ranges[:, 0],
                    seed_joints - self._joint_deviation,
                ),
                [root[2] - self._root_deviation],
            )
        )
        upper = np.concatenate(
            (
                np.minimum(
                    selected_ranges[:, 1],
                    seed_joints + self._joint_deviation,
                ),
                [root[2] + self._root_deviation],
            )
        )
        initial = seed.copy()
        for selected_order in range(len(selected_feet)):
            offset = selected_order * 6
            initial[offset] -= 0.075
            initial[offset + 3] += 0.15
            initial[offset + 4] -= 0.075
        initial = np.minimum(np.maximum(initial, lower), upper)
        posture_scale = np.full(len(selected_dofs), 0.003, dtype=np.float64)
        # Position-only IK otherwise uses extreme ankle pitch/roll to satisfy
        # unequal tread heights, tipping a sole through the neighboring tread.
        for selected_order in range(len(selected_feet)):
            posture_scale[selected_order * 6 + 4 : selected_order * 6 + 6] = 0.3
        root_scale = 0.01
        orientation_scale = 0.2
        position_scale = 20.0

        def set_state(value: np.ndarray) -> None:
            data.qpos[selected_qpos] = value[:-1]
            data.qpos[2] = value[-1]
            mujoco.mj_forward(model, data)

        def residual(value: np.ndarray) -> np.ndarray:
            set_state(value)
            feet = np.concatenate(
                [
                    position_scale
                    * (
                        data.xpos[kinematics._foot_body_ids[index]]
                        - targets[index]
                    )
                    for index in selected_feet
                ]
            )
            orientations = np.concatenate(
                [
                    orientation_scale
                    * (
                        data.xmat[
                            kinematics._foot_body_ids[index]
                        ].reshape(3, 3)[:, 2]
                        - np.array((0.0, 0.0, 1.0))
                    )
                    for index in selected_feet
                ]
            )
            return np.concatenate(
                (
                    feet,
                    orientations,
                    posture_scale * (value[:-1] - seed_joints),
                    [root_scale * (value[-1] - root[2])],
                )
            )

        def jacobian(value: np.ndarray) -> np.ndarray:
            set_state(value)
            foot_rows = []
            orientation_rows = []
            for index in selected_feet:
                row = np.zeros((3, model.nv), dtype=np.float64)
                rotation_row = np.zeros(
                    (3, model.nv), dtype=np.float64
                )
                mujoco.mj_jacBody(
                    model,
                    data,
                    row,
                    rotation_row,
                    int(kinematics._foot_body_ids[index]),
                )
                foot_rows.append(
                    position_scale
                    * np.concatenate(
                        (
                            row[:, selected_dofs],
                            np.array(((0.0,), (0.0,), (1.0,))),
                        ),
                        axis=1,
                    )
                )
                body_z = data.xmat[
                    kinematics._foot_body_ids[index]
                ].reshape(3, 3)[:, 2]
                skew = np.array(
                    (
                        (0.0, -body_z[2], body_z[1]),
                        (body_z[2], 0.0, -body_z[0]),
                        (-body_z[1], body_z[0], 0.0),
                    )
                )
                orientation_rows.append(
                    np.concatenate(
                        (
                            -orientation_scale
                            * skew
                            @ rotation_row[:, selected_dofs],
                            np.zeros((3, 1)),
                        ),
                        axis=1,
                    )
                )
            posture = np.concatenate(
                (
                    np.diag(posture_scale),
                    np.zeros((len(selected_dofs), 1)),
                ),
                axis=1,
            )
            root_row = np.zeros((1, len(selected_dofs) + 1))
            root_row[0, -1] = root_scale
            return np.concatenate(
                (*foot_rows, *orientation_rows, posture, root_row),
                axis=0,
            )

        result = least_squares(
            residual,
            initial,
            jac=jacobian,
            bounds=(lower, upper),
            max_nfev=256,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        set_state(result.x)
        errors = [
            np.linalg.norm(
                data.xpos[kinematics._foot_body_ids[index]]
                - targets[index]
            )
            for index in selected_feet
        ]
        if max(errors) > 0.005:
            raise ContractError(
                "horizontal terrain retarget target is unreachable: "
                f"{max(errors):.6f} m"
            )
        source_joints = np.asarray(data.qpos[7:], dtype=np.float64)
        target_joints = source_joints[
            np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION, dtype=np.int64)
        ]
        solved_root = root.copy()
        solved_root[2] = result.x[-1]
        return (
            np.ascontiguousarray(target_joints),
            np.ascontiguousarray(solved_root),
        )
