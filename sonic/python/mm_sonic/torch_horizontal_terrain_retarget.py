"""Bounded per-foot terrain retargeting for offline horizontal routes."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .joints import ContractError, PINNED_TARGET_TO_SOURCE_PERMUTATION
from .torch_g1_fk import MujocoG1FootKinematics, target_state_qpos


def stance_segment_anchors(
    *,
    foot_position_world: object,
    support_mask: object,
    preserve_segment_endpoints: bool = False,
) -> np.ndarray:
    """Hold each continuous support interval at its touchdown position."""

    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    if (
        feet.ndim != 3
        or feet.shape[1:] != (2, 3)
        or len(feet) < 1
        or support.shape != feet.shape[:2]
        or support.dtype != np.bool_
        or type(preserve_segment_endpoints) is not bool
        or not np.isfinite(feet).all()
        or not bool(support.any(axis=1).all())
    ):
        raise ContractError("stance segment anchor input is invalid")
    anchors = np.full_like(feet, np.nan)
    for foot in range(2):
        starts = np.flatnonzero(
            support[:, foot]
            & np.concatenate(
                (np.ones(1, dtype=np.bool_), ~support[:-1, foot])
            )
        )
        stops = np.flatnonzero(
            support[:, foot]
            & np.concatenate(
                (~support[1:, foot], np.ones(1, dtype=np.bool_))
            )
        )
        for start, stop in zip(starts, stops):
            count = int(stop - start + 1)
            if preserve_segment_endpoints and count > 1:
                alpha = np.linspace(0.0, 1.0, count)
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                anchors[start : stop + 1, foot] = (
                    (1.0 - smooth[:, None]) * feet[start, foot]
                    + smooth[:, None] * feet[stop, foot]
                )
            else:
                anchors[start : stop + 1, foot] = feet[start, foot]
    return anchors


def raise_penetrating_stance_anchors(
    *,
    stance_anchor_world: object,
    support_mask: object,
    minimum_sole_clearance_by_foot_m: object,
    penetration_threshold_m: float = -0.025,
    clearance_target_m: float = -0.020,
) -> np.ndarray:
    """Lift an entire support interval when its touchdown penetrates terrain."""

    anchors = np.asarray(stance_anchor_world, dtype=np.float64)
    support = np.asarray(support_mask)
    clearance = np.asarray(
        minimum_sole_clearance_by_foot_m, dtype=np.float64
    )
    if (
        anchors.ndim != 3
        or anchors.shape[1:] != (2, 3)
        or support.shape != anchors.shape[:2]
        or support.dtype != np.bool_
        or clearance.shape != support.shape
        or not bool(support.any(axis=1).all())
        or not np.isfinite(anchors[support]).all()
        or not np.isfinite(clearance).all()
        or not math.isfinite(float(penetration_threshold_m))
        or not math.isfinite(float(clearance_target_m))
        or penetration_threshold_m >= clearance_target_m
    ):
        raise ContractError("stance anchor clearance input is invalid")
    output = anchors.copy()
    for foot in range(2):
        start: int | None = None
        for frame in range(len(anchors)):
            if support[frame, foot] and start is None:
                start = frame
                if clearance[frame, foot] < penetration_threshold_m:
                    lift = clearance_target_m - clearance[frame, foot]
                    output[frame, foot, 2] += lift
            elif support[frame, foot] and start is not None:
                output[frame, foot, 2] = output[start, foot, 2]
            else:
                start = None
    return output


def contact_repair_targets(
    *,
    foot_position_world: object,
    support_mask: object,
    stance_anchor_world: object,
    minimum_sole_clearance_by_foot_m: object,
    swing_clearance_margin_m: float,
    maximum_swing_lift_per_solve_m: float,
    swing_collision_threshold_m: float = -0.020,
) -> tuple[np.ndarray, np.ndarray]:
    """Constrain stance anchors and incrementally lift colliding swing feet."""

    feet = np.asarray(foot_position_world, dtype=np.float64)
    support = np.asarray(support_mask)
    anchors = np.asarray(stance_anchor_world, dtype=np.float64)
    clearance = np.asarray(
        minimum_sole_clearance_by_foot_m, dtype=np.float64
    )
    scalar_values = (
        swing_clearance_margin_m,
        maximum_swing_lift_per_solve_m,
        swing_collision_threshold_m,
    )
    if (
        feet.shape != (2, 3)
        or support.shape != (2,)
        or support.dtype != np.bool_
        or not bool(support.any())
        or anchors.shape != (2, 3)
        or clearance.shape != (2,)
        or not np.isfinite(feet).all()
        or not np.isfinite(anchors[support]).all()
        or not np.isfinite(clearance).all()
        or any(not math.isfinite(float(value)) for value in scalar_values)
        or swing_clearance_margin_m < 0.0
        or maximum_swing_lift_per_solve_m <= 0.0
        or swing_collision_threshold_m >= swing_clearance_margin_m
    ):
        raise ContractError("contact repair target input is invalid")
    colliding_swing = (~support) & (
        clearance < float(swing_collision_threshold_m)
    )
    targets = feet.copy()
    targets[support] = anchors[support]
    targets[colliding_swing, 2] += np.minimum(
        float(maximum_swing_lift_per_solve_m),
        float(swing_clearance_margin_m) - clearance[colliding_swing],
    )
    return support | colliding_swing, targets


def nearest_valid_sole_translation(
    *,
    sole_point_scene_xy: object,
    support_surface_height_m: float,
    sample_height: Callable[[np.ndarray], np.ndarray],
    maximum_shift_m: float = 0.08,
    search_resolution_m: float = 0.01,
    surface_tolerance_m: float = 0.025,
    safety_margin_m: float = 0.0,
) -> np.ndarray:
    """Find the smallest XY shift that puts an entire sole on one surface."""

    sole = np.asarray(sole_point_scene_xy, dtype=np.float64)
    scalar_values = (
        support_surface_height_m,
        maximum_shift_m,
        search_resolution_m,
        surface_tolerance_m,
        safety_margin_m,
    )
    if (
        sole.ndim != 2
        or sole.shape[1:] != (2,)
        or len(sole) < 3
        or not np.isfinite(sole).all()
        or any(not math.isfinite(float(value)) for value in scalar_values)
        or maximum_shift_m <= 0.0
        or search_resolution_m <= 0.0
        or search_resolution_m > maximum_shift_m
        or surface_tolerance_m <= 0.0
        or safety_margin_m < 0.0
        or not callable(sample_height)
    ):
        raise ContractError("sole-footprint search input is invalid")
    steps = int(math.ceil(maximum_shift_m / search_resolution_m))
    values = np.arange(-steps, steps + 1, dtype=np.float64)
    values *= float(search_resolution_m)
    dx, dy = np.meshgrid(values, values, indexing="ij")
    offsets = np.stack((dx.ravel(), dy.ravel()), axis=1)
    norm = np.linalg.norm(offsets, axis=1)
    offsets = offsets[norm <= maximum_shift_m + 1.0e-12]
    norm = np.linalg.norm(offsets, axis=1)
    order = np.lexsort(
        (
            offsets[:, 1],
            offsets[:, 0],
            np.abs(offsets[:, 1]),
            np.abs(offsets[:, 0]),
            norm,
        )
    )
    offsets = offsets[order]
    points = sole[None, :, :] + offsets[:, None, :]
    height = np.asarray(sample_height(points), dtype=np.float64)
    if (
        height.shape != points.shape[:-1]
        or not np.isfinite(height).all()
    ):
        raise ContractError("sole-footprint terrain samples are invalid")
    relative = height - float(support_surface_height_m)
    minimum_support_points = max(2, int(math.ceil(0.4 * len(sole))))
    if safety_margin_m > 0.0:
        margin = float(safety_margin_m)
        expanded = np.concatenate(
            [
                points + delta
                for delta in (
                    np.array((margin, 0.0)),
                    np.array((-margin, 0.0)),
                    np.array((0.0, margin)),
                    np.array((0.0, -margin)),
                )
            ],
            axis=1,
        )
        expanded_height = np.asarray(
            sample_height(expanded), dtype=np.float64
        )
        if (
            expanded_height.shape != expanded.shape[:-1]
            or not np.isfinite(expanded_height).all()
        ):
            raise ContractError(
                "sole-footprint safety samples are invalid"
            )
        safe = (
            expanded_height - float(support_surface_height_m)
            <= float(surface_tolerance_m)
        ).all(axis=1)
    else:
        safe = np.ones(len(offsets), dtype=np.bool_)
    valid = (
        safe
        & (relative <= float(surface_tolerance_m)).all(axis=1)
        & (
            (np.abs(relative) <= float(surface_tolerance_m)).sum(axis=1)
            >= minimum_support_points
        )
    )
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        raise ContractError("no valid sole footprint exists nearby")
    return np.ascontiguousarray(offsets[int(candidates[0])])


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
        maximum_root_horizontal_deviation_m: float = 1.0e-6,
        maximum_target_error_m: float = 0.005,
        maximum_warm_joint_step_rad: float = 0.25,
        foot_position_scale: float = 100.0,
        foot_orientation_scale: float = 5.0,
        center_of_mass_scale: float = 10.0,
        hip_yaw_posture_scale: float = 0.003,
    ) -> None:
        if (
            not math.isfinite(float(maximum_joint_deviation_rad))
            or maximum_joint_deviation_rad <= 0.0
            or not math.isfinite(float(maximum_root_height_deviation_m))
            or maximum_root_height_deviation_m <= 0.0
            or not math.isfinite(
                float(maximum_root_horizontal_deviation_m)
            )
            or maximum_root_horizontal_deviation_m <= 0.0
            or not math.isfinite(float(maximum_target_error_m))
            or maximum_target_error_m <= 0.0
            or not math.isfinite(float(maximum_warm_joint_step_rad))
            or maximum_warm_joint_step_rad <= 0.0
            or not math.isfinite(float(foot_position_scale))
            or foot_position_scale <= 0.0
            or not math.isfinite(float(foot_orientation_scale))
            or foot_orientation_scale <= 0.0
            or not math.isfinite(float(center_of_mass_scale))
            or center_of_mass_scale <= 0.0
            or not math.isfinite(float(hip_yaw_posture_scale))
            or hip_yaw_posture_scale <= 0.0
        ):
            raise ContractError("horizontal terrain retarget bounds are invalid")
        self._kinematics = MujocoG1FootKinematics(g1_xml)
        self._joint_deviation = float(maximum_joint_deviation_rad)
        self._root_deviation = float(maximum_root_height_deviation_m)
        self._root_horizontal_deviation = float(
            maximum_root_horizontal_deviation_m
        )
        self._maximum_target_error = float(maximum_target_error_m)
        self._maximum_warm_joint_step = float(
            maximum_warm_joint_step_rad
        )
        self._foot_position_scale = float(foot_position_scale)
        self._foot_orientation_scale = float(foot_orientation_scale)
        self._center_of_mass_scale = float(center_of_mass_scale)
        self._hip_yaw_posture_scale = float(hip_yaw_posture_scale)

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
        enforce_target_error_feet: object | None = None,
        target_foot_position_world: object,
        target_foot_position_weights: object | None = None,
        target_foot_rotation_world: object | None = None,
        target_center_of_mass_world_xy: object | None = None,
        level_feet: object | None = None,
        initial_joint_position: object | None = None,
        initial_root_position_world: object | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joint_position, dtype=np.float64)
        root = np.asarray(root_position_world, dtype=np.float64)
        quaternion = np.asarray(
            root_orientation_world_wxyz, dtype=np.float64
        )
        mask = np.asarray(solve_feet)
        required_mask = (
            np.asarray(enforce_target_error_feet)
            if enforce_target_error_feet is not None
            else mask.copy()
        )
        targets = np.asarray(target_foot_position_world, dtype=np.float64)
        position_weights = (
            np.asarray(target_foot_position_weights, dtype=np.float64)
            if target_foot_position_weights is not None
            else np.ones(2, dtype=np.float64)
        )
        rotation_targets = (
            np.asarray(target_foot_rotation_world, dtype=np.float64)
            if target_foot_rotation_world is not None
            else None
        )
        center_of_mass_target = (
            np.asarray(
                target_center_of_mass_world_xy, dtype=np.float64
            )
            if target_center_of_mass_world_xy is not None
            else None
        )
        level_mask = (
            np.asarray(level_feet)
            if level_feet is not None
            else mask.copy()
        )
        has_initial = (
            initial_joint_position is not None
            or initial_root_position_world is not None
        )
        initial_joints = (
            np.asarray(initial_joint_position, dtype=np.float64)
            if initial_joint_position is not None
            else None
        )
        initial_root = (
            np.asarray(initial_root_position_world, dtype=np.float64)
            if initial_root_position_world is not None
            else None
        )
        if (
            joints.shape != (29,)
            or root.shape != (3,)
            or quaternion.shape != (4,)
            or mask.dtype != np.bool_
            or mask.shape != (2,)
            or not bool(mask.any())
            or required_mask.dtype != np.bool_
            or required_mask.shape != (2,)
            or not bool(required_mask.any())
            or bool((required_mask & ~mask).any())
            or targets.shape != (2, 3)
            or position_weights.shape != (2,)
            or not np.isfinite(position_weights).all()
            or bool((position_weights <= 0.0).any())
            or (
                center_of_mass_target is not None
                and (
                    center_of_mass_target.shape != (2,)
                    or not np.isfinite(center_of_mass_target).all()
                )
            )
            or (
                rotation_targets is not None
                and (
                    rotation_targets.shape != (2, 3, 3)
                    or not np.isfinite(rotation_targets).all()
                    or not np.allclose(
                        np.swapaxes(rotation_targets, 1, 2)
                        @ rotation_targets,
                        np.eye(3),
                        atol=1.0e-4,
                    )
                    or not np.allclose(
                        np.linalg.det(rotation_targets),
                        1.0,
                        atol=1.0e-4,
                    )
                )
            )
            or level_mask.dtype != np.bool_
            or level_mask.shape != (2,)
            or bool((level_mask & ~mask).any())
            or not all(
                np.isfinite(value).all()
                for value in (joints, root, quaternion, targets)
            )
            or (
                has_initial
                and (
                    initial_joints is None
                    or initial_root is None
                    or initial_joints.shape != (29,)
                    or initial_root.shape != (3,)
                    or not np.isfinite(initial_joints).all()
                    or not np.isfinite(initial_root).all()
                )
            )
        ):
            raise ContractError("horizontal terrain retarget frame is invalid")
        kinematics = self._kinematics
        data = kinematics._data
        model = kinematics._model
        mujoco = kinematics._mujoco
        data.qpos[:] = target_state_qpos(joints, root, quaternion)
        mujoco.mj_forward(model, data)
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
        orientation_targets = {
            int(index): (
                np.array((0.0, 0.0, 1.0), dtype=np.float64)
                if level_mask[index]
                else (
                    rotation_targets[index, :, 2].copy()
                    if rotation_targets is not None
                    else data.xmat[
                        kinematics._foot_body_ids[index]
                    ].reshape(3, 3)[:, 2].copy()
                )
            )
            for index in selected_feet
        }
        heading_targets = {}
        if rotation_targets is not None:
            for index in selected_feet:
                target_z = orientation_targets[int(index)]
                target_x = rotation_targets[index, :, 0].copy()
                target_x -= float(target_x @ target_z) * target_z
                norm = float(np.linalg.norm(target_x))
                if norm <= 1.0e-6:
                    raise ContractError(
                        "horizontal terrain retarget foot heading is invalid"
                    )
                heading_targets[int(index)] = target_x / norm
        seed_joints = data.qpos[selected_qpos].copy()
        seed = np.concatenate((seed_joints, root))
        lower = np.concatenate(
            (
                np.maximum(
                    selected_ranges[:, 0],
                    seed_joints - self._joint_deviation,
                ),
                [
                    root[0] - self._root_horizontal_deviation,
                    root[1] - self._root_horizontal_deviation,
                    root[2] - self._root_deviation,
                ],
            )
        )
        upper = np.concatenate(
            (
                np.minimum(
                    selected_ranges[:, 1],
                    seed_joints + self._joint_deviation,
                ),
                [
                    root[0] + self._root_horizontal_deviation,
                    root[1] + self._root_horizontal_deviation,
                    root[2] + self._root_deviation,
                ],
            )
        )
        if has_initial:
            initial_qpos = target_state_qpos(
                initial_joints, initial_root, quaternion
            )
            initial = np.concatenate(
                (initial_qpos[selected_qpos], initial_root)
            )
            base_lower = lower.copy()
            base_upper = upper.copy()
            warm_lower = lower.copy()
            warm_upper = upper.copy()
            warm_lower[:-3] = np.maximum(
                warm_lower[:-3],
                initial[:-3] - self._maximum_warm_joint_step,
            )
            warm_upper[:-3] = np.minimum(
                warm_upper[:-3],
                initial[:-3] + self._maximum_warm_joint_step,
            )
            root_step = np.array((0.02, 0.02, 0.02))
            warm_lower[-3:] = np.maximum(
                warm_lower[-3:], initial_root - root_step
            )
            warm_upper[-3:] = np.minimum(
                warm_upper[-3:], initial_root + root_step
            )
            disjoint = warm_lower > warm_upper
            lower = np.where(disjoint, base_lower, warm_lower)
            upper = np.where(disjoint, base_upper, warm_upper)
            collapsed = upper - lower < 2.0e-9
            if bool(collapsed.any()):
                middle = 0.5 * (lower[collapsed] + upper[collapsed])
                lower[collapsed] = middle - 1.0e-9
                upper[collapsed] = middle + 1.0e-9
        else:
            initial = seed.copy()
            for selected_order in range(len(selected_feet)):
                offset = selected_order * 6
                initial[offset] -= 0.075
                initial[offset + 3] += 0.15
                initial[offset + 4] -= 0.075
        initial = np.minimum(np.maximum(initial, lower), upper)
        continuity_target = initial[:-3].copy()
        continuity_scale = 0.3 if has_initial else 0.0
        posture_scale = np.full(len(selected_dofs), 0.003, dtype=np.float64)
        # Position-only IK otherwise uses extreme ankle pitch/roll to satisfy
        # unequal tread heights, tipping a sole through the neighboring tread.
        for selected_order in range(len(selected_feet)):
            posture_scale[selected_order * 6 + 2] = (
                self._hip_yaw_posture_scale
            )
            posture_scale[selected_order * 6 + 4 : selected_order * 6 + 6] = 0.3
        root_scale = np.array((0.1, 0.1, 0.01), dtype=np.float64)
        orientation_scales = self._foot_orientation_scale * np.where(
            position_weights >= 20.0, 2.0, 1.0
        )
        position_scales = self._foot_position_scale * position_weights

        def set_state(value: np.ndarray) -> None:
            data.qpos[selected_qpos] = value[:-3]
            data.qpos[:3] = value[-3:]
            mujoco.mj_forward(model, data)

        def residual(value: np.ndarray) -> np.ndarray:
            set_state(value)
            feet = np.concatenate(
                [
                    position_scales[index]
                    * (
                        data.xpos[kinematics._foot_body_ids[index]]
                        - targets[index]
                    )
                    for index in selected_feet
                ]
            )
            orientations = np.concatenate(
                [
                    orientation_scales[index]
                    * (
                        data.xmat[
                            kinematics._foot_body_ids[index]
                        ].reshape(3, 3)[:, 2]
                        - orientation_targets[int(index)]
                    )
                    for index in selected_feet
                ]
            )
            headings = (
                np.concatenate(
                    [
                        orientation_scales[index]
                        * (
                            data.xmat[
                                kinematics._foot_body_ids[index]
                            ].reshape(3, 3)[:, 0]
                            - heading_targets[int(index)]
                        )
                        for index in selected_feet
                    ]
                )
                if heading_targets
                else np.empty(0, dtype=np.float64)
            )
            center_of_mass = (
                self._center_of_mass_scale
                * (
                    data.subtree_com[kinematics._pelvis_body_id, :2]
                    - center_of_mass_target
                )
                if center_of_mass_target is not None
                else np.empty(0, dtype=np.float64)
            )
            values = [
                feet,
                orientations,
                headings,
                center_of_mass,
                posture_scale * (value[:-3] - seed_joints),
                root_scale * (value[-3:] - root),
            ]
            if has_initial:
                values.append(
                    continuity_scale * (value[:-3] - continuity_target)
                )
            return np.concatenate(values)

        def jacobian(value: np.ndarray) -> np.ndarray:
            set_state(value)
            foot_rows = []
            orientation_rows = []
            heading_rows = []
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
                    position_scales[index]
                    * np.concatenate(
                        (
                            row[:, selected_dofs],
                            row[:, :3],
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
                            -orientation_scales[index]
                            * skew
                            @ rotation_row[:, selected_dofs],
                            np.zeros((3, 3)),
                        ),
                        axis=1,
                    )
                )
                if heading_targets:
                    body_x = data.xmat[
                        kinematics._foot_body_ids[index]
                    ].reshape(3, 3)[:, 0]
                    x_skew = np.array(
                        (
                            (0.0, -body_x[2], body_x[1]),
                            (body_x[2], 0.0, -body_x[0]),
                            (-body_x[1], body_x[0], 0.0),
                        )
                    )
                    heading_rows.append(
                        np.concatenate(
                            (
                                -orientation_scales[index]
                                * x_skew
                                @ rotation_row[:, selected_dofs],
                                np.zeros((3, 3)),
                            ),
                            axis=1,
                        )
                    )
            posture = np.concatenate(
                (
                    np.diag(posture_scale),
                    np.zeros((len(selected_dofs), 3)),
                ),
                axis=1,
            )
            root_row = np.zeros((3, len(selected_dofs) + 3))
            root_row[:, -3:] = np.diag(root_scale)
            center_of_mass_rows = []
            if center_of_mass_target is not None:
                center_of_mass_jacobian = np.zeros(
                    (3, model.nv), dtype=np.float64
                )
                mujoco.mj_jacSubtreeCom(
                    model,
                    data,
                    center_of_mass_jacobian,
                    kinematics._pelvis_body_id,
                )
                center_of_mass_rows.append(
                    self._center_of_mass_scale
                    * np.concatenate(
                        (
                            center_of_mass_jacobian[
                                :2, selected_dofs
                            ],
                            center_of_mass_jacobian[:2, :3],
                        ),
                        axis=1,
                    )
                )
            rows = [
                *foot_rows,
                *orientation_rows,
                *heading_rows,
                *center_of_mass_rows,
                posture,
                root_row,
            ]
            if has_initial:
                rows.append(
                    np.concatenate(
                        (
                            continuity_scale
                            * np.eye(len(selected_dofs)),
                            np.zeros((len(selected_dofs), 3)),
                        ),
                        axis=1,
                    )
                )
            return np.concatenate(rows, axis=0)

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
        errors = {
            int(index): float(
                np.linalg.norm(
                    data.xpos[kinematics._foot_body_ids[index]]
                    - targets[index]
                )
            )
            for index in selected_feet
        }
        error_vectors = {
            int(index): (
                data.xpos[kinematics._foot_body_ids[index]]
                - targets[index]
            ).tolist()
            for index in selected_feet
        }
        required_errors = [
            errors[int(index)] for index in np.flatnonzero(required_mask)
        ]
        if max(required_errors) > self._maximum_target_error:
            raise ContractError(
                "horizontal terrain retarget target is unreachable: "
                f"{max(required_errors):.6f} m "
                f"(per-foot={errors}, delta={error_vectors})"
            )
        source_joints = np.asarray(data.qpos[7:], dtype=np.float64)
        target_joints = source_joints[
            np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION, dtype=np.int64)
        ]
        solved_root = np.asarray(result.x[-3:], dtype=np.float64)
        return (
            np.ascontiguousarray(target_joints),
            np.ascontiguousarray(solved_root),
        )
