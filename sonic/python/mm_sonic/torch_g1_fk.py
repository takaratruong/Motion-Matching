"""Authoritative native-MuJoCo forward kinematics for emitted G1 states."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
)


_G1_NQ = 36
_G1_JOINT_COUNT = 29
_FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
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
_IK_MAX_ITERATIONS = 32
_IK_DAMPING = 1.0e-4
_IK_MAX_STEP_RAD = 0.10
_IK_TOLERANCE_M = 0.005


def _numpy_float64(value: object) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().to("cpu").numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=np.float64)


def _finite_array(
    value: object,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = _numpy_float64(value)
    if array.shape != shape:
        raise ContractError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ContractError(f"{label} must be finite")
    return np.ascontiguousarray(array)


def target_state_qpos(
    joint_position: object,
    root_position_world: object,
    root_orientation_world_wxyz: object,
) -> np.ndarray:
    """Convert target-ordered emitted state into native G1 qpos ordering."""

    target_joints = _finite_array(
        joint_position, (_G1_JOINT_COUNT,), "joint position"
    )
    root = _finite_array(root_position_world, (3,), "root position")
    quaternion = _finite_array(
        root_orientation_world_wxyz, (4,), "root orientation"
    )
    if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-4:
        raise ContractError("root orientation must be unit length")

    source_joints = np.empty(_G1_JOINT_COUNT, dtype=np.float64)
    source_joints[
        np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION, dtype=np.int64)
    ] = target_joints
    return np.concatenate((root, quaternion, source_joints))


class MujocoG1FootKinematics:
    """Evaluate both G1 ankle-roll body origins for emitted states."""

    def __init__(self, g1_xml: str | Path) -> None:
        path = Path(g1_xml).resolve()
        if not path.is_file():
            raise ContractError(f"G1 XML is missing or invalid: {path}")
        try:
            import mujoco
        except ImportError as error:
            raise ContractError("G1 foot kinematics requires mujoco") from error
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception as error:
            raise ContractError(f"failed to compile G1 XML: {path}") from error
        self._initialize(mujoco, model)

    @classmethod
    def _from_compiled_model_for_test(cls, model: object):
        """Exercise compiled-model contract checks without compiling XML."""

        instance = cls.__new__(cls)
        instance._initialize(None, model)
        return instance

    def _initialize(self, mujoco_module: object, model: object) -> None:
        if getattr(model, "nq", None) != _G1_NQ:
            raise ContractError(
                f"G1 MuJoCo model nq must be {_G1_NQ}, got "
                f"{getattr(model, 'nq', None)}"
            )
        if mujoco_module is None:
            raise ContractError("compiled G1 model test double is incomplete")
        try:
            foot_body_ids = np.asarray(
                [model.body(name).id for name in _FOOT_BODY_NAMES],
                dtype=np.int64,
            )
        except Exception as error:
            raise ContractError("G1 model is missing ankle-roll foot bodies") from error
        self._mujoco = mujoco_module
        self._model = model
        self._data = mujoco_module.MjData(model)
        self._foot_body_ids = foot_body_ids
        try:
            leg_joint_ids = tuple(
                tuple(int(model.joint(name).id) for name in names)
                for names in _LEG_JOINT_NAMES
            )
        except Exception as error:
            raise ContractError("G1 model is missing leg joints") from error
        self._leg_qpos_addresses = tuple(
            np.asarray(
                [int(model.jnt_qposadr[index]) for index in joint_ids],
                dtype=np.int64,
            )
            for joint_ids in leg_joint_ids
        )
        self._leg_dof_addresses = tuple(
            np.asarray(
                [int(model.jnt_dofadr[index]) for index in joint_ids],
                dtype=np.int64,
            )
            for joint_ids in leg_joint_ids
        )
        self._leg_ranges = tuple(
            np.asarray(model.jnt_range[list(joint_ids)], dtype=np.float64)
            for joint_ids in leg_joint_ids
        )

    def foot_positions(
        self,
        joint_positions: object,
        root_positions_world: object,
        root_orientations_world_wxyz: object,
    ) -> np.ndarray:
        """Return world foot positions with shape ``(frames, 2, 3)``."""

        joints = _numpy_float64(joint_positions)
        roots = _numpy_float64(root_positions_world)
        quaternions = _numpy_float64(root_orientations_world_wxyz)
        if (
            joints.ndim != 2
            or joints.shape[1:] != (_G1_JOINT_COUNT,)
            or roots.ndim != 2
            or roots.shape[1:] != (3,)
            or quaternions.ndim != 2
            or quaternions.shape[1:] != (4,)
            or not (joints.shape[0] == roots.shape[0] == quaternions.shape[0])
        ):
            raise ContractError(
                "G1 FK batch shapes must be (frames,29), (frames,3), "
                "and (frames,4) with matching frame counts"
            )
        if not (
            np.isfinite(joints).all()
            and np.isfinite(roots).all()
            and np.isfinite(quaternions).all()
        ):
            raise ContractError("G1 FK batch values must be finite")
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-4):
            raise ContractError("G1 FK batch root orientations must be unit length")

        output = np.empty((joints.shape[0], 2, 3), dtype=np.float64)
        for frame in range(joints.shape[0]):
            self._data.qpos[:] = target_state_qpos(
                joints[frame], roots[frame], quaternions[frame]
            )
            self._mujoco.mj_forward(self._model, self._data)
            output[frame] = self._data.xpos[self._foot_body_ids]
        return output

    def solve_leg_positions(
        self,
        joint_position: object,
        root_position_world: object,
        root_orientation_world_wxyz: object,
        solve_feet: object,
        target_foot_position_world: object,
    ) -> np.ndarray:
        """Solve selected G1 legs to world ankle targets without moving root."""

        joints = _finite_array(
            joint_position, (_G1_JOINT_COUNT,), "joint position"
        )
        root = _finite_array(root_position_world, (3,), "root position")
        quaternion = _finite_array(
            root_orientation_world_wxyz, (4,), "root orientation"
        )
        if abs(float(np.linalg.norm(quaternion)) - 1.0) > 1e-4:
            raise ContractError("root orientation must be unit length")
        mask = np.asarray(solve_feet)
        if mask.dtype != np.bool_ or mask.shape != (2,) or not bool(mask.any()):
            raise ContractError(
                "G1 foot IK solve mask must be boolean shape (2,) and nonempty"
            )
        targets = _finite_array(
            target_foot_position_world, (2, 3), "G1 foot IK target"
        )

        self._data.qpos[:] = target_state_qpos(joints, root, quaternion)
        selected_feet = np.flatnonzero(mask)
        selected_dofs = np.concatenate(
            [self._leg_dof_addresses[index] for index in selected_feet]
        )
        selected_qpos = np.concatenate(
            [self._leg_qpos_addresses[index] for index in selected_feet]
        )
        selected_ranges = np.concatenate(
            [self._leg_ranges[index] for index in selected_feet], axis=0
        )

        for _ in range(_IK_MAX_ITERATIONS):
            self._mujoco.mj_forward(self._model, self._data)
            error = np.concatenate(
                [
                    targets[index]
                    - self._data.xpos[self._foot_body_ids[index]]
                    for index in selected_feet
                ]
            )
            if all(
                np.linalg.norm(error[offset : offset + 3])
                <= _IK_TOLERANCE_M
                for offset in range(0, error.size, 3)
            ):
                source = np.asarray(self._data.qpos[7:], dtype=np.float64)
                return np.ascontiguousarray(
                    source[
                        np.asarray(
                            PINNED_TARGET_TO_SOURCE_PERMUTATION,
                            dtype=np.int64,
                        )
                    ]
                )

            rows = []
            for index in selected_feet:
                jacobian = np.zeros((3, self._model.nv), np.float64)
                self._mujoco.mj_jacBody(
                    self._model,
                    self._data,
                    jacobian,
                    None,
                    int(self._foot_body_ids[index]),
                )
                rows.append(jacobian[:, selected_dofs])
            jacobian = np.concatenate(rows, axis=0)
            system = (
                jacobian @ jacobian.T
                + _IK_DAMPING * np.eye(error.size, dtype=np.float64)
            )
            try:
                delta = jacobian.T @ np.linalg.solve(system, error)
            except np.linalg.LinAlgError as error_value:
                raise ContractError(
                    "G1 foot IK target is unreachable"
                ) from error_value
            delta = np.clip(
                delta, -_IK_MAX_STEP_RAD, _IK_MAX_STEP_RAD
            )
            updated = np.clip(
                self._data.qpos[selected_qpos] + delta,
                selected_ranges[:, 0],
                selected_ranges[:, 1],
            )
            self._data.qpos[selected_qpos] = updated

        raise ContractError("G1 foot IK target is unreachable")
