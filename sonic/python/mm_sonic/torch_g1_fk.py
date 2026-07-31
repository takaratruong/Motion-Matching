"""Authoritative native-MuJoCo forward kinematics for emitted G1 states."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .joints import ContractError, PINNED_TARGET_TO_SOURCE_PERMUTATION


_G1_NQ = 36
_G1_JOINT_COUNT = 29
_FOOT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


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
