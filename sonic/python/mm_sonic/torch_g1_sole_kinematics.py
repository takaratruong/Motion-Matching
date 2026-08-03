"""Native-MuJoCo world samples for the complete G1 sole footprint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .joints import ContractError
from .torch_g1_fk import target_state_qpos


_FOOT_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_LOCAL_SOLE_POINTS_M = np.array(
    (
        (0.035, 0.0, -0.05),
        (0.12, 0.0, -0.05),
        (-0.05, 0.0, -0.05),
        (0.12, 0.04, -0.05),
        (0.12, -0.04, -0.05),
        (-0.05, 0.04, -0.05),
        (-0.05, -0.04, -0.05),
    ),
    dtype=np.float64,
)


class MujocoG1SoleKinematics:
    """Evaluate seven oriented sole samples per foot for emitted G1 states."""

    def __init__(self, g1_xml: str | Path) -> None:
        path = Path(g1_xml).resolve()
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"G1 XML is missing or invalid: {path}")
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_path(str(path))
            body_ids = np.asarray(
                [model.body(name).id for name in _FOOT_BODY_NAMES],
                dtype=np.int64,
            )
        except Exception as error:
            raise ContractError("failed to compile G1 sole kinematics") from error
        if model.nq != 36:
            raise ContractError("G1 sole kinematics requires nq=36")
        self._mujoco = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        self._body_ids = body_ids

    def sole_points(
        self,
        joint_positions: object,
        root_positions_world: object,
        root_orientations_world_wxyz: object,
    ) -> np.ndarray:
        joints = np.asarray(joint_positions, dtype=np.float64)
        roots = np.asarray(root_positions_world, dtype=np.float64)
        quaternions = np.asarray(
            root_orientations_world_wxyz, dtype=np.float64
        )
        if (
            joints.ndim != 2
            or joints.shape[1:] != (29,)
            or roots.shape != (joints.shape[0], 3)
            or quaternions.shape != (joints.shape[0], 4)
            or not np.isfinite(joints).all()
            or not np.isfinite(roots).all()
            or not np.isfinite(quaternions).all()
            or np.any(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0) > 1e-4)
        ):
            raise ContractError("G1 sole kinematics batch is invalid")
        output = np.empty((joints.shape[0], 2, 7, 3), dtype=np.float64)
        for frame in range(joints.shape[0]):
            self._data.qpos[:] = target_state_qpos(
                joints[frame], roots[frame], quaternions[frame]
            )
            self._mujoco.mj_forward(self._model, self._data)
            for foot, body in enumerate(self._body_ids):
                rotation = self._data.xmat[body].reshape(3, 3)
                output[frame, foot] = (
                    self._data.xpos[body]
                    + _LOCAL_SOLE_POINTS_M @ rotation.T
                )
        return output
