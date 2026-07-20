"""Pure Motion-Matching root-direction scoring oracle (Stage R1, Task 2).

`signed_root_projection` scores a committed :class:`TargetChunk` by projecting
its final-minus-initial virtual-root displacement onto a normalized horizontal
MuJoCo x-y axis. Vertical z displacement is ignored. The function is pure and
deterministic; the same oracle is reused to score live MM candidates.

The synthetic tests that exercise this module prove the scoring oracle only.
They are not evidence that live Motion Matching follows a command.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from .joints import ContractError


# Registered horizontal MuJoCo command axes. MuJoCo is x-forward, y-left,
# z-up, so backward and right are the negated forward and left axes.
REGISTERED_DIRECTION_AXES_MUJOCO = MappingProxyType(
    {
        "forward": (1.0, 0.0, 0.0),
        "backward": (-1.0, 0.0, 0.0),
        "left": (0.0, 1.0, 0.0),
        "right": (0.0, -1.0, 0.0),
    }
)

# Minimum signed projection (metres) a passing generation must reach along its
# commanded axis. A single 0.4 s chunk of deliberate travel clears this floor;
# neutral travel does not.
REGISTERED_MINIMUM_PROJECTION_M = MappingProxyType(
    {
        "forward": 0.05,
        "backward": 0.05,
        "left": 0.05,
        "right": 0.05,
    }
)


def _horizontal_unit_axis(axis_mujoco: object) -> np.ndarray:
    if type(axis_mujoco) not in (tuple, list) or len(axis_mujoco) != 3:
        raise ContractError("direction axis_mujoco must have width 3")
    values = []
    for item in axis_mujoco:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ContractError("direction axis_mujoco must contain finite values")
        values.append(float(item))
    horizontal = np.array([values[0], values[1], 0.0], np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ContractError(
            "direction axis_mujoco must have a nonzero horizontal component"
        )
    return horizontal / norm


def signed_root_projection(target: object, axis_mujoco: object) -> float:
    """Signed projection of virtual-root travel onto a horizontal axis.

    The displacement is the final virtual-root row minus the initial row of
    ``target.virtual_root_position``; only the x-y plane contributes.
    """
    axis = _horizontal_unit_axis(axis_mujoco)
    position = getattr(target, "virtual_root_position", None)
    if position is None:
        raise ContractError("direction target must expose virtual_root_position")
    path = np.asarray(position)
    if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 1:
        raise ContractError(
            "direction target virtual_root_position must be an (N, 3) path"
        )
    path = path.astype(np.float64)
    if not np.all(np.isfinite(path)):
        raise ContractError("direction target virtual_root_position must be finite")
    displacement = path[-1] - path[0]
    displacement[2] = 0.0
    return float(np.dot(displacement, axis))
