"""Pinned GEAR policy-action to received LowCmd target reconstruction."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


_ISAACLAB_TO_MUJOCO = (
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
)

_DEFAULT_ANGLES = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0,
    0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)

_NATURAL_FREQ = 10 * 2.0 * 3.1415926535
_STIFFNESS = {
    "5020": 0.003609725 * _NATURAL_FREQ * _NATURAL_FREQ,
    "7520_14": 0.010177520 * _NATURAL_FREQ * _NATURAL_FREQ,
    "7520_22": 0.025101925 * _NATURAL_FREQ * _NATURAL_FREQ,
    "4010": 0.00425 * _NATURAL_FREQ * _NATURAL_FREQ,
}
_EFFORT = {
    "5020": 25.0,
    "7520_14": 88.0,
    "7520_22": 139.0,
    "4010": 5.0,
}
_MOTOR_KINDS = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020", "5020", "5020", "5020", "5020",
    "5020", "4010", "4010", "5020", "5020", "5020", "5020",
    "5020", "4010", "4010",
)
_ACTION_SCALES = tuple(
    0.25 * _EFFORT[kind] / _STIFFNESS[kind] for kind in _MOTOR_KINDS
)


def policy_action_to_lowcmd_target(
    action: Sequence[float],
) -> tuple[float, ...]:
    """Reproduce pinned C++ double arithmetic and final float32 target cast."""

    if len(action) != 29:
        raise ValueError("policy action must contain exactly 29 values")
    action32: list[np.float32] = []
    for index, value in enumerate(action):
        if type(value) not in (int, float, np.float32, np.float64):
            raise ValueError(f"policy action[{index}] must be finite numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"policy action[{index}] must be finite numeric")
        action32.append(np.float32(number))
    return tuple(
        float(
            np.float32(
                _DEFAULT_ANGLES[target_index]
                + float(action32[source_index]) * _ACTION_SCALES[target_index]
            )
        )
        for target_index, source_index in enumerate(_ISAACLAB_TO_MUJOCO)
    )


__all__ = ["policy_action_to_lowcmd_target"]
