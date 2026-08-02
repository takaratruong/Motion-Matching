"""Native quality descriptors for immutable terrain contact actions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .torch_contact_oracle_actions import (
    ContactPhaseAction,
    ContactPhaseActionIndex,
)


def _readonly_vector(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeActionQuality:
    action_index: int
    source_key: tuple[int, int, int, bool]
    entry_support: tuple[bool, bool]
    swing_foot: int
    landing_frame_offset: int
    natural_landing_local_xyz: np.ndarray
    root_displacement_local_xy: np.ndarray
    root_yaw_delta_rad: float
    source_stance_drift_m: float
    minimum_swing_clearance_m: float
    entry_joint_speed_norm: float
    terminal_joint_speed_norm: float
    exact_successor_index: int | None

    def __post_init__(self) -> None:
        if (
            type(self.action_index) is not int
            or self.action_index < 0
            or not isinstance(self.source_key, tuple)
            or len(self.source_key) != 4
            or any(type(value) is not int or value < 0 for value in self.source_key[:3])
            or type(self.source_key[3]) is not bool
            or not isinstance(self.entry_support, tuple)
            or len(self.entry_support) != 2
            or any(type(value) is not bool for value in self.entry_support)
            or self.swing_foot not in (0, 1)
            or type(self.landing_frame_offset) is not int
            or self.landing_frame_offset < 1
            or (
                self.exact_successor_index is not None
                and (
                    type(self.exact_successor_index) is not int
                    or self.exact_successor_index < 0
                )
            )
        ):
            raise ValueError("native action quality identity is invalid")
        object.__setattr__(
            self,
            "natural_landing_local_xyz",
            _readonly_vector(
                self.natural_landing_local_xyz,
                (3,),
                "natural landing",
            ),
        )
        object.__setattr__(
            self,
            "root_displacement_local_xy",
            _readonly_vector(
                self.root_displacement_local_xy,
                (2,),
                "root displacement",
            ),
        )
        for name in (
            "root_yaw_delta_rad",
            "source_stance_drift_m",
            "minimum_swing_clearance_m",
            "entry_joint_speed_norm",
            "terminal_joint_speed_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (
                    name
                    in {
                        "source_stance_drift_m",
                        "entry_joint_speed_norm",
                        "terminal_joint_speed_norm",
                    }
                    and float(value) < 0.0
                )
            ):
                raise ValueError(f"native action quality {name} is invalid")
            object.__setattr__(self, name, float(value))


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def describe_native_action_quality(
    action_index: int,
    action: ContactPhaseAction,
    exact_successor_index: int | None,
) -> NativeActionQuality:
    """Describe natural landing and contact quality without scene composition."""

    if type(action_index) is not int or action_index < 0:
        raise ValueError("native action index must be non-negative")
    if not isinstance(action, ContactPhaseAction):
        raise ValueError("native action quality requires ContactPhaseAction")
    if exact_successor_index is not None and (
        type(exact_successor_index) is not int or exact_successor_index < 0
    ):
        raise ValueError("native action successor index is invalid")

    feet = _cpu(action.foot_position_local).astype(np.float64, copy=False)
    support = _cpu(action.support_mask).astype(bool, copy=False)
    if action.frame_count > 1:
        horizontal = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2)
        stance = support[:-1] & support[1:]
        stance_drift = float(np.where(stance, horizontal, 0.0).sum())
    else:
        stance_drift = 0.0
    root = _cpu(action.root_position_local).astype(np.float64, copy=False)
    yaw = _cpu(action.root_yaw_local).astype(np.float64, copy=False)
    yaw_delta = math.atan2(
        math.sin(float(yaw[-1] - yaw[0])),
        math.cos(float(yaw[-1] - yaw[0])),
    )
    velocity = action.joint_velocity
    return NativeActionQuality(
        action_index=action_index,
        source_key=action.source_key,
        entry_support=tuple(bool(value) for value in action.entry_support.tolist()),
        swing_foot=action.swing_foot,
        landing_frame_offset=action.frame_count - 1,
        natural_landing_local_xyz=feet[-1, action.swing_foot],
        root_displacement_local_xy=root[-1, :2] - root[0, :2],
        root_yaw_delta_rad=yaw_delta,
        source_stance_drift_m=stance_drift,
        minimum_swing_clearance_m=action.minimum_swing_clearance_m,
        entry_joint_speed_norm=float(torch.linalg.vector_norm(velocity[0]).item()),
        terminal_joint_speed_norm=float(
            torch.linalg.vector_norm(velocity[-1]).item()
        ),
        exact_successor_index=exact_successor_index,
    )


def build_native_quality_index(
    index: ContactPhaseActionIndex,
) -> tuple[NativeActionQuality, ...]:
    """Describe every action in stable inventory order."""

    if not isinstance(index, ContactPhaseActionIndex):
        raise ValueError("native quality index requires ContactPhaseActionIndex")
    return tuple(
        describe_native_action_quality(
            action_index,
            action,
            index.exact_successor_indices[action_index],
        )
        for action_index, action in enumerate(index.actions)
    )
