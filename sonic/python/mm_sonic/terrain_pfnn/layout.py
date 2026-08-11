from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VectorLayout:
    fields: tuple[tuple[str, int], ...]

    @property
    def size(self) -> int:
        return sum(width for _, width in self.fields)

    def __getitem__(self, name: str) -> slice:
        offset = 0
        for field, width in self.fields:
            if field == name:
                return slice(offset, offset + width)
            offset += width
        raise KeyError(name)


INPUT_LAYOUT = VectorLayout((
    ("trajectory_position", 24),
    ("trajectory_direction", 24),
    ("terrain_height", 36),
    ("semantic_intent", 24),
    ("previous_body_position", 90),
    ("previous_body_velocity", 90),
))

CLASSIC_G1_INPUT_LAYOUT_V3 = VectorLayout((
    *INPUT_LAYOUT.fields,
    ("joint_position", 29),
    ("joint_velocity", 29),
))

OUTPUT_LAYOUT = VectorLayout((
    ("trajectory_position", 24),
    ("trajectory_direction", 24),
    ("body_position", 90),
    ("body_velocity", 90),
    ("root_height", 1),
    ("root_tilt", 2),
    ("joint_position", 29),
    ("root_planar_velocity", 2),
    ("root_yaw_velocity", 1),
    ("phase_advance", 1),
    ("contact_logit", 4),
))

assert INPUT_LAYOUT.size == 288
assert CLASSIC_G1_INPUT_LAYOUT_V3.size == 346
assert OUTPUT_LAYOUT.size == 268

TRAJECTORY_TIMES_S = np.array(
    [-1.0, -5/6, -2/3, -1/2, -1/3, -1/6, 0.0, 1/6, 1/3, 1/2, 2/3, 5/6],
    dtype=np.float64,
)
assert TRAJECTORY_TIMES_S.shape == (12,)

CONTACT_ORDER = ("left_heel", "left_toe", "right_heel", "right_toe")
