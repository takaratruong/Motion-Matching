"""Probe P1: the pure Motion-Matching root-direction scoring oracle.

These synthetic tests validate the scoring oracle only. They are not evidence
that live Motion Matching follows a command; the controller runs the real
four-reset MM probe separately.
"""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from mm_sonic.direction_probe import (
    REGISTERED_DIRECTION_AXES_MUJOCO,
    REGISTERED_MINIMUM_PROJECTION_M,
    signed_root_projection,
)
from mm_sonic.joints import ContractError, load_joint_contract
from mm_sonic.timeline import TargetTimeline
from tests.python.test_sonic_resample import (
    JOINT_CONTRACT,
    initial_from_chunk,
    make_source_chunk,
)


_TARGET_ROWS = 20


def base_target():
    contract = load_joint_contract(JOINT_CONTRACT)
    chunk = make_source_chunk(contract)
    timeline = TargetTimeline(initial_from_chunk(chunk), contract)
    return timeline.prepare(chunk).target


def target_with_root_path(per_frame_delta) -> object:
    """A target whose virtual root travels a fixed per-frame delta from origin."""
    delta = np.asarray(per_frame_delta, np.float64)
    steps = np.arange(_TARGET_ROWS, dtype=np.float64)[:, np.newaxis]
    path = (steps * delta[np.newaxis, :]).astype(np.float32)
    return replace(base_target(), virtual_root_position=path)


class SignedRootProjectionTests(unittest.TestCase):
    def test_backward_left_right_forward_have_expected_signed_root_projection(self):
        # A large horizontal step per direction plus deliberate z drift and a
        # small orthogonal leak that must not dominate the signed projection.
        magnitude = 0.02
        commanded = {
            "forward": (magnitude, 0.0),
            "backward": (-magnitude, 0.0),
            "left": (0.0, magnitude),
            "right": (0.0, -magnitude),
        }
        for direction, (dx, dy) in commanded.items():
            axis = REGISTERED_DIRECTION_AXES_MUJOCO[direction]
            minimum = REGISTERED_MINIMUM_PROJECTION_M[direction]
            target = target_with_root_path((dx, dy, -0.05))
            projection = signed_root_projection(target, axis)
            self.assertGreater(projection, 0.0, direction)
            self.assertGreaterEqual(projection, minimum, direction)

            # Vertical z displacement is ignored entirely.
            flat = target_with_root_path((dx, dy, 0.0))
            self.assertAlmostEqual(
                projection, signed_root_projection(flat, axis), places=5
            )

            # The orthogonal in-plane axis sees near-zero leakage.
            orthogonal = (-axis[1], axis[0], 0.0)
            self.assertAlmostEqual(
                signed_root_projection(target, orthogonal), 0.0, places=5
            )

    def test_neutral_after_direction_reduces_generated_travel(self):
        for direction, axis in REGISTERED_DIRECTION_AXES_MUJOCO.items():
            active = target_with_root_path(
                (axis[0] * 0.02, axis[1] * 0.02, 0.0)
            )
            neutral = target_with_root_path((0.0, 0.0, 0.0))
            self.assertGreater(
                signed_root_projection(active, axis),
                signed_root_projection(neutral, axis),
                direction,
            )

    def test_projection_normalizes_the_axis(self):
        target = target_with_root_path((0.02, 0.0, 0.0))
        unit = signed_root_projection(target, (1.0, 0.0, 0.0))
        scaled = signed_root_projection(target, (5.0, 0.0, 0.0))
        self.assertAlmostEqual(unit, scaled, places=6)

    def test_rejects_zero_or_non_finite_axis(self):
        target = target_with_root_path((0.02, 0.0, 0.0))
        for axis in ((0.0, 0.0, 0.0), (0.0, 0.0, 5.0)):
            with self.assertRaises(ContractError):
                signed_root_projection(target, axis)
        for axis in ((float("nan"), 0.0, 0.0), (float("inf"), 1.0, 0.0)):
            with self.assertRaises(ContractError):
                signed_root_projection(target, axis)
        with self.assertRaises(ContractError):
            signed_root_projection(target, (1.0, 0.0))

    def test_rejects_malformed_target(self):
        with self.assertRaises(ContractError):
            signed_root_projection(object(), (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
