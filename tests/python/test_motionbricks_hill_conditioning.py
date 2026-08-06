from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from mm_sonic.motionbricks_hill import GentleHillProfile
from mm_sonic.motionbricks_hill_conditioning import (
    MotionBricksHillConditioner,
    canonical_targets_to_world_xy,
    terrain_height_deltas,
)


class CoordinateConversionTest(unittest.TestCase):
    def test_swaps_motion_planar_axes_at_zero_heading(self) -> None:
        targets = np.asarray(((2.0, 4.0), (1.0, 3.0)))
        world = canonical_targets_to_world_xy(targets, 0.0, (10.0, 20.0))
        np.testing.assert_allclose(world, ((11.0, 22.0), (13.0, 24.0)))

    def test_rotates_targets_by_canonicalization_heading(self) -> None:
        targets = np.asarray(((2.0,), (1.0,)))
        world = canonical_targets_to_world_xy(
            targets, 0.5 * math.pi, (10.0, 20.0)
        )
        np.testing.assert_allclose(world, ((8.0, 21.0),), atol=1.0e-12)

    def test_terrain_deltas_are_relative_to_current_support(self) -> None:
        hill = GentleHillProfile()
        current = (1.0, 0.0)
        targets = np.asarray(((2.0, 0.0), (5.0, 0.0), (8.0, 0.0), (9.0, 0.0)))
        deltas = terrain_height_deltas(current, targets, hill.height)
        expected = np.asarray([hill.height(target) for target in targets])
        np.testing.assert_allclose(deltas, expected)
        self.assertGreater(deltas[0], 0.0)
        self.assertGreater(deltas[1], deltas[0])
        self.assertLess(deltas[2], deltas[1])
        self.assertEqual(deltas[3], 0.0)


class _FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def _generate_target_joint_transforms(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls += 1
        joint_positions = torch.zeros((1, 4, 2, 3), dtype=torch.float32)
        joint_rotations = torch.zeros((1, 4, 2, 3, 3), dtype=torch.float32)
        root_positions = torch.tensor(
            [[[0.0, 0.80, 0.0],
              [0.0, 0.84, 0.0],
              [0.0, 0.79, 0.0],
              [0.0, 0.82, 0.0]]],
            dtype=torch.float32,
        )
        return joint_positions, joint_rotations, root_positions


def _inputs(
    *,
    target_root_positions: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    raw_context = torch.zeros((1, 4, 36), dtype=torch.float32)
    raw_context[0, -1, :2] = torch.tensor((1.0, 0.0))
    return {
        "target_root_positions": (
            torch.tensor(
                [[[0.0, 0.0, 0.0, 0.0], [2.0, 5.0, 8.0, 9.0]]],
                dtype=torch.float32,
            )
            if target_root_positions is None
            else target_root_positions
        ),
        "first_frame_heading_angle": torch.tensor((0.0,), dtype=torch.float32),
        "first_frame_position": torch.tensor(
            ((0.0, 0.0, 0.0),), dtype=torch.float32
        ),
        "raw_context_mujoco_qpos": raw_context,
    }


class MotionBricksHillConditionerTest(unittest.TestCase):
    def test_hook_adds_deltas_without_replacing_clip_bob(self) -> None:
        hill = GentleHillProfile()
        agent = _FakeAgent()
        conditioner = MotionBricksHillConditioner(hill.height)
        conditioner.install(agent)

        _, _, conditioned = agent._generate_target_joint_transforms(_inputs())

        original_vertical = np.asarray((0.80, 0.84, 0.79, 0.82))
        expected_delta = np.asarray(
            [hill.height((x, 0.0)) for x in (2.0, 5.0, 8.0, 9.0)]
        )
        np.testing.assert_allclose(
            conditioned[0, :, 1].numpy(),
            original_vertical + expected_delta,
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            np.diff(conditioned[0, :, 1].numpy() - expected_delta),
            np.diff(original_vertical),
            atol=1.0e-6,
        )
        self.assertIsNotNone(conditioner.latest_trace)
        np.testing.assert_allclose(
            conditioner.latest_trace.height_deltas, expected_delta
        )

    def test_remove_restores_unconditioned_method(self) -> None:
        agent = _FakeAgent()
        conditioner = MotionBricksHillConditioner(lambda xy: float(xy[0]))
        conditioner.install(agent)
        conditioner.remove()

        _, _, roots = agent._generate_target_joint_transforms(_inputs())

        np.testing.assert_allclose(
            roots[0, :, 1].numpy(), (0.80, 0.84, 0.79, 0.82)
        )
        self.assertFalse(conditioner.is_installed)

    def test_rejects_unexpected_batch_shape(self) -> None:
        agent = _FakeAgent()
        conditioner = MotionBricksHillConditioner(lambda xy: 0.0)
        conditioner.install(agent)
        bad = torch.zeros((2, 2, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "batch"):
            agent._generate_target_joint_transforms(
                _inputs(target_root_positions=bad)
            )

    def test_rejects_unexpected_target_frame_count(self) -> None:
        agent = _FakeAgent()
        conditioner = MotionBricksHillConditioner(lambda xy: 0.0)
        conditioner.install(agent)
        bad = torch.zeros((1, 2, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "target frame"):
            agent._generate_target_joint_transforms(
                _inputs(target_root_positions=bad)
            )


if __name__ == "__main__":
    unittest.main()
