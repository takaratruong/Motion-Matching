from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import unittest

from mm_sonic.commands import CommandSample
from mm_sonic.joints import ContractError
from mm_sonic.operator import OperatorLimits, OperatorSampler, OperatorState


class OperatorValueTests(unittest.TestCase):
    def test_state_and_limits_are_frozen_validated_values(self) -> None:
        state = OperatorState()
        self.assertEqual(
            (
                state.forward,
                state.backward,
                state.left,
                state.right,
                state.heading_left,
                state.heading_right,
                state.stand,
                state.terminate,
            ),
            (False,) * 8,
        )
        with self.assertRaises(FrozenInstanceError):
            state.forward = True

        limits = OperatorLimits(
            forward_mps=1.0,
            backward_mps=0.4,
            lateral_mps=0.3,
            heading_step_rad=math.radians(5.0),
        )
        with self.assertRaises(FrozenInstanceError):
            limits.forward_mps = 2.0

        for field, value in (
            ("forward_mps", 0.0),
            ("backward_mps", -1.0),
            ("lateral_mps", math.inf),
            ("heading_step_rad", True),
        ):
            fields = {
                "forward_mps": 1.0,
                "backward_mps": 0.4,
                "lateral_mps": 0.3,
                "heading_step_rad": 0.1,
            }
            fields[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ContractError, field):
                    OperatorLimits(**fields)

        with self.assertRaisesRegex(ContractError, "boolean"):
            OperatorState(forward=1)


class OperatorSamplerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = OperatorLimits(
            forward_mps=0.5,
            backward_mps=0.4,
            lateral_mps=0.3,
            heading_step_rad=math.radians(5.0),
        )
        self.sampler = OperatorSampler(self.limits)

    def assertTupleClose(
        self, actual: tuple[float, ...], expected: tuple[float, ...]
    ) -> None:
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected, strict=True):
            self.assertAlmostEqual(left, right, places=12)

    def test_update_queues_only_the_newest_complete_state(self) -> None:
        self.assertIsNone(self.sampler.update(OperatorState(forward=True)))
        self.assertIsNone(
            self.sampler.update(OperatorState(backward=True, right=True))
        )
        command = self.sampler.sample_boundary(0)
        self.assertIsInstance(command, CommandSample)
        assert command is not None
        component = self.limits.backward_mps / math.sqrt(2.0)
        self.assertTupleClose(
            command.requested_velocity_mujoco,
            (-component, -component, 0.0),
        )

        with self.assertRaisesRegex(ContractError, "OperatorState"):
            self.sampler.update(object())
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ContractError, "chunk_index"):
                    self.sampler.sample_boundary(value)

    def test_diagonals_preserve_direction_and_obey_both_speed_caps(self) -> None:
        self.sampler.update(OperatorState(forward=True, left=True))
        command = self.sampler.sample_boundary(0)
        assert command is not None
        self.assertTupleClose(
            command.requested_velocity_mujoco,
            (self.limits.lateral_mps, self.limits.lateral_mps, 0.0),
        )
        self.assertLessEqual(
            math.hypot(*command.requested_velocity_mujoco[:2]),
            self.limits.forward_mps,
        )

        self.sampler.update(OperatorState(left=True))
        lateral = self.sampler.sample_boundary(1)
        assert lateral is not None
        self.assertEqual(
            lateral.requested_velocity_mujoco,
            (0.0, self.limits.lateral_mps, 0.0),
        )

    def test_conflicts_cancel_and_stand_overrides_translation(self) -> None:
        self.sampler.update(
            OperatorState(
                forward=True,
                backward=True,
                left=True,
                right=True,
                heading_left=True,
                heading_right=True,
            )
        )
        cancelled = self.sampler.sample_boundary(0)
        assert cancelled is not None
        self.assertEqual(cancelled.requested_velocity_mujoco, (0.0, 0.0, 0.0))
        self.assertEqual(cancelled.desired_heading_mujoco_wxyz, (1.0, 0.0, 0.0, 0.0))

        self.sampler.update(OperatorState(forward=True, stand=True))
        standing = self.sampler.sample_boundary(1)
        assert standing is not None
        self.assertEqual(standing.requested_velocity_mujoco, (0.0, 0.0, 0.0))

    def test_heading_changes_once_per_boundary_and_persists(self) -> None:
        self.sampler.update(OperatorState(heading_left=True, stand=True))
        turned = self.sampler.sample_boundary(0)
        assert turned is not None
        half = 0.5 * self.limits.heading_step_rad
        self.assertTupleClose(
            turned.desired_heading_mujoco_wxyz,
            (math.cos(half), 0.0, 0.0, math.sin(half)),
        )

        held = self.sampler.sample_boundary(1)
        assert held is not None
        self.assertEqual(
            held.desired_heading_mujoco_wxyz,
            turned.desired_heading_mujoco_wxyz,
        )
        self.sampler.update(OperatorState(heading_right=True))
        returned = self.sampler.sample_boundary(2)
        assert returned is not None
        self.assertTupleClose(
            returned.desired_heading_mujoco_wxyz,
            (1.0, 0.0, 0.0, 0.0),
        )

    def test_termination_is_sticky(self) -> None:
        self.sampler.update(OperatorState(terminate=True))
        self.assertIsNone(self.sampler.sample_boundary(0))
        self.sampler.update(OperatorState(forward=True))
        self.assertIsNone(self.sampler.sample_boundary(1))


if __name__ == "__main__":
    unittest.main()
