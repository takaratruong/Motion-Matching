from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch
import mm_sonic.terrain_pfnn as terrain_pfnn

from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT, TRAJECTORY_TIMES_S
from mm_sonic.terrain_pfnn.recurrence import (
    PlannedTrajectory,
    RecurrentTrajectoryState,
    advance_recurrent_state,
    derive_training_desired_velocity,
    initialize_recurrent_state,
    pack_recurrent_input,
    plan_recurrent_trajectory,
)


class TerrainPFNNRecurrenceTests(unittest.TestCase):
    dtype = torch.float64

    def make_state(self) -> RecurrentTrajectoryState:
        batch_size = 2
        trajectory_position = torch.zeros(
            (batch_size, 12, 2), dtype=self.dtype
        )
        future_time = torch.as_tensor(
            TRAJECTORY_TIMES_S[6:], dtype=self.dtype
        )
        trajectory_position[:, 6:, 0] = 0.12 * future_time
        trajectory_direction = torch.zeros_like(trajectory_position)
        trajectory_direction[..., 0] = 1.0
        semantic = torch.zeros_like(trajectory_position)
        semantic[..., 0] = 1.0
        return initialize_recurrent_state(
            trajectory_position_local=trajectory_position,
            trajectory_direction_local=trajectory_direction,
            semantic_intent=semantic,
            previous_body_position_local=torch.zeros(
                (batch_size, 30, 3), dtype=self.dtype
            ),
            previous_body_velocity_local=torch.zeros(
                (batch_size, 30, 3), dtype=self.dtype
            ),
            phase=torch.tensor((0.25, 2.0 * math.pi - 0.05), dtype=self.dtype),
            root_world_xy=torch.zeros((batch_size, 2), dtype=self.dtype),
            root_yaw_world=torch.zeros(batch_size, dtype=self.dtype),
        )

    def make_physical_output(self) -> torch.Tensor:
        output = torch.zeros((2, OUTPUT_LAYOUT.size), dtype=self.dtype)
        local_position = output[
            :, OUTPUT_LAYOUT["trajectory_position"]
        ].reshape(2, 12, 2)
        local_position[..., 0] = torch.linspace(-0.2, 0.9, 12, dtype=self.dtype)
        local_position[..., 1] = torch.tensor((0.1, -0.2), dtype=self.dtype)[:, None]
        local_direction = output[
            :, OUTPUT_LAYOUT["trajectory_direction"]
        ].reshape(2, 12, 2)
        local_direction[..., 0] = 1.0
        output[:, OUTPUT_LAYOUT["root_planar_velocity"]] = torch.tensor(
            (0.3, 0.0), dtype=self.dtype
        )
        output[:, OUTPUT_LAYOUT["root_yaw_velocity"]] = 0.6
        output[:, OUTPUT_LAYOUT["phase_advance"]] = 0.1
        output[:, OUTPUT_LAYOUT["body_position"]] = 1.5
        output[:, OUTPUT_LAYOUT["body_velocity"]] = -2.0
        return output

    def test_initialize_reconstructs_thirty_one_history_samples(self) -> None:
        position = torch.zeros((1, 12, 2), dtype=self.dtype)
        position[0, :7, 0] = torch.arange(7, dtype=self.dtype)
        direction = torch.zeros_like(position)
        direction[..., 0] = 1.0
        direction[0, 1, :2] = torch.tensor((0.0, 1.0), dtype=self.dtype)
        semantic = torch.zeros_like(position)
        semantic[..., 0] = 1.0
        semantic[:, 1:3] = torch.tensor((0.0, 1.0), dtype=self.dtype)
        yaw = torch.tensor((math.pi / 2.0,), dtype=self.dtype)

        state = initialize_recurrent_state(
            trajectory_position_local=position,
            trajectory_direction_local=direction,
            semantic_intent=semantic,
            previous_body_position_local=torch.zeros((1, 30, 3), dtype=self.dtype),
            previous_body_velocity_local=torch.zeros((1, 30, 3), dtype=self.dtype),
            phase=torch.tensor((0.4,), dtype=self.dtype),
            root_world_xy=torch.tensor(((2.0, -1.0),), dtype=self.dtype),
            root_yaw_world=yaw,
        )

        self.assertEqual(state.history_position_world_xy.shape, (1, 31, 2))
        torch.testing.assert_close(
            state.history_position_world_xy[0, 1],
            torch.tensor((2.0, -0.8), dtype=self.dtype),
        )
        torch.testing.assert_close(
            torch.linalg.vector_norm(state.history_direction_world_xy, dim=-1),
            torch.ones((1, 31), dtype=self.dtype),
        )
        self.assertEqual(state.history_semantic_intent[0, 2].tolist(), [1.0, 0.0])
        self.assertEqual(state.history_semantic_intent[0, 3].tolist(), [0.0, 1.0])

    def test_planner_is_deterministic_and_derives_future_semantics(self) -> None:
        state = self.make_state()
        desired_velocity_world = torch.tensor(
            ((0.0, 0.0), (0.3, 0.0)), dtype=self.dtype
        )

        planned = plan_recurrent_trajectory(state, desired_velocity_world)

        self.assertEqual(planned.position_world_xy.shape, (2, 12, 2))
        self.assertEqual(planned.direction_world_xy.shape, (2, 12, 2))
        self.assertEqual(planned.semantic_intent.shape, (2, 12, 2))
        torch.testing.assert_close(planned.position_world_xy[:, 6], state.root_world_xy)
        torch.testing.assert_close(
            torch.linalg.vector_norm(planned.direction_world_xy[:, 7:], dim=-1),
            torch.ones((2, 5), dtype=torch.float64),
        )
        self.assertEqual(planned.semantic_intent[0, 11].tolist(), [1.0, 0.0])
        self.assertEqual(planned.semantic_intent[1, 11].tolist(), [0.0, 1.0])
        weight = 1.0 - math.sqrt(1.0 - 0.2)
        expected_velocity = torch.tensor(
            ((1.0 - weight) * 0.12, (1.0 - weight) * 0.12 + weight * 0.3),
            dtype=self.dtype,
        )
        torch.testing.assert_close(planned.position_world_xy[:, 7, 0], expected_velocity / 6.0)
        repeated = plan_recurrent_trajectory(state, desired_velocity_world)
        torch.testing.assert_close(repeated.position_world_xy, planned.position_world_xy)

    def test_planner_uses_published_direction_bias(self) -> None:
        state = self.make_state()
        desired_velocity_world = torch.tensor(
            ((0.0, 0.3), (0.0, 0.3)), dtype=self.dtype
        )

        planned = plan_recurrent_trajectory(state, desired_velocity_world)

        u = 0.2
        weight = 1.0 - (1.0 - u) ** 2
        expected = torch.tensor((1.0 - weight, weight), dtype=self.dtype)
        expected = expected / torch.linalg.vector_norm(expected)
        torch.testing.assert_close(planned.direction_world_xy[0, 7], expected)

    def test_training_command_uses_future_displacement_and_idle_semantic(self) -> None:
        position = torch.zeros((2, 12, 2), dtype=self.dtype)
        position[:, 11] = torch.tensor(((0.5, 0.0), (0.5, 0.0)), dtype=self.dtype)
        semantic = torch.zeros_like(position)
        semantic[..., 1] = 1.0
        semantic[0, 11] = torch.tensor((1.0, 0.0), dtype=self.dtype)
        yaw = torch.tensor((0.0, math.pi / 2.0), dtype=self.dtype)

        desired = derive_training_desired_velocity(position, semantic, yaw)

        torch.testing.assert_close(desired[0], torch.zeros(2, dtype=self.dtype))
        torch.testing.assert_close(desired[1], torch.tensor((0.0, 0.6), dtype=self.dtype))

    def test_packer_normalizes_once_and_scales_body_after_normalization(self) -> None:
        state = self.make_state()
        planned = plan_recurrent_trajectory(
            state, torch.tensor(((0.0, 0.0), (0.3, 0.0)), dtype=self.dtype)
        )
        state = replace(
            state,
            previous_body_position_local=torch.full((2, 30, 3), 3.0, dtype=self.dtype),
            previous_body_velocity_local=torch.full((2, 30, 3), -1.0, dtype=self.dtype),
        )
        terrain = torch.arange(72, dtype=self.dtype).reshape(2, 12, 3) / 10.0
        x_mean = torch.ones(INPUT_LAYOUT.size, dtype=self.dtype)
        x_std = torch.full((INPUT_LAYOUT.size,), 2.0, dtype=self.dtype)

        packed = pack_recurrent_input(
            state=state,
            planned=planned,
            terrain_height=terrain,
            x_mean=x_mean,
            x_std=x_std,
        )

        self.assertEqual(packed.shape, (2, INPUT_LAYOUT.size))
        torch.testing.assert_close(
            packed[:, INPUT_LAYOUT["previous_body_position"]],
            torch.full((2, 90), 0.1, dtype=self.dtype),
        )
        torch.testing.assert_close(
            packed[:, INPUT_LAYOUT["previous_body_velocity"]],
            torch.full((2, 90), -0.1, dtype=self.dtype),
        )
        torch.testing.assert_close(
            packed[:, INPUT_LAYOUT["terrain_height"]],
            (terrain.reshape(2, -1) - 1.0) / 2.0,
        )

    def test_state_advance_integrates_thirty_hz_and_transforms_prediction(self) -> None:
        state = self.make_state()
        planned = plan_recurrent_trajectory(
            state, torch.tensor(((0.0, 0.0), (0.3, 0.0)), dtype=self.dtype)
        )
        physical_output = self.make_physical_output()

        advanced = advance_recurrent_state(
            state,
            planned,
            physical_output,
            phase_advance_cap=torch.tensor((0.3, 0.3), dtype=self.dtype),
        )

        torch.testing.assert_close(
            advanced.root_world_xy,
            torch.tensor(((0.01, 0.0), (0.01, 0.0)), dtype=self.dtype),
        )
        torch.testing.assert_close(
            advanced.root_yaw_world, torch.tensor((0.02, 0.02), dtype=self.dtype)
        )
        torch.testing.assert_close(
            advanced.phase,
            torch.tensor((0.35, 0.05), dtype=self.dtype),
            atol=1.0e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            advanced.history_position_world_xy[:, :-1],
            state.history_position_world_xy[:, 1:],
        )
        torch.testing.assert_close(
            advanced.history_position_world_xy[:, -1], advanced.root_world_xy
        )
        local_position = physical_output[
            :, OUTPUT_LAYOUT["trajectory_position"]
        ].reshape(2, 12, 2)
        cosine, sine = math.cos(0.02), math.sin(0.02)
        rotation = torch.tensor(((cosine, -sine), (sine, cosine)), dtype=self.dtype)
        expected_world = advanced.root_world_xy[:, None] + local_position @ rotation.T
        torch.testing.assert_close(advanced.predicted_position_world_xy, expected_world)
        self.assertEqual(advanced.history_semantic_intent[0, -1].tolist(), [1.0, 0.0])
        self.assertEqual(advanced.history_semantic_intent[1, -1].tolist(), [0.0, 1.0])

    def test_planning_packing_and_advancement_remain_differentiable(self) -> None:
        state = self.make_state()
        predicted = state.predicted_position_world_xy.clone().requires_grad_()
        state = replace(state, predicted_position_world_xy=predicted)
        desired = torch.tensor(
            ((0.1, 0.0), (0.3, 0.1)), dtype=self.dtype, requires_grad=True
        )
        planned = plan_recurrent_trajectory(state, desired)
        packed = pack_recurrent_input(
            state=state,
            planned=planned,
            terrain_height=torch.zeros((2, 12, 3), dtype=self.dtype),
            x_mean=torch.zeros(INPUT_LAYOUT.size, dtype=self.dtype),
            x_std=torch.ones(INPUT_LAYOUT.size, dtype=self.dtype),
        )
        physical = self.make_physical_output().requires_grad_()
        advanced = advance_recurrent_state(
            state,
            planned,
            physical,
            phase_advance_cap=torch.full((2,), 0.3, dtype=self.dtype),
        )

        loss = packed.square().sum() + advanced.predicted_position_world_xy.square().sum()
        loss.backward()

        self.assertIsNotNone(desired.grad)
        self.assertIsNotNone(predicted.grad)
        self.assertIsNotNone(physical.grad)
        self.assertTrue(torch.isfinite(desired.grad).all())
        self.assertGreater(float(torch.linalg.vector_norm(desired.grad)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(physical.grad)), 0.0)

    def test_invalid_shapes_dtypes_directions_semantics_and_values_fail_closed(self) -> None:
        state = self.make_state()
        with self.assertRaisesRegex(ValueError, "desired_velocity_world"):
            plan_recurrent_trajectory(
                state, torch.zeros((2, 3), dtype=self.dtype)
            )
        with self.assertRaisesRegex(ValueError, "shared floating dtype and device"):
            plan_recurrent_trajectory(
                state, torch.zeros((2, 2), dtype=torch.float32)
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            plan_recurrent_trajectory(
                state, torch.full((2, 2), float("nan"), dtype=self.dtype)
            )
        planned = plan_recurrent_trajectory(
            state, torch.zeros((2, 2), dtype=self.dtype)
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            pack_recurrent_input(
                state=state,
                planned=planned,
                terrain_height=torch.zeros((2, 12, 3), dtype=self.dtype),
                x_mean=torch.full(
                    (INPUT_LAYOUT.size,), -torch.finfo(self.dtype).max, dtype=self.dtype
                ),
                x_std=torch.full((INPUT_LAYOUT.size,), 0.5, dtype=self.dtype),
            )
        with self.assertRaisesRegex(ValueError, "normalized"):
            PlannedTrajectory(
                position_world_xy=torch.zeros((2, 12, 2), dtype=self.dtype),
                direction_world_xy=torch.zeros((2, 12, 2), dtype=self.dtype),
                semantic_intent=torch.nn.functional.one_hot(
                    torch.zeros((2, 12), dtype=torch.long), num_classes=2
                ).to(self.dtype),
            )
        invalid_semantic = torch.full((2, 12, 2), 0.5, dtype=self.dtype)
        with self.assertRaisesRegex(ValueError, "one-hot"):
            PlannedTrajectory(
                position_world_xy=torch.zeros((2, 12, 2), dtype=self.dtype),
                direction_world_xy=state.predicted_direction_world_xy,
                semantic_intent=invalid_semantic,
            )

    def test_shared_recurrence_surface_is_exported_from_package(self) -> None:
        expected = {
            "RecurrentTrajectoryState": RecurrentTrajectoryState,
            "PlannedTrajectory": PlannedTrajectory,
            "initialize_recurrent_state": initialize_recurrent_state,
            "derive_training_desired_velocity": derive_training_desired_velocity,
            "plan_recurrent_trajectory": plan_recurrent_trajectory,
            "pack_recurrent_input": pack_recurrent_input,
            "advance_recurrent_state": advance_recurrent_state,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(terrain_pfnn, name), value)


if __name__ == "__main__":
    unittest.main()
