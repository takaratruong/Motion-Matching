"""Exact command shaping and dense Torch search contract tests."""

import math
import unittest

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_motion_features import CommandTrajectory, TorchMotionDatabase
from mm_sonic.torch_motion_matcher import (
    MatcherConfig,
    active_transition_penalty,
    bounded_velocity_step,
    bounded_yaw_step,
    predict_command_trajectory,
    search_is_due,
    select_exact_candidate,
)


def _database(features, clip_indices, frame_indices, device):
    tensor = torch.tensor(features, dtype=torch.float32, device=device)
    return TorchMotionDatabase(
        folder=None,
        device=torch.device(device),
        normalization=None,
        reset_row=0,
        _search_features=tensor,
        _search_clip_index=torch.tensor(clip_indices, device=device),
        _search_frame_index=torch.tensor(frame_indices, device=device),
        _source_row_map={
            (int(clip), int(frame)): row
            for row, (clip, frame) in enumerate(zip(clip_indices, frame_indices))
        },
    )


class CommandShapingTests(unittest.TestCase):
    def test_acceleration_deceleration_reversal_and_yaw_caps_are_exact(self):
        cfg = MatcherConfig()
        zero = torch.zeros(2, dtype=torch.float32)
        forward = torch.tensor([1.0, 0.0], dtype=torch.float32)

        accelerated = bounded_velocity_step(zero, forward, config=cfg)
        np.testing.assert_allclose(accelerated.numpy(), [0.03, 0.0], atol=1e-7)

        decelerated = bounded_velocity_step(forward, zero, config=cfg)
        np.testing.assert_allclose(decelerated.numpy(), [0.96, 0.0], atol=1e-7)

        reversed_step = bounded_velocity_step(forward, -forward, config=cfg)
        np.testing.assert_allclose(reversed_step.numpy(), [0.96, 0.0], atol=1e-7)

        lateral = bounded_velocity_step(
            zero, torch.tensor([0.3, 0.4]), config=cfg
        )
        np.testing.assert_allclose(lateral.numpy(), [0.018, 0.024], atol=1e-7)

        yaw_step = bounded_yaw_step(
            torch.tensor(math.radians(179.0)),
            torch.tensor(math.radians(-179.0)),
            config=cfg,
        )
        self.assertAlmostEqual(float(yaw_step), math.radians(-179.0), places=6)

        capped = bounded_yaw_step(
            torch.tensor(0.0), torch.tensor(math.pi), config=cfg
        )
        self.assertAlmostEqual(
            abs(float(capped)), math.radians(120.0) * 0.02, places=6
        )

    def test_prediction_samples_steps_15_30_45(self):
        cfg = MatcherConfig(
            acceleration_mps2=1000.0,
            deceleration_mps2=1000.0,
            yaw_rate_rad_s=1000.0,
        )
        command = predict_command_trajectory(
            torch.tensor([1.0, -2.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor(0.0),
            torch.tensor([1.0, 0.0]),
            torch.tensor(math.pi / 2.0),
            config=cfg,
        )
        self.assertIsInstance(command.trajectory, CommandTrajectory)
        np.testing.assert_allclose(
            command.trajectory.position_world_xy.cpu().numpy(),
            [[1.3, -2.0], [1.6, -2.0], [1.9, -2.0]],
            atol=2e-6,
        )
        expected_facing = np.tile(np.array([[0.0, 1.0]]), (3, 1))
        np.testing.assert_allclose(
            command.trajectory.facing_world_xy.cpu().numpy(),
            expected_facing,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            command.velocity_world_xy.cpu().numpy(), [1.0, 0.0], atol=1e-7
        )
        self.assertAlmostEqual(float(command.heading_world_yaw), math.pi / 2.0)

    def test_start_stop_reversal_and_clip_end_force_search(self):
        cfg = MatcherConfig()

        def prediction(old, requested, *, successor=True):
            return predict_command_trajectory(
                torch.zeros(2),
                torch.tensor(old, dtype=torch.float32),
                torch.tensor(0.0),
                torch.tensor(requested, dtype=torch.float32),
                torch.tensor(0.0),
                has_valid_successor=successor,
                config=cfg,
            )

        self.assertTrue(prediction((0.0, 0.0), (0.2, 0.0)).force_search)
        self.assertTrue(prediction((0.2, 0.0), (0.0, 0.0)).force_search)
        self.assertTrue(prediction((0.2, 0.0), (-0.2, 0.0)).force_search)
        self.assertTrue(
            prediction((0.2, 0.0), (0.2, 0.0), successor=False).force_search
        )
        self.assertFalse(prediction((0.2, 0.0), (0.2, 0.0)).force_search)

        self.assertTrue(search_is_due(0, False, cfg))
        self.assertFalse(search_is_due(1, False, cfg))
        self.assertTrue(search_is_due(5, False, cfg))
        self.assertTrue(search_is_due(3, True, cfg))


class ExactSearchTests(unittest.TestCase):
    def _oracle(
        self,
        features,
        query,
        clips,
        frames,
        current_clip,
        current_frame,
        incumbent,
        config,
    ):
        feature_costs = np.square(
            np.asarray(features, dtype=np.float32)
            - np.asarray(query, dtype=np.float32)[None, :]
        ).sum(axis=1, dtype=np.float32)
        eligible = np.ones(len(features), dtype=bool)
        eligible &= ~(
            (np.asarray(clips) == current_clip)
            & (np.abs(np.asarray(frames) - current_frame) <= config.exclusion_frames)
        )
        if incumbent is not None:
            eligible[incumbent] = True
        total = feature_costs + np.float32(config.transition_penalty)
        if incumbent is not None:
            total[incumbent] = feature_costs[incumbent]
        total[~eligible] = np.inf
        return int(np.argmin(total)), feature_costs, total

    def test_cpu_and_cuda_match_float32_numpy_oracle(self):
        rng = np.random.default_rng(1947)
        features = rng.normal(size=(64, 27)).astype(np.float32)
        query = rng.normal(size=27).astype(np.float32)
        clips = np.repeat([0, 1], 32)
        frames = np.tile(np.arange(32), 2)
        cfg = MatcherConfig()
        expected, feature_costs, totals = self._oracle(
            features, query, clips, frames, 0, 12, 13, cfg
        )

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        decisions = []
        for device in devices:
            db = _database(features, clips, frames, device)
            decision = select_exact_candidate(
                db,
                torch.tensor(query, device=device),
                current_clip_index=0,
                current_frame_index=12,
                incumbent_row=13,
                search=True,
                config=cfg,
            )
            self.assertEqual(decision.selected_row, expected)
            self.assertAlmostEqual(
                decision.selected_feature_cost,
                float(feature_costs[expected]),
                places=4,
            )
            self.assertAlmostEqual(
                decision.selected_total_cost, float(totals[expected]), places=4
            )
            decisions.append(decision)
        self.assertTrue(all(d.selected_row == decisions[0].selected_row for d in decisions))

    def test_local_20_frame_exclusion_applies_only_to_current_clip(self):
        features = np.full((5, 1), 10.0, dtype=np.float32)
        features[0, 0] = 0.0  # current clip, frame 30: excluded
        features[1, 0] = 1.0  # current clip, frame 9: outside exclusion
        features[2, 0] = 0.0  # other clip, same frame: eligible and wins
        db = _database(features, [0, 0, 1, 0, 0], [30, 9, 30, 31, 50], "cpu")
        decision = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=30,
            incumbent_row=3,
            search=True,
            config=MatcherConfig(),
        )
        self.assertEqual(decision.selected_row, 2)

    def test_candidate_must_beat_incumbent_after_point_one_penalty(self):
        db = _database([[math.sqrt(1.0)], [math.sqrt(0.91)]], [0, 1], [1, 1], "cpu")
        decision = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=0,
            incumbent_row=0,
            search=True,
            config=MatcherConfig(),
        )
        self.assertEqual(decision.selected_row, 0)
        self.assertFalse(decision.transitioned)

        db = _database([[math.sqrt(1.0)], [math.sqrt(0.89)]], [0, 1], [1, 1], "cpu")
        decision = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=0,
            incumbent_row=0,
            search=True,
            config=MatcherConfig(),
        )
        self.assertEqual(decision.selected_row, 1)
        self.assertTrue(decision.transitioned)

    def test_active_transition_penalty_decays_linearly_and_can_be_disabled(self):
        cfg = MatcherConfig(
            transition_settle_duration_s=0.20,
            transition_settle_penalty=10.0,
        )
        self.assertAlmostEqual(active_transition_penalty(0.0, cfg), 10.0)
        self.assertAlmostEqual(active_transition_penalty(0.05, cfg), 7.5)
        self.assertAlmostEqual(active_transition_penalty(0.20, cfg), 0.0)
        self.assertAlmostEqual(active_transition_penalty(1.0, cfg), 0.0)
        self.assertEqual(
            active_transition_penalty(
                0.0,
                MatcherConfig(
                    transition_settle_duration_s=0.20,
                    transition_settle_penalty=0.0,
                ),
            ),
            0.0,
        )
        accumulated_age = sum(0.02 for _ in range(10))
        self.assertLess(accumulated_age, 0.20)
        self.assertEqual(
            active_transition_penalty(accumulated_age, cfg),
            0.0,
        )
        self.assertEqual(
            active_transition_penalty(
                0.0,
                MatcherConfig(
                    transition_settle_duration_s=0.0,
                    transition_settle_penalty=10.0,
                ),
            ),
            0.0,
        )

    def test_active_transition_penalty_rejects_invalid_inputs(self):
        valid = MatcherConfig(
            transition_settle_duration_s=0.20,
            transition_settle_penalty=10.0,
        )
        for age in (-0.01, math.inf, -math.inf, math.nan):
            with self.subTest(age=age):
                with self.assertRaises(ContractError):
                    active_transition_penalty(age, valid)
        for duration in (-0.01, math.inf, math.nan):
            with self.subTest(duration=duration):
                with self.assertRaises(ContractError):
                    active_transition_penalty(
                        0.0,
                        MatcherConfig(
                            transition_settle_duration_s=duration,
                            transition_settle_penalty=10.0,
                        ),
                    )
        for magnitude in (-0.01, math.inf, math.nan):
            with self.subTest(magnitude=magnitude):
                with self.assertRaises(ContractError):
                    active_transition_penalty(
                        0.0,
                        MatcherConfig(
                            transition_settle_duration_s=0.20,
                            transition_settle_penalty=magnitude,
                        ),
                    )

    def test_additional_penalty_never_applies_to_incumbent(self):
        db = _database(
            [[math.sqrt(1.0)], [math.sqrt(0.5)]],
            [0, 1],
            [1, 1],
            "cpu",
        )
        held = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=0,
            incumbent_row=0,
            search=True,
            config=MatcherConfig(),
            additional_transition_penalty=0.5,
        )
        self.assertEqual(held.selected_row, 0)
        self.assertAlmostEqual(held.selected_total_cost, 1.0)

        switched = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=0,
            incumbent_row=0,
            search=True,
            config=MatcherConfig(),
            additional_transition_penalty=0.3,
        )
        self.assertEqual(switched.selected_row, 1)
        self.assertAlmostEqual(switched.selected_total_cost, 0.9, places=6)

    def test_additional_transition_penalty_must_be_finite_and_non_negative(self):
        db = _database([[1.0], [0.0]], [0, 1], [1, 1], "cpu")
        for penalty in (-0.01, math.inf, -math.inf, math.nan):
            with self.subTest(penalty=penalty):
                with self.assertRaises(ContractError):
                    select_exact_candidate(
                        db,
                        torch.zeros(1),
                        current_clip_index=0,
                        current_frame_index=0,
                        incumbent_row=0,
                        search=True,
                        config=MatcherConfig(),
                        additional_transition_penalty=penalty,
                    )

    def test_missing_incumbent_requires_lowest_cost_valid_candidate(self):
        db = _database([[0.0], [2.0], [1.0]], [0, 1, 1], [5, 2, 3], "cpu")
        decision = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=5,
            incumbent_row=None,
            search=True,
            config=MatcherConfig(),
        )
        self.assertEqual(decision.selected_row, 2)
        self.assertIsNone(decision.incumbent_row)
        self.assertTrue(math.isinf(decision.incumbent_cost))
        self.assertTrue(decision.transitioned)

    def test_equal_cost_chooses_smallest_global_row(self):
        db = _database([[1.0], [1.0], [2.0]], [1, 1, 1], [0, 1, 2], "cpu")
        decision = select_exact_candidate(
            db,
            torch.zeros(1),
            current_clip_index=0,
            current_frame_index=100,
            incumbent_row=None,
            search=True,
            config=MatcherConfig(),
        )
        self.assertEqual(decision.selected_row, 0)


if __name__ == "__main__":
    unittest.main()
