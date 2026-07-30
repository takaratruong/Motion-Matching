from copy import deepcopy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import MappingProxyType
from unittest import mock

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_terrain_directional_rollout import (
    ASCENT_STOP,
    REVERSAL_STOP,
    STEP_COUNT,
    TRANSITION_NEIGHBORHOOD_RADIUS,
    ChallengeCommand,
    ChallengeMatrix,
    ChallengeScenario,
    ChallengeScenarioRun,
    DirectionalRollout,
    _challenge_run_hash,
    _challenge_exception_evidence,
    _compute_challenge_metrics,
    build_directional_argument_parser,
    challenge_scenarios,
    compute_directional_metrics,
    directional_phase,
    evaluate_challenge_matrix,
    main,
    run_challenge_matrix,
    run_directional_rollout,
    save_challenge_matrix,
    save_directional_rollout,
    transition_neighborhood_mask,
)
from tests.python import test_sonic_torch_terrain_rollout as terrain_rollout_test


def _readonly(value):
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fixture_rollout() -> DirectionalRollout:
    arrays = MappingProxyType(
        {
            "joint_position": _readonly(
                np.array([[0.0], [1.0], [4.0], [10.0]], np.float32)
            ),
            "step_time_ns": _readonly(
                np.array([11, 12, 13, 14], np.int64)
            ),
        }
    )
    resolved = MappingProxyType(
        {
            "matcher": {
                "transition_joint_position_weight": 0.1,
                "transition_joint_velocity_weight": 0.25,
            },
            "directional_step_count": STEP_COUNT,
        }
    )
    metrics = MappingProxyType(
        {
            "position_weight": 0.1,
            "velocity_weight": 0.25,
            "deterministic_sha256": "a" * 64,
            "gates": {"completed_640_frames": True},
            "qualified": True,
        }
    )
    events = (
        MappingProxyType(
            {
                "sequence": 1,
                "transitioned": True,
                "selected_total_cost": 2.0,
            }
        ),
    )
    return DirectionalRollout(
        arrays=arrays,
        metrics=metrics,
        events=events,
        resolved_config=resolved,
        resolved_config_sha256="b" * 64,
        deterministic_sha256="a" * 64,
    )


def _metric_arrays(frame_count=STEP_COUNT):
    return {
        "joint_position": np.zeros((frame_count, 1), np.float64),
        "joint_velocity": np.zeros((frame_count, 1), np.float64),
        "root_position_world": np.column_stack(
            (
                np.zeros(frame_count),
                np.zeros(frame_count),
                np.full(frame_count, 0.75),
            )
        ),
        "foot_position_world": np.zeros((frame_count, 2, 3), np.float64),
        "foot_clearance_m": np.zeros((frame_count, 2), np.float64),
        "selected_clip_index": np.zeros(frame_count, np.int32),
        "previous_selected_clip_index": np.zeros(frame_count, np.int32),
        "transitioned": np.zeros(frame_count, np.bool_),
        "transition_rejected": np.zeros(frame_count, np.bool_),
        "terrain_safety_override": np.zeros(frame_count, np.bool_),
        "terrain_safety_override_rank": np.zeros(frame_count, np.int32),
        "selected_transition_position_cost": np.zeros(
            frame_count, np.float64
        ),
        "selected_transition_velocity_cost": np.zeros(
            frame_count, np.float64
        ),
        "selected_transition_continuity_cost": np.zeros(
            frame_count, np.float64
        ),
    }


def _challenge_metric(
    name,
    *,
    jerk=100.0,
    clearance=0.0,
    completed=True,
    error=None,
    position_weight=0.0,
    velocity_weight=0.0,
):
    return {
        "scenario_name": name,
        "command_sha256": name.encode("utf-8").hex().ljust(64, "0")[:64],
        "completed_without_exception": completed,
        "minimum_clearance_m": clearance,
        "p95_joint_jerk_rad_s3": jerk,
        "exception": error,
        "position_weight": position_weight,
        "velocity_weight": velocity_weight,
    }


def _fixture_challenge_matrix(
    *,
    jerk=100.0,
    position_weight=0.0,
    velocity_weight=0.0,
) -> ChallengeMatrix:
    resolved_config = {
        "fixture": True,
        "matcher": {
            "transition_joint_position_weight": position_weight,
            "transition_joint_velocity_weight": velocity_weight,
        },
    }
    resolved_config_sha = hashlib.sha256(
        (
            json.dumps(
                resolved_config,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    runs = []
    for scenario in challenge_scenarios():
        command_count = len(scenario.commands)
        arrays = MappingProxyType(
            {
                "command_velocity_world_xy": _readonly(
                    np.asarray(
                        [
                            command.velocity_world_xy
                            for command in scenario.commands
                        ],
                        np.float32,
                    )
                ),
                "command_heading_world_yaw": _readonly(
                    np.asarray(
                        [
                            command.heading_world_yaw
                            for command in scenario.commands
                        ],
                        np.float32,
                    )
                ),
                "reset_before": _readonly(
                    np.asarray(
                        [
                            command.reset_before
                            for command in scenario.commands
                        ],
                        np.bool_,
                    )
                ),
                "joint_position": _readonly(
                    np.zeros((command_count, 1), np.float32)
                ),
            }
        )
        command_identity = {
            "name": scenario.name,
            "commands": [
                {
                    "velocity_world_xy": list(command.velocity_world_xy),
                    "heading_world_yaw": command.heading_world_yaw,
                    "reset_before": command.reset_before,
                }
                for command in scenario.commands
            ],
        }
        command_sha = hashlib.sha256(
            (
                json.dumps(
                    command_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        metrics = MappingProxyType(
            _challenge_metric(
                scenario.name,
                jerk=jerk,
                position_weight=position_weight,
                velocity_weight=velocity_weight,
            )
        )
        runs.append(
            ChallengeScenarioRun(
                scenario=scenario,
                arrays=arrays,
                metrics=metrics,
                events=(),
                command_sha256=command_sha,
                deterministic_sha256=("d" * 63 + str(len(runs))),
            )
        )
    return ChallengeMatrix(
        runs=tuple(runs),
        metrics=MappingProxyType(
            {
                "position_weight": position_weight,
                "velocity_weight": velocity_weight,
                "scenario_count": 6,
            }
        ),
        resolved_config=MappingProxyType(resolved_config),
        resolved_config_sha256=resolved_config_sha,
        deterministic_sha256="e" * 64,
    )


def _challenge_oracle():
    frame_count = 10
    scenario = ChallengeScenario(
        "oracle",
        tuple(
            ChallengeCommand(
                (1.0, 0.0),
                0.0,
                reset_before=index in (0, 5),
            )
            for index in range(frame_count)
        ),
    )
    joint_episode = np.array([0.0, 1.0, 4.0, 10.0, 20.0])
    root_episode = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
    arrays = {
        "joint_position": np.concatenate(
            (joint_episode, joint_episode + 1000.0)
        )[:, None],
        "root_position_world": np.column_stack(
            (
                np.concatenate(
                    (root_episode, root_episode + 1000.0)
                ),
                np.zeros(frame_count),
                np.zeros(frame_count),
            )
        ),
        "foot_position_world": np.zeros(
            (frame_count, 2, 3), np.float64
        ),
        "foot_clearance_m": np.zeros((frame_count, 2), np.float64),
        "reset_before": np.array(
            [True, False, False, False, False] * 2, np.bool_
        ),
        "selected_clip_index": np.zeros(frame_count, np.int32),
        "previous_selected_clip_index": np.zeros(
            frame_count, np.int32
        ),
        "transitioned": np.array(
            [False, False, False, False, True]
            + [False] * 5,
            np.bool_,
        ),
        "transition_rejected": np.zeros(frame_count, np.bool_),
        "terrain_safety_override": np.zeros(frame_count, np.bool_),
        "terrain_safety_override_rank": np.zeros(
            frame_count, np.int32
        ),
        "selected_transition_position_cost": np.zeros(
            frame_count, np.float64
        ),
        "selected_transition_velocity_cost": np.zeros(
            frame_count, np.float64
        ),
        "selected_transition_continuity_cost": np.zeros(
            frame_count, np.float64
        ),
    }
    return scenario, arrays


class ChallengeMetricAndIdentityTests(unittest.TestCase):
    def test_identity_excludes_only_runtime_timing_and_includes_events(self):
        arrays = {
            "time_s": np.array([0.0], np.float32),
            "joint_position": np.array([[1.0]], np.float32),
            "step_time_ns": np.array([11], np.int64),
            "search_time_ns": np.array([7], np.int64),
        }
        events = (
            {
                "command_index": 0,
                "time_s": 0.0,
                "selected_frame": 3,
                "step_time_ns": 11,
                "search_time_ns": 7,
            },
        )

        def identity(
            array_values=arrays,
            event_values=events,
            exception=None,
        ):
            return _challenge_run_hash(
                "a" * 64,
                "b" * 64,
                array_values,
                event_values,
                exception,
            )

        expected = identity()
        timing_arrays = deepcopy(arrays)
        timing_arrays["step_time_ns"][0] = 999
        timing_arrays["search_time_ns"][0] = 888
        self.assertEqual(identity(timing_arrays), expected)
        timing_events = deepcopy(events)
        timing_events[0]["step_time_ns"] = 999
        timing_events[0]["search_time_ns"] = 888
        self.assertEqual(identity(event_values=timing_events), expected)

        behavior_array = deepcopy(arrays)
        behavior_array["joint_position"][0, 0] = 2.0
        self.assertNotEqual(identity(behavior_array), expected)
        behavior_event = deepcopy(events)
        behavior_event[0]["selected_frame"] = 4
        self.assertNotEqual(
            identity(event_values=behavior_event), expected
        )
        command_time = deepcopy(events)
        command_time[0]["time_s"] = 0.02
        self.assertNotEqual(identity(event_values=command_time), expected)
        self.assertNotEqual(
            identity(
                exception={
                    "type": "RuntimeError",
                    "message": "failed",
                    "command_index": 0,
                    "stage": "command",
                }
            ),
            expected,
        )

    def test_boundary_exception_convention_never_claims_command_zero(self):
        self.assertEqual(
            _challenge_exception_evidence(
                RuntimeError("finalize failed"),
                stage="scenario_finalize",
                command_index=None,
            ),
            {
                "type": "RuntimeError",
                "message": "finalize failed",
                "command_index": None,
                "stage": "scenario_finalize",
            },
        )
        with self.assertRaises(ContractError):
            _challenge_exception_evidence(
                RuntimeError("invalid"),
                stage="scenario_setup",
                command_index=0,
            )

    def test_reset_local_derivative_and_transition_neighborhood_oracle(self):
        scenario, arrays = _challenge_oracle()
        metrics = _compute_challenge_metrics(
            scenario,
            arrays,
            dt=1.0,
            position_weight=0.0,
            velocity_weight=0.0,
            exception=None,
        )
        aggregate = metrics["aggregate"]
        self.assertEqual(
            aggregate["joint_jerk_rad_s3"]["samples"],
            [1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(
            aggregate["root_acceleration_m_s2"]["samples"],
            [2.0] * 6,
        )
        self.assertEqual(
            aggregate[
                "transition_neighborhood_joint_jerk_rad_s3"
            ]["samples"],
            [1.0, 1.0],
        )
        self.assertEqual(
            aggregate[
                "transition_neighborhood_joint_jerk_output_frames"
            ],
            [3, 4],
        )
        self.assertEqual(aggregate["root_jerk_m_s3"]["maximum"], 0.0)


class ChallengeScenarioGenerationTests(unittest.TestCase):
    def test_exact_named_scenarios_and_segment_commands(self):
        scenarios = challenge_scenarios()
        self.assertEqual(
            tuple(scenario.name for scenario in scenarios),
            (
                "rapid-reversal",
                "lateral-switch",
                "independent-octants",
                "stair-stop-restart",
                "upper-landing-side-exit",
                "seeded-random",
            ),
        )
        rapid = scenarios[0].commands
        for index, command in enumerate(rapid):
            self.assertEqual(
                command.velocity_world_xy,
                ((1.0, 0.0) if (index // 10) % 2 == 0 else (-1.0, 0.0)),
            )
            self.assertEqual(command.heading_world_yaw, 0.0)

        lateral = scenarios[1].commands
        for index, command in enumerate(lateral):
            self.assertEqual(
                command.velocity_world_xy,
                ((0.0, 1.0) if (index // 15) % 2 == 0 else (0.0, -1.0)),
            )
            self.assertEqual(command.heading_world_yaw, 0.0)

    def test_octants_are_normalized_and_heading_is_independent(self):
        commands = challenge_scenarios()[2].commands
        travel = {
            tuple(np.round(command.velocity_world_xy, 12))
            for command in commands
        }
        root_half = round(math.sqrt(0.5), 12)
        self.assertEqual(
            travel,
            {
                (1.0, 0.0),
                (root_half, root_half),
                (0.0, 1.0),
                (-root_half, root_half),
                (-1.0, 0.0),
                (-root_half, -root_half),
                (0.0, -1.0),
                (root_half, -root_half),
            },
        )
        differing = 0
        for command in commands:
            travel_yaw = math.atan2(*command.velocity_world_xy[::-1])
            delta = math.atan2(
                math.sin(command.heading_world_yaw - travel_yaw),
                math.cos(command.heading_world_yaw - travel_yaw),
            )
            differing += not math.isclose(delta, 0.0, abs_tol=1e-12)
        self.assertGreaterEqual(differing, 4)

    def test_scripted_stair_scenarios_preserve_clock_and_reset_episodes(self):
        by_name = {
            scenario.name: scenario for scenario in challenge_scenarios()
        }
        side_exit = by_name["upper-landing-side-exit"].commands
        ascent = tuple((1.0, 0.0) for _ in range(ASCENT_STOP))
        self.assertEqual(
            tuple(command.velocity_world_xy for command in side_exit[:ASCENT_STOP]),
            ascent,
        )
        resets = [
            index
            for index, command in enumerate(side_exit)
            if command.reset_before
        ]
        self.assertEqual(resets, [0, 400])
        self.assertEqual(
            tuple(
                command.velocity_world_xy
                for command in side_exit[ASCENT_STOP:400]
            ),
            tuple((0.0, 1.0) for _ in range(120)),
        )
        self.assertEqual(
            tuple(
                command.velocity_world_xy
                for command in side_exit[400:680]
            ),
            ascent,
        )
        self.assertEqual(
            tuple(command.velocity_world_xy for command in side_exit[680:]),
            tuple((0.0, -1.0) for _ in range(120)),
        )

        stop_restart = by_name["stair-stop-restart"].commands
        zero_runs = []
        run_start = None
        for index, command in enumerate(stop_restart + (ChallengeCommand((1.0, 0.0), 0.0),)):
            if command.velocity_world_xy == (0.0, 0.0) and run_start is None:
                run_start = index
            elif command.velocity_world_xy != (0.0, 0.0) and run_start is not None:
                zero_runs.append((run_start, index))
                run_start = None
        self.assertEqual(zero_runs, [(80, 100), (200, 220), (320, 340)])
        self.assertEqual(len(stop_restart), 340)

    def test_seeded_random_is_deterministic_bounded_and_independent(self):
        first = challenge_scenarios(seed=20260730)[-1].commands
        second = challenge_scenarios(seed=20260730)[-1].commands
        different = challenge_scenarios(seed=20260731)[-1].commands
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        reset_indices = [
            index for index, command in enumerate(first) if command.reset_before
        ]
        self.assertEqual(len(reset_indices), 8)
        episode_stops = reset_indices[1:] + [len(first)]
        allowed = {
            (0.0, 0.0),
            (1.0, 0.0),
            (round(math.sqrt(0.5), 12), round(math.sqrt(0.5), 12)),
            (0.0, 1.0),
            (-round(math.sqrt(0.5), 12), round(math.sqrt(0.5), 12)),
            (-1.0, 0.0),
            (-round(math.sqrt(0.5), 12), -round(math.sqrt(0.5), 12)),
            (0.0, -1.0),
            (round(math.sqrt(0.5), 12), -round(math.sqrt(0.5), 12)),
        }
        headings = {
            round(index * math.pi / 4.0, 12) for index in range(-4, 4)
        }
        for start, stop in zip(reset_indices, episode_stops):
            self.assertLessEqual(stop - start, 400)
            segment_start = start
            while segment_start < stop:
                identity = (
                    first[segment_start].velocity_world_xy,
                    first[segment_start].heading_world_yaw,
                )
                segment_stop = segment_start + 1
                while (
                    segment_stop < stop
                    and (
                        first[segment_stop].velocity_world_xy,
                        first[segment_stop].heading_world_yaw,
                    )
                    == identity
                ):
                    segment_stop += 1
                self.assertGreaterEqual(segment_stop - segment_start, 5)
                self.assertLessEqual(segment_stop - segment_start, 40)
                segment_start = segment_stop
            for command in first[start:stop]:
                self.assertIn(
                    tuple(round(value, 12) for value in command.velocity_world_xy),
                    allowed,
                )
                self.assertIn(round(command.heading_world_yaw, 12), headings)
        self.assertTrue(
            any(
                command.velocity_world_xy != (0.0, 0.0)
                and not math.isclose(
                    command.heading_world_yaw,
                    math.atan2(
                        command.velocity_world_xy[1],
                        command.velocity_world_xy[0],
                    ),
                    abs_tol=1e-12,
                )
                for command in first
            )
        )

    def test_challenge_inputs_are_frozen_and_validated(self):
        command = ChallengeCommand((1.0, 0.0), 0.0, True)
        scenario = ChallengeScenario("valid", (command,))
        self.assertEqual(scenario.commands, (command,))
        with self.assertRaises(Exception):
            command.heading_world_yaw = 1.0
        for velocity, heading, reset in (
            ((math.nan, 0.0), 0.0, False),
            ((0.0, 0.0), math.inf, False),
            ((0.0,), 0.0, False),
            ((0.0, 0.0), 0.0, 1),
        ):
            with self.subTest(velocity=velocity, heading=heading, reset=reset):
                with self.assertRaises(ContractError):
                    ChallengeCommand(velocity, heading, reset)
        for name, commands in (("", (command,)), ("valid", ())):
            with self.subTest(name=name):
                with self.assertRaises(ContractError):
                    ChallengeScenario(name, commands)
        for kwargs in (
            {"seed": True},
            {"seed": 1.5},
            {"speed": math.nan},
            {"speed": math.inf},
            {"forward_heading_world_yaw": math.nan},
            {"forward_heading_world_yaw": math.inf},
            {"max_episode_frames": 0},
            {"max_episode_frames": -1},
            {"max_episode_frames": 1.5},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ContractError):
                    challenge_scenarios(**kwargs)

    def test_scenarios_rotate_velocity_and_heading_into_stair_frame(self):
        scenarios = challenge_scenarios(
            speed=2.0,
            forward_heading_world_yaw=math.pi / 2.0,
        )
        rapid = scenarios[0].commands
        np.testing.assert_allclose(
            rapid[0].velocity_world_xy, (0.0, 2.0), atol=1e-12
        )
        np.testing.assert_allclose(
            rapid[10].velocity_world_xy, (0.0, -2.0), atol=1e-12
        )
        self.assertAlmostEqual(
            rapid[0].heading_world_yaw, math.pi / 2.0
        )
        lateral = scenarios[1].commands
        np.testing.assert_allclose(
            lateral[0].velocity_world_xy, (-2.0, 0.0), atol=1e-12
        )
        self.assertAlmostEqual(
            lateral[0].heading_world_yaw, math.pi / 2.0
        )
        octants = scenarios[2].commands
        self.assertEqual(
            {
                round(math.hypot(*command.velocity_world_xy), 12)
                for command in octants
            },
            {2.0},
        )
        self.assertGreaterEqual(
            sum(
                not math.isclose(
                    command.heading_world_yaw,
                    math.atan2(
                        command.velocity_world_xy[1],
                        command.velocity_world_xy[0],
                    ),
                    abs_tol=1e-12,
                )
                for command in octants
            ),
            4,
        )


class ChallengeMatrixAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.names = tuple(
            scenario.name for scenario in challenge_scenarios()
        )
        self.baseline = {
            name: _challenge_metric(name, jerk=100.0)
            for name in self.names
        }
        self.retained = {
            name: _challenge_metric(
                name, jerk=80.0 if index < 4 else 100.0
            )
            for index, name in enumerate(self.names)
        }

    def test_four_improvements_and_all_nonregressions_pass(self):
        verdict = evaluate_challenge_matrix(self.baseline, self.retained)
        self.assertTrue(verdict["matrix_pass"])
        self.assertEqual(verdict["material_improvement_count"], 4)
        self.assertTrue(all(verdict["gates"].values()))

    def test_clearance_exception_or_one_110_percent_regression_fails(self):
        cases = {}
        low_clearance = deepcopy(self.retained)
        low_clearance[self.names[0]]["minimum_clearance_m"] = -0.030001
        cases["minimum_clearance"] = low_clearance
        crashed = deepcopy(self.retained)
        crashed[self.names[1]]["completed_without_exception"] = False
        crashed[self.names[1]]["exception"] = {
            "type": "RuntimeError",
            "message": "fixture",
            "command_index": 17,
        }
        cases["completed_without_exception"] = crashed
        regression = deepcopy(self.retained)
        regression[self.names[2]]["p95_joint_jerk_rad_s3"] = 110.000001
        cases["scenario_nonregression"] = regression
        for gate, retained in cases.items():
            with self.subTest(gate=gate):
                verdict = evaluate_challenge_matrix(
                    self.baseline, retained
                )
                self.assertFalse(verdict["gates"][gate])
                self.assertFalse(verdict["matrix_pass"])

    def test_exact_zero_baseline_requires_zero_and_never_improves(self):
        baseline = deepcopy(self.baseline)
        retained = deepcopy(self.retained)
        baseline[self.names[0]]["p95_joint_jerk_rad_s3"] = 0.0
        retained[self.names[0]]["p95_joint_jerk_rad_s3"] = 0.0
        verdict = evaluate_challenge_matrix(baseline, retained)
        scenario = verdict["scenarios"][self.names[0]]
        self.assertIsNone(scenario["jerk_ratio"])
        self.assertTrue(scenario["scenario_nonregression"])
        self.assertFalse(scenario["material_improvement"])

        retained[self.names[0]]["p95_joint_jerk_rad_s3"] = 0.001
        verdict = evaluate_challenge_matrix(baseline, retained)
        self.assertFalse(
            verdict["scenarios"][self.names[0]][
                "scenario_nonregression"
            ]
        )
        self.assertFalse(verdict["matrix_pass"])

    def test_evaluation_authenticates_names_and_command_identity(self):
        missing = deepcopy(self.baseline)
        missing.pop(self.names[-1])
        with self.assertRaises(ContractError):
            evaluate_challenge_matrix(missing, self.retained)
        changed = deepcopy(self.baseline)
        changed[self.names[0]]["command_sha256"] = "f" * 64
        with self.assertRaises(ContractError):
            evaluate_challenge_matrix(changed, self.retained)


class ChallengeMatrixExecutionAndArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_run_continues_after_scenario_exception_with_exact_evidence(self):
        scenarios = (
            ChallengeScenario(
                "first",
                (
                    ChallengeCommand((1.0, 0.0), 0.0, True),
                    ChallengeCommand((1.0, 0.0), 0.0),
                ),
            ),
            ChallengeScenario(
                "second",
                (ChallengeCommand((0.0, 0.0), math.pi / 2.0, True),),
            ),
        )
        completed = ChallengeScenarioRun(
            scenario=scenarios[1],
            arrays=MappingProxyType({}),
            metrics=MappingProxyType(
                {
                    "scenario_name": "second",
                    "completed_without_exception": True,
                    "minimum_clearance_m": 0.0,
                    "p95_joint_jerk_rad_s3": 0.0,
                }
            ),
            events=(),
            command_sha256="b" * 64,
            deterministic_sha256="c" * 64,
        )
        sentinel_resolved = mock.Mock()
        sentinel_resolved.resolved_config = {
            "dt": 0.02,
            "command_speed_mps": 1.0,
            "reference_direction_matcher_xy": [1.0, 0.0],
        }
        sentinel_resolved.device = __import__("torch").device("cpu")
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.challenge_scenarios",
                return_value=scenarios,
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout._run_challenge_scenario",
                side_effect=[RuntimeError("boom"), completed],
            ) as execute,
        ):
            matrix = run_challenge_matrix(
                sentinel_resolved,
                device="cpu",
                position_weight=0.1,
                velocity_weight=0.25,
            )
        self.assertEqual(execute.call_count, 2)
        failed = matrix.runs[0]
        self.assertFalse(failed.metrics["completed_without_exception"])
        self.assertEqual(
            failed.metrics["exception"],
            {
                "type": "RuntimeError",
                "message": "boom",
                "command_index": None,
                "stage": "scenario_boundary_unknown",
            },
        )
        self.assertEqual(matrix.runs[1], completed)

    def test_real_scenario_loop_preserves_prefix_before_command_exception(self):
        torch = __import__("torch")
        scenario = ChallengeScenario(
            "partial",
            (
                ChallengeCommand((1.0, 0.0), 0.0, True),
                ChallengeCommand((1.0, 0.0), 0.0),
            ),
        )
        diagnostics = SimpleNamespace(
            selected_clip_path="reset",
            selected_frame=3,
            searched=True,
            transitioned=True,
            transition_rejected=False,
            terrain_safety_override=False,
            terrain_safety_override_rank=0,
            motion_feature_cost=1.0,
            extension_feature_cost=2.0,
            selected_feature_cost=3.0,
            selected_total_cost=4.0,
            selected_transition_position_cost=5.0,
            selected_transition_velocity_cost=6.0,
            selected_transition_continuity_cost=7.0,
            step_time_ns=11,
            search_time_ns=7,
            sequence=1,
        )
        result = SimpleNamespace(
            diagnostics=diagnostics,
            dense_feature_body_position_window=torch.zeros(
                (1, 3, 3), dtype=torch.float32
            ),
            dense_feature_body_velocity_window=torch.zeros(
                (1, 3, 3), dtype=torch.float32
            ),
            joint_position=torch.tensor([1.0]),
            joint_velocity=torch.tensor([2.0]),
            root_position_world=torch.tensor([0.0, 0.0, 0.75]),
            root_orientation_world_wxyz=torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            ),
        )
        matcher = mock.Mock()
        matcher.motion_inventory_sha256 = "m" * 64
        matcher.reset.return_value = SimpleNamespace(
            diagnostics=SimpleNamespace(selected_clip_path="reset")
        )
        matcher.prepare_step.side_effect = ["first", "second"]
        matcher.commit.side_effect = [result, RuntimeError("second failed")]
        resolved = mock.Mock()
        resolved.device = torch.device("cpu")
        resolved.resolved_config = {
            "dt": 0.02,
            "matcher": {},
            "reset_clip": "reset",
            "command_speed_mps": 1.0,
            "reference_direction_matcher_xy": [1.0, 0.0],
        }
        resolved.dataset = SimpleNamespace(
            root=Path("dataset"),
            manifest_sha256="d" * 64,
            folder=SimpleNamespace(
                clips=(SimpleNamespace(relative_path="reset"),)
            ),
        )
        resolved.measurement_extension = SimpleNamespace(
            alignment=SimpleNamespace(
                matcher_to_scene_xy=lambda value: value
            ),
            query_grid=SimpleNamespace(
                sample_xy=lambda value: torch.zeros(
                    2, dtype=torch.float32
                )
            ),
        )
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.challenge_scenarios",
                return_value=(scenario,),
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.TorchMotionMatcher.from_folder",
                return_value=matcher,
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.matcher_config_from_resolved",
                return_value=object(),
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.terrain_transition_validator_from_resolved",
                return_value=object(),
            ),
        ):
            matrix = run_challenge_matrix(
                resolved,
                device="cpu",
                position_weight=0.0,
                velocity_weight=0.0,
            )
        run = matrix.runs[0]
        self.assertEqual(run.metrics["frame_count"], 1)
        self.assertEqual(len(run.events), 1)
        np.testing.assert_array_equal(
            run.arrays["joint_position"], np.array([[1.0]])
        )
        self.assertEqual(
            run.metrics["exception"],
            {
                "type": "RuntimeError",
                "message": "second failed",
                "command_index": 1,
                "stage": "command",
            },
        )

    def test_real_scenario_setup_exception_has_no_command_index(self):
        torch = __import__("torch")
        scenario = ChallengeScenario(
            "setup",
            (ChallengeCommand((1.0, 0.0), 0.0, True),),
        )
        resolved = mock.Mock()
        resolved.device = torch.device("cpu")
        resolved.resolved_config = {
            "dt": 0.02,
            "matcher": {},
            "reset_clip": "reset",
            "command_speed_mps": 1.0,
            "reference_direction_matcher_xy": [1.0, 0.0],
        }
        resolved.dataset = SimpleNamespace(
            root=Path("dataset"),
            folder=SimpleNamespace(clips=None),
        )
        resolved.measurement_extension = object()
        matcher = mock.Mock()
        matcher.motion_inventory_sha256 = "m" * 64
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.challenge_scenarios",
                return_value=(scenario,),
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.matcher_config_from_resolved",
                return_value=object(),
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.terrain_transition_validator_from_resolved",
                return_value=object(),
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.TorchMotionMatcher.from_folder",
                return_value=matcher,
            ),
        ):
            matrix = run_challenge_matrix(
                resolved,
                device="cpu",
                position_weight=0.0,
                velocity_weight=0.0,
            )
        exception = matrix.runs[0].metrics["exception"]
        self.assertEqual(exception["type"], "TypeError")
        self.assertIsNone(exception["command_index"])
        self.assertEqual(
            exception["stage"],
            "scenario_setup",
        )

    def test_run_generates_commands_from_resolved_stair_direction_and_speed(self):
        sentinel_resolved = mock.Mock()
        sentinel_resolved.resolved_config = {
            "dt": 0.02,
            "matcher": {},
            "command_speed_mps": 0.3784,
            "reference_direction_matcher_xy": [0.8380, -0.5457],
        }
        sentinel_resolved.device = __import__("torch").device("cpu")
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.challenge_scenarios",
                wraps=challenge_scenarios,
            ) as generate,
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout._run_challenge_scenario",
                side_effect=RuntimeError("fixture"),
            ),
        ):
            run_challenge_matrix(
                sentinel_resolved,
                device="cpu",
                position_weight=0.1,
                velocity_weight=0.25,
            )
        generate.assert_called_once_with(
            speed=0.3784,
            forward_heading_world_yaw=math.atan2(-0.5457, 0.8380),
        )

    def test_transactional_save_writes_all_scenarios_and_authenticates_baseline(self):
        baseline = _fixture_challenge_matrix(jerk=100.0)
        baseline_root = self.root / "baseline"
        save_challenge_matrix(baseline, baseline_root)
        top = json.loads(
            (baseline_root / "matrix.json").read_text("utf-8")
        )
        self.assertTrue(top["baseline"])
        self.assertEqual(top["position_weight"], 0.0)
        self.assertEqual(top["velocity_weight"], 0.0)
        self.assertIsNone(top["matrix_pass"])
        self.assertEqual(
            tuple(top["scenario_names"]),
            tuple(scenario.name for scenario in challenge_scenarios()),
        )
        for scenario in challenge_scenarios():
            scenario_root = baseline_root / scenario.name
            self.assertTrue((scenario_root / "rollout.npz").is_file())
            self.assertTrue((scenario_root / "metrics.json").is_file())
            self.assertTrue((scenario_root / "events.jsonl").is_file())
            metrics = json.loads(
                (scenario_root / "metrics.json").read_text("utf-8")
            )
            self.assertEqual(metrics["position_weight"], 0.0)
            self.assertEqual(metrics["velocity_weight"], 0.0)
            self.assertIn("rollout_npz_sha256", metrics)
        resolved_config = json.loads(
            (baseline_root / "resolved_config.json").read_text("utf-8")
        )
        self.assertEqual(
            resolved_config["matcher"][
                "transition_joint_position_weight"
            ],
            0.0,
        )
        self.assertEqual(
            resolved_config["matcher"][
                "transition_joint_velocity_weight"
            ],
            0.0,
        )

        retained = _fixture_challenge_matrix(jerk=80.0)
        retained_root = self.root / "retained"
        save_challenge_matrix(
            retained,
            retained_root,
            baseline_root=baseline_root,
        )
        verdict = json.loads(
            (retained_root / "matrix.json").read_text("utf-8")
        )
        self.assertFalse(verdict["baseline"])
        self.assertTrue(verdict["matrix_pass"])
        self.assertEqual(verdict["material_improvement_count"], 6)

        metrics_path = baseline_root / "rapid-reversal" / "metrics.json"
        altered = json.loads(metrics_path.read_text("utf-8"))
        altered["p95_joint_jerk_rad_s3"] = 1.0
        metrics_path.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaises(ContractError):
            save_challenge_matrix(
                retained,
                self.root / "rejected",
                baseline_root=baseline_root,
            )
        self.assertFalse((self.root / "rejected").exists())

    def test_baseline_save_rejects_nonzero_weights_without_partial_output(self):
        for position_weight, velocity_weight in (
            (0.1, 0.0),
            (0.0, 0.25),
        ):
            with self.subTest(
                position_weight=position_weight,
                velocity_weight=velocity_weight,
            ):
                output = self.root / (
                    f"invalid-{position_weight}-{velocity_weight}"
                )
                with self.assertRaises(ContractError):
                    save_challenge_matrix(
                        _fixture_challenge_matrix(
                            position_weight=position_weight,
                            velocity_weight=velocity_weight,
                        ),
                        output,
                    )
                self.assertFalse(output.exists())

    def test_baseline_load_rejects_self_consistent_nonzero_weight_tampering(self):
        retained = _fixture_challenge_matrix(jerk=80.0)

        for tamper in ("top", "config", "scenario"):
            with self.subTest(tamper=tamper):
                baseline_root = self.root / f"baseline-{tamper}"
                save_challenge_matrix(
                    _fixture_challenge_matrix(), baseline_root
                )
                top_path = baseline_root / "matrix.json"
                top = json.loads(top_path.read_text("utf-8"))
                if tamper == "top":
                    top["position_weight"] = 0.1
                elif tamper == "config":
                    config_path = baseline_root / "resolved_config.json"
                    config = json.loads(config_path.read_text("utf-8"))
                    config["matcher"][
                        "transition_joint_position_weight"
                    ] = 0.1
                    config_path.write_bytes(_canonical_bytes(config))
                    top["resolved_config_json_sha256"] = hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest()
                    top["resolved_config_sha256"] = hashlib.sha256(
                        _canonical_bytes(config)
                    ).hexdigest()
                else:
                    name = "rapid-reversal"
                    metrics_path = (
                        baseline_root / name / "metrics.json"
                    )
                    metrics = json.loads(
                        metrics_path.read_text("utf-8")
                    )
                    metrics["position_weight"] = 0.1
                    metrics_path.write_bytes(_canonical_bytes(metrics))
                    top["scenario_artifacts"][name][
                        "metrics_json_sha256"
                    ] = hashlib.sha256(
                        metrics_path.read_bytes()
                    ).hexdigest()
                top_path.write_bytes(_canonical_bytes(top))
                output = self.root / f"rejected-{tamper}"
                with self.assertRaises(ContractError):
                    save_challenge_matrix(
                        retained,
                        output,
                        baseline_root=baseline_root,
                    )
                self.assertFalse(output.exists())

    def test_save_rejects_output_symlink_without_changing_target(self):
        target = self.root / "target"
        target.mkdir()
        sentinel = target / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        output = self.root / "output"
        output.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ContractError):
            save_challenge_matrix(_fixture_challenge_matrix(), output)
        self.assertTrue(output.is_symlink())
        self.assertEqual(sentinel.read_text("utf-8"), "keep")

    def test_parser_and_main_select_challenge_mode_with_baseline_root(self):
        parser = build_directional_argument_parser()
        args = parser.parse_args(
            [
                "--dataset",
                "DATASET",
                "--config",
                "CONFIG",
                "--challenge-matrix",
                "--baseline-root",
                "BASELINE",
                "--output",
                "OUTPUT",
            ]
        )
        self.assertTrue(args.challenge_matrix)
        self.assertEqual(args.baseline_root, "BASELINE")
        matrix = _fixture_challenge_matrix()
        sentinel_resolved = mock.Mock()
        sentinel_resolved.resolved_config = {
            "matcher": {
                "transition_joint_position_weight": 0.07,
                "transition_joint_velocity_weight": 0.19,
            }
        }
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.load_experiment_config",
                return_value={"schema": "fixture"},
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.resolve_stair_config",
                return_value=sentinel_resolved,
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.run_challenge_matrix",
                return_value=matrix,
            ) as run,
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.save_challenge_matrix",
            ) as save,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(
                main(
                    [
                        "--dataset",
                        "DATASET",
                        "--config",
                        "CONFIG",
                        "--device",
                        "cpu",
                        "--challenge-matrix",
                        "--baseline-root",
                        "BASELINE",
                        "--output",
                        "OUTPUT",
                    ]
                ),
                0,
            )
        run.assert_called_once_with(
            sentinel_resolved,
            device="cpu",
            position_weight=0.07,
            velocity_weight=0.19,
        )
        save.assert_called_once_with(
            matrix,
            Path("OUTPUT").absolute(),
            baseline_root=Path("BASELINE"),
        )


class DirectionalPhaseAndMetricTests(unittest.TestCase):
    def test_phase_boundaries_are_exact(self):
        self.assertEqual(ASCENT_STOP, 280)
        self.assertEqual(REVERSAL_STOP, 380)
        self.assertEqual(STEP_COUNT, 640)
        self.assertEqual(TRANSITION_NEIGHBORHOOD_RADIUS, 4)
        self.assertEqual(directional_phase(0), "ascent")
        self.assertEqual(directional_phase(279), "ascent")
        self.assertEqual(directional_phase(280), "reversal")
        self.assertEqual(directional_phase(379), "reversal")
        self.assertEqual(directional_phase(380), "descent")
        self.assertEqual(directional_phase(639), "descent")

    def test_phase_rejects_indices_outside_the_script(self):
        for index in (-1, 640, 1.5, True):
            with self.subTest(index=index):
                with self.assertRaises(Exception):
                    directional_phase(index)

    def test_transition_neighborhood_marks_exact_plus_or_minus_four(self):
        transitioned = np.zeros(16, dtype=np.bool_)
        transitioned[[1, 10]] = True
        expected = np.zeros(16, dtype=np.bool_)
        expected[0:6] = True
        expected[6:15] = True
        np.testing.assert_array_equal(
            transition_neighborhood_mask(transitioned),
            expected,
        )

    def test_metrics_match_hand_computed_finite_difference_oracle(self):
        # x = [0, 1, 4, 10] at dt=1 has second differences [2, 3] and
        # third difference [1].  The same construction is used for the root.
        arrays = {
            "joint_position": np.array(
                [[0.0], [1.0], [4.0], [10.0]], np.float64
            ),
            "joint_velocity": np.array(
                [[1.0], [2.0], [3.0], [4.0]], np.float64
            ),
            "root_position_world": np.array(
                [
                    [0.0, 0.0, 0.75],
                    [1.0, 0.0, 0.75],
                    [4.0, 0.0, 0.75],
                    [10.0, 0.0, 0.75],
                ],
                np.float64,
            ),
            "foot_position_world": np.array(
                [
                    [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]],
                    [[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]],
                    [[3.0, 0.0, 0.1], [5.0, 0.0, 0.2]],
                    [[6.0, 0.0, 0.1], [9.0, 0.0, 0.2]],
                ],
                np.float64,
            ),
            "foot_clearance_m": np.array(
                [[0.10, 0.20], [0.05, 0.20], [-0.02, 0.10], [0.0, 0.1]]
            ),
            "selected_clip_index": np.array([0, 1, 1, 2], np.int32),
            "previous_selected_clip_index": np.array(
                [0, 0, 1, 1], np.int32
            ),
            "transitioned": np.array([False, True, False, True]),
            "transition_rejected": np.array([False, False, True, False]),
            "terrain_safety_override": np.array([False, True, False, False]),
            "terrain_safety_override_rank": np.array([0, 2, 0, 0]),
            "selected_transition_position_cost": np.array(
                [0.0, 4.0, 0.0, 8.0]
            ),
            "selected_transition_velocity_cost": np.array(
                [0.0, 5.0, 0.0, 10.0]
            ),
            "selected_transition_continuity_cost": np.array(
                [0.0, 9.0, 0.0, 18.0]
            ),
        }
        metrics = compute_directional_metrics(
            arrays,
            dt=1.0,
            position_weight=0.1,
            velocity_weight=0.25,
        )

        aggregate = metrics["aggregate"]
        self.assertEqual(aggregate["frame_count"], 4)
        self.assertEqual(
            aggregate["joint_acceleration_rad_s2"]["samples"], [2.0, 3.0]
        )
        self.assertEqual(
            aggregate["joint_jerk_rad_s3"]["samples"], [1.0]
        )
        self.assertEqual(
            aggregate["root_jerk_m_s3"]["samples"], [1.0]
        )
        self.assertEqual(aggregate["accepted_transition_count"], 2)
        self.assertEqual(aggregate["rejected_transition_count"], 1)
        self.assertEqual(aggregate["cross_clip_transition_count"], 2)
        self.assertEqual(aggregate["minimum_foot_clearance_m"], -0.02)
        self.assertEqual(aggregate["final_root_height_m"], 0.75)
        self.assertEqual(
            aggregate["transition_neighborhood_joint_jerk_rad_s3"][
                "samples"
            ],
            [1.0],
        )
        self.assertEqual(
            aggregate["transition_continuity_cost"]["samples"], [9.0, 18.0]
        )
        self.assertEqual(metrics["position_weight"], 0.1)
        self.assertEqual(metrics["velocity_weight"], 0.25)

    def test_reversal_derivatives_start_inside_frame_280_boundary(self):
        arrays = _metric_arrays()
        arrays["joint_position"][279, 0] = 100.0
        arrays["root_position_world"][279, 0] = 100.0
        arrays["foot_position_world"][279, :, 0] = 100.0
        arrays["transitioned"][280] = True
        metrics = compute_directional_metrics(
            arrays,
            dt=1.0,
            position_weight=0.1,
            velocity_weight=0.25,
        )

        reversal = metrics["phases"]["reversal"]
        self.assertEqual(
            reversal["joint_acceleration_rad_s2"]["count"], 98
        )
        self.assertEqual(reversal["joint_jerk_rad_s3"]["count"], 97)
        self.assertEqual(reversal["root_jerk_m_s3"]["count"], 97)
        self.assertEqual(reversal["contact_foot_speed_m_s"]["count"], 99)
        self.assertEqual(
            reversal["joint_acceleration_rad_s2"]["maximum"], 0.0
        )
        self.assertEqual(reversal["joint_jerk_rad_s3"]["maximum"], 0.0)
        self.assertEqual(reversal["root_jerk_m_s3"]["maximum"], 0.0)
        self.assertEqual(reversal["contact_foot_speed_m_s"]["maximum"], 0.0)
        self.assertEqual(
            reversal["transition_neighborhood_joint_jerk_output_frames"],
            [283, 284],
        )

    def test_descent_derivatives_start_inside_frame_380_boundary(self):
        arrays = _metric_arrays()
        arrays["joint_position"][379, 0] = 100.0
        arrays["root_position_world"][379, 0] = 100.0
        arrays["foot_position_world"][379, :, 0] = 100.0
        arrays["transitioned"][380] = True
        metrics = compute_directional_metrics(
            arrays,
            dt=1.0,
            position_weight=0.1,
            velocity_weight=0.25,
        )

        descent = metrics["phases"]["descent"]
        self.assertEqual(
            descent["joint_acceleration_rad_s2"]["count"], 258
        )
        self.assertEqual(descent["joint_jerk_rad_s3"]["count"], 257)
        self.assertEqual(descent["root_jerk_m_s3"]["count"], 257)
        self.assertEqual(descent["contact_foot_speed_m_s"]["count"], 259)
        self.assertEqual(
            descent["joint_acceleration_rad_s2"]["maximum"], 0.0
        )
        self.assertEqual(descent["joint_jerk_rad_s3"]["maximum"], 0.0)
        self.assertEqual(descent["root_jerk_m_s3"]["maximum"], 0.0)
        self.assertEqual(descent["contact_foot_speed_m_s"]["maximum"], 0.0)
        self.assertEqual(
            descent["transition_neighborhood_joint_jerk_output_frames"],
            [383, 384],
        )

    def test_frame_zero_cross_clip_compares_reset_selection(self):
        arrays = _metric_arrays(frame_count=4)
        arrays["previous_selected_clip_index"][0] = 7
        arrays["selected_clip_index"][0] = 3
        arrays["transitioned"][0] = True
        metrics = compute_directional_metrics(
            arrays,
            dt=1.0,
            position_weight=0.1,
            velocity_weight=0.25,
        )
        self.assertEqual(
            metrics["aggregate"]["cross_clip_transition_count"], 1
        )

    def test_all_frozen_gates_and_qualification_are_explicit(self):
        passing = _metric_arrays()
        nonzero = compute_directional_metrics(
            passing,
            dt=1.0,
            position_weight=0.1,
            velocity_weight=0.25,
        )
        self.assertEqual(
            nonzero["gates"],
            {
                "completed_640_frames": True,
                "returned_to_lower_height": True,
                "minimum_clearance": True,
                "descent_transition_count": True,
                "descent_joint_jerk_p95": True,
                "descent_transition_max_jerk": True,
                "ascent_joint_jerk_p95": True,
            },
        )
        self.assertTrue(nonzero["qualified"])

        baseline = compute_directional_metrics(
            passing,
            dt=1.0,
            position_weight=0.0,
            velocity_weight=0.0,
        )
        self.assertTrue(baseline["baseline"])
        self.assertTrue(all(baseline["gates"].values()))
        self.assertFalse(baseline["qualified"])

        cases = {}
        incomplete = _metric_arrays(frame_count=639)
        cases["completed_640_frames"] = incomplete
        high_root = _metric_arrays()
        high_root["root_position_world"][-1, 2] = 0.8201
        cases["returned_to_lower_height"] = high_root
        low_clearance = _metric_arrays()
        low_clearance["foot_clearance_m"][100, 0] = -0.0301
        cases["minimum_clearance"] = low_clearance
        many_transitions = _metric_arrays()
        many_transitions["transitioned"][380:392] = True
        cases["descent_transition_count"] = many_transitions
        descent_jerk = _metric_arrays()
        descent_index = np.arange(STEP_COUNT - REVERSAL_STOP)
        descent_jerk["joint_position"][REVERSAL_STOP:, 0] = (
            3000.0 * descent_index**3
        )
        cases["descent_joint_jerk_p95"] = descent_jerk
        transition_jerk = _metric_arrays()
        transition_jerk["joint_position"][400, 0] = 60000.0
        transition_jerk["transitioned"][400] = True
        cases["descent_transition_max_jerk"] = transition_jerk
        ascent_jerk = _metric_arrays()
        ascent_index = np.arange(ASCENT_STOP)
        ascent_jerk["joint_position"][:ASCENT_STOP, 0] = (
            2500.0 * ascent_index**3
        )
        cases["ascent_joint_jerk_p95"] = ascent_jerk

        for gate, arrays in cases.items():
            with self.subTest(gate=gate):
                metrics = compute_directional_metrics(
                    arrays,
                    dt=1.0,
                    position_weight=0.1,
                    velocity_weight=0.25,
                )
                self.assertFalse(metrics["gates"][gate])
                self.assertFalse(metrics["qualified"])

    def test_zero_weight_baseline_with_failed_improvement_gates_still_saves(self):
        arrays = _metric_arrays()
        ascent_index = np.arange(ASCENT_STOP)
        arrays["joint_position"][:ASCENT_STOP, 0] = (
            2500.0 * ascent_index**3
        )
        descent_index = np.arange(STEP_COUNT - REVERSAL_STOP)
        arrays["joint_position"][REVERSAL_STOP:, 0] = (
            3000.0 * descent_index**3
        )
        arrays["joint_position"][400, 0] += 60000.0
        arrays["transitioned"][400] = True
        metrics = compute_directional_metrics(
            arrays,
            dt=1.0,
            position_weight=0.0,
            velocity_weight=0.0,
        )
        for gate in (
            "descent_joint_jerk_p95",
            "descent_transition_max_jerk",
            "ascent_joint_jerk_p95",
        ):
            self.assertFalse(metrics["gates"][gate])
        self.assertFalse(metrics["qualified"])

        rollout = DirectionalRollout(
            arrays=MappingProxyType(
                {
                    name: _readonly(value)
                    for name, value in arrays.items()
                }
            ),
            metrics=MappingProxyType(metrics),
            events=(),
            resolved_config=MappingProxyType({"baseline": True}),
            resolved_config_sha256="b" * 64,
            deterministic_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "baseline"
            save_directional_rollout(rollout, output)
            saved = json.loads(
                (output / "metrics.json").read_text("utf-8")
            )
        self.assertTrue(saved["baseline"])
        self.assertFalse(saved["qualified"])


class DirectionalCliAndArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_parser_accepts_exact_weight_overrides(self):
        args = build_directional_argument_parser().parse_args(
            [
                "--dataset",
                "DATASET",
                "--config",
                "CONFIG",
                "--device",
                "cpu",
                "--position-weight",
                "0.1",
                "--velocity-weight",
                "0.25",
                "--output",
                "OUTPUT",
            ]
        )
        self.assertEqual(args.position_weight, 0.1)
        self.assertEqual(args.velocity_weight, 0.25)

    def test_parser_leaves_omitted_weight_overrides_unset(self):
        args = build_directional_argument_parser().parse_args(
            [
                "--dataset",
                "DATASET",
                "--config",
                "CONFIG",
                "--output",
                "OUTPUT",
            ]
        )
        self.assertIsNone(args.position_weight)
        self.assertIsNone(args.velocity_weight)

    def test_parser_rejects_negative_and_nonfinite_weights(self):
        parser = build_directional_argument_parser()
        for option in ("--position-weight", "--velocity-weight"):
            for value in ("-0.1", "nan", "inf", "-inf"):
                with self.subTest(option=option, value=value):
                    argv = [
                        "--dataset",
                        "DATASET",
                        "--config",
                        "CONFIG",
                        option,
                        value,
                        "--output",
                        "OUTPUT",
                    ]
                    with mock.patch("sys.stderr"):
                        with self.assertRaises(SystemExit):
                            parser.parse_args(argv)

    def test_main_records_exact_overrides_and_saves_fixture(self):
        output = self.root / "output"
        fixture = _fixture_rollout()
        sentinel_resolved = object()
        argv = [
            "--dataset",
            "DATASET",
            "--config",
            "CONFIG",
            "--device",
            "cpu",
            "--position-weight",
            "0.1",
            "--velocity-weight",
            "0.25",
            "--output",
            str(output),
        ]
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.load_experiment_config",
                return_value={"schema": "fixture"},
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.resolve_stair_config",
                return_value=sentinel_resolved,
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.run_directional_rollout",
                return_value=fixture,
            ) as run,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(main(argv), 0)
        run.assert_called_once_with(
            sentinel_resolved,
            device="cpu",
            position_weight=0.1,
            velocity_weight=0.25,
        )
        resolved = json.loads(
            (output / "resolved_config.json").read_text("utf-8")
        )
        self.assertEqual(
            resolved["matcher"]["transition_joint_position_weight"], 0.1
        )
        self.assertEqual(
            resolved["matcher"]["transition_joint_velocity_weight"], 0.25
        )

    def test_main_uses_resolved_weights_when_flags_are_omitted(self):
        output = self.root / "resolved-weights"
        fixture = _fixture_rollout()
        sentinel_resolved = mock.Mock()
        sentinel_resolved.resolved_config = {
            "matcher": {
                "transition_joint_position_weight": 0.07,
                "transition_joint_velocity_weight": 0.19,
            }
        }
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.load_experiment_config",
                return_value={"schema": "fixture"},
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.resolve_stair_config",
                return_value=sentinel_resolved,
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.run_directional_rollout",
                return_value=fixture,
            ) as run,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(
                main(
                    [
                        "--dataset",
                        "DATASET",
                        "--config",
                        "CONFIG",
                        "--device",
                        "cpu",
                        "--output",
                        str(output),
                    ]
                ),
                0,
            )
        run.assert_called_once_with(
            sentinel_resolved,
            device="cpu",
            position_weight=0.07,
            velocity_weight=0.19,
        )

    def test_saved_artifacts_are_pickle_free_canonical_and_finite(self):
        rollout = _fixture_rollout()
        output = self.root / "saved"
        save_directional_rollout(rollout, output)
        with np.load(output / "rollout.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(rollout.arrays))
            for name in archive.files:
                self.assertNotEqual(archive[name].dtype, np.dtype(object))

        for name in (
            "metrics.json",
            "resolved_config.json",
        ):
            raw = (output / name).read_bytes()
            value = json.loads(raw)
            self.assertEqual(
                raw,
                (
                    json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8"),
            )
        event_lines = (output / "events.jsonl").read_bytes().splitlines(
            keepends=True
        )
        self.assertEqual(len(event_lines), 1)
        event = json.loads(event_lines[0])
        self.assertEqual(
            event_lines[0],
            (
                json.dumps(
                    event,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )

    def test_save_rejects_non_directory_output_without_replacing_it(self):
        output = self.root / "not-a-directory"
        output.write_text("keep", encoding="utf-8")
        with self.assertRaises(ContractError):
            save_directional_rollout(_fixture_rollout(), output)
        self.assertEqual(output.read_text("utf-8"), "keep")

    def test_save_rejects_directory_symlink_without_changing_target(self):
        target = self.root / "target"
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        output = self.root / "output-link"
        output.symlink_to(target, target_is_directory=True)

        with self.assertRaises(Exception):
            save_directional_rollout(_fixture_rollout(), output)

        self.assertTrue(output.is_symlink())
        self.assertEqual(sentinel.read_text("utf-8"), "keep")

    def test_main_does_not_resolve_away_output_symlink(self):
        target = self.root / "main-target"
        target.mkdir()
        sentinel = target / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        output = self.root / "main-output-link"
        output.symlink_to(target, target_is_directory=True)
        with (
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.load_experiment_config",
                return_value={"schema": "fixture"},
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.resolve_stair_config",
                return_value=object(),
            ),
            mock.patch(
                "mm_sonic.torch_terrain_directional_rollout.run_directional_rollout",
                return_value=_fixture_rollout(),
            ),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(ContractError):
                main(
                    [
                        "--dataset",
                        "DATASET",
                        "--config",
                        "CONFIG",
                        "--output",
                        str(output),
                    ]
                )
        self.assertTrue(output.is_symlink())
        self.assertEqual(sentinel.read_text("utf-8"), "keep")


class DirectionalMatcherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        terrain_rollout_test.TerrainRolloutTests.setUpClass()
        cls.resolved = terrain_rollout_test.TerrainRolloutTests.resolved

    @classmethod
    def tearDownClass(cls):
        terrain_rollout_test.TerrainRolloutTests.tearDownClass()

    def test_real_matcher_emits_exact_dense_640_frame_contract(self):
        input_config = deepcopy(dict(self.resolved.resolved_config))
        first = run_directional_rollout(
            self.resolved,
            device="cpu",
            position_weight=0.1,
            velocity_weight=0.25,
        )
        second = run_directional_rollout(
            self.resolved,
            device="cpu",
            position_weight=0.1,
            velocity_weight=0.25,
        )

        self.assertEqual(first.resolved_config["step_count"], 640)
        self.assertEqual(first.resolved_config["duration_s"], 12.8)
        self.assertEqual(
            first.resolved_config["matcher"][
                "transition_joint_position_weight"
            ],
            0.1,
        )
        self.assertEqual(
            first.resolved_config["matcher"][
                "transition_joint_velocity_weight"
            ],
            0.25,
        )
        self.assertEqual(
            first.deterministic_sha256, second.deterministic_sha256
        )
        self.assertEqual(
            dict(self.resolved.resolved_config),
            input_config,
        )
        arrays = first.arrays
        reset_clip_index = next(
            index
            for index, clip in enumerate(
                self.resolved.dataset.folder.clips
            )
            if clip.relative_path
            == self.resolved.resolved_config["reset_clip"]
        )
        self.assertEqual(
            int(arrays["previous_selected_clip_index"][0]),
            reset_clip_index,
        )
        np.testing.assert_array_equal(
            arrays["previous_selected_clip_index"][1:],
            arrays["selected_clip_index"][:-1],
        )
        for name, trailing in (
            ("joint_position", (29,)),
            ("joint_velocity", (29,)),
            ("root_position_world", (3,)),
            ("foot_position_world", (2, 3)),
            ("foot_velocity_world", (2, 3)),
            ("foot_clearance_m", (2,)),
            ("selected_clip_index", ()),
            ("previous_selected_clip_index", ()),
            ("selected_frame", ()),
            ("transitioned", ()),
            ("transition_rejected", ()),
            ("terrain_safety_override", ()),
            ("terrain_safety_override_rank", ()),
            ("selected_transition_position_cost", ()),
            ("selected_transition_velocity_cost", ()),
            ("selected_transition_continuity_cost", ()),
            ("motion_feature_cost", ()),
            ("terrain_feature_cost", ()),
            ("total_feature_cost", ()),
            ("selected_total_cost", ()),
        ):
            self.assertEqual(arrays[name].shape, (640,) + trailing)
        direction = np.asarray(
            self.resolved.resolved_config[
                "reference_direction_matcher_xy"
            ]
        )
        command = direction * float(
            self.resolved.resolved_config["command_speed_mps"]
        )
        np.testing.assert_allclose(
            arrays["command_velocity_world_xy"][:ASCENT_STOP],
            np.repeat(command[None, :], ASCENT_STOP, axis=0),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            arrays["command_velocity_world_xy"][ASCENT_STOP:],
            np.repeat(
                -command[None, :], STEP_COUNT - ASCENT_STOP, axis=0
            ),
            atol=1e-7,
        )
        self.assertAlmostEqual(
            float(arrays["command_heading_world_yaw"][279]),
            math.atan2(float(command[1]), float(command[0])),
            places=6,
        )
        self.assertAlmostEqual(
            float(arrays["command_heading_world_yaw"][280]),
            math.atan2(float(-command[1]), float(-command[0])),
            places=6,
        )
        extension = self.resolved.measurement_extension
        foot = arrays["foot_position_world"][123]
        scene_xy = extension.alignment.matcher_to_scene_xy(
            __import__("torch").tensor(foot[:, :2])
        )
        surface = extension.query_grid.sample_xy(scene_xy).numpy()
        np.testing.assert_allclose(
            arrays["foot_clearance_m"][123],
            foot[:, 2] - surface,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
