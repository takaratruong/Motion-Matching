from copy import deepcopy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_terrain_directional_rollout import (
    ASCENT_STOP,
    REVERSAL_STOP,
    STEP_COUNT,
    TRANSITION_NEIGHBORHOOD_RADIUS,
    DirectionalRollout,
    build_directional_argument_parser,
    compute_directional_metrics,
    directional_phase,
    main,
    run_directional_rollout,
    save_directional_rollout,
    transition_neighborhood_mask,
)
from tests.python import test_sonic_torch_terrain_rollout as terrain_rollout_test


def _readonly(value):
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


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
