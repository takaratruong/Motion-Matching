import json
import math
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest import mock

import numpy as np

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
        with self.assertRaises(Exception):
            save_directional_rollout(_fixture_rollout(), output)
        self.assertEqual(output.read_text("utf-8"), "keep")


class DirectionalMatcherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        terrain_rollout_test.TerrainRolloutTests.setUpClass()
        cls.resolved = terrain_rollout_test.TerrainRolloutTests.resolved

    @classmethod
    def tearDownClass(cls):
        terrain_rollout_test.TerrainRolloutTests.tearDownClass()

    def test_real_matcher_emits_exact_dense_640_frame_contract(self):
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
        arrays = first.arrays
        for name, trailing in (
            ("joint_position", (29,)),
            ("joint_velocity", (29,)),
            ("root_position_world", (3,)),
            ("foot_position_world", (2, 3)),
            ("foot_velocity_world", (2, 3)),
            ("foot_clearance_m", (2,)),
            ("selected_clip_index", ()),
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
