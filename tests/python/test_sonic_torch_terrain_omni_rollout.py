import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from mm_sonic.torch_terrain_omni_rollout import (
    KinematicSample,
    evaluate_route_outcome,
    run_omni_matrix,
    save_omni_matrix,
)
from mm_sonic.torch_terrain_omni_routes import (
    OmniRoute,
    RouteOutcomeContract,
    RouteCommand,
    StairFrame,
)
from mm_sonic.torch_terrain_rollout import load_experiment_config


def _routes():
    return (
        OmniRoute(
            "first",
            (
                RouteCommand((0.2, 0.0), 0.0, 4, "move", True),
                RouteCommand((0.0, 0.0), 0.0, 4, "stop"),
            ),
            "mixed",
        ),
        OmniRoute(
            "second",
            (
                RouteCommand((0.0, 0.2), 1.0, 2, "move", True),
                RouteCommand((0.0, 0.0), 1.0, 3, "stop"),
            ),
            "mixed",
        ),
    )


class _Diagnostics:
    def __init__(self, frame, timing):
        self.selected_clip_path = "flat/motion.npz"
        self.selected_frame = frame
        self.terrain_safety_override = frame % 3 == 0
        self.step_time_ns = timing + frame
        self.search_time_ns = timing + frame + 1


class _Result:
    def __init__(self, frame, timing):
        self.frame = frame
        self.diagnostics = _Diagnostics(frame, timing)


class _FakeMatcher:
    def __init__(self, *, fail_at, timing, ledger):
        self._frame = 0
        self._fail_at = fail_at
        self._timing = timing
        self._ledger = ledger

    def reset(self):
        self._ledger.append(("reset", None))
        return _Result(-1, self._timing)

    def prepare_step(self, velocity, heading, *, dt):
        self._ledger.append(("prepare", (velocity, heading, dt)))
        return self._frame

    def commit(self, prepared):
        if prepared == self._fail_at:
            raise RuntimeError("synthetic route failure")
        self._ledger.append(("commit", prepared))
        result = _Result(self._frame, self._timing)
        self._frame += 1
        return result


def _kinematics(result):
    frame = float(result.frame)
    feet = np.array(
        [[frame * 0.001, -0.1, 0.035], [frame * 0.001, 0.1, 0.035]]
    )
    return KinematicSample(
        qpos=np.full(36, frame),
        joint_position=np.full(29, frame),
        joint_velocity=np.zeros(29),
        root_position_world=np.array([frame * 0.001, 0.0, 0.8]),
        root_yaw_world=0.0,
        foot_position_world=feet,
    )


class OmnidirectionalRolloutContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = StairFrame((0.0, 0.0), 0.0, 0.6223, 0.3302, 0.1778, 3)

    def test_every_command_frame_commits_once_and_arrays_are_complete(self):
        ledgers = []

        def factory(_route):
            ledger = []
            ledgers.append(ledger)
            return _FakeMatcher(fail_at=None, timing=100, ledger=ledger)

        matrix = run_omni_matrix(
            routes=_routes(),
            stair_frame=self.frame,
            matcher_factory=factory,
            kinematics=_kinematics,
            terrain_sampler=lambda xy: np.zeros(2),
            dataset_identity="dataset-a",
            config_identity="config-a",
        )

        self.assertTrue(matrix.matrix_pass)
        self.assertEqual([run.completed_frames for run in matrix.runs], [8, 5])
        for route, run, ledger in zip(_routes(), matrix.runs, ledgers):
            self.assertEqual(sum(item[0] == "reset" for item in ledger), 1)
            self.assertEqual(
                sum(item[0] == "commit" for item in ledger),
                sum(command.frames for command in route.commands),
            )
            self.assertEqual(
                set(run.arrays),
                {
                    "command_velocity_world_xy",
                    "command_heading_world_yaw",
                    "command_segment_index",
                    "qpos",
                    "joint_position",
                    "joint_velocity",
                    "root_position_world",
                    "root_yaw_world",
                    "foot_position_world",
                    "foot_surface_height_m",
                    "selected_clip_path",
                    "selected_source_frame",
                    "terrain_rescue",
                    "step_time_ns",
                    "search_time_ns",
                },
            )

    def test_route_exception_is_isolated_and_timing_is_not_hashed(self):
        def execute(timing):
            return run_omni_matrix(
                routes=_routes(),
                stair_frame=self.frame,
                matcher_factory=lambda route: _FakeMatcher(
                    fail_at=7 if route.name == "first" else None,
                    timing=timing,
                    ledger=[],
                ),
                kinematics=_kinematics,
                terrain_sampler=lambda xy: np.zeros(2),
                dataset_identity="dataset-a",
                config_identity="config-a",
            )

        first = execute(100)
        second = execute(900)
        self.assertFalse(first.matrix_pass)
        self.assertEqual(first.runs[0].completed_frames, 7)
        self.assertEqual(first.runs[0].failure.stage, "commit")
        self.assertEqual(first.runs[0].failure.frame_index, 7)
        self.assertEqual(first.runs[0].failure.exception_type, "RuntimeError")
        self.assertTrue(first.runs[1].completed_without_exception)
        self.assertEqual(first.deterministic_sha256, second.deterministic_sha256)

    def test_matrix_artifacts_are_saved_transactionally_and_pickle_free(self):
        matrix = run_omni_matrix(
            routes=_routes(),
            stair_frame=self.frame,
            matcher_factory=lambda _route: _FakeMatcher(
                fail_at=None, timing=100, ledger=[]
            ),
            kinematics=_kinematics,
            terrain_sampler=lambda xy: np.zeros(2),
            dataset_identity="dataset-a",
            config_identity="config-a",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "matrix"
            save_omni_matrix(matrix, output)
            self.assertTrue((output / "matrix.json").is_file())
            for route in _routes():
                route_dir = output / "routes" / route.name
                self.assertTrue((route_dir / "arrays.npz").is_file())
                self.assertTrue((route_dir / "metrics.json").is_file())
                with np.load(route_dir / "arrays.npz", allow_pickle=False) as arrays:
                    self.assertIn("foot_surface_height_m", arrays)
                payload = json.loads((route_dir / "metrics.json").read_text())
                self.assertTrue(payload["outcome"]["completed"])
                self.assertEqual(payload["outcome"]["failure_reasons"], [])
            with self.assertRaises(FileExistsError):
                save_omni_matrix(matrix, output)

    def test_route_outcome_requires_progress_terrain_and_final_state(self):
        route = OmniRoute(
            "qualified-mount",
            (RouteCommand((0.2, 0.0), 0.0, 51, "mount", True),),
            "mount",
            RouteOutcomeContract(
                required_segments=("mount",),
                min_segment_progress_ratio=0.5,
                min_elevated_foot_samples=20,
                final_surface="elevated",
                final_heading_error_max_rad=0.2,
            ),
        )
        frame_count = 51
        root = np.zeros((frame_count, 3))
        root[:, 0] = np.linspace(0.0, 0.15, frame_count)
        arrays = {
            "root_position_world": root,
            "root_yaw_world": np.zeros(frame_count),
            "command_velocity_world_xy": np.tile((0.2, 0.0), (frame_count, 1)),
            "command_heading_world_yaw": np.zeros(frame_count),
            "command_segment_index": np.zeros(frame_count, dtype=np.int32),
            "foot_surface_height_m": np.full((frame_count, 2), 0.1778),
        }

        outcome = evaluate_route_outcome(route, arrays)

        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.failure_reasons, ())
        self.assertAlmostEqual(outcome.segment_progress_ratio[0][1], 0.75)

        no_progress = dict(arrays)
        no_progress["root_position_world"] = np.zeros_like(root)
        failed = evaluate_route_outcome(route, no_progress)
        self.assertFalse(failed.completed)
        self.assertIn("segment:mount:progress", failed.failure_reasons)

    def test_matrix_pass_requires_behavioral_outcome_completion(self):
        route = OmniRoute(
            "cannot-mount",
            (RouteCommand((0.2, 0.0), 0.0, 4, "move", True),),
            "mount",
            RouteOutcomeContract(
                required_segments=("move",),
                min_segment_progress_ratio=0.1,
                min_elevated_foot_samples=1,
                final_surface="elevated",
            ),
        )
        matrix = run_omni_matrix(
            routes=(route,),
            stair_frame=self.frame,
            matcher_factory=lambda _route: _FakeMatcher(
                fail_at=None, timing=100, ledger=[]
            ),
            kinematics=_kinematics,
            terrain_sampler=lambda xy: np.zeros(2),
            dataset_identity="dataset-a",
            config_identity="config-a",
        )

        self.assertTrue(matrix.runs[0].completed_without_exception)
        self.assertFalse(
            matrix.runs[0].metrics.required_outcome_completed
        )
        self.assertFalse(matrix.matrix_pass)

    def test_retained_omni_configs_pin_terrain_weight_three(self):
        project = Path(__file__).resolve().parents[2]
        expanded = load_experiment_config(
            project
            / "sonic/configs/experiments/torch_terrain_expanded.json"
        )
        normalization = load_experiment_config(
            project
            / "sonic/configs/experiments/"
            "torch_stair_small_terrain_weight3.json"
        )

        self.assertEqual(expanded["conditions"]["dense"]["weight"], 3.0)
        self.assertEqual(
            normalization["conditions"]["dense"]["weight"], 3.0
        )
        self.assertEqual(expanded["conditions"]["legacy"]["weight"], 4.0)
        self.assertEqual(normalization["conditions"]["legacy"]["weight"], 4.0)


if __name__ == "__main__":
    unittest.main()
