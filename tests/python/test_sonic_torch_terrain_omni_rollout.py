import unittest

import numpy as np

from mm_sonic.torch_terrain_omni_rollout import (
    KinematicSample,
    run_omni_matrix,
)
from mm_sonic.torch_terrain_omni_routes import (
    OmniRoute,
    RouteCommand,
    StairFrame,
)


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


if __name__ == "__main__":
    unittest.main()
