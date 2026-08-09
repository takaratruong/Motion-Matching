from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.evaluate_g1_pfnn_vertical_slice import (
    evaluate_script,
    vertical_slice_command_script,
)
from mm_sonic.terrain_pfnn.runtime import TerrainSample


class _Runtime:
    def __init__(self, *, hold_tick: int | None = None) -> None:
        self.tick = 0
        self.hold_tick = hold_tick
        self.x = 0.0

    def step(self, command, camera_yaw=0.0):
        del camera_yaw
        command = np.asarray(command, dtype=np.float64)
        self.x += float(command[0]) / 30.0
        diagnostics = {"realized_speed_m_s": abs(float(command[0]))}
        if self.tick == self.hold_tick:
            diagnostics["hold_reason"] = "fixture"
        self.tick += 1
        return SimpleNamespace(
            root_position_world=np.array((self.x, 0.0, 0.8)),
            joint_position_isaaclab=np.zeros(29),
            phase=0.1,
            contact_probability=np.array((1.0, 1.0, 0.0, 0.0)),
            diagnostics=diagnostics,
        )


def _terrain(xy):
    point = np.asarray(xy, dtype=np.float64)
    return TerrainSample(float(0.1 * point[0]), np.array((0.1, 0.0)))


class EvaluateG1PFNNVerticalSliceTest(unittest.TestCase):
    def test_script_contains_start_turns_signed_grades_and_stop(self) -> None:
        script = vertical_slice_command_script()
        self.assertEqual(sum(segment.ticks for segment in script), 600)
        self.assertEqual(
            {segment.name for segment in script},
            {"start", "straight", "left_turn", "right_turn", "ascent", "descent", "stop"},
        )

    def test_script_rejects_first_hold_and_accepts_finite_signed_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "hold_reason at tick 7"):
            evaluate_script(_Runtime(hold_tick=7), _terrain, vertical_slice_command_script())
        result = evaluate_script(
            _Runtime(), _terrain, vertical_slice_command_script(), minimum_grade_degrees=1.0
        )
        self.assertEqual(result["ticks"], 600)
        self.assertGreater(result["maximum_signed_grade_degrees"], 1.0)
        self.assertLess(result["minimum_signed_grade_degrees"], -1.0)


if __name__ == "__main__":
    unittest.main()
