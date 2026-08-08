from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import math
import unittest

import numpy as np


class TerrainPFNNViewerTests(unittest.TestCase):
    def test_help_and_interactive_defaults_do_not_load_runtime(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _parser, main

        arguments = _parser().parse_args([])
        self.assertGreaterEqual(arguments.max_steps, 1_000_000)
        self.assertEqual(arguments.device, "cuda")
        self.assertFalse(arguments.strict_envelope)

        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--checkpoint", output.getvalue())
        self.assertIn("--no-viewer", output.getvalue())
        self.assertIn("--strict-envelope", output.getvalue())

    def test_wasd_command_is_bounded_and_release_stops(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _command_from_pressed

        self.assertTrue(np.array_equal(_command_from_pressed(set(), 0.8), (0.0, 0.0)))
        self.assertTrue(np.array_equal(_command_from_pressed({"w"}, 0.8), (0.8, 0.0)))
        self.assertTrue(np.array_equal(_command_from_pressed({"s"}, 0.8), (-0.8, 0.0)))
        diagonal = _command_from_pressed({"w", "a"}, 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(diagonal)), 0.8, places=12)

    def test_runtime_callback_uses_the_rendered_triangle_height_and_grade(self) -> None:
        from mm_sonic.terrain_pfnn.hill_map import TerrainPFNNHillMap
        from mm_sonic.terrain_pfnn_viewer import _HillTerrainCallback

        terrain = TerrainPFNNHillMap()
        callback = _HillTerrainCallback(terrain)
        hill = terrain.hills[-1]
        point = np.asarray((hill.start_x + terrain.blend_m + 0.5, 0.0))
        sample = callback(point)

        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample.height_m, terrain.height_at(point))
        measured = math.degrees(math.atan(float(sample.gradient_xy[0])))
        self.assertAlmostEqual(measured, 18.9, delta=0.05)
        self.assertIsNone(callback((terrain.x_max + 0.1, 0.0)))


if __name__ == "__main__":
    unittest.main()
