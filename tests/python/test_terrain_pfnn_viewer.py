from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


class TerrainPFNNViewerTests(unittest.TestCase):
    def test_viewer_loads_raw_pfnn_runtime(self) -> None:
        from mm_sonic.terrain_pfnn.runtime import TerrainPFNNRuntime
        from mm_sonic.terrain_pfnn_viewer import _load_runtime, _parser

        arguments = _parser().parse_args([])
        terrain = object()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                '{"dataset_digest_sha256":"' + ("a" * 64) + '"}',
                encoding="utf-8",
            )
            with mock.patch.object(
                TerrainPFNNRuntime, "from_checkpoint", return_value=object()
            ) as factory:
                _load_runtime(
                    arguments,
                    Path("checkpoint.pt"),
                    manifest,
                    Path("g1.xml"),
                    terrain,
                )

        self.assertIs(factory.call_args.kwargs["command_driven_root"], False)

    def test_help_and_interactive_defaults_do_not_load_runtime(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _parser, main

        arguments = _parser().parse_args([])
        self.assertGreaterEqual(arguments.max_steps, 1_000_000)
        self.assertEqual(arguments.device, "cuda")
        self.assertFalse(hasattr(arguments, "strict_envelope"))

        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--checkpoint", output.getvalue())
        self.assertIn("--no-viewer", output.getvalue())
        self.assertNotIn("--strict-envelope", output.getvalue())

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

    def test_default_viewer_course_has_lateral_recovery_room(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _viewer_terrain_map

        terrain = _viewer_terrain_map()
        self.assertGreaterEqual(terrain.y_max, 10.0)
        x = terrain.hills[1].start_x + terrain.blend_m + 0.25
        self.assertEqual(terrain.height_at((x, -9.0)), terrain.height_at((x, 9.0)))

    def test_camera_defaults_show_the_longitudinal_hill_profile(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _configure_camera

        viewer = SimpleNamespace(
            cam=SimpleNamespace(azimuth=0.0, elevation=0.0, distance=0.0)
        )
        _configure_camera(viewer)
        self.assertEqual(viewer.cam.azimuth, 90.0)
        self.assertEqual(viewer.cam.elevation, -18.0)
        self.assertEqual(viewer.cam.distance, 4.0)

    def test_preview_root_quaternion_keeps_yaw_and_removes_tilt(self) -> None:
        from mm_sonic.terrain_pfnn_viewer import _upright_yaw_quaternion

        rolled = np.asarray((math.cos(0.2), math.sin(0.2), 0.0, 0.0))
        np.testing.assert_allclose(
            _upright_yaw_quaternion(rolled), (1.0, 0.0, 0.0, 0.0), atol=1.0e-12
        )
        yaw = 0.75
        yawed = np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
        np.testing.assert_allclose(
            _upright_yaw_quaternion(yawed), yawed, atol=1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
