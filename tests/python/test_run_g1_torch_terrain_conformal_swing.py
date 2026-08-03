import tempfile
import unittest
from pathlib import Path

import torch

from resources.run_g1_torch_terrain_conformal_swing import (
    _quality,
    build_parser,
    require_new_output,
    resolve_frame_interval,
)


class TerrainConformalSwingCliTests(unittest.TestCase):
    def test_parser_requires_explicit_real_inputs(self):
        arguments = build_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "terrain.json",
                "--g1-xml", "g1.xml",
                "--arrays", "route/arrays.npz",
                "--foot", "1",
                "--start-frame", "217",
                "--end-frame-exclusive", "248",
                "--output", "evidence",
                "--device", "cuda:3",
                "--reference-weight", "2.0",
                "--smoothness-weight", "80.0",
                "--project-landing",
                "--foothold-search-radius", "0.16",
            ]
        )

        self.assertEqual(arguments.foot, 1)
        self.assertEqual(arguments.seed, 41)
        self.assertEqual(arguments.device, "cuda:3")
        self.assertEqual(arguments.reference_weight, 2.0)
        self.assertEqual(arguments.smoothness_weight, 80.0)
        self.assertEqual(arguments.clearance_weight, 500.0)
        self.assertTrue(arguments.project_landing)
        self.assertEqual(arguments.foothold_search_radius, 0.16)
        self.assertEqual(arguments.foothold_search_step, 0.01)
        self.assertEqual(arguments.maximum_foothold_height_range, 0.025)

    def test_frame_interval_is_strict_and_end_exclusive(self):
        self.assertEqual(resolve_frame_interval(300, 217, 248), (217, 248))
        with self.assertRaisesRegex(ValueError, "interval"):
            resolve_frame_interval(300, 248, 248)
        with self.assertRaisesRegex(ValueError, "interval"):
            resolve_frame_interval(300, 217, 301)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            require_new_output(output)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                require_new_output(output)

    def test_quality_lifts_planar_foot_offsets_into_world_points(self):
        path = torch.tensor(
            ((0.0, 0.0, 0.10), (0.1, 0.0, 0.20), (0.2, 0.0, 0.10))
        )
        toe = torch.tensor(((0.08, 0.0),) * 3)
        heel = torch.tensor(((-0.05, 0.0),) * 3)

        quality = _quality(
            path,
            toe,
            heel,
            lambda points: torch.zeros(
                points.shape[:-1], dtype=points.dtype, device=points.device
            ),
        )

        self.assertEqual(quality["clearance_violation_count"], 0)
        self.assertAlmostEqual(quality["minimum_point_clearance_m"], 0.10)


if __name__ == "__main__":
    unittest.main()
