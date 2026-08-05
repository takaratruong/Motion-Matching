import importlib.util
from pathlib import Path
import unittest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_heading_footprint_planner.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_heading_footprint_planner", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class HeadingFootprintPlannerRunnerTests(unittest.TestCase):
    def test_parser_requires_heading_distance_and_speed(self):
        args = _MODULE.parser().parse_args(
            [
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--start-state",
                "start.npz",
                "--heading-degrees",
                "45",
                "--distance-m",
                "1.0",
                "--speed-mps",
                "0.4",
                "--output",
                "out",
            ]
        )

        self.assertEqual(args.heading_degrees, 45.0)
        self.assertEqual(args.distance_m, 1.0)
        self.assertFalse(hasattr(args, "lane_scene_y"))
        self.assertFalse(hasattr(args, "source_start"))

    def test_failure_json_distinguishes_motion_coverage(self):
        record = _MODULE.failure_record(
            code="no_motion_coverage",
            heading_degrees=45.0,
            step_index=3,
            attempted_footprints=8,
            attempted_actions=512,
            reasons=("height tolerance",),
        )

        self.assertEqual(record["code"], "no_motion_coverage")
        self.assertEqual(record["step_index"], 3)
        self.assertEqual(
            record["schema"], "g1-heading-footprint-failure/v1"
        )


if __name__ == "__main__":
    unittest.main()
