import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_stair_pivot_viewer",
    ROOT / "resources" / "run_g1_stair_pivot_viewer.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunG1StairPivotViewerTest(unittest.TestCase):
    def test_parser_exposes_headless_contact_sheet(self):
        args = MODULE._parser().parse_args(
            (
                "--connector",
                "connector.npz",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--render-contact-sheet",
                "sheet.png",
            )
        )

        self.assertEqual(args.render_contact_sheet, Path("sheet.png"))

    def test_parser_accepts_native_motionbricks_frame_rate(self):
        args = MODULE._parser().parse_args(
            (
                "--connector",
                "connector.npz",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--frames-per-second",
                "30",
            )
        )

        self.assertEqual(args.frames_per_second, 30.0)

    def test_loader_accepts_exact_connector_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connector.npz"
            np.savez_compressed(
                path,
                joint_position=np.zeros((5, 29)),
                root_position_world=np.zeros((5, 3)),
                root_orientation_world_wxyz=np.tile(
                    (1.0, 0.0, 0.0, 0.0),
                    (5, 1),
                ),
            )

            arrays = MODULE._load_connector(path)

        self.assertEqual(arrays["joint_position"].shape, (5, 29))


if __name__ == "__main__":
    unittest.main()
