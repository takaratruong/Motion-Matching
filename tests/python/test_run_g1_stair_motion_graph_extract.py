import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_stair_motion_graph_extract",
    ROOT / "resources" / "run_g1_stair_motion_graph_extract.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunG1StairMotionGraphExtractTest(unittest.TestCase):
    def test_load_metrics_salvages_safe_prefix_from_failed_route(self):
        payload = {
            "completed_frames": 266,
            "completed_without_exception": False,
            "deterministic_sha256": "a" * 64,
            "failure": {
                "exception_type": "HorizonSearchFailure",
                "frame_index": 266,
            },
            "metrics": {
                "penetration_depth_m": {"maximum": 0.0},
                "support_height_error_m": {"p95": 0.019},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory)
            (route / "metrics.json").write_text(json.dumps(payload))

            loaded = MODULE._load_metrics(route)

        self.assertEqual(loaded["completed_frames"], 266)

    def test_parser_accepts_multiple_matrix_shards(self):
        args = MODULE._parser().parse_args(
            (
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--matrix-directory",
                "shard-a",
                "--matrix-directory",
                "shard-b",
                "--output",
                "graph.json",
            )
        )

        self.assertEqual(
            args.matrix_directory,
            (Path("shard-a"), Path("shard-b")),
        )

    def test_parser_requires_full_sole_validation_inputs(self):
        with self.assertRaises(SystemExit):
            MODULE._parser().parse_args(
                (
                    "--dataset",
                    "dataset",
                    "--matrix-directory",
                    "shard-a",
                    "--output",
                    "graph.json",
                )
            )

    def test_load_metrics_rejects_non_numeric_contact_metrics(self):
        payload = {
            "completed_frames": 3,
            "deterministic_sha256": "a" * 64,
            "metrics": {
                "penetration_depth_m": {"maximum": "invalid"},
                "support_height_error_m": {"p95": 0.0},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory)
            (route / "metrics.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ContractError, "contact-safe"):
                MODULE._load_metrics(route)

    def test_sole_inputs_reject_missing_or_malformed_qpos(self):
        arrays = {
            "joint_position": np.zeros((3, 29)),
            "root_position_world": np.zeros((3, 3)),
        }
        with self.assertRaisesRegex(ContractError, "arrays"):
            MODULE._sole_inputs(arrays)

        arrays["qpos"] = np.zeros((2, 36))
        with self.assertRaisesRegex(ContractError, "arrays"):
            MODULE._sole_inputs(arrays)


if __name__ == "__main__":
    unittest.main()
