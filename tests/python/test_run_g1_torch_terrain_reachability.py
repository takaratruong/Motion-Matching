import json
import os
from pathlib import Path
import tempfile
import unittest

from resources.run_g1_torch_terrain_reachability import build_parser, main


class TerrainReachabilityCliTests(unittest.TestCase):
    def test_parser_requires_frozen_inputs_and_output(self):
        parser = build_parser()
        args = parser.parse_args(
            (
                "--dataset",
                "expanded",
                "--config",
                "expanded.json",
                "--normalization-dataset",
                "small",
                "--normalization-config",
                "small.json",
                "--output",
                "evidence.json",
            )
        )
        self.assertEqual(args.dataset, "expanded")
        self.assertEqual(args.config, "expanded.json")
        self.assertEqual(args.normalization_dataset, "small")
        self.assertEqual(args.normalization_config, "small.json")
        self.assertEqual(args.output, "evidence.json")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.route, [])

    @unittest.skipUnless(
        os.environ.get("RUN_G1_TERRAIN_REACHABILITY_ORACLE") == "1",
        "real captured-state reachability replay is opt-in",
    )
    def test_real_replay_matches_canonical_stable_evidence(self):
        repository = Path(__file__).resolve().parents[2]
        expected_path = (
            repository
            / "docs"
            / "superpowers"
            / "results"
            / (
                "2026-07-30-g1-curb-and-transition-"
                "reachability-evidence.json"
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            observed_path = Path(temporary) / "evidence.json"
            exit_code = main(
                [
                    "--dataset",
                    str(
                        repository
                        / "build"
                        / "torch-terrain-expanded-combined-v42"
                    ),
                    "--config",
                    str(
                        repository
                        / "sonic"
                        / "configs"
                        / "experiments"
                        / "torch_terrain_expanded.json"
                    ),
                    "--normalization-dataset",
                    str(repository / "build" / "torch-stair-small"),
                    "--normalization-config",
                    str(
                        repository
                        / "sonic"
                        / "configs"
                        / "experiments"
                        / "torch_stair_small_terrain_weight3.json"
                    ),
                    "--output",
                    str(observed_path),
                    "--device",
                    "cuda",
                ]
            )
            self.assertEqual(exit_code, 0)
            observed = json.loads(observed_path.read_text("utf-8"))
        expected = json.loads(expected_path.read_text("utf-8"))
        for payload in (expected, observed):
            for run in payload["runs"]:
                for diagnostic in run["diagnostics"]:
                    diagnostic.pop("planner_compute_ns")
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
