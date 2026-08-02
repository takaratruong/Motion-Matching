import unittest
from unittest import mock

from resources import run_g1_torch_motion_quality_oracle as cli


class MotionQualityOracleCliTests(unittest.TestCase):
    def test_parser_requires_all_research_identities(self):
        args = cli.build_parser().parse_args(
            [
                "--dataset", "dataset",
                "--terrain-config", "terrain.json",
                "--oracle-config", "oracle.json",
                "--g1-xml", "g1.xml",
                "--baseline-artifacts", "baseline",
                "--output", "output",
                "--device", "cuda:7",
            ]
        )
        self.assertEqual(args.dataset, "dataset")
        self.assertEqual(args.baseline_artifacts, "baseline")
        self.assertEqual(args.device, "cuda:7")

    def test_main_passes_resolved_arguments_to_one_orchestration_boundary(self):
        result = {"deterministic_sha256": "b" * 64, "state_count": 7}
        with mock.patch.object(cli, "run_quality_oracle", return_value=result) as run:
            status = cli.main(
                [
                    "--dataset", "dataset",
                    "--terrain-config", "terrain.json",
                    "--oracle-config", "oracle.json",
                    "--g1-xml", "g1.xml",
                    "--baseline-artifacts", "baseline",
                    "--output", "output",
                    "--device", "cpu",
                ]
            )

        self.assertEqual(status, 0)
        run.assert_called_once_with(
            dataset="dataset",
            terrain_config="terrain.json",
            oracle_config="oracle.json",
            g1_xml="g1.xml",
            baseline_artifacts="baseline",
            output="output",
            device="cpu",
        )


if __name__ == "__main__":
    unittest.main()
