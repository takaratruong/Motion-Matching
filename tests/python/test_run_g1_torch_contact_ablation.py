import unittest
from unittest import mock

from resources import run_g1_torch_contact_ablation as cli


class ContactAblationCliTests(unittest.TestCase):
    def test_main_forwards_qualified_inputs(self):
        result = {"deterministic_sha256": "a" * 64, "state_count": 92}
        with mock.patch.object(cli, "run_contact_ablation", return_value=result) as run:
            status = cli.main(
                [
                    "--dataset", "dataset",
                    "--terrain-config", "terrain.json",
                    "--oracle-config", "oracle.json",
                    "--g1-xml", "g1.xml",
                    "--baseline-artifacts", "baseline",
                    "--quality-oracle-artifacts", "quality",
                    "--output", "output",
                    "--device", "cuda:0",
                    "--max-actions-per-state", "3",
                    "--projection-strategy", "stance-root",
                    "--selection-strategy", "contact-anchored",
                    "--root-correction-scale", "0.5",
                    "--root-smoothing-passes", "2",
                    "--joint-smoothing-passes", "3",
                    "--reproject-smoothed-joints",
                    "--reprojection-blend", "0.1",
                ]
            )

        self.assertEqual(status, 0)
        run.assert_called_once_with(
            dataset="dataset",
            terrain_config="terrain.json",
            oracle_config="oracle.json",
            g1_xml="g1.xml",
            baseline_artifacts="baseline",
            quality_oracle_artifacts="quality",
            output="output",
            device="cuda:0",
            maximum_actions_per_state=3,
            projection_strategy="stance-root",
            selection_strategy="contact-anchored",
            root_correction_scale=0.5,
            root_smoothing_passes=2,
            joint_smoothing_passes=3,
            reproject_smoothed_joints=True,
            reprojection_blend=0.1,
        )


if __name__ == "__main__":
    unittest.main()
