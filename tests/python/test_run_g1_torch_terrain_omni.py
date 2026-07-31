import unittest

from resources.run_g1_torch_terrain_omni import build_parser


class TerrainOmniCliTests(unittest.TestCase):
    def test_contact_policy_and_latency_flags_are_explicit(self):
        arguments = build_parser().parse_args(
            [
                "--dataset",
                "motions",
                "--config",
                "contact.json",
                "--g1-xml",
                "g1.xml",
                "--output",
                "matrix",
                "--contact-segments",
                "--maximum-step-time-ms",
                "1000",
            ]
        )

        self.assertTrue(arguments.contact_segments)
        self.assertEqual(arguments.maximum_step_time_ms, 1000.0)


if __name__ == "__main__":
    unittest.main()
