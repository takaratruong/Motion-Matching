import unittest

from resources.run_g1_torch_terrain_omni import build_parser


class TerrainOmniCliTests(unittest.TestCase):
    def test_foothold_ablation_arm_is_explicit(self):
        arguments = build_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "contact.json",
                "--g1-xml", "g1.xml",
                "--output", "matrix",
                "--contact-segments",
                "--foothold-arm", "hybrid",
                "--foothold-height-tolerance-m", "0.08",
            ]
        )

        self.assertEqual(arguments.foothold_arm, "hybrid")
        self.assertEqual(arguments.foothold_height_tolerance_m, 0.08)

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

    def test_transition_continuity_overrides_are_explicit(self):
        arguments = build_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "contact.json",
                "--g1-xml", "g1.xml",
                "--output", "matrix",
                "--transition-joint-position-weight", "1.0",
                "--transition-joint-velocity-weight", "2.0",
            ]
        )

        self.assertEqual(arguments.transition_joint_position_weight, 1.0)
        self.assertEqual(arguments.transition_joint_velocity_weight, 2.0)


if __name__ == "__main__":
    unittest.main()
