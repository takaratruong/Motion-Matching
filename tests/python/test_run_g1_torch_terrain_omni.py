import unittest

from resources.run_g1_torch_terrain_omni import build_parser


class TerrainOmniCliTests(unittest.TestCase):
    def test_layered_graph_arm_is_available_for_ablation(self):
        arguments = build_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "contact.json",
                "--g1-xml", "g1.xml",
                "--output", "matrix",
                "--contact-segments",
                "--foothold-arm", "layered-graph-hybrid",
            ]
        )

        self.assertEqual(arguments.foothold_arm, "layered-graph-hybrid")

    def test_turn_path_cost_weights_are_explicit(self):
        arguments = build_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "contact.json",
                "--g1-xml", "g1.xml",
                "--output", "matrix",
                "--turn-root-lateral-cost-weight", "1200",
                "--turn-root-progress-cost-weight", "75",
                "--turn-sequence-candidate-count", "32",
                "--heading-maintenance-yaw-cost-weight", "12",
                "--turn-lateral-root-warp-gain", "0.4",
                "--small-turn-lateral-root-warp-gain", "0.3",
                "--reversal-lateral-root-warp-gain", "0.2",
                "--strafe-action-gate",
            ]
        )

        self.assertEqual(arguments.turn_root_lateral_cost_weight, 1200.0)
        self.assertEqual(arguments.turn_root_progress_cost_weight, 75.0)
        self.assertEqual(arguments.turn_sequence_candidate_count, 32)
        self.assertEqual(arguments.heading_maintenance_yaw_cost_weight, 12.0)
        self.assertEqual(arguments.turn_lateral_root_warp_gain, 0.4)
        self.assertEqual(arguments.small_turn_lateral_root_warp_gain, 0.3)
        self.assertEqual(arguments.reversal_lateral_root_warp_gain, 0.2)
        self.assertTrue(arguments.strafe_action_gate)

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
