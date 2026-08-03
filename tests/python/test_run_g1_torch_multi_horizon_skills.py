import io
import json
import math
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from mm_sonic.torch_terrain_skill_horizon_rollout import HorizonChunkEvent
from mm_sonic.torch_terrain_skill_horizon_search import HorizonCost
from resources.run_g1_torch_multi_horizon_skills import (
    build_parser,
    canonical_chunk_events,
    main,
    search_config_from_experiment,
    select_routes,
)


class MultiHorizonSkillsCliTest(unittest.TestCase):
    def test_parser_requires_explicit_device_and_accepts_qualification_slice(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "--dataset", "data", "--config", "config.json",
                    "--g1-xml", "g1.xml", "--output", "out",
                ]
            )
        args = build_parser().parse_args(
            [
                "--dataset", "data", "--config", "config.json",
                "--g1-xml", "g1.xml", "--output", "out",
                "--device", "cuda:3", "--qualification-slice",
            ]
        )
        self.assertEqual(args.device, "cuda:3")
        self.assertTrue(args.qualification_slice)
        self.assertEqual(args.ablation, "combined")
        self.assertFalse(args.foot_lock)
        self.assertFalse(args.contact_phase_gate)
        self.assertFalse(args.continuous_skill)
        self.assertFalse(args.emitted_contact_preview)
        self.assertEqual(args.maximum_emitted_contact_candidates, 64)
        self.assertIsNone(args.maximum_emitted_contact_rescue_candidates)
        self.assertIsNone(args.swing_clearance_margin_m)
        self.assertIsNone(args.swing_plan_sigma_frames)
        self.assertIsNone(args.maximum_source_contact_p95_m)
        self.assertIsNone(args.normalization_source)
        self.assertIsNone(args.turning_source_corpus)
        self.assertEqual(args.maximum_translation_warp_m, 0.0)
        self.assertEqual(args.maximum_yaw_warp_rad, 0.0)
        self.assertEqual(args.minimum_endpoint_warp_yaw_rad, 0.0)
        self.assertAlmostEqual(args.maximum_endpoint_warp_yaw_rad, math.pi)
        self.assertTrue(math.isinf(args.maximum_endpoint_warp_terrain_delta_m))
        self.assertEqual(
            args.minimum_endpoint_warp_velocity_heading_alignment, -1.0
        )
        self.assertEqual(args.foot_correction_halflife_s, 0.04)
        self.assertEqual(args.maximum_output_joint_speed_rad_s, 13.0)
        self.assertIsNone(args.root_height_correction_halflife_s)
        self.assertIsNone(args.touchdown_projection_max_shift_m)
        self.assertFalse(args.anticipatory_touchdown_projection)
        self.assertEqual(args.maximum_contact_anchor_yaw_rad, 0.0)
        locked = build_parser().parse_args(
            [
                "--dataset", "data", "--config", "config.json",
                "--g1-xml", "g1.xml", "--output", "out",
                "--device", "cuda:3", "--foot-lock", "--contact-phase-gate",
                "--continuous-skill",
                "--emitted-contact-preview",
                "--maximum-emitted-contact-candidates", "512",
                "--maximum-emitted-contact-rescue-candidates", "1024",
                "--swing-clearance-margin-m", "0.01",
                "--swing-plan-sigma-frames", "4.0",
                "--foot-correction-halflife-s", "0.02",
                "--maximum-output-joint-speed-rad-s", "16.0",
                "--root-height-correction-halflife-s", "0.05",
                "--touchdown-projection-max-shift-m", "0.05",
                "--anticipatory-touchdown-projection",
                "--maximum-contact-anchor-yaw-rad", "0.2",
                "--maximum-source-contact-p95-m", "0.02",
                "--normalization-source", "baseline",
                "--turning-source-corpus", "baseline",
                "--maximum-translation-warp-m", "0.15",
                "--maximum-yaw-warp-rad", "0.30",
                "--minimum-endpoint-warp-yaw-rad", "0.30",
                "--maximum-endpoint-warp-yaw-rad", "0.90",
                "--maximum-endpoint-warp-terrain-delta-m", "0.02",
                "--minimum-endpoint-warp-velocity-heading-alignment", "0.80",
            ]
        )
        self.assertTrue(locked.foot_lock)
        self.assertTrue(locked.contact_phase_gate)
        self.assertTrue(locked.continuous_skill)
        self.assertTrue(locked.emitted_contact_preview)
        self.assertEqual(locked.maximum_emitted_contact_candidates, 512)
        self.assertEqual(
            locked.maximum_emitted_contact_rescue_candidates, 1024
        )
        self.assertEqual(locked.swing_clearance_margin_m, 0.01)
        self.assertEqual(locked.swing_plan_sigma_frames, 4.0)
        self.assertEqual(locked.foot_correction_halflife_s, 0.02)
        self.assertEqual(locked.maximum_output_joint_speed_rad_s, 16.0)
        self.assertEqual(locked.root_height_correction_halflife_s, 0.05)
        self.assertEqual(locked.touchdown_projection_max_shift_m, 0.05)
        self.assertTrue(locked.anticipatory_touchdown_projection)
        self.assertEqual(locked.maximum_contact_anchor_yaw_rad, 0.2)
        self.assertEqual(locked.maximum_source_contact_p95_m, 0.02)
        self.assertEqual(locked.normalization_source, "baseline")
        self.assertEqual(locked.turning_source_corpus, "baseline")
        self.assertEqual(locked.maximum_translation_warp_m, 0.15)
        self.assertEqual(locked.maximum_yaw_warp_rad, 0.30)
        self.assertEqual(locked.minimum_endpoint_warp_yaw_rad, 0.30)
        self.assertEqual(locked.maximum_endpoint_warp_yaw_rad, 0.90)
        self.assertEqual(locked.maximum_endpoint_warp_terrain_delta_m, 0.02)
        self.assertEqual(
            locked.minimum_endpoint_warp_velocity_heading_alignment, 0.80
        )

    def test_route_selection_preserves_frozen_order_and_rejects_bad_names(self):
        routes = select_routes(
            ["diagonal-down-left", "cross-tread-left-to-right"], False
        )
        self.assertEqual(
            [route.name for route in routes],
            ["cross-tread-left-to-right", "diagonal-down-left"],
        )
        self.assertEqual(len(select_routes([], True)), 6)
        with self.assertRaisesRegex(ValueError, "repeated"):
            select_routes(["diagonal-down-left", "diagonal-down-left"], False)
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_routes(["does-not-exist"], False)

    def test_main_propagates_output_speed_cap_to_rollout_and_summary(self):
        matrix = SimpleNamespace(
            runs=[SimpleNamespace(completed_without_exception=True)],
            matrix_pass=True,
            deterministic_sha256="matrix-sha",
        )
        output = io.StringIO()
        module = "resources.run_g1_torch_multi_horizon_skills"
        with (
            patch(f"{module}.load_experiment_config", return_value={}),
            patch(
                f"{module}.search_config_from_experiment",
                return_value=SimpleNamespace(turn_gate_rad=0.5),
            ),
            patch(f"{module}.resolve_stair_config", return_value="resolved"),
            patch(
                f"{module}.run_resolved_horizon_matrix",
                return_value=(matrix, {"route": ()}),
            ) as run_matrix,
            patch(f"{module}.save_horizon_matrix"),
            redirect_stdout(output),
        ):
            return_code = main(
                [
                    "--dataset", "data",
                    "--config", "config.json",
                    "--g1-xml", "g1.xml",
                    "--output", "out",
                    "--device", "cuda:3",
                    "--maximum-output-joint-speed-rad-s", "16.0",
                ]
            )

        self.assertEqual(return_code, 0)
        self.assertEqual(
            run_matrix.call_args.kwargs["maximum_output_joint_speed_rad_s"],
            16.0,
        )
        self.assertEqual(
            json.loads(output.getvalue())["maximum_output_joint_speed_rad_s"],
            16.0,
        )

    def test_search_config_is_strict_and_pins_horizons(self):
        descriptor = {
            "horizons": [25, 50, 100],
            "maximum_endpoint_lateness_frames": 25,
            "entry_weight": 1.0 / 27.0,
            "displacement_weight": 8.0,
            "yaw_weight": 3.0,
            "height_weight": 4.0,
            "duration_weight": 0.25,
            "stall_weight": 0.05,
            "moving_speed_mps": 0.1,
            "minimum_progress_m": 0.05,
            "turn_gate_rad": math.radians(30.0),
            "surface_gate_m": 0.05,
        }
        config = search_config_from_experiment({"multi_horizon": descriptor})
        self.assertEqual(config.displacement_weight, 8.0)
        with self.assertRaisesRegex(ValueError, "fields"):
            search_config_from_experiment(
                {"multi_horizon": {**descriptor, "unexpected": 1}}
            )

    def test_chunk_event_json_is_canonical_and_mapping_order_independent(self):
        event = HorizonChunkEvent(
            entry_row=4,
            skill_index=2,
            target_frames=50,
            endpoint_frame_exclusive=61,
            cost=HorizonCost(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0, 21.0),
            rejected_by_reason={"terrain": 2, "turn": 1},
            release_reason="endpoint",
        )
        first = canonical_chunk_events((event,))
        reordered = HorizonChunkEvent(
            **{**event.__dict__, "rejected_by_reason": {"turn": 1, "terrain": 2}}
        )
        self.assertEqual(first, canonical_chunk_events((reordered,)))
        payload = json.loads(first)
        self.assertEqual(payload["schema"], "g1-terrain-horizon-chunks/v1")
        self.assertEqual(payload["chunks"][0]["total_cost"], 21.0)
        self.assertEqual(len(payload["deterministic_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
