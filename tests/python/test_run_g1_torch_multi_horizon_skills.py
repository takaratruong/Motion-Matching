import json
import math
import unittest

from mm_sonic.torch_terrain_skill_horizon_rollout import HorizonChunkEvent
from mm_sonic.torch_terrain_skill_horizon_search import HorizonCost
from resources.run_g1_torch_multi_horizon_skills import (
    build_parser,
    canonical_chunk_events,
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
        self.assertIsNone(args.swing_clearance_margin_m)
        self.assertIsNone(args.swing_plan_sigma_frames)
        self.assertFalse(args.swing_foot_geometry)
        self.assertEqual(args.foot_correction_halflife_s, 0.04)
        locked = build_parser().parse_args(
            [
                "--dataset", "data", "--config", "config.json",
                "--g1-xml", "g1.xml", "--output", "out",
                "--device", "cuda:3", "--foot-lock", "--contact-phase-gate",
                "--swing-clearance-margin-m", "0.01",
                "--swing-plan-sigma-frames", "4.0",
                "--swing-foot-geometry",
                "--foot-correction-halflife-s", "0.02",
            ]
        )
        self.assertTrue(locked.foot_lock)
        self.assertTrue(locked.contact_phase_gate)
        self.assertEqual(locked.swing_clearance_margin_m, 0.01)
        self.assertEqual(locked.swing_plan_sigma_frames, 4.0)
        self.assertTrue(locked.swing_foot_geometry)
        self.assertEqual(locked.foot_correction_halflife_s, 0.02)

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
