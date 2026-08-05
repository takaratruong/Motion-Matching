import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.torch_heading_footprint_path import (
    NominalFootprint,
    NominalFootprintPath,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    TerrainFootprintCandidate,
    TerrainFootprintLayer,
)

_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_heading_footprint_planner.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_heading_footprint_planner", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class HeadingFootprintPlannerRunnerTests(unittest.TestCase):
    def test_two_contact_action_rejects_hidden_recontact(self):
        action = SimpleNamespace(
            start_frame=2,
            end_frame=12,
            landing_feet=(0, 1),
            landing_frame_offsets=(3, 8),
        )
        support = np.ones((14, 2), dtype=np.bool_)
        support[3:5, 0] = False
        support[3:5, 1] = False
        support[6:10, 1] = False

        self.assertFalse(
            _MODULE.action_has_exact_declared_contacts(action, support)
        )
        support[3:5, 1] = True
        self.assertTrue(
            _MODULE.action_has_exact_declared_contacts(action, support)
        )
        support[11, 0] = False
        self.assertFalse(
            _MODULE.action_has_exact_declared_contacts(action, support)
        )

    def test_candidate_pruning_keeps_lowest_cost_terrain_placements(self):
        nominal = NominalFootprint(
            step_index=0,
            foot=0,
            center_scene_xy=torch.zeros(2),
            center_heading_xy=torch.zeros(2),
            yaw_scene_rad=0.0,
            contact_frame=1,
        )
        candidates = tuple(
            TerrainFootprintCandidate(
                step_index=0,
                foot=0,
                center_scene_xy=torch.tensor((float(index), 0.0)),
                center_heading_xy=torch.tensor((float(index), 0.0)),
                offset_heading_xy=torch.zeros(2),
                yaw_scene_rad=0.0,
                surface_height_m=0.0,
                placement_cost=float(cost),
            )
            for index, cost in enumerate((3.0, 1.0, 2.0, 0.5))
        )

        result = _MODULE.prune_footprint_layers(
            (
                TerrainFootprintLayer(
                    step_index=0,
                    nominal=nominal,
                    candidates=candidates,
                ),
            ),
            maximum_candidates=2,
        )

        self.assertEqual(
            [item.placement_cost for item in result[0].candidates],
            [0.5, 1.0],
        )

    def test_candidate_pruning_preserves_distinct_terrain_levels(self):
        nominal = NominalFootprint(
            step_index=0,
            foot=0,
            center_scene_xy=torch.zeros(2),
            center_heading_xy=torch.zeros(2),
            yaw_scene_rad=0.0,
            contact_frame=1,
        )
        candidates = tuple(
            TerrainFootprintCandidate(
                step_index=0,
                foot=0,
                center_scene_xy=torch.tensor((float(index), 0.0)),
                center_heading_xy=torch.tensor((float(index), 0.0)),
                offset_heading_xy=torch.zeros(2),
                yaw_scene_rad=0.0,
                surface_height_m=height,
                placement_cost=cost,
            )
            for index, (height, cost) in enumerate(
                ((0.0, 0.1), (0.0, 0.2), (0.20, 4.0))
            )
        )

        result = _MODULE.prune_footprint_layers(
            (
                TerrainFootprintLayer(
                    step_index=0,
                    nominal=nominal,
                    candidates=candidates,
                ),
            ),
            maximum_candidates=2,
        )

        self.assertEqual(
            [item.surface_height_m for item in result[0].candidates],
            [0.0, 0.20],
        )

    def test_terminal_swing_geometry_is_expressed_in_source_heading(self):
        body_position = np.zeros((5, 3, 3), dtype=np.float64)
        body_position[3, 1] = (1.0, 2.0, 0.4)
        body_position[3, 2] = (0.8, 2.4, 0.3)
        body_quaternion = np.zeros((5, 3, 4), dtype=np.float64)
        body_quaternion[..., 0] = 1.0
        body_quaternion[0, 0] = (
            math.sqrt(0.5),
            0.0,
            0.0,
            math.sqrt(0.5),
        )

        relative = _MODULE.terminal_swing_relative_heading(
            body_position_world=body_position,
            body_quaternion_world_wxyz=body_quaternion,
            start_frame=0,
            terminal_frame=3,
            root_body_index=0,
            foot_body_indices=(1, 2),
            swing_foot=1,
        )

        np.testing.assert_allclose(relative, (0.4, 0.2, -0.1))

    def test_edge_approach_inserts_same_level_contacts_before_drop(self):
        heading = torch.tensor((1.0, 0.0))
        nominal = tuple(
            NominalFootprint(
                step_index=index,
                foot=index,
                center_scene_xy=torch.tensor((0.3 + 0.05 * index, 0.0)),
                center_heading_xy=torch.tensor(
                    (0.3 + 0.05 * index, 0.1 - 0.2 * index)
                ),
                yaw_scene_rad=0.0,
                contact_frame=30 + 30 * index,
            )
            for index in range(2)
        )
        path = NominalFootprintPath(
            origin_scene_xy=torch.zeros(2),
            heading_scene_xy=heading,
            footprints=nominal,
            progress_m=0.35,
        )
        prefix = tuple(
            TerrainFootprintCandidate(
                step_index=index,
                foot=index,
                center_scene_xy=item.center_scene_xy
                + torch.tensor((0.02, 0.0)),
                center_heading_xy=item.center_heading_xy
                + torch.tensor((0.02, 0.0)),
                offset_heading_xy=torch.zeros(2),
                yaw_scene_rad=0.0,
                surface_height_m=0.4 + 0.1 * index,
                placement_cost=0.0,
            )
            for index, item in enumerate(nominal)
        )

        result = _MODULE.edge_approach_path(
            path=path,
            prefix_footprints=prefix,
            advance_m=0.15,
            speed_mps=0.4,
            frames_per_second=50.0,
        )

        self.assertEqual(len(result.footprints), 4)
        self.assertEqual(
            [item.foot for item in result.footprints], [0, 1, 0, 1]
        )
        self.assertAlmostEqual(
            float(result.footprints[0].center_heading_xy[0]), 0.32
        )
        self.assertAlmostEqual(
            float(result.footprints[2].center_heading_xy[0]), 0.47
        )
        self.assertAlmostEqual(
            float(result.footprints[3].center_heading_xy[0]), 0.52
        )
        self.assertGreater(
            result.footprints[2].contact_frame,
            result.footprints[1].contact_frame,
        )

    def test_parser_requires_heading_distance_and_speed(self):
        args = _MODULE.parser().parse_args(
            [
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--g1-xml",
                "g1.xml",
                "--start-state",
                "start.npz",
                "--heading-degrees",
                "45",
                "--distance-m",
                "1.0",
                "--speed-mps",
                "0.4",
                "--output",
                "out",
                "--motionbricks",
                "motionbricks",
            ]
        )

        self.assertEqual(args.heading_degrees, 45.0)
        self.assertEqual(args.distance_m, 1.0)
        self.assertFalse(hasattr(args, "lane_scene_y"))
        self.assertFalse(hasattr(args, "source_start"))
        self.assertEqual(args.motionbricks, Path("motionbricks"))
        self.assertEqual(args.motionbricks_num_tokens, 8)
        self.assertEqual(args.edge_approach_m, 0.15)
        self.assertFalse(hasattr(args, "motionbricks_drop_template"))

    def test_failure_json_distinguishes_motion_coverage(self):
        record = _MODULE.failure_record(
            code="no_motion_coverage",
            heading_degrees=45.0,
            step_index=3,
            attempted_footprints=8,
            attempted_actions=512,
            reasons=("height tolerance",),
        )

        self.assertEqual(record["code"], "no_motion_coverage")
        self.assertEqual(record["step_index"], 3)
        self.assertEqual(
            record["schema"], "g1-heading-footprint-failure/v1"
        )


if __name__ == "__main__":
    unittest.main()
