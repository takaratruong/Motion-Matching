import math
import unittest

import torch

from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    nominal_footprint_path,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    FootprintCandidateFailure,
    terrain_footprint_layers,
)


def request(heading=(1.0, 0.0)):
    return ConstantHeadingRequest(
        start_foot_scene_xy=torch.tensor(
            ((0.0, 0.12), (0.0, -0.12)), dtype=torch.float32
        ),
        start_support=torch.tensor((True, True)),
        heading_scene_xy=torch.tensor(heading, dtype=torch.float32),
        distance_m=0.5,
        speed_mps=0.4,
        stride_m=0.25,
        step_width_m=0.24,
        frames_per_second=50.0,
    )


SOLE = torch.tensor(
    (
        (-0.10, -0.04),
        (-0.10, 0.04),
        (0.12, -0.04),
        (0.12, 0.04),
    ),
    dtype=torch.float32,
)


class TerrainFootprintCandidateTests(unittest.TestCase):
    def test_center_valid_but_toe_over_edge_is_rejected(self):
        path = nominal_footprint_path(request())

        def edge(points):
            return torch.where(
                points[..., 0] <= 0.30,
                torch.zeros(points.shape[:-1]),
                torch.full(points.shape[:-1], -0.20),
            )

        layers = terrain_footprint_layers(
            path=path,
            sole_offsets_by_foot=(SOLE, SOLE),
            sample_surface=edge,
            forward_offsets_m=(0.0, -0.05, -0.10),
            lateral_offsets_m=(0.0,),
            maximum_surface_variation_m=0.025,
            edge_safety_margin_m=0.01,
        )

        self.assertTrue(layers[0].candidates)
        self.assertAlmostEqual(
            layers[0].candidates[0].offset_heading_xy[0].item(),
            -0.10,
        )
        self.assertLess(
            layers[0].candidates[0].center_scene_xy[0],
            path.footprints[0].center_scene_xy[0],
        )

    def test_global_rotation_preserves_local_candidate_offsets(self):
        outputs = []
        for heading in (
            (1.0, 0.0),
            (math.sqrt(0.5), math.sqrt(0.5)),
        ):
            outputs.append(
                terrain_footprint_layers(
                    path=nominal_footprint_path(request(heading)),
                    sole_offsets_by_foot=(SOLE, SOLE),
                    sample_surface=lambda points: torch.zeros(
                        points.shape[:-1], dtype=points.dtype
                    ),
                    forward_offsets_m=(-0.05, 0.0, 0.05),
                    lateral_offsets_m=(-0.05, 0.0, 0.05),
                    maximum_surface_variation_m=0.025,
                    edge_safety_margin_m=0.01,
                )
            )

        first = torch.stack(
            [
                candidate.offset_heading_xy
                for candidate in outputs[0][0].candidates
            ]
        )
        second = torch.stack(
            [
                candidate.offset_heading_xy
                for candidate in outputs[1][0].candidates
            ]
        )
        torch.testing.assert_close(first, second)

    def test_empty_layer_is_classified_as_terrain_failure(self):
        path = nominal_footprint_path(request())

        def discontinuous(points):
            return 4.0 * points[..., 0]

        with self.assertRaises(FootprintCandidateFailure) as caught:
            terrain_footprint_layers(
                path=path,
                sole_offsets_by_foot=(SOLE, SOLE),
                sample_surface=discontinuous,
                forward_offsets_m=(0.0,),
                lateral_offsets_m=(0.0,),
                maximum_surface_variation_m=0.025,
                edge_safety_margin_m=0.01,
            )

        self.assertEqual(caught.exception.code, "no_terrain_footprint")
        self.assertEqual(caught.exception.step_index, 0)
        self.assertEqual(caught.exception.attempted_offsets, 1)


if __name__ == "__main__":
    unittest.main()
