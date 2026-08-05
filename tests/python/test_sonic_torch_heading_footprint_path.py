import math
import unittest

import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    heading_local_to_scene,
    nominal_footprint_path,
    scene_to_heading_local,
)


class HeadingFootprintPathTests(unittest.TestCase):
    def test_heading_coordinates_round_trip_at_arbitrary_yaw(self):
        heading = torch.tensor(
            (math.sqrt(0.5), math.sqrt(0.5)), dtype=torch.float32
        )
        origin = torch.tensor((1.0, -0.3), dtype=torch.float32)
        local = torch.tensor(
            ((0.4, -0.1), (0.9, 0.2)), dtype=torch.float32
        )

        scene = heading_local_to_scene(local, origin, heading)

        torch.testing.assert_close(
            scene_to_heading_local(scene, origin, heading), local
        )

    def test_path_alternates_and_finishes_on_contact_boundary(self):
        request = ConstantHeadingRequest(
            start_foot_scene_xy=torch.tensor(
                ((0.0, 0.12), (0.0, -0.12)), dtype=torch.float32
            ),
            start_support=torch.tensor((True, True)),
            heading_scene_xy=torch.tensor((0.6, 0.8)),
            distance_m=1.0,
            speed_mps=0.4,
            stride_m=0.25,
            step_width_m=0.24,
            frames_per_second=50.0,
        )

        path = nominal_footprint_path(request)

        self.assertEqual(len(path.footprints), 4)
        self.assertTrue(
            all(
                left.foot != right.foot
                for left, right in zip(
                    path.footprints, path.footprints[1:]
                )
            )
        )
        self.assertAlmostEqual(path.progress_m, 1.0)
        self.assertEqual(path.footprints[-1].contact_frame, 125)

    def test_single_support_moves_the_swing_foot_first(self):
        request = ConstantHeadingRequest(
            start_foot_scene_xy=torch.tensor(
                ((0.0, 0.12), (0.0, -0.12)), dtype=torch.float32
            ),
            start_support=torch.tensor((True, False)),
            heading_scene_xy=torch.tensor((1.0, 0.0)),
            distance_m=0.5,
            speed_mps=0.5,
            stride_m=0.25,
            step_width_m=0.24,
            frames_per_second=50.0,
        )

        path = nominal_footprint_path(request)

        self.assertEqual(path.footprints[0].foot, 1)

    def test_zero_heading_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "heading"):
            ConstantHeadingRequest(
                start_foot_scene_xy=torch.zeros((2, 2)),
                start_support=torch.tensor((True, True)),
                heading_scene_xy=torch.zeros(2),
                distance_m=0.5,
                speed_mps=0.5,
                stride_m=0.25,
                step_width_m=0.24,
                frames_per_second=50.0,
            )


if __name__ == "__main__":
    unittest.main()
