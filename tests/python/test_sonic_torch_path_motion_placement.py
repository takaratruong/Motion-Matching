import math
import unittest

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    nominal_footprint_path,
)
from mm_sonic.torch_path_motion_placement import (
    contact_signature_cost,
    extract_raw_motion_windows,
    path_contact_signature,
)


class PathMotionPlacementTests(unittest.TestCase):
    @staticmethod
    def nominal_path():
        return nominal_footprint_path(
            ConstantHeadingRequest(
                start_foot_scene_xy=torch.tensor(
                    ((0.0, 0.1), (0.0, -0.1))
                ),
                start_support=torch.tensor((True, True)),
                heading_scene_xy=torch.tensor((1.0, 0.0)),
                distance_m=1.0,
                speed_mps=0.5,
                stride_m=0.25,
                step_width_m=0.2,
                frames_per_second=50.0,
            )
        )

    def test_path_signature_uses_heading_coordinates_and_relative_heights(self):
        path = self.nominal_path()

        def surface(points):
            return torch.where(
                (points[..., 0] >= 0.45) & (points[..., 0] <= 0.8),
                torch.where(
                    points[..., 1] < 0.0,
                    torch.full(points.shape[:-1], 0.18),
                    torch.zeros(points.shape[:-1]),
                ),
                torch.zeros(points.shape[:-1]),
            )

        signature = path_contact_signature(path=path, sample_surface=surface)

        self.assertEqual(signature.foot_order, (0, 1, 0, 1))
        self.assertEqual(signature.forward_m, (0.25, 0.5, 0.75, 1.0))
        np.testing.assert_allclose(
            signature.lateral_m, (0.1, -0.1, 0.1, -0.1)
        )
        self.assertEqual(signature.contact_frame, (25, 50, 75, 100))
        np.testing.assert_allclose(
            signature.height_pattern_m, (0.0, 0.18, 0.0, 0.0)
        )

    def test_extracts_label_independent_level_split_level_window(self):
        frames = 90
        roots = np.zeros((frames, 3))
        roots[:, 0] = np.linspace(0.0, 1.4, frames)
        quaternions = np.zeros((frames, 4))
        quaternions[:, 0] = 1.0
        feet = np.zeros((frames, 2, 3))
        feet[:, 0, 1] = 0.1
        feet[:, 1, 1] = -0.1
        support = np.zeros((frames, 2), dtype=np.bool_)
        surface = np.zeros((frames, 2))
        events = ((0, 0, 0.0), (0, 1, 0.0), (15, 0, 0.0),
                  (30, 1, 0.18), (45, 0, 0.0), (60, 1, 0.18),
                  (75, 0, 0.0), (82, 1, 0.0))
        for frame, foot, height in events:
            stop = min(frames, frame + 7)
            support[frame:stop, foot] = True
            feet[frame:stop, foot, 0] = roots[frame, 0]
            feet[frame:stop, foot, 2] = height + 0.035
            surface[frame:stop, foot] = height

        windows = extract_raw_motion_windows(
            source_clip="arbitrary-family-name",
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            foot_position_world=feet,
            support_mask=support,
            foot_surface_height_m=surface,
        )

        self.assertTrue(windows)
        self.assertTrue(
            any(
                len(window.events) >= 6
                and max(event.surface_height_m for event in window.events)
                >= 0.17
                and window.start_frame == 0
                for window in windows
            )
        )

    def test_contact_cost_prefers_matching_height_pattern_and_heading(self):
        path = self.nominal_path()

        def surface(points):
            return torch.where(
                points[..., 1] < 0.0,
                torch.full(points.shape[:-1], 0.18),
                torch.zeros(points.shape[:-1]),
            )

        signature = path_contact_signature(path=path, sample_surface=surface)
        roots = np.zeros((70, 3))
        roots[:, 0] = np.linspace(0.0, 1.0, 70)
        quaternions = np.zeros((70, 4))
        quaternions[:, 0] = 1.0
        feet = np.zeros((70, 2, 3))
        feet[:, 0, 1] = 0.1
        feet[:, 1, 1] = -0.1
        support = np.zeros((70, 2), dtype=np.bool_)
        heights = np.zeros((70, 2))
        for frame, foot in ((5, 0), (20, 1), (35, 0), (50, 1)):
            support[frame:frame + 5, foot] = True
            feet[frame:frame + 5, foot, 0] = roots[frame, 0]
            value = 0.18 if foot == 1 else 0.0
            heights[frame:frame + 5, foot] = value
            feet[frame:frame + 5, foot, 2] = value + 0.035
        matching = extract_raw_motion_windows(
            source_clip="matching",
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            foot_position_world=feet,
            support_mask=support,
            foot_surface_height_m=heights,
            minimum_events=4,
            maximum_events=4,
        )[0]
        flat_heights = np.zeros_like(heights)
        flat_feet = feet.copy()
        flat_feet[..., 2] = 0.035
        flat = extract_raw_motion_windows(
            source_clip="flat",
            root_position_world=roots,
            root_orientation_world_wxyz=quaternions,
            foot_position_world=flat_feet,
            support_mask=support,
            foot_surface_height_m=flat_heights,
            minimum_events=4,
            maximum_events=4,
        )[0]

        self.assertLess(
            contact_signature_cost(signature, matching),
            contact_signature_cost(signature, flat),
        )

    def test_rejects_zero_event_bounds(self):
        with self.assertRaises(ContractError):
            extract_raw_motion_windows(
                source_clip="clip",
                root_position_world=np.zeros((4, 3)),
                root_orientation_world_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (4, 1)),
                foot_position_world=np.zeros((4, 2, 3)),
                support_mask=np.zeros((4, 2), dtype=np.bool_),
                foot_surface_height_m=np.zeros((4, 2)),
                minimum_events=0,
            )


if __name__ == "__main__":
    unittest.main()
