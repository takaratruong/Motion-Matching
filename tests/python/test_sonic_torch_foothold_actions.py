from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import torch

from mm_sonic.joints import ContractError

try:
    from mm_sonic.torch_foothold_actions import (
        FootholdAction,
        FootholdActionIndex,
        action_from_profiles,
    )
except ImportError:
    @dataclass(frozen=True)
    class FootholdAction:
        clip_index: int
        start_frame: int
        end_frame: int
        start_support: tuple[bool, bool]
        landing_feet: tuple[int, int]
        landing_frame_offsets: tuple[int, int]
        landing_xy_start_frame_m: torch.Tensor
        landing_height_delta_m: torch.Tensor
        root_displacement_m: torch.Tensor
        root_yaw_delta_rad: torch.Tensor
        minimum_swing_clearance_m: float
        maximum_unsupported_frames: int

        def __post_init__(self):
            raise AssertionError("foothold action descriptor is missing")

    def action_from_profiles(*_args, **_kwargs):
        raise AssertionError("foothold action extraction is missing")

    class FootholdActionIndex:
        @classmethod
        def from_dataset(cls, *_args, **_kwargs):
            raise AssertionError("foothold action index is missing")


def _two_contact_profiles():
    support = torch.tensor(
        [
            [True, True],
            [True, False],
            [True, False],
            [True, True],
            [False, True],
            [True, True],
        ],
        dtype=torch.bool,
    )
    foot_xy = torch.tensor(
        [
            [[0.00, 0.00], [0.00, 0.20]],
            [[0.00, 0.00], [0.00, 0.20]],
            [[0.00, 0.00], [0.20, 0.20]],
            [[0.00, 0.00], [0.30, 0.20]],
            [[0.00, 0.00], [0.30, 0.20]],
            [[0.30, 0.00], [0.30, 0.20]],
        ],
        dtype=torch.float32,
    )
    surface = torch.tensor(
        [
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.18],
            [0.00, 0.18],
            [0.18, 0.18],
        ],
        dtype=torch.float32,
    )
    root_xy = torch.tensor(
        [
            [1.00, 2.00],
            [1.00, 2.00],
            [1.05, 2.00],
            [1.15, 2.00],
            [1.20, 2.00],
            [1.30, 2.00],
        ],
        dtype=torch.float32,
    )
    root_yaw = torch.full((6,), torch.pi / 2, dtype=torch.float32)
    return support, foot_xy, surface, root_xy, root_yaw


class FootholdActionTests(unittest.TestCase):
    def test_two_contact_action_records_alternating_landings(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )

        action = action_from_profiles(
            clip_index=2,
            start_frame=1,
            support_mask=support,
            foot_xy_m=foot_xy,
            foot_surface_height_m=surface,
            root_xy_m=root_xy,
            root_yaw_rad=root_yaw,
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.start_support, (True, False))
        self.assertEqual(action.landing_feet, (1, 0))
        self.assertEqual(action.landing_frame_offsets, (2, 4))
        self.assertEqual(action.end_frame, 6)
        torch.testing.assert_close(
            action.landing_height_delta_m,
            torch.tensor((0.18, 0.18)),
        )
        torch.testing.assert_close(
            action.root_displacement_m,
            torch.tensor(((0.00, -0.15), (0.00, -0.30))),
            atol=1e-6,
            rtol=0.0,
        )

    def test_action_rejects_repeated_same_foot_landing(self):
        with self.assertRaisesRegex(ContractError, "alternate"):
            FootholdAction(
                clip_index=0,
                start_frame=0,
                end_frame=5,
                start_support=(True, False),
                landing_feet=(0, 0),
                landing_frame_offsets=(2, 4),
                landing_xy_start_frame_m=torch.zeros((2, 2)),
                landing_height_delta_m=torch.zeros(2),
                root_displacement_m=torch.zeros((2, 2)),
                root_yaw_delta_rad=torch.zeros(2),
                minimum_swing_clearance_m=0.04,
                maximum_unsupported_frames=0,
            )

    def test_action_returns_none_without_two_alternating_landings(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )
        support[5, 0] = False

        action = action_from_profiles(
            clip_index=2,
            start_frame=1,
            support_mask=support,
            foot_xy_m=foot_xy,
            foot_surface_height_m=surface,
            root_xy_m=root_xy,
            root_yaw_rad=root_yaw,
        )

        self.assertIsNone(action)

    def test_simultaneous_double_support_onset_is_not_two_landings(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )
        support[:] = torch.tensor(
            [
                [True, True],
                [False, False],
                [False, False],
                [True, True],
                [False, True],
                [True, True],
            ],
            dtype=torch.bool,
        )

        action = action_from_profiles(
            clip_index=2,
            start_frame=0,
            support_mask=support,
            foot_xy_m=foot_xy,
            foot_surface_height_m=surface,
            root_xy_m=root_xy,
            root_yaw_rad=root_yaw,
        )

        self.assertEqual(action.landing_frame_offsets, (3, 5))
        self.assertEqual(action.landing_feet, (1, 0))

    def test_dataset_index_maps_only_exact_action_entry_rows(self):
        support, foot_xy, surface, root_xy, root_yaw = (
            _two_contact_profiles()
        )
        body = torch.zeros((6, 3, 3), dtype=torch.float32)
        body[:, 0, :2] = root_xy
        body[:, 0, 2] = 0.8
        body[:, 1:, :2] = foot_xy
        body[:, 1:, 2] = surface + 0.035
        half_yaw = root_yaw / 2.0
        root_quaternion = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=-1,
        )
        quaternion = torch.zeros((6, 3, 4), dtype=torch.float32)
        quaternion[..., 0] = 1.0
        quaternion[:, 0] = root_quaternion
        clip = SimpleNamespace(
            body_position_world=body.numpy(),
            body_quaternion_world_wxyz=quaternion.numpy(),
        )

        class StepGrid:
            @staticmethod
            def sample_xy(points):
                return torch.where(
                    points[..., 0] >= 0.25,
                    torch.full_like(points[..., 0], 0.18),
                    torch.zeros_like(points[..., 0]),
                )

        class IdentityAlignment:
            @staticmethod
            def matcher_to_scene_xy(points):
                return points

        dataset = SimpleNamespace(
            folder=SimpleNamespace(
                clips=(clip,),
                layout=SimpleNamespace(
                    root_body_index=0,
                    left_foot_body_index=1,
                    right_foot_body_index=2,
                ),
            ),
            clip_grids=(StepGrid(),),
            clip_alignments=(IdentityAlignment(),),
            device=torch.device("cpu"),
        )
        segment_index = SimpleNamespace(
            segments=(SimpleNamespace(clip_index=0, start_frame=1),),
            support_mask=lambda clip_index: support.clone(),
        )

        index = FootholdActionIndex.from_dataset(dataset, segment_index)
        database = SimpleNamespace(
            _search_clip_index=torch.tensor((0, 0, 0)),
            _search_frame_index=torch.tensor((0, 1, 2)),
            device=torch.device("cpu"),
        )

        self.assertEqual(len(index.actions), 1)
        rows = index.rows_for_database(database)
        self.assertIsNone(rows[0])
        self.assertIs(rows[1], index.actions[0])
        self.assertIsNone(rows[2])
        torch.testing.assert_close(
            index.actions[0].landing_height_delta_m,
            torch.tensor((0.18, 0.18)),
        )


if __name__ == "__main__":
    unittest.main()
