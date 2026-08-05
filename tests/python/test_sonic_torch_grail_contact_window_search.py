import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_grail_contact_window_search import (
    ContactWindow,
    best_contact_window_path,
    contact_window_cost,
    contact_windows_from_source,
)


def window(
    window_id,
    role,
    *,
    support_start=(True, False),
    support_stop=(True, False),
    height_error=0.0,
    timing_error=0.0,
    endpoint_pose=(0.0, 0.0),
    start_pose=(0.0, 0.0),
    endpoint_velocity=(0.0, 0.0),
    start_velocity=(0.0, 0.0),
):
    return ContactWindow(
        window_id=window_id,
        role=role,
        source_clip=f"clip-{window_id}",
        start_frame=10,
        stop_frame=40,
        support_start=support_start,
        support_stop=support_stop,
        height_pattern_error_m=height_error,
        timing_error_frames=timing_error,
        sole_transform_error_m=0.0,
        heading_error_rad=0.0,
        pelvis_error_m=0.0,
        start_pose=start_pose,
        stop_pose=endpoint_pose,
        start_velocity=start_velocity,
        stop_velocity=endpoint_velocity,
    )


class GrailContactWindowSearchTests(unittest.TestCase):
    @staticmethod
    def synthetic_source(*, split_height=True):
        frames = 90
        joints = np.zeros((frames, 4), dtype=np.float64)
        roots = np.zeros((frames, 3), dtype=np.float64)
        roots[:, 0] = np.linspace(0.0, 0.9, frames)
        quaternions = np.zeros((frames, 4), dtype=np.float64)
        quaternions[:, 0] = 1.0
        velocities = np.zeros((frames, 4), dtype=np.float64)
        velocities[:, 0] = 0.5
        feet = np.zeros((frames, 2, 3), dtype=np.float64)
        surface = np.zeros((frames, 2), dtype=np.float64)
        surface[:, 0] = 0.35
        surface[:, 1] = 0.53 if split_height else 0.35
        support = np.zeros((frames, 2), dtype=np.bool_)
        support[:5] = True
        touchdowns = ((10, 0), (25, 1), (40, 0), (55, 1))
        for index, (frame, foot) in enumerate(touchdowns):
            support[frame : frame + 10, foot] = True
            if index + 1 < len(touchdowns):
                next_frame, next_foot = touchdowns[index + 1]
                support[frame : next_frame + 3, foot] = True
                support[next_frame : next_frame + 3, next_foot] = True
        support[55:] = True
        feet[..., 0] = roots[:, None, 0]
        feet[:, 0, 1] = 0.12
        feet[:, 1, 1] = -0.12
        feet[..., 2] = surface + 0.035
        return (
            joints,
            roots,
            quaternions,
            velocities,
            feet,
            support,
            surface,
        )

    @staticmethod
    def synthetic_exit_source(*, allow_flight=False):
        arrays = list(
            GrailContactWindowSearchTests.synthetic_source(
                split_height=True
            )
        )
        frames = len(arrays[0])
        roots = arrays[1]
        feet = arrays[4]
        support = np.zeros((frames, 2), dtype=np.bool_)
        surface = np.empty((frames, 2), dtype=np.float64)
        surface[:, 0] = 0.35
        surface[:, 1] = 0.53
        support[:10] = True
        support[10:20, 1] = True
        if not allow_flight:
            support[20:25] = True
        support[20:, 0] = True
        surface[20:, 0] = 0.0
        support[25:40, 1] = False
        support[40:, 1] = True
        surface[40:, 1] = 0.0
        if allow_flight:
            support[10:20] = False
        feet[..., 0] = roots[:, None, 0]
        feet[:, 0, 1] = 0.12
        feet[:, 1, 1] = -0.12
        feet[..., 2] = surface + 0.035
        arrays[4] = feet
        arrays[5] = support
        arrays[6] = surface
        return tuple(arrays)

    def test_contact_cost_rejects_height_mismatch(self):
        candidate = window("bad-height", "uneven_walk", height_error=0.09)

        with self.assertRaisesRegex(ContractError, "height pattern"):
            contact_window_cost(candidate)

    def test_path_uses_boundary_pose_and_velocity_not_greedy_match_cost(self):
        entry = window(
            "entry",
            "entry",
            endpoint_pose=(0.0, 0.0),
            endpoint_velocity=(0.2, 0.0),
        )
        locally_best = window(
            "locally-best",
            "uneven_walk",
            height_error=0.0,
            start_pose=(1.0, 1.0),
            start_velocity=(-1.0, 0.0),
        )
        compatible = window(
            "compatible",
            "uneven_walk",
            height_error=0.01,
            start_pose=(0.02, 0.0),
            start_velocity=(0.21, 0.0),
        )

        path = best_contact_window_path(
            ("entry", "uneven_walk"),
            (locally_best, entry, compatible),
        )

        self.assertEqual(
            tuple(item.window_id for item in path.windows),
            ("entry", "compatible"),
        )

    def test_path_rejects_incompatible_support_identity(self):
        entry = window(
            "entry",
            "entry",
            support_stop=(True, False),
        )
        middle = window(
            "middle",
            "uneven_walk",
            support_start=(False, True),
        )

        with self.assertRaisesRegex(ContractError, "no contact-window path"):
            best_contact_window_path(
                ("entry", "uneven_walk"), (entry, middle)
            )

    def test_path_is_deterministic_under_equal_cost(self):
        entry_b = window("entry-b", "entry")
        entry_a = window("entry-a", "entry")
        middle_b = window("middle-b", "uneven_walk")
        middle_a = window("middle-a", "uneven_walk")

        first = best_contact_window_path(
            ("entry", "uneven_walk"),
            (middle_b, entry_b, middle_a, entry_a),
        )
        second = best_contact_window_path(
            ("entry", "uneven_walk"),
            (entry_a, middle_a, entry_b, middle_b),
        )

        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.window_id for item in first.windows),
            ("entry-a", "middle-a"),
        )

    def test_missing_role_is_explicit_failure(self):
        with self.assertRaisesRegex(ContractError, "no contact-window path"):
            best_contact_window_path(
                ("entry", "uneven_walk"),
                (window("entry", "entry"),),
            )

    def test_extracts_sustained_split_height_uneven_window(self):
        windows = contact_windows_from_source(
            source_clip="synthetic",
            joint_position=self.synthetic_source()[0],
            root_position_world=self.synthetic_source()[1],
            root_orientation_world_wxyz=self.synthetic_source()[2],
            generalized_velocity=self.synthetic_source()[3],
            foot_position_world=self.synthetic_source()[4],
            support_mask=self.synthetic_source()[5],
            foot_surface_height_m=self.synthetic_source()[6],
            target_split_height_m=0.18,
        )

        uneven = [item for item in windows if item.role == "uneven_walk"]
        self.assertTrue(uneven)
        self.assertLessEqual(uneven[0].height_pattern_error_m, 0.01)

    def test_flat_source_is_not_uneven_coverage(self):
        arrays = self.synthetic_source(split_height=False)

        windows = contact_windows_from_source(
            source_clip="flat",
            joint_position=arrays[0],
            root_position_world=arrays[1],
            root_orientation_world_wxyz=arrays[2],
            generalized_velocity=arrays[3],
            foot_position_world=arrays[4],
            support_mask=arrays[5],
            foot_surface_height_m=arrays[6],
            target_split_height_m=0.18,
        )

        self.assertFalse(
            any(item.role == "uneven_walk" for item in windows)
        )

    def test_extracts_supported_two_contact_exit(self):
        arrays = self.synthetic_exit_source()

        windows = contact_windows_from_source(
            source_clip="exit",
            joint_position=arrays[0],
            root_position_world=arrays[1],
            root_orientation_world_wxyz=arrays[2],
            generalized_velocity=arrays[3],
            foot_position_world=arrays[4],
            support_mask=arrays[5],
            foot_surface_height_m=arrays[6],
            target_split_height_m=0.18,
        )

        exits = [item for item in windows if item.role == "exit"]
        self.assertTrue(exits)
        self.assertEqual(exits[0].support_stop, (True, True))

    def test_exit_with_global_flight_is_rejected(self):
        arrays = self.synthetic_exit_source(allow_flight=True)

        windows = contact_windows_from_source(
            source_clip="flight",
            joint_position=arrays[0],
            root_position_world=arrays[1],
            root_orientation_world_wxyz=arrays[2],
            generalized_velocity=arrays[3],
            foot_position_world=arrays[4],
            support_mask=arrays[5],
            foot_surface_height_m=arrays[6],
            target_split_height_m=0.18,
        )

        self.assertFalse(any(item.role == "exit" for item in windows))


if __name__ == "__main__":
    unittest.main()
