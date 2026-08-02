from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.torch_terrain_foot_lock import TerrainFootLockFilter


class _LinearFeet:
    @staticmethod
    def foot_positions(joints, roots, quaternions):
        del quaternions
        joints = np.asarray(joints)
        roots = np.asarray(roots)
        return joints[:, :6].reshape(-1, 2, 3) + roots[:, None, :]

    @staticmethod
    def solve_leg_positions(joints, root, quaternion, mask, targets):
        del quaternion
        output = np.array(joints, copy=True)
        for foot in range(2):
            if mask[foot]:
                output[foot * 3 : foot * 3 + 3] = targets[foot] - root
        return output


def _result(frame, root_x):
    joints = torch.zeros(29)
    joints[:3] = torch.tensor((0.0, 0.1, -0.5))
    joints[3:6] = torch.tensor((0.0, -0.1, -0.5))
    return SimpleNamespace(
        joint_position=joints,
        joint_velocity=torch.zeros(29),
        root_position_world=torch.tensor((root_x, 0.0, 0.5)),
        root_orientation_world_wxyz=torch.tensor((1.0, 0.0, 0.0, 0.0)),
        root_linear_velocity_world=torch.tensor((0.5, 0.0, 0.0)),
        diagnostics=SimpleNamespace(
            selected_clip_path="clip.npz", selected_frame=frame
        ),
    )


class TerrainFootLockTests(unittest.TestCase):
    def test_failed_solve_unlocks_until_the_next_contact(self):
        class _FailingFeet(_LinearFeet):
            def __init__(self):
                self.solve_calls = 0

            def solve_leg_positions(self, *args):
                self.solve_calls += 1
                raise ValueError("unreachable")

        feet = _FailingFeet()
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=feet,
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
        )

        foot_lock.apply(_result(0, 0.0))
        foot_lock.apply(_result(1, 0.01))

        self.assertEqual(feet.solve_calls, 1)
        self.assertEqual(foot_lock.failure_count, 1)

    def test_base_matcher_result_does_not_require_root_velocity_field(self):
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
        )
        result = _result(0, 0.0)
        delattr(result, "root_linear_velocity_world")

        corrected = foot_lock.apply(result)

        self.assertEqual(tuple(corrected.joint_position.shape), (29,))

    def test_persistent_support_stays_locked_while_swing_leg_is_untouched(self):
        support = torch.tensor(
            ((True, False), (True, False), (False, True))
        )
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
        )

        first = foot_lock.apply(_result(0, 0.0))
        second = foot_lock.apply(_result(1, 0.1))

        first_feet = _LinearFeet.foot_positions(
            first.joint_position[None],
            first.root_position_world[None],
            first.root_orientation_world_wxyz[None],
        )[0]
        second_feet = _LinearFeet.foot_positions(
            second.joint_position[None],
            second.root_position_world[None],
            second.root_orientation_world_wxyz[None],
        )[0]
        native_second = _LinearFeet.foot_positions(
            _result(1, 0.1).joint_position[None],
            _result(1, 0.1).root_position_world[None],
            _result(1, 0.1).root_orientation_world_wxyz[None],
        )[0]
        self.assertLess(
            np.linalg.norm(second_feet[0] - first_feet[0]),
            np.linalg.norm(native_second[0] - first_feet[0]),
        )
        np.testing.assert_allclose(
            second.joint_position[3:6], _result(1, 0.1).joint_position[3:6]
        )
        self.assertGreater(float(torch.linalg.vector_norm(second.joint_velocity)), 0.0)

    def test_touchdown_releases_old_lock_and_acquires_new_foot(self):
        support = torch.tensor(
            ((True, False), (True, False), (False, True), (False, True))
        )
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.full((xy.shape[0],), 0.1),
            device=torch.device("cpu"),
        )
        foot_lock.apply(_result(0, 0.0))
        foot_lock.apply(_result(1, 0.1))
        touchdown = foot_lock.apply(_result(2, 0.2))
        continued = foot_lock.apply(_result(3, 0.3))

        touchdown_feet = _LinearFeet.foot_positions(
            touchdown.joint_position[None],
            touchdown.root_position_world[None],
            touchdown.root_orientation_world_wxyz[None],
        )[0]
        continued_feet = _LinearFeet.foot_positions(
            continued.joint_position[None],
            continued.root_position_world[None],
            continued.root_orientation_world_wxyz[None],
        )[0]
        self.assertLess(
            np.linalg.norm(continued_feet[1] - touchdown_feet[1]), 0.1
        )
        self.assertGreater(continued_feet[1, 2], 0.035)
        self.assertEqual(foot_lock.failure_count, 0)

    def test_large_lock_deviation_unlocks_instead_of_contorting_pose(self):
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
            unlock_radius_m=0.05,
        )
        foot_lock.apply(_result(0, 0.0))

        corrected = foot_lock.apply(_result(1, 0.2))

        self.assertLess(
            float(torch.max(torch.abs(corrected.joint_position))), 0.51
        )

    def test_output_joint_speed_is_bounded_during_correction(self):
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
            maximum_output_joint_speed_rad_s=2.0,
        )
        foot_lock.apply(_result(0, 0.0))

        corrected = foot_lock.apply(_result(1, 0.1))

        self.assertLessEqual(
            float(torch.max(torch.abs(corrected.joint_velocity))), 2.0 + 1e-6
        )

    def test_optional_swing_clearance_lifts_only_unsupported_penetrating_foot(self):
        support = torch.tensor(((True, False),), dtype=torch.bool)
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.where(
                xy[:, 1] > 0.0,
                torch.zeros(xy.shape[0]),
                torch.full((xy.shape[0],), 0.1),
            ),
            device=torch.device("cpu"),
            swing_clearance_margin_m=0.01,
        )
        native = _result(0, 0.0)
        native.joint_position[2] = -0.465

        corrected = foot_lock.apply(native)

        self.assertGreater(
            float(corrected.joint_position[5]),
            float(native.joint_position[5]),
        )
        self.assertEqual(
            float(corrected.joint_position[2]),
            float(native.joint_position[2]),
        )

    def test_unknown_source_fails_closed(self):
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
        )
        result = _result(0, 0.0)
        result.diagnostics.selected_clip_path = "missing.npz"
        with self.assertRaisesRegex(ValueError, "source"):
            foot_lock.apply(result)


if __name__ == "__main__":
    unittest.main()
