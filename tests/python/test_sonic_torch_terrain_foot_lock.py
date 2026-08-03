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


class _CompactSoles:
    @staticmethod
    def sole_points(joints, roots, quaternions):
        del quaternions
        feet = _LinearFeet.foot_positions(
            joints,
            roots,
            np.zeros((len(np.asarray(joints)), 4)),
        )
        offsets = np.asarray(
            ((-0.03, -0.02, -0.05), (-0.03, 0.02, -0.05),
             (0.03, -0.02, -0.05), (0.03, 0.02, -0.05)),
            dtype=np.float64,
        )
        return feet[:, :, None, :] + offsets[None, None, :, :]


class _TargetOrderedLinearFeet:
    _position_indices = ((0, 3, 6), (1, 4, 7))

    @classmethod
    def foot_positions(cls, joints, roots, quaternions):
        del quaternions
        joints = np.asarray(joints)
        roots = np.asarray(roots)
        return np.stack(
            [joints[:, indices] for indices in cls._position_indices],
            axis=1,
        ) + roots[:, None, :]

    @classmethod
    def solve_leg_positions(cls, joints, root, quaternion, mask, targets):
        del quaternion
        output = np.array(joints, copy=True)
        for foot, indices in enumerate(cls._position_indices):
            if mask[foot]:
                output[list(indices)] = targets[foot] - root
        return output


class _TargetOrderedCompactSoles(_CompactSoles):
    @staticmethod
    def sole_points(joints, roots, quaternions):
        feet = _TargetOrderedLinearFeet.foot_positions(
            joints, roots, quaternions
        )
        offsets = np.asarray(
            ((-0.03, -0.02, -0.05), (-0.03, 0.02, -0.05),
             (0.03, -0.02, -0.05), (0.03, 0.02, -0.05)),
            dtype=np.float64,
        )
        return feet[:, :, None, :] + offsets[None, None, :, :]


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
    def test_preview_restores_all_persistent_filter_state(self):
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
            root_height_correction_halflife_s=0.04,
        )
        foot_lock.apply(_result(0, 0.0))
        before = (
            foot_lock._lock_position.clone(),
            foot_lock._previous_support.clone(),
            foot_lock._locked.clone(),
            foot_lock._projected.clone(),
            foot_lock._anticipated.clone(),
            foot_lock._swing_projected.clone(),
            foot_lock._swing_projection_shift.clone(),
            foot_lock._swing_projection_touchdown.clone(),
            foot_lock._swing_projection_clip.clone(),
            foot_lock._joint_offset.clone(),
            foot_lock._previous_joint_position.clone(),
            foot_lock._root_height_offset_z.clone(),
            foot_lock.failure_count,
        )

        preview = foot_lock.preview((_result(1, 0.1),))

        self.assertEqual(len(preview), 1)
        after = (
            foot_lock._lock_position,
            foot_lock._previous_support,
            foot_lock._locked,
            foot_lock._projected,
            foot_lock._anticipated,
            foot_lock._swing_projected,
            foot_lock._swing_projection_shift,
            foot_lock._swing_projection_touchdown,
            foot_lock._swing_projection_clip,
            foot_lock._joint_offset,
            foot_lock._previous_joint_position,
            foot_lock._root_height_offset_z,
        )
        for expected, actual in zip(before[:-1], after):
            self.assertTrue(torch.allclose(expected, actual, equal_nan=True))
        self.assertEqual(foot_lock.failure_count, before[-1])

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

    def test_optional_support_aware_root_height_reduces_leg_correction(self):
        support = torch.ones((1, 2), dtype=torch.bool)
        common = dict(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.full((xy.shape[0],), 0.18),
            device=torch.device("cpu"),
            correction_halflife_s=0.001,
        )
        joint_only = TerrainFootLockFilter(**common)
        support_root = TerrainFootLockFilter(
            **common,
            root_height_correction_halflife_s=0.001,
        )
        gated_root = TerrainFootLockFilter(
            **common,
            root_height_correction_halflife_s=0.001,
            terrain_base_m=0.18,
        )
        native = _result(0, 0.0)

        joint_result = joint_only.apply(native)
        initial_root_result = support_root.apply(native)
        root_result = support_root.apply(native)
        gated_root.apply(native)
        gated_result = gated_root.apply(native)

        self.assertEqual(
            float(initial_root_result.root_position_world[2]),
            float(native.root_position_world[2]),
        )
        self.assertGreater(
            float(root_result.root_position_world[2]),
            float(native.root_position_world[2]),
        )
        self.assertLess(
            float(
                torch.linalg.vector_norm(
                    root_result.joint_position - native.joint_position
                )
            ),
            float(
                torch.linalg.vector_norm(
                    joint_result.joint_position - native.joint_position
                )
            ),
        )
        self.assertEqual(
            float(gated_result.root_position_world[2]),
            float(native.root_position_world[2]),
        )

    def test_support_root_offset_resets_when_new_chunk_is_rebased(self):
        support_root = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((4, 2), dtype=torch.bool),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.full((xy.shape[0],), 0.18),
            device=torch.device("cpu"),
            correction_halflife_s=0.001,
            root_height_correction_halflife_s=0.001,
        )
        native = _result(0, 0.0)
        support_root.apply(native)
        corrected = support_root.apply(_result(1, 0.0))
        self.assertGreater(
            float(corrected.root_position_world[2]),
            float(native.root_position_world[2]),
        )
        rebased = _result(2, 0.0)
        rebased.root_position_world[2] = corrected.root_position_world[2]
        rebased.diagnostics.terrain_chunk_start = True

        output = support_root.apply(rebased)

        self.assertEqual(
            float(output.root_position_world[2]),
            float(rebased.root_position_world[2]),
        )

    def test_touchdown_projection_moves_complete_sole_onto_one_surface(self):
        support = torch.tensor(
            ((False, False), (True, False), (True, False))
        )

        def step_surface(xy):
            return torch.where(
                xy[:, 0] >= 0.0,
                torch.full_like(xy[:, 0], 0.18),
                torch.zeros_like(xy[:, 0]),
            )

        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sole_kinematics=_TargetOrderedCompactSoles(),
            sample_surface=step_surface,
            device=torch.device("cpu"),
            touchdown_projection_max_shift_m=0.05,
            correction_halflife_s=0.25,
        )
        first = _result(0, -0.01)
        first.joint_position[6:8] = -0.5
        foot_lock.apply(first)

        native = _result(1, -0.01)
        native.joint_position[6:8] = -0.5
        corrected = foot_lock.apply(native)

        projected_x = float(foot_lock._lock_position[0, 0])
        self.assertLessEqual(projected_x, -0.03)
        right_leg = torch.tensor((1, 4, 7, 10, 14, 18))
        self.assertTrue(
            torch.equal(
                corrected.joint_position[right_leg],
                native.joint_position[right_leg],
            )
        )
        sole = _TargetOrderedCompactSoles.sole_points(
            corrected.joint_position.numpy()[None],
            corrected.root_position_world.numpy()[None],
            corrected.root_orientation_world_wxyz.numpy()[None],
        )[0, 0]
        heights = step_surface(torch.tensor(sole[:, :2])).numpy()
        self.assertLess(float(heights.max() - heights.min()), 0.01)

        continued_native = _result(2, 0.0)
        continued_native.joint_position[6:8] = -0.5
        continued = foot_lock.apply(continued_native)
        continued_sole = _TargetOrderedCompactSoles.sole_points(
            continued.joint_position.numpy()[None],
            continued.root_position_world.numpy()[None],
            continued.root_orientation_world_wxyz.numpy()[None],
        )[0, 0]
        self.assertTrue(
            np.allclose(continued_sole[:, :2], sole[:, :2], atol=1e-6)
        )

    def test_touchdown_projection_leaves_safe_native_contact_unchanged(self):
        support = torch.tensor(((False, False), (True, False)))
        common = dict(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sample_surface=lambda xy: torch.zeros(xy.shape[0]),
            device=torch.device("cpu"),
            correction_halflife_s=0.25,
        )
        legacy = TerrainFootLockFilter(**common)
        projected = TerrainFootLockFilter(
            **common,
            sole_kinematics=_TargetOrderedCompactSoles(),
            touchdown_projection_max_shift_m=0.05,
        )
        first = _result(0, -0.10)
        first.joint_position[6:8] = -0.5
        second = _result(1, -0.10)
        second.joint_position[6:8] = -0.5
        legacy.apply(first)
        projected.apply(first)

        expected = legacy.apply(second)
        actual = projected.apply(second)

        self.assertFalse(bool(projected._projected.any().item()))
        self.assertTrue(
            torch.equal(actual.joint_position, expected.joint_position)
        )

    def test_direct_touchdown_projection_respects_output_speed_limit(self):
        support = torch.tensor(((False, False), (True, False)))

        def step_surface(xy):
            return torch.where(
                xy[:, 0] >= 0.0,
                torch.full_like(xy[:, 0], 0.18),
                torch.zeros_like(xy[:, 0]),
            )

        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sole_kinematics=_TargetOrderedCompactSoles(),
            sample_surface=step_surface,
            device=torch.device("cpu"),
            touchdown_projection_max_shift_m=0.05,
            maximum_output_joint_speed_rad_s=2.0,
        )
        first = _result(0, -0.01)
        first.joint_position[6:8] = -0.5
        second = _result(1, -0.01)
        second.joint_position[6:8] = -0.5
        foot_lock.apply(first)

        projected = foot_lock.apply(second)

        self.assertTrue(bool(foot_lock._projected[0].item()))
        self.assertLessEqual(
            float(torch.max(torch.abs(projected.joint_velocity))), 2.0 + 1e-6
        )

    def test_touchdown_projection_blends_during_swing_and_bounds_speed(self):
        support = torch.tensor(
            ((True, True), (False, True), (False, True), (True, True))
        )
        source_feet = torch.zeros((4, 2, 3), dtype=torch.float32)
        source_feet[:, 0, 0] = torch.tensor((-0.10, -0.07, -0.04, -0.01))

        def step_surface(xy):
            return torch.where(
                xy[:, 0] >= 0.0,
                torch.full_like(xy[:, 0], 0.18),
                torch.zeros_like(xy[:, 0]),
            )

        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            source_foot_positions=(source_feet,),
            source_root_yaws=(torch.zeros(4),),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sole_kinematics=_TargetOrderedCompactSoles(),
            sample_surface=step_surface,
            device=torch.device("cpu"),
            touchdown_projection_max_shift_m=0.05,
            anticipatory_touchdown_projection=True,
            correction_halflife_s=0.001,
            maximum_output_joint_speed_rad_s=2.0,
        )
        outputs = []
        for frame, root_x in enumerate((-0.10, -0.07, -0.04, -0.01)):
            native = _result(frame, root_x)
            native.joint_position[1] = 0.2
            native.joint_position[6:8] = -0.5
            outputs.append(foot_lock.apply(native))

        last_swing_foot = _TargetOrderedLinearFeet.foot_positions(
            outputs[2].joint_position.numpy()[None],
            outputs[2].root_position_world.numpy()[None],
            outputs[2].root_orientation_world_wxyz.numpy()[None],
        )[0, 0]
        self.assertLess(float(last_swing_foot[0]), -0.05)
        touchdown_sole = _TargetOrderedCompactSoles.sole_points(
            outputs[3].joint_position.numpy()[None],
            outputs[3].root_position_world.numpy()[None],
            outputs[3].root_orientation_world_wxyz.numpy()[None],
        )[0, 0]
        heights = step_surface(torch.tensor(touchdown_sole[:, :2]))
        self.assertLess(float(heights.max() - heights.min()), 0.01)
        self.assertLessEqual(
            max(
                float(torch.max(torch.abs(output.joint_velocity)))
                for output in outputs[1:]
            ),
            2.0 + 1e-6,
        )

    def test_touchdown_projection_does_not_change_early_swing(self):
        support = torch.tensor(
            (
                (True, True),
                (False, True),
                (False, True),
                (False, True),
                (False, True),
                (False, True),
                (True, True),
            )
        )
        source_feet = torch.zeros((7, 2, 3), dtype=torch.float32)
        source_feet[:, 0, 0] = torch.linspace(-0.10, -0.01, 7)

        def step_surface(xy):
            return torch.where(
                xy[:, 0] >= 0.0,
                torch.full_like(xy[:, 0], 0.18),
                torch.zeros_like(xy[:, 0]),
            )

        common = dict(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            source_foot_positions=(source_feet,),
            source_root_yaws=(torch.zeros(7),),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sample_surface=step_surface,
            device=torch.device("cpu"),
            correction_halflife_s=0.001,
        )
        legacy = TerrainFootLockFilter(**common)
        projected = TerrainFootLockFilter(
            **common,
            sole_kinematics=_TargetOrderedCompactSoles(),
            touchdown_projection_max_shift_m=0.05,
            anticipatory_touchdown_projection=True,
        )

        for frame, root_x in enumerate((-0.10, -0.085, -0.07)):
            native = _result(frame, root_x)
            native.joint_position[1] = -0.1
            native.joint_position[6:8] = -0.5
            expected = legacy.apply(native)
            actual = projected.apply(native)
            self.assertTrue(
                torch.equal(actual.joint_position, expected.joint_position)
            )

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

    def test_supported_leg_correction_remains_speed_bounded_and_soft(self):
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sample_surface=lambda xy: torch.full((xy.shape[0],), 0.465),
            device=torch.device("cpu"),
            correction_halflife_s=10.0,
            maximum_output_joint_speed_rad_s=12.0,
            terrain_base_m=0.0,
        )
        first = foot_lock.apply(_result(0, 0.0))
        target = _TargetOrderedLinearFeet.foot_positions(
            first.joint_position.numpy()[None],
            first.root_position_world.numpy()[None],
            first.root_orientation_world_wxyz.numpy()[None],
        )[0]

        moved = foot_lock.apply(_result(1, 0.02))
        actual = _TargetOrderedLinearFeet.foot_positions(
            moved.joint_position.numpy()[None],
            moved.root_position_world.numpy()[None],
            moved.root_orientation_world_wxyz.numpy()[None],
        )[0]

        self.assertGreater(float(actual[0, 0] - target[0, 0]), 0.019)
        self.assertLessEqual(
            float(torch.max(torch.abs(moved.joint_velocity))), 12.0 + 1e-6
        )

        flat_gated = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(torch.ones((2, 2), dtype=torch.bool),),
            foot_kinematics=_TargetOrderedLinearFeet(),
            sample_surface=lambda xy: torch.full((xy.shape[0],), 0.465),
            device=torch.device("cpu"),
            correction_halflife_s=10.0,
            terrain_base_m=0.465,
        )
        flat_gated.apply(_result(0, 0.0))
        gated_moved = flat_gated.apply(_result(1, 0.02))
        gated_feet = _TargetOrderedLinearFeet.foot_positions(
            gated_moved.joint_position.numpy()[None],
            gated_moved.root_position_world.numpy()[None],
            gated_moved.root_orientation_world_wxyz.numpy()[None],
        )[0]
        self.assertGreater(float(gated_feet[0, 0] - target[0, 0]), 0.019)

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

    def test_optional_source_swing_plan_anticipates_future_path_riser(self):
        support = torch.tensor(
            ((True, False), (True, False), (True, True)), dtype=torch.bool
        )
        source_feet = torch.zeros((3, 2, 3), dtype=torch.float32)
        source_feet[:, 1, 0] = torch.tensor((0.0, 0.1, 0.1))
        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            source_foot_positions=(source_feet,),
            source_root_yaws=(torch.zeros(3),),
            foot_kinematics=_LinearFeet(),
            sample_surface=lambda xy: torch.where(
                xy[:, 0] >= 0.08,
                torch.full((xy.shape[0],), 0.1),
                torch.zeros(xy.shape[0]),
            ),
            device=torch.device("cpu"),
            swing_clearance_margin_m=0.01,
            swing_plan_sigma_frames=2.0,
        )
        native = _result(0, 0.0)

        corrected = foot_lock.apply(native)

        self.assertGreater(
            float(corrected.joint_position[5]),
            float(native.joint_position[5]),
        )

    def test_source_swing_plan_falls_back_at_terrain_grid_boundary(self):
        support = torch.tensor(
            ((True, False), (True, False), (True, True)), dtype=torch.bool
        )
        source_feet = torch.zeros((3, 2, 3), dtype=torch.float32)
        source_feet[:, 1, 0] = torch.tensor((0.0, 0.1, 0.1))

        def bounded_surface(xy):
            if bool((xy[:, 0] > 0.05).any()):
                raise ValueError("outside grid")
            return torch.zeros(xy.shape[0])

        foot_lock = TerrainFootLockFilter(
            clip_paths=("clip.npz",),
            support_masks=(support,),
            source_foot_positions=(source_feet,),
            source_root_yaws=(torch.zeros(3),),
            foot_kinematics=_LinearFeet(),
            sample_surface=bounded_surface,
            device=torch.device("cpu"),
            swing_clearance_margin_m=0.01,
            swing_plan_sigma_frames=2.0,
        )

        corrected = foot_lock.apply(_result(0, 0.0))

        self.assertEqual(tuple(corrected.joint_position.shape), (29,))
        self.assertEqual(foot_lock.failure_count, 1)

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
