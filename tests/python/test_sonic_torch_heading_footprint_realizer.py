from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from mm_sonic.torch_foothold_actions import (
    FootholdAction,
    FootholdActionIndex,
)
from mm_sonic.torch_heading_footprint_search import (
    FootprintActionEdge,
    HeadingFootprintPlan,
)
from mm_sonic.torch_heading_footprint_realizer import (
    RealizationFailure,
    blend_action_boundary,
    delay_swing_progress_for_terrain,
    lift_swing_targets_over_terrain,
    realize_heading_footprint_plan,
    report_support_after_touchdown_settles,
    smooth_swing_terrain_height,
    smooth_swing_terrain_lift,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    TerrainFootprintCandidate,
)


def _action():
    return FootholdAction(
        clip_index=0,
        start_frame=0,
        end_frame=21,
        start_support=(True, True),
        landing_feet=(0, 1),
        landing_frame_offsets=(10, 20),
        landing_xy_start_frame_m=torch.tensor(
            ((0.25, 0.12), (0.50, -0.12))
        ),
        landing_height_delta_m=torch.zeros(2),
        root_displacement_m=torch.tensor(
            ((0.25, 0.0), (0.50, 0.0))
        ),
        root_yaw_delta_rad=torch.zeros(2),
        minimum_swing_clearance_m=0.04,
        maximum_unsupported_frames=0,
    )


def _plan(heading=(0.0, 1.0)):
    heading_tensor = torch.tensor(heading, dtype=torch.float32)
    heading_tensor /= torch.linalg.vector_norm(heading_tensor)
    lateral = torch.tensor(
        (-heading_tensor[1], heading_tensor[0])
    )
    candidates = []
    for step, (progress, foot) in enumerate(((0.25, 0), (0.50, 1))):
        local = torch.tensor(
            (progress, 0.12 if foot == 0 else -0.12)
        )
        scene = (
            progress * heading_tensor
            + local[1] * lateral
        )
        candidates.append(
            TerrainFootprintCandidate(
                step_index=step,
                foot=foot,
                center_scene_xy=scene,
                center_heading_xy=local,
                offset_heading_xy=torch.zeros(2),
                yaw_scene_rad=float(
                    torch.atan2(heading_tensor[1], heading_tensor[0])
                ),
                surface_height_m=0.0,
                placement_cost=0.0,
            )
        )
    edge = FootprintActionEdge(
        action_key=(0, 0),
        candidate_indices=(0, 0),
        descriptor_cost=0.0,
        transition_cost=0.0,
    )
    return HeadingFootprintPlan(
        heading_scene_xy=heading_tensor,
        footprints=tuple(candidates),
        edges=(edge,),
        action_keys=((0, 0),),
        total_cost=0.0,
    )


def _source(
    *,
    terminal_double_support=True,
    extra_recontact=False,
    departure_swing=False,
):
    frames = 21
    root = np.zeros((frames, 3), dtype=np.float64)
    root[:, 0] = np.linspace(0.0, 0.50, frames)
    root[:, 2] = 0.80
    feet = np.zeros((frames, 2, 3), dtype=np.float64)
    feet[:, 0] = (0.0, 0.12, 0.035)
    feet[:, 1] = (0.0, -0.12, 0.035)
    feet[1:11, 0, 0] = np.linspace(0.0, 0.25, 10)
    feet[10:, 0, 0] = 0.25
    feet[11:, 1, 0] = np.linspace(0.0, 0.50, 10)
    joints = np.zeros((frames, 29), dtype=np.float64)
    joints[:, :6] = feet.reshape(frames, 6)
    body_position = np.zeros((frames, 30, 3), dtype=np.float64)
    body_position[:, 0] = root
    body_position[:, 18:20] = feet
    quaternions = np.zeros((frames, 30, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    support = np.ones((frames, 2), dtype=np.bool_)
    support[1:10, 0] = False
    support[11:20, 1] = False
    if extra_recontact:
        support[11, 0] = False
        support[11:15, 1] = True
    if not terminal_double_support:
        support[-1, 1] = False
    if departure_swing:
        extra = 9
        root = np.concatenate(
            (root, np.repeat(root[-1:], extra, axis=0)), axis=0
        )
        feet = np.concatenate(
            (feet, np.repeat(feet[-1:], extra, axis=0)), axis=0
        )
        joints = np.concatenate(
            (joints, np.repeat(joints[-1:], extra, axis=0)), axis=0
        )
        body_position = np.concatenate(
            (
                body_position,
                np.repeat(body_position[-1:], extra, axis=0),
            ),
            axis=0,
        )
        quaternions = np.concatenate(
            (
                quaternions,
                np.repeat(quaternions[-1:], extra, axis=0),
            ),
            axis=0,
        )
        support = np.concatenate(
            (support, np.ones((extra, 2), dtype=np.bool_)), axis=0
        )
        support[23:29, 0] = False
        feet[23:29, 0, 0] = np.linspace(0.25, 0.65, 6)
        joints[23:29, :6] = feet[23:29].reshape(6, 6)
        body_position[23:29, 18:20] = feet[23:29]
    action = _action()
    return SimpleNamespace(
        clips=(
            SimpleNamespace(
                joint_position=joints,
                body_position_world=body_position,
                body_quaternion_world_wxyz=quaternions,
            ),
        ),
        root_body_index=0,
        foot_body_indices=(18, 19),
        action_index=FootholdActionIndex(
            actions=(action,), _entries={(0, 0): action}
        ),
        support_mask=lambda _clip_index: support,
    )


class _SyntheticKinematics:
    @staticmethod
    def foot_positions(joints, _root, _quaternion):
        joints = np.asarray(joints)
        return joints[:, :6].reshape(len(joints), 2, 3)

    @staticmethod
    def center_of_mass_positions(_joints, root, _quaternion):
        return np.asarray(root)


class _RecordingRetargeter:
    def __init__(self):
        self.calls = []

    def solve_frame(self, **kwargs):
        self.calls.append(kwargs)
        joints = np.asarray(
            kwargs["joint_position"], dtype=np.float64
        ).copy()
        joints[:6] = np.asarray(
            kwargs["target_foot_position_world"]
        ).reshape(6)
        return joints, np.asarray(
            kwargs["root_position_world"], dtype=np.float64
        ).copy()


class _FlatTerrain:
    @staticmethod
    def sample_surface(points_xy):
        points = np.asarray(points_xy)
        return np.zeros(points.shape[:-1], dtype=np.float64)


class _StepTerrain:
    @staticmethod
    def sample_surface(points_xy):
        points = np.asarray(points_xy)
        # A finite obstacle inside the authenticated swing, rather than a
        # raised landing whose height would disagree with _plan().
        return 0.02 * (
            (points[..., 0] > 0.181) & (points[..., 0] < 0.19)
        )


class HeadingFootprintRealizerTests(unittest.TestCase):
    def test_touchdown_support_is_reported_after_one_settle_frame(self):
        support = np.ones((12, 2), dtype=np.bool_)
        support[1:10, 0] = False

        reported = report_support_after_touchdown_settles(support)

        self.assertFalse(bool(reported[10, 0]))
        self.assertTrue(bool(reported[11, 0]))
        self.assertTrue(reported[-1].all())
        np.testing.assert_array_equal(reported[:, 1], support[:, 1])

    def test_swing_lift_samples_position_uncertainty_around_the_sole(self):
        targets = np.zeros((2, 2, 3), dtype=np.float64)
        targets[..., 2] = 0.035
        support = np.ones((2, 2), dtype=np.bool_)
        support[1, 0] = False

        def sample_surface(points_xy):
            points = np.asarray(points_xy)
            return 0.20 * (points[..., 0] > 0.18)

        lifted = lift_swing_targets_over_terrain(
            target_foot_position_world=targets,
            support_mask=support,
            target_heading_world_xy=np.array((1.0, 0.0)),
            sample_surface=sample_surface,
            ankle_origin_sole_m=0.035,
            minimum_clearance_m=0.02,
            sample_uncertainty_m=0.08,
        )

        self.assertAlmostEqual(float(lifted[1, 0, 2]), 0.255)
        self.assertAlmostEqual(float(lifted[0, 0, 2]), 0.035)

    def test_large_terrain_lift_delays_then_recovers_swing_progress(self):
        targets = np.zeros((11, 2, 3), dtype=np.float64)
        targets[:, 0, 0] = np.linspace(0.0, 0.50, 11)
        support = np.ones((11, 2), dtype=np.bool_)
        support[1:10, 0] = False
        raw_lift = np.zeros((11, 2), dtype=np.float64)
        raw_lift[3:7, 0] = 0.16

        delayed = delay_swing_progress_for_terrain(
            target_foot_position_world=targets,
            support_mask=support,
            raw_terrain_lift_m=raw_lift,
            lift_threshold_m=0.08,
            maximum_delay_frames=5,
        )

        self.assertLess(float(delayed[3, 0, 0]), 0.05)
        self.assertAlmostEqual(float(delayed[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(delayed[-1, 0, 0]), 0.50)
        np.testing.assert_allclose(delayed[:, 1], targets[:, 1])

    def test_swing_lift_is_anticipated_with_a_bounded_height_slope(self):
        support = np.ones((11, 2), dtype=np.bool_)
        support[1:10, 0] = False
        raw_lift = np.zeros((11, 2), dtype=np.float64)
        raw_lift[5, 0] = 0.10

        envelope = smooth_swing_terrain_lift(
            raw_lift_m=raw_lift,
            support_mask=support,
            maximum_height_step_m=0.02,
        )

        self.assertAlmostEqual(float(envelope[5, 0]), 0.10)
        self.assertGreater(float(envelope[3, 0]), 0.0)
        self.assertTrue(
            (
                np.abs(np.diff(envelope[:, 0]))
                <= 0.02 + 1.0e-12
            ).all()
        )
        np.testing.assert_allclose(envelope[support], 0.0)

    def test_absolute_swing_height_has_a_bounded_slope(self):
        support = np.ones((11, 2), dtype=np.bool_)
        support[1:10, 0] = False
        required = np.zeros((11, 2), dtype=np.float64)
        required[:, 1] = 0.20
        required[1:10, 0] = np.array(
            (0.02, 0.05, 0.09, 0.13, 0.17, 0.13, 0.09, 0.05, 0.02)
        )

        height = smooth_swing_terrain_height(
            required_height_m=required,
            support_mask=support,
            maximum_height_step_m=0.04,
        )

        self.assertTrue((height >= required - 1.0e-12).all())
        self.assertTrue(
            (
                np.abs(np.diff(height[:, 0]))
                <= 0.04 + 1.0e-12
            ).all()
        )
        np.testing.assert_allclose(height[support], required[support])

    def test_terrain_lifted_swing_target_is_a_required_ik_constraint(self):
        retargeter = _RecordingRetargeter()

        realize_heading_footprint_plan(
            plan=_plan(heading=(1.0, 0.0)),
            source=_source(),
            terrain=_StepTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=retargeter,
        )

        self.assertTrue(
            any(
                bool(
                    np.asarray(
                        call["enforce_target_error_feet"]
                    ).all()
                )
                for call in retargeter.calls[1:10]
            )
        )

    def test_action_time_scale_adds_interpolated_50hz_frames(self):
        source = _source()
        action = replace(
            source.action_index.actions[0],
            landing_height_delta_m=torch.tensor((0.20, 0.20)),
        )
        source.action_index = FootholdActionIndex(
            actions=(action,), _entries={(0, 0): action}
        )
        traversal = realize_heading_footprint_plan(
            plan=_plan(heading=(1.0, 0.0)),
            source=source,
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=_RecordingRetargeter(),
            action_time_scale=2.0,
        )

        self.assertEqual(len(traversal.joint_position), 41)
        self.assertTrue(traversal.source_support_mask[-1].all())
        self.assertEqual(
            traversal.source_frame_provenance.dtype, np.int64
        )

    def test_realizer_preserves_the_solved_boundary_frame(self):
        first = _plan()
        second_footprints = tuple(
            replace(
                footprint,
                step_index=footprint.step_index + 2,
                center_scene_xy=footprint.center_scene_xy
                + torch.tensor((0.5, 0.0)),
                center_heading_xy=footprint.center_heading_xy
                + torch.tensor((0.5, 0.0)),
            )
            for footprint in first.footprints
        )
        plan = HeadingFootprintPlan(
            heading_scene_xy=first.heading_scene_xy,
            footprints=(*first.footprints, *second_footprints),
            edges=(*first.edges, first.edges[0]),
            action_keys=(*first.action_keys, first.action_keys[0]),
            total_cost=0.0,
        )

        result = realize_heading_footprint_plan(
            plan=plan,
            source=_source(),
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=_RecordingRetargeter(),
        )

        self.assertEqual(len(result.joint_position), 42)
        np.testing.assert_array_equal(
            result.source_frame_provenance[20:22],
            ((0, 20), (0, 0)),
        )

    def test_swing_target_clears_highest_surface_under_full_sole(self):
        targets = np.zeros((2, 2, 3), dtype=np.float64)
        support = np.array(((True, False), (False, True)))

        lifted = lift_swing_targets_over_terrain(
            target_foot_position_world=targets,
            support_mask=support,
            target_heading_world_xy=np.array((1.0, 0.0)),
            sample_surface=lambda points: (
                np.asarray(points)[..., 0] > 0.05
            ).astype(np.float64),
            ankle_origin_sole_m=0.035,
            minimum_clearance_m=0.02,
        )

        np.testing.assert_allclose(lifted[support], 0.0)
        np.testing.assert_allclose(lifted[~support, 2], 1.055)

    def test_boundary_blend_starts_exactly_at_previous_state(self):
        joints = np.ones((10, 29), dtype=np.float64)
        root = np.ones((10, 3), dtype=np.float64)
        quaternion = np.zeros((10, 4), dtype=np.float64)
        quaternion[:, 0] = 1.0
        previous_joint = np.zeros(29, dtype=np.float64)
        previous_root = np.zeros(3, dtype=np.float64)
        previous_quaternion = np.array(
            (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
        )

        blended = blend_action_boundary(
            joint_position=joints,
            root_position_world=root,
            root_orientation_world_wxyz=quaternion,
            previous_joint_position=previous_joint,
            previous_root_position_world=previous_root,
            previous_root_orientation_world_wxyz=previous_quaternion,
            blend_frames=6,
        )

        np.testing.assert_allclose(blended[0][0], previous_joint)
        np.testing.assert_allclose(blended[1][0], previous_root)
        np.testing.assert_allclose(
            blended[2][0], previous_quaternion
        )
        np.testing.assert_allclose(blended[0][5:], joints[5:])
        np.testing.assert_allclose(blended[1][5:], root[5:])
        np.testing.assert_allclose(blended[2][5:], quaternion[5:])
        np.testing.assert_array_equal(joints, 1.0)

    def test_realizer_rotates_window_and_locks_stance_feet(self):
        retargeter = _RecordingRetargeter()
        result = realize_heading_footprint_plan(
            plan=_plan(heading=(0.0, 1.0)),
            source=_source(),
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=retargeter,
        )

        self.assertLess(result.maximum_stance_error_m, 0.005)
        self.assertGreater(result.root_position_world[-1, 1], 0.49)
        self.assertAlmostEqual(
            result.root_position_world[-1, 0], 0.0, places=6
        )
        self.assertTrue(result.source_support_mask[-1].all())
        self.assertEqual(
            tuple(result.source_frame_provenance[-1]), (0, 20)
        )
        self.assertEqual(len(retargeter.calls), 21)
        np.testing.assert_allclose(
            retargeter.calls[0]["target_foot_rotation_world"][
                :, :, 0
            ],
            np.tile((0.0, 1.0, 0.0), (2, 1)),
            atol=1.0e-7,
        )

    def test_realizer_rejects_terminal_mid_swing(self):
        with self.assertRaisesRegex(
            RealizationFailure, "terminal_mid_swing"
        ):
            realize_heading_footprint_plan(
                plan=_plan(),
                source=_source(terminal_double_support=False),
                terrain=_FlatTerrain(),
                kinematics=_SyntheticKinematics(),
                retargeter=_RecordingRetargeter(),
            )

    def test_terminal_extension_stops_before_next_touchdown(self):
        result = realize_heading_footprint_plan(
            plan=_plan(),
            source=_source(departure_swing=True),
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=_RecordingRetargeter(),
            extend_terminal_swing=True,
        )

        self.assertEqual(
            tuple(result.source_frame_provenance[-1]), (0, 28)
        )
        np.testing.assert_array_equal(
            result.source_support_mask[-1], (False, True)
        )

    def test_separate_departure_action_stops_before_first_touchdown(self):
        retargeter = _RecordingRetargeter()
        result = realize_heading_footprint_plan(
            plan=_plan(),
            source=_source(),
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=retargeter,
            departure_action_key=(0, 0),
            departure_swing_clearance_m=0.10,
            departure_swing_target_xy_world=np.array((0.60, 0.12)),
        )

        self.assertEqual(
            tuple(result.source_frame_provenance[-1]), (0, 9)
        )
        np.testing.assert_array_equal(
            result.source_support_mask[-1], (False, True)
        )
        self.assertAlmostEqual(
            result.joint_position[-1, 2], 0.135, places=6
        )
        np.testing.assert_allclose(
            result.joint_position[-1, :2], (0.60, 0.12)
        )
        terminal_call = retargeter.calls[-1]
        weights = terminal_call["target_foot_position_weights"]
        self.assertGreater(float(weights[1]), float(weights[0]))
        np.testing.assert_array_equal(
            terminal_call["enforce_target_error_feet"], (True, True)
        )

    def test_realizer_preserves_an_extra_same_foot_recontact(self):
        result = realize_heading_footprint_plan(
            plan=_plan(),
            source=_source(extra_recontact=True),
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=_RecordingRetargeter(),
        )

        self.assertFalse(result.source_support_mask[12, 0])
        self.assertTrue(result.source_support_mask[13, 0])
        self.assertLess(result.maximum_stance_error_m, 0.005)

    def test_first_window_uses_actual_start_stance_feet(self):
        source = _source()
        source.start_foot_position_world = np.array(
            ((0.10, 0.12, 0.035), (0.10, -0.12, 0.035))
        )

        result = realize_heading_footprint_plan(
            plan=_plan(heading=(1.0, 0.0)),
            source=source,
            terrain=_FlatTerrain(),
            kinematics=_SyntheticKinematics(),
            retargeter=_RecordingRetargeter(),
        )

        np.testing.assert_allclose(
            result.joint_position[0, :6].reshape(2, 3),
            source.start_foot_position_world,
        )


if __name__ == "__main__":
    unittest.main()
