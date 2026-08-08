import unittest
import math
from types import SimpleNamespace

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_ordered_terrain_levels import (
    derive_ordered_terrain_levels,
    match_ordered_touchdown_levels,
)
from mm_sonic.torch_path_motion_placement import (
    RawContactEvent,
    RawMotionWindow,
)
from mm_sonic.torch_heading_footprint_realizer import blend_action_boundary
from resources.run_g1_root_path_motion_search import (
    _best_transition,
    _best_certified_transition,
    _candidate_naturalness_penalty,
    _certified_traversal_archive,
    _contact_preserving_boundary_blend,
    _certify_transition,
    _extend_window_to_terminal_double_support,
    _extend_window_through_stable_liftoff,
    _extend_window_through_next_touchdown,
    _minimum_search_window_length,
    _parser,
    _ordered_phase_kind,
    _prepare_candidate_window,
    _placed_touchdowns,
    _resolve_path,
    _requires_terminal_settle,
    _search_window_reaches_path_end,
    _scanner_job,
)


class _FakeSoleKinematics:
    def __init__(self, sole_height):
        self.sole_height = np.broadcast_to(
            np.asarray(sole_height, dtype=np.float64), (3,)
        )

    def sole_points(self, joints, roots, quaternion):
        output = np.zeros((len(joints), 2, 3, 3), dtype=np.float64)
        output[..., 2] = self.sole_height
        return output


class _RootRelativeSoleKinematics:
    def sole_points(self, joints, roots, quaternion):
        del quaternion
        output = np.zeros((len(joints), 2, 3, 3), dtype=np.float64)
        output[..., 0] = roots[:, None, None, 0]
        output[..., 1] = roots[:, None, None, 1]
        output[..., 2] = (
            roots[:, None, None, 2]
            + joints[:, None, None, 0]
        )
        return output


class RootPathMotionSearchTests(unittest.TestCase):
    def test_candidate_naturalness_penalizes_crossing_and_double_support(self):
        frames = 40
        feet = np.zeros((frames, 2, 3), dtype=np.float64)
        feet[:, 0, 1] = 0.10
        feet[:, 1, 1] = -0.10
        alternating = np.zeros((frames, 2), dtype=np.bool_)
        alternating[:20, 0] = True
        alternating[20:, 1] = True

        clean = _candidate_naturalness_penalty(
            support=alternating,
            feet_scene=feet,
            heading_scene_xy=(1.0, 0.0),
        )
        crossed_feet = np.array(feet, copy=True)
        crossed_feet[:, 0, 1] = -0.13
        crossed = _candidate_naturalness_penalty(
            support=alternating,
            feet_scene=crossed_feet,
            heading_scene_xy=(1.0, 0.0),
        )
        double = _candidate_naturalness_penalty(
            support=np.ones((frames, 2), dtype=np.bool_),
            feet_scene=feet,
            heading_scene_xy=(1.0, 0.0),
        )

        self.assertEqual(clean, 0.0)
        self.assertGreater(crossed, clean)
        self.assertGreater(double, clean)

    def test_nonterminal_candidate_keeps_clean_source_splice_boundary(self):
        window = RawMotionWindow(
            source_clip="clip",
            start_frame=0,
            stop_frame=5,
            events=(
                RawContactEvent(0, 0, (0.0, 0.0), 0.0),
                RawContactEvent(3, 1, (0.3, 0.0), 0.0),
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=0.3,
            heading_error_rad=0.0,
        )
        support = np.array(
            [[True, True]] * 5
            + [[True, False]] * 3
            + [[True, True]] * 4,
            dtype=np.bool_,
        )
        feet = np.zeros((len(support), 2, 3), dtype=np.float64)
        surface = np.zeros((len(support), 2), dtype=np.float64)

        prepared = _prepare_candidate_window(
            window,
            support,
            feet,
            surface,
            require_terminal_double_support=False,
        )

        self.assertIs(prepared, window)
        self.assertEqual(prepared.stop_frame, 5)

    def test_certified_archive_preserves_composed_support_mask(self):
        frames = 3
        arrays = {
            "joint_position": np.zeros((frames, 29), dtype=np.float64),
            "root_position_world": np.array(
                ((0.0, 0.0, 0.8), (0.1, 0.0, 0.8), (0.2, 0.0, 0.8)),
                dtype=np.float64,
            ),
            "root_orientation_world_wxyz": np.tile(
                np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
                (frames, 1),
            ),
        }
        support = np.array(
            ((True, False), (True, True), (False, True)), dtype=np.bool_
        )

        archive = _certified_traversal_archive(
            arrays,
            support,
            target_root_scene_xy=np.array((0.0, 0.0), dtype=np.float64),
            alignment_yaw=0.0,
        )

        np.testing.assert_array_equal(archive["source_support_mask"], support)
        self.assertIsNot(archive["source_support_mask"], support)

    def test_nonterminal_double_support_extends_through_next_touchdown(self):
        window = RawMotionWindow(
            source_clip="clip",
            start_frame=0,
            stop_frame=5,
            events=(
                RawContactEvent(0, 0, (0.0, 0.0), 0.0),
                RawContactEvent(3, 1, (0.3, 0.0), 0.2),
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=0.3,
            heading_error_rad=0.0,
        )
        support = np.array(
            [[True, True]] * 6
            + [[False, True]] * 4
            + [[True, True]] * 6,
            dtype=np.bool_,
        )
        feet = np.zeros((len(support), 2, 3), dtype=np.float64)
        feet[10, 0, :2] = (0.7, 0.1)
        surface = np.zeros((len(support), 2), dtype=np.float64)
        surface[10, 0] = 0.2

        extended = _extend_window_through_next_touchdown(
            window, support, feet, surface, settle_frames=3
        )

        self.assertEqual(extended.stop_frame, len(support))
        self.assertEqual(extended.events[-1].frame, 10)
        self.assertEqual(extended.events[-1].foot, 0)
        self.assertEqual(extended.events[-1].position_world_xy, (0.7, 0.1))
        self.assertAlmostEqual(extended.events[-1].surface_height_m, 0.2)

    def test_nonterminal_extension_keeps_two_following_touchdowns(self):
        window = RawMotionWindow(
            source_clip="clip",
            start_frame=0,
            stop_frame=4,
            events=(
                RawContactEvent(0, 0, (0.0, 0.0), 0.0),
                RawContactEvent(2, 1, (0.2, 0.0), 0.2),
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=0.2,
            heading_error_rad=0.0,
        )
        support = np.array(
            [[True, True]] * 5
            + [[False, True]] * 3
            + [[True, True]] * 4
            + [[True, False]] * 3
            + [[True, True]] * 5,
            dtype=np.bool_,
        )
        feet = np.zeros((len(support), 2, 3), dtype=np.float64)
        surface = np.zeros((len(support), 2), dtype=np.float64)

        extended = _extend_window_through_next_touchdown(
            window,
            support,
            feet,
            surface,
            settle_frames=2,
            touchdown_count=2,
        )

        self.assertEqual([event.frame for event in extended.events], [0, 2, 8, 15])
        self.assertEqual(extended.stop_frame, len(support))

    def test_nonterminal_double_support_extends_through_stable_liftoff(self):
        window = RawMotionWindow(
            source_clip="clip",
            start_frame=0,
            stop_frame=6,
            events=(
                RawContactEvent(0, 0, (0.0, 0.0), 0.0),
                RawContactEvent(4, 1, (0.4, 0.0), 0.0),
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=0.4,
            heading_error_rad=0.0,
        )
        support = np.array(
            [[True, True]] * 8 + [[False, True]] * 6,
            dtype=np.bool_,
        )

        extended = _extend_window_through_stable_liftoff(
            window, support, settle_frames=3
        )

        self.assertEqual(extended.stop_frame, 11)
        self.assertEqual(extended.events, window.events)

    def test_boundary_blend_reprojects_root_to_incoming_support(self):
        frames = 4
        incoming_joints = np.zeros((frames, 29), dtype=np.float64)
        incoming_roots = np.zeros((frames, 3), dtype=np.float64)
        incoming_roots[:, 2] = 0.20
        incoming_quaternions = np.zeros((frames, 4), dtype=np.float64)
        incoming_quaternions[:, 0] = 1.0
        previous_joints = np.zeros(29, dtype=np.float64)
        previous_joints[0] = -0.10
        previous_root = np.array((0.0, 0.0, 0.20))
        support = np.array([[True, False]] * frames)
        kinematics = _RootRelativeSoleKinematics()

        joints, roots, quaternions = _contact_preserving_boundary_blend(
            joint_position=incoming_joints,
            root_position_world=incoming_roots,
            root_orientation_world_wxyz=incoming_quaternions,
            previous_joint_position=previous_joints,
            previous_root_position_world=previous_root,
            previous_root_orientation_world_wxyz=np.array(
                (1.0, 0.0, 0.0, 0.0)
            ),
            support_mask=support,
            blend_frames=frames,
            sole_kinematics=kinematics,
        )

        target = kinematics.sole_points(
            incoming_joints, incoming_roots, incoming_quaternions
        )
        actual = kinematics.sole_points(joints, roots, quaternions)
        np.testing.assert_allclose(actual[:, 0], target[:, 0], atol=1e-12)
        self.assertAlmostEqual(roots[0, 2], 0.30)
        self.assertAlmostEqual(roots[-1, 2], incoming_roots[-1, 2])

    def test_boundary_blend_interpolates_correction_through_flight(self):
        frames = 5
        incoming_joints = np.zeros((frames, 29), dtype=np.float64)
        incoming_roots = np.zeros((frames, 3), dtype=np.float64)
        incoming_roots[:, 2] = 0.20
        quaternions = np.zeros((frames, 4), dtype=np.float64)
        quaternions[:, 0] = 1.0
        previous_joints = np.zeros(29, dtype=np.float64)
        previous_joints[0] = -0.10
        support = np.array(
            [
                [True, False],
                [True, False],
                [False, False],
                [False, True],
                [False, True],
            ]
        )

        _, roots, _ = _contact_preserving_boundary_blend(
            joint_position=incoming_joints,
            root_position_world=incoming_roots,
            root_orientation_world_wxyz=quaternions,
            previous_joint_position=previous_joints,
            previous_root_position_world=np.array((0.0, 0.0, 0.20)),
            previous_root_orientation_world_wxyz=np.array(
                (1.0, 0.0, 0.0, 0.0)
            ),
            support_mask=support,
            blend_frames=frames,
            sole_kinematics=_RootRelativeSoleKinematics(),
        )

        self.assertTrue(np.isfinite(roots).all())
        self.assertGreater(roots[2, 2], 0.20)
        self.assertLess(roots[2, 2], 0.30)

    def test_boundary_blend_supports_certifiable_pose_only_strategy(self):
        frames = 4
        incoming_joints = np.full((frames, 29), 0.2, dtype=np.float64)
        incoming_roots = np.column_stack(
            (
                np.linspace(0.1, 0.13, frames),
                np.zeros(frames),
                np.full(frames, 0.8),
            )
        )
        incoming_quaternions = np.tile(
            (1.0, 0.0, 0.0, 0.0), (frames, 1)
        )
        previous_joints = np.zeros(29, dtype=np.float64)
        previous_root = np.array((0.0, 0.0, 0.8))
        previous_quaternion = np.array((1.0, 0.0, 0.0, 0.0))

        actual = _contact_preserving_boundary_blend(
            joint_position=incoming_joints,
            root_position_world=incoming_roots,
            root_orientation_world_wxyz=incoming_quaternions,
            previous_joint_position=previous_joints,
            previous_root_position_world=previous_root,
            previous_root_orientation_world_wxyz=previous_quaternion,
            support_mask=np.tile((True, False), (frames, 1)),
            blend_frames=frames,
            sole_kinematics=_RootRelativeSoleKinematics(),
            blend_mode="pose",
        )
        expected = blend_action_boundary(
            joint_position=incoming_joints,
            root_position_world=incoming_roots,
            root_orientation_world_wxyz=incoming_quaternions,
            previous_joint_position=previous_joints,
            previous_root_position_world=previous_root,
            previous_root_orientation_world_wxyz=previous_quaternion,
            blend_frames=frames,
        )

        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_allclose(actual_array, expected_array)

    def test_short_segments_keep_a_terminal_search_anchor(self):
        self.assertEqual(_minimum_search_window_length(0.60), 0.60)
        self.assertEqual(_minimum_search_window_length(0.90), 0.65)

    def test_only_the_final_search_window_requires_a_settle(self):
        self.assertFalse(
            _search_window_reaches_path_end(
                anchor_m=2.0, length_m=0.6, path_length_m=3.5
            )
        )
        self.assertTrue(
            _search_window_reaches_path_end(
                anchor_m=2.9, length_m=0.6, path_length_m=3.5
            )
        )

    def test_flat_and_ordered_paths_share_terminal_settle_invariant(self):
        self.assertTrue(
            _requires_terminal_settle(
                anchor_m=1.85,
                length_m=0.65,
                path_length_m=2.5,
            )
        )
        self.assertFalse(
            _requires_terminal_settle(
                anchor_m=1.4,
                length_m=0.9,
                path_length_m=2.5,
            )
        )

    def test_explicit_path_endpoints_resolve_diagonal_heading(self):
        start, stop, heading, length = _resolve_path(
            SimpleNamespace(
                path_start=(-1.0, 1.0),
                path_stop=(1.0, -1.0),
                start_x=-1.25,
                stop_x=1.25,
                path_y=None,
            )
        )

        np.testing.assert_allclose(start, (-1.0, 1.0))
        np.testing.assert_allclose(stop, (1.0, -1.0))
        np.testing.assert_allclose(
            heading, (np.sqrt(0.5), -np.sqrt(0.5))
        )
        self.assertAlmostEqual(length, np.sqrt(8.0))

    def test_legacy_path_arguments_resolve_horizontal_heading(self):
        start, stop, heading, length = _resolve_path(
            SimpleNamespace(
                path_start=None,
                path_stop=None,
                start_x=-1.0,
                stop_x=2.0,
                path_y=0.4,
            )
        )

        np.testing.assert_allclose(start, (-1.0, 0.4))
        np.testing.assert_allclose(stop, (2.0, 0.4))
        np.testing.assert_allclose(heading, (1.0, 0.0))
        self.assertAlmostEqual(length, 3.0)

    def test_path_resolution_rejects_mixed_and_degenerate_inputs(self):
        with self.assertRaisesRegex(ContractError, "ambiguous"):
            _resolve_path(
                SimpleNamespace(
                    path_start=(-1.0, 1.0),
                    path_stop=(1.0, -1.0),
                    start_x=-1.0,
                    stop_x=1.0,
                    path_y=0.4,
                )
            )
        with self.assertRaisesRegex(ContractError, "invalid"):
            _resolve_path(
                SimpleNamespace(
                    path_start=(0.0, 0.0),
                    path_stop=(0.0, 0.0),
                    start_x=-1.0,
                    stop_x=1.0,
                    path_y=None,
                )
            )

    def test_scanner_job_includes_phase_kinds(self):
        queries = (("query-a",), ("query-b",))
        lengths = (0.8, 0.9)
        phases = (
            SimpleNamespace(kind="mount"),
            SimpleNamespace(kind="interior"),
        )

        job = _scanner_job(
            "dataset", {"logical_name": "clip"}, queries, lengths, phases
        )

        self.assertEqual(len(job), 5)
        self.assertEqual(job[:4], ("dataset", {"logical_name": "clip"}, queries, lengths))
        self.assertEqual(job[4], ("mount", "interior"))

    def test_transforms_actual_touchdowns_into_target_path_observations(self):
        window = RawMotionWindow(
            source_clip="clip",
            start_frame=0,
            stop_frame=3,
            events=(
                RawContactEvent(0, 0, (0.0, 0.0), 0.0),
                RawContactEvent(2, 1, (1.0, 0.0), 0.0),
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=1.0,
            heading_error_rad=0.0,
        )

        progress, height, feet, frames = _placed_touchdowns(
            window=window,
            yaw_scene_rad=0.5 * math.pi,
            translation_scene_xyz=(1.0, 2.0, 0.0),
            path_start_scene_xy=(1.0, 2.0),
            path_heading_scene_xy=(0.0, 1.0),
            sample_surface=lambda points: 0.187 * (points[..., 1] > 2.5),
        )

        np.testing.assert_allclose(progress, (0.0, 1.0))
        np.testing.assert_allclose(height, (0.0, 0.187))
        self.assertEqual(feet, (0, 1))
        self.assertEqual(frames, (0, 2))

    def test_observed_two_level_jump_is_rejected(self):
        def surface(points):
            y = np.asarray(points)[..., 1]
            return np.select(
                (y > 0.90, y > 0.65, y > 0.40),
                (0.0, 0.187, 0.349),
                default=0.0,
            )

        contract = derive_ordered_terrain_levels(
            path_start_scene_xy=(-0.5, 1.4),
            path_stop_scene_xy=(0.7, 0.2),
            sample_surface=surface,
        )

        with self.assertRaisesRegex(
            ContractError, "skipped-required-level"
        ):
            match_ordered_touchdown_levels(
                contract,
                touchdown_progress_m=(0.10, 1.08),
                touchdown_height_m=(0.0, 0.349),
                touchdown_foot=(0, 1),
                require_complete=False,
            )

    def test_cli_enables_ordered_contact_levels_explicitly(self):
        args = _parser().parse_args(
            [
                "--source-dataset",
                "dataset",
                "--target-scene",
                "scene",
                "--g1-xml",
                "g1.xml",
                "--path-start",
                "0",
                "1",
                "--path-stop",
                "1",
                "0",
                "--ordered-contact-levels",
                "--output",
                "output",
            ]
        )

        self.assertTrue(args.ordered_contact_levels)

    def test_ordered_contract_labels_mount_and_dismount_windows(self):
        def surface(points):
            y = np.asarray(points)[..., 1]
            return np.select(
                (y > 0.90, y > 0.65, y > 0.40),
                (0.0, 0.187, 0.349),
                default=0.0,
            )

        contract = derive_ordered_terrain_levels(
            path_start_scene_xy=(-0.5, 1.4),
            path_stop_scene_xy=(0.7, 0.2),
            sample_surface=surface,
        )

        self.assertEqual(_ordered_phase_kind(contract, 0.0, 0.9), "mount")
        self.assertEqual(
            _ordered_phase_kind(contract, 0.8, 1.3), "interior"
        )
        self.assertEqual(
            _ordered_phase_kind(contract, 1.10, contract.path_length_m),
            "dismount",
        )
        self.assertEqual(
            _ordered_phase_kind(contract, 0.80, contract.path_length_m),
            "interior",
        )

    def test_an_ascending_terminal_boundary_is_a_mount(self):
        def surface(points):
            progress = np.asarray(points)[..., 0]
            return np.select(
                (progress >= 1.2, progress >= 0.8, progress >= 0.4),
                (0.6, 0.4, 0.2),
                default=0.0,
            )

        contract = derive_ordered_terrain_levels(
            path_start_scene_xy=(0.0, 0.0),
            path_stop_scene_xy=(1.6, 0.0),
            sample_surface=surface,
        )

        self.assertEqual(
            _ordered_phase_kind(contract, 1.0, 1.6), "mount"
        )

    def test_dismount_window_extends_through_terminal_double_support(self):
        window = RawMotionWindow(
            source_clip="clip",
            start_frame=2,
            stop_frame=10,
            events=(
                RawContactEvent(5, 0, (0.0, 0.0), 0.2),
                RawContactEvent(9, 1, (0.2, 0.0), 0.0),
            ),
            root_start_world_xy=(0.0, 0.0),
            forward_progress_m=0.2,
            heading_error_rad=0.0,
        )
        support = np.zeros((20, 2), dtype=np.bool_)
        support[12:16] = True

        extended = _extend_window_to_terminal_double_support(
            window, support
        )

        self.assertEqual(extended.stop_frame, 16)
        self.assertEqual(extended.events, window.events)

    def candidate(
        self, candidate_id, anchor, support, *, source_clip=None
    ):
        frames = len(support)
        root = np.zeros((frames, 3), dtype=np.float64)
        root[:, 0] = anchor + np.linspace(0.0, 0.5, frames)
        quaternion = np.zeros((frames, 4), dtype=np.float64)
        quaternion[:, 0] = 1.0
        return {
            "candidate_id": candidate_id,
            "anchor_m": anchor,
            "interval_start_m": anchor,
            "interval_stop_m": anchor + 0.5,
            "row": {
                "source_clip": source_clip or candidate_id,
                "start_frame": 0,
                "stop_frame": frames,
                "events": [
                    {"surface_height_m": 0.0},
                    {"surface_height_m": 0.0},
                ],
            },
            "scene": {
                "joint_position": np.zeros((frames, 29), dtype=np.float64),
                "root_position_world": root,
                "root_orientation_world_wxyz": quaternion,
            },
            "support": np.asarray(support, dtype=np.bool_),
            "path_progress_m": anchor + np.linspace(
                0.0, 0.5, frames
            ),
        }

    def test_transition_search_cannot_invent_a_support_contact(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 8
        )
        second = self.candidate(
            "second", 0.4, [[True, True]] * 8
        )

        self.assertIsNone(_best_transition(first, second))

    def test_transition_can_release_one_foot_from_double_support(self):
        first = self.candidate(
            "first", 0.0, [[True, True]] * 8
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 8
        )

        self.assertIsNotNone(_best_transition(first, second))

    def test_transition_prefers_continuing_the_same_source_window(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 12,
            source_clip="shared",
        )
        same = self.candidate(
            "same", 0.4, [[True, False]] * 12,
            source_clip="shared",
        )
        same["row"]["start_frame"] = 1
        same["row"]["stop_frame"] = 13
        different = self.candidate(
            "different", 0.4, [[True, False]] * 12,
            source_clip="other",
        )

        same_result = _best_transition(first, same)
        different_result = _best_transition(first, different)

        self.assertIsNotNone(same_result)
        self.assertIsNotNone(different_result)
        self.assertLess(same_result[0], different_result[0])

    def test_same_anchor_can_transition_to_a_farther_reaching_window(self):
        first = self.candidate(
            "first", 1.0, [[True, False]] * 12
        )
        second = self.candidate(
            "second", 1.0, [[True, False]] * 16
        )
        second["interval_stop_m"] = 1.8
        second["scene"]["root_position_world"][:, 0] = np.linspace(
            1.0, 1.8, 16
        )
        second["path_progress_m"] = np.linspace(1.0, 1.8, 16)

        self.assertIsNotNone(_best_transition(first, second))

    def test_transition_search_checks_phase_matches_at_overlap_edge(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 13
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 13
        )
        first["interval_stop_m"] = 1.2
        second["interval_stop_m"] = 1.4
        first["path_progress_m"] = np.linspace(0.0, 1.2, 13)
        second["path_progress_m"] = np.linspace(0.4, 1.4, 13)
        first["scene"]["root_position_world"][:, 0] = first[
            "path_progress_m"
        ]
        second["scene"]["root_position_world"][:, 0] = second[
            "path_progress_m"
        ]
        second["scene"]["joint_position"][:] = 10.0
        second["scene"]["joint_position"][-4:] = 0.0

        result = _best_transition(first, second)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result[2], 9)

    def test_transition_search_tries_next_pose_match_when_best_is_uncertified(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 13
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 13
        )
        first["interval_stop_m"] = 1.2
        second["interval_stop_m"] = 1.4
        first["path_progress_m"] = np.linspace(0.0, 1.2, 13)
        second["path_progress_m"] = np.linspace(0.4, 1.4, 13)
        first["scene"]["root_position_world"][:, 0] = first[
            "path_progress_m"
        ]
        second["scene"]["root_position_world"][:, 0] = second[
            "path_progress_m"
        ]
        rejected = _best_transition(first, second)
        self.assertIsNotNone(rejected)
        calls = []

        def certify(first_frame, second_frame):
            calls.append((first_frame, second_frame))
            if (first_frame, second_frame) == rejected[1:]:
                return None
            return {"certified": 1.0}

        result = _best_certified_transition(
            first, second, certify=certify, maximum_attempts=16
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(calls), 2)
        self.assertNotEqual(result[0][1:], rejected[1:])
        self.assertEqual(result[1], {"certified": 1.0})

    def test_transition_progress_is_independent_of_global_x(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 12
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 12
        )
        first["scene"]["root_position_world"][:, 0] = 0.0
        second["scene"]["root_position_world"][:, 0] = 0.0

        self.assertIsNotNone(_best_transition(first, second))

    def test_transition_allows_a_certifiable_eight_frame_root_blend(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 12
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 12
        )
        second["scene"]["root_position_world"][:, 2] = 0.17

        self.assertIsNotNone(_best_transition(first, second))

    def test_transition_preserves_a_two_touchdown_terminal_unit(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 16
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 16
        )
        first["interval_stop_m"] = 0.9
        first["scene"]["root_position_world"][:, 0] = np.linspace(
            0.0, 0.9, 16
        )
        first["path_progress_m"] = np.linspace(0.0, 0.9, 16)
        second["interval_stop_m"] = 1.3
        second["scene"]["root_position_world"][:, 0] = np.linspace(
            0.4, 1.3, 16
        )
        second["path_progress_m"] = np.linspace(0.4, 1.3, 16)
        second["ordered_levels"] = (3, 3)
        second["touchdown_frames"] = (4, 10)
        second["support"][-1] = (True, True)

        result = _best_transition(first, second)

        self.assertIsNotNone(result)
        self.assertLessEqual(result[2], 4)

    def test_transition_certification_rejects_blended_penetration(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 8
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 8
        )
        second["scene"]["root_position_world"][:, 0] = np.linspace(
            0.4, 0.6, 8
        )

        result = _certify_transition(
            first=first,
            second=second,
            first_frame=6,
            second_frame=0,
            blend_frames=4,
            sole_kinematics=_FakeSoleKinematics(-0.04),
            sample_surface=lambda points: np.zeros(points.shape[:-1]),
        )

        self.assertIsNone(result)

    def test_transition_certification_accepts_supported_clearance(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 8
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 8
        )
        second["scene"]["root_position_world"][:, 0] = np.linspace(
            0.4, 0.6, 8
        )

        result = _certify_transition(
            first=first,
            second=second,
            first_frame=6,
            second_frame=0,
            blend_frames=4,
            sole_kinematics=_FakeSoleKinematics(0.01),
            sample_surface=lambda points: np.zeros(points.shape[:-1]),
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result["maximum_supported_sole_error_m"], 0.01
        )

    def test_transition_certification_rejects_boundary_pose_jump(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 8
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 8
        )
        second["scene"]["joint_position"][:, 0] = 0.40

        result = _certify_transition(
            first=first,
            second=second,
            first_frame=6,
            second_frame=0,
            blend_frames=4,
            sole_kinematics=_FakeSoleKinematics(0.01),
            sample_surface=lambda points: np.zeros(points.shape[:-1]),
        )

        self.assertIsNone(result)

    def test_transition_requires_three_nearby_support_samples(self):
        first = self.candidate(
            "first", 0.0, [[True, False]] * 8
        )
        second = self.candidate(
            "second", 0.4, [[True, False]] * 8
        )

        result = _certify_transition(
            first=first,
            second=second,
            first_frame=6,
            second_frame=0,
            blend_frames=4,
            sole_kinematics=_FakeSoleKinematics((0.01, 0.20, 0.20)),
            sample_surface=lambda points: np.zeros(points.shape[:-1]),
        )

        self.assertIsNone(result)

    def test_reused_source_window_cannot_rewind_source_time(self):
        first = self.candidate(
            "first-placement",
            0.0,
            [[True, False]] * 8,
            source_clip="same-window",
        )
        second = self.candidate(
            "second-placement",
            0.4,
            [[True, False]] * 8,
            source_clip="same-window",
        )

        self.assertIsNone(_best_transition(first, second))


if __name__ == "__main__":
    unittest.main()
