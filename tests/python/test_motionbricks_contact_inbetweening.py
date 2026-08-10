from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.build_motionbricks_contact_terrain_pilots import (
    _audit_partial_rigid_support_contact,
    _refit_motion_to_sole_targets,
    _repair_stance_reach_with_root_lowering,
    _repair_stance_reach_with_pelvis_tilt,
    _smoothly_subdivide_motion_steps,
)
from mm_sonic.canonical_terrain_matcher import RegularGridHeightField
from mm_sonic.postprocess_continuous_terrain_contacts import (
    _crop_processed_motion,
    _maximum_root_acceleration_m_s2,
    _minimal_vertical_swing_clearance,
    _resample_trace_arrays,
    _rigid_stance_extras_from_trace,
    _smooth_root_positions,
    _terrain_index_near_motion,
)
from mm_sonic.terrain_oracle.stitch import FrameProvenance, StitchedMotion


class MotionBricksContactInbetweeningTest(unittest.TestCase):
    def test_trace_labels_follow_local_time_subdivision(self) -> None:
        coordinate = np.asarray((0.0, 0.5, 1.0, 2.0, 3.0), dtype=np.float64)
        result = _resample_trace_arrays(
            {
                "frame_value": np.asarray((0.0, 1.0, 2.0, 3.0)),
                "contact": np.asarray((False, True, False)),
                "clip": np.asarray((4, 5, 6), dtype=np.int32),
            },
            coordinate,
            original_frame_count=4,
        )

        np.testing.assert_allclose(
            result["frame_value"], (0.0, 0.5, 1.0, 2.0, 3.0)
        )
        np.testing.assert_array_equal(
            result["contact"], (False, False, True, False)
        )
        np.testing.assert_array_equal(result["clip"], (4, 4, 5, 6))

    def test_minimal_swing_guidance_preserves_xy_and_only_lifts_flight(self) -> None:
        field = RegularGridHeightField(
            np.full((5, 5), 0.2, dtype=np.float64),
            spacing_m=0.5,
            origin_xy=(-1.0, -1.0),
        )
        nominal = np.zeros((5, 2, 4, 3), dtype=np.float32)
        nominal[..., 2] = 0.10
        nominal[:, 0, :, 1] = 0.10
        nominal[:, 1, :, 1] = -0.10
        planted = nominal.copy()
        planted[[0, 4], :, :, 2] = 0.205
        stance = np.zeros((5, 2), dtype=bool)
        stance[[0, 4]] = True

        class Adapter:
            @staticmethod
            def sole_sphere_radii():
                value = np.full(4, 0.005, dtype=np.float64)
                return value, value

        updated, report = _minimal_vertical_swing_clearance(
            {
                "authored_stance_mask": stance,
                "planned_nominal_sole_points_world": nominal,
                "target_sole_points_world": planted,
            },
            field=field,
            adapter=Adapter(),
        )

        target = updated["target_sole_points_world"]
        np.testing.assert_allclose(target[..., :2], nominal[..., :2])
        np.testing.assert_allclose(target[stance], planted[stance])
        self.assertTrue(np.all(target[1:4, :, :, 2] >= 0.213 - 1.0e-6))
        self.assertGreater(report["active_frame_foot_count"], 0)
        self.assertEqual(report["maximum_planar_adjustment_m"], 0.0)

    def test_exact_audit_terrain_is_cropped_to_motion_corridor(self) -> None:
        field = RegularGridHeightField(
            np.zeros((31, 61), dtype=np.float64),
            spacing_m=0.1,
            origin_xy=(-3.0, -1.5),
        )
        frames = 5
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = np.linspace(-0.2, 0.2, frames)
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=root,
            root_quaternion_world_wxyz=np.tile(
                np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                (frames, 1),
            ),
            joint_position=np.zeros((frames, 1), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frames)
            ),
            seam_indices=(),
        )

        local, report = _terrain_index_near_motion(
            field.index, motion, horizontal_margin_m=0.5
        )

        self.assertLess(
            report["selected_face_count"], report["source_face_count"]
        )
        for point in root[:, :2]:
            hit = local.raycast(
                np.asarray((point[0], point[1], 1.0)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            self.assertIsNotNone(hit)
            self.assertAlmostEqual(float(hit.position_world[2]), 0.0)

    def test_processed_crop_keeps_hidden_context_out_of_saved_arrays(self) -> None:
        frames = 7
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=np.arange(frames * 3, dtype=np.float32).reshape(
                frames, 3
            ),
            root_quaternion_world_wxyz=np.tile(
                np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                (frames, 1),
            ),
            joint_position=np.zeros((frames, 1), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frames)
            ),
            seam_indices=(1, 4, 6),
        )
        cropped, extras, arrays = _crop_processed_motion(
            motion,
            {"frame": np.arange(frames), "constant": np.asarray(3)},
            {
                "frame": np.arange(frames),
                "interval": np.arange(frames - 1),
            },
            start=2,
            stop=6,
        )

        self.assertEqual(len(cropped.root_position_world), 4)
        self.assertEqual(cropped.seam_indices, (2,))
        self.assertEqual(extras["frame"].tolist(), [2, 3, 4, 5])
        self.assertEqual(arrays["interval"].tolist(), [2, 3, 4])

    def test_stance_only_refit_does_not_solve_unconstrained_flight_frames(self) -> None:
        frame_count = 5
        root = np.zeros((frame_count, 3), dtype=np.float32)
        root[:, 0] = np.arange(frame_count, dtype=np.float32)
        quaternion = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        )
        joints = np.zeros((frame_count, 1), dtype=np.float32)
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=root,
            root_quaternion_world_wxyz=quaternion,
            joint_position=joints,
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        points = np.zeros((frame_count, 2, 4, 3), dtype=np.float32)
        points[:, 0, :, 1] = 0.10
        points[:, 1, :, 1] = -0.10
        points[:, :, :, 0] = root[:, None, None, 0]
        points[2, 0, :, 2] = 0.01
        points[3, 1, :, 2] = 0.01
        stance = np.zeros((frame_count, 2), dtype=bool)
        stance[2, 0] = True
        additional = np.zeros((frame_count, 2), dtype=bool)
        additional[3, 1] = True
        extras = {
            "target_sole_points_world": points,
            "authored_stance_mask": stance,
        }

        class Adapter:
            solved_root_x: list[float] = []

            @staticmethod
            def sole_positions_for_pose(*, root_position, **_pose):
                value = np.zeros((2, 4, 3), dtype=np.float64)
                value[0, :, 1] = 0.10
                value[1, :, 1] = -0.10
                value[:, :, 0] = float(root_position[0])
                return value[0], value[1]

            def adapt_to_targets(self, *, root_position, authored_joints, **_fit):
                self.solved_root_x.append(float(root_position[0]))
                return np.asarray(authored_joints).copy(), 0.0, 0.01

        adapter = Adapter()
        repaired, updated, _report = _refit_motion_to_sole_targets(
            motion,
            extras,
            adapter=adapter,
            stance_only=True,
            additional_target_mask=additional,
        )

        self.assertTrue(adapter.solved_root_x)
        self.assertEqual(set(adapter.solved_root_x), {2.0, 3.0})
        self.assertFalse(np.any(updated["sole_target_point_mask"][3, 0]))
        self.assertTrue(np.all(updated["sole_target_point_mask"][3, 1]))
        np.testing.assert_allclose(repaired.joint_position, joints)

    def test_continuous_root_impulse_is_smoothed_before_plants_are_locked(self) -> None:
        root = np.zeros((61, 3), dtype=np.float64)
        root[:, 2] = 0.78 + 0.005 * np.sin(np.arange(61) * 0.25)
        velocity = np.full(60, 0.008, dtype=np.float64)
        velocity[30:] = 0.018
        root[1:, 0] = np.cumsum(velocity)

        smoothed, report = _smooth_root_positions(
            root, fps=50.0, target_maximum_acceleration_m_s2=25.0
        )

        self.assertTrue(report["applied"])
        self.assertLessEqual(
            _maximum_root_acceleration_m_s2(smoothed, fps=50.0), 25.0
        )
        self.assertLess(report["maximum_root_adjustment_m"], 0.03)
        np.testing.assert_allclose(smoothed[0], root[0], atol=1.0e-12)

    def test_trace_stance_uses_only_slow_contact_cores(self) -> None:
        frame_count = 21
        root = np.zeros((frame_count, 3), dtype=np.float32)
        root[:, 2] = 0.78
        quaternion = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        )
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=root,
            root_quaternion_world_wxyz=quaternion,
            joint_position=np.zeros((frame_count, 1), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        contact = np.zeros((frame_count - 1, 2), dtype=bool)
        contact[2:18, 0] = True

        class Adapter:
            frame = -1

            def sole_positions_for_pose(self, **_pose):
                self.frame += 1
                left_x = 0.001 * self.frame
                if 8 <= self.frame <= 12:
                    left_x += 0.03 * (self.frame - 8)
                quad = np.asarray(
                    (
                        (-0.09, -0.05, 0.0),
                        (0.09, -0.05, 0.0),
                        (0.09, 0.05, 0.0),
                        (-0.09, 0.05, 0.0),
                    ),
                    dtype=np.float64,
                )
                left = quad.copy()
                left[:, 0] += left_x
                right = quad.copy()
                right[:, 1] += 0.20
                return left, right

        extras, report = _rigid_stance_extras_from_trace(
            motion,
            contact,
            adapter=Adapter(),
            maximum_stance_speed_mps=0.18,
            minimum_stance_run_frames=3,
        )

        stance = extras["authored_stance_mask"]
        self.assertTrue(np.any(stance[:8, 0]))
        self.assertFalse(np.any(stance[8:13, 0]))
        self.assertTrue(np.any(stance[14:, 0]))
        self.assertEqual(report["stance_span_count"], 2)

    def test_trace_stance_falls_back_to_slow_exact_surface_geometry(self) -> None:
        frame_count = 31
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=np.tile(
                np.asarray((0.0, 0.0, 0.78), dtype=np.float32),
                (frame_count, 1),
            ),
            root_quaternion_world_wxyz=np.tile(
                np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                (frame_count, 1),
            ),
            joint_position=np.zeros((frame_count, 1), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        field = RegularGridHeightField(
            np.zeros((21, 41), dtype=np.float64),
            spacing_m=0.1,
            origin_xy=(-1.0, -1.0),
        )

        class Adapter:
            frame = -1

            @staticmethod
            def sole_sphere_radii():
                value = np.full(4, 0.02, dtype=np.float64)
                return value, value

            def sole_positions_for_pose(self, **_pose):
                self.frame += 1
                quad = np.asarray(
                    (
                        (-0.09, -0.05, 0.0),
                        (0.09, -0.05, 0.0),
                        (0.09, 0.05, 0.0),
                        (-0.09, 0.05, 0.0),
                    ),
                    dtype=np.float64,
                )
                left = quad.copy()
                right = quad.copy()
                if self.frame <= 12:
                    left[:, 0] += 0.0
                    left[:, 2] = 0.02
                else:
                    left[:, 0] += 0.03 * (self.frame - 12)
                    left[:, 2] = 0.12
                if self.frame >= 16:
                    right[:, 0] += 0.45
                    right[:, 2] = 0.02
                else:
                    right[:, 0] += 0.45 - 0.03 * (16 - self.frame)
                    right[:, 2] = 0.12
                right[:, 1] += 0.20
                return left, right

        extras, report = _rigid_stance_extras_from_trace(
            motion,
            np.zeros((frame_count - 1, 2), dtype=bool),
            adapter=Adapter(),
            maximum_stance_speed_mps=0.18,
            minimum_stance_run_frames=4,
            target_height_field=field,
        )

        stance = extras["authored_stance_mask"]
        self.assertTrue(report["geometry_fallback_used"])
        self.assertEqual(report["source_stance_frame_foot_count"], 0)
        self.assertTrue(np.any(stance[:12, 0]))
        self.assertFalse(np.any(stance[16:, 0]))
        self.assertFalse(np.any(stance[:12, 1]))
        self.assertTrue(np.any(stance[18:, 1]))
        self.assertEqual(report["stance_span_count"], 2)

        native_speed = np.full((frame_count - 1, 2), 1.0, dtype=np.float64)
        native_speed[:11, 0] = 0.002
        native_speed[17:, 1] = 0.003
        native_extras, native_report = _rigid_stance_extras_from_trace(
            motion,
            np.zeros((frame_count - 1, 2), dtype=bool),
            adapter=Adapter(),
            maximum_stance_speed_mps=0.18,
            minimum_stance_run_frames=4,
            target_height_field=field,
            source_sole_speed_mps=native_speed,
        )
        self.assertTrue(native_report["source_kinematic_fallback_used"])
        self.assertFalse(native_report["geometry_fallback_used"])
        self.assertTrue(np.any(native_extras["authored_stance_mask"][:10, 0]))
        self.assertTrue(np.any(native_extras["authored_stance_mask"][20:, 1]))

        source_height = np.full((frame_count, 2), 0.08, dtype=np.float64)
        source_height[:11, 0] = 0.006
        source_height[17:, 1] = 0.005
        height_extras, height_report = _rigid_stance_extras_from_trace(
            motion,
            np.zeros((frame_count - 1, 2), dtype=bool),
            adapter=Adapter(),
            maximum_stance_speed_mps=0.18,
            minimum_stance_run_frames=4,
            target_height_field=field,
            source_sole_speed_mps=np.ones((frame_count, 2)),
            source_sole_height_above_ground_m=source_height,
            maximum_source_sole_ground_clearance_m=0.015,
        )
        self.assertTrue(height_report["source_ground_height_stance_used"])
        self.assertFalse(height_report["geometry_fallback_used"])
        np.testing.assert_array_equal(
            height_extras["authored_stance_mask"],
            source_height <= 0.015,
        )

    def test_unreachable_slope_stance_uses_a_smooth_bounded_pelvis_tilt(self) -> None:
        frame_count = 25
        root = np.zeros((frame_count, 3), dtype=np.float32)
        root[:, 2] = 0.78
        quaternion = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        )
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=root,
            root_quaternion_world_wxyz=quaternion,
            joint_position=np.zeros((frame_count, 1), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        target = np.zeros((frame_count, 2, 4, 3), dtype=np.float32)
        target[..., 0] = np.asarray((-0.09, 0.09, 0.09, -0.09))
        target[..., 1] = np.asarray((-0.05, -0.05, 0.05, 0.05))
        target[:8, ..., 2] = 0.02
        target[17:, ..., 2] = 0.02
        stance = np.ones((frame_count, 2), dtype=bool)
        initial_error = np.zeros((frame_count, 2), dtype=np.float32)
        initial_error[8:17] = 0.02
        extras = {
            "target_sole_points_world": target,
            "sole_target_point_mask": np.ones(
                (frame_count, 2, 4), dtype=bool
            ),
            "authored_stance_mask": stance,
            "per_frame_sole_target_error_by_foot_m": initial_error,
            "planned_pelvis_height_minimum_world_m": np.full(
                frame_count, 0.70, dtype=np.float32
            ),
            "planned_pelvis_height_maximum_world_m": np.full(
                frame_count, 0.85, dtype=np.float32
            ),
        }

        class Adapter:
            @staticmethod
            def sole_positions_for_pose(*, root_quaternion_wxyz, **_pose):
                points = target[0, 0].astype(np.float64).copy()
                # A negative local pitch brings this synthetic sole toward
                # the fixed slope target; root height deliberately has no
                # effect so the search must exercise quaternion conversion.
                points[:, 2] = 0.02 + 0.20 * float(root_quaternion_wxyz[2])
                return points.copy(), points.copy()

            @staticmethod
            def adapt_to_targets(*, authored_joints, **_fit):
                return np.asarray(authored_joints).copy(), 0.0, 0.0

        repaired, repaired_extras, report = (
            _repair_stance_reach_with_pelvis_tilt(
                motion,
                extras,
                adapter=Adapter(),
                transition_frames=4,
            )
        )

        self.assertTrue(report["applied"])
        self.assertGreater(report["maximum_local_pitch_adjustment_rad"], 0.0)
        self.assertLess(report["final_maximum_stance_error_m"], 0.02)
        correction = repaired_extras["stance_reach_pelvis_pose_adjustment"]
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(correction[:, 2])))),
            np.deg2rad(3.0),
        )
        self.assertFalse(
            np.allclose(repaired.root_quaternion_world_wxyz, quaternion)
        )

    def test_overextended_double_support_is_lowered_smoothly(self) -> None:
        frame_count = 25
        root = np.zeros((frame_count, 3), dtype=np.float32)
        root[:, 2] = 0.80
        quaternion = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        )
        joints = np.zeros((frame_count, 1), dtype=np.float32)
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=root,
            root_quaternion_world_wxyz=quaternion,
            joint_position=joints,
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        target = np.zeros((frame_count, 2, 4, 3), dtype=np.float32)
        target[..., 0] = np.asarray((-0.09, 0.09, 0.09, -0.09))
        target[..., 1] = np.asarray((-0.05, -0.05, 0.05, 0.05))
        stance = np.ones((frame_count, 2), dtype=bool)
        extras = {
            "target_sole_points_world": target,
            "authored_stance_mask": stance,
            "per_frame_sole_target_error_by_foot_m": np.full(
                (frame_count, 2), 0.03, dtype=np.float32
            ),
            "planned_pelvis_height_minimum_world_m": np.full(
                frame_count, 0.70, dtype=np.float32
            ),
        }

        class Adapter:
            @staticmethod
            def sole_positions_for_pose(*, root_position, **_pose):
                points = target[0, 0].astype(np.float64).copy()
                points[:, 2] = float(root_position[2]) - 0.77
                return points.copy(), points.copy()

            def adapt_to_targets(self, *, authored_joints, root_position, **_fit):
                points = self.sole_positions_for_pose(root_position=root_position)
                error = max(
                    float(np.max(np.linalg.norm(points[foot] - target[0, foot], axis=1)))
                    for foot in range(2)
                )
                return np.asarray(authored_joints).copy(), 0.0, error

        repaired, repaired_extras, report = (
            _repair_stance_reach_with_root_lowering(
                motion,
                extras,
                adapter=Adapter(),
                maximum_iterations=1,
            )
        )

        self.assertTrue(report["applied"])
        self.assertGreater(report["maximum_root_lower_m"], 0.02)
        self.assertLess(report["final_maximum_stance_error_m"], 0.01)
        self.assertLessEqual(report["maximum_root_lower_step_m"], 1.0e-6)
        self.assertLess(
            float(
                np.max(
                    repaired_extras["per_frame_sole_target_error_by_foot_m"]
                )
            ),
            0.01,
        )
        self.assertTrue(np.all(repaired.root_position_world[:, 2] < 0.78))

    def test_joint_transition_is_inbetweened_without_root_speed_impulse(self) -> None:
        frame_count = 41
        root = np.zeros((frame_count, 3), dtype=np.float32)
        root[:, 0] = 0.01 * np.arange(frame_count)
        quaternion = np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        )
        joints = np.zeros((frame_count, 3), dtype=np.float32)
        joints[20:, 0] = 0.42
        provenance = tuple(
            FrameProvenance(0, frame, "synthetic")
            for frame in range(frame_count)
        )
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=root,
            root_quaternion_world_wxyz=quaternion,
            joint_position=joints,
            provenance=provenance,
            seam_indices=(),
        )

        result, extras, report = _smoothly_subdivide_motion_steps(
            motion,
            {"authored_stance_mask": np.ones((frame_count, 2), dtype=bool)},
            target_joint_step_rad=0.14,
        )

        self.assertGreater(report["added_frame_count"], 0.0)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(result.joint_position, axis=0)))),
            0.14,
        )
        root_acceleration = (
            np.diff(result.root_position_world.astype(np.float64), n=2, axis=0)
            * result.fps**2
        )
        self.assertLess(
            float(np.max(np.linalg.norm(root_acceleration, axis=1))), 3.0
        )
        self.assertEqual(
            extras["authored_stance_mask"].shape,
            (len(result.root_position_world), 2),
        )

    def test_partial_contact_requires_real_collision_sphere_support(self) -> None:
        frame_count = 4
        motion = StitchedMotion(
            fps=50.0,
            root_position_world=np.zeros((frame_count, 3), dtype=np.float32),
            root_quaternion_world_wxyz=np.tile(
                np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
                (frame_count, 1),
            ),
            joint_position=np.zeros((frame_count, 3), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(0, frame, "synthetic")
                for frame in range(frame_count)
            ),
            seam_indices=(),
        )
        field = RegularGridHeightField(
            np.zeros((11, 11), dtype=np.float64),
            spacing_m=0.05,
            origin_xy=(-0.25, -0.25),
        )
        extras = {
            "authored_stance_mask": np.ones((frame_count, 2), dtype=bool),
            "planned_partial_rigid_support_mask": np.tile(
                np.asarray((True, False), dtype=bool), (frame_count, 1)
            ),
            "per_frame_sole_target_error_by_foot_m": np.zeros(
                (frame_count, 2), dtype=np.float32
            ),
        }

        class Adapter:
            def __init__(self, partial_heights: tuple[float, ...]) -> None:
                self.partial_heights = partial_heights

            def sole_support_points_for_pose(self, **_pose):
                xy = np.asarray(
                    ((-0.09, -0.05), (0.09, -0.05), (0.09, 0.05), (-0.09, 0.05)),
                    dtype=np.float64,
                )
                partial = np.column_stack((xy, self.partial_heights))
                strict = np.column_stack((xy, np.zeros(4)))
                return partial, strict

        accepted = _audit_partial_rigid_support_contact(
            motion,
            extras,
            adapter=Adapter((0.0, 0.0, 0.02, 0.02)),
            target_mesh=field.index,
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["minimum_partial_contact_probe_count"], 2)

        rejected = _audit_partial_rigid_support_contact(
            motion,
            extras,
            adapter=Adapter((0.0, 0.02, 0.02, 0.02)),
            target_mesh=field.index,
        )
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["minimum_partial_contact_probe_count"], 1)


if __name__ == "__main__":
    unittest.main()
