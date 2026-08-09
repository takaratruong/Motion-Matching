from __future__ import annotations

import unittest

import numpy as np

from mm_sonic.build_motionbricks_contact_terrain_pilots import (
    _audit_partial_rigid_support_contact,
    _repair_stance_reach_with_root_lowering,
    _smoothly_subdivide_motion_steps,
)
from mm_sonic.canonical_terrain_matcher import RegularGridHeightField
from mm_sonic.terrain_oracle.stitch import FrameProvenance, StitchedMotion


class MotionBricksContactInbetweeningTest(unittest.TestCase):
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
