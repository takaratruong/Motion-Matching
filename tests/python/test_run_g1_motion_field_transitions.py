import importlib.util
import unittest
from pathlib import Path

import numpy as np
from mm_sonic.joints import ContractError

_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_motion_field_transitions.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_motion_field_transitions", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _yaw_quaternion(degrees):
    radians = np.deg2rad(degrees)
    return np.array(
        (np.cos(0.5 * radians), 0.0, 0.0, np.sin(0.5 * radians)),
        dtype=np.float64,
    )


def _route(frames, *, x_start=0.0, yaw_start=0.0):
    joints = np.zeros((frames, 29), dtype=np.float64)
    joints[:, 0] = np.linspace(0.0, 0.4, frames)
    roots = np.zeros((frames, 3), dtype=np.float64)
    roots[:, 0] = x_start + np.linspace(0.0, 0.4, frames)
    roots[:, 2] = 0.8
    quaternions = np.stack(
        [_yaw_quaternion(yaw_start) for _ in range(frames)]
    )
    support = np.tile((True, False), (frames, 1))
    return {
        "joint_position": joints,
        "root_position_world": roots,
        "root_orientation_world_wxyz": quaternions,
        "source_support_mask": support,
    }


class MotionFieldTransitionRunnerTests(unittest.TestCase):
    def test_rigid_footprint_relocation_preserves_authentic_motion(self):
        route = _route(5)

        relocated = _MODULE.translate_route_xy(
            route=route,
            translation_xy_m=(0.008, 0.024),
        )

        np.testing.assert_array_equal(
            relocated["joint_position"], route["joint_position"]
        )
        np.testing.assert_array_equal(
            relocated["root_orientation_world_wxyz"],
            route["root_orientation_world_wxyz"],
        )
        np.testing.assert_array_equal(
            relocated["source_support_mask"], route["source_support_mask"]
        )
        np.testing.assert_allclose(
            relocated["root_position_world"][:, :2],
            route["root_position_world"][:, :2] + (0.008, 0.024),
        )
        np.testing.assert_array_equal(
            relocated["root_position_world"][:, 2],
            route["root_position_world"][:, 2],
        )

    def test_footprint_relocation_occurs_only_through_support_transfer(self):
        route = _route(8)
        route["source_support_mask"][:3] = (False, True)
        route["source_support_mask"][3:5] = (False, False)
        route["source_support_mask"][5:] = (True, False)

        relocated = _MODULE.translate_route_xy_through_first_support_transfer(
            route=route,
            translation_xy_m=(0.008, 0.024),
        )
        correction = (
            relocated["root_position_world"] - route["root_position_world"]
        )

        np.testing.assert_allclose(correction[:3], 0.0)
        np.testing.assert_allclose(
            correction[5:, :2], np.tile((0.008, 0.024), (3, 1))
        )
        np.testing.assert_allclose(correction[:, 2], 0.0)
        self.assertGreater(correction[4, 1], correction[3, 1])
        np.testing.assert_array_equal(
            relocated["joint_position"], route["joint_position"]
        )

    def test_minimum_root_xy_contact_correction_repairs_only_invalid_frames(self):
        soles = np.zeros((5, 2, 4, 3), dtype=np.float64)
        soles[..., 0] = 0.10
        soles[..., 2] = 0.01
        soles[2, 1, 2:, 0] = -0.001
        support = np.tile((False, True), (5, 1))
        support[3] = (False, False)

        def surface(points):
            xy = np.asarray(points)
            return np.where(xy[..., 0] >= 0.0, 0.0, -0.5)

        correction = _MODULE.minimum_root_xy_contact_correction_schedule(
            sole_position_matcher=soles,
            support_mask=support,
            matcher_to_scene_xy=lambda points: points,
            sample_scene_surface=surface,
            search_radius_m=0.004,
            search_step_m=0.002,
        )

        np.testing.assert_allclose(correction[[0, 1, 3, 4]], 0.0)
        np.testing.assert_allclose(correction[2], (0.002, 0.0))

    def test_minimum_root_xy_contact_correction_rejects_unrepairable_frame(self):
        soles = np.zeros((3, 2, 4, 3), dtype=np.float64)
        soles[..., 2] = 0.50
        support = np.tile((True, False), (3, 1))

        with self.assertRaisesRegex(ContractError, "no local contact correction"):
            _MODULE.minimum_root_xy_contact_correction_schedule(
                sole_position_matcher=soles,
                support_mask=support,
                matcher_to_scene_xy=lambda points: points,
                sample_scene_surface=lambda points: np.zeros(
                    np.asarray(points).shape[:-1]
                ),
                search_radius_m=0.004,
                search_step_m=0.002,
            )

    def test_apply_root_xy_correction_preserves_pose_and_exact_endpoints(self):
        route = _route(5)
        correction = np.array(
            ((0.0, 0.0), (0.002, 0.0), (0.002, -0.001), (0.0, 0.0), (0.0, 0.0))
        )

        repaired = _MODULE.apply_root_xy_correction_schedule(
            route=route,
            correction_xy_m=correction,
        )

        np.testing.assert_array_equal(
            repaired["joint_position"], route["joint_position"]
        )
        np.testing.assert_array_equal(
            repaired["root_orientation_world_wxyz"],
            route["root_orientation_world_wxyz"],
        )
        np.testing.assert_allclose(
            repaired["root_position_world"][:, :2],
            route["root_position_world"][:, :2] + correction,
        )
        np.testing.assert_allclose(
            repaired["root_position_world"][[0, -1]],
            route["root_position_world"][[0, -1]],
        )

    def test_minimum_root_height_clearance_lift_repairs_only_collision(self):
        clearance = np.full((5, 2, 4), 0.005, dtype=np.float64)
        clearance[2, 1, 0] = -0.030
        support = np.tile((False, True), (5, 1))

        correction = _MODULE.minimum_root_z_clearance_correction_schedule(
            sole_clearance_m=clearance,
            support_mask=support,
            maximum_lift_m=0.02,
            search_step_m=0.001,
        )

        np.testing.assert_allclose(correction, (0.0, 0.0, 0.005, 0.0, 0.0))

    def test_apply_root_height_correction_preserves_pose_and_boundaries(self):
        route = _route(5)
        correction = np.asarray((0.0, 0.002, 0.006, 0.002, 0.0))

        repaired = _MODULE.apply_root_z_correction_schedule(
            route=route, correction_z_m=correction
        )

        np.testing.assert_array_equal(
            repaired["joint_position"], route["joint_position"]
        )
        np.testing.assert_allclose(
            repaired["root_position_world"][:, 2],
            route["root_position_world"][:, 2] + correction,
        )
        np.testing.assert_allclose(
            repaired["root_position_world"][[0, -1]],
            route["root_position_world"][[0, -1]],
        )

    def test_exact_successor_edge_removes_duplicate_boundary_frames(self):
        incoming = _route(5)
        connector = _route(4, x_start=0.4)
        connector["joint_position"][0] = incoming["joint_position"][-1]
        connector["root_position_world"][0] = incoming["root_position_world"][-1]
        successor = _route(6, x_start=0.8)
        successor["joint_position"][0] = connector["joint_position"][-1]
        successor["root_position_world"][0] = connector["root_position_world"][-1]

        assembled = _MODULE.assemble_exact_successor_edge(
            incoming_route=incoming,
            connector=connector,
            successor=successor,
        )

        self.assertEqual(len(assembled["joint_position"]), 5 + 3 + 5)
        np.testing.assert_allclose(
            assembled["root_position_world"][4],
            incoming["root_position_world"][-1],
        )
        np.testing.assert_allclose(
            assembled["root_position_world"][7],
            connector["root_position_world"][-1],
        )

    def test_exact_successor_accepts_equivalent_nearly_normalized_quaternions(self):
        incoming = _route(5)
        connector = _route(4, x_start=0.4)
        connector["joint_position"][0] = incoming["joint_position"][-1]
        connector["root_position_world"][0] = incoming["root_position_world"][-1]
        connector["root_orientation_world_wxyz"][0] = (
            -incoming["root_orientation_world_wxyz"][-1] * 0.99999
        )
        successor = _route(6, x_start=0.8)
        successor["joint_position"][0] = connector["joint_position"][-1]
        successor["root_position_world"][0] = connector["root_position_world"][-1]
        successor["root_orientation_world_wxyz"][0] *= 0.99999

        assembled = _MODULE.assemble_exact_successor_edge(
            incoming_route=incoming,
            connector=connector,
            successor=successor,
        )

        self.assertEqual(len(assembled["joint_position"]), 13)

    def test_sagittal_reflection_is_an_exact_route_involution(self):
        route = _route(7)
        route["root_position_world"][:, 1] = np.linspace(-0.2, 0.4, 7)
        route["root_orientation_world_wxyz"] = np.stack(
            [_yaw_quaternion(value) for value in np.linspace(5.0, -40.0, 7)]
        )
        route["source_support_mask"][:3] = (True, False)
        route["source_support_mask"][3:] = (False, True)

        mirrored = _MODULE.reflect_route_across_heading_axis(
            route=route, axis_y_m=0.1
        )
        restored = _MODULE.reflect_route_across_heading_axis(
            route=mirrored, axis_y_m=0.1
        )

        for name in (
            "joint_position",
            "root_position_world",
            "root_orientation_world_wxyz",
        ):
            np.testing.assert_allclose(restored[name], route[name], atol=1.0e-12)
        np.testing.assert_array_equal(
            restored["source_support_mask"], route["source_support_mask"]
        )
        np.testing.assert_allclose(
            mirrored["root_position_world"][:, 1],
            0.2 - route["root_position_world"][:, 1],
        )
        np.testing.assert_array_equal(
            mirrored["source_support_mask"],
            route["source_support_mask"][:, ::-1],
        )

    def test_planar_reflection_supports_rotated_symmetry_axis(self):
        route = _route(4)
        route["root_position_world"][:, :2] = (1.0, 0.0)

        reflected = _MODULE.reflect_route_across_planar_axis(
            route=route,
            axis_origin_xy=(0.0, 0.0),
            axis_heading_rad=np.pi / 4.0,
        )

        np.testing.assert_allclose(
            reflected["root_position_world"][:, :2],
            np.tile((0.0, 1.0), (4, 1)),
            atol=1.0e-12,
        )

    def test_stance_anchored_prefix_blend_preserves_contact_and_suffix(self):
        route = _route(7)
        route["joint_position"][:, 0] = (0.0, 0.8, -0.4, 0.7, 0.4, 0.5, 0.6)
        route["root_position_world"][:, 0] = (0.0, 0.2, 0.1, -0.1, -0.16, 0.2, 0.3)
        original_suffix = route["root_position_world"][4:].copy()

        class FakeFeet:
            @staticmethod
            def foot_positions(joints, roots, quaternions):
                feet = np.repeat(np.asarray(roots)[:, None, :], 2, axis=1)
                feet[:, :, 0] += np.asarray(joints)[:, :1] ** 2
                return feet

        repaired = _MODULE.stance_anchored_prefix_blend(
            route=route,
            stance_foot=0,
            blend_stop_frame=4,
            foot_kinematics=FakeFeet(),
        )
        feet = FakeFeet.foot_positions(
            repaired["joint_position"],
            repaired["root_position_world"],
            repaired["root_orientation_world_wxyz"],
        )

        np.testing.assert_allclose(
            feet[:5, 0], np.tile(feet[0, 0], (5, 1)), atol=1.0e-12
        )
        np.testing.assert_allclose(
            repaired["root_position_world"][4:], original_suffix
        )
        np.testing.assert_array_equal(
            repaired["source_support_mask"], route["source_support_mask"]
        )

    def test_stance_anchored_terminal_blend_reaches_exact_target(self):
        route = _route(7)
        route["joint_position"][:, 0] = 0.0
        route["root_position_world"][:, 0] = 0.0

        class FakeFeet:
            @staticmethod
            def foot_positions(joints, roots, quaternions):
                feet = np.repeat(np.asarray(roots)[:, None, :], 2, axis=1)
                feet[:, :, 0] += np.asarray(joints)[:, :1] ** 2
                return feet

        target_joint = route["joint_position"][-1].copy()
        target_joint[0] = 0.4
        target_root = route["root_position_world"][-1].copy()
        target_root[0] = -0.16
        repaired = _MODULE.stance_anchored_terminal_blend(
            route=route,
            target_joint_position=target_joint,
            target_root_position_world=target_root,
            target_root_orientation_world_wxyz=route[
                "root_orientation_world_wxyz"
            ][-1],
            stance_foot=0,
            blend_start_frame=2,
            foot_kinematics=FakeFeet(),
        )
        feet = FakeFeet.foot_positions(
            repaired["joint_position"],
            repaired["root_position_world"],
            repaired["root_orientation_world_wxyz"],
        )

        np.testing.assert_allclose(repaired["joint_position"][-1], target_joint)
        np.testing.assert_allclose(repaired["root_position_world"][-1], target_root)
        np.testing.assert_allclose(
            feet[2:, 0], np.tile(feet[-1, 0], (5, 1)), atol=1.0e-12
        )

    def test_terminal_blend_accepts_sub_gate_contact_projection_residual(self):
        route = _route(7)
        route["joint_position"][:, 0] = 0.0
        route["root_position_world"][:, 0] = 0.0

        class FakeFeet:
            @staticmethod
            def foot_positions(joints, roots, quaternions):
                feet = np.repeat(np.asarray(roots)[:, None, :], 2, axis=1)
                feet[:, :, 0] += np.asarray(joints)[:, :1] ** 2
                return feet

        target_joint = route["joint_position"][-1].copy()
        target_joint[0] = 0.4
        target_root = route["root_position_world"][-1].copy()
        target_root[0] = -0.1575

        repaired = _MODULE.stance_anchored_terminal_blend(
            route=route,
            target_joint_position=target_joint,
            target_root_position_world=target_root,
            target_root_orientation_world_wxyz=route[
                "root_orientation_world_wxyz"
            ][-1],
            stance_foot=0,
            blend_start_frame=2,
            foot_kinematics=FakeFeet(),
        )

        np.testing.assert_allclose(repaired["joint_position"][-1], target_joint)
        np.testing.assert_allclose(repaired["root_position_world"][-1], target_root)

    def test_stance_boundary_connector_is_exact_and_locks_support_foot(self):
        incoming = _route(3)
        target = _route(3, x_start=0.2)
        target["joint_position"][0, 0] = 0.0
        target["root_position_world"][0, 0] = (
            incoming["root_position_world"][-1, 0] + 0.16
        )

        class FakeFeet:
            @staticmethod
            def foot_positions(joints, roots, quaternions):
                feet = np.repeat(np.asarray(roots)[:, None, :], 2, axis=1)
                feet[:, :, 0] += np.asarray(joints)[:, :1] ** 2
                return feet

        connector = _MODULE.stance_anchored_boundary_connector(
            incoming_joint_position=incoming["joint_position"][-1],
            incoming_root_position_world=incoming["root_position_world"][-1],
            incoming_root_orientation_world_wxyz=incoming[
                "root_orientation_world_wxyz"
            ][-1],
            target_joint_position=target["joint_position"][0],
            target_root_position_world=target["root_position_world"][0],
            target_root_orientation_world_wxyz=target[
                "root_orientation_world_wxyz"
            ][0],
            stance_foot=0,
            frame_count=12,
            foot_kinematics=FakeFeet(),
        )
        feet = FakeFeet.foot_positions(
            connector["joint_position"],
            connector["root_position_world"],
            connector["root_orientation_world_wxyz"],
        )

        np.testing.assert_allclose(
            connector["joint_position"][[0, -1]],
            (incoming["joint_position"][-1], target["joint_position"][0]),
        )
        np.testing.assert_allclose(feet[:, 0], np.tile(feet[0, 0], (12, 1)))
        np.testing.assert_array_equal(
            connector["source_support_mask"],
            np.tile((True, False), (12, 1)),
        )

    def test_authentic_placement_preserves_every_source_joint_frame(self):
        source = _route(9)
        source["joint_position"][:, 3] = np.linspace(-0.4, 0.7, 9)
        source["root_position_world"][:, 1] = np.linspace(0.0, -0.2, 9)
        source["root_orientation_world_wxyz"] = np.stack(
            [_yaw_quaternion(value) for value in np.linspace(10.0, -35.0, 9)]
        )
        original_joints = source["joint_position"].copy()
        original_support = source["source_support_mask"].copy()

        placed = _MODULE.place_authentic_transition(
            source_route=source,
            incoming_root_position_world=np.array((1.0, 2.0, 0.9)),
            incoming_root_orientation_world_wxyz=_yaw_quaternion(25.0),
            outgoing_root_position_world=np.array((1.25, 1.75, 0.88)),
        )

        np.testing.assert_array_equal(placed["joint_position"], original_joints)
        np.testing.assert_array_equal(
            placed["source_support_mask"], original_support
        )
        np.testing.assert_allclose(
            placed["root_position_world"][[0, -1]],
            ((1.0, 2.0, 0.9), (1.25, 1.75, 0.88)),
        )
        self.assertAlmostEqual(
            float(_MODULE._yaw_wxyz(placed["root_orientation_world_wxyz"][0])),
            np.deg2rad(25.0),
        )

    def test_aligned_clearance_samples_scene_coordinates_not_matcher_coordinates(self):
        sole = np.zeros((2, 2, 3, 3), dtype=np.float64)
        sole[..., 0] = 1.0
        sole[..., 1] = 2.0
        sole[..., 2] = 0.8
        seen = []

        def matcher_to_scene(points):
            return np.asarray(points) + np.array((10.0, -4.0))

        def sample_scene(points):
            scene = np.asarray(points)
            seen.append(scene.copy())
            return 0.1 * scene[..., 0]

        clearance = _MODULE.aligned_terrain_clearance(
            sole_position_matcher=sole,
            matcher_to_scene_xy=matcher_to_scene,
            sample_scene_surface=sample_scene,
        )

        np.testing.assert_allclose(seen[0], sole[..., :2] + (10.0, -4.0))
        np.testing.assert_allclose(clearance, -0.3)

    def test_contact_alignment_translates_root_without_editing_authentic_pose(self):
        placed = _route(7)
        original_joints = placed["joint_position"].copy()
        original_orientation = placed["root_orientation_world_wxyz"].copy()

        aligned = _MODULE.align_authentic_transition_contact(
            placed_route=placed,
            placed_start_foot_position_world=np.array((1.0, 2.0, 0.3)),
            target_start_foot_position_world=np.array((1.04, 1.97, 0.3)),
        )

        np.testing.assert_array_equal(aligned["joint_position"], original_joints)
        np.testing.assert_array_equal(
            aligned["root_orientation_world_wxyz"], original_orientation
        )
        np.testing.assert_allclose(
            aligned["root_position_world"] - placed["root_position_world"],
            np.tile((0.04, -0.03, 0.0), (7, 1)),
        )
        np.testing.assert_array_equal(
            aligned["source_support_mask"], placed["source_support_mask"]
        )

    def test_contact_transfer_alignment_fades_only_between_support_feet(self):
        placed = _route(8)
        placed["source_support_mask"][:3] = (True, False)
        placed["source_support_mask"][3:5] = (False, False)
        placed["source_support_mask"][5:] = (False, True)
        original_joints = placed["joint_position"].copy()

        aligned = _MODULE.align_authentic_transition_through_support_transfer(
            placed_route=placed,
            placed_start_foot_position_world=np.array((1.0, 2.0, 0.3)),
            target_start_foot_position_world=np.array((1.04, 1.97, 0.3)),
        )

        correction = (
            aligned["root_position_world"] - placed["root_position_world"]
        )
        np.testing.assert_allclose(
            correction[:3], np.tile((0.04, -0.03, 0.0), (3, 1))
        )
        np.testing.assert_allclose(correction[5:], 0.0, atol=1.0e-12)
        self.assertGreater(correction[3, 0], correction[4, 0])
        self.assertGreater(correction[4, 0], correction[5, 0])
        np.testing.assert_array_equal(aligned["joint_position"], original_joints)

    def test_root_height_schedule_reaches_target_at_opposite_landing(self):
        support = np.zeros((9, 2), dtype=np.bool_)
        support[:3] = (False, True)
        support[3:5] = (False, False)
        support[5:] = (True, False)
        roots = np.zeros((9, 3), dtype=np.float64)
        roots[:, 0] = np.linspace(0.0, 0.4, 9)
        roots[:, 2] = 1.25 + 0.01 * np.sin(np.linspace(0.0, np.pi, 9))

        scheduled = _MODULE.support_transfer_root_height_schedule(
            root_position_world=roots,
            support_mask=support,
            target_root_height_world=1.08,
        )

        np.testing.assert_allclose(scheduled[:, :2], roots[:, :2])
        self.assertAlmostEqual(float(scheduled[0, 2]), float(roots[0, 2]))
        self.assertAlmostEqual(float(scheduled[5, 2]), 1.08)
        np.testing.assert_allclose(scheduled[5:, 2], 1.08)
        self.assertTrue(np.all(np.diff(scheduled[:6, 2]) <= 1.0e-12))

    def test_return_support_height_schedule_holds_first_landing_then_descends(self):
        support = np.zeros((12, 2), dtype=np.bool_)
        support[:3] = (False, True)
        support[3:5] = (False, False)
        support[5:8] = (True, False)
        support[8:10] = (False, False)
        support[10:] = (False, True)
        roots = np.zeros((12, 3), dtype=np.float64)
        roots[:, 2] = 1.25

        scheduled = _MODULE.return_support_root_height_schedule(
            root_position_world=roots,
            support_mask=support,
            target_root_height_world=1.08,
        )

        np.testing.assert_allclose(scheduled[:6, 2], 1.25)
        self.assertGreater(float(scheduled[7, 2]), float(scheduled[9, 2]))
        self.assertAlmostEqual(float(scheduled[10, 2]), 1.08)
        np.testing.assert_allclose(scheduled[10:, 2], 1.08)

    def test_landing_root_correction_peaks_at_opposite_and_returns_to_zero(self):
        support = np.zeros((12, 2), dtype=np.bool_)
        support[:3] = (False, True)
        support[3:5] = (True, True)
        support[5:8] = (True, False)
        support[8:10] = (True, True)
        support[10:] = (False, True)

        correction = _MODULE.landing_root_correction_schedule(
            support_mask=support,
            opposite_landing_correction_m=0.13,
        )

        self.assertAlmostEqual(float(correction[0]), 0.0)
        self.assertAlmostEqual(float(correction[5]), 0.13)
        np.testing.assert_allclose(correction[5:8], 0.13)
        self.assertAlmostEqual(float(correction[10]), 0.0)
        np.testing.assert_allclose(correction[10:], 0.0)
        self.assertTrue(np.all(np.diff(correction[:6]) >= -1.0e-12))
        self.assertTrue(np.all(np.diff(correction[5:11]) <= 1.0e-12))

    def test_endpoint_correction_preserves_turn_interior_and_exact_boundaries(self):
        source = _route(9)
        source["root_position_world"][:, 1] = np.linspace(0.0, -0.2, 9)
        source["root_orientation_world_wxyz"] = np.stack(
            [_yaw_quaternion(value) for value in np.linspace(0.0, -45.0, 9)]
        )
        source["source_support_mask"][:4] = (True, False)
        source["source_support_mask"][4:] = (False, True)
        incoming_joint = source["joint_position"][0].copy()
        incoming_root = np.array((1.0, 2.0, 0.9))
        incoming_quaternion = _yaw_quaternion(15.0)
        outgoing_joint = source["joint_position"][-1].copy()
        outgoing_root = np.array((1.25, 1.75, 0.9))
        outgoing_quaternion = _yaw_quaternion(-30.0)

        connector = _MODULE.endpoint_corrected_transition(
            source_route=source,
            incoming_joint_position=incoming_joint,
            incoming_root_position_world=incoming_root,
            incoming_root_orientation_world_wxyz=incoming_quaternion,
            outgoing_joint_position=outgoing_joint,
            outgoing_root_position_world=outgoing_root,
            outgoing_root_orientation_world_wxyz=outgoing_quaternion,
            endpoint_blend_frames=2,
        )

        np.testing.assert_allclose(connector["joint_position"][0], incoming_joint)
        np.testing.assert_allclose(connector["joint_position"][-1], outgoing_joint)
        np.testing.assert_allclose(connector["root_position_world"][0], incoming_root)
        np.testing.assert_allclose(connector["root_position_world"][-1], outgoing_root)
        np.testing.assert_allclose(
            connector["root_orientation_world_wxyz"][0], incoming_quaternion
        )
        np.testing.assert_allclose(
            connector["root_orientation_world_wxyz"][-1], outgoing_quaternion
        )
        np.testing.assert_allclose(
            connector["joint_position"][4], source["joint_position"][4]
        )
        np.testing.assert_array_equal(
            connector["source_support_mask"], source["source_support_mask"]
        )

    def test_assembly_preserves_routes_outside_the_local_transition(self):
        incoming = _route(8)
        outgoing = _route(7, x_start=0.35, yaw_start=-45.0)
        incoming_frame = 5
        outgoing_frame = 2
        outgoing["source_support_mask"][:] = (False, True)
        source = _route(9)
        source["source_support_mask"][:4] = (True, False)
        source["source_support_mask"][4:] = (False, True)
        connector = _MODULE.endpoint_corrected_transition(
            source_route=source,
            incoming_joint_position=incoming["joint_position"][incoming_frame],
            incoming_root_position_world=incoming["root_position_world"][incoming_frame],
            incoming_root_orientation_world_wxyz=incoming[
                "root_orientation_world_wxyz"
            ][incoming_frame],
            outgoing_joint_position=outgoing["joint_position"][outgoing_frame],
            outgoing_root_position_world=outgoing["root_position_world"][outgoing_frame],
            outgoing_root_orientation_world_wxyz=outgoing[
                "root_orientation_world_wxyz"
            ][outgoing_frame],
            endpoint_blend_frames=2,
        )

        assembled = _MODULE.assemble_transition_routes(
            incoming_route=incoming,
            outgoing_route=outgoing,
            incoming_frame=incoming_frame,
            outgoing_frame=outgoing_frame,
            connector=connector,
        )

        np.testing.assert_array_equal(
            assembled["joint_position"][:incoming_frame],
            incoming["joint_position"][:incoming_frame],
        )
        np.testing.assert_array_equal(
            assembled["joint_position"][
                -(len(outgoing["joint_position"]) - outgoing_frame - 1) :
            ],
            outgoing["joint_position"][outgoing_frame + 1 :],
        )
        self.assertEqual(
            len(assembled["joint_position"]),
            incoming_frame
            + len(connector["joint_position"])
            + len(outgoing["joint_position"])
            - outgoing_frame
            - 1,
        )

    def test_assembly_rejects_support_phase_invention(self):
        incoming = _route(8)
        outgoing = _route(7, yaw_start=-45.0)
        outgoing["source_support_mask"][:] = (False, True)
        connector = _route(9)
        connector["source_support_mask"][:] = (False, True)

        with self.assertRaisesRegex(ContractError, "support"):
            _MODULE.assemble_transition_routes(
                incoming_route=incoming,
                outgoing_route=outgoing,
                incoming_frame=5,
                outgoing_frame=2,
                connector=connector,
            )

    def test_assembly_accepts_normalization_tolerant_identical_orientations(self):
        incoming = _route(8)
        outgoing = _route(7, x_start=0.35, yaw_start=-45.0)
        outgoing["source_support_mask"][:] = (False, True)
        source = _route(9)
        source["source_support_mask"][:4] = (True, False)
        source["source_support_mask"][4:] = (False, True)
        connector = _MODULE.endpoint_corrected_transition(
            source_route=source,
            incoming_joint_position=incoming["joint_position"][5],
            incoming_root_position_world=incoming["root_position_world"][5],
            incoming_root_orientation_world_wxyz=(
                incoming["root_orientation_world_wxyz"][5] * 0.99999
            ),
            outgoing_joint_position=outgoing["joint_position"][2],
            outgoing_root_position_world=outgoing["root_position_world"][2],
            outgoing_root_orientation_world_wxyz=(
                outgoing["root_orientation_world_wxyz"][2] * 0.99999
            ),
            endpoint_blend_frames=2,
        )
        incoming["root_orientation_world_wxyz"][5] *= 0.99999
        outgoing["root_orientation_world_wxyz"][2] *= 0.99999

        assembled = _MODULE.assemble_transition_routes(
            incoming_route=incoming,
            outgoing_route=outgoing,
            incoming_frame=5,
            outgoing_frame=2,
            connector=connector,
        )

        self.assertGreater(len(assembled["joint_position"]), 9)

    def test_candidate_gate_rejects_contact_heading_and_continuity_failures(self):
        clean = {
            "maximum_joint_step_rad": 0.20,
            "maximum_root_step_m": 0.03,
            "minimum_sole_clearance_m": -0.02,
            "maximum_stance_contact_error_m": 0.02,
            "maximum_stance_horizontal_step_m": 0.005,
            "minimum_supported_sole_points": 3,
            "heading_change_error_degrees": 4.0,
        }
        self.assertEqual(_MODULE.transition_candidate_rejections(clean), ())

        broken = dict(
            clean,
            maximum_joint_step_rad=0.40,
            minimum_sole_clearance_m=-0.05,
            maximum_stance_contact_error_m=0.08,
            maximum_stance_horizontal_step_m=0.03,
            heading_change_error_degrees=20.0,
        )
        self.assertEqual(
            set(_MODULE.transition_candidate_rejections(broken)),
            {
                "joint_discontinuity",
                "terrain_penetration",
                "stance_contact_failure",
                "stance_foot_slide",
                "heading_change_mismatch",
            },
        )

    def test_entry_gate_rejects_terrain_valid_turn_that_needs_large_root_move(self):
        self.assertEqual(
            _MODULE.transition_entry_candidate_rejections(
                {
                    "required_contact_root_translation_m": 0.104,
                    "starting_pose_max_joint_delta_rad": 0.59,
                    "support_compatible": True,
                }
            ),
            (),
        )
        self.assertEqual(
            _MODULE.transition_entry_candidate_rejections(
                {
                    "required_contact_root_translation_m": 0.170,
                    "starting_pose_max_joint_delta_rad": 0.59,
                    "support_compatible": True,
                }
            ),
            ("entry_root_relocation",),
        )

    def test_finite_radius_edge_allows_bounded_connector_to_turn_join(self):
        incoming = _route(9)
        incoming_frame = 6
        connector = _route(5)
        for name in (
            "joint_position",
            "root_position_world",
            "root_orientation_world_wxyz",
            "source_support_mask",
        ):
            connector[name][0] = incoming[name][incoming_frame]
        connector["joint_position"][0, 0] += 3.0e-8
        connector["root_orientation_world_wxyz"][0] *= 0.99999
        turn = _route(7, x_start=float(connector["root_position_world"][-1, 0]))
        turn["joint_position"][0] = connector["joint_position"][-1] + 0.12
        turn["root_position_world"][0] = (
            connector["root_position_world"][-1] + (0.003, -0.002, 0.0)
        )

        edge = _MODULE.assemble_finite_radius_transition_edge(
            incoming_route=incoming,
            incoming_frame=incoming_frame,
            connector=connector,
            authentic_turn=turn,
            preview_start_frame=2,
        )

        np.testing.assert_array_equal(
            edge["joint_position"][: incoming_frame - 2],
            incoming["joint_position"][2:incoming_frame],
        )
        self.assertEqual(
            len(edge["joint_position"]),
            incoming_frame - 2 + len(connector["joint_position"]) + 7,
        )
        np.testing.assert_array_equal(
            edge["joint_position"][-7:], turn["joint_position"]
        )

    def test_finite_radius_edge_allows_final_incoming_frame_boundary(self):
        incoming = _route(9)
        incoming_frame = 8
        connector = _route(5)
        for name in (
            "joint_position",
            "root_position_world",
            "root_orientation_world_wxyz",
            "source_support_mask",
        ):
            connector[name][0] = incoming[name][incoming_frame]
        turn = _route(7, x_start=float(connector["root_position_world"][-1, 0]))
        turn["joint_position"][0] = connector["joint_position"][-1]
        turn["root_position_world"][0] = connector["root_position_world"][-1]
        turn["root_orientation_world_wxyz"][0] = connector[
            "root_orientation_world_wxyz"
        ][-1]

        edge = _MODULE.assemble_finite_radius_transition_edge(
            incoming_route=incoming,
            incoming_frame=incoming_frame,
            connector=connector,
            authentic_turn=turn,
            preview_start_frame=6,
        )

        np.testing.assert_array_equal(
            edge["joint_position"][:2], incoming["joint_position"][6:8]
        )

    def test_finite_radius_edge_rejects_excessive_connector_to_turn_jump(self):
        incoming = _route(9)
        connector = _route(5)
        for name in (
            "joint_position",
            "root_position_world",
            "root_orientation_world_wxyz",
            "source_support_mask",
        ):
            connector[name][0] = incoming[name][6]
        turn = _route(7, x_start=float(connector["root_position_world"][-1, 0]))
        turn["joint_position"][0] = connector["joint_position"][-1] + 0.30

        with self.assertRaisesRegex(ContractError, "connector-to-turn"):
            _MODULE.assemble_finite_radius_transition_edge(
                incoming_route=incoming,
                incoming_frame=6,
                connector=connector,
                authentic_turn=turn,
                preview_start_frame=2,
            )

    def test_terrain_contact_metrics_do_not_compare_walking_feet_to_final_pose(self):
        clearance = np.full((4, 2, 4), 0.20, dtype=np.float64)
        support = np.array(
            ((True, False), (True, False), (False, True), (False, True)),
            dtype=np.bool_,
        )
        clearance[0, 0] = (-0.020, -0.010, 0.005, 0.040)
        clearance[1, 0] = (-0.015, -0.005, 0.010, 0.050)
        clearance[2, 1] = (-0.010, 0.000, 0.020, 0.060)
        clearance[3, 1] = (-0.005, 0.005, 0.025, 0.070)

        metrics = _MODULE.terrain_transition_contact_metrics(
            sole_clearance_m=clearance,
            support_mask=support,
        )

        self.assertAlmostEqual(metrics["minimum_sole_clearance_m"], -0.020)
        self.assertEqual(metrics["minimum_supported_sole_points"], 3)
        self.assertAlmostEqual(metrics["maximum_stance_contact_error_m"], 0.005)


if __name__ == "__main__":
    unittest.main()
