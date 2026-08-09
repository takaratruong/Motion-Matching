import math
from types import SimpleNamespace

import numpy as np

from mm_sonic.build_bones_side_on_stair_pilots import (
    _adaptively_subdivide_motion_steps,
    _attenuate_upper_limb_motion,
    _conform_stance_sole_targets_to_mesh,
    _flat_support_mesh,
    _lateral_stance_level_schedule,
    _refit_motion_to_sole_targets,
    _repairable_warp_accepted,
    _retime_motion_and_extras,
    _select_route_length_segment,
)
from mm_sonic.build_stairs500_omnidirectional_pilots import (
    _fill_short_stance_gaps,
    _remove_short_stance_runs,
)
from mm_sonic.terrain_oracle.stitch import FrameProvenance, StitchedMotion


def test_short_contact_blips_are_not_treated_as_stance() -> None:
    stance = np.zeros((12, 2), dtype=bool)
    stance[1:3, 0] = True
    stance[5:10, 0] = True
    stance[2:6, 1] = True

    filtered = _remove_short_stance_runs(stance, minimum_run_frames=4)

    assert not np.any(filtered[1:3, 0])
    assert np.all(filtered[5:10, 0])
    assert np.all(filtered[2:6, 1])


def test_upper_limb_attenuation_reduces_swing_without_changing_legs() -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 80)
    joints = np.stack(
        (
            0.4 * np.sin(phase),
            0.8 * np.sin(phase),
            0.2 * np.cos(phase),
        ),
        axis=1,
    )

    reduced, diagnostics = _attenuate_upper_limb_motion(
        joints,
        ("left_shoulder_pitch_joint", "right_wrist_yaw_joint", "left_knee_joint"),
        fps=50.0,
        arm_swing_scale=0.35,
        wrist_swing_scale=0.10,
        smoothing_sigma_frames=2.0,
    )

    np.testing.assert_allclose(reduced[:, 2], joints[:, 2])
    assert np.ptp(reduced[:, 0]) < 0.36 * np.ptp(joints[:, 0])
    assert np.ptp(reduced[:, 1]) < 0.11 * np.ptp(joints[:, 1])
    assert (
        diagnostics["after"]["maximum_arm_velocity_rad_s"]
        < diagnostics["before"]["maximum_arm_velocity_rad_s"]
    )


def test_sole_refit_keeps_valid_authored_pose_over_bad_ik_branch() -> None:
    class BranchingAdapter:
        @staticmethod
        def sole_positions_for_pose(*, joints, **_kwargs):
            point = np.asarray((float(joints[0]), 0.0, 0.0), dtype=np.float64)
            sole = np.tile(point, (4, 1))
            return sole.copy(), sole.copy()

        @staticmethod
        def adapt_to_targets(**_kwargs):
            return np.asarray((1.0,), dtype=np.float64), 1.0, 0.95

    frame_count = 6
    motion = StitchedMotion(
        fps=50.0,
        root_position_world=np.zeros((frame_count, 3), dtype=np.float32),
        root_quaternion_world_wxyz=np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        ),
        joint_position=np.zeros((frame_count, 1), dtype=np.float32),
        provenance=tuple(
            FrameProvenance(-1, frame, "test") for frame in range(frame_count)
        ),
        seam_indices=(),
    )
    targets = np.zeros((frame_count, 2, 4, 3), dtype=np.float32)
    targets[..., 0] = 0.05

    refitted, _extras, metrics = _refit_motion_to_sole_targets(
        motion,
        {"target_sole_points_world": targets},
        adapter=BranchingAdapter(),
    )

    np.testing.assert_allclose(refitted.joint_position, 0.0)
    assert metrics["maximum_sole_target_error_m"] < 0.051
    assert metrics["maximum_joint_step_rad"] == 0.0


def test_stance_only_refit_does_not_chase_free_swing_target() -> None:
    class TwoFootAdapter:
        @staticmethod
        def sole_positions_for_pose(*, joints, **_kwargs):
            left = np.tile(
                np.asarray((float(joints[0]), 0.0, 0.0)), (4, 1)
            )
            right = np.tile(
                np.asarray((float(joints[1]), 0.0, 0.0)), (4, 1)
            )
            return left, right

        @staticmethod
        def adapt_to_targets(*, sole_targets_world, **_kwargs):
            joints = np.asarray(
                (
                    sole_targets_world[0, 0, 0],
                    sole_targets_world[1, 0, 0],
                )
            )
            return joints, float(np.max(np.abs(joints))), 0.0

    frame_count = 4
    motion = StitchedMotion(
        fps=50.0,
        root_position_world=np.zeros((frame_count, 3), dtype=np.float32),
        root_quaternion_world_wxyz=np.tile(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frame_count, 1),
        ),
        joint_position=np.zeros((frame_count, 2), dtype=np.float32),
        provenance=tuple(
            FrameProvenance(-1, frame, "test") for frame in range(frame_count)
        ),
        seam_indices=(),
    )
    targets = np.zeros((frame_count, 2, 4, 3), dtype=np.float32)
    targets[:, 0, :, 0] = 0.1
    targets[:, 1, :, 0] = 5.0
    stance = np.zeros((frame_count, 2), dtype=np.bool_)
    stance[:, 0] = True

    refitted, extras, _metrics = _refit_motion_to_sole_targets(
        motion,
        {
            "target_sole_points_world": targets,
            "authored_stance_mask": stance,
        },
        adapter=TwoFootAdapter(),
        stance_only=True,
    )

    np.testing.assert_allclose(refitted.joint_position[:, 0], 0.1)
    np.testing.assert_allclose(refitted.joint_position[:, 1], 0.0)
    np.testing.assert_allclose(
        extras["target_sole_points_world"][:, 1], 0.0
    )

def test_short_contact_dropout_is_merged_into_one_stance() -> None:
    stance = np.zeros((14, 2), dtype=bool)
    stance[2:7, 0] = True
    stance[9:13, 0] = True

    merged = _fill_short_stance_gaps(stance, maximum_gap_frames=2)

    assert np.all(merged[2:13, 0])
    assert not np.any(merged[:, 1])


def test_segment_selector_preserves_a_true_side_on_span() -> None:
    frame_count = 240
    root_xy = np.zeros((frame_count, 2), dtype=np.float64)
    root_xy[:, 0] = np.linspace(0.0, 2.4, frame_count)
    yaw = np.full(frame_count, math.pi / 2.0, dtype=np.float64)
    quaternion = np.zeros((frame_count, 4), dtype=np.float64)
    quaternion[:, 0] = np.cos(0.5 * yaw)
    quaternion[:, 3] = np.sin(0.5 * yaw)

    start, stop, facing_delta = _select_route_length_segment(
        root_xy,
        quaternion,
        required_distance_m=1.3,
        minimum_frames=120,
    )

    assert stop - start >= 120
    assert root_xy[stop - 1, 0] - root_xy[start, 0] >= 1.3
    assert abs(abs(math.degrees(facing_delta)) - 90.0) < 1.0e-6


def test_side_on_schedule_uses_lead_up_then_trailing_foot_catch() -> None:
    spans = [
        SimpleNamespace(foot_index=0, start_frame=0, stop_frame=10),
        SimpleNamespace(foot_index=1, start_frame=1, stop_frame=5),
        SimpleNamespace(foot_index=1, start_frame=12, stop_frame=17),
        SimpleNamespace(foot_index=0, start_frame=18, stop_frame=23),
        SimpleNamespace(foot_index=1, start_frame=25, stop_frame=30),
        SimpleNamespace(foot_index=0, start_frame=31, stop_frame=36),
    ]

    schedule = _lateral_stance_level_schedule(
        spans, leading_foot_index=1, level_count=3
    )

    assert list(schedule.values()) == [0, 0, 1, 1, 2, 2]


def test_side_on_schedule_does_not_skip_lead_foot_base_support() -> None:
    spans = [
        SimpleNamespace(foot_index=0, start_frame=0, stop_frame=8),
        SimpleNamespace(foot_index=1, start_frame=3, stop_frame=10),
        SimpleNamespace(foot_index=1, start_frame=14, stop_frame=20),
        SimpleNamespace(foot_index=0, start_frame=21, stop_frame=27),
    ]

    schedule = _lateral_stance_level_schedule(
        spans, leading_foot_index=1, level_count=2
    )

    assert schedule[(0, 0, 8)] == 0
    assert schedule[(1, 3, 10)] == 0
    assert schedule[(1, 14, 20)] == 1
    assert schedule[(0, 21, 27)] == 1


def test_side_on_schedule_orders_overlapping_touchdowns_by_start() -> None:
    spans = [
        SimpleNamespace(foot_index=0, start_frame=0, stop_frame=20),
        SimpleNamespace(foot_index=1, start_frame=1, stop_frame=5),
        SimpleNamespace(foot_index=0, start_frame=30, stop_frame=42),
        SimpleNamespace(foot_index=1, start_frame=32, stop_frame=50),
    ]

    schedule = _lateral_stance_level_schedule(
        spans, leading_foot_index=0, level_count=2
    )

    assert schedule[(0, 0, 20)] == 0
    assert schedule[(1, 1, 5)] == 0
    assert schedule[(0, 30, 42)] == 1
    assert schedule[(1, 32, 50)] == 1


def test_repairable_gate_allows_one_boundary_outlier_but_not_broad_error() -> None:
    stance = np.zeros((120, 2), dtype=bool)
    stance[:, 0] = True
    stance[::2, 1] = True
    error = np.full((120, 2), 0.002, dtype=np.float64)
    error[20, 0] = 0.105
    support = np.full((120, 2), 4, dtype=np.int16)
    support[20, 0] = 0
    diagnostics = SimpleNamespace(
        maximum_joint_correction_rad=1.1,
        maximum_joint_step_rad=0.2,
        maximum_root_translation_step_m=0.04,
        maximum_root_rotation_step_rad=0.02,
        maximum_root_acceleration_m_s2=42.0,
    )

    accepted, metrics = _repairable_warp_accepted(
        diagnostics,
        {
            "authored_stance_mask": stance,
            "per_frame_sole_target_error_by_foot_m": error,
            "target_stance_support_point_count": support,
        },
    )

    assert accepted
    assert metrics["unsupported_stance_frame_foot_count"] == 1

    error[stance] = 0.04
    rejected, _ = _repairable_warp_accepted(
        diagnostics,
        {
            "authored_stance_mask": stance,
            "per_frame_sole_target_error_by_foot_m": error,
            "target_stance_support_point_count": support,
        },
    )
    assert not rejected


def test_stance_sole_conformance_flattens_a_tilted_foot_on_a_tread() -> None:
    frame_count = 10
    stance = np.zeros((frame_count, 2), dtype=bool)
    stance[2:8, 0] = True
    targets = np.zeros((frame_count, 2, 4, 3), dtype=np.float64)
    footprint = np.asarray(
        ((-0.10, -0.04), (-0.10, 0.04), (0.10, -0.04), (0.10, 0.04))
    )
    targets[:, 0, :, :2] = footprint[None] + np.asarray((0.5, 0.0))
    targets[:, 0, :, 2] = 0.02 + 0.20 * footprint[:, 0]
    targets[:, 1, :, :2] = footprint[None] + np.asarray((0.5, 0.25))
    targets[:, 1, :, 2] = 0.02
    adapter = SimpleNamespace(
        sole_sphere_radii=lambda: (
            np.full(4, 0.02, dtype=np.float64),
            np.full(4, 0.02, dtype=np.float64),
        )
    )

    conformed, diagnostics = _conform_stance_sole_targets_to_mesh(
        {
            "authored_stance_mask": stance,
            "target_sole_points_world": targets,
        },
        adapter=adapter,
        target_mesh=_flat_support_mesh(
            np.asarray(((0.0, 0.0), (1.0, 0.0))), height_m=0.0
        ),
        ground_fallback_height_m=0.0,
    )

    core = conformed["target_sole_points_world"][3:7, 0, :, 2]
    np.testing.assert_allclose(core, 0.02, atol=1.0e-6)
    assert diagnostics["maximum_surface_alignment_deg"] > 10.0


def test_stance_sole_conformance_uses_flat_runway_outside_finite_mesh() -> None:
    stance = np.zeros((8, 2), dtype=bool)
    stance[1:7, 0] = True
    targets = np.zeros((8, 2, 4, 3), dtype=np.float64)
    targets[:, 0, :, :2] = np.asarray(
        ((20.0, -0.04), (20.0, 0.04), (20.2, -0.04), (20.2, 0.04))
    )
    targets[:, 0, :, 2] = 0.02
    adapter = SimpleNamespace(
        sole_sphere_radii=lambda: (
            np.full(4, 0.02, dtype=np.float64),
            np.full(4, 0.02, dtype=np.float64),
        )
    )

    conformed, diagnostics = _conform_stance_sole_targets_to_mesh(
        {
            "authored_stance_mask": stance,
            "target_sole_points_world": targets,
        },
        adapter=adapter,
        target_mesh=_flat_support_mesh(
            np.asarray(((0.0, 0.0), (1.0, 0.0))), height_m=0.0
        ),
        ground_fallback_height_m=0.0,
    )

    np.testing.assert_allclose(
        conformed["target_sole_points_world"][2:6, 0, :, 2],
        0.02,
        atol=1.0e-6,
    )
    assert diagnostics["ground_fallback_run_count"] == 1.0


def test_post_warp_retime_preserves_endpoints_and_reduces_steps() -> None:
    motion = StitchedMotion(
        fps=50.0,
        root_position_world=np.asarray(
            ((0.0, 0.0, 0.8), (0.1, 0.0, 0.9), (0.2, 0.0, 1.0)),
            dtype=np.float32,
        ),
        root_quaternion_world_wxyz=np.asarray(
            ((1.0, 0.0, 0.0, 0.0),) * 3, dtype=np.float32
        ),
        joint_position=np.asarray(((0.0,), (0.2,), (0.4,)), dtype=np.float32),
        provenance=tuple(
            FrameProvenance(1, frame, "source") for frame in range(3)
        ),
        seam_indices=(),
    )
    extras = {
        "continuous": np.asarray((0.0, 1.0, 2.0), dtype=np.float32),
        "stance": np.asarray((False, True, False), dtype=np.bool_),
    }

    retimed, retimed_extras = _retime_motion_and_extras(
        motion, extras, temporal_scale=2.0
    )

    assert len(retimed.root_position_world) == 5
    np.testing.assert_allclose(
        retimed.root_position_world[[0, -1]],
        motion.root_position_world[[0, -1]],
    )
    assert np.isclose(
        float(np.max(np.abs(np.diff(retimed.joint_position, axis=0)))), 0.1
    )
    np.testing.assert_allclose(
        retimed_extras["continuous"], (0.0, 0.5, 1.0, 1.5, 2.0)
    )
    assert retimed_extras["stance"].dtype == np.bool_


def test_retime_does_not_label_interpolated_takeoff_as_stance() -> None:
    motion = StitchedMotion(
        fps=50.0,
        root_position_world=np.zeros((3, 3), dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            ((1.0, 0.0, 0.0, 0.0),) * 3, dtype=np.float32
        ),
        joint_position=np.zeros((3, 1), dtype=np.float32),
        provenance=tuple(
            FrameProvenance(1, frame, "source") for frame in range(3)
        ),
        seam_indices=(),
    )
    stance = np.asarray(
        ((False, False), (True, False), (False, False)), dtype=np.bool_
    )

    _retimed, extras = _retime_motion_and_extras(
        motion,
        {"authored_stance_mask": stance},
        temporal_scale=2.0,
    )

    np.testing.assert_array_equal(
        extras["authored_stance_mask"][:, 0],
        (False, False, True, False, False),
    )


def test_adaptive_subdivision_only_inserts_frames_at_large_joint_step() -> None:
    motion = StitchedMotion(
        fps=50.0,
        root_position_world=np.zeros((3, 3), dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            ((1.0, 0.0, 0.0, 0.0),) * 3, dtype=np.float32
        ),
        joint_position=np.asarray(((0.0,), (0.40,), (0.42,)), dtype=np.float32),
        provenance=tuple(
            FrameProvenance(1, frame, "source") for frame in range(3)
        ),
        seam_indices=(),
    )

    subdivided, _extras, diagnostics = _adaptively_subdivide_motion_steps(
        motion,
        {},
        target_joint_step_rad=0.14,
    )

    assert diagnostics["added_frame_count"] == 2.0
    assert len(subdivided.joint_position) == 5
    maximum_step = float(
        np.max(np.abs(np.diff(subdivided.joint_position, axis=0)))
    )
    assert maximum_step <= 0.14 + 1.0e-6
