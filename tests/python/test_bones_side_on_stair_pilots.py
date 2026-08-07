import math
from types import SimpleNamespace

import numpy as np

from mm_sonic.build_bones_side_on_stair_pilots import (
    _adaptively_subdivide_motion_steps,
    _lateral_stance_level_schedule,
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
        SimpleNamespace(foot_index=0, start_frame=0, stop_frame=5),
        SimpleNamespace(foot_index=1, start_frame=5, stop_frame=10),
        SimpleNamespace(foot_index=0, start_frame=10, stop_frame=15),
        SimpleNamespace(foot_index=1, start_frame=15, stop_frame=20),
        SimpleNamespace(foot_index=0, start_frame=20, stop_frame=25),
    ]

    schedule = _lateral_stance_level_schedule(
        spans, leading_foot_index=1, level_count=3
    )

    assert list(schedule.values()) == [0, 1, 1, 2, 2]


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
