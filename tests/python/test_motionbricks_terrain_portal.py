import sys
import types

import numpy as np
import pytest
import torch

from mm_sonic.motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    TerrainCoursePlayback,
    select_terrain_seam_portal,
    select_terrain_portal,
)
from mm_sonic.build_backward_facing_terrain_portals import (
    _reverse_motion,
    _trim_hovering_flat_margins,
)
from mm_sonic.motionbricks_global_terrain_viewer import (
    MotionBricksCommandBufferInvalidator,
    MotionBricksExitFootLock,
    PORTAL_ENTRY_LEAD_TIME_S,
    _guard_flat_velocity,
    _minimum_sole_clearance_m,
    _flat_pose_support_error,
    _laterally_registered_course,
    _registration_matches_terrain,
    _submit_motionbricks,
    _submit_responsive_motionbricks,
    _verified_motionbricks_context,
)
from mm_sonic.motionbricks_global_terrain_safety_probe import (
    _left_multiply_yaw,
)


class _FlatTerrain:
    @staticmethod
    def raycast(origin, _direction):
        return types.SimpleNamespace(
            position_world=np.asarray(
                (float(origin[0]), float(origin[1]), 0.0)
            )
        )


class _MutableBoundaryTerrain:
    def __init__(self) -> None:
        self.straddling = True

    def raycast(self, origin, _direction):
        height = 0.15 if self.straddling and float(origin[1]) > 0.05 else 0.0
        return types.SimpleNamespace(
            position_world=np.asarray(
                (float(origin[0]), float(origin[1]), height)
            )
        )


class _EncodedSoleAdapter:
    """Tiny deterministic two-foot model for exit-lock state tests."""

    @staticmethod
    def _centres(joints):
        values = np.asarray(joints, dtype=np.float64)
        return (
            np.asarray(((0.0, -0.1, values[0]),), dtype=np.float64),
            np.asarray(((0.0, 0.1, values[3]),), dtype=np.float64),
        )

    def sole_positions_for_pose(self, *, joints, **_kwargs):
        return self._centres(joints)

    def sole_support_points_for_pose(self, *, joints, **_kwargs):
        return self._centres(joints)


class _EncodedSoleIk(_EncodedSoleAdapter):
    def adapt_to_targets(
        self,
        *,
        authored_joints,
        sole_targets_world,
        **_kwargs,
    ):
        authored = np.asarray(authored_joints, dtype=np.float64)
        result = authored.copy()
        result[0] = float(np.asarray(sole_targets_world[0])[0, 2])
        result[3] = float(np.asarray(sole_targets_world[1])[0, 2])
        return result, float(np.max(np.abs(result - authored))), 0.0


def _encoded_sole_qpos(left_height: float, right_height: float) -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float64)
    qpos[3] = 1.0
    qpos[7] = float(left_height)
    qpos[8] = float(right_height)
    return qpos


def test_command_buffer_ignores_small_joystick_jitter() -> None:
    buffer = MotionBricksCommandBufferInvalidator()
    buffer.observe_generated(
        velocity_world_xy=(0.40, 0.0),
        facing_yaw_world=0.0,
        mode_name="walk",
    )
    assert buffer.invalidation_reasons(
        velocity_world_xy=(0.42, 0.03),
        facing_yaw_world=np.deg2rad(8.0),
        mode_name="walk",
    ) == ()


@pytest.mark.parametrize(
    ("velocity", "facing", "mode", "expected"),
    (
        ((-0.40, 0.0), 0.0, "walk", "direction_change"),
        ((0.0, 0.0), 0.0, "idle", "start_stop"),
        ((0.40, 0.0), np.deg2rad(40.0), "walk", "facing_change"),
    ),
)
def test_command_buffer_invalidates_material_operator_changes(
    velocity, facing, mode, expected
) -> None:
    buffer = MotionBricksCommandBufferInvalidator()
    buffer.observe_generated(
        velocity_world_xy=(0.40, 0.0),
        facing_yaw_world=0.0,
        mode_name="walk",
    )
    assert expected in buffer.invalidation_reasons(
        velocity_world_xy=velocity,
        facing_yaw_world=facing,
        mode_name=mode,
    )


def test_verified_motionbricks_context_pads_only_the_oldest_pose() -> None:
    previous = np.zeros(36, dtype=np.float64)
    current = np.ones(36, dtype=np.float64)
    context = _verified_motionbricks_context((previous, current))
    assert context.shape == (4, 36)
    np.testing.assert_allclose(context[:3], np.repeat(previous[None], 3, axis=0))
    np.testing.assert_allclose(context[3], current)


def test_safety_probe_yaw_offset_is_a_normalized_left_rotation() -> None:
    rotated = _left_multiply_yaw((1.0, 0.0, 0.0, 0.0), 0.5 * np.pi)
    np.testing.assert_allclose(
        rotated,
        np.asarray((2.0**-0.5, 0.0, 0.0, 2.0**-0.5)),
    )
    assert np.linalg.norm(rotated) == pytest.approx(1.0)


def _course() -> MotionBricksTerrainCourse:
    root = np.zeros((100, 3), dtype=np.float64)
    root[:, 0] = np.linspace(0.0, 2.0, len(root))
    root[:, 2] = 0.78
    quaternion = np.zeros((len(root), 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    joints = np.zeros((len(root), 29), dtype=np.float64)
    joints[:, 0] = np.linspace(0.0, 0.2, len(root))
    return MotionBricksTerrainCourse(
        path=None,  # type: ignore[arg-type]
        fps=50.0,
        root_position_world=root,
        root_quaternion_world_wxyz=quaternion,
        joint_position_isaaclab=joints,
        seam_indices=(40, 70),
    )


def test_native_qpos_and_portal_capture_use_the_same_joint_pose() -> None:
    course = _course()
    qpos = course.native_mujoco_qpos()[0]
    capture = course.portal_capture(qpos, np.asarray((0.3, 0.0)))
    assert capture.accepted
    assert capture.position_error_m == 0.0
    assert capture.lower_body_rmse_rad == 0.0
    assert not course.portal_capture(qpos, np.asarray((-0.3, 0.0))).accepted


def test_entry_blend_starts_at_live_pose_and_finishes_on_authored_course() -> None:
    course = _course()
    current = course.native_mujoco_qpos()[0]
    current[:2] += (0.05, -0.03)
    current[7] += 0.10
    playback = TerrainCoursePlayback(course, current, blend_frames=18)
    first = playback.next_qpos()
    np.testing.assert_allclose(first, current, atol=1.0e-10)
    values = [playback.next_qpos() for _ in range(17)]
    np.testing.assert_allclose(values[-1], course.native_mujoco_qpos()[17])


def test_just_in_time_entry_skips_the_long_authored_approach() -> None:
    course = _course()
    entry_frame = course.entry_frame_index(0.40)
    assert entry_frame == 20
    current = course.native_mujoco_qpos()[entry_frame]
    current[:2] += (0.03, -0.02)
    selected = select_terrain_portal(
        (course,),
        current,
        np.asarray((0.3, 0.0)),
        entry_lead_time_s=0.40,
    )
    assert selected.capture.accepted
    assert selected.entry_frame_index == entry_frame

    playback = TerrainCoursePlayback(
        course,
        current,
        blend_frames=18,
        start_frame=selected.entry_frame_index,
    )
    np.testing.assert_allclose(playback.next_qpos(), current, atol=1.0e-10)
    values = [playback.next_qpos() for _ in range(17)]
    np.testing.assert_allclose(values[-1], course.native_mujoco_qpos()[37])


def test_tight_landing_capture_can_enter_time_reverse_at_terrain_seam() -> None:
    course = _course()
    seam = course.seam_indices[0]
    current = course.native_mujoco_qpos()[seam]
    selected = select_terrain_seam_portal(
        (course,), current, np.asarray((0.3, 0.0))
    )
    assert selected is not None
    assert selected.capture.accepted
    assert selected.entry_frame_index == seam

    playback = TerrainCoursePlayback(
        course,
        current,
        blend_frames=18,
        start_frame=seam,
        allow_terrain_seam_start=True,
    )
    np.testing.assert_allclose(playback.next_qpos(), current, atol=1.0e-10)
    values = [playback.next_qpos() for _ in range(17)]
    np.testing.assert_allclose(values[-1], course.native_mujoco_qpos()[seam + 17])


def test_landing_capture_rejects_wrong_travel_and_nonmatching_pose() -> None:
    course = _course()
    seam = course.seam_indices[0]
    current = course.native_mujoco_qpos()[seam]
    assert select_terrain_seam_portal(
        (course,), current, np.asarray((-0.3, 0.0))
    ) is None
    current[7] += 2.0
    assert select_terrain_seam_portal(
        (course,), current, np.asarray((0.3, 0.0))
    ) is None


def test_motionbricks_context_is_resampled_at_thirty_hertz() -> None:
    course = _course()
    context = course.resampled_context_qpos(at_end=True, target_fps=30.0)
    assert context.shape == (4, 36)
    np.testing.assert_allclose(context[-1], course.native_mujoco_qpos()[-1])
    np.testing.assert_allclose(
        np.diff(context[:, 0]),
        np.full(3, (2.0 / 99.0) * (50.0 / 30.0)),
        rtol=1.0e-6,
    )


def test_terminal_course_with_one_entry_seam_loads(tmp_path) -> None:
    course = _course()
    path = tmp_path / "terminal_course.npz"
    np.savez_compressed(
        path,
        fps=np.asarray(course.fps),
        root_position_world=course.root_position_world,
        root_quaternion_world_wxyz=course.root_quaternion_world_wxyz,
        joint_position=course.joint_position_isaaclab,
        seam_indices=np.asarray((40,), dtype=np.int64),
    )
    loaded = MotionBricksTerrainCourse.load(path)
    assert loaded.seam_indices == (40,)
    assert loaded.frame_count == course.frame_count


def test_complete_course_reversal_turns_exit_into_flat_entry(tmp_path) -> None:
    course = _course()
    source = tmp_path / "forward.npz"
    np.savez_compressed(
        source,
        fps=np.asarray(course.fps),
        root_position_world=course.root_position_world,
        root_quaternion_world_wxyz=course.root_quaternion_world_wxyz,
        joint_position=course.joint_position_isaaclab,
        seam_indices=np.asarray((20, 50), dtype=np.int64),
        source_frame=np.arange(course.frame_count, dtype=np.int64),
    )
    destination = tmp_path / "backward.npz"
    receipt = _reverse_motion(source, destination)
    reversed_course = MotionBricksTerrainCourse.load(destination)
    assert reversed_course.seam_indices == (50, 80)
    np.testing.assert_allclose(
        reversed_course.root_position_world,
        course.root_position_world[::-1],
    )
    with np.load(destination, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["source_frame"], np.arange(course.frame_count)[::-1]
        )
    assert receipt["flat_approach_frame_count"] == 50


def test_hover_trim_removes_only_unsupported_outer_flat_tail(tmp_path) -> None:
    course = _course()
    path = tmp_path / "hovering_tail.npz"
    root = np.concatenate(
        (course.root_position_world, course.root_position_world[:40]), axis=0
    )
    quaternion = np.concatenate(
        (
            course.root_quaternion_world_wxyz,
            course.root_quaternion_world_wxyz[:40],
        ),
        axis=0,
    )
    joints = np.concatenate(
        (course.joint_position_isaaclab, course.joint_position_isaaclab[:40]),
        axis=0,
    )
    np.savez_compressed(
        path,
        fps=np.asarray(course.fps),
        root_position_world=root,
        root_quaternion_world_wxyz=quaternion,
        joint_position=joints,
        seam_indices=np.asarray((50, 100), dtype=np.int64),
        source_frame=np.arange(len(root), dtype=np.int64),
    )
    receipt = _trim_hovering_flat_margins(
        path,
        {
            "threshold_exceedance_frame_indices": list(range(120, 140)),
            "endpoint_accepted": False,
            "first_grounded_frame_index": 0,
            "last_grounded_frame_index": 119,
        },
    )
    assert receipt is not None
    assert receipt["frame_range"] == [0, 120]
    trimmed = MotionBricksTerrainCourse.load(path)
    assert trimmed.frame_count == 120
    assert trimmed.seam_indices == (50, 100)


def test_hover_trim_never_cuts_an_unsupported_terrain_traversal(tmp_path) -> None:
    course = _course()
    path = tmp_path / "hovering_terrain.npz"
    np.savez_compressed(
        path,
        fps=np.asarray(course.fps),
        root_position_world=course.root_position_world,
        root_quaternion_world_wxyz=course.root_quaternion_world_wxyz,
        joint_position=course.joint_position_isaaclab,
        seam_indices=np.asarray((40, 70), dtype=np.int64),
    )
    assert _trim_hovering_flat_margins(
        path, {"threshold_exceedance_frame_indices": [55]}
    ) is None


def test_portal_catalog_prefers_the_compatible_entry_family() -> None:
    first = _course()
    shifted_root = first.root_position_world.copy()
    shifted_root[:, 1] += 0.8
    second = MotionBricksTerrainCourse(
        path=None,  # type: ignore[arg-type]
        fps=first.fps,
        root_position_world=shifted_root,
        root_quaternion_world_wxyz=first.root_quaternion_world_wxyz,
        joint_position_isaaclab=first.joint_position_isaaclab,
        seam_indices=first.seam_indices,
    )
    qpos = second.native_mujoco_qpos()[0]
    selected = select_terrain_portal(
        (first, second), qpos, np.asarray((0.3, 0.0))
    )
    assert selected.course_index == 1
    assert selected.capture.accepted


def test_portal_lane_registration_preserves_longitudinal_trajectory() -> None:
    course = _course()
    entry = course.entry_frame_index(0.40)
    registered, shift = _laterally_registered_course(
        course,
        course.root_position_world[entry, :2] + np.asarray((0.0, 0.65)),
        entry_lead_time_s=0.40,
    )
    assert shift == 0.65
    np.testing.assert_allclose(
        registered.root_position_world[:, 0], course.root_position_world[:, 0]
    )
    np.testing.assert_allclose(
        registered.root_position_world[:, 1],
        course.root_position_world[:, 1] + 0.65,
    )


def test_manifest_axis_ignores_authored_lateral_root_wiggle() -> None:
    course = _course()
    root = course.root_position_world.copy()
    root[:, 1] = np.linspace(0.0, 0.25, len(root))
    wiggly = MotionBricksTerrainCourse(
        path=course.path,
        fps=course.fps,
        root_position_world=root,
        root_quaternion_world_wxyz=course.root_quaternion_world_wxyz,
        joint_position_isaaclab=course.joint_position_isaaclab,
        seam_indices=course.seam_indices,
    )
    entry = wiggly.entry_frame_index(0.40)
    registered, shift = _laterally_registered_course(
        wiggly,
        wiggly.root_position_world[entry, :2] + np.asarray((0.0, 0.65)),
        entry_lead_time_s=0.40,
        lateral_axis_world_xy=(0.0, 1.0),
    )
    assert shift == 0.65
    np.testing.assert_allclose(
        registered.root_position_world[:, 0], wiggly.root_position_world[:, 0]
    )
    np.testing.assert_allclose(
        registered.root_position_world[:, 1],
        wiggly.root_position_world[:, 1] + 0.65,
    )


def test_flat_guard_stops_before_ramp_instead_of_carrying_root_up() -> None:
    class Terrain:
        @staticmethod
        def raycast(origin, _direction):
            x = float(origin[0])
            height = max(0.0, 0.125 * x)
            return types.SimpleNamespace(
                position_world=np.asarray((x, float(origin[1]), height))
            )

    terrain = Terrain()
    velocity = np.asarray((0.5, 0.0))
    guarded, blocked = _guard_flat_velocity(
        terrain,  # type: ignore[arg-type]
        np.asarray((-0.05, 0.0, 0.8)),
        velocity,
        ray_origin_z=2.0,
    )
    assert blocked
    np.testing.assert_allclose(guarded, np.zeros(2))

    flat, blocked = _guard_flat_velocity(
        terrain,  # type: ignore[arg-type]
        np.asarray((-1.0, 0.0, 0.8)),
        velocity,
        ray_origin_z=2.0,
    )
    assert not blocked
    np.testing.assert_allclose(flat, velocity)


def test_decoded_flat_pose_is_rejected_when_a_foot_enters_a_ramp() -> None:
    class Terrain:
        @staticmethod
        def raycast(origin, _direction):
            x = float(origin[0])
            height = max(0.0, 0.20 * x)
            return types.SimpleNamespace(
                position_world=np.asarray((x, float(origin[1]), height))
            )

    class Soles:
        @staticmethod
        def sole_support_points_for_pose(**_kwargs):
            return (
                np.asarray(((0.14, -0.08, 0.0), (0.22, -0.08, 0.0))),
                np.asarray(((-0.12, 0.08, 0.0), (-0.04, 0.08, 0.0))),
            )

    qpos = np.zeros(36, dtype=np.float64)
    qpos[2] = 0.80
    qpos[3] = 1.0
    error = _flat_pose_support_error(
        qpos,
        sole_adapter=Soles(),  # type: ignore[arg-type]
        terrain=Terrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
        support_height_world=0.0,
    )
    assert error == pytest.approx(0.044)


def test_decoded_flat_pose_accepts_all_samples_on_current_platform() -> None:
    class Terrain:
        @staticmethod
        def raycast(origin, _direction):
            return types.SimpleNamespace(
                position_world=np.asarray(
                    (float(origin[0]), float(origin[1]), 0.25)
                )
            )

    class Soles:
        @staticmethod
        def sole_support_points_for_pose(**_kwargs):
            return (
                np.asarray(((0.14, -0.08, 0.25), (0.22, -0.08, 0.25))),
                np.asarray(((-0.12, 0.08, 0.25), (-0.04, 0.08, 0.25))),
            )

    qpos = np.zeros(36, dtype=np.float64)
    qpos[2] = 1.05
    qpos[3] = 1.0
    assert _flat_pose_support_error(
        qpos,
        sole_adapter=Soles(),  # type: ignore[arg-type]
        terrain=Terrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
        support_height_world=0.25,
    ) == 0.0


def test_minimum_sole_clearance_detects_hover_above_lower_surface() -> None:
    class Terrain:
        @staticmethod
        def raycast(origin, _direction):
            return types.SimpleNamespace(
                position_world=np.asarray(
                    (float(origin[0]), float(origin[1]), 0.10)
                )
            )

    class Soles:
        @staticmethod
        def sole_support_points_for_pose(**_kwargs):
            return (
                np.asarray(((0.0, 0.0, 0.18), (0.1, 0.0, 0.19))),
                np.asarray(((0.0, 0.1, 0.32), (0.1, 0.1, 0.33))),
            )

    qpos = np.zeros(36, dtype=np.float64)
    qpos[3] = 1.0
    assert _minimum_sole_clearance_m(
        qpos,
        sole_adapter=Soles(),  # type: ignore[arg-type]
        terrain=Terrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
    ) == pytest.approx(0.08)


def test_exit_foot_lock_holds_course_support_until_live_support_arrives() -> None:
    adapter = _EncodedSoleAdapter()
    lock = MotionBricksExitFootLock(
        sole_adapter=adapter,  # type: ignore[arg-type]
        ik_adapter=_EncodedSoleIk(),  # type: ignore[arg-type]
        terrain=_FlatTerrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
        release_confirmation_frames=2,
        release_frames=3,
    )
    receipt = lock.arm(_encoded_sole_qpos(0.005, 0.20))
    assert receipt["support_foot"] == 0

    first = lock.apply(_encoded_sole_qpos(0.12, 0.12))
    assert first.triggered
    assert first.active
    assert first.raw_minimum_clearance_m == pytest.approx(0.12)
    assert first.corrected_minimum_clearance_m == pytest.approx(0.005)

    second = lock.apply(_encoded_sole_qpos(0.12, 0.02))
    assert second.active
    third = lock.apply(_encoded_sole_qpos(0.12, 0.02))
    assert third.active
    assert third.corrected_minimum_clearance_m <= 0.02
    fourth = lock.apply(_encoded_sole_qpos(0.12, 0.02))
    assert fourth.active
    fifth = lock.apply(_encoded_sole_qpos(0.12, 0.02))
    assert fifth.active
    sixth = lock.apply(_encoded_sole_qpos(0.12, 0.02))
    assert sixth.active
    seventh = lock.apply(_encoded_sole_qpos(0.12, 0.02))
    assert seventh.released
    assert not seventh.active
    assert not lock.armed


def test_exit_foot_lock_does_not_touch_an_already_supported_live_phase() -> None:
    adapter = _EncodedSoleAdapter()
    lock = MotionBricksExitFootLock(
        sole_adapter=adapter,  # type: ignore[arg-type]
        ik_adapter=_EncodedSoleIk(),  # type: ignore[arg-type]
        terrain=_FlatTerrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
    )
    lock.arm(_encoded_sole_qpos(0.005, 0.20))
    proposal = _encoded_sole_qpos(0.02, 0.12)
    first = lock.apply(proposal)
    assert not first.triggered
    assert first.active
    second = lock.apply(proposal)
    assert second.released
    assert not second.active
    np.testing.assert_array_equal(second.qpos, proposal)


def test_supported_exit_stays_guarded_until_footprint_clears_boundary() -> None:
    adapter = _EncodedSoleAdapter()
    terrain = _MutableBoundaryTerrain()
    lock = MotionBricksExitFootLock(
        sole_adapter=adapter,  # type: ignore[arg-type]
        ik_adapter=_EncodedSoleIk(),  # type: ignore[arg-type]
        terrain=terrain,  # type: ignore[arg-type]
        ray_origin_z=2.0,
    )
    lock.arm(_encoded_sole_qpos(0.005, 0.20))
    proposal = _encoded_sole_qpos(0.02, 0.17)
    boundary = lock.apply(proposal)
    assert boundary.active
    assert not boundary.released
    assert boundary.joint_correction_rad == 0.0

    terrain.straddling = False
    assert lock.apply(proposal).active
    released = lock.apply(proposal)
    assert released.released
    assert not released.active


def test_portal_capture_opens_before_flat_guard_at_a_step() -> None:
    class Terrain:
        @staticmethod
        def raycast(origin, _direction):
            x = float(origin[0])
            height = 0.20 if x >= 0.0 else 0.0
            return types.SimpleNamespace(
                position_world=np.asarray((x, float(origin[1]), height))
            )

    base = _course()
    root = base.root_position_world.copy()
    root[:, 0] -= 0.21
    course = MotionBricksTerrainCourse(
        path=base.path,
        fps=base.fps,
        root_position_world=root,
        root_quaternion_world_wxyz=base.root_quaternion_world_wxyz,
        joint_position_isaaclab=base.joint_position_isaaclab,
        seam_indices=base.seam_indices,
    )
    entry = course.entry_frame_index(PORTAL_ENTRY_LEAD_TIME_S)
    assert entry == 0
    current = course.native_mujoco_qpos()[entry]
    current[0] = -0.34
    velocity = np.asarray((0.5, 0.0))
    assert course.portal_capture(current, velocity, frame_index=entry).accepted
    guarded, blocked = _guard_flat_velocity(
        Terrain(),  # type: ignore[arg-type]
        current[:3],
        velocity,
        ray_origin_z=2.0,
    )
    assert not blocked
    np.testing.assert_allclose(guarded, velocity)


def test_lane_registration_requires_the_same_support_profile() -> None:
    class Terrain:
        @staticmethod
        def raycast(origin, _direction):
            x, y = float(origin[0]), float(origin[1])
            if abs(y) > 1.0:
                return None
            return types.SimpleNamespace(
                position_world=np.asarray((x, y, 0.1 * x))
            )

    course = _course()
    entry = course.entry_frame_index(0.40)
    shifted, _ = _laterally_registered_course(
        course,
        course.root_position_world[entry, :2] + np.asarray((0.0, 0.40)),
        entry_lead_time_s=0.40,
    )
    assert _registration_matches_terrain(
        course,
        shifted,
        terrain=Terrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
        entry_frame=entry,
    )

    outside, _ = _laterally_registered_course(
        course,
        course.root_position_world[entry, :2] + np.asarray((0.0, 1.40)),
        entry_lead_time_s=0.40,
    )
    assert not _registration_matches_terrain(
        course,
        outside,
        terrain=Terrain(),  # type: ignore[arg-type]
        ray_origin_z=2.0,
        entry_frame=entry,
    )


def test_raised_support_is_restored_once_per_generated_batch(monkeypatch) -> None:
    clips = types.ModuleType("motionbricks.motion_backbone.demo.clips")
    clips.clip_holder_G1 = types.SimpleNamespace(CLIPS={"walk": object()})
    for name in (
        "motionbricks",
        "motionbricks.motion_backbone",
        "motionbricks.motion_backbone.demo",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, clips.__name__, clips)

    class Controller:
        @staticmethod
        def get_default_allowed_pred_num_tokens(_mode: int) -> int:
            return 12

        @staticmethod
        def get_controller_dt() -> float:
            return 1.0 / 30.0

    class Agent:
        def __init__(self) -> None:
            self.frames = {"mujoco_qpos": torch.zeros((1, 1, 36))}
            self.generate_next = False
            self.last_signals = None

        def generate_new_frames(
            self, signals, _duration: float, *, force_generation: bool
        ) -> None:
            self.last_signals = signals
            if force_generation or self.generate_next:
                generated = torch.zeros((1, 8, 36))
                generated[..., 2] = 0.70
                self.frames["mujoco_qpos"] = generated
                self.generate_next = False

    context = np.zeros((4, 36), dtype=np.float32)
    context[:, 2] = 1.03
    agent = Agent()
    controller = Controller()
    arguments = {
        "context_qpos": context,
        "velocity_world_xy": np.asarray((0.3, 0.0)),
        "facing_yaw_world": 0.0,
        "mode_name": "walk",
        "random_seed": 0,
        "support_height_world": 0.33,
    }

    _submit_motionbricks(agent, controller, force=True, **arguments)
    np.testing.assert_allclose(
        agent.last_signals["context_mujoco_qpos"][..., 2], 0.70, atol=1.0e-6
    )
    np.testing.assert_allclose(
        agent.frames["mujoco_qpos"][..., 2], 1.03, atol=1.0e-6
    )

    # A cached batch is returned unchanged; its world support offset must not
    # be accumulated again on every controller tick.
    _submit_motionbricks(agent, controller, force=False, **arguments)
    np.testing.assert_allclose(
        agent.frames["mujoco_qpos"][..., 2], 1.03, atol=1.0e-6
    )

    # When the agent really generates a new batch, restore the offset once.
    agent.generate_next = True
    _submit_motionbricks(agent, controller, force=False, **arguments)
    np.testing.assert_allclose(
        agent.frames["mujoco_qpos"][..., 2], 1.03, atol=1.0e-6
    )


def test_responsive_submission_discards_verified_prefix_on_reversal(
    monkeypatch,
) -> None:
    clips = types.ModuleType("motionbricks.motion_backbone.demo.clips")
    clips.clip_holder_G1 = types.SimpleNamespace(
        CLIPS={"idle": object(), "walk": object()}
    )
    for name in (
        "motionbricks",
        "motionbricks.motion_backbone",
        "motionbricks.motion_backbone.demo",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, clips.__name__, clips)

    class Controller:
        @staticmethod
        def get_default_allowed_pred_num_tokens(_mode: int) -> int:
            return 12

        @staticmethod
        def get_controller_dt() -> float:
            return 1.0 / 30.0

    class Agent:
        def __init__(self) -> None:
            self.frames = {"mujoco_qpos": torch.zeros((1, 8, 36))}
            self.next_count = 0
            self.last_signals = None

        def generate_new_frames(
            self, signals, _duration: float, *, force_generation: bool
        ) -> None:
            self.last_signals = signals
            if force_generation:
                self.frames["mujoco_qpos"] = torch.zeros((1, 12, 36))

        def get_next_frame(self):
            self.next_count += 1
            return np.zeros(36)

        def get_context_mujoco_qpos(self):
            return torch.zeros((1, 4, 36))

    history = []
    for index in range(4):
        qpos = np.zeros(36, dtype=np.float64)
        qpos[0] = index
        history.append(qpos)
    buffer = MotionBricksCommandBufferInvalidator()
    buffer.observe_generated(
        velocity_world_xy=(0.4, 0.0),
        facing_yaw_world=0.0,
        mode_name="walk",
    )
    agent = Agent()
    receipt = _submit_responsive_motionbricks(
        agent,
        Controller(),
        command_buffer=buffer,
        verified_history=history,
        velocity_world_xy=(-0.4, 0.0),
        facing_yaw_world=0.0,
        mode_name="walk",
    )
    assert receipt.invalidated
    assert receipt.discarded_conditioning_frame_count == 4
    assert agent.next_count == 4
    np.testing.assert_allclose(
        agent.last_signals["context_mujoco_qpos"][0, :, 0],
        np.arange(4),
    )

    cached = _submit_responsive_motionbricks(
        agent,
        Controller(),
        command_buffer=buffer,
        verified_history=history,
        velocity_world_xy=(-0.4, 0.0),
        facing_yaw_world=0.0,
        mode_name="walk",
    )
    assert not cached.invalidated
    assert agent.next_count == 4
