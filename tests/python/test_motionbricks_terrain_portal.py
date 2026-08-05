import numpy as np

from mm_sonic.motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    TerrainCoursePlayback,
    select_terrain_portal,
)


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
