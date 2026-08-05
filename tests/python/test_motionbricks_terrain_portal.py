import sys
import types

import numpy as np
import torch

from mm_sonic.motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    TerrainCoursePlayback,
    select_terrain_portal,
)
from mm_sonic.motionbricks_global_terrain_viewer import (
    PORTAL_ENTRY_LEAD_TIME_S,
    _guard_flat_velocity,
    _laterally_registered_course,
    _registration_matches_terrain,
    _submit_motionbricks,
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
