import sys
import types

import numpy as np
import torch

from mm_sonic.motionbricks_terrain_portal import (
    MotionBricksTerrainCourse,
    TerrainCoursePlayback,
    select_terrain_portal,
)
from mm_sonic.motionbricks_global_terrain_viewer import _submit_motionbricks


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
