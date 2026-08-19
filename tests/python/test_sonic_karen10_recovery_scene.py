from __future__ import annotations

import pytest
import numpy as np

from mm_sonic.build_karen10_recovery_scene import (
    KAREN_LATERAL_BOUNDS_M,
    KAREN_STEP_TOPS_M,
    KAREN_TOP_PLATFORM_DEPTH_M,
    KAREN_TREAD_PITCH_M,
    karen10_profile,
    build_rigid,
)
from mm_sonic.build_synthesized_recovery_reference import (
    isaaclab_to_mujoco_joints,
)
from mm_sonic.grail_terrain_source import mujoco_to_isaaclab_joints
from mm_sonic.build_trace_recovery_reference import transform_root_to_reference


def test_karen10_profile_has_ten_exact_risers_and_deep_platform() -> None:
    profile = karen10_profile()
    vertical = [
        (left, right)
        for left, right in zip(profile, profile[1:])
        if left[0] == right[0] and left[1] != right[1]
    ]

    assert len(vertical) == 10
    assert [right[1] for _, right in vertical] == list(KAREN_STEP_TOPS_M)
    assert vertical[-1][0][0] == pytest.approx(9 * KAREN_TREAD_PITCH_M)
    assert profile[-1][0] == pytest.approx(
        9 * KAREN_TREAD_PITCH_M + KAREN_TOP_PLATFORM_DEPTH_M
    )
    assert profile[-1][1] == pytest.approx(1.7558225)


def test_recovery_reference_joint_permutation_round_trips() -> None:
    mujoco = np.arange(29, dtype=np.float32)[None, :]
    isaaclab = mujoco_to_isaaclab_joints(mujoco)

    np.testing.assert_array_equal(isaaclab_to_mujoco_joints(isaaclab), mujoco)


def test_success_trace_root_is_rebased_without_pose_snapping() -> None:
    roots = np.asarray(
        ((1.2, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0),), dtype=np.float32
    )

    transformed = transform_root_to_reference(
        roots,
        learner_front_xy=(1.0, 0.0),
        reference_front_xy=(0.0, 0.0),
        yaw_delta_rad=0.0,
    )

    np.testing.assert_allclose(
        transformed[0, :3], (0.2, 0.1, 0.8), atol=1.0e-6
    )
    np.testing.assert_allclose(transformed[0, 3:], roots[0, 3:])


def test_rigid_karen_asset_has_one_body_and_ten_collision_meshes(
    tmp_path,
) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(build_rigid(tmp_path / "karen10.usda")))
    model = stage.GetPrimAtPath("/model")
    steps = [
        stage.GetPrimAtPath(f"/model/geometry/step_{index:02d}")
        for index in range(1, 11)
    ]

    assert "PhysicsRigidBodyAPI" in model.GetAppliedSchemas()
    assert UsdPhysics.MassAPI(model).GetMassAttr().Get() == pytest.approx(1.0)
    assert all("PhysicsCollisionAPI" in step.GetAppliedSchemas() for step in steps)
    for step, lateral_bounds in zip(steps, KAREN_LATERAL_BOUNDS_M):
        points = step.GetAttribute("points").Get()
        assert min(point[1] for point in points) == pytest.approx(lateral_bounds[0])
        assert max(point[1] for point in points) == pytest.approx(lateral_bounds[1])
    assert stage.GetDefaultPrim() == model
