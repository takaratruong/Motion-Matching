import math

import numpy as np

from mm_sonic.build_global_terrain_motion_index import select_nearest_routes
from mm_sonic.compose_motionbricks_terrain_course import _phase_candidates
from mm_sonic.compose_privileged_stair_route import MotionSegment
from mm_sonic.generate_motionbricks_terrain_transitions import (
    _apply_planar_transform,
    _bezier,
    _yaw_wxyz,
)
from mm_sonic.terrain_oracle.stitch import FrameProvenance


def test_bezier_preserves_requested_entry_tangent() -> None:
    control = np.asarray(
        ((-2.0, 1.0), (-1.3, 0.5), (-0.3, 0.0), (0.5, 0.0))
    )
    path, tangent = _bezier(control, np.linspace(0.0, 1.0, 101))
    np.testing.assert_allclose(path[0], control[0])
    np.testing.assert_allclose(path[-1], control[-1])
    np.testing.assert_allclose(tangent[-1], (1.0, 0.0), atol=1.0e-8)


def test_planar_transform_moves_root_and_yaw_without_touching_joints() -> None:
    qpos = np.zeros(36, dtype=np.float64)
    qpos[:3] = (1.0, 0.0, 0.75)
    qpos[3] = 1.0
    qpos[7:] = np.linspace(-0.5, 0.5, 29)
    transformed = _apply_planar_transform(
        qpos,
        yaw_offset=math.pi / 2.0,
        translation=np.asarray((2.0, 3.0, 0.1)),
    )
    np.testing.assert_allclose(transformed[:3], (2.0, 4.0, 0.85))
    assert math.isclose(_yaw_wxyz(transformed[3:7]), math.pi / 2.0)
    np.testing.assert_array_equal(transformed[7:], qpos[7:])


def test_route_index_never_crosses_traversal_direction() -> None:
    index = {
        "routes": (
            {
                "target_clip_index": 1,
                "traversal": "up",
                "nominal_geometry": {
                    "rise_m": 0.20,
                    "tread_m": 0.30,
                    "step_count": 7,
                },
            },
            {
                "target_clip_index": 2,
                "traversal": "down",
                "nominal_geometry": {
                    "rise_m": 0.20,
                    "tread_m": 0.30,
                    "step_count": 7,
                },
            },
        )
    }
    selected = select_nearest_routes(
        index,
        traversal="down",
        rise_m=0.20,
        tread_m=0.30,
        step_count=7,
    )
    assert [row["target_clip_index"] for row in selected] == [2]


def test_reusable_template_global_yaw_is_a_free_registration() -> None:
    source_root = np.zeros((50, 3), dtype=np.float32)
    source_root[:, 0] = np.arange(50, dtype=np.float32) * 0.01
    source_quaternion = np.zeros((50, 4), dtype=np.float32)
    source_quaternion[:, 0] = 1.0
    target_root = np.asarray(((3.0, 4.0, 0.0), (3.0, 4.01, 0.0)))
    target_quaternion = np.tile(
        np.asarray((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))), (2, 1)
    )

    def segment(label: str, root: np.ndarray, quaternion: np.ndarray) -> MotionSegment:
        return MotionSegment(
            label=label,
            root_position_world=root,
            root_quaternion_world_wxyz=quaternion,
            joint_position=np.zeros((len(root), 29), dtype=np.float32),
            provenance=tuple(
                FrameProvenance(-1, frame, label) for frame in range(len(root))
            ),
        )

    candidates = _phase_candidates(
        segment("template", source_root, source_quaternion),
        segment("rotated_stair", target_root, target_quaternion),
        source_is_approach=True,
        maximum_position_correction_m=0.55,
        maximum_yaw_correction_rad=math.radians(32.0),
        candidate_limit=4,
        planar_registration_is_free=True,
    )
    assert candidates
    assert all(
        math.isclose(candidate.yaw_correction_rad, math.pi / 2.0)
        for candidate in candidates
    )
