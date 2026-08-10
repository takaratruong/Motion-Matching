from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

from mm_sonic.canonical_terrain_matcher import (
    CanonicalTerrainLibrary,
    ContinuousTerrainMotionMatcher,
    ContinuousTerrainRuntimeConfig,
    GenericTerrainSearchConfig,
    MeshHeightField,
    RegularGridHeightField,
    TerrainRowFeatures,
    _exact_sole_corners_world,
    _kinematic_stance_from_sole_motion,
)
from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    CanonicalTerrainMesh,
)
from mm_sonic.terrain_oracle.contact import SoleGeometry
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.terrain_mesh import TerrainMeshIndex
from mm_sonic.evaluate_continuous_terrain_course import (
    _command_for_step,
    _reset_alignment,
    build_flat_course,
)
from mm_sonic.torch_motion_data import MotionFolder
from mm_sonic.torch_motion_matcher import MatcherConfig, TorchMotionMatcher
from tests.python.torch_motion_test_utils import (
    build_varying_takara_arrays,
    write_takara_arrays,
)


def _identity() -> RigidTransform:
    return RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )


def test_height_field_is_generic_over_flat_ramp_and_raised_patch() -> None:
    # Large sloped plane z=x plus a smaller horizontal patch at z=2.  The
    # query must return the top surface without knowing either semantic label.
    vertices = np.asarray(
        [
            (-2.0, -2.0, -2.0),
            (2.0, -2.0, 2.0),
            (2.0, 2.0, 2.0),
            (-2.0, 2.0, -2.0),
            (-0.5, -0.5, 2.0),
            (0.5, -0.5, 2.0),
            (0.5, 0.5, 2.0),
            (-0.5, 0.5, 2.0),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)), dtype=np.int32
    )
    mesh = CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=faces,
        valid_faces=np.ones(4, dtype=np.bool_),
        source_asset_sha256="a" * 64,
    )
    field = MeshHeightField(TerrainMeshIndex(mesh, _identity()))
    height, normal, hit = field.sample(
        np.asarray(((-1.0, 1.0), (0.0, 0.0), (1.0, -1.0), (3.0, 0.0)))
    )
    np.testing.assert_allclose(height[:3], (-1.0, 2.0, 1.0), atol=1.0e-6)
    assert hit.tolist() == [True, True, True, False]
    assert np.all(normal[:3, 2] > 0.0)


def test_height_field_preserves_batch_shape() -> None:
    vertices = np.asarray(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    mesh = CanonicalTerrainMesh(
        vertices_local=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        valid_faces=np.ones(2, dtype=np.bool_),
        source_asset_sha256="b" * 64,
    )
    field = MeshHeightField(TerrainMeshIndex(mesh, _identity()))
    height, normal, hit = field.sample(np.zeros((3, 2, 4, 2), dtype=np.float32))
    assert height.shape == (3, 2, 4)
    assert normal.shape == (3, 2, 4, 3)
    assert hit.shape == (3, 2, 4)
    assert np.all(hit)


def test_regular_grid_height_field_interpolates_height_and_normal() -> None:
    # z = x + 2y sampled on a one-metre grid.
    y, x = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
    field = RegularGridHeightField(
        x + 2.0 * y,
        spacing_m=1.0,
        origin_xy=(-1.0, -2.0),
    )
    height, normal, hit = field.sample(
        np.asarray(((-0.5, -1.5), (1.25, -0.75), (4.0, 0.0)))
    )
    np.testing.assert_allclose(height[:2], (1.5, 4.75), atol=1.0e-6)
    expected = np.asarray((-1.0, -2.0, 1.0)) / np.sqrt(6.0)
    np.testing.assert_allclose(
        normal[:2], np.repeat(expected[None], 2, axis=0), atol=1.0e-6
    )
    assert hit.tolist() == [True, True, False]


def test_exact_sole_corners_use_named_ankle_pose_and_rigid_geometry() -> None:
    corners = np.asarray(
        (
            ((-0.05, -0.02, -0.03), (-0.05, 0.02, -0.03),
             (0.11, -0.02, -0.03), (0.11, 0.02, -0.03)),
            ((-0.05, -0.02, -0.03), (-0.05, 0.02, -0.03),
             (0.11, -0.02, -0.03), (0.11, 0.02, -0.03)),
        ),
        dtype=np.float32,
    )
    geometry = SoleGeometry(
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        corner_positions_body=corners,
    )
    body_position = np.zeros((2, len(ISAACLAB_BODY_NAMES), 3), np.float32)
    body_quaternion = np.zeros((2, len(ISAACLAB_BODY_NAMES), 4), np.float32)
    body_quaternion[..., 0] = 1.0
    left = ISAACLAB_BODY_NAMES.index("left_ankle_roll_link")
    right = ISAACLAB_BODY_NAMES.index("right_ankle_roll_link")
    body_position[:, left] = (1.0, 2.0, 3.0)
    body_position[:, right] = (-1.0, -2.0, 4.0)
    clip = SimpleNamespace(
        frame_count=2,
        body_names=ISAACLAB_BODY_NAMES,
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_quaternion,
    )
    world = _exact_sole_corners_world(clip, geometry)
    np.testing.assert_allclose(
        world[:, 0],
        np.repeat(corners[None, 0] + (1.0, 2.0, 3.0), 2, axis=0),
    )
    np.testing.assert_allclose(
        world[:, 1],
        np.repeat(corners[None, 1] + (-1.0, -2.0, 4.0), 2, axis=0),
    )


def test_kinematic_stance_recovers_alternating_plants_and_rejects_apex() -> None:
    frames = 40
    sole = np.zeros((frames, 2, 3), dtype=np.float64)
    sole[:, 0, 1] = 0.10
    sole[:, 1, 1] = -0.10
    sole[16:, 0, 0] = np.arange(frames - 16) * 0.02
    sole[:20, 1, 0] = np.arange(20) * 0.02
    sole[20:, 1, 0] = sole[19, 1, 0]
    # A two-frame stationary flight apex is too short to become a plant.
    sole[7:9, 1] = sole[7, 1]

    stance = _kinematic_stance_from_sole_motion(sole, fps=50.0)

    assert np.all(stance[1:15, 0])
    assert not np.any(stance[18:, 0])
    assert not np.any(stance[:18, 1])
    assert np.all(stance[21:-1, 1])


def test_two_stick_profile_is_closed_and_spans_backward_travel() -> None:
    field, metadata = build_flat_course()
    height, _normal, hit = field.sample(
        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    )
    assert np.all(hit) and np.allclose(height, 0.0)
    assert metadata["segments"] == ["flat closed-loop two-stick arena"]
    commands = [
        _command_for_step(
            "two_stick",
            step=step,
            root_position_world=np.zeros(3),
            speed_mps=0.4,
            start_x=0.0,
        )
        for step in range(640)
    ]
    velocity = np.asarray([row[0] for row in commands])
    facing = np.asarray([row[1] for row in commands])
    np.testing.assert_allclose(np.sum(velocity, axis=0), 0.0, atol=1.0e-12)
    travel_yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
    separation = np.abs(
        np.arctan2(
            np.sin(travel_yaw - facing), np.cos(travel_yaw - facing)
        )
    )
    assert float(np.max(separation)) == np.pi


def test_terrain_omnidirectional_profiles_keep_travel_and_facing_independent() -> None:
    expected_heading = {
        "side_on_left": np.pi / 2.0,
        "side_on_right": -np.pi / 2.0,
        "backward": np.pi,
    }
    for profile, heading in expected_heading.items():
        velocity, actual_heading, target_y = _command_for_step(
            profile,
            step=0,
            root_position_world=np.zeros(3),
            speed_mps=0.4,
            start_x=0.0,
        )
        np.testing.assert_allclose(velocity, (0.4, 0.0))
        np.testing.assert_allclose(actual_heading, heading)
        assert target_y == 0.0


def test_reset_alignment_preserves_requested_nonzero_world_yaw() -> None:
    field, _metadata = build_flat_course()
    source_yaw = 0.4
    source = SimpleNamespace(
        root_position_world=np.asarray(((0.5, -0.2, 0.8),)),
        root_quaternion_world_wxyz=np.asarray(
            ((np.cos(source_yaw / 2.0), 0.0, 0.0, np.sin(source_yaw / 2.0)),)
        ),
        sole_position_world=np.asarray(
            ((((0.4, -0.3, 0.0), (0.4, -0.1, 0.0))),)
        ).reshape(1, 2, 3),
    )
    target_yaw = np.pi / 2.0
    alignment = _reset_alignment(
        source,
        0,
        field,
        field,
        (2.0, 1.0),
        target_yaw_rad=target_yaw,
    )
    np.testing.assert_allclose(
        alignment.yaw_offset_rad,
        target_yaw - source_yaw,
    )
    cosine = np.cos(alignment.yaw_offset_rad)
    sine = np.sin(alignment.yaw_offset_rad)
    source_root = source.root_position_world[0]
    rotated_xy = np.asarray(
        (
            cosine * source_root[0] - sine * source_root[1],
            sine * source_root[0] + cosine * source_root[1],
        )
    )
    np.testing.assert_allclose(
        rotated_xy + np.asarray(alignment.translation_world_xyz[:2]),
        (2.0, 1.0),
    )


class _FlatField:
    def sample(self, points_world_xy: object):
        points = np.asarray(points_world_xy)
        shape = points.shape[:-1]
        height = np.zeros(shape, dtype=np.float64)
        normal = np.zeros((*shape, 3), dtype=np.float64)
        normal[..., 2] = 1.0
        return height, normal, np.ones(shape, dtype=np.bool_)


def test_continuous_matcher_searches_then_advances_transactionally() -> None:
    arrays = build_varying_takara_arrays(frames=100)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_takara_arrays(root / "walk", arrays)
        folder = MotionFolder.load(root)
        base = TorchMotionMatcher.from_motion_folder(
            folder,
            device="cpu",
            config=MatcherConfig(trajectory_model="legacy"),
        )
        rows = base.database.feature_shape[0]
        source_height = torch.full((rows, 2, 4), -0.8)
        source_normal = torch.zeros((rows, 2, 4, 3))
        source_normal[..., 2] = 1.0
        terrain_rows = TerrainRowFeatures(
            future_foot_local_xy=torch.zeros((rows, 2, 4, 2)),
            future_foot_height_from_root=source_height.clone(),
            source_height_from_root=source_height,
            source_normal_local=source_normal,
            source_hit=torch.ones((rows, 2, 4), dtype=torch.bool),
            source_path_height_from_root=torch.full((rows, 10), -0.8),
            source_path_normal_local=torch.nn.functional.pad(
                torch.ones((rows, 10, 1)), (2, 0)
            ),
            source_path_hit=torch.ones((rows, 10), dtype=torch.bool),
            contact_phase=torch.zeros(rows, dtype=torch.int8),
            contact_future=torch.zeros((rows, 2, 4)),
            joint_position=torch.zeros((rows, 29)),
            joint_velocity=torch.zeros((rows, 29)),
            maximum_future_joint_step_rad=torch.zeros(rows),
            maximum_future_root_step_m=torch.zeros(rows),
            maximum_future_root_yaw_step_rad=torch.zeros(rows),
        )
        canonical = SimpleNamespace(
            body_position_world=arrays["body_pos_w"],
            body_quaternion_world_wxyz=arrays["body_quat_w"],
            root_position_world=arrays["body_pos_w"][:, 0],
            root_quaternion_world_wxyz=arrays["body_quat_w"][:, 0],
            joint_position=arrays["joint_pos"],
            sole_position_world=arrays["body_pos_w"][:, (18, 19)],
            heel_position_world=arrays["body_pos_w"][:, (18, 19)],
            toe_position_world=arrays["body_pos_w"][:, (18, 19)],
            contact=np.zeros((len(arrays["joint_pos"]), 2), dtype=np.float32),
        )
        library = CanonicalTerrainLibrary(
            corpus_root=root,
            clip_ids=("synthetic/flat",),
            canonical_clips=(canonical,),
            height_fields=(_FlatField(),),
            folder=folder,
        )
        matcher = ContinuousTerrainMotionMatcher(
            library,
            base,
            _FlatField(),
            terrain_rows=terrain_rows,
            search_config=GenericTerrainSearchConfig(preselection_count=32),
            runtime_config=ContinuousTerrainRuntimeConfig(vertical_adaptation=False),
        )
        reset = matcher.reset()
        prepared = matcher.prepare_step((0.4, 0.0), 0.0)
        assert reset.diagnostics.sequence == 0
        assert matcher.last_search_event is None
        result = matcher.commit(prepared)
        assert result.diagnostics.sequence == 1
        assert matcher.last_search_event is not None
        assert matcher.last_search_event.sequence == 1
        assert torch.linalg.vector_norm(
            result.root_position_world - reset.root_position_world
        ) < 0.06
