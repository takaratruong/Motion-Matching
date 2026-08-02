from __future__ import annotations

from dataclasses import fields, replace
import unittest
from pathlib import Path

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.canonical import (
    CanonicalClip,
    CanonicalTerrainMesh,
)
from mm_sonic.terrain_oracle.contact import (
    CONTACT_RECONSTRUCTION_TAG,
    CanonicalMeshQuery,
    ContactConfig,
    SoleGeometry,
    reconstruct_contacts,
)
from mm_sonic.terrain_oracle.math3d import RigidTransform
from tests.python.terrain_oracle_test_utils import synthetic_canonical_clip


MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)
LEFT_ANKLE = 18
RIGHT_ANKLE = 19


def _identity_transform() -> RigidTransform:
    return RigidTransform(
        np.zeros(3, dtype=np.float32),
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )


def _mesh(
    vertices: object,
    faces: object,
    valid_faces: object,
) -> CanonicalTerrainMesh:
    return CanonicalTerrainMesh(
        vertices_local=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        valid_faces=np.asarray(valid_faces, dtype=np.bool_),
        source_asset_sha256="f" * 64,
    )


def _plane_query(*, toe_only: bool = False) -> CanonicalMeshQuery:
    x_min = 0.05 if toe_only else -2.0
    plane = _mesh(
        (
            (x_min, -2.0, 0.0),
            (2.0, -2.0, 0.0),
            (2.0, 2.0, 0.0),
            (x_min, 2.0, 0.0),
        ),
        ((0, 1, 2), (0, 2, 3)),
        (True, True),
    )
    return CanonicalMeshQuery(plane, _identity_transform())


def _tread_with_nearby_riser_query() -> CanonicalMeshQuery:
    terrain = _mesh(
        (
            (-2.0, -2.0, 0.0),
            (2.0, -2.0, 0.0),
            (2.0, 2.0, 0.0),
            (-2.0, 2.0, 0.0),
            (0.101, -2.0, 0.0),
            (0.101, 2.0, 0.0),
            (0.101, 2.0, 0.2),
            (0.101, -2.0, 0.2),
        ),
        ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)),
        (True, True, True, True),
    )
    return CanonicalMeshQuery(terrain, _identity_transform())


def _analytic_geometry() -> SoleGeometry:
    return SoleGeometry(
        body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        corner_positions_body=np.array(
            [
                (
                    (-0.10, 0.04, 0.0),
                    (-0.10, -0.04, 0.0),
                    (0.10, 0.04, 0.0),
                    (0.10, -0.04, 0.0),
                ),
                (
                    (-0.10, 0.04, 0.0),
                    (-0.10, -0.04, 0.0),
                    (0.10, 0.04, 0.0),
                    (0.10, -0.04, 0.0),
                ),
            ],
            dtype=np.float32,
        ),
    )


def _config() -> ContactConfig:
    return ContactConfig(
        geometry=_analytic_geometry(),
        enter_distance_m=0.010,
        leave_distance_m=0.020,
        enter_tangential_speed_m_s=0.10,
        leave_tangential_speed_m_s=0.20,
        enter_angular_speed_rad_s=0.50,
        leave_angular_speed_rad_s=1.00,
        minimum_surface_normal_z=0.50,
    )


def _analytic_clip(
    *,
    sole_height: float | np.ndarray,
    foot_speed: float = 0.0,
    angular_speed: float = 0.0,
    frames: int = 4,
) -> CanonicalClip:
    clip = synthetic_canonical_clip(frames=frames)
    height = np.broadcast_to(
        np.asarray(sole_height, dtype=np.float32), (frames,)
    )
    body_position = np.array(clip.body_position_world, copy=True)
    body_position[:, LEFT_ANKLE, 1] = 0.20
    body_position[:, RIGHT_ANKLE, 1] = -0.20
    body_position[:, (LEFT_ANKLE, RIGHT_ANKLE), 2] = height[:, None]
    body_velocity = np.array(clip.body_linear_velocity_world, copy=True)
    body_velocity[:, (LEFT_ANKLE, RIGHT_ANKLE), 0] = foot_speed
    body_angular_velocity = np.array(
        clip.body_angular_velocity_world, copy=True
    )
    body_angular_velocity[:, (LEFT_ANKLE, RIGHT_ANKLE), 2] = angular_speed
    return replace(
        clip,
        body_position_world=body_position,
        body_linear_velocity_world=body_velocity,
        body_angular_velocity_world=body_angular_velocity,
        action_tags=("walk", "custom-source-tag", "contacts-unreconstructed"),
    )


@unittest.skipUnless(MODEL.is_file(), "real G1 MuJoCo model is unavailable")
class SoleGeometryTests(unittest.TestCase):
    def test_exact_model_selects_only_four_collision_spheres_per_ankle(self):
        """Catches selecting the ankle visual mesh or unrelated limb geoms."""

        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MODEL))

        geometry = SoleGeometry.from_model(model)

        expected = np.array(
            (
                (-0.05, -0.025, -0.035),
                (-0.05, 0.025, -0.035),
                (0.12, -0.030, -0.035),
                (0.12, 0.030, -0.035),
            ),
            dtype=np.float32,
        )
        self.assertEqual(
            geometry.body_names,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
        )
        np.testing.assert_allclose(
            geometry.corner_positions_body[0], expected, atol=1.0e-7, rtol=0.0
        )
        np.testing.assert_allclose(
            geometry.corner_positions_body[1], expected, atol=1.0e-7, rtol=0.0
        )
        np.testing.assert_allclose(
            geometry.heel_positions_body[:, 0], -0.05, atol=1.0e-7
        )
        np.testing.assert_allclose(
            geometry.toe_positions_body[:, 0], 0.12, atol=1.0e-7
        )

    def test_model_geometry_must_remain_descended_from_named_ankle(self):
        """Catches unchecked geom indices surviving a model-layout change."""

        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MODEL))
        left_ankle = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        left_collision_geoms = np.flatnonzero(
            (model.geom_bodyid == left_ankle)
            & (model.geom_contype != 0)
            & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
        )
        model.geom_bodyid[left_collision_geoms[0]] = 1

        with self.assertRaisesRegex(ContractError, "left_ankle_roll_link.*four"):
            SoleGeometry.from_model(model)


class CanonicalMeshQueryTests(unittest.TestCase):
    def test_rejects_invalid_indices_and_valid_face_shape_at_query_boundary(self):
        """Catches trusting corrupt mesh arrays merely because shapes look close."""

        valid = _mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ((0, 1, 2),),
            (True,),
        )
        bad_indices = object.__new__(CanonicalTerrainMesh)
        for name in (
            "vertices_local",
            "valid_faces",
            "source_asset_sha256",
        ):
            object.__setattr__(bad_indices, name, getattr(valid, name))
        object.__setattr__(
            bad_indices, "faces", np.array(((0, 1, 3),), dtype=np.int32)
        )
        with self.assertRaisesRegex(ContractError, "indices"):
            CanonicalMeshQuery(bad_indices, _identity_transform())

        bad_mask = object.__new__(CanonicalTerrainMesh)
        for name in (
            "vertices_local",
            "faces",
            "source_asset_sha256",
        ):
            object.__setattr__(bad_mask, name, getattr(valid, name))
        object.__setattr__(
            bad_mask, "valid_faces", np.array((True, False), dtype=np.bool_)
        )
        with self.assertRaisesRegex(ContractError, "valid_faces"):
            CanonicalMeshQuery(bad_mask, _identity_transform())

    def test_transform_is_applied_once_and_queries_exact_triangle_geometry(self):
        """Catches double-applying terrain placement or using a height-field proxy."""

        local_mesh = _mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ((0, 1, 2),),
            (True,),
        )
        transform = RigidTransform(
            np.array((1.0, 2.0, 3.0), dtype=np.float32),
            np.array(
                (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)),
                dtype=np.float32,
            ),
        )
        query = CanonicalMeshQuery(local_mesh, transform)
        points_world = transform.apply_points(
            np.array(((0.25, 0.25, 0.40), (1.0, 1.0, 0.0)))
        )

        result = query.query(points_world)

        expected_closest = transform.apply_points(
            np.array(((0.25, 0.25, 0.0), (0.5, 0.5, 0.0)))
        )
        np.testing.assert_allclose(
            result.closest_point_world,
            expected_closest,
            atol=1.0e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.distance_m,
            (0.4, np.sqrt(0.5)),
            atol=1.0e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.surface_normal_world,
            ((0, 0, 1), (0, 0, 1)),
            atol=1.0e-6,
            rtol=0.0,
        )
        self.assertAlmostEqual(result.downward_ray_distance_m[0], 0.4, places=6)
        self.assertTrue(np.isinf(result.downward_ray_distance_m[1]))

    def test_invalid_faces_are_excluded_from_ray_and_closest_point_queries(self):
        """Catches an invalid triangle winning a nearest-surface query."""

        mesh = _mesh(
            (
                (-1, -1, 0),
                (1, -1, 0),
                (0, 1, 0),
                (-1, -1, 0.5),
                (1, -1, 0.5),
                (0, 1, 0.5),
            ),
            ((0, 1, 2), (3, 4, 5)),
            (True, False),
        )

        result = CanonicalMeshQuery(mesh, _identity_transform()).query(
            np.array(((0.0, 0.0, 0.6),), dtype=np.float32)
        )

        self.assertAlmostEqual(result.distance_m[0], 0.6, places=6)
        self.assertAlmostEqual(result.closest_point_world[0, 2], 0.0, places=6)
        self.assertAlmostEqual(result.downward_ray_distance_m[0], 0.6, places=6)


class ContactReconstructionTests(unittest.TestCase):
    def test_grounded_complete_sole_contacts_but_hovering_or_moving_does_not(self):
        """Catches inferring support from foot height without velocity and geometry."""

        grounded = reconstruct_contacts(
            _analytic_clip(sole_height=0.0),
            _plane_query(),
            _config(),
        )
        hovering = reconstruct_contacts(
            _analytic_clip(sole_height=0.04),
            _plane_query(),
            _config(),
        )
        translating = reconstruct_contacts(
            _analytic_clip(sole_height=0.0, foot_speed=0.5),
            _plane_query(),
            _config(),
        )
        rotating = reconstruct_contacts(
            _analytic_clip(sole_height=0.0, angular_speed=2.0),
            _plane_query(),
            _config(),
        )

        self.assertTrue(np.all(grounded.contact))
        self.assertFalse(np.any(hovering.contact))
        self.assertFalse(np.any(translating.contact))
        self.assertFalse(np.any(rotating.contact))
        self.assertTrue(np.all(np.isfinite(grounded.contact_confidence)))
        self.assertTrue(
            np.all(
                (grounded.contact_confidence >= 0.0)
                & (grounded.contact_confidence <= 1.0)
            )
        )

    def test_enter_leave_hysteresis_prevents_threshold_chatter(self):
        """Catches using one distance threshold for both contact transitions."""

        clip = _analytic_clip(
            sole_height=np.array((0.005, 0.015, 0.025, 0.015)),
            frames=4,
        )

        reconstruction = reconstruct_contacts(
            clip, _plane_query(), _config()
        )

        np.testing.assert_array_equal(
            reconstruction.contact[:, 0], (True, True, False, False)
        )
        np.testing.assert_array_equal(
            reconstruction.contact[:, 1], (True, True, False, False)
        )

    def test_valid_tread_ray_wins_over_nearer_vertical_riser(self):
        """Catches a closer riser normal hiding valid support 8 mm below."""

        reconstruction = reconstruct_contacts(
            _analytic_clip(sole_height=0.008),
            _tread_with_nearby_riser_query(),
            _config(),
        )

        self.assertTrue(np.all(reconstruction.contact))
        np.testing.assert_allclose(
            reconstruction.contact_confidence,
            np.full((4, 2), 0.6, dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_tread_ray_normal_drives_tangential_speed_near_riser(self):
        """Catches selecting tread distance but projecting speed on the riser."""

        reconstruction = reconstruct_contacts(
            _analytic_clip(sole_height=0.008, foot_speed=0.15),
            _tread_with_nearby_riser_query(),
            _config(),
        )

        self.assertFalse(np.any(reconstruction.contact))
        np.testing.assert_allclose(
            reconstruction.contact_confidence,
            np.full((4, 2), 0.15, dtype=np.float32),
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_toe_only_surface_is_partial_confidence_not_full_sole_contact(self):
        """Catches a toe or riser probe being promoted to complete-sole support."""

        reconstruction = reconstruct_contacts(
            _analytic_clip(sole_height=0.0),
            _plane_query(toe_only=True),
            _config(),
        )

        self.assertFalse(np.any(reconstruction.contact))
        self.assertTrue(np.all(reconstruction.contact_confidence > 0.0))
        self.assertTrue(np.all(reconstruction.contact_confidence < 1.0))

    def test_reconstruction_requires_an_explicit_surface_query(self):
        """Catches silently inventing a world z=0 terrain plane."""

        with self.assertRaisesRegex(ContractError, "surface query"):
            reconstruct_contacts(
                _analytic_clip(sole_height=0.0), None, _config()
            )

    def test_apply_replaces_only_contact_geometry_and_provenance(self):
        """Catches dropping source/action metadata while replacing placeholders."""

        clip = _analytic_clip(sole_height=0.0)
        reconstruction = reconstruct_contacts(
            clip, _plane_query(), _config()
        )

        applied = reconstruction.apply(clip)

        reconstructed_fields = {
            "sole_position_world",
            "sole_quaternion_world_wxyz",
            "heel_position_world",
            "toe_position_world",
            "contact",
            "contact_confidence",
            "action_tags",
        }
        for field in fields(CanonicalClip):
            if field.name in reconstructed_fields:
                continue
            before = getattr(clip, field.name)
            after = getattr(applied, field.name)
            with self.subTest(field=field.name):
                if isinstance(before, np.ndarray):
                    np.testing.assert_array_equal(after, before)
                else:
                    self.assertEqual(after, before)
        self.assertEqual(
            applied.action_tags,
            ("walk", "custom-source-tag", CONTACT_RECONSTRUCTION_TAG),
        )
        self.assertNotIn("contacts-unreconstructed", applied.action_tags)
        self.assertTrue(np.all(applied.contact))
        self.assertTrue(np.all(applied.contact_confidence > 0.0))
        self.assertFalse(applied.sole_position_world.flags.writeable)
        applied.validate()


if __name__ == "__main__":
    unittest.main()
