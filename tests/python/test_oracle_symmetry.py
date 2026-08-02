from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.offline_corpus import (
    BODY_MIRROR_PERMUTATION,
    JOINT_MIRROR_PERMUTATION,
    JOINT_MIRROR_SIGNS,
)
from mm_sonic.terrain_oracle.canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CanonicalTerrainMesh,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
    TerrainBinding,
)
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.storage import write_mesh
from mm_sonic.terrain_oracle.symmetry import (
    assert_mirror_involution,
    mirror_clip,
    mirror_terrain,
)


_POLAR_SIGNS = np.array((1.0, -1.0, 1.0), dtype=np.float32)
_AXIAL_SIGNS = np.array((-1.0, 1.0, -1.0), dtype=np.float32)
_QUATERNION_SIGNS = np.array((1.0, -1.0, 1.0, -1.0), dtype=np.float32)


def _normalized_quaternions(leading_shape: tuple[int, ...], start: int) -> np.ndarray:
    count = int(np.prod(leading_shape))
    values = np.arange(start, start + count * 4, dtype=np.float32).reshape(
        (*leading_shape, 4)
    )
    values[..., 1::2] *= -1.0
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def _asymmetric_clip(*, mesh_sha256: str = "b" * 64) -> CanonicalClip:
    frames_count = 4
    frame = np.arange(frames_count, dtype=np.float32)
    root_position = np.stack(
        (0.2 + frame, -0.3 - 0.1 * frame, 0.8 + 0.05 * frame), axis=1
    )
    root_linear_velocity = np.stack(
        (1.0 + frame, -2.0 - frame, 3.0 + 0.5 * frame), axis=1
    )
    root_angular_velocity = np.stack(
        (-0.2 - frame, 0.3 + 2.0 * frame, -0.4 - 3.0 * frame), axis=1
    )
    joint_position = (
        np.arange(frames_count * 29, dtype=np.float32).reshape(frames_count, 29)
        / 17.0
        - 2.0
    )
    joint_velocity = joint_position * np.float32(-1.7) + np.float32(0.11)
    body_position = (
        np.arange(frames_count * 30 * 3, dtype=np.float32).reshape(
            frames_count, 30, 3
        )
        / 31.0
        - 4.0
    )
    body_linear_velocity = body_position * np.float32(-0.4) + np.float32(0.3)
    body_angular_velocity = body_position * np.float32(0.6) - np.float32(0.8)
    sole_position = np.array(
        [
            [[0.1, 0.2, 0.3], [1.1, -1.2, 1.3]],
            [[2.1, 2.2, 2.3], [3.1, -3.2, 3.3]],
            [[4.1, 4.2, 4.3], [5.1, -5.2, 5.3]],
            [[6.1, 6.2, 6.3], [7.1, -7.2, 7.3]],
        ],
        dtype=np.float32,
    )
    terrain = TerrainBinding(
        asset_path="/read-only/asymmetric-stair.usd",
        asset_size_bytes=8192,
        asset_sha256="a" * 64,
        asset_license_id="CC-BY-4.0",
        mesh_sha256=mesh_sha256,
        world_from_terrain=RigidTransform(
            translation_world=np.array((1.2, -2.3, 3.4), dtype=np.float32),
            quaternion_world_from_local_wxyz=np.array(
                (0.8, 0.2, -0.3, 0.45), dtype=np.float32
            )
            / np.linalg.norm((0.8, 0.2, -0.3, 0.45)),
        ),
        validity_mask_path="/read-only/asymmetric-stair.valid.npy",
    )
    clip = CanonicalClip(
        clip_id="asymmetric-stair-ascent",
        fps=50.0,
        source=SourceIdentity(
            source_format="synthetic-full-contract",
            source_path="fixture://asymmetric-stair-ascent",
            source_size_bytes=123456,
            source_sha256="c" * 64,
            source_license_id="CC0-1.0",
            coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
            quaternion_convention="wxyz",
            pose_origin=CLEAN_POSE_ORIGIN,
        ),
        joint_names=ISAACLAB_JOINT_NAMES,
        body_names=ISAACLAB_BODY_NAMES,
        root_position_world=root_position,
        root_quaternion_world_wxyz=_normalized_quaternions(
            (frames_count,), 1
        ),
        joint_position=joint_position,
        root_linear_velocity_world=root_linear_velocity,
        root_angular_velocity_world=root_angular_velocity,
        joint_velocity=joint_velocity,
        body_position_world=body_position,
        body_quaternion_world_wxyz=_normalized_quaternions(
            (frames_count, 30), 101
        ),
        body_linear_velocity_world=body_linear_velocity,
        body_angular_velocity_world=body_angular_velocity,
        sole_position_world=sole_position,
        sole_quaternion_world_wxyz=_normalized_quaternions(
            (frames_count, 2), 1001
        ),
        heel_position_world=sole_position + np.array(
            (-0.13, 0.07, -0.02), dtype=np.float32
        ),
        toe_position_world=sole_position + np.array(
            (0.21, -0.09, 0.03), dtype=np.float32
        ),
        contact=np.array(
            ((0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)),
            dtype=np.float32,
        ),
        contact_confidence=np.array(
            ((0.91, 0.11), (0.82, 0.22), (0.73, 0.33), (0.64, 0.44)),
            dtype=np.float32,
        ),
        commands=CommandTrack(
            observed_travel_stick_xy=np.array(
                ((0.1, 0.2), (-0.3, 0.4), (0.5, -0.6), (-0.7, -0.8)),
                dtype=np.float32,
            ),
            observed_facing_stick_xy=np.array(
                ((-0.2, 0.3), (0.4, -0.5), (-0.6, -0.7), (0.8, 0.9)),
                dtype=np.float32,
            ),
            observed_mask=np.array((True, False, True, False)),
            inferred_velocity_local_xy=np.array(
                ((1.1, -1.2), (2.1, 2.2), (-3.1, 3.2), (-4.1, -4.2)),
                dtype=np.float32,
            ),
            inferred_facing_local_xy=np.array(
                ((0.6, 0.8), (-0.8, 0.6), (-0.6, -0.8), (0.8, -0.6)),
                dtype=np.float32,
            ),
            inferred_yaw_rate_rad_s=np.array(
                (0.25, -0.5, 0.75, -1.0), dtype=np.float32
            ),
        ),
        terrain=terrain,
        action_tags=("terrain", "stairs", "up", "authored-tag"),
    )
    clip.validate()
    return clip


def _asymmetric_mesh() -> CanonicalTerrainMesh:
    return CanonicalTerrainMesh(
        vertices_local=np.array(
            (
                (-0.4, -0.7, 0.0),
                (1.2, -0.6, 0.1),
                (-0.3, 0.9, 0.2),
                (1.1, 0.8, 0.35),
                (0.2, 0.1, -0.4),
            ),
            dtype=np.float32,
        ),
        faces=np.array(((0, 1, 2), (1, 3, 2), (0, 4, 1)), dtype=np.int32),
        valid_faces=np.array((True, True, False)),
        source_asset_sha256="a" * 64,
    )


def _assert_values_exact(
    test: unittest.TestCase, actual: object, expected: object, path: str
) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif is_dataclass(expected):
        test.assertEqual(type(actual), type(expected), path)
        for field in fields(expected):
            _assert_values_exact(
                test,
                getattr(actual, field.name),
                getattr(expected, field.name),
                f"{path}.{field.name}",
            )
    else:
        test.assertEqual(actual, expected, path)


def _assert_arrays_read_only(
    test: unittest.TestCase, value: object, path: str
) -> None:
    if isinstance(value, np.ndarray):
        test.assertTrue(value.flags.c_contiguous, path)
        test.assertFalse(value.flags.writeable, path)
    elif is_dataclass(value):
        for field in fields(value):
            _assert_arrays_read_only(
                test,
                getattr(value, field.name),
                f"{path}.{field.name}",
            )


def _rotation_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


class CanonicalClipSymmetryTests(unittest.TestCase):
    def test_mirror_twice_restores_every_history_command_contact_and_provenance_field(
        self,
    ):
        """Catches omission of any history field or reflection on a wrong axis."""

        original = _asymmetric_clip()
        mirrored = mirror_clip(original)
        restored = mirror_clip(mirrored)

        _assert_values_exact(self, restored, original, "clip")
        self.assertEqual(mirrored.clip_id, "asymmetric-stair-ascent__mirror")
        self.assertEqual(mirrored.mirror_of, original.clip_id)
        self.assertEqual(mirrored.source, original.source)
        self.assertEqual(mirrored.action_tags, original.action_tags)
        np.testing.assert_array_equal(
            mirrored.root_position_world,
            original.root_position_world * _POLAR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.root_linear_velocity_world,
            original.root_linear_velocity_world * _POLAR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.root_angular_velocity_world,
            original.root_angular_velocity_world * _AXIAL_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.root_quaternion_world_wxyz,
            original.root_quaternion_world_wxyz * _QUATERNION_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.joint_position,
            original.joint_position[:, JOINT_MIRROR_PERMUTATION]
            * JOINT_MIRROR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.joint_velocity,
            original.joint_velocity[:, JOINT_MIRROR_PERMUTATION]
            * JOINT_MIRROR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.body_position_world,
            original.body_position_world[:, BODY_MIRROR_PERMUTATION]
            * _POLAR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.body_linear_velocity_world,
            original.body_linear_velocity_world[:, BODY_MIRROR_PERMUTATION]
            * _POLAR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.body_angular_velocity_world,
            original.body_angular_velocity_world[:, BODY_MIRROR_PERMUTATION]
            * _AXIAL_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.body_quaternion_world_wxyz,
            original.body_quaternion_world_wxyz[:, BODY_MIRROR_PERMUTATION]
            * _QUATERNION_SIGNS,
        )
        for field_name in (
            "sole_position_world",
            "heel_position_world",
            "toe_position_world",
        ):
            np.testing.assert_array_equal(
                getattr(mirrored, field_name),
                getattr(original, field_name)[:, (1, 0)] * _POLAR_SIGNS,
            )
        np.testing.assert_array_equal(
            mirrored.sole_quaternion_world_wxyz,
            original.sole_quaternion_world_wxyz[:, (1, 0)]
            * _QUATERNION_SIGNS,
        )
        np.testing.assert_array_equal(mirrored.contact, original.contact[:, (1, 0)])
        np.testing.assert_array_equal(
            mirrored.contact_confidence,
            original.contact_confidence[:, (1, 0)],
        )
        np.testing.assert_array_equal(
            mirrored.commands.observed_travel_stick_xy,
            original.commands.observed_travel_stick_xy
            * np.array((-1.0, 1.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mirrored.commands.observed_facing_stick_xy,
            original.commands.observed_facing_stick_xy
            * np.array((-1.0, 1.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mirrored.commands.observed_mask, original.commands.observed_mask
        )
        np.testing.assert_array_equal(
            mirrored.commands.inferred_velocity_local_xy,
            original.commands.inferred_velocity_local_xy
            * np.array((1.0, -1.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mirrored.commands.inferred_facing_local_xy,
            original.commands.inferred_facing_local_xy
            * np.array((1.0, -1.0), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            mirrored.commands.inferred_yaw_rate_rad_s,
            -original.commands.inferred_yaw_rate_rad_s,
        )
        self.assertIsNotNone(mirrored.terrain)
        np.testing.assert_array_equal(
            mirrored.terrain.world_from_terrain.translation_world,
            original.terrain.world_from_terrain.translation_world * _POLAR_SIGNS,
        )
        np.testing.assert_array_equal(
            mirrored.terrain.world_from_terrain.quaternion_world_from_local_wxyz,
            original.terrain.world_from_terrain.quaternion_world_from_local_wxyz
            * _QUATERNION_SIGNS,
        )
        self.assertEqual(mirrored.terrain.mesh_sha256, original.terrain.mesh_sha256)
        assert_mirror_involution(original, mirrored)
        mirrored.validate()
        _assert_arrays_read_only(self, mirrored, "mirrored")
        _assert_arrays_read_only(self, restored, "restored")

    def test_root_never_uses_the_body_left_right_permutation(self):
        """Catches root pose accidentally being replaced by a permuted body pose."""

        original = _asymmetric_clip()
        mirrored = mirror_clip(original)

        np.testing.assert_array_equal(
            mirrored.root_position_world[:, 0], original.root_position_world[:, 0]
        )
        self.assertFalse(
            np.array_equal(
                mirrored.root_position_world,
                mirrored.body_position_world[:, 0],
            )
        )

    def test_rejects_wrong_types_orders_and_ambiguous_pretagged_identity(self):
        """Catches reflection proceeding under an unknown layout or mirror lineage."""

        original = _asymmetric_clip()
        with self.assertRaisesRegex(ContractError, "CanonicalClip"):
            mirror_clip(object())
        with self.assertRaisesRegex(ContractError, "joint_names"):
            mirror_clip(
                replace(
                    original,
                    joint_names=(
                        original.joint_names[1],
                        original.joint_names[0],
                        *original.joint_names[2:],
                    ),
                )
            )
        with self.assertRaisesRegex(ContractError, "body_names"):
            mirror_clip(
                replace(
                    original,
                    body_names=(
                        original.body_names[1],
                        original.body_names[0],
                        *original.body_names[2:],
                    ),
                )
            )
        with self.assertRaisesRegex(ContractError, "mirror_of"):
            mirror_clip(replace(original, mirror_of="unrelated-source"))
        with self.assertRaisesRegex(ContractError, "cyclic"):
            mirror_clip(replace(original, mirror_of=original.clip_id))

    def test_assert_mirror_involution_reports_each_perturbed_field(self):
        """Catches a shallow involution check that ignores history or provenance."""

        original = _asymmetric_clip()
        mirrored = mirror_clip(original)
        mutations = {
            "joint_position": replace(
                mirrored,
                joint_position=np.asarray(mirrored.joint_position)
                + np.eye(4, 29, dtype=np.float32),
            ),
            "commands.inferred_yaw_rate_rad_s": replace(
                mirrored,
                commands=replace(
                    mirrored.commands,
                    inferred_yaw_rate_rad_s=(
                        np.asarray(mirrored.commands.inferred_yaw_rate_rad_s)
                        + np.array((0.0, 0.0, 0.125, 0.0), dtype=np.float32)
                    ),
                ),
            ),
            "contact": replace(
                mirrored,
                contact=np.asarray(mirrored.contact)
                + np.array(
                    ((0.01, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
                    dtype=np.float32,
                ),
            ),
            "source": replace(
                mirrored,
                source=replace(mirrored.source, source_path="fixture://forged"),
            ),
            "action_tags": replace(
                mirrored, action_tags=(*mirrored.action_tags, "forged")
            ),
            "terrain.world_from_terrain.translation_world": replace(
                mirrored,
                terrain=replace(
                    mirrored.terrain,
                    world_from_terrain=RigidTransform(
                        np.asarray(
                            mirrored.terrain.world_from_terrain.translation_world
                        )
                        + np.array((0.0, 0.1, 0.0), dtype=np.float32),
                        mirrored.terrain.world_from_terrain
                        .quaternion_world_from_local_wxyz,
                    ),
                ),
            ),
        }
        for field_name, perturbed in mutations.items():
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(
                    ContractError, field_name.replace(".", r"\.")
                ):
                    assert_mirror_involution(original, perturbed)


class TerrainSymmetryTests(unittest.TestCase):
    def _binding_for(self, mesh: CanonicalTerrainMesh, root: Path) -> TerrainBinding:
        record = write_mesh(root / "meshes", mesh)
        return TerrainBinding(
            asset_path="/read-only/asymmetric-stair.usd",
            asset_size_bytes=8192,
            asset_sha256=mesh.source_asset_sha256,
            asset_license_id="CC-BY-4.0",
            mesh_sha256=record.sha256,
            world_from_terrain=RigidTransform(
                translation_world=np.array((1.2, -2.3, 3.4), dtype=np.float32),
                quaternion_world_from_local_wxyz=np.array(
                    (0.8, 0.2, -0.3, 0.45), dtype=np.float32
                )
                / np.linalg.norm((0.8, 0.2, -0.3, 0.45)),
            ),
            validity_mask_path="/read-only/asymmetric-stair.valid.npy",
        )

    def test_mirrors_mesh_and_proper_transform_once_with_reversed_winding(self):
        """Catches double reflection, an improper rotation, or stale face winding."""

        mesh = _asymmetric_mesh()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = self._binding_for(mesh, root)
            mirrored_binding, mirrored_mesh = mirror_terrain(binding, mesh)

            world = binding.world_from_terrain.apply_points(mesh.vertices_local)
            mirrored_world = mirrored_binding.world_from_terrain.apply_points(
                mirrored_mesh.vertices_local
            )
            np.testing.assert_allclose(
                mirrored_world, world * _POLAR_SIGNS, rtol=0.0, atol=2.0e-6
            )
            np.testing.assert_array_equal(
                mirrored_mesh.vertices_local,
                mesh.vertices_local * _POLAR_SIGNS,
            )
            np.testing.assert_array_equal(
                mirrored_mesh.faces, mesh.faces[:, (0, 2, 1)]
            )
            np.testing.assert_array_equal(
                mirrored_mesh.valid_faces, mesh.valid_faces
            )
            self.assertEqual(
                mirrored_mesh.source_asset_sha256, mesh.source_asset_sha256
            )
            self.assertEqual(mirrored_binding.asset_path, binding.asset_path)
            self.assertEqual(mirrored_binding.asset_sha256, binding.asset_sha256)
            self.assertEqual(
                mirrored_binding.asset_license_id, binding.asset_license_id
            )
            mirrored_record = write_mesh(root / "mirrored-meshes", mirrored_mesh)
            self.assertEqual(mirrored_binding.mesh_sha256, mirrored_record.sha256)

            original_rotation = _rotation_matrix_wxyz(
                binding.world_from_terrain.quaternion_world_from_local_wxyz
            )
            mirrored_rotation = _rotation_matrix_wxyz(
                mirrored_binding.world_from_terrain.quaternion_world_from_local_wxyz
            )
            reflection = np.diag((1.0, -1.0, 1.0))
            np.testing.assert_allclose(
                mirrored_rotation,
                reflection @ original_rotation @ reflection,
                rtol=0.0,
                atol=2.0e-6,
            )
            original_triangle = world[mesh.faces[0]]
            mirrored_triangle = mirrored_world[mirrored_mesh.faces[0]]
            original_normal = np.cross(
                original_triangle[1] - original_triangle[0],
                original_triangle[2] - original_triangle[0],
            )
            mirrored_normal = np.cross(
                mirrored_triangle[1] - mirrored_triangle[0],
                mirrored_triangle[2] - mirrored_triangle[0],
            )
            original_normal /= np.linalg.norm(original_normal)
            mirrored_normal /= np.linalg.norm(mirrored_normal)
            np.testing.assert_allclose(
                mirrored_normal, original_normal * _POLAR_SIGNS, atol=2.0e-6
            )
            self.assertGreater(original_normal[2], 0.0)
            self.assertGreater(mirrored_normal[2], 0.0)

    def test_double_terrain_mirror_restores_binding_mesh_and_digest_exactly(self):
        """Catches a mirrored digest or transform that is not an exact involution."""

        mesh = _asymmetric_mesh()
        with tempfile.TemporaryDirectory() as temporary:
            binding = self._binding_for(mesh, Path(temporary))
            mirrored_binding, mirrored_mesh = mirror_terrain(binding, mesh)
            restored_binding, restored_mesh = mirror_terrain(
                mirrored_binding, mirrored_mesh
            )

        _assert_values_exact(self, restored_binding, binding, "binding")
        _assert_values_exact(self, restored_mesh, mesh, "mesh")
        self.assertFalse(mirrored_mesh.vertices_local.flags.writeable)
        self.assertFalse(mirrored_mesh.faces.flags.writeable)
        self.assertFalse(mirrored_mesh.valid_faces.flags.writeable)

    def test_clip_and_terrain_pairing_is_publishable_and_double_pairs_exactly(self):
        """Catches publishing the intermediate clip with its stale pre-mirror digest."""

        mesh = _asymmetric_mesh()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binding = self._binding_for(mesh, root)
            original = replace(_asymmetric_clip(), terrain=binding)

            intermediate = mirror_clip(original)
            self.assertEqual(intermediate.terrain.mesh_sha256, binding.mesh_sha256)
            mirrored_binding, mirrored_mesh = mirror_terrain(binding, mesh)
            mirrored = replace(intermediate, terrain=mirrored_binding)
            mirrored.validate()
            mirrored_record = write_mesh(root / "paired-mirrored", mirrored_mesh)
            self.assertEqual(mirrored.terrain.mesh_sha256, mirrored_record.sha256)
            np.testing.assert_allclose(
                mirrored.terrain.world_from_terrain.apply_points(
                    mirrored_mesh.vertices_local
                ),
                binding.world_from_terrain.apply_points(mesh.vertices_local)
                * _POLAR_SIGNS,
                rtol=0.0,
                atol=2.0e-6,
            )

            restored_intermediate = mirror_clip(mirrored)
            restored_binding, restored_mesh = mirror_terrain(
                mirrored_binding, mirrored_mesh
            )
            restored = replace(
                restored_intermediate, terrain=restored_binding
            )

        _assert_values_exact(self, restored, original, "clip")
        _assert_values_exact(self, restored_binding, binding, "binding")
        _assert_values_exact(self, restored_mesh, mesh, "mesh")

    def test_rejects_wrong_types_mismatched_source_and_stale_mesh_digest(self):
        """Catches pairing a terrain binding with unrelated or unhashed geometry."""

        mesh = _asymmetric_mesh()
        with tempfile.TemporaryDirectory() as temporary:
            binding = self._binding_for(mesh, Path(temporary))
        with self.assertRaisesRegex(ContractError, "TerrainBinding"):
            mirror_terrain(object(), mesh)
        with self.assertRaisesRegex(ContractError, "CanonicalTerrainMesh"):
            mirror_terrain(binding, object())
        with self.assertRaisesRegex(ContractError, "source asset"):
            mirror_terrain(
                binding,
                replace(mesh, source_asset_sha256="d" * 64),
            )
        with self.assertRaisesRegex(ContractError, "mesh_sha256"):
            mirror_terrain(replace(binding, mesh_sha256="e" * 64), mesh)
        with self.assertRaisesRegex(ContractError, "CanonicalClip"):
            assert_mirror_involution(object(), mirror_clip(_asymmetric_clip()))


if __name__ == "__main__":
    unittest.main()
