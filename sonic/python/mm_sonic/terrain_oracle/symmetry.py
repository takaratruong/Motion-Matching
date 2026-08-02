"""Exact sagittal symmetry for the complete canonical terrain-motion contract."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import hashlib

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.offline_corpus import (
    BODY_MIRROR_PERMUTATION,
    BODY_NAMES,
    JOINT_MIRROR_PERMUTATION,
    JOINT_MIRROR_SIGNS,
    JOINT_NAMES,
)

from .canonical import (
    CanonicalClip,
    CanonicalTerrainMesh,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    TerrainBinding,
)
from .math3d import RigidTransform
from .storage import _canonical_json_bytes, _deterministic_npz_bytes


_MIRROR_SUFFIX = "__mirror"
_LEFT_RIGHT_PERMUTATION = np.asarray((1, 0), dtype=np.int64)
_POLAR_SIGNS = np.asarray((1.0, -1.0, 1.0), dtype=np.float32)
_AXIAL_SIGNS = np.asarray((-1.0, 1.0, -1.0), dtype=np.float32)
_QUATERNION_SIGNS = np.asarray((1.0, -1.0, 1.0, -1.0), dtype=np.float32)
_OBSERVED_STICK_SIGNS = np.asarray((-1.0, 1.0), dtype=np.float32)
_INFERRED_LOCAL_SIGNS = np.asarray((1.0, -1.0), dtype=np.float32)


def _validate_mirror_order(clip: CanonicalClip) -> None:
    if JOINT_NAMES != ISAACLAB_JOINT_NAMES:
        raise ContractError(
            "offline JOINT_NAMES no longer equal the canonical joint order"
        )
    if BODY_NAMES != ISAACLAB_BODY_NAMES:
        raise ContractError(
            "offline BODY_NAMES no longer equal the canonical body order"
        )
    if clip.joint_names != JOINT_NAMES:
        raise ContractError(
            "joint_names must exactly match the validated offline mirror order"
        )
    if clip.body_names != BODY_NAMES:
        raise ContractError(
            "body_names must exactly match the validated offline mirror order"
        )
    if (
        JOINT_MIRROR_PERMUTATION.shape != (len(JOINT_NAMES),)
        or BODY_MIRROR_PERMUTATION.shape != (len(BODY_NAMES),)
        or JOINT_MIRROR_SIGNS.shape != (len(JOINT_NAMES),)
        or not np.array_equal(
            JOINT_MIRROR_PERMUTATION[JOINT_MIRROR_PERMUTATION],
            np.arange(len(JOINT_NAMES)),
        )
        or not np.array_equal(
            BODY_MIRROR_PERMUTATION[BODY_MIRROR_PERMUTATION],
            np.arange(len(BODY_NAMES)),
        )
        or not np.all(JOINT_MIRROR_SIGNS * JOINT_MIRROR_SIGNS == 1.0)
    ):
        raise ContractError("validated offline mirror constants are not involutions")


def _mirrored_identity(clip: CanonicalClip) -> tuple[str, str | None]:
    if clip.mirror_of is None:
        if clip.clip_id.endswith(_MIRROR_SUFFIX):
            raise ContractError(
                f"clip_id suffix {_MIRROR_SUFFIX!r} is reserved for clips "
                "with mirror_of provenance"
            )
        return f"{clip.clip_id}{_MIRROR_SUFFIX}", clip.clip_id
    if clip.mirror_of == clip.clip_id:
        raise ContractError("cyclic mirror_of provenance is not permitted")
    if clip.clip_id != f"{clip.mirror_of}{_MIRROR_SUFFIX}":
        raise ContractError(
            "mirror_of provenance is ambiguous: clip_id must equal "
            f"{clip.mirror_of!r} + {_MIRROR_SUFFIX!r}"
        )
    return clip.mirror_of, None


def _mirror_transform(transform: RigidTransform) -> RigidTransform:
    return RigidTransform(
        translation_world=transform.translation_world * _POLAR_SIGNS,
        quaternion_world_from_local_wxyz=(
            transform.quaternion_world_from_local_wxyz * _QUATERNION_SIGNS
        ),
    )


def mirror_clip(
    clip: CanonicalClip,
    terrain_mesh: CanonicalTerrainMesh | None = None,
) -> CanonicalClip:
    """Reflect every canonical clip field about local/world ``y=0``.

    Observed stick arrays follow their binding convention: channel zero is
    lateral.  Inferred arrays are robot-local ``(forward, lateral)``, so channel
    one is lateral.  Terrain-bound clips require their referenced mesh so the
    returned binding always contains the exact mirrored content digest.
    """

    if not isinstance(clip, CanonicalClip):
        raise ContractError("mirror_clip requires a CanonicalClip")
    clip.validate()
    _validate_mirror_order(clip)
    clip_id, mirror_of = _mirrored_identity(clip)
    if clip.terrain is None:
        if terrain_mesh is not None:
            raise ContractError(
                "cannot supply a terrain mesh for a terrain-unbound clip"
            )
        mirrored_terrain = None
    else:
        if terrain_mesh is None:
            raise ContractError(
                "terrain-bound clip mirroring requires its terrain mesh"
            )
        mirrored_terrain, _mirrored_mesh = mirror_terrain(
            clip.terrain, terrain_mesh
        )
    commands = CommandTrack(
        observed_travel_stick_xy=(
            clip.commands.observed_travel_stick_xy * _OBSERVED_STICK_SIGNS
        ),
        observed_facing_stick_xy=(
            clip.commands.observed_facing_stick_xy * _OBSERVED_STICK_SIGNS
        ),
        observed_mask=clip.commands.observed_mask,
        inferred_velocity_local_xy=(
            clip.commands.inferred_velocity_local_xy * _INFERRED_LOCAL_SIGNS
        ),
        inferred_facing_local_xy=(
            clip.commands.inferred_facing_local_xy * _INFERRED_LOCAL_SIGNS
        ),
        inferred_yaw_rate_rad_s=-clip.commands.inferred_yaw_rate_rad_s,
    )
    mirrored = replace(
        clip,
        clip_id=clip_id,
        root_position_world=clip.root_position_world * _POLAR_SIGNS,
        root_quaternion_world_wxyz=(
            clip.root_quaternion_world_wxyz * _QUATERNION_SIGNS
        ),
        joint_position=(
            clip.joint_position[:, JOINT_MIRROR_PERMUTATION]
            * JOINT_MIRROR_SIGNS
        ),
        root_linear_velocity_world=(
            clip.root_linear_velocity_world * _POLAR_SIGNS
        ),
        root_angular_velocity_world=(
            clip.root_angular_velocity_world * _AXIAL_SIGNS
        ),
        joint_velocity=(
            clip.joint_velocity[:, JOINT_MIRROR_PERMUTATION]
            * JOINT_MIRROR_SIGNS
        ),
        body_position_world=(
            clip.body_position_world[:, BODY_MIRROR_PERMUTATION]
            * _POLAR_SIGNS
        ),
        body_quaternion_world_wxyz=(
            clip.body_quaternion_world_wxyz[:, BODY_MIRROR_PERMUTATION]
            * _QUATERNION_SIGNS
        ),
        body_linear_velocity_world=(
            clip.body_linear_velocity_world[:, BODY_MIRROR_PERMUTATION]
            * _POLAR_SIGNS
        ),
        body_angular_velocity_world=(
            clip.body_angular_velocity_world[:, BODY_MIRROR_PERMUTATION]
            * _AXIAL_SIGNS
        ),
        sole_position_world=(
            clip.sole_position_world[:, _LEFT_RIGHT_PERMUTATION] * _POLAR_SIGNS
        ),
        sole_quaternion_world_wxyz=(
            clip.sole_quaternion_world_wxyz[:, _LEFT_RIGHT_PERMUTATION]
            * _QUATERNION_SIGNS
        ),
        heel_position_world=(
            clip.heel_position_world[:, _LEFT_RIGHT_PERMUTATION] * _POLAR_SIGNS
        ),
        toe_position_world=(
            clip.toe_position_world[:, _LEFT_RIGHT_PERMUTATION] * _POLAR_SIGNS
        ),
        contact=clip.contact[:, _LEFT_RIGHT_PERMUTATION],
        contact_confidence=clip.contact_confidence[:, _LEFT_RIGHT_PERMUTATION],
        commands=commands,
        terrain=mirrored_terrain,
        mirror_of=mirror_of,
    )
    mirrored.validate()
    return mirrored


def _mesh_digest(mesh: CanonicalTerrainMesh) -> str:
    payload = _deterministic_npz_bytes(
        {
            "vertices_local": mesh.vertices_local,
            "faces": mesh.faces,
            "valid_faces": mesh.valid_faces,
            "metadata_json": np.frombuffer(
                _canonical_json_bytes(
                    {"source_asset_sha256": mesh.source_asset_sha256},
                    "mesh metadata",
                ),
                dtype=np.uint8,
            ),
        }
    )
    return hashlib.sha256(payload).hexdigest()


def mirror_terrain(
    binding: TerrainBinding, mesh: CanonicalTerrainMesh
) -> tuple[TerrainBinding, CanonicalTerrainMesh]:
    """Reflect mesh-local geometry and proper placement exactly once.

    Reversing triangle winding compensates for the handedness change.  The
    returned binding references the exact canonical bytes of the returned mesh;
    this function performs no disk publication.
    """

    if not isinstance(binding, TerrainBinding):
        raise ContractError("mirror_terrain requires a TerrainBinding")
    if not isinstance(mesh, CanonicalTerrainMesh):
        raise ContractError("mirror_terrain requires a CanonicalTerrainMesh")
    if binding.asset_sha256 != mesh.source_asset_sha256:
        raise ContractError(
            "terrain binding and mesh must identify the same source asset"
        )
    actual_digest = _mesh_digest(mesh)
    if binding.mesh_sha256 != actual_digest:
        raise ContractError(
            "terrain binding mesh_sha256 does not match the supplied canonical mesh"
        )
    mirrored_mesh = CanonicalTerrainMesh(
        vertices_local=mesh.vertices_local * _POLAR_SIGNS,
        faces=mesh.faces[:, (0, 2, 1)],
        valid_faces=mesh.valid_faces,
        source_asset_sha256=mesh.source_asset_sha256,
    )
    mirrored_binding = replace(
        binding,
        mesh_sha256=_mesh_digest(mirrored_mesh),
        world_from_terrain=_mirror_transform(binding.world_from_terrain),
    )
    return mirrored_binding, mirrored_mesh


def _different_fields(actual: object, expected: object, path: str = "") -> list[str]:
    if isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
            return [path]
        return []
    if is_dataclass(expected):
        if type(actual) is not type(expected):
            return [path]
        failures: list[str] = []
        for field in fields(expected):
            child = f"{path}.{field.name}" if path else field.name
            failures.extend(
                _different_fields(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    child,
                )
            )
        return failures
    return [] if actual == expected else [path]


def assert_mirror_involution(
    original: CanonicalClip,
    mirrored: CanonicalClip,
    terrain_mesh: CanonicalTerrainMesh | None = None,
) -> None:
    """Fail with field paths unless ``mirrored`` is the exact canonical mirror."""

    if not isinstance(original, CanonicalClip) or not isinstance(
        mirrored, CanonicalClip
    ):
        raise ContractError(
            "assert_mirror_involution requires two CanonicalClip values"
        )
    original.validate()
    mirrored.validate()
    expected = mirror_clip(original, terrain_mesh)
    failures = _different_fields(mirrored, expected)
    if failures:
        raise ContractError(
            "canonical mirror mismatch for fields: " + ", ".join(failures)
        )
    if original.terrain is None:
        mirrored_mesh = None
    else:
        assert terrain_mesh is not None
        _mirrored_binding, mirrored_mesh = mirror_terrain(
            original.terrain, terrain_mesh
        )
    restored = mirror_clip(mirrored, mirrored_mesh)
    failures = _different_fields(restored, original)
    if failures:
        raise ContractError(
            "canonical mirror round-trip mismatch for fields: "
            + ", ".join(failures)
        )
