"""Deterministic, immutable storage for canonical terrain-oracle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence
import zipfile

import numpy as np

from mm_sonic.joints import ContractError

from .canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    CanonicalClip,
    CanonicalTerrainMesh,
    CommandTrack,
    SourceIdentity,
    TerrainBinding,
)
from .math3d import RigidTransform


_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_CLIP_ARRAY_NAMES = (
    "root_position_world",
    "root_quaternion_world_wxyz",
    "joint_position",
    "root_linear_velocity_world",
    "root_angular_velocity_world",
    "joint_velocity",
    "body_position_world",
    "body_quaternion_world_wxyz",
    "body_linear_velocity_world",
    "body_angular_velocity_world",
    "sole_position_world",
    "sole_quaternion_world_wxyz",
    "heel_position_world",
    "toe_position_world",
    "contact",
    "contact_confidence",
    "observed_travel_stick_xy",
    "observed_facing_stick_xy",
    "observed_mask",
    "inferred_velocity_local_xy",
    "inferred_facing_local_xy",
    "inferred_yaw_rate_rad_s",
)


@dataclass(frozen=True)
class ClipRecord:
    relative_path: str
    sha256: str
    clip_id: str
    frame_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "clip_id": self.clip_id,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ClipRecord":
        digest = _sha256(value.get("sha256"))
        return cls(
            relative_path=_record_relative_path(value, "clips", digest),
            sha256=digest,
            clip_id=_string(value, "clip_id"),
            frame_count=_positive_int(value, "frame_count"),
        )


@dataclass(frozen=True)
class MeshRecord:
    relative_path: str
    sha256: str
    vertex_count: int
    face_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "MeshRecord":
        digest = _sha256(value.get("sha256"))
        return cls(
            relative_path=_record_relative_path(value, "meshes", digest),
            sha256=digest,
            vertex_count=_positive_int(value, "vertex_count"),
            face_count=_positive_int(value, "face_count"),
        )


@dataclass(frozen=True)
class CorpusManifest:
    schema: str
    coordinate_frame: str
    quaternion_convention: str
    fps: float
    joint_order: tuple[str, ...]
    body_order: tuple[str, ...]
    clips: tuple[ClipRecord, ...]
    meshes: tuple[MeshRecord, ...]
    metadata: dict[str, object]


def _string(value: Mapping[str, object], name: str) -> str:
    result = value.get(name)
    if type(result) is not str or not result:
        raise ContractError(f"{name} must be a nonempty string")
    return result


def _positive_int(value: Mapping[str, object], name: str) -> int:
    result = value.get(name)
    if type(result) is not int or result < 1:
        raise ContractError(f"{name} must be a positive integer")
    return result


def _sha256(value: object) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError("sha256 must be a lowercase SHA-256 digest")
    return value


def _record_relative_path(
    value: Mapping[str, object], directory: str, digest: str
) -> str:
    relative_path = _string(value, "relative_path")
    expected = f"{directory}/{digest}.npz"
    if relative_path != expected:
        raise ContractError(f"relative_path must equal {expected}")
    return relative_path


def _canonical_json_bytes(value: object, label: str) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be finite JSON data") from error


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Encode arrays through ``np.savez`` and normalize ZIP metadata."""

    raw = io.BytesIO()
    np.savez(raw, **arrays)
    normalized = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw.getvalue()), "r") as source:
        with zipfile.ZipFile(normalized, "w", compression=zipfile.ZIP_STORED) as target:
            for name in sorted(source.namelist()):
                member = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
                member.compress_type = zipfile.ZIP_STORED
                member.external_attr = 0o600 << 16
                target.writestr(member, source.read(name))
    return normalized.getvalue()


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    ) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), str(destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_no_replace(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _rename_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _clip_metadata(clip: CanonicalClip) -> dict[str, object]:
    terrain = None
    if clip.terrain is not None:
        terrain = {
            "asset_path": clip.terrain.asset_path,
            "asset_size_bytes": clip.terrain.asset_size_bytes,
            "asset_sha256": clip.terrain.asset_sha256,
            "asset_license_id": clip.terrain.asset_license_id,
            "mesh_sha256": clip.terrain.mesh_sha256,
            "world_from_terrain": {
                "translation_world": clip.terrain.world_from_terrain.translation_world.tolist(),
                "quaternion_world_from_local_wxyz": clip.terrain.world_from_terrain.quaternion_world_from_local_wxyz.tolist(),
            },
            "validity_mask_path": clip.terrain.validity_mask_path,
        }
    return {
        "clip_id": clip.clip_id,
        "fps": clip.fps,
        "source": {
            "source_format": clip.source.source_format,
            "source_path": clip.source.source_path,
            "source_size_bytes": clip.source.source_size_bytes,
            "source_sha256": clip.source.source_sha256,
            "source_license_id": clip.source.source_license_id,
            "coordinate_convention": clip.source.coordinate_convention,
            "quaternion_convention": clip.source.quaternion_convention,
            "pose_origin": clip.source.pose_origin,
        },
        "joint_names": list(clip.joint_names),
        "body_names": list(clip.body_names),
        "terrain": terrain,
        "action_tags": list(clip.action_tags),
        "mirror_of": clip.mirror_of,
    }


def _clip_arrays(clip: CanonicalClip) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(getattr(clip, name)) for name in _CLIP_ARRAY_NAMES[:16]}
    arrays.update(
        {
            name: np.asarray(getattr(clip.commands, name))
            for name in _CLIP_ARRAY_NAMES[16:]
        }
    )
    arrays["metadata_json"] = np.frombuffer(
        _canonical_json_bytes(_clip_metadata(clip), "clip metadata"), dtype=np.uint8
    )
    return arrays


def _read_json_array(archive: np.lib.npyio.NpzFile, name: str) -> dict[str, object]:
    try:
        return json.loads(bytes(np.asarray(archive[name], dtype=np.uint8)).decode("ascii"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid {name} in canonical artifact") from error


def _verified_npz_bytes(path: Path) -> bytes:
    expected = _sha256(path.stem)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError(f"canonical artifact digest does not match its filename: {path}")
    return payload


def write_clip(output: Path, clip: CanonicalClip) -> ClipRecord:
    """Validate and publish one content-addressed canonical clip NPZ."""

    if not isinstance(clip, CanonicalClip):
        raise ContractError("write_clip requires a CanonicalClip")
    clip.validate()
    payload = _deterministic_npz_bytes(_clip_arrays(clip))
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(output)
    _write_no_replace(directory / f"{digest}.npz", payload)
    return ClipRecord(f"{directory.name}/{digest}.npz", digest, clip.clip_id, clip.frame_count)


def read_clip(path: Path) -> CanonicalClip:
    """Load a canonical clip without permitting pickle deserialization."""

    try:
        with np.load(io.BytesIO(_verified_npz_bytes(Path(path))), allow_pickle=False) as archive:
            metadata = _read_json_array(archive, "metadata_json")
            source = SourceIdentity(**metadata["source"])
            terrain_data = metadata["terrain"]
            terrain = None
            if terrain_data is not None:
                transform = terrain_data["world_from_terrain"]
                terrain = TerrainBinding(
                    asset_path=terrain_data["asset_path"],
                    asset_size_bytes=terrain_data["asset_size_bytes"],
                    asset_sha256=terrain_data["asset_sha256"],
                    asset_license_id=terrain_data["asset_license_id"],
                    mesh_sha256=terrain_data["mesh_sha256"],
                    world_from_terrain=RigidTransform(**transform),
                    validity_mask_path=terrain_data["validity_mask_path"],
                )
            commands = CommandTrack(
                **{name: archive[name] for name in _CLIP_ARRAY_NAMES[16:]}
            )
            clip = CanonicalClip(
                clip_id=metadata["clip_id"],
                fps=metadata["fps"],
                source=source,
                joint_names=tuple(metadata["joint_names"]),
                body_names=tuple(metadata["body_names"]),
                **{name: archive[name] for name in _CLIP_ARRAY_NAMES[:16]},
                commands=commands,
                terrain=terrain,
                action_tags=tuple(metadata["action_tags"]),
                mirror_of=metadata["mirror_of"],
            )
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid canonical clip artifact: {path}") from error
    clip.validate()
    return clip


def write_mesh(output: Path, mesh: CanonicalTerrainMesh) -> MeshRecord:
    """Publish one content-addressed canonical terrain mesh NPZ."""

    if not isinstance(mesh, CanonicalTerrainMesh):
        raise ContractError("write_mesh requires a CanonicalTerrainMesh")
    payload = _deterministic_npz_bytes(
        {
            "vertices_local": mesh.vertices_local,
            "faces": mesh.faces,
            "valid_faces": mesh.valid_faces,
            "metadata_json": np.frombuffer(
                _canonical_json_bytes(
                    {"source_asset_sha256": mesh.source_asset_sha256}, "mesh metadata"
                ),
                dtype=np.uint8,
            ),
        }
    )
    digest = hashlib.sha256(payload).hexdigest()
    directory = Path(output)
    _write_no_replace(directory / f"{digest}.npz", payload)
    return MeshRecord(
        f"{directory.name}/{digest}.npz", digest, len(mesh.vertices_local), len(mesh.faces)
    )


def read_mesh(path: Path) -> CanonicalTerrainMesh:
    """Load a canonical terrain mesh without permitting pickle deserialization."""

    try:
        with np.load(io.BytesIO(_verified_npz_bytes(Path(path))), allow_pickle=False) as archive:
            metadata = _read_json_array(archive, "metadata_json")
            return CanonicalTerrainMesh(
                vertices_local=archive["vertices_local"],
                faces=archive["faces"],
                valid_faces=archive["valid_faces"],
                source_asset_sha256=metadata["source_asset_sha256"],
            )
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"invalid canonical mesh artifact: {path}") from error


def _metadata_and_mesh_records(
    metadata: Mapping[str, object],
) -> tuple[dict[str, object], tuple[MeshRecord, ...]]:
    copied = dict(metadata)
    supplied = copied.pop("mesh_records", [])
    if not isinstance(supplied, list):
        raise ContractError("metadata.mesh_records must be a list of mesh records")
    try:
        mesh_records = tuple(
            MeshRecord.from_dict(
                value.to_dict() if isinstance(value, MeshRecord) else value
            )
            for value in supplied
        )
    except (AttributeError, TypeError) as error:
        raise ContractError("metadata.mesh_records must contain mesh records") from error
    return (
        json.loads(_canonical_json_bytes(copied, "corpus metadata")),
        mesh_records,
    )


def publish_corpus(
    output: Path, records: Sequence[ClipRecord], metadata: dict,
) -> Path:
    """Atomically publish one immutable corpus manifest directory."""

    if type(metadata) is not dict:
        raise ContractError("corpus metadata must be a dict")
    if not records or any(not isinstance(record, ClipRecord) for record in records):
        raise ContractError("corpus requires one or more ClipRecord values")
    records = tuple(ClipRecord.from_dict(record.to_dict()) for record in records)
    manifest_metadata, mesh_records = _metadata_and_mesh_records(metadata)
    manifest = {
        "schema": "terrain-oracle-corpus/v1",
        "coordinate_frame": CANONICAL_COORDINATE_CONVENTION,
        "quaternion_convention": "wxyz",
        "fps": 50.0,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "body_order": list(ISAACLAB_BODY_NAMES),
        "clips": [record.to_dict() for record in records],
        "meshes": [record.to_dict() for record in mesh_records],
        "metadata": manifest_metadata,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_no_replace(temporary / "manifest.json", _canonical_json_bytes(manifest, "corpus manifest"))
        _fsync_directory(temporary)
        _rename_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return destination


def load_corpus(path: Path) -> CorpusManifest:
    """Load the canonical JSON manifest published by :func:`publish_corpus`."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    try:
        document = json.loads(manifest_path.read_text("ascii"))
        clips = tuple(ClipRecord.from_dict(value) for value in document["clips"])
        meshes = tuple(MeshRecord.from_dict(value) for value in document["meshes"])
        metadata = document["metadata"]
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
        raise ContractError(f"invalid corpus manifest: {manifest_path}") from error
    if (
        document.get("schema") != "terrain-oracle-corpus/v1"
        or document.get("coordinate_frame") != CANONICAL_COORDINATE_CONVENTION
        or document.get("quaternion_convention") != "wxyz"
        or document.get("fps") != 50.0
        or tuple(document.get("joint_order", ())) != ISAACLAB_JOINT_NAMES
        or tuple(document.get("body_order", ())) != ISAACLAB_BODY_NAMES
        or type(metadata) is not dict
    ):
        raise ContractError("corpus manifest violates the canonical contract")
    return CorpusManifest(
        schema=document["schema"],
        coordinate_frame=document["coordinate_frame"],
        quaternion_convention=document["quaternion_convention"],
        fps=document["fps"],
        joint_order=tuple(document["joint_order"]),
        body_order=tuple(document["body_order"]),
        clips=clips,
        meshes=meshes,
        metadata=metadata,
    )
