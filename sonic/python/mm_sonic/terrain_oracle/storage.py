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
import secrets
import shutil
import stat
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
COMPLETION_MARKER = ".terrain-oracle-complete.json"
_OWNER_MARKER = ".terrain-oracle-publishing-owner"
_COMPLETE_SCHEMA = "terrain-oracle-complete/v1"
_CORPUS_EXTENSIONS = ("coverage.json", "render-audit")
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
        try:
            _rename_noreplace(temporary, destination)
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
            try:
                os.link(
                    temporary,
                    destination,
                    follow_symlinks=False,
                )
            finally:
                temporary.unlink(missing_ok=True)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_regular_file_nofollow(path: Path) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError(f"publication entry is not regular: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _normalized_excluded_top_level(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        type(item) is not str for item in value
    ):
        raise ContractError(
            "completion excluded_top_level must be a string list"
        )
    result = tuple(value)
    if result not in ((), _CORPUS_EXTENSIONS):
        raise ContractError(
            "completion excluded_top_level is not a canonical profile"
        )
    return result


def _validate_extension_entries(
    root: Path,
    excluded_top_level: tuple[str, ...],
) -> None:
    for name in excluded_top_level:
        path = root / name
        if not os.path.lexists(path):
            continue
        if path.is_symlink():
            raise ContractError("corpus extension must not be a symlink")
        if name == "coverage.json" and not path.is_file():
            raise ContractError("coverage.json extension must be a regular file")
        if name == "render-audit" and not path.is_dir():
            raise ContractError("render-audit extension must be a directory")


def _directory_tree_evidence(
    root: Path,
    excluded_top_level: tuple[str, ...],
) -> tuple[list[str], list[dict[str, object]], str]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError("publication tree root must be a regular directory")
    _validate_extension_entries(root, excluded_top_level)
    directories: list[str] = []
    files: list[dict[str, object]] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative_path = path.relative_to(root).as_posix()
        parts = Path(relative_path).parts
        if relative_path in (COMPLETION_MARKER, _OWNER_MARKER):
            continue
        if parts and parts[0] in excluded_top_level:
            continue
        if path.is_symlink():
            raise ContractError("publication tree must not contain symlinks")
        if path.is_dir():
            directories.append(relative_path)
            continue
        if not path.is_file():
            raise ContractError(
                "publication tree contains a non-regular entry"
            )
        payload = _read_regular_file_nofollow(path)
        files.append(
            {
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    evidence = {
        "directories": directories,
        "files": files,
    }
    tree_sha256 = hashlib.sha256(
        _canonical_json_bytes(evidence, "publication tree evidence")
    ).hexdigest()
    return directories, files, tree_sha256


def _seal_directory(
    root: Path,
    *,
    excluded_top_level: Sequence[str] = (),
) -> dict[str, object]:
    excluded = tuple(excluded_top_level)
    if excluded not in ((), _CORPUS_EXTENSIONS):
        raise ContractError(
            "publication exclusions must use the canonical corpus profile"
        )
    if os.path.lexists(root / COMPLETION_MARKER):
        raise FileExistsError(root / COMPLETION_MARKER)
    if os.path.lexists(root / _OWNER_MARKER):
        raise ContractError("publication staging tree contains an owner marker")
    for name in excluded:
        if os.path.lexists(root / name):
            raise ContractError(
                "excluded corpus extensions must be absent while sealing"
            )
    _directories, files, tree_sha256 = _directory_tree_evidence(
        root,
        excluded,
    )
    marker = {
        "schema": _COMPLETE_SCHEMA,
        "excluded_top_level": list(excluded),
        "file_count": len(files),
        "tree_sha256": tree_sha256,
    }
    _write_no_replace(
        root / COMPLETION_MARKER,
        _canonical_json_bytes(marker, "publication completion marker"),
    )
    _fsync_directory(root)
    return marker


def _validate_complete_directory(root: Path) -> dict[str, object]:
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ContractError("published tree is missing or a symlink")
    if os.path.lexists(directory / _OWNER_MARKER):
        raise ContractError("published tree is still owned by a writer")
    marker_path = directory / COMPLETION_MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ContractError("published tree completion marker is missing")
    payload = _read_regular_file_nofollow(marker_path)
    try:
        marker = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("invalid publication completion marker") from error
    if (
        not isinstance(marker, dict)
        or set(marker) != {
            "schema",
            "excluded_top_level",
            "file_count",
            "tree_sha256",
        }
        or marker["schema"] != _COMPLETE_SCHEMA
        or payload
        != _canonical_json_bytes(
            marker,
            "publication completion marker",
        )
        or type(marker["file_count"]) is not int
        or marker["file_count"] < 1
    ):
        raise ContractError("publication completion marker fields are stale")
    excluded = _normalized_excluded_top_level(
        marker["excluded_top_level"]
    )
    _directories, files, tree_sha256 = _directory_tree_evidence(
        directory,
        excluded,
    )
    if (
        marker["file_count"] != len(files)
        or _sha256(marker["tree_sha256"]) != tree_sha256
    ):
        raise ContractError("published tree completion digest is stale")
    return marker


def _cleanup_owned_destination(
    destination: Path,
    *,
    identity: tuple[int, int],
    owner_payload: bytes,
) -> None:
    try:
        metadata = os.lstat(destination)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
            or _read_regular_file_nofollow(destination / _OWNER_MARKER)
            != owner_payload
        ):
            return
    except (FileNotFoundError, OSError, ContractError):
        return
    shutil.rmtree(destination)


def _copy_directory_claim_noreplace(
    source: Path,
    destination: Path,
) -> None:
    marker = _validate_complete_directory(source)
    marker_payload = _read_regular_file_nofollow(
        source / COMPLETION_MARKER
    )
    owner_payload = (
        f"terrain-oracle-owner-v1:{secrets.token_hex(32)}\n"
    ).encode("ascii")
    os.mkdir(destination, mode=0o700)
    claimed = os.lstat(destination)
    identity = (claimed.st_dev, claimed.st_ino)
    try:
        _write_no_replace(destination / _OWNER_MARKER, owner_payload)
        directories = [
            path
            for path in sorted(
                source.rglob("*"),
                key=lambda item: (
                    len(item.relative_to(source).parts),
                    item.relative_to(source).as_posix(),
                ),
            )
            if path.is_dir() and not path.is_symlink()
        ]
        for directory in directories:
            relative = directory.relative_to(source)
            os.mkdir(destination / relative)
        for path in sorted(
            source.rglob("*"),
            key=lambda item: item.relative_to(source).as_posix(),
        ):
            if path.is_symlink():
                raise ContractError(
                    "publication source tree contains a symlink"
                )
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if relative.as_posix() == COMPLETION_MARKER:
                continue
            _write_no_replace(
                destination / relative,
                _read_regular_file_nofollow(path),
            )
        _directories, files, tree_sha256 = _directory_tree_evidence(
            destination,
            tuple(marker["excluded_top_level"]),
        )
        if (
            len(files) != marker["file_count"]
            or tree_sha256 != marker["tree_sha256"]
        ):
            raise ContractError(
                "NFS publication copy does not match sealed staging tree"
            )
        for directory in sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _write_no_replace(
            destination / COMPLETION_MARKER,
            marker_payload,
        )
        _fsync_directory(destination)
        (destination / _OWNER_MARKER).unlink()
        _fsync_directory(destination)
        _validate_complete_directory(destination)
    except BaseException:
        _cleanup_owned_destination(
            destination,
            identity=identity,
            owner_payload=owner_payload,
        )
        raise
    shutil.rmtree(source)


def _publish_directory_no_replace(
    source: Path,
    destination: Path,
) -> None:
    _validate_complete_directory(source)
    try:
        _rename_noreplace(source, destination)
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
        _copy_directory_claim_noreplace(source, destination)
    _fsync_directory(destination.parent)
    _validate_complete_directory(destination)


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


def _clip_payload(clip: CanonicalClip) -> bytes:
    if not isinstance(clip, CanonicalClip):
        raise ContractError("clip payload requires a CanonicalClip")
    clip.validate()
    return _deterministic_npz_bytes(_clip_arrays(clip))


def clip_digest(clip: CanonicalClip) -> str:
    """Return the SHA-256 of the exact canonical bytes used by ``write_clip``."""

    return hashlib.sha256(_clip_payload(clip)).hexdigest()


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
    payload = _clip_payload(clip)
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


def _mesh_payload(mesh: CanonicalTerrainMesh) -> bytes:
    if not isinstance(mesh, CanonicalTerrainMesh):
        raise ContractError("mesh payload requires a CanonicalTerrainMesh")
    return _deterministic_npz_bytes(
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


def mesh_digest(mesh: CanonicalTerrainMesh) -> str:
    """Return the SHA-256 of the exact canonical bytes published by ``write_mesh``."""

    return hashlib.sha256(_mesh_payload(mesh)).hexdigest()


def write_mesh(output: Path, mesh: CanonicalTerrainMesh) -> MeshRecord:
    """Publish one content-addressed canonical terrain mesh NPZ."""

    payload = _mesh_payload(mesh)
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
        _seal_directory(temporary)
        _publish_directory_no_replace(temporary, destination)
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
        _validate_complete_directory(manifest_path)
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
