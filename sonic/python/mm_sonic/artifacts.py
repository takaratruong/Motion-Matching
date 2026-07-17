"""Confined, append-only evidence bundles for MM-to-SONIC runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import BinaryIO, Mapping
import uuid

from .joints import ContractError


_MANIFEST_SCHEMA = "mm-sonic-run-manifest/v1"
_INVENTORY_SCHEMA = "mm-sonic-evidence-inventory/v1"
_TERMINAL_STATUSES = frozenset(("complete", "failed", "not_run"))
_TERMINAL_MANIFEST_KEYS = frozenset(
    (
        "external",
        "repositories",
        "artifact_hashes",
        "command_script",
        "perturbation",
        "coordinate_transform",
        "processes",
    )
)
_ARTIFACT_HASH_KEYS = frozenset(
    (
        "policy",
        "encoder",
        "observation_config",
        "model",
        "source_mjcf",
        "motion",
        "terrain",
        "joint_map",
        "scene",
    )
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RESERVED_OUTPUTS = frozenset(("manifest.json", "inventory.json"))

_EXTERNAL_KEYS = frozenset(
    (
        "gear_checkout",
        "gear_commit",
        "gear_dirty",
        "policy",
        "observation_config",
        "encoder",
        "terrain_dir",
        "source_mjcf",
        "hashes",
    )
)
_MANDATORY_COMPLETE_HASHES = _ARTIFACT_HASH_KEYS - {"encoder"}
_SCENE_REGISTRATION_KEYS = frozenset(
    (
        "scene_id",
        "route_id",
        "source_kind",
        "source_hashes",
        "coordinate_source",
        "coordinate_target",
        "transform_matrix",
        "output_hashes",
        "allowed_foot_geoms",
        "forbidden_geom_groups",
    )
)
_FORBIDDEN_GEOM_GROUPS = frozenset(("pelvis", "knees", "torso", "hands"))
_HOLDEN_COORDINATE_SIGNATURE = "holden-y-up-right-handed-forward-plus-z"
_MUJOCO_COORDINATE_SIGNATURE = "mujoco-z-up-right-handed-forward-plus-x"
_HOLDEN_TO_MUJOCO_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _identifier(value: object, label: str) -> str:
    if (
        type(value) is not str
        or value in (".", "..")
        or _IDENTIFIER.fullmatch(value) is None
    ):
        raise ContractError(
            f"{label} identifier must use only letters, digits, '.', '_', or '-'"
        )
    return value


def _relative_parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    path = Path(value)
    if path.is_absolute():
        raise ContractError("output path must be relative to the run bundle")
    parts = path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ContractError("output path must be a confined relative path")
    if any("\x00" in part or "\\" in part for part in parts):
        raise ContractError("output path contains an invalid component")
    return tuple(parts)


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _regular_read_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _inode_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _opened_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while recording run evidence")
        view = view[written:]


def _json_bytes(value: object, label: str) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be finite JSON data") from error
    return (text + "\n").encode("ascii")


def _json_copy(value: object, label: str) -> object:
    return json.loads(_json_bytes(value, label))


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ContractError(f"terminal manifest {label} must be an object")
    return value


def _exact_keys(
    value: object, expected: frozenset[str], label: str
) -> dict[str, object]:
    output = _mapping(value, label)
    if set(output) != expected:
        raise ContractError(f"terminal manifest {label} has invalid keys")
    return output


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"terminal manifest {label} must be a nonempty string")
    return value


def _sha256_or_null(value: object, label: str) -> None:
    if value is not None and (
        type(value) is not str or _SHA256.fullmatch(value) is None
    ):
        raise ContractError(
            f"terminal manifest {label} must be a SHA-256 digest or null"
        )


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ContractError(f"terminal manifest {label} must be a finite number")
    return float(value)


def _timestamp(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ContractError(
            f"terminal manifest {label} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise ContractError(
            f"terminal manifest {label} must include a timezone"
        )
    return text


def _hash_mapping(value: object, label: str) -> None:
    hashes = _mapping(value, label)
    if not hashes:
        raise ContractError(f"terminal manifest {label} must not be empty")
    for name, digest in hashes.items():
        _nonempty_string(name, f"{label} key")
        _sha256_or_null(digest, f"{label}.{name}")
        if digest is None:
            raise ContractError(
                f"terminal manifest {label}.{name} cannot be null"
            )


def _geom_ids(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not list or not value:
        raise ContractError(
            f"terminal manifest {label} must be a nonempty array"
        )
    if any(type(item) is not int or item < 0 for item in value):
        raise ContractError(
            f"terminal manifest {label} must contain nonnegative integers"
        )
    if len(set(value)) != len(value):
        raise ContractError(f"terminal manifest {label} contains duplicates")
    return tuple(value)


def _validate_scene_registration(value: object) -> None:
    if type(value) is not dict or set(value) not in (
        _SCENE_REGISTRATION_KEYS,
        _SCENE_REGISTRATION_KEYS | {"terrain_geoms"},
    ):
        raise ContractError(
            "terminal manifest scene_registration has invalid keys"
        )
    scene = value
    _nonempty_string(scene["scene_id"], "scene_registration.scene_id")
    route_id = scene["route_id"]
    if route_id is not None:
        _nonempty_string(route_id, "scene_registration.route_id")
    _nonempty_string(scene["source_kind"], "scene_registration.source_kind")
    _hash_mapping(scene["source_hashes"], "scene_registration.source_hashes")
    source = _nonempty_string(
        scene["coordinate_source"], "scene_registration.coordinate_source"
    )
    target = _nonempty_string(
        scene["coordinate_target"], "scene_registration.coordinate_target"
    )
    if (
        source != _HOLDEN_COORDINATE_SIGNATURE
        or target != _MUJOCO_COORDINATE_SIGNATURE
    ):
        raise ContractError(
            "terminal manifest scene_registration coordinate signatures changed"
        )
    matrix = scene["transform_matrix"]
    if type(matrix) is not list or len(matrix) != 3:
        raise ContractError(
            "terminal manifest scene_registration.transform_matrix must be 3x3"
        )
    normalized_matrix: list[tuple[float, ...]] = []
    for row_index, row in enumerate(matrix):
        if type(row) is not list or len(row) != 3:
            raise ContractError(
                "terminal manifest scene_registration.transform_matrix must be 3x3"
            )
        normalized_matrix.append(
            tuple(
                _finite_number(
                    item,
                    "scene_registration.transform_matrix"
                    f"[{row_index}][{column_index}]",
                )
                for column_index, item in enumerate(row)
            )
        )
    if tuple(normalized_matrix) != _HOLDEN_TO_MUJOCO_MATRIX:
        raise ContractError(
            "terminal manifest scene_registration.transform_matrix changed"
        )
    _hash_mapping(scene["output_hashes"], "scene_registration.output_hashes")
    allowed = set(
        _geom_ids(
            scene["allowed_foot_geoms"],
            "scene_registration.allowed_foot_geoms",
        )
    )
    groups = _exact_keys(
        scene["forbidden_geom_groups"],
        _FORBIDDEN_GEOM_GROUPS,
        "scene_registration.forbidden_geom_groups",
    )
    used = set(allowed)
    terrain = set()
    if "terrain_geoms" in scene:
        terrain = set(
            _geom_ids(
                scene["terrain_geoms"],
                "scene_registration.terrain_geoms",
            )
        )
        if not terrain:
            raise ContractError(
                "terminal manifest scene_registration.terrain_geoms is empty"
            )
        if terrain & used:
            raise ContractError(
                "terminal manifest scene_registration geom groups overlap"
            )
        used.update(terrain)
    for name in sorted(_FORBIDDEN_GEOM_GROUPS):
        current = set(
            _geom_ids(
                groups[name],
                f"scene_registration.forbidden_geom_groups.{name}",
            )
        )
        if current & used:
            raise ContractError(
                "terminal manifest scene_registration geom groups overlap"
            )
        used.update(current)


def _validate_terminal_metadata(
    manifest: Mapping[str, object], status: str
) -> None:
    missing = sorted(_TERMINAL_MANIFEST_KEYS - set(manifest))
    if missing:
        raise ContractError(f"terminal manifest is missing fields: {missing}")
    if status == "complete" and "scene_registration" not in manifest:
        raise ContractError(
            "complete terminal manifest requires scene_registration"
        )
    if "scene_registration" in manifest:
        _validate_scene_registration(manifest["scene_registration"])

    external = _exact_keys(manifest["external"], _EXTERNAL_KEYS, "external")
    for name in (
        "gear_checkout",
        "policy",
        "observation_config",
        "terrain_dir",
        "source_mjcf",
    ):
        _nonempty_string(external[name], f"external.{name}")
    encoder = external["encoder"]
    if encoder is not None:
        _nonempty_string(encoder, "external.encoder")
    if external["gear_commit"] is None and status == "not_run":
        pass
    elif (
        type(external["gear_commit"]) is not str
        or _COMMIT.fullmatch(external["gear_commit"]) is None
    ):
        raise ContractError(
            "terminal manifest external.gear_commit must be a commit digest"
        )
    if external["gear_dirty"] is None and status == "not_run":
        pass
    elif type(external["gear_dirty"]) is not bool:
        raise ContractError(
            "terminal manifest external.gear_dirty must be a boolean"
        )
    external_hashes = _mapping(external["hashes"], "external.hashes")
    if not external_hashes:
        raise ContractError("terminal manifest external.hashes must not be empty")
    for name, value in external_hashes.items():
        _nonempty_string(name, "external.hashes key")
        _sha256_or_null(value, f"external.hashes.{name}")
        if value is None and status != "not_run":
            raise ContractError(
                f"terminal manifest external.hashes.{name} cannot be null"
            )

    repositories = _mapping(manifest["repositories"], "repositories")
    if not repositories:
        raise ContractError("terminal manifest repositories must not be empty")
    for name, value in repositories.items():
        _nonempty_string(name, "repositories key")
        repository = _exact_keys(
            value, frozenset(("commit", "dirty")), f"repositories.{name}"
        )
        if repository["commit"] is None and status == "not_run":
            pass
        elif (
            type(repository["commit"]) is not str
            or _COMMIT.fullmatch(repository["commit"]) is None
        ):
            raise ContractError(
                f"terminal manifest repositories.{name}.commit is invalid"
            )
        if repository["dirty"] is None and status == "not_run":
            pass
        elif type(repository["dirty"]) is not bool:
            raise ContractError(
                f"terminal manifest repositories.{name}.dirty must be a boolean"
            )

    artifact_hashes = _exact_keys(
        manifest["artifact_hashes"], _ARTIFACT_HASH_KEYS, "artifact_hashes"
    )
    for name, value in artifact_hashes.items():
        _sha256_or_null(value, f"artifact_hashes.{name}")
    if status == "complete":
        for name in sorted(_MANDATORY_COMPLETE_HASHES):
            if artifact_hashes[name] is None:
                raise ContractError(
                    "complete terminal manifest requires "
                    f"artifact_hashes.{name}"
                )

    command = _mapping(manifest["command_script"], "command_script")
    if not {"id", "sha256"}.issubset(command):
        raise ContractError("terminal manifest command_script has invalid keys")
    _nonempty_string(command["id"], "command_script.id")
    _sha256_or_null(command["sha256"], "command_script.sha256")

    perturbation = _exact_keys(
        manifest["perturbation"],
        frozenset(("id", "lateral_offset_m", "yaw_offset_rad")),
        "perturbation",
    )
    _nonempty_string(perturbation["id"], "perturbation.id")
    _finite_number(
        perturbation["lateral_offset_m"], "perturbation.lateral_offset_m"
    )
    _finite_number(perturbation["yaw_offset_rad"], "perturbation.yaw_offset_rad")

    transform = _exact_keys(
        manifest["coordinate_transform"],
        frozenset(("source", "target", "matrix")),
        "coordinate_transform",
    )
    _nonempty_string(transform["source"], "coordinate_transform.source")
    _nonempty_string(transform["target"], "coordinate_transform.target")
    matrix = transform["matrix"]
    if type(matrix) is not list or len(matrix) != 3:
        raise ContractError(
            "terminal manifest coordinate_transform.matrix must be 3x3"
        )
    for row_index, row in enumerate(matrix):
        if type(row) is not list or len(row) != 3:
            raise ContractError(
                "terminal manifest coordinate_transform.matrix must be 3x3"
            )
        for column_index, value in enumerate(row):
            _finite_number(
                value,
                "coordinate_transform.matrix"
                f"[{row_index}][{column_index}]",
            )

    processes = manifest["processes"]
    if type(processes) is not list or not processes:
        raise ContractError("terminal manifest processes must be a nonempty array")
    for index, value in enumerate(processes):
        process = _mapping(value, f"processes[{index}]")
        if not {"name", "argv"}.issubset(process):
            raise ContractError(
                f"terminal manifest processes[{index}] has invalid keys"
            )
        _nonempty_string(process["name"], f"processes[{index}].name")
        argv = process["argv"]
        if type(argv) is not list or not argv:
            raise ContractError(
                f"terminal manifest processes[{index}].argv must not be empty"
            )
        for argument_index, argument in enumerate(argv):
            _nonempty_string(
                argument, f"processes[{index}].argv[{argument_index}]"
            )
        invocation_cwd = process.get("invocation_cwd")
        if invocation_cwd is not None:
            _nonempty_string(
                invocation_cwd, f"processes[{index}].invocation_cwd"
            )
            if not Path(invocation_cwd).is_absolute():
                raise ContractError(
                    f"terminal manifest processes[{index}].invocation_cwd "
                    "must be absolute"
                )


def _validate_terminal_manifest(manifest: object) -> dict[str, object]:
    value = _mapping(manifest, "root")
    required = {
        "schema",
        "experiment_id",
        "run_id",
        "status",
        "created_utc",
        "finalization_started_utc",
        "finalized_utc",
        "outcome",
        "evidence",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ContractError(f"terminal manifest is missing fields: {missing}")
    if value["schema"] != _MANIFEST_SCHEMA:
        raise ContractError("terminal manifest has an unsupported schema")
    _identifier(value["experiment_id"], "experiment")
    _identifier(value["run_id"], "run")
    status = value["status"]
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ContractError(
            "terminal manifest status must be complete, failed, or not_run"
        )
    _timestamp(value["created_utc"], "created_utc")
    _timestamp(value["finalization_started_utc"], "finalization_started_utc")
    _timestamp(value["finalized_utc"], "finalized_utc")
    _mapping(value["outcome"], "outcome")
    evidence = _exact_keys(
        value["evidence"],
        frozenset(("inventory_sha256", "file_count")),
        "evidence",
    )
    inventory_sha256 = evidence["inventory_sha256"]
    if (
        type(inventory_sha256) is not str
        or _SHA256.fullmatch(inventory_sha256) is None
    ):
        raise ContractError(
            "terminal manifest evidence.inventory_sha256 must be a SHA-256 digest"
        )
    file_count = evidence["file_count"]
    if type(file_count) is not int or file_count < 0:
        raise ContractError(
            "terminal manifest evidence.file_count must be a nonnegative integer"
        )
    _validate_terminal_metadata(value, status)
    return value


def _terminal_manifest_core(manifest: Mapping[str, object]) -> dict[str, object]:
    core = dict(manifest)
    core.pop("evidence", None)
    return core


def _sha256_descriptor(
    descriptor: int, relative: str
) -> tuple[str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"evidence is not a regular file: {relative}")
    if before.st_nlink != 1:
        raise ContractError(f"evidence is a hard link alias: {relative}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if _opened_identity(before) != _opened_identity(after) or size != before.st_size:
        raise ContractError(f"evidence changed while hashing: {relative}")
    return digest.hexdigest(), size


def _read_descriptor_bytes(descriptor: int, relative: str) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"evidence is not a regular file: {relative}")
    if before.st_nlink != 1:
        raise ContractError(f"evidence is a hard link alias: {relative}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if _opened_identity(before) != _opened_identity(after) or size != before.st_size:
        raise ContractError(f"evidence changed while reading: {relative}")
    return b"".join(chunks)


def _read_regular_at(directory_fd: int, filename: str) -> bytes:
    try:
        observed = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise ContractError(f"cannot inspect evidence file: {filename}") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ContractError(f"evidence is not a safe regular file: {filename}")
    descriptor = -1
    try:
        descriptor = os.open(
            filename, _regular_read_flags(), dir_fd=directory_fd
        )
        opened = os.fstat(descriptor)
        if _inode_identity(observed) != _inode_identity(opened):
            raise ContractError(f"evidence file changed while opening: {filename}")
        return _read_descriptor_bytes(descriptor, filename)
    except OSError as error:
        raise ContractError(f"cannot read evidence file safely: {filename}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _walk_directory_fd(
    directory_fd: int,
    prefix: str = "",
):
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        raise ContractError("cannot enumerate retained run directory") from error
    for name in names:
        relative = name if not prefix else f"{prefix}/{name}"
        try:
            observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise ContractError(f"cannot inspect evidence entry: {relative}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ContractError(f"evidence tree contains a symlink: {relative}")
        if stat.S_ISDIR(observed.st_mode):
            try:
                child = os.open(name, _directory_flags(), dir_fd=directory_fd)
            except OSError as error:
                raise ContractError(
                    f"cannot open evidence directory safely: {relative}"
                ) from error
            try:
                opened = os.fstat(child)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _inode_identity(observed) != _inode_identity(opened)
                ):
                    raise ContractError(
                        f"evidence directory changed while opening: {relative}"
                    )
                yield "directory", relative, child, opened
                yield from _walk_directory_fd(child, relative)
            finally:
                os.close(child)
            continue
        if stat.S_ISREG(observed.st_mode):
            try:
                descriptor = os.open(
                    name, _regular_read_flags(), dir_fd=directory_fd
                )
            except OSError as error:
                raise ContractError(
                    f"cannot open evidence file safely: {relative}"
                ) from error
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _inode_identity(observed) != _inode_identity(opened)
                ):
                    raise ContractError(
                        f"evidence file changed while opening: {relative}"
                    )
                if opened.st_nlink != 1:
                    raise ContractError(
                        f"evidence is a hard link alias: {relative}"
                    )
                yield "file", relative, descriptor, opened
            finally:
                os.close(descriptor)
            continue
        raise ContractError(f"evidence tree contains a special entry: {relative}")


class RunBundle:
    """Own one exclusively created run directory and its evidence lifecycle."""

    def __init__(
        self,
        path: Path,
        directory_fd: int,
        manifest: dict[str, object],
    ) -> None:
        self._path = path
        self._directory_fd = directory_fd
        self._manifest = manifest
        self._status = "running"
        self._sequence = 0

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str],
        experiment_id: str,
        run_id: str,
    ) -> "RunBundle":
        experiment = _identifier(experiment_id, "experiment")
        run = _identifier(run_id, "run")
        root_path = Path(root).expanduser()
        try:
            root_path.mkdir(parents=True, exist_ok=True)
            root_path = root_path.resolve(strict=True)
        except OSError as error:
            raise ContractError(f"cannot create output root: {root}") from error
        if not root_path.is_dir():
            raise ContractError(f"output root is not a directory: {root_path}")

        root_fd = os.open(root_path, _directory_flags())
        experiment_fd = -1
        run_fd = -1
        try:
            try:
                os.mkdir(experiment, mode=0o755, dir_fd=root_fd)
                os.fsync(root_fd)
            except FileExistsError:
                pass
            try:
                experiment_fd = os.open(
                    experiment, _directory_flags(), dir_fd=root_fd
                )
            except OSError as error:
                raise ContractError(
                    f"experiment output is not a safe directory: {experiment}"
                ) from error
            try:
                os.mkdir(run, mode=0o755, dir_fd=experiment_fd)
                os.fsync(experiment_fd)
            except FileExistsError as error:
                raise ContractError(
                    f"run bundle already exists: {experiment}/{run}"
                ) from error
            run_fd = os.open(run, _directory_flags(), dir_fd=experiment_fd)
        finally:
            if experiment_fd >= 0:
                os.close(experiment_fd)
            os.close(root_fd)

        path = root_path / experiment / run
        manifest: dict[str, object] = {
            "schema": _MANIFEST_SCHEMA,
            "experiment_id": experiment,
            "run_id": run,
            "status": "running",
            "created_utc": _utc_now(),
        }
        bundle = cls(path, run_fd, manifest)
        try:
            bundle._replace_internal("manifest.json", _json_bytes(manifest, "manifest"))
        except BaseException:
            os.close(run_fd)
            raise
        return bundle

    def __del__(self) -> None:
        descriptor = getattr(self, "_directory_fd", -1)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._directory_fd = -1

    @property
    def path(self) -> Path:
        return self._path

    @property
    def status(self) -> str:
        return self._status

    def _ensure_running(self) -> None:
        if self._status != "running":
            raise ContractError(
                "run bundle is already finalized or finalizing: "
                f"{self._status}"
            )

    def _require_path_identity(self) -> None:
        expected = os.fstat(self._directory_fd)
        try:
            observed = self._path.lstat()
        except OSError as error:
            raise ContractError(
                "run path identity is missing or inaccessible"
            ) from error
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ContractError("run path identity was replaced")
        descriptor = -1
        try:
            descriptor = os.open(self._path, _directory_flags())
            opened = os.fstat(descriptor)
        except OSError as error:
            raise ContractError("run path identity cannot be opened safely") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if (
            _inode_identity(observed) != _inode_identity(opened)
            or _inode_identity(expected) != _inode_identity(opened)
        ):
            raise ContractError(
                "run path identity no longer matches the retained directory"
            )

    def _open_parent(
        self, parts: tuple[str, ...], *, create: bool
    ) -> tuple[int, str]:
        descriptor = os.dup(self._directory_fd)
        try:
            for part in parts[:-1]:
                try:
                    child = os.open(part, _directory_flags(), dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(part, mode=0o755, dir_fd=descriptor)
                    os.fsync(descriptor)
                    child = os.open(part, _directory_flags(), dir_fd=descriptor)
                except OSError as error:
                    raise ContractError(
                        f"output path contains a symlink or non-directory: {part}"
                    ) from error
                os.close(descriptor)
                descriptor = child
            return descriptor, parts[-1]
        except BaseException:
            os.close(descriptor)
            raise

    def _create_output(self, relative: str | os.PathLike[str], data: bytes) -> None:
        parts = _relative_parts(relative)
        if parts[0] in _RESERVED_OUTPUTS:
            raise ContractError(f"output path is reserved: {parts[0]}")
        parent, filename = self._open_parent(parts, create=True)
        descriptor = -1
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                descriptor = os.open(filename, flags, 0o644, dir_fd=parent)
            except FileExistsError as error:
                raise ContractError(f"evidence output already exists: {relative}") from error
            except OSError as error:
                raise ContractError(f"cannot create confined output: {relative}") from error
            _write_all(descriptor, data)
            os.fsync(descriptor)
            os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def write_bytes(
        self, relative: str | os.PathLike[str], data: bytes | bytearray | memoryview
    ) -> Path:
        self._ensure_running()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ContractError("evidence data must be bytes")
        self._create_output(relative, bytes(data))
        return self._path.joinpath(*_relative_parts(relative))

    def write_text(self, relative: str | os.PathLike[str], text: str) -> Path:
        if type(text) is not str:
            raise ContractError("evidence text must be a string")
        return self.write_bytes(relative, text.encode("utf-8"))

    def output_exists(self, relative: str | os.PathLike[str]) -> bool:
        """Check an output through the retained bundle descriptor."""

        parts = _relative_parts(relative)
        try:
            parent, filename = self._open_parent(parts, create=False)
        except FileNotFoundError:
            return False
        try:
            try:
                os.stat(filename, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return False
            except OSError as error:
                raise ContractError(
                    f"cannot inspect confined output: {relative}"
                ) from error
            return True
        finally:
            os.close(parent)

    def archive_transmission(
        self,
        message: bytes | bytearray | memoryview,
        *,
        first_frame_index: int,
        last_frame_index: int,
        phase: str = "timeline",
        attempt: int | None = None,
    ) -> Mapping[str, object]:
        """Confine one exact packed message and its digest to this run."""

        self._ensure_running()
        if not isinstance(message, (bytes, bytearray, memoryview)):
            raise ContractError("transmitted message must be bytes")
        if (
            type(first_frame_index) is not int
            or type(last_frame_index) is not int
            or first_frame_index < 0
            or last_frame_index < first_frame_index
        ):
            raise ContractError(
                "transmitted frame range must be nonnegative and ordered"
            )
        if phase == "timeline":
            if attempt is not None:
                raise ContractError("timeline transmission phase cannot have an attempt")
        elif phase == "readiness":
            if type(attempt) is not int or attempt <= 0:
                raise ContractError(
                    f"{phase} transmission phase requires a positive attempt"
                )
        elif phase in ("logical", "padding", "receipt_fence"):
            if attempt is not None:
                raise ContractError(
                    f"{phase} transmission phase cannot have an attempt"
                )
        else:
            raise ContractError(
                "transmission phase must be timeline, readiness, "
                "logical, padding, or receipt_fence"
            )
        payload = bytes(message)
        digest = hashlib.sha256(payload).hexdigest()
        stem = f"{first_frame_index:06d}-{last_frame_index:06d}"
        if phase == "readiness":
            assert attempt is not None
            leaf_stem = f"attempt-{attempt:06d}__{stem}"
            message_path = f"transmitted/{phase}/{leaf_stem}.bin"
            digest_path = f"transmitted/{phase}/{leaf_stem}.sha256"
        elif phase in ("logical", "padding", "receipt_fence"):
            leaf_stem = stem
            message_path = f"transmitted/{phase}/{leaf_stem}.bin"
            digest_path = f"transmitted/{phase}/{leaf_stem}.sha256"
        else:
            leaf_stem = stem
            message_path = f"transmitted/{leaf_stem}.bin"
            digest_path = f"transmitted/{leaf_stem}.sha256"
        self.write_bytes(message_path, payload)
        self.write_text(
            digest_path,
            f"{digest}  {leaf_stem}.bin\n",
        )
        return MappingProxyType(
            {
                "first_frame_index": first_frame_index,
                "last_frame_index": last_frame_index,
                "phase": phase,
                "attempt": attempt,
                "message_path": message_path,
                "digest_path": digest_path,
                "sha256": digest,
                "size": len(payload),
            }
        )

    def open_input(self, path: str | os.PathLike[str]) -> BinaryIO:
        self._ensure_running()
        candidate = Path(path).expanduser()
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise ContractError(f"input does not exist: {candidate}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"input symlink is not allowed: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ContractError(f"cannot resolve input: {candidate}") from error
        if resolved == self._path or resolved.is_relative_to(self._path):
            raise ContractError("run outputs cannot be reopened as external inputs")
        try:
            descriptor = os.open(resolved, _regular_read_flags())
        except OSError as error:
            raise ContractError(f"cannot open input read-only: {resolved}") from error
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            os.close(descriptor)
            raise ContractError(f"input is not a regular file: {resolved}")
        return os.fdopen(descriptor, "rb", closefd=True)

    def _append_record(self, kind: str, filename: str, record: Mapping[str, object]) -> None:
        self._ensure_running()
        if not isinstance(record, Mapping):
            raise ContractError(f"{kind} record must be a mapping")
        copied = _json_copy(dict(record), f"{kind} record")
        sequence = self._sequence + 1
        payload = _json_bytes(
            {
                "kind": kind,
                "sequence": sequence,
                "recorded_utc": _utc_now(),
                "record": copied,
            },
            f"{kind} record",
        )
        parent, leaf = self._open_parent((filename,), create=True)
        descriptor = -1
        try:
            flags = (
                os.O_WRONLY
                | os.O_APPEND
                | os.O_CREAT
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(leaf, flags, 0o644, dir_fd=parent)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ContractError(
                    f"{kind} evidence target is not a regular file"
                )
            if opened.st_nlink != 1:
                raise ContractError(
                    f"{kind} evidence target is a hard link alias"
                )
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fsync(parent)
        except OSError as error:
            raise ContractError(f"cannot append {kind} evidence") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)
        self._sequence = sequence

    def record_candidate(self, record: Mapping[str, object]) -> None:
        self._append_record("candidate", "candidates.jsonl", record)

    def record_abort(self, record: Mapping[str, object]) -> None:
        self._append_record("abort", "aborts.jsonl", record)

    def record_accept(self, record: Mapping[str, object]) -> None:
        self._append_record("accept", "accepts.jsonl", record)

    def record_timing(self, record: Mapping[str, object]) -> None:
        self._append_record("timing", "timings.jsonl", record)

    def write_readiness(self, readiness: object) -> None:
        """Persist the exact official-log identity and scoring boundary once."""

        self._append_record(
            "readiness",
            "readiness.jsonl",
            {
                "session_id": getattr(readiness, "session_id"),
                "official_log_path": getattr(readiness, "official_log_path"),
                "log_identity": [
                    getattr(readiness, "log_device"),
                    getattr(readiness, "log_inode"),
                ],
                "readiness_log_start_offset": getattr(
                    readiness, "readiness_log_start_offset"
                ),
                "scoring_log_offset": getattr(
                    readiness, "scoring_log_offset"
                ),
                "readiness_attempts": getattr(
                    readiness, "readiness_attempts"
                ),
                "expected_row_sha256": getattr(
                    readiness, "expected_row_sha256"
                ),
            },
        )

    def write_prepared(self, source: object, target: object) -> None:
        """Append the compact source/target identity before local publication."""

        try:
            frame_value = getattr(target, "frame_index")
        except AttributeError:
            frame_value = getattr(getattr(target, "buffer"), "frame_index")
        frames = list(frame_value)
        if (
            not frames
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in frames
            )
        ):
            raise ContractError(
                "prepared target frame indices must be nonempty integers"
            )
        frame_indices = [int(value) for value in frames]
        if frame_indices[0] < 0 or any(
            right != left + 1
            for left, right in zip(frame_indices, frame_indices[1:])
        ):
            raise ContractError(
                "prepared target frame indices must be nonnegative and contiguous"
            )
        hashes = dict(getattr(target, "hashes", {}))
        self.record_candidate(
            {
                "session_id": getattr(source, "session_id"),
                "candidate_id": getattr(source, "candidate_id"),
                "predecessor_id": getattr(source, "predecessor_id"),
                "source_rows": len(getattr(source, "timestamps_s")),
                "accepted_chunk_id": getattr(target, "accepted_chunk_id"),
                "source_candidate_id": getattr(target, "source_candidate_id"),
                "target_frames": [frame_indices[0], frame_indices[-1]],
                "target_hashes": hashes,
            }
        )

    def write_rejection(self, rejection: object) -> None:
        self._append_record(
            "rejection",
            "rejections.jsonl",
            {
                "candidate_id": getattr(rejection, "candidate_id"),
                "failure_site": getattr(rejection, "failure_site"),
                "error_type": getattr(rejection, "error_type"),
                "error_message": getattr(rejection, "error_message"),
                "mm_alive": getattr(rejection, "mm_alive"),
                "abort_required": getattr(rejection, "abort_required"),
                "abort_sent": getattr(rejection, "abort_sent"),
                "timeline_abort_required": getattr(
                    rejection, "timeline_abort_required"
                ),
            },
        )

    def write_abort_decision(self, decision: object) -> None:
        self.record_abort(
            {
                "candidate_id": getattr(decision, "candidate_id"),
                "mm_abort_required": getattr(decision, "mm_abort_required"),
                "mm_abort_sent": getattr(decision, "mm_abort_sent"),
                "timeline_abort_required": getattr(
                    decision, "timeline_abort_required"
                ),
                "timeline_aborted": getattr(decision, "timeline_aborted"),
                "errors": list(getattr(decision, "errors")),
            }
        )

    def write_accepted(self, accepted: object) -> None:
        target = getattr(accepted, "target")
        advance = getattr(accepted, "advance")
        frame_value = getattr(getattr(target, "buffer"), "frame_index")
        frames = list(frame_value)
        self.record_accept(
            {
                "accepted_chunk_id": getattr(target, "accepted_chunk_id"),
                "source_candidate_id": getattr(target, "source_candidate_id"),
                "target_frames": [int(frames[0]), int(frames[-1])],
                "target_hashes": dict(getattr(target, "hashes", {})),
                "advance": {
                    "steps": getattr(advance, "steps"),
                    "sim_time_start_s": getattr(advance, "sim_time_start_s"),
                    "sim_time_end_s": getattr(advance, "sim_time_end_s"),
                    "state_rows": getattr(advance, "state_rows"),
                    "contact_rows": getattr(advance, "contact_rows"),
                },
                "timings_ns": dict(getattr(accepted, "timings_ns")),
                "wall_started_ns": getattr(accepted, "wall_started_ns"),
                "wall_finished_ns": getattr(accepted, "wall_finished_ns"),
            }
        )

    def write_timing(self, timing: object) -> None:
        self.record_timing(
            {
                "candidate_id": getattr(timing, "candidate_id"),
                "durations_ns": dict(getattr(timing, "durations_ns")),
                "failed_stage": getattr(timing, "failed_stage"),
                "wall_started_ns": getattr(timing, "wall_started_ns"),
                "wall_finished_ns": getattr(timing, "wall_finished_ns"),
            }
        )

    def write_terminal_verdict(self, verdict: object) -> None:
        audit = getattr(verdict, "delivery_audit")
        audit_value = None
        if audit is not None:
            audit_value = {
                "expected_rows": getattr(audit, "expected_rows"),
                "observed_rows": getattr(audit, "observed_rows"),
                "transmitted_indices_exact": getattr(
                    audit, "transmitted_indices_exact"
                ),
                "official_rows_exact": getattr(audit, "official_rows_exact"),
                "evidence_sha256": getattr(audit, "evidence_sha256"),
            }
        payload = {
            "schema": "mm-sonic-terminal-verdict/v1",
            "status": getattr(verdict, "status"),
            "failure_phase": getattr(verdict, "failure_phase"),
            "failure_site": getattr(verdict, "failure_site"),
            "error_type": getattr(verdict, "error_type"),
            "error_message": getattr(verdict, "error_message"),
            "accepted_chunks": getattr(verdict, "accepted_chunks"),
            "simulation_paused": getattr(verdict, "simulation_paused"),
            "delivery_audit_required": getattr(
                verdict, "delivery_audit_required"
            ),
            "delivery_audit_completed": getattr(
                verdict, "delivery_audit_completed"
            ),
            "delivery_audit": audit_value,
        }
        self.write_bytes(
            "terminal-verdict.json", _json_bytes(payload, "terminal verdict")
        )

    def update_manifest(self, fields: Mapping[str, object]) -> None:
        self._ensure_running()
        if not isinstance(fields, Mapping):
            raise ContractError("manifest update must be a mapping")
        immutable = {
            "schema",
            "experiment_id",
            "run_id",
            "status",
            "created_utc",
            "finalization_started_utc",
            "finalized_utc",
            "outcome",
            "evidence",
        }
        overlap = immutable.intersection(fields)
        if overlap:
            raise ContractError(f"manifest update contains reserved keys: {sorted(overlap)}")
        copied = _json_copy(dict(fields), "manifest update")
        assert isinstance(copied, dict)
        candidate = dict(self._manifest)
        candidate.update(copied)
        self._replace_internal("manifest.json", _json_bytes(candidate, "manifest"))
        self._manifest = candidate

    def _replace_internal(
        self, filename: str, data: bytes, *, mode: int = 0o644
    ) -> None:
        temporary = f".{filename}.{uuid.uuid4().hex}.tmp"
        descriptor = -1
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, mode, dir_fd=self._directory_fd)
            _write_all(descriptor, data)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(
                temporary,
                filename,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            os.fsync(self._directory_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=self._directory_fd)
            except FileNotFoundError:
                pass

    def _inventory(
        self, terminal_manifest_core_sha256: str
    ) -> dict[str, object]:
        files: dict[str, object] = {}
        directories: set[str] = set()
        for kind, relative, descriptor, _ in _walk_directory_fd(
            self._directory_fd
        ):
            parts = _relative_parts(relative)
            if "/".join(parts) != relative:
                raise ContractError(f"evidence path is not canonical: {relative}")
            if kind == "directory":
                directories.add(relative)
                continue
            if relative in _RESERVED_OUTPUTS:
                continue
            digest, size = _sha256_descriptor(descriptor, relative)
            files[relative] = {"sha256": digest, "size": size}
        expected_directories: set[str] = set()
        for relative in files:
            parts = _relative_parts(relative)
            expected_directories.update(
                "/".join(parts[:index]) for index in range(1, len(parts))
            )
        extra_directories = sorted(directories - expected_directories)
        if extra_directories:
            raise ContractError(
                f"run contains unregistered directories: {extra_directories}"
            )
        return {
            "schema": _INVENTORY_SCHEMA,
            "terminal_manifest_core_sha256": terminal_manifest_core_sha256,
            "files": files,
        }

    def _seal_evidence(self) -> None:
        for kind, relative, descriptor, metadata in _walk_directory_fd(
            self._directory_fd
        ):
            if kind == "directory":
                continue
            if metadata.st_nlink != 1:
                raise ContractError(f"evidence is a hard link alias: {relative}")
            os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode) & ~0o222)
            os.fsync(descriptor)
        os.fsync(self._directory_fd)

    def finalize(
        self,
        status: str,
        *,
        outcome: Mapping[str, object],
    ) -> Mapping[str, object]:
        self._ensure_running()
        if status not in _TERMINAL_STATUSES:
            raise ContractError(
                "terminal status must be complete, failed, or not_run"
            )
        if not isinstance(outcome, Mapping):
            raise ContractError("run outcome must be a mapping")
        _validate_terminal_metadata(self._manifest, status)
        copied_outcome = _json_copy(dict(outcome), "run outcome")
        assert isinstance(copied_outcome, dict)

        finalizing_manifest = dict(self._manifest)
        finalizing_manifest.update(
            {
                "status": "finalizing",
                "finalization_started_utc": _utc_now(),
            }
        )
        self._status = "finalizing"
        self._manifest = finalizing_manifest
        self._replace_internal(
            "manifest.json",
            _json_bytes(finalizing_manifest, "finalizing manifest"),
        )

        self._require_path_identity()
        terminal_core = dict(finalizing_manifest)
        terminal_core.update(
            {
                "status": status,
                "finalized_utc": _utc_now(),
                "outcome": copied_outcome,
            }
        )
        terminal_core_sha256 = hashlib.sha256(
            _json_bytes(terminal_core, "terminal manifest core")
        ).hexdigest()
        inventory = self._inventory(terminal_core_sha256)
        inventory_bytes = _json_bytes(inventory, "evidence inventory")
        self._replace_internal("inventory.json", inventory_bytes)
        self._seal_evidence()
        evidence = {
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "file_count": len(inventory["files"]),
        }
        terminal_manifest = dict(terminal_core)
        terminal_manifest["evidence"] = evidence
        _validate_terminal_manifest(terminal_manifest)
        self._replace_internal(
            "manifest.json",
            _json_bytes(terminal_manifest, "terminal manifest"),
            mode=0o444,
        )
        self._manifest = terminal_manifest
        self._status = status
        return MappingProxyType(inventory)


def verify_run_inventory(path: str | os.PathLike[str]) -> bool:
    root = Path(path).expanduser()
    try:
        observed_root = root.lstat()
    except OSError as error:
        raise ContractError(f"cannot inspect run bundle: {root}") from error
    if stat.S_ISLNK(observed_root.st_mode) or not stat.S_ISDIR(
        observed_root.st_mode
    ):
        raise ContractError(f"run bundle is not a safe directory: {root}")

    root_fd = -1
    try:
        root_fd = os.open(root, _directory_flags())
        opened_root = os.fstat(root_fd)
        if _inode_identity(observed_root) != _inode_identity(opened_root):
            raise ContractError("run bundle changed while opening")

        inventory_bytes = _read_regular_at(root_fd, "inventory.json")
        manifest_bytes = _read_regular_at(root_fd, "manifest.json")
        inventory = json.loads(inventory_bytes)
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        if root_fd >= 0:
            os.close(root_fd)
            root_fd = -1
        raise ContractError("run inventory or manifest is unreadable") from error
    except BaseException:
        if root_fd >= 0:
            os.close(root_fd)
            root_fd = -1
        raise
    try:
        terminal = _validate_terminal_manifest(manifest)
        if type(inventory) is not dict or set(inventory) != {
            "schema",
            "terminal_manifest_core_sha256",
            "files",
        }:
            raise ContractError("run inventory has invalid keys")
        if inventory["schema"] != _INVENTORY_SCHEMA:
            raise ContractError("run inventory has an unsupported schema")
        core_digest = inventory["terminal_manifest_core_sha256"]
        if type(core_digest) is not str or _SHA256.fullmatch(core_digest) is None:
            raise ContractError(
                "run inventory terminal manifest core SHA-256 is invalid"
            )
        files = inventory["files"]
        if type(files) is not dict:
            raise ContractError("run inventory files must be an object")

        evidence = terminal["evidence"]
        assert isinstance(evidence, dict)
        actual_inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
        if evidence["inventory_sha256"] != actual_inventory_hash:
            raise ContractError("inventory SHA-256 does not match the manifest")
        actual_core_hash = hashlib.sha256(
            _json_bytes(
                _terminal_manifest_core(terminal), "terminal manifest core"
            )
        ).hexdigest()
        if core_digest != actual_core_hash:
            raise ContractError(
                "terminal manifest core SHA-256 does not match the inventory"
            )
        if evidence["file_count"] != len(files):
            raise ContractError(
                "terminal manifest evidence.file_count does not match the inventory"
            )

        registered_files: set[str] = set()
        expected_directories: set[str] = set()
        for relative, registered in files.items():
            if type(relative) is not str or type(registered) is not dict:
                raise ContractError("run inventory entry is invalid")
            parts = _relative_parts(relative)
            if "/".join(parts) != relative or relative in _RESERVED_OUTPUTS:
                raise ContractError(
                    f"run inventory path is not canonical evidence: {relative}"
                )
            if set(registered) != {"sha256", "size"}:
                raise ContractError(f"run inventory entry has invalid keys: {relative}")
            digest = registered["sha256"]
            size = registered["size"]
            if type(digest) is not str or _SHA256.fullmatch(digest) is None:
                raise ContractError(
                    f"run inventory entry has invalid SHA-256: {relative}"
                )
            if type(size) is not int or size < 0:
                raise ContractError(
                    f"run inventory entry has invalid size: {relative}"
                )
            registered_files.add(relative)
            expected_directories.update(
                "/".join(parts[:index]) for index in range(1, len(parts))
            )

        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        for kind, relative, descriptor, _ in _walk_directory_fd(root_fd):
            if kind == "directory":
                actual_directories.add(relative)
                continue
            actual_files.add(relative)
            if relative not in registered_files:
                continue
            digest, size = _sha256_descriptor(descriptor, relative)
            registered = files[relative]
            assert isinstance(registered, dict)
            if registered["sha256"] != digest:
                raise ContractError(f"evidence SHA-256 mismatch: {relative}")
            if registered["size"] != size:
                raise ContractError(f"evidence size mismatch: {relative}")

        expected_files = registered_files | _RESERVED_OUTPUTS
        extra_files = sorted(actual_files - expected_files)
        if extra_files:
            raise ContractError(
                f"run contains unregistered evidence: {extra_files}"
            )
        missing_files = sorted(expected_files - actual_files)
        if missing_files:
            raise ContractError(
                f"run is missing registered evidence: {missing_files}"
            )
        extra_directories = sorted(actual_directories - expected_directories)
        if extra_directories:
            raise ContractError(
                f"run contains unregistered directories: {extra_directories}"
            )
        missing_directories = sorted(expected_directories - actual_directories)
        if missing_directories:
            raise ContractError(
                f"run is missing registered directories: {missing_directories}"
            )
        return True
    finally:
        if root_fd >= 0:
            os.close(root_fd)
