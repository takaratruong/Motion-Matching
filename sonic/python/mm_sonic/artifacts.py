"""Confined, append-only evidence bundles for MM-to-SONIC runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
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
_RESERVED_OUTPUTS = frozenset(("manifest.json", "inventory.json"))


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


def _sha256_regular(path: Path) -> tuple[str, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ContractError(f"cannot inspect evidence file: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"evidence path is a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"evidence path is not a regular file: {path}")
    try:
        descriptor = os.open(path, _regular_read_flags())
    except OSError as error:
        raise ContractError(f"cannot open evidence read-only: {path}") from error
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or size != before.st_size:
            raise ContractError(f"evidence changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


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
            raise ContractError(f"run bundle is already finalized as {self._status}")

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
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(leaf, flags, 0o644, dir_fd=parent)
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

    def _replace_internal(self, filename: str, data: bytes) -> None:
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
            descriptor = os.open(temporary, flags, 0o644, dir_fd=self._directory_fd)
            _write_all(descriptor, data)
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

    def _inventory(self) -> dict[str, object]:
        files: dict[str, object] = {}
        for candidate in sorted(self._path.rglob("*")):
            relative = candidate.relative_to(self._path).as_posix()
            metadata = candidate.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(f"evidence tree contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if relative in _RESERVED_OUTPUTS:
                continue
            digest, size = _sha256_regular(candidate)
            files[relative] = {"sha256": digest, "size": size}
        return {"schema": _INVENTORY_SCHEMA, "files": files}

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
        missing = sorted(_TERMINAL_MANIFEST_KEYS - set(self._manifest))
        if missing:
            raise ContractError(f"terminal manifest is missing fields: {missing}")
        artifact_hashes = self._manifest["artifact_hashes"]
        if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != _ARTIFACT_HASH_KEYS:
            raise ContractError("terminal manifest artifact_hashes has invalid keys")
        if any(
            value is not None
            and (type(value) is not str or _SHA256.fullmatch(value) is None)
            for value in artifact_hashes.values()
        ):
            raise ContractError(
                "terminal manifest artifact_hashes must be SHA-256 digests or null"
            )
        copied_outcome = _json_copy(dict(outcome), "run outcome")
        assert isinstance(copied_outcome, dict)

        inventory = self._inventory()
        inventory_bytes = _json_bytes(inventory, "evidence inventory")
        self._replace_internal("inventory.json", inventory_bytes)
        evidence = {
            "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            "file_count": len(inventory["files"]),
        }
        terminal_manifest = dict(self._manifest)
        terminal_manifest.update(
            {
                "status": status,
                "finalized_utc": _utc_now(),
                "outcome": copied_outcome,
                "evidence": evidence,
            }
        )
        self._replace_internal(
            "manifest.json", _json_bytes(terminal_manifest, "terminal manifest")
        )

        for candidate in sorted(self._path.rglob("*")):
            metadata = candidate.lstat()
            if stat.S_ISREG(metadata.st_mode):
                candidate.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        os.fsync(self._directory_fd)
        self._manifest = terminal_manifest
        self._status = status
        return MappingProxyType(inventory)


def verify_run_inventory(path: str | os.PathLike[str]) -> bool:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ContractError(f"run bundle is not a directory: {root}")
    inventory_path = root / "inventory.json"
    manifest_path = root / "manifest.json"
    try:
        inventory_bytes = inventory_path.read_bytes()
        inventory = json.loads(inventory_bytes)
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("run inventory or manifest is unreadable") from error
    if type(inventory) is not dict or set(inventory) != {"schema", "files"}:
        raise ContractError("run inventory has invalid keys")
    if inventory["schema"] != _INVENTORY_SCHEMA or type(inventory["files"]) is not dict:
        raise ContractError("run inventory has an unsupported schema")
    expected_inventory_hash = manifest.get("evidence", {}).get("inventory_sha256")
    actual_inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    if expected_inventory_hash != actual_inventory_hash:
        raise ContractError("inventory SHA-256 does not match the manifest")
    for relative, registered in inventory["files"].items():
        if type(relative) is not str or type(registered) is not dict:
            raise ContractError("run inventory entry is invalid")
        parts = _relative_parts(relative)
        candidate = root.joinpath(*parts)
        digest, size = _sha256_regular(candidate)
        if registered.get("sha256") != digest:
            raise ContractError(f"evidence SHA-256 mismatch: {relative}")
        if registered.get("size") != size:
            raise ContractError(f"evidence size mismatch: {relative}")
    actual_files: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        metadata = candidate.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"run evidence contains a symlink: {relative}")
        if stat.S_ISREG(metadata.st_mode) and relative not in _RESERVED_OUTPUTS:
            actual_files.add(relative)
    registered_files = set(inventory["files"])
    extra = sorted(actual_files - registered_files)
    if extra:
        raise ContractError(f"run contains unregistered evidence: {extra}")
    missing = sorted(registered_files - actual_files)
    if missing:
        raise ContractError(f"run is missing registered evidence: {missing}")
    return True
