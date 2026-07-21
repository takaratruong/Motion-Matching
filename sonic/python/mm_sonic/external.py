"""Fail-closed identity checks for inputs outside the SONIC package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping


LOCK_SCHEMA = "gear-sonic-lock/v1"
LOCK_REPOSITORY = "https://github.com/NVlabs/GR00T-WholeBodyControl.git"
CURRENT_FRAME_ADVANCEMENT_SHA256 = (
    "07f183c4abab53157beba3bbc12bc021789ae00ec8c1b8f69dce5596e33e7f6b"
)

LOCK_PATH_KEYS = (
    "known_good_reference",
    "joint_names_source",
    "policy_parameters_source",
    "zmq_example_source",
    "zmq_decoder_source",
    "stream_merger_source",
    "current_frame_advancement_source",
)

SOURCE_PATH_KEYS = LOCK_PATH_KEYS[1:]

TARGET_TO_SOURCE_PERMUTATION = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

KNOWN_GOOD_REFERENCE_FILES = (
    "body_ang_vel.csv",
    "body_lin_vel.csv",
    "body_pos.csv",
    "body_quat.csv",
    "info.txt",
    "joint_pos.csv",
    "joint_vel.csv",
    "metadata.txt",
)

_LOCK_KEYS = frozenset(
    (
        "schema",
        "repository",
        "commit",
        "current_frame_advancement_sha256",
        *LOCK_PATH_KEYS,
    )
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PERMUTATION_RE = re.compile(
    r"\bmujoco_to_isaaclab\b\s*=\s*\{(?P<values>[^}]*)\}",
    re.MULTILINE,
)


class ExternalInputError(ValueError):
    """An external input cannot be authenticated safely."""


@dataclass(frozen=True)
class ExternalInputs:
    gear_checkout: Path
    policy: Path
    observation_config: Path
    encoder: Path | None
    terrain_dir: Path
    source_mjcf: Path

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> "ExternalInputs":
        """Resolve, type-check, hash, and isolate explicit CLI inputs."""

        gear_checkout = _required_directory(args, "gear_checkout")
        policy = _required_file(args, "policy")
        observation_config = _required_file(args, "observation_config")
        encoder = _optional_file(args, "encoder")
        terrain_dir = _required_directory(args, "terrain_dir")
        source_mjcf = _required_file(args, "source_mjcf")
        output_root = _output_root(args)

        inputs = cls(
            gear_checkout=gear_checkout,
            policy=policy,
            observation_config=observation_config,
            encoder=encoder,
            terrain_dir=terrain_dir,
            source_mjcf=source_mjcf,
        )
        _reject_output_in_inputs(output_root, inputs)

        # Open every explicit file read-only now so a bad or unreadable input
        # fails before any run directory can be created.
        _sha256_file(policy)
        _sha256_file(observation_config)
        if encoder is not None:
            _sha256_file(encoder)
        _sha256_file(source_mjcf)
        _sha256_directory(terrain_dir)
        return inputs


@dataclass(frozen=True)
class VerifiedExternal:
    inputs: ExternalInputs
    gear_commit: str
    hashes: Mapping[str, str]
    known_good_reference: Path


@dataclass(frozen=True)
class VerifiedGearCheckout:
    gear_checkout: Path
    gear_commit: str
    gear_dirty: bool
    hashes: Mapping[str, str]
    known_good_reference: Path


def verify_gear_checkout(
    path: Path,
    lock: Path,
    *,
    scored: bool = True,
) -> VerifiedGearCheckout:
    """Authenticate a GEAR checkout without writing through its path."""

    checkout = _resolve_directory(path, "gear_checkout")
    lock_data = _load_lock(lock)

    top_level = _run_git(checkout, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(top_level).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ExternalInputError(
            f"gear_checkout has an invalid Git top level: {top_level!r}"
        ) from error
    if git_root != checkout:
        raise ExternalInputError(
            f"gear_checkout must be a Git worktree root: {checkout}"
        )

    commit = _run_git(checkout, "rev-parse", "--verify", "HEAD")
    expected_commit = lock_data["commit"]
    if commit != expected_commit:
        raise ExternalInputError(
            "GEAR checkout commit mismatch: "
            f"expected {expected_commit}, found {commit}"
        )

    status = _run_git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    dirty = bool(status)
    if scored and dirty:
        raise ExternalInputError("GEAR checkout is dirty for a scored run")

    hashes: dict[str, str] = {}
    for key in SOURCE_PATH_KEYS:
        source = _checkout_path(checkout, lock_data[key], key, directory=False)
        if key == "policy_parameters_source":
            contents = _read_file(source)
            hashes[f"gear:{key}"] = hashlib.sha256(contents).hexdigest()
            _verify_permutation(source, contents)
        else:
            hashes[f"gear:{key}"] = _sha256_file(source)
        if (
            key == "current_frame_advancement_source"
            and hashes[f"gear:{key}"]
            != lock_data["current_frame_advancement_sha256"]
        ):
            raise ExternalInputError(
                "pinned CurrentFrameAdvancement source hash mismatch: "
                f"expected {lock_data['current_frame_advancement_sha256']}, "
                f"found {hashes[f'gear:{key}']}"
            )

    reference = _checkout_path(
        checkout,
        lock_data["known_good_reference"],
        "known_good_reference",
        directory=True,
    )
    for name in KNOWN_GOOD_REFERENCE_FILES:
        reference_file = _checkout_path(
            reference,
            name,
            f"known_good_reference/{name}",
            directory=False,
        )
        hashes[f"gear:known_good_reference/{name}"] = _sha256_file(
            reference_file
        )

    final_commit = _run_git(checkout, "rev-parse", "--verify", "HEAD")
    final_status = _run_git(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if final_commit != commit or final_status != status:
        raise ExternalInputError("GEAR checkout changed during verification")

    return VerifiedGearCheckout(
        gear_checkout=checkout,
        gear_commit=commit,
        gear_dirty=dirty,
        hashes=MappingProxyType(dict(sorted(hashes.items()))),
        known_good_reference=reference,
    )


def locked_gear_capability_paths(
    gear_checkout: Path,
    lock: Path,
) -> tuple[Path, ...]:
    """Return the exact GEAR files authenticated through the lock contract."""

    lock_data = _load_lock(lock)
    sources = tuple(
        gear_checkout.joinpath(*PurePosixPath(lock_data[key]).parts)
        for key in SOURCE_PATH_KEYS
    )
    reference = gear_checkout.joinpath(
        *PurePosixPath(lock_data["known_good_reference"]).parts
    )
    return sources + tuple(
        reference / name for name in KNOWN_GOOD_REFERENCE_FILES
    )


def verify_external(
    inputs: ExternalInputs,
    lock: Path,
    *,
    scored: bool = True,
) -> VerifiedExternal:
    """Hash all explicit inputs and authenticate their GEAR checkout."""

    if not isinstance(inputs, ExternalInputs):
        raise ExternalInputError("inputs must be an ExternalInputs instance")

    checkout = verify_gear_checkout(inputs.gear_checkout, lock, scored=scored)
    hashes = dict(checkout.hashes)
    hashes.update(
        {
            "policy": _sha256_file(inputs.policy),
            "observation_config": _sha256_file(inputs.observation_config),
            "terrain_dir": _sha256_directory(inputs.terrain_dir),
            "source_mjcf": _sha256_file(inputs.source_mjcf),
        }
    )
    if inputs.encoder is not None:
        hashes["encoder"] = _sha256_file(inputs.encoder)

    return VerifiedExternal(
        inputs=inputs,
        gear_commit=checkout.gear_commit,
        hashes=MappingProxyType(dict(sorted(hashes.items()))),
        known_good_reference=checkout.known_good_reference,
    )


def _required_directory(args: argparse.Namespace, name: str) -> Path:
    return _resolve_directory(_namespace_value(args, name), name)


def _required_file(args: argparse.Namespace, name: str) -> Path:
    return _resolve_file(_namespace_value(args, name), name)


def _optional_file(args: argparse.Namespace, name: str) -> Path | None:
    value = _namespace_value(args, name)
    if value is None:
        return None
    return _resolve_file(value, name)


def _namespace_value(args: argparse.Namespace, name: str):
    if not hasattr(args, name):
        raise ExternalInputError(f"missing explicit --{name.replace('_', '-')}")
    return getattr(args, name)


def _resolve_file(value, label: str) -> Path:
    path = _resolve_existing(value, label)
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ExternalInputError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISREG(mode):
        raise ExternalInputError(f"{label} must be a regular file: {path}")
    return path


def _resolve_directory(value, label: str) -> Path:
    path = _resolve_existing(value, label)
    try:
        mode = path.stat().st_mode
    except OSError as error:
        raise ExternalInputError(f"cannot inspect {label}: {path}") from error
    if not stat.S_ISDIR(mode):
        raise ExternalInputError(f"{label} must be a directory: {path}")
    return path


def _resolve_existing(value, label: str) -> Path:
    if value is None or str(value) == "":
        raise ExternalInputError(f"{label} is required")
    try:
        return Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ExternalInputError(f"{label} does not exist: {value}") from error


def _output_root(args: argparse.Namespace) -> Path:
    value = _namespace_value(args, "output_root")
    if value is None or str(value) == "":
        raise ExternalInputError("output_root is required")
    try:
        output = Path(value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ExternalInputError(f"invalid output_root: {value}") from error
    if output.exists() and not output.is_dir():
        raise ExternalInputError(f"output_root must be a directory: {output}")
    return output


def _reject_output_in_inputs(output: Path, inputs: ExternalInputs) -> None:
    directory_inputs = (inputs.gear_checkout, inputs.terrain_dir)
    for input_directory in directory_inputs:
        if output == input_directory or output.is_relative_to(input_directory):
            raise ExternalInputError(
                f"output_root {output} is nested under input {input_directory}"
            )

    file_inputs = (
        inputs.policy,
        inputs.observation_config,
        inputs.encoder,
        inputs.source_mjcf,
    )
    for input_file in file_inputs:
        if input_file is not None and output == input_file:
            raise ExternalInputError(
                f"output_root {output} duplicates input file {input_file}"
            )


def _load_lock(lock: Path) -> dict[str, str]:
    lock_path = _resolve_file(lock, "gear lock")
    try:
        raw = _read_file(lock_path).decode("utf-8")
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalInputError(f"invalid GEAR lock JSON: {lock_path}") from error
    if not isinstance(data, dict):
        raise ExternalInputError("GEAR lock must contain one JSON object")

    keys = frozenset(data)
    if keys != _LOCK_KEYS:
        missing = sorted(_LOCK_KEYS - keys)
        extra = sorted(keys - _LOCK_KEYS)
        raise ExternalInputError(
            f"GEAR lock keys mismatch; missing={missing}, extra={extra}"
        )
    if data["schema"] != LOCK_SCHEMA:
        raise ExternalInputError(f"unsupported GEAR lock schema: {data['schema']!r}")
    if data["repository"] != LOCK_REPOSITORY:
        raise ExternalInputError("GEAR lock repository does not match the contract")
    if not isinstance(data["commit"], str) or not _COMMIT_RE.fullmatch(
        data["commit"]
    ):
        raise ExternalInputError("GEAR lock commit must be 40 lowercase hex digits")
    advancement_digest = data["current_frame_advancement_sha256"]
    if (
        not isinstance(advancement_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", advancement_digest) is None
    ):
        raise ExternalInputError(
            "GEAR lock CurrentFrameAdvancement digest must be lowercase SHA-256"
        )

    for key in LOCK_PATH_KEYS:
        value = data[key]
        if not isinstance(value, str) or not value:
            raise ExternalInputError(f"GEAR lock {key} must be a relative path")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise ExternalInputError(f"GEAR lock {key} must be a confined path")
    return data


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExternalInputError(f"duplicate JSON key in GEAR lock: {key}")
        result[key] = value
    return result


def _run_git(checkout: Path, *args: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ("git", "-C", str(checkout), *args),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise ExternalInputError("cannot execute Git for GEAR checkout") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ExternalInputError(
            f"Git cannot verify GEAR checkout ({' '.join(args)}): {detail}"
        )
    return result.stdout.strip()


def _checkout_path(
    root: Path,
    relative: str,
    label: str,
    *,
    directory: bool,
) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ExternalInputError(f"missing official {label}: {candidate}") from error
    if not resolved.is_relative_to(root):
        raise ExternalInputError(f"official {label} escapes its checkout: {candidate}")
    if directory:
        return _resolve_directory(resolved, f"official {label}")
    return _resolve_file(resolved, f"official {label}")


def _verify_permutation(parameters: Path, contents: bytes) -> None:
    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ExternalInputError(
            f"policy parameters are not UTF-8: {parameters}"
        ) from error
    declarations = tuple(_PERMUTATION_RE.finditer(text))
    if len(declarations) != 1:
        raise ExternalInputError(
            "pinned mujoco_to_isaaclab permutation declaration is missing or ambiguous"
        )
    values = tuple(
        int(value)
        for value in re.findall(r"(?<![A-Za-z_])-?\d+", declarations[0]["values"])
    )
    if values != TARGET_TO_SOURCE_PERMUTATION:
        raise ExternalInputError(
            "pinned mujoco_to_isaaclab permutation does not match GEAR contract"
        )


def _read_file(path: Path) -> bytes:
    descriptor = _open_read_only(path)
    try:
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        _require_unchanged(path, before, after)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    descriptor = _open_read_only(path)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        _require_unchanged(path, before, after)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _open_read_only(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ExternalInputError(f"cannot open input read-only: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExternalInputError(f"input is not a regular file: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _require_unchanged(path: Path, before, after) -> None:
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
    if identity_before != identity_after:
        raise ExternalInputError(f"input changed while it was being hashed: {path}")


def _sha256_directory(root: Path) -> str:
    directory = _resolve_directory(root, "directory input")
    initial_snapshot = _directory_snapshot(directory)
    digest = hashlib.sha256()
    for relative, kind, _identity in initial_snapshot:
        if not relative:
            continue
        relative_bytes = relative.encode("utf-8")
        digest.update(kind + b"\0" + relative_bytes + b"\0")
        if kind == b"F":
            entry = directory.joinpath(*PurePosixPath(relative).parts)
            digest.update(bytes.fromhex(_sha256_file(entry)))

    if _directory_snapshot(directory) != initial_snapshot:
        raise ExternalInputError(f"directory input changed while hashing: {directory}")
    return digest.hexdigest()


def _directory_snapshot(directory: Path):
    try:
        entries = sorted(
            directory.rglob("*"),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
        root_metadata = directory.lstat()
    except OSError as error:
        raise ExternalInputError(
            f"cannot enumerate directory input: {directory}"
        ) from error

    snapshot = [("", b"D", _metadata_identity(root_metadata))]
    for entry in entries:
        relative = entry.relative_to(directory).as_posix()
        try:
            metadata = entry.lstat()
        except OSError as error:
            raise ExternalInputError(
                f"cannot inspect directory input: {entry}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ExternalInputError(f"directory input contains a symlink: {entry}")
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"F"
        else:
            raise ExternalInputError(
                f"directory input contains a special file: {entry}"
            )
        snapshot.append((relative, kind, _metadata_identity(metadata)))
    return tuple(snapshot)


def _metadata_identity(metadata):
    return (
        metadata.st_mode,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


__all__ = [
    "ExternalInputError",
    "ExternalInputs",
    "VerifiedExternal",
    "VerifiedGearCheckout",
    "locked_gear_capability_paths",
    "verify_external",
    "verify_gear_checkout",
]
