#!/usr/bin/env python3
"""Pin, acquire, and verify the GRAIL terrain inputs without loading them."""

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath


MANIFEST_SCHEMA = "g1-grail-terrain-inputs/v1"
INVENTORY_SCHEMA = "g1-grail-terrain-inventory/v1"
HASH_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ENTRY_KEYS = {"bytes", "path", "sha256"}
_REPOSITORY_KEYS = {"id", "revision", "type"}
_MANIFEST_KEYS = {"modalities", "repository", "schema", "source_coverage"}
_MODALITY_KEYS = {
    "allowed_globs", "byte_count", "canonical_inventory_sha256",
    "file_count", "partitions",
}
_PARTITION_KEYS = {"byte_count", "file_count", "path_prefix"}
_SOURCE_COVERAGE_KEYS = {
    "canonical_basename_sha256", "file_count", "modalities",
}
_SOURCE_COVERAGE_MODALITY_KEYS = {"path_prefix", "suffix"}
_SOURCE_COVERAGE_MODALITIES = {"object_usd", "objects", "robot"}


def canonical_json_bytes(value):
    return (json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def _canonical_entry(value):
    if type(value) is not dict or set(value) != _ENTRY_KEYS:
        raise ValueError(
            f"inventory entry keys must be {sorted(_ENTRY_KEYS)}")
    path = value["path"]
    if type(path) is not str or not path or "\\" in path or "\x00" in path:
        raise ValueError("inventory path must be a relative POSIX path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts) \
            or pure.as_posix() != path:
        raise ValueError("inventory path must be a normalized relative POSIX path")
    byte_count = value["bytes"]
    if type(byte_count) is not int:
        raise TypeError("inventory byte size must be an integer")
    if byte_count < 0:
        raise ValueError("inventory byte size must be nonnegative")
    digest = value["sha256"]
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise ValueError("inventory SHA-256 must be 64 lowercase hex digits")
    return {"bytes": byte_count, "path": path, "sha256": digest}


def _canonical_entries(entries):
    if not isinstance(entries, (list, tuple)):
        raise TypeError("inventory entries must be a list or tuple")
    normalized = [_canonical_entry(value) for value in entries]
    paths = [value["path"] for value in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate inventory path")
    return sorted(normalized, key=lambda value: value["path"])


def canonical_inventory_bytes(entries):
    """Encode sorted {bytes,path,sha256} entries as canonical JSON."""
    return canonical_json_bytes(_canonical_entries(entries))


def canonical_inventory_sha256(entries):
    return hashlib.sha256(canonical_inventory_bytes(entries)).hexdigest()


def canonical_basenames_bytes(basenames):
    if not isinstance(basenames, (list, tuple)):
        raise TypeError("source basenames must be a list or tuple")
    normalized = []
    for basename in basenames:
        if type(basename) is not str:
            raise TypeError("source basename must be a string")
        pure = PurePosixPath(basename)
        if not basename or pure.name != basename or pure.suffix \
                or "\\" in basename or "\x00" in basename:
            raise ValueError("source basename must be an extensionless file basename")
        normalized.append(basename)
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate source basename")
    return canonical_json_bytes(sorted(normalized))


def canonical_basenames_sha256(basenames):
    return hashlib.sha256(canonical_basenames_bytes(basenames)).hexdigest()


def sha256_stream(stream, chunk_size=HASH_CHUNK_BYTES):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("hash chunk size must be a positive integer")
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def sha256_file(path):
    with open(path, "rb") as stream:
        return sha256_stream(stream)


def _require_exact_keys(value, expected, label):
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label} keys must be {sorted(expected)}")


def _require_count(value, label):
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _glob_prefix(pattern):
    wildcard_positions = [
        position for token in "*?["
        if (position := pattern.find(token)) >= 0
    ]
    end = min(wildcard_positions) if wildcard_positions else len(pattern)
    return pattern[:end].rstrip("/")


def _validate_repository(repository):
    _require_exact_keys(repository, _REPOSITORY_KEYS, "repository")
    if type(repository["id"]) is not str or not repository["id"]:
        raise ValueError("repository id must be a nonempty string")
    if type(repository["revision"]) is not str \
            or not _REVISION.fullmatch(repository["revision"]):
        raise ValueError("repository revision must be a pinned lowercase commit")
    if repository["type"] != "dataset":
        raise ValueError("repository type must be dataset")


def validate_manifest_data(value):
    _require_exact_keys(value, _MANIFEST_KEYS, "manifest")
    if value["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    _validate_repository(value["repository"])
    modalities = value["modalities"]
    if type(modalities) is not dict or not modalities:
        raise ValueError("manifest modalities must be a nonempty object")
    for name, config in modalities.items():
        if type(name) is not str or not name:
            raise ValueError("modality names must be nonempty strings")
        _require_exact_keys(config, _MODALITY_KEYS, f"{name} modality")
        globs = config["allowed_globs"]
        if type(globs) is not list or not globs \
                or any(type(pattern) is not str or not pattern for pattern in globs):
            raise ValueError(f"{name} allowed globs must be nonempty strings")
        if globs != sorted(set(globs)):
            raise ValueError(f"{name} allowed globs must be unique and lexical")
        for pattern in globs:
            prefix = _glob_prefix(pattern)
            if not prefix or pattern.startswith("/") or "\\" in pattern \
                    or ".." in PurePosixPath(prefix).parts:
                raise ValueError(f"{name} allowed glob must be a relative POSIX path")
        partitions = config["partitions"]
        if type(partitions) is not dict or not partitions:
            raise ValueError(f"{name} partitions must be a nonempty object")
        partition_files = 0
        partition_bytes = 0
        for partition_name, partition in partitions.items():
            if type(partition_name) is not str or not partition_name:
                raise ValueError("partition names must be nonempty strings")
            _require_exact_keys(
                partition, _PARTITION_KEYS, f"{name}.{partition_name}")
            prefix = partition["path_prefix"]
            if type(prefix) is not str or not prefix.endswith("/") \
                    or prefix.startswith("/") or "\\" in prefix \
                    or ".." in PurePosixPath(prefix).parts:
                raise ValueError("partition path prefix must be relative POSIX")
            _require_count(partition["file_count"], "partition file count")
            _require_count(partition["byte_count"], "partition byte count")
            partition_files += partition["file_count"]
            partition_bytes += partition["byte_count"]
        _require_count(config["file_count"], f"{name} file count")
        _require_count(config["byte_count"], f"{name} byte count")
        if partition_files != config["file_count"]:
            raise ValueError(f"{name} partition file counts do not match total")
        if partition_bytes != config["byte_count"]:
            raise ValueError(f"{name} partition byte counts do not match total")
        digest = config["canonical_inventory_sha256"]
        if type(digest) is not str or not _SHA256.fullmatch(digest):
            raise ValueError(
                f"{name} canonical inventory SHA-256 must be lowercase hex")
    source_coverage = value["source_coverage"]
    if type(source_coverage) is not dict:
        raise TypeError("source coverage must be an object")
    for source, coverage in source_coverage.items():
        if type(source) is not str or not source:
            raise ValueError("source coverage names must be nonempty strings")
        _require_exact_keys(
            coverage, _SOURCE_COVERAGE_KEYS, f"{source} source coverage")
        _require_count(
            coverage["file_count"], f"{source} source coverage file count")
        if coverage["file_count"] == 0:
            raise ValueError("source coverage file count must be positive")
        digest = coverage["canonical_basename_sha256"]
        if type(digest) is not str or not _SHA256.fullmatch(digest):
            raise ValueError(
                f"{source} canonical basename SHA-256 must be lowercase hex")
        coverage_modalities = coverage["modalities"]
        _require_exact_keys(
            coverage_modalities, _SOURCE_COVERAGE_MODALITIES,
            f"{source} source coverage modalities")
        prefixes = []
        for name, source_modality in coverage_modalities.items():
            _require_exact_keys(
                source_modality, _SOURCE_COVERAGE_MODALITY_KEYS,
                f"{source}.{name} source coverage modality")
            prefix = source_modality["path_prefix"]
            suffix = source_modality["suffix"]
            if type(prefix) is not str or not prefix.endswith("/") \
                    or prefix.startswith("/") or "\\" in prefix \
                    or ".." in PurePosixPath(prefix).parts:
                raise ValueError(
                    "source coverage path prefix must be relative POSIX")
            if type(suffix) is not str or not suffix.startswith(".") \
                    or suffix.count(".") != 1 or "/" in suffix \
                    or "\\" in suffix or len(suffix) == 1:
                raise ValueError("source coverage suffix must be a file suffix")
            prefixes.append(prefix)
            if name in modalities and source in modalities[name]["partitions"]:
                locked_prefix = modalities[name]["partitions"][source][
                    "path_prefix"]
                if prefix != locked_prefix:
                    raise ValueError(
                        f"{source}.{name} source coverage partition changed")
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("source coverage path prefixes must be unique")
    return json.loads(canonical_json_bytes(value))


def _read_canonical_json(path, label):
    path = Path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing {label}: {path}") from None
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} may not be a symlink: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON: {path}")
    return value


def load_manifest(path):
    return validate_manifest_data(_read_canonical_json(path, "manifest"))


def _selected_modalities(manifest, modalities):
    if not isinstance(modalities, (list, tuple)) or not modalities:
        raise ValueError("at least one modality is required")
    if any(type(name) is not str for name in modalities):
        raise TypeError("modality names must be strings")
    if len(modalities) != len(set(modalities)):
        raise ValueError("duplicate modality selection")
    unknown = sorted(set(modalities) - set(manifest["modalities"]))
    if unknown:
        raise ValueError(f"unknown modality selection: {unknown}")
    return tuple(sorted(modalities))


def _path_matches(path, pattern):
    return PurePosixPath(path).match(pattern)


def _modality_for_path(manifest, path):
    matches = [
        name for name, config in manifest["modalities"].items()
        if any(_path_matches(path, pattern)
               for pattern in config["allowed_globs"])
    ]
    if len(matches) != 1:
        raise ValueError(f"unexpected or ambiguous inventory path: {path}")
    return matches[0]


def _validate_modality_entries(name, config, entries):
    if len(entries) != config["file_count"]:
        raise ValueError(
            f"{name} file count changed: {len(entries)} != {config['file_count']}")
    byte_count = sum(entry["bytes"] for entry in entries)
    if byte_count != config["byte_count"]:
        raise ValueError(
            f"{name} byte count changed: {byte_count} != {config['byte_count']}")
    partition_entries = {partition: [] for partition in config["partitions"]}
    for entry in entries:
        matches = [
            partition for partition, locked in config["partitions"].items()
            if entry["path"].startswith(locked["path_prefix"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{name} inventory path has no unique partition: {entry['path']}")
        partition_entries[matches[0]].append(entry)
    for partition, locked in config["partitions"].items():
        observed = partition_entries[partition]
        if len(observed) != locked["file_count"]:
            raise ValueError(f"{name}.{partition} file count changed")
        if sum(entry["bytes"] for entry in observed) != locked["byte_count"]:
            raise ValueError(f"{name}.{partition} byte count changed")
    digest = canonical_inventory_sha256(entries)
    if digest != config["canonical_inventory_sha256"]:
        raise ValueError(
            f"{name} canonical inventory SHA-256 changed: {digest}")


def _coverage_entry_basenames(entries, source_modality):
    prefix = source_modality["path_prefix"]
    suffix = source_modality["suffix"]
    basenames = []
    for entry in entries:
        path = entry["path"]
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix):]
        if "/" in relative or not relative.endswith(suffix):
            raise ValueError(f"invalid source coverage input: {path}")
        basenames.append(relative[:-len(suffix)])
    return basenames


def _validate_source_coverage_inventory(manifest, by_modality):
    for source, coverage in manifest["source_coverage"].items():
        for name, source_modality in coverage["modalities"].items():
            if name not in by_modality or name not in manifest["modalities"]:
                continue
            if source not in manifest["modalities"][name]["partitions"]:
                continue
            basenames = _coverage_entry_basenames(
                by_modality[name], source_modality)
            if len(basenames) != coverage["file_count"]:
                raise ValueError(
                    f"{source}.{name} source coverage file count changed")
            digest = canonical_basenames_sha256(basenames)
            if digest != coverage["canonical_basename_sha256"]:
                raise ValueError(
                    f"{source}.{name} source coverage basename SHA-256 changed")


def validate_inventory_document(manifest, inventory, required_modalities):
    manifest = validate_manifest_data(manifest)
    _require_exact_keys(
        inventory, {"files", "modalities", "repository", "schema"},
        "inventory")
    if inventory["schema"] != INVENTORY_SCHEMA:
        raise ValueError(f"inventory schema must be {INVENTORY_SCHEMA}")
    _validate_repository(inventory["repository"])
    if inventory["repository"] != manifest["repository"]:
        raise ValueError("inventory repository or revision changed")
    document_modalities = inventory["modalities"]
    if type(document_modalities) is not list \
            or document_modalities != sorted(set(document_modalities)):
        raise ValueError("inventory modalities must be unique and lexical")
    unknown_document = sorted(
        set(document_modalities) - set(manifest["modalities"]))
    if unknown_document:
        raise ValueError(f"inventory has unknown modalities: {unknown_document}")
    if not isinstance(required_modalities, (list, tuple)):
        raise TypeError("required modalities must be a list or tuple")
    if required_modalities:
        selected = _selected_modalities(manifest, required_modalities)
        missing = sorted(set(selected) - set(document_modalities))
        if missing:
            raise ValueError(f"inventory is missing modalities: {missing}")
    files = inventory["files"]
    canonical_files = _canonical_entries(files)
    if files != canonical_files:
        raise ValueError("inventory paths must be in canonical lexical order")
    by_modality = {name: [] for name in document_modalities}
    for entry in canonical_files:
        name = _modality_for_path(manifest, entry["path"])
        if name not in by_modality:
            raise ValueError(
                f"inventory path belongs to undeclared modality {name}: "
                f"{entry['path']}")
        by_modality[name].append(entry)
    for name in document_modalities:
        _validate_modality_entries(
            name, manifest["modalities"][name], by_modality[name])
    _validate_source_coverage_inventory(manifest, by_modality)
    _validate_basename_coverage(manifest, by_modality)
    return {
        "schema": INVENTORY_SCHEMA,
        "repository": dict(inventory["repository"]),
        "modalities": list(document_modalities),
        "files": canonical_files,
    }


def load_inventory(path, manifest, required_modalities):
    return validate_inventory_document(
        manifest, _read_canonical_json(path, "inventory"),
        required_modalities)


def _remote_entry(remote, repository, hf_download=None):
    path = getattr(remote, "path", None)
    size = getattr(remote, "size", None)
    if size is None:
        return None
    if type(size) is not int or size < 0:
        raise ValueError(f"remote byte size changed: {path}")
    lfs = getattr(remote, "lfs", None)
    if lfs is not None:
        lfs_size = getattr(lfs, "size", None)
        if type(lfs_size) is not int or size != lfs_size:
            raise ValueError(f"remote LFS byte size changed: {path}")
        digest = getattr(lfs, "sha256", None)
    else:
        if hf_download is None:
            from huggingface_hub import hf_hub_download
            hf_download = hf_hub_download
        downloaded = Path(hf_download(
            repo_id=repository["id"], filename=path,
            repo_type=repository["type"], revision=repository["revision"]))
        try:
            metadata = downloaded.stat()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"remote geometry download is missing: {path}") from None
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"remote geometry download is not regular: {path}")
        if metadata.st_size != size:
            raise ValueError(
                f"remote regular-file byte size changed: {path}")
        digest = sha256_file(downloaded)
    return _canonical_entry({
        "bytes": size, "path": path, "sha256": digest,
    })


def build_remote_inventory(manifest, modalities, api=None, hf_download=None):
    manifest = validate_manifest_data(manifest)
    selected = _selected_modalities(manifest, modalities)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    entries = []
    visited_prefixes = set()
    repository = manifest["repository"]
    for name in selected:
        config = manifest["modalities"][name]
        target_suffixes = {
            PurePosixPath(pattern).suffix
            for pattern in config["allowed_globs"]
            if PurePosixPath(pattern).suffix
        }
        for pattern in config["allowed_globs"]:
            prefix = _glob_prefix(pattern)
            if prefix in visited_prefixes:
                continue
            visited_prefixes.add(prefix)
            remote_values = api.list_repo_tree(
                repository["id"], path_in_repo=prefix,
                recursive=True, expand=False,
                revision=repository["revision"],
                repo_type=repository["type"])
            for remote in remote_values:
                remote_path = getattr(remote, "path", None)
                if type(remote_path) is not str:
                    continue
                if not any(
                        _path_matches(remote_path, allowed)
                        for allowed in config["allowed_globs"]):
                    if PurePosixPath(remote_path).suffix in target_suffixes:
                        raise ValueError(
                            f"unexpected remote input: {remote_path}")
                    continue
                entry = _remote_entry(
                    remote, repository, hf_download=hf_download)
                if entry is None:
                    continue
                observed_name = _modality_for_path(manifest, entry["path"])
                if observed_name != name:
                    raise ValueError(
                        f"unexpected remote modality at {entry['path']}")
                entries.append(entry)
    document = {
        "schema": INVENTORY_SCHEMA,
        "repository": dict(repository),
        "modalities": list(selected),
        "files": _canonical_entries(entries),
    }
    return validate_inventory_document(manifest, document, selected)


def _ensure_no_symlink_components(root, target):
    root = Path(root)
    target = Path(target)
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing dataset root: {root}") from None
    if stat.S_ISLNK(root_metadata.st_mode):
        raise ValueError(f"dataset root may not be a symlink: {root}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"dataset root must be a directory: {root}")
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise ValueError(f"dataset path escapes root: {target}") from None
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"dataset path contains symlink: {current}")


def _regular_file_metadata(root, entry):
    path = Path(root) / entry["path"]
    _ensure_no_symlink_components(root, path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"missing dataset file: {entry['path']}") from None
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"dataset file may not be a symlink: {entry['path']}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"dataset input must be a regular file: {entry['path']}")
    return path, metadata


def _entries_for_modalities(manifest, inventory, modalities):
    selected = set(_selected_modalities(manifest, modalities))
    return [
        entry for entry in inventory["files"]
        if _modality_for_path(manifest, entry["path"]) in selected
    ]


def _local_source_coverage_basenames(root, source, name, source_modality):
    prefix = source_modality["path_prefix"]
    suffix = source_modality["suffix"]
    directory = Path(root) / prefix.rstrip("/")
    _ensure_no_symlink_components(root, directory)
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"missing {source}.{name} source coverage directory: {prefix}") \
            from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(
            f"{source}.{name} source coverage path must be a directory")
    basenames = []
    for current, directories, files in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for child_name in directories:
            child = current_path / child_name
            child_metadata = child.lstat()
            if stat.S_ISLNK(child_metadata.st_mode):
                raise ValueError(
                    f"{source}.{name} source coverage contains symlink: {child}")
            if not stat.S_ISDIR(child_metadata.st_mode):
                raise ValueError(
                    f"{source}.{name} source coverage contains non-directory: "
                    f"{child}")
        for child_name in files:
            child = current_path / child_name
            child_metadata = child.lstat()
            if child.suffix != suffix:
                continue
            if current_path != directory:
                raise ValueError(
                    f"{source}.{name} source coverage contains nested input: "
                    f"{child}")
            if stat.S_ISLNK(child_metadata.st_mode) \
                    or not stat.S_ISREG(child_metadata.st_mode):
                raise ValueError(
                    f"{source}.{name} source coverage input is not regular: "
                    f"{child}")
            basenames.append(child.stem)
    return basenames


def _verify_source_coverage(manifest, root, is_modality_selected):
    for source, coverage in manifest["source_coverage"].items():
        if not all(
                is_modality_selected(name)
                for name in coverage["modalities"]):
            continue
        for name, source_modality in coverage["modalities"].items():
            basenames = _local_source_coverage_basenames(
                root, source, name, source_modality)
            if len(basenames) != coverage["file_count"]:
                raise ValueError(
                    f"{source}.{name} source coverage file count changed")
            digest = canonical_basenames_sha256(basenames)
            if digest != coverage["canonical_basename_sha256"]:
                raise ValueError(
                    f"{source}.{name} source coverage basename SHA-256 changed")


def _scan_local_inputs(manifest, inventory, root, modalities):
    selected = _selected_modalities(manifest, modalities)
    expected = {
        entry["path"] for entry in _entries_for_modalities(
            manifest, inventory, selected)
    }
    prefixes = sorted({
        locked["path_prefix"].rstrip("/")
        for name in selected
        for locked in manifest["modalities"][name]["partitions"].values()
    })
    selected_patterns = [
        pattern
        for name in selected
        for pattern in manifest["modalities"][name]["allowed_globs"]
    ]
    selected_suffixes = {
        PurePosixPath(pattern).suffix
        for pattern in selected_patterns
        if PurePosixPath(pattern).suffix
    }
    for prefix in prefixes:
        directory = Path(root) / prefix
        _ensure_no_symlink_components(root, directory)
        if not directory.exists():
            continue
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"dataset partition must be a directory: {prefix}")
        for current, directories, files in os.walk(directory, followlinks=False):
            current_path = Path(current)
            for name in list(directories):
                child = current_path / name
                child_metadata = child.lstat()
                if stat.S_ISLNK(child_metadata.st_mode):
                    raise ValueError(f"dataset path contains symlink: {child}")
                if not stat.S_ISDIR(child_metadata.st_mode):
                    raise ValueError(f"dataset path is not a directory: {child}")
            for name in files:
                child = current_path / name
                child_metadata = child.lstat()
                relative = child.relative_to(root).as_posix()
                if stat.S_ISLNK(child_metadata.st_mode):
                    raise ValueError(f"dataset file may not be a symlink: {relative}")
                if not stat.S_ISREG(child_metadata.st_mode):
                    raise ValueError(
                        f"dataset input must be a regular file: {relative}")
                if relative not in expected and any(
                        _path_matches(relative, pattern)
                        for pattern in selected_patterns):
                    raise ValueError(f"unexpected local dataset file: {relative}")
                if relative not in expected \
                        and PurePosixPath(relative).suffix in selected_suffixes:
                    raise ValueError(f"unexpected local dataset file: {relative}")


def _partition_basename_keys(name, config, entries):
    keys = []
    for entry in entries:
        matches = [
            partition
            for partition, locked in config["partitions"].items()
            if entry["path"].startswith(locked["path_prefix"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{name} inventory path has no unique partition: "
                f"{entry['path']}")
        keys.append((matches[0], PurePosixPath(entry["path"]).stem))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} has duplicate partition basename coverage")
    return set(keys)


def _validate_basename_coverage(manifest, by_modality):
    if not {"objects", "object_usd"}.issubset(by_modality):
        return
    object_keys = _partition_basename_keys(
        "objects", manifest["modalities"]["objects"],
        by_modality["objects"])
    geometry_keys = _partition_basename_keys(
        "object_usd", manifest["modalities"]["object_usd"],
        by_modality["object_usd"])
    if object_keys != geometry_keys:
        raise ValueError("object/geometry basename coverage mismatch")
    if "robot" not in by_modality:
        return
    robot_keys = _partition_basename_keys(
        "robot", manifest["modalities"]["robot"], by_modality["robot"])
    terrain_keys = {
        key for key in object_keys
        if key[0] in manifest["modalities"]["robot"]["partitions"]
    }
    if robot_keys != terrain_keys:
        raise ValueError("robot/object/geometry basename coverage mismatch")


def _verify_openable_geometry(root, entries):
    from pxr import Usd, UsdGeom
    for entry in entries:
        path = Path(root) / entry["path"]
        stage = Usd.Stage.Open(str(path))
        if stage is None or not any(
                prim.IsA(UsdGeom.Mesh) for prim in stage.Traverse()):
            raise ValueError(
                f"object USD lacks openable mesh geometry: {entry['path']}")


def verify_local(manifest, inventory, dataset_root, modalities):
    manifest = validate_manifest_data(manifest)
    selected = _selected_modalities(manifest, modalities)
    inventory = validate_inventory_document(manifest, inventory, selected)
    root = Path(dataset_root)
    _verify_source_coverage(manifest, root, set(selected).__contains__)
    _scan_local_inputs(manifest, inventory, root, selected)
    entries = _entries_for_modalities(manifest, inventory, selected)
    for entry in entries:
        path, metadata = _regular_file_metadata(root, entry)
        if metadata.st_size != entry["bytes"]:
            raise ValueError(
                f"dataset byte size changed for {entry['path']}: "
                f"{metadata.st_size} != {entry['bytes']}")
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ValueError(
                f"dataset SHA-256 changed for {entry['path']}: {observed}")
    if "object_usd" in selected:
        geometry_entries = [
            entry for entry in entries
            if _modality_for_path(manifest, entry["path"]) == "object_usd"
        ]
        _verify_openable_geometry(root, geometry_entries)
    return {
        "file_count": len(entries),
        "byte_count": sum(entry["bytes"] for entry in entries),
        "canonical_inventory_sha256": canonical_inventory_sha256(entries),
    }


def _prepare_download_parent(root, relative_path):
    current = Path(root)
    for part in PurePosixPath(relative_path).parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"download path contains symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"download parent is not a directory: {current}")


def _download_needed(root, entry):
    path = Path(root) / entry["path"]
    _ensure_no_symlink_components(root, path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return True, False
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"dataset file may not be a symlink: {entry['path']}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"dataset input must be a regular file: {entry['path']}")
    if metadata.st_size != entry["bytes"]:
        return True, True
    if sha256_file(path) != entry["sha256"]:
        return True, True
    return False, False


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def download_inventory(
        manifest, inventory, dataset_root, modalities, inventory_path,
        hf_download=None):
    manifest = validate_manifest_data(manifest)
    selected = _selected_modalities(manifest, modalities)
    inventory = validate_inventory_document(manifest, inventory, selected)
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_no_symlink_components(root, root)
    if hf_download is None:
        from huggingface_hub import hf_hub_download
        hf_download = hf_hub_download
    repository = manifest["repository"]
    for entry in _entries_for_modalities(manifest, inventory, selected):
        _prepare_download_parent(root, entry["path"])
        needed, force = _download_needed(root, entry)
        if not needed:
            continue
        hf_download(
            repo_id=repository["id"], filename=entry["path"],
            repo_type=repository["type"], revision=repository["revision"],
            local_dir=str(root), force_download=force)
        path, metadata = _regular_file_metadata(root, entry)
        if metadata.st_size != entry["bytes"] \
                or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"downloaded file identity changed: {entry['path']}")
    summary = verify_local(manifest, inventory, root, selected)
    atomic_write_json(inventory_path, inventory)
    return summary


def merge_inventory_documents(manifest, existing, update):
    existing = validate_inventory_document(manifest, existing, ())
    update = validate_inventory_document(manifest, update, ())
    by_path = {entry["path"]: entry for entry in existing["files"]}
    for entry in update["files"]:
        prior = by_path.get(entry["path"])
        if prior is not None and prior != entry:
            raise ValueError(f"inventory identity conflict: {entry['path']}")
        by_path[entry["path"]] = entry
    document = {
        "schema": INVENTORY_SCHEMA,
        "repository": dict(manifest["repository"]),
        "modalities": sorted(set(
            existing["modalities"] + update["modalities"])),
        "files": sorted(by_path.values(), key=lambda value: value["path"]),
    }
    return validate_inventory_document(manifest, document, ())


def _inventory_target(dataset_root, value):
    if value is not None:
        return Path(value)
    return Path(dataset_root) / "g1_mm_inventory.json"


def _summary_line(action, summary):
    return (
        f"{action} {summary['file_count']} files, "
        f"{summary['byte_count']} bytes, canonical inventory SHA-256 "
        f"{summary['canonical_inventory_sha256']}"
    )


def _add_common_arguments(parser):
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--inventory")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Acquire pinned GRAIL inputs without loading pickle data")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "download", "verify"):
        _add_common_arguments(commands.add_parser(name))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    selected = _selected_modalities(manifest, args.modalities)
    inventory_path = _inventory_target(args.dataset_root, args.inventory)

    if args.command == "verify":
        inventory = load_inventory(inventory_path, manifest, selected)
        summary = verify_local(
            manifest, inventory, args.dataset_root, selected)
        print(_summary_line("verified", summary))
        return 0

    existing = None
    if inventory_path.exists():
        existing = load_inventory(inventory_path, manifest, ())
    if args.command == "inventory":
        update = build_remote_inventory(manifest, selected)
    else:
        missing = selected if existing is None else tuple(
            name for name in selected if name not in existing["modalities"])
        update = build_remote_inventory(manifest, missing) if missing else None
    if existing is None:
        inventory = update
    elif update is None:
        inventory = existing
    else:
        inventory = merge_inventory_documents(manifest, existing, update)

    if args.command == "inventory":
        atomic_write_json(inventory_path, inventory)
        entries = _entries_for_modalities(manifest, inventory, selected)
        summary = {
            "file_count": len(entries),
            "byte_count": sum(entry["bytes"] for entry in entries),
            "canonical_inventory_sha256":
                canonical_inventory_sha256(entries),
        }
        print(_summary_line("inventoried", summary))
        return 0

    summary = download_inventory(
        manifest, inventory, args.dataset_root, selected, inventory_path)
    print(_summary_line("verified", summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
