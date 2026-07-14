#!/usr/bin/env python3
"""Independently reload and validate a published G1 terrain artifact set."""

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
from contextlib import contextmanager
from functools import lru_cache

import numpy as np


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from resources.g1_terrain_builder.artifacts import (
    SUPPORT_COLUMNS,
    read_support_sidecar,
    read_terrain_sidecar,
)
from resources.g1_terrain_builder.database import read_holden_database
from resources.g1_terrain_builder.scenes import (
    COORDINATE_SIGNATURE,
    REQUIRED_SCENE_IDS,
    TERRAIN_DISTANCES,
    build_scene,
    procedural_scene_definitions,
)
from resources.g1_terrain_builder.terrain import (
    HEIGHTFIELD_DIAGONAL,
    HEIGHTFIELD_INTERPOLATION,
    GrailTerrain,
    HeightGrid,
    surface_semantics,
    surface_semantics_signature,
)


SCHEMA = "g1-terrain-artifacts/v2"
OUTPUT_FPS = 25.0
FEATURE_DIMENSIONS = 31
TERRAIN_DIMENSIONS = 4
SUPPORT_DIMENSIONS = 3
SCENE_CELL_SIZE = float(np.float32(0.02))
WALKABILITY_HALO = float(np.float32(0.25))
ENDPOINT_RADIUS = float(np.float32(0.20))
MANIFEST_KEYS = {
    "schema", "output_fps", "feature_dimensions", "terrain_dimensions",
    "support_dimensions", "terrain_feature_distances_m", "total_clips",
    "grail_clips", "skipped_clips", "database_frames", "diagnostic_mode",
    "sources", "skeleton", "contact", "surface", "database", "sidecars",
    "scene_index", "validation_file", "validation",
}
SOURCE_KEYS = {
    "name", "terrain_id", "source_fps", "source_frames", "output_frames",
    "range_start", "range_stop", "source_frame_map",
}
SCENE_KEYS = {
    "schema", "id", "label", "provenance", "coordinate_signature",
    "surface_signature", "terrain_feature_distances_m", "heightfield",
    "mesh", "walkability", "bounds", "spawn", "regions", "routes",
}
GRAIL_EXPECTED_BASES = {
    "grail-curb-default": "terrain_curbs__curb_000__000",
    "grail-curb-low": "terrain_curbs__curb_186__004",
    "grail-curb-medium": "terrain_curbs__curb_022__001",
    "grail-curb-high": "terrain_curbs__curb_165__006",
}
GRAIL_LABELS = {
    "grail-curb-default": "GRAIL Default Curb",
    "grail-curb-low": "GRAIL Low Curb",
    "grail-curb-medium": "GRAIL Medium Curb",
    "grail-curb-high": "GRAIL High Curb",
}
GRAIL_TARGETS = {
    "grail-curb-default": None,
    "grail-curb-low": 0.12,
    "grail-curb-medium": 0.24,
    "grail-curb-high": 0.36,
}
GRAIL_EXPECTED_HEIGHTS = {
    "grail-curb-default": 0.2921024334377573,
    "grail-curb-low": 0.12238701526200782,
    "grail-curb-medium": 0.24007104328948528,
    "grail-curb-high": 0.3599740964554129,
}
GRAIL_EXPECTED_CLASSES = {
    "grail-curb-default": 2,
    "grail-curb-low": 1,
    "grail-curb-medium": 2,
    "grail-curb-high": 2,
}
LOCKED_SCENE_IDS = (
    "grail-curb-default",
    "grail-curb-low",
    "grail-curb-medium",
    "grail-curb-high",
    "stairs-shallow",
    "stairs-standard",
    "stairs-unseen-variable",
    "ramp-05-up-down",
    "ramp-10-up-down",
    "ramp-15-stress",
    "cross-slope-05",
    "cross-slope-10",
    "mixed-multilevel",
    "blocked-course",
)
if tuple(REQUIRED_SCENE_IDS) != LOCKED_SCENE_IDS:
    raise RuntimeError("producer required-scene order differs from validator lock")
EXPECTED_ROUTE_IDS = {
    "grail-curb-default": ("curb-forward",),
    "grail-curb-low": ("curb-forward",),
    "grail-curb-medium": ("curb-forward",),
    "grail-curb-high": ("curb-forward",),
    "stairs-shallow": ("ascent-landing-descent",),
    "stairs-standard": ("ascent-landing-descent",),
    "stairs-unseen-variable": ("ascent-landing-descent",),
    "ramp-05-up-down": ("up-landing-down",),
    "ramp-10-up-down": ("up-landing-down",),
    "ramp-15-stress": ("up-landing-down",),
    "cross-slope-05": ("forward-cross-slope",),
    "cross-slope-10": ("forward-cross-slope",),
    "mixed-multilevel": ("full-course",),
    "blocked-course": ("wall-safe-stop", "ramp-safe-stop"),
}
EXPECTED_WAYPOINT_COUNTS = {
    scene_id: (5,) for scene_id in LOCKED_SCENE_IDS
}
EXPECTED_WAYPOINT_COUNTS["mixed-multilevel"] = (8,)
EXPECTED_WAYPOINT_COUNTS["blocked-course"] = (3, 3)
G1_SKELETON_NAMES = (
    "Simulation", "Hips",
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee",
    "LeftAnkle", "LeftToe",
    "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
    "RightAnkle", "RightToe",
    "Spine", "Spine1", "Spine2",
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
    "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWrist",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
    "RightElbow", "RightWristRoll", "RightWristPitch", "RightWrist",
)
G1_SKELETON_PARENTS = (
    -1, 0,
    1, 2, 3, 4, 5, 6,
    1, 8, 9, 10, 11, 12,
    1, 14, 15,
    16, 17, 18, 19, 20, 21, 22,
    16, 24, 25, 26, 27, 28, 29,
)
_HEIGHTFIELD_HEADER = struct.Struct("<4sIII4f")
_WALKABILITY_HEADER = struct.Struct("<4sIII")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_BYTES = 1024 * 1024
_MAX_SCENE_JSON_BYTES = 64 * 1024
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 1_000_000
_MAX_JSON_LIST = 50_000
_MAX_CLIPS = 1770
_MAX_FRAMES = 459682
_MAX_SOURCE_FRAMES = 50_000
_CANONICAL_SOURCE_FPS = (25.0, 50.0)
_MAX_GRID_AXIS = 2048
_MAX_GRID_CELLS = 200_000
_MAX_OBJ_BYTES = 16 * 1024 * 1024
_MAX_DATABASE_BYTES = 176 + 8 * _MAX_CLIPS + 1614 * _MAX_FRAMES
_MAX_FEATURE_BYTES = 16 + 16 * _MAX_FRAMES
_MAX_SUPPORT_BYTES = 16 + 12 * _MAX_FRAMES
_ROOT_FILE_LIMITS = {
    "database.bin": _MAX_DATABASE_BYTES,
    "terrain_features.bin": _MAX_FEATURE_BYTES,
    "terrain_support.bin": _MAX_SUPPORT_BYTES,
    "scenes/index.json": _MAX_SCENE_JSON_BYTES,
    "validation.json": _MAX_VALIDATION_BYTES,
}
_LEGACY_OBJ_INDEX = re.compile(r"[1-9][0-9]*\Z")


def _require(condition, contract):
    if not condition:
        raise ValueError(contract)


def _canonical_json_bytes(value):
    def normalize(child, label="JSON"):
        if type(child) is dict:
            _require(all(type(key) is str for key in child),
                     f"{label} object keys must be strings")
            return {
                key: normalize(item, f"{label}.{key}")
                for key, item in child.items()
            }
        if type(child) in (list, tuple):
            return [
                normalize(item, f"{label}[{index}]")
                for index, item in enumerate(child)
            ]
        if type(child) is float:
            _require(np.isfinite(child), f"{label} must be finite")
            return 0.0 if child == 0.0 else child
        if type(child) is int or type(child) in (str, bool) or child is None:
            return child
        raise ValueError(
            f"{label} contains non-JSON scalar {type(child).__name__}")

    return (json.dumps(
        normalize(value), indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def _json_exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return set(actual) == set(expected) and all(
            _json_exact(actual[key], expected[key]) for key in actual)
    if type(actual) in (list, tuple):
        return len(actual) == len(expected) and all(
            _json_exact(left, right)
            for left, right in zip(actual, expected))
    if type(actual) is float:
        return struct.pack("<d", actual) == struct.pack("<d", expected)
    return actual == expected


def _read_regular_bytes(path, maximum_size, label):
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing regular {label}") from error
    _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
    _require(0 < before.st_size <= maximum_size,
             f"{label} exceeds its byte-size contract")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open regular {label}") from error
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode)
                 and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 f"{label} changed during open")
        _require(0 < opened.st_size <= maximum_size,
                 f"{label} exceeds its byte-size contract")
        chunks = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"{label} changed during read")
            chunks.append(block)
            remaining -= len(block)
        _require(not os.read(descriptor, 1), f"{label} changed during read")
        after = os.fstat(descriptor)
        _require((opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                 == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_json_complexity(value, label):
    nodes = 0
    pending = [(value, 1)]
    while pending:
        child, depth = pending.pop()
        nodes += 1
        _require(nodes <= _MAX_JSON_NODES,
                 f"{label} exceeds the JSON node limit")
        _require(depth <= _MAX_JSON_DEPTH,
                 f"{label} exceeds the JSON depth limit")
        if type(child) is dict:
            pending.extend((item, depth + 1) for item in child.values())
        elif type(child) is list:
            _require(len(child) <= _MAX_JSON_LIST,
                     f"{label} contains an oversized JSON list")
            pending.extend((item, depth + 1) for item in child)


def _read_json(path, maximum_size, label):
    def object_without_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = _read_regular_bytes(path, maximum_size, label)
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")),
        )
    except (
        OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError,
    ) as error:
        raise ValueError(f"{path} is invalid JSON: {error}") from error
    _validate_json_complexity(value, label)
    _require(
        payload == _canonical_json_bytes(value),
        f"{path} is not canonical sorted indented newline JSON")
    return value


def _json_int(value, label, minimum=None):
    _require(type(value) is int, f"{label} must be an integer, not bool")
    if minimum is not None:
        _require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def _json_float(value, label):
    _require(type(value) is float and np.isfinite(value),
             f"{label} must be a finite JSON float")
    _require(value != 0.0 or struct.pack("<d", value) == struct.pack("<d", 0.0),
             f"{label} must encode positive zero")
    return value


def _json_f32(value, label, positive=False):
    value = _json_float(value, label)
    try:
        encoded = struct.pack("<f", value)
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{label} is outside float32") from error
    bits = struct.unpack("<I", encoded)[0]
    exponent = bits & 0x7f800000
    magnitude = bits & 0x7fffffff
    _require(exponent != 0x7f800000 and not (
        magnitude != 0 and exponent == 0),
        f"{label} must be normal-or-positive-zero float32")
    decoded = struct.unpack("<f", encoded)[0]
    _require(float(decoded) == value,
             f"{label} is not an exact promoted float32")
    if positive:
        _require(bits & 0x80000000 == 0 and exponent not in (0, 0x7f800000),
                 f"{label} must be positive-normal float32")
    return value


def _f32_round(value, label):
    try:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError(f"{label} is outside float32") from error
    exponent = bits & 0x7f800000
    magnitude = bits & 0x7fffffff
    _require(exponent != 0x7f800000 and not (
        magnitude != 0 and exponent == 0),
        f"{label} must be normal-or-positive-zero float32")
    decoded = struct.unpack("<f", struct.pack("<I", bits))[0]
    return 0.0 if decoded == 0.0 else float(decoded)


def _f32_add(left, right, label):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) + np.float32(right))
    return _f32_round(value, label)


def _f32_sub(left, right, label):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) - np.float32(right))
    return _f32_round(value, label)


def _f32_mul(left, right, label):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) * np.float32(right))
    return _f32_round(value, label)


def _f32_div(left, right, label):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = np.float32(np.float32(left) / np.float32(right))
    return _f32_round(value, label)


def _f32_sqrt(value, label):
    with np.errstate(invalid="ignore"):
        result = np.float32(np.sqrt(np.float32(value)))
    return _f32_round(result, label)


def _sha256_file(path, maximum_size):
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing regular file {path}") from error
    _require(stat.S_ISREG(before.st_mode), f"non-regular file {path}")
    _require(0 < before.st_size <= maximum_size,
             f"file exceeds hash byte-size contract: {path}")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open regular file {path}") from error
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode)
                 and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 f"file changed during hash open: {path}")
        _require(0 < opened.st_size <= maximum_size,
                 f"file exceeds hash byte-size contract: {path}")
        digest = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"file changed during hash: {path}")
            digest.update(block)
            remaining -= len(block)
        _require(not os.read(descriptor, 1),
                 f"file changed during hash: {path}")
        after = os.fstat(descriptor)
        identity = (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        )
        _require(identity == (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ), f"file changed during hash: {path}")
        return digest.hexdigest(), identity
    finally:
        os.close(descriptor)


def _require_file_identity(path, expected, label):
    try:
        node = os.lstat(path)
    except OSError as error:
        raise ValueError(f"{label} changed after authentication") from error
    actual = (
        node.st_dev, node.st_ino, node.st_size,
        node.st_mtime_ns, node.st_ctime_ns,
    )
    _require(stat.S_ISREG(node.st_mode) and actual == expected,
             f"{label} changed after authentication")


@contextmanager
def _authenticated_reader_path(
    path, expected_digest, expected_size, label, preflight,
):
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing authenticated {label}") from error
    _require(stat.S_ISREG(before.st_mode) and before.st_size == expected_size,
             f"{label} size changed before authenticated read")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open authenticated {label}") from error
    try:
        opened = os.fstat(descriptor)
        _require(stat.S_ISREG(opened.st_mode)
                 and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino)
                 and opened.st_size == expected_size,
                 f"{label} changed during authenticated open")
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            _require(bool(block), f"{label} changed during authenticated read")
            digest.update(block)
            remaining -= len(block)
        _require(not os.read(descriptor, 1),
                 f"{label} changed during authenticated read")
        _require(digest.hexdigest() == expected_digest,
                 f"{label} SHA-256 mismatch")
        preflight(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        yield f"/proc/self/fd/{descriptor}"
        after = os.fstat(descriptor)
        _require((
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        ) == (
            opened.st_dev, opened.st_ino, opened.st_size,
            opened.st_mtime_ns, opened.st_ctime_ns,
        ), f"{label} changed during authenticated parse")
    finally:
        os.close(descriptor)


def _expected_tree():
    files = {
        "database.bin", "terrain_features.bin", "terrain_support.bin",
        "manifest.json", "validation.json", os.path.join("scenes", "index.json"),
    }
    directories = {".", "scenes"}
    for scene_id in LOCKED_SCENE_IDS:
        scene_dir = os.path.join("scenes", scene_id)
        directories.add(scene_dir)
        for name in ("scene.json", "terrain.bin", "terrain.obj", "walkability.bin"):
            files.add(os.path.join(scene_dir, name))
    return files, directories


def _validate_exact_file_tree(root):
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        raise ValueError("artifact directory does not exist") from error
    _require(stat.S_ISDIR(root_stat.st_mode),
             "artifact directory does not exist or is a symlink")
    expected_files, expected_directories = _expected_tree()
    actual_files = set()
    actual_directories = {"."}
    pending = [(root, ".")]
    while pending:
        directory, relative_directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = entry.name if relative_directory == "." else \
                        os.path.join(relative_directory, entry.name)
                    try:
                        node = entry.stat(follow_symlinks=False)
                    except OSError as error:
                        raise ValueError(
                            f"cannot inspect artifact node {relative}") from error
                    if stat.S_ISLNK(node.st_mode):
                        raise ValueError(f"artifact tree contains symlink {relative}")
                    if stat.S_ISDIR(node.st_mode):
                        _require(relative in expected_directories,
                                 "artifact tree has an unexpected directory")
                        actual_directories.add(relative)
                        pending.append((entry.path, relative))
                    elif stat.S_ISREG(node.st_mode):
                        _require(relative in expected_files,
                                 "artifact tree has an unexpected file")
                        _require(node.st_size > 0,
                                 f"artifact tree contains empty file {relative}")
                        actual_files.add(relative)
                    else:
                        raise ValueError(
                            f"artifact tree contains non-regular node {relative}")
        except OSError as error:
            raise ValueError(
                f"cannot inspect artifact directory {relative_directory}") from error
    _require(actual_files == expected_files,
             "artifact tree has missing, stale, or unexpected files")
    _require(actual_directories == expected_directories,
             "artifact tree has missing, stale, or unexpected directories")


def _regular_relative_file(root, relative, label):
    _require(type(relative) is str and relative,
             f"{label} path must be a string")
    _require("\\" not in relative and not os.path.isabs(relative),
             f"{label} path is unsafe")
    components = relative.split("/")
    _require(all(component not in ("", ".", "..") for component in components)
             and relative == "/".join(components),
             f"{label} path is unsafe")
    path = os.path.join(root, *components)
    try:
        node = os.lstat(path)
    except OSError as error:
        raise ValueError(f"missing or non-regular {label}") from error
    _require(stat.S_ISREG(node.st_mode), f"missing or non-regular {label}")
    root_real = os.path.realpath(root)
    _require(os.path.commonpath([root_real, os.path.realpath(path)]) == root_real,
             f"{label} path escapes artifact directory")
    return path


def _hash_descriptor(root, value, label, fixed):
    _require(type(value) is dict and set(value) == set(fixed) | {"sha256"},
             f"{label} descriptor keys are invalid")
    for key, expected in fixed.items():
        _require(_json_exact(value[key], expected), f"{label} {key} mismatch")
    digest = value["sha256"]
    _require(type(digest) is str and _HEX_SHA256.fullmatch(digest),
             f"{label} SHA-256 is invalid")
    path = _regular_relative_file(root, fixed["path"], label)
    maximum_size = _ROOT_FILE_LIMITS[fixed["path"]]
    actual_digest, identity = _sha256_file(path, maximum_size)
    _require(actual_digest == digest, f"{label} SHA-256 mismatch")
    return path, identity


def _validate_manifest_header(manifest):
    _require(type(manifest) is dict, "manifest.json root must be an object")
    _require(set(manifest) == MANIFEST_KEYS, "manifest.json key set changed")
    fixed = {
        "schema": SCHEMA,
        "output_fps": OUTPUT_FPS,
        "feature_dimensions": FEATURE_DIMENSIONS,
        "terrain_dimensions": TERRAIN_DIMENSIONS,
        "support_dimensions": SUPPORT_DIMENSIONS,
        "terrain_feature_distances_m": list(TERRAIN_DISTANCES),
    }
    for key, expected in fixed.items():
        _require(_json_exact(manifest[key], expected), f"manifest {key} changed")
    _require(type(manifest["diagnostic_mode"]) is bool,
             "diagnostic_mode must be boolean")
    for name, minimum in (
        ("total_clips", 1), ("grail_clips", 0), ("skipped_clips", 0),
        ("database_frames", 1),
    ):
        _json_int(manifest[name], name, minimum)
    _require(manifest["total_clips"] <= _MAX_CLIPS,
             "total_clips exceeds the canonical corpus bound")
    _require(manifest["database_frames"] <= _MAX_FRAMES,
             "database_frames exceeds the canonical corpus bound")
    _require(manifest["skipped_clips"] == 0, "skipped_clips must be zero")
    if not manifest["diagnostic_mode"]:
        _require(manifest["total_clips"] == 1770,
                 "full pack total_clips must be 1770")
        _require(manifest["grail_clips"] == 1769,
                 "full pack grail_clips must be 1769")
        _require(manifest["database_frames"] == 459682,
                 "full pack database_frames must be 459682")


def _validate_motion_descriptors(root, manifest):
    surface = manifest["surface"]
    _require(type(surface) is dict and set(surface) == {"semantics", "signature"},
             "surface descriptor keys are invalid")
    _require(_json_exact(surface["semantics"], surface_semantics()),
             "surface semantics changed")
    _require(type(surface["signature"]) is str
             and _HEX_SHA256.fullmatch(surface["signature"]),
             "surface signature is invalid")
    _require(surface["signature"] == surface_semantics_signature(),
             "surface signature changed")
    database_path = _hash_descriptor(root, manifest["database"], "database", {
        "path": "database.bin", "schema": "holden-database/v1",
    })
    sidecars = manifest["sidecars"]
    _require(type(sidecars) is dict
             and set(sidecars) == {"terrain_features", "terrain_support"},
             "sidecar descriptor keys are invalid")
    features_path = _hash_descriptor(
        root, sidecars["terrain_features"], "terrain features", {
            "path": "terrain_features.bin", "schema": "G1TF/v1",
            "version": 1, "dimensions": 4,
        })
    support_path = _hash_descriptor(
        root, sidecars["terrain_support"], "terrain support", {
            "path": "terrain_support.bin", "schema": "G1SP/v1",
            "version": 1, "dimensions": 3,
            "columns": list(SUPPORT_COLUMNS),
        })
    index_path = _hash_descriptor(root, manifest["scene_index"], "scene index", {
        "path": "scenes/index.json", "schema": "g1-terrain-scene-index/v1",
    })
    validation_path = _hash_descriptor(
        root, manifest["validation_file"], "validation file", {
            "path": "validation.json", "schema": "g1-terrain-validation/v1",
        })
    return database_path, features_path, support_path, index_path, validation_path


def _load_database(path):
    try:
        return read_holden_database(path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"database.bin is invalid: {error}") from error


def _load_terrain_features(path):
    try:
        return read_terrain_sidecar(path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"terrain_features.bin is invalid: {error}") from error


def _load_terrain_support(path):
    try:
        return read_support_sidecar(path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"terrain_support.bin is invalid: {error}") from error


def _pread_exact(descriptor, size, offset, label):
    chunks = []
    remaining = size
    while remaining:
        block = os.pread(descriptor, remaining, offset)
        _require(bool(block), f"truncated {label} header")
        chunks.append(block)
        offset += len(block)
        remaining -= len(block)
    return b"".join(chunks)


def _preflight_database_payload(descriptor, expected_size, frames, clips):
    offset = 0

    def read_exact(size, label):
        nonlocal offset
        payload = _pread_exact(descriptor, size, offset, label)
        offset += size
        return payload

    for label, columns, components, itemsize in (
        ("positions", FEATURE_DIMENSIONS, 3, 4),
        ("velocities", FEATURE_DIMENSIONS, 3, 4),
        ("rotations", FEATURE_DIMENSIONS, 4, 4),
        ("angular velocities", FEATURE_DIMENSIONS, 3, 4),
    ):
        rows, actual_columns = struct.unpack(
            "<II", read_exact(8, label))
        _require((rows, actual_columns) == (frames, columns),
                 f"database {label} dimensions changed")
        offset += rows * actual_columns * components * itemsize
    parents, = struct.unpack("<I", read_exact(4, "parents"))
    _require(parents == FEATURE_DIMENSIONS,
             "database parent count changed")
    offset += parents * 4
    for label in ("range starts", "range stops"):
        count, = struct.unpack("<I", read_exact(4, label))
        _require(count == clips, f"database {label} count changed")
        offset += count * 4
    rows, columns = struct.unpack("<II", read_exact(8, "contacts"))
    _require((rows, columns) == (frames, 2),
             "database contact dimensions changed")
    offset += rows * columns
    _require(offset == expected_size and not os.pread(descriptor, 1, offset),
             "database layout differs from canonical dimensions")


def _preflight_sidecar_payload(
    descriptor, expected_size, frames, magic, dimensions, label,
):
    _require(expected_size >= 16, f"{label} size is invalid")
    header = _pread_exact(descriptor, 16, 0, label)
    actual_magic, version, actual_frames, actual_dimensions = \
        struct.unpack("<4sIII", header)
    _require((actual_magic, version, actual_frames, actual_dimensions)
             == (magic, 1, frames, dimensions),
             f"{label} header differs from manifest dimensions")


def _motion_payload_sizes(frames, clips):
    return (
        176 + 8 * clips + 1614 * frames,
        16 + 16 * frames,
        16 + 12 * frames,
    )


def _manifest_signature(names, parents):
    payload = json.dumps(
        {"names": names, "parents": parents},
        separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_skeleton(manifest, database):
    skeleton = manifest["skeleton"]
    _require(type(skeleton) is dict
             and set(skeleton) == {"names", "parents", "signature"},
             "skeleton keys are invalid")
    names = skeleton["names"]
    parents = skeleton["parents"]
    _require(type(names) is list
             and all(type(name) is str and name for name in names),
             "skeleton names must be non-empty strings")
    _require(tuple(names) == G1_SKELETON_NAMES,
             "skeleton names do not match the canonical G1 hierarchy")
    _require(type(parents) is list
             and all(type(parent) is int for parent in parents),
             "skeleton parents must contain exact integers, not bool")
    _require(tuple(parents) == G1_SKELETON_PARENTS,
             "skeleton parents do not match the canonical G1 hierarchy")
    _require(parents == np.asarray(database.parents, np.int64).tolist(),
             "skeleton parents do not match database.bin")
    signature = skeleton["signature"]
    _require(type(signature) is str and _HEX_SHA256.fullmatch(signature),
             "skeleton signature is invalid")
    _require(signature == _manifest_signature(names, parents),
             "skeleton signature does not match names and parents")
    _require(database.positions.shape[1] == FEATURE_DIMENSIONS,
             "database.bin bone count does not match the G1 skeleton")
    return names


def _expected_output_frames(source_frames, source_fps):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        duration = np.float64(source_frames - 1) / np.float64(source_fps)
        count = duration * np.float64(OUTPUT_FPS)
    _require(np.isfinite(count) and count <= _MAX_FRAMES,
             "source duration exceeds the canonical frame bound")
    return int(np.floor(count + 1e-9)) + 1


def _validate_sources(manifest, database):
    sources = manifest["sources"]
    _require(type(sources) is list and sources, "sources must be non-empty")
    _require(len(sources) <= _MAX_CLIPS,
             "sources exceeds the canonical clip bound")
    _require(len(sources) == manifest["total_clips"],
             "total_clips does not match sources")
    _require(manifest["grail_clips"] == len(sources) - 1,
             "grail_clips must count every source after Takara")
    _require(manifest["database_frames"] == len(database.positions),
             "database_frames does not match database.bin")
    _require(len(database.range_starts) == len(sources),
             "database.bin range count does not match total_clips")
    cursor = 0
    names = []
    source_map_total = 0
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        _require(type(source) is dict and set(source) == SOURCE_KEYS,
                 f"{label} keys are invalid")
        name = source["name"]
        terrain_id = source["terrain_id"]
        _require(type(name) is str and name, f"{label} name is invalid")
        _require(name not in names, f"duplicate source name {name}")
        names.append(name)
        _require(type(terrain_id) is str and terrain_id,
                 f"{label} terrain_id is invalid")
        if index == 0:
            _require(name == "takara_walk_50hz", "first source must be Takara")
            _require(terrain_id == "flat", "Takara terrain_id must be flat")
        else:
            _require(terrain_id == name, f"{label} terrain_id must match name")
        source_fps = _json_float(source["source_fps"], f"{label} source_fps")
        _require(any(_json_exact(source_fps, expected)
                     for expected in _CANONICAL_SOURCE_FPS),
                 f"{label} source_fps must be canonical 25.0 or 50.0")
        source_frames = _json_int(source["source_frames"], f"{label} source_frames", 1)
        _require(source_frames <= _MAX_SOURCE_FRAMES,
                 f"{label} source_frames exceeds the canonical input bound")
        output_frames = _json_int(source["output_frames"], f"{label} output_frames", 1)
        range_start = _json_int(source["range_start"], f"{label} range_start", 0)
        range_stop = _json_int(source["range_stop"], f"{label} range_stop", 1)
        _require(output_frames == _expected_output_frames(source_frames, source_fps),
                 f"{label} output_frames violates the 25 Hz duration contract")
        _require(range_start == cursor, f"{label} range_start is not contiguous")
        _require(range_stop == range_start + output_frames,
                 f"{label} range_stop does not match output_frames")
        _require(range_stop <= len(database.positions),
                 f"{label} range exceeds database.bin")
        _require(int(database.range_starts[index]) == range_start
                 and int(database.range_stops[index]) == range_stop,
                 f"{label} range does not match database.bin")
        source_map = source["source_frame_map"]
        _require(type(source_map) is list and len(source_map) == output_frames,
                 f"{label} source frame map length does not match output_frames")
        _require(all(type(value) is int for value in source_map),
                 f"{label} source frame map must contain exact integers")
        output_t = np.arange(output_frames, dtype=np.float64) / OUTPUT_FPS
        expected_map = np.rint(output_t * source_fps).astype(np.int64)
        expected_map = np.clip(expected_map, 0, source_frames - 1).tolist()
        _require(source_map == expected_map,
                 f"{label} source frame map violates 25 Hz provenance")
        source_map_total += len(source_map)
        _require(source_map_total <= _MAX_FRAMES,
                 "source frame maps exceed the canonical frame bound")
        cursor = range_stop
    _require(names[1:] == sorted(names[1:]), "GRAIL sources must be lexically sorted")
    _require(cursor == len(database.positions),
             "source ranges do not cover database.bin frames")
    return sources


def _validate_parameters(manifest, clip_count):
    contact = manifest["contact"]
    expected_contact = {
        "speed_threshold": 0.15,
        "height_threshold": 0.06,
        "median_filter_frames": 3,
    }
    _require(_json_exact(contact, expected_contact), "contact parameters changed")
    validation = manifest["validation"]
    _require(type(validation) is dict and set(validation) == {
        "schema", "duration_error_s", "fk_max_error_m",
        "quaternion_norm_max_error",
    }, "validation keys are invalid")
    _require(validation["schema"] == "g1-terrain-validation/v1",
             "validation schema changed")
    limits = {
        "fk_max_error_m": 0.001,
        "duration_error_s": 1.0 / OUTPUT_FPS + 1e-12,
        "quaternion_norm_max_error": 1e-4,
    }
    for name, limit in limits.items():
        values = validation[name]
        _require(type(values) is list and len(values) == clip_count,
                 f"validation {name} must contain one value per clip")
        numbers = [
            _json_float(value, f"validation {name}[{index}]")
            for index, value in enumerate(values)
        ]
        _require(all(0.0 <= value <= limit for value in numbers),
                 f"validation {name} exceeds {limit}")


def _normal_or_positive_zero_f32(values):
    encoded = np.asarray(values, dtype="<f4")
    bits = encoded.view("<u4")
    exponent = bits & np.uint32(0x7f800000)
    return (bits == 0) | ((exponent != 0) & (exponent != np.uint32(0x7f800000)))


def _positive_normal_f32(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = bits & 0x7f800000
    return (bits & 0x80000000) == 0 and exponent not in (0, 0x7f800000)


def _parse_heightfield(path, metadata):
    payload = _read_regular_bytes(
        path, _HEIGHTFIELD_HEADER.size + 4 * _MAX_GRID_CELLS,
        "scene heightfield")
    _require(len(payload) >= _HEIGHTFIELD_HEADER.size,
             f"{path}: truncated G1HF header")
    magic, version, nx, nz, ox, oz, cell, exterior = \
        _HEIGHTFIELD_HEADER.unpack_from(payload)
    _require(magic == b"G1HF" and version == 2,
             f"{path}: scene heightfield must be G1HF/v2")
    _require(nx >= 2 and nz >= 2, f"{path}: invalid G1HF dimensions")
    _require(nx <= _MAX_GRID_AXIS and nz <= _MAX_GRID_AXIS,
             f"{path}: G1HF axis exceeds the canonical bound")
    available = len(payload) - _HEIGHTFIELD_HEADER.size
    _require(available % 4 == 0, f"{path}: misaligned G1HF payload")
    cells = available // 4
    _require(cells <= _MAX_GRID_CELLS,
             f"{path}: G1HF cell count exceeds the canonical bound")
    _require(nz > 0 and nx <= cells // nz and nx * nz == cells,
             f"{path}: truncated, trailing, or overflowing G1HF payload")
    _require(_normal_or_positive_zero_f32([ox, oz, exterior]).all()
             and _positive_normal_f32(cell),
             f"{path}: G1HF/v2 header has invalid binary32 values")
    heights = np.frombuffer(
        payload, "<f4", cells, _HEIGHTFIELD_HEADER.size).reshape(nz, nx).copy()
    _require(_normal_or_positive_zero_f32(heights).all(),
             f"{path}: G1HF/v2 heights have invalid binary32 values")
    expected_keys = {
        "path", "schema", "version", "nx", "nz", "origin_x", "origin_z",
        "cell_size_m", "exterior_height_m", "interpolation", "diagonal",
        "sha256",
    }
    _require(type(metadata) is dict and set(metadata) == expected_keys,
             "scene heightfield metadata keys are invalid")
    _json_int(metadata["version"], "heightfield version", 1)
    _json_int(metadata["nx"], "heightfield nx", 2)
    _json_int(metadata["nz"], "heightfield nz", 2)
    fixed = {
        "path": "terrain.bin", "schema": "G1HF/v2", "version": 2,
        "nx": nx, "nz": nz, "interpolation": HEIGHTFIELD_INTERPOLATION,
        "diagonal": HEIGHTFIELD_DIAGONAL,
    }
    for key, expected in fixed.items():
        _require(_json_exact(metadata[key], expected),
                 f"scene heightfield {key} mismatch")
    for key, actual in (
        ("origin_x", ox), ("origin_z", oz),
        ("cell_size_m", cell), ("exterior_height_m", exterior),
    ):
        expected = _json_f32(
            metadata[key], f"heightfield {key}", positive=key == "cell_size_m")
        _require(struct.pack("<f", expected) == struct.pack("<f", actual),
                 f"scene heightfield {key} mismatch")
    _require(metadata["cell_size_m"] == SCENE_CELL_SIZE
             and cell == SCENE_CELL_SIZE,
             "scene heightfield cell size must be 0.02f")
    _require(metadata["exterior_height_m"] == 0.0 and exterior == 0.0,
             "scene heightfield exterior height must be zero")
    digest = metadata["sha256"]
    _require(type(digest) is str and _HEX_SHA256.fullmatch(digest),
             "scene heightfield SHA-256 is invalid")
    _require(hashlib.sha256(payload).hexdigest() == digest,
             "scene heightfield SHA-256 mismatch")
    try:
        return HeightGrid(heights, float(ox), float(oz), float(cell), float(exterior))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid G1HF runtime grid: {error}") from error


def _parse_walkability(path, metadata, grid):
    payload = _read_regular_bytes(
        path, _WALKABILITY_HEADER.size + _MAX_GRID_CELLS,
        "scene walkability")
    _require(len(payload) >= _WALKABILITY_HEADER.size,
             f"{path}: truncated G1WM header")
    magic, version, nx, nz = _WALKABILITY_HEADER.unpack_from(payload)
    _require(magic == b"G1WM" and version == 1,
             f"{path}: walkability must be G1WM/v1")
    _require(nx >= 2 and nz >= 2, f"{path}: invalid G1WM dimensions")
    _require(nx <= _MAX_GRID_AXIS and nz <= _MAX_GRID_AXIS,
             f"{path}: G1WM axis exceeds the canonical bound")
    available = len(payload) - _WALKABILITY_HEADER.size
    _require(available <= _MAX_GRID_CELLS,
             f"{path}: G1WM cell count exceeds the canonical bound")
    _require(nz > 0 and nx <= available // nz and nx * nz == available,
             f"{path}: truncated, trailing, or overflowing G1WM payload")
    values = np.frombuffer(
        payload, np.uint8, available, _WALKABILITY_HEADER.size,
    ).reshape(nz, nx).copy()
    _require(np.isin(values, np.array([0, 1, 2], np.uint8)).all(),
             f"{path}: invalid G1WM class")
    _require((nx, nz) == (grid.nx, grid.nz),
             "walkability dimensions differ from heightfield")
    expected = {
        "path": "walkability.bin", "schema": "G1WM/v1", "version": 1,
        "nx": grid.nx, "nz": grid.nz,
        "classes": {"blocked": 0, "certified": 1, "stress": 2},
        "sha256": metadata.get("sha256") if type(metadata) is dict else None,
    }
    _require(type(metadata) is dict and set(metadata) == set(expected),
             "walkability metadata keys are invalid")
    digest = metadata["sha256"]
    _require(type(digest) is str and _HEX_SHA256.fullmatch(digest),
             "walkability SHA-256 is invalid")
    expected["sha256"] = digest
    _require(_json_exact(metadata, expected),
             "walkability metadata values are invalid")
    _require(hashlib.sha256(payload).hexdigest() == digest,
             "walkability SHA-256 mismatch")
    return values


def _runtime_obj_coordinate(origin, index, cell):
    encoded = np.float32(float(origin) + int(index) * float(cell))
    return 0.0 if encoded == 0.0 else float(encoded)


def _independent_obj_bytes(grid):
    lines = []
    for iz in range(grid.nz):
        z = _runtime_obj_coordinate(grid.origin_z, iz, grid.cell_size)
        for ix in range(grid.nx):
            x = _runtime_obj_coordinate(grid.origin_x, ix, grid.cell_size)
            y = float(grid.heights[iz, ix])
            lines.append(f"v {x:.9g} {y:.9g} {z:.9g}\n")
    for iz in range(grid.nz - 1):
        for ix in range(grid.nx - 1):
            p00 = iz * grid.nx + ix + 1
            p10 = p00 + 1
            p01 = p00 + grid.nx
            p11 = p01 + 1
            lines.append(f"f {p00} {p11} {p10}\n")
            lines.append(f"f {p00} {p01} {p11}\n")
    payload = "".join(lines).encode("ascii")
    _require(len(payload) <= _MAX_OBJ_BYTES,
             "independent OBJ rendering exceeds the canonical bound")
    return payload


def _validate_obj(path, metadata, grid):
    _require(type(metadata) is dict and set(metadata) == {"path", "schema", "sha256"},
             "mesh metadata keys are invalid")
    digest = metadata["sha256"]
    _require(type(digest) is str and _HEX_SHA256.fullmatch(digest),
             "mesh SHA-256 is invalid")
    _require(_json_exact(metadata, {
        "path": "terrain.obj", "schema": "obj/v1", "sha256": digest,
    }), "mesh path or schema mismatch")
    payload = _read_regular_bytes(path, _MAX_OBJ_BYTES, "scene OBJ")
    _require(hashlib.sha256(payload).hexdigest() == digest,
             "mesh SHA-256 mismatch")
    _require(payload == _independent_obj_bytes(grid),
             "terrain.obj is not the exact fixed-diagonal heightfield mesh")


def _walkability_indices(grid, x, z):
    x = _f32_round(x, "route sample x")
    z = _f32_round(z, "route sample z")
    gx = _f32_div(
        _f32_sub(x, grid.origin_x, "walkability gx numerator"),
        grid.cell_size, "walkability gx")
    gz = _f32_div(
        _f32_sub(z, grid.origin_z, "walkability gz numerator"),
        grid.cell_size, "walkability gz")
    _require(0.0 <= gx <= grid.nx - 1 and 0.0 <= gz <= grid.nz - 1,
             "route or region sample is outside walkability grid")
    ix = min(int(np.floor(np.float32(
        _f32_add(gx, 0.5, "walkability gx tie offset")))), grid.nx - 1)
    iz = min(int(np.floor(np.float32(
        _f32_add(gz, 0.5, "walkability gz tie offset")))), grid.nz - 1)
    return ix, iz


def _route_samples(points, maximum_step, budget):
    output = [points[0]]
    step = _f32_round(maximum_step, "route maximum sample step")
    _require(step > 0.0, "route maximum sample step must be positive")
    required = 1
    segments = []
    for start, stop in zip(points, points[1:]):
        dx = _f32_sub(stop[0], start[0], "route segment dx")
        dz = _f32_sub(stop[1], start[1], "route segment dz")
        squared = _f32_add(
            _f32_mul(dx, dx, "route segment dx squared"),
            _f32_mul(dz, dz, "route segment dz squared"),
            "route segment squared length")
        distance = _f32_sqrt(squared, "route segment length")
        count = max(1, int(np.ceil(np.float32(
            _f32_div(distance, step, "route segment sample ratio")))))
        required += count
        _require(required <= budget,
                 "scene route exceeds its grid-derived sample budget")
        segments.append((start, stop, dx, dz, count))
    for start, stop, dx, dz, count in segments:
        for index in range(1, count + 1):
            if index == count:
                output.append(stop)
            else:
                alpha = _f32_div(index, count, "route sample alpha")
                output.append((
                    _f32_add(start[0], _f32_mul(
                        alpha, dx, "route sample dx"), "route sample x"),
                    _f32_add(start[1], _f32_mul(
                        alpha, dz, "route sample dz"), "route sample z"),
                ))
    return output


def _route_cover_classes(grid, values, points):
    node_count = grid.nx * grid.nz
    samples = _route_samples(
        points, _f32_div(grid.cell_size, 2.0, "route half-cell step"),
        4 * node_count + 1)
    output = []
    for start, stop in zip(samples, samples[1:]):
        start_ix, start_iz = _walkability_indices(grid, *start)
        stop_ix, stop_iz = _walkability_indices(grid, *stop)
        _require(abs(stop_ix - start_ix) <= 1 and abs(stop_iz - start_iz) <= 1,
                 "route half-cell subsegment skipped a grid cell")
        output.append(frozenset(
            int(values[iz, ix])
            for iz in range(min(start_iz, stop_iz), max(start_iz, stop_iz) + 1)
            for ix in range(min(start_ix, stop_ix), max(start_ix, stop_ix) + 1)
        ))
    return output


def _endpoint_footprint_classes(grid, values, x, z):
    radius = np.float32(ENDPOINT_RADIUS)
    minimum_x = float(grid.origin_x)
    maximum_x = minimum_x + (grid.nx - 1) * float(grid.cell_size)
    minimum_z = float(grid.origin_z)
    maximum_z = minimum_z + (grid.nz - 1) * float(grid.cell_size)
    _require(minimum_x <= float(x) - float(radius)
             and float(x) + float(radius) <= maximum_x
             and minimum_z <= float(z) - float(radius)
             and float(z) + float(radius) <= maximum_z,
             "route endpoint footprint leaves heightfield")
    origin_x = np.float32(grid.origin_x)
    origin_z = np.float32(grid.origin_z)
    cell = np.float32(grid.cell_size)
    center_x = np.float32(x)
    center_z = np.float32(z)
    xs = np.asarray([
        np.float32(origin_x + np.float32(np.float32(ix) * cell))
        for ix in range(grid.nx)
    ], np.float32)
    zs = np.asarray([
        np.float32(origin_z + np.float32(np.float32(iz) * cell))
        for iz in range(grid.nz)
    ], np.float32)
    dx = np.float32(xs[np.newaxis, :] - center_x)
    dz = np.float32(zs[:, np.newaxis] - center_z)
    radius_squared = np.float32(radius * radius)
    limit = np.float32(radius_squared + np.float32(1e-8))
    inside = np.float32(
        np.float32(dx * dx) + np.float32(dz * dz)) <= limit
    _require(bool(np.any(inside)), "route endpoint footprint contains no grid node")
    return frozenset(int(value) for value in values[inside])


def _validate_route(route, grid, walkability, lookahead):
    _require(type(route) is dict and set(route) == {
        "id", "waypoints_xz", "expected_outcome", "walkability_class",
        "landing_hold_seconds",
    }, "scene route keys are invalid")
    _require(type(route["id"]) is str and _SAFE_ID.fullmatch(route["id"]),
             "scene route ID is invalid")
    raw_points = route["waypoints_xz"]
    node_count = grid.nx * grid.nz
    _require(type(raw_points) is list and 2 <= len(raw_points) <= node_count + 1,
             "scene route point count exceeds its grid-derived bound")
    points = []
    for index, point in enumerate(raw_points):
        _require(type(point) is list and len(point) == 2,
                 "scene route points are invalid")
        x = _json_f32(point[0], f"route waypoint {index} x")
        z = _json_f32(point[1], f"route waypoint {index} z")
        _require(lookahead[0] <= x <= lookahead[1]
                 and lookahead[2] <= z <= lookahead[3],
                 "scene route leaves lookahead bounds")
        points.append((x, z))
    _require(all(start != stop for start, stop in zip(points, points[1:])),
             "scene route requires every nonzero segment")
    hold = _json_f32(route["landing_hold_seconds"], "landing hold")
    _require(hold >= 0.0, "landing hold must be nonnegative")
    _require(hold == 0.0 or len(points) >= 4,
             "landing hold requires waypoint 2 and a later exit")
    outcome = route["expected_outcome"]
    _require(type(outcome) is str, "scene route outcome must be a string")
    expected_class = {
        "traverse": 1, "safe-stop": 0, "traverse-or-safe-stop": 2,
    }.get(outcome)
    route_class = _json_int(route["walkability_class"], "route walkability class")
    _require(expected_class == route_class, "scene route outcome/class mismatch")
    covers = _route_cover_classes(grid, walkability, points)
    _require(covers, "scene route has no cell covers")
    if route_class in (1, 2):
        _require(all(classes == {route_class} for classes in covers),
                 "expected route enters wrong walkability class")
        endpoint_classes = ({route_class}, {route_class})
    else:
        transitions = [
            index for index, classes in enumerate(covers)
            if classes == {0, 1}
        ]
        _require(len(transitions) == 1,
                 "safe-stop route needs exactly one certified-blocked cover")
        split = transitions[0]
        _require(split > 0 and split < len(covers) - 1
                 and all(classes == {1} for classes in covers[:split])
                 and all(classes == {0} for classes in covers[split + 1:]),
                 "safe-stop route must have pure certified prefix and blocked suffix")
        endpoint_classes = ({1}, {0})
    for point, expected in zip((points[0], points[-1]), endpoint_classes):
        _require(_endpoint_footprint_classes(
            grid, walkability, *point) == expected,
            "route endpoint footprint enters wrong walkability class")
    return tuple(points)


def _bounds_pair(metadata, minimum_key, maximum_key, dimensions, label, f32):
    raw_minimum = metadata[minimum_key]
    raw_maximum = metadata[maximum_key]
    _require(type(raw_minimum) is list and type(raw_maximum) is list
             and len(raw_minimum) == dimensions and len(raw_maximum) == dimensions,
             f"{label} bounds are invalid")
    scalar = _json_f32 if f32 else _json_float
    minimum = np.array([
        scalar(value, f"{label} minimum[{index}]")
        for index, value in enumerate(raw_minimum)
    ], np.float64)
    maximum = np.array([
        scalar(value, f"{label} maximum[{index}]")
        for index, value in enumerate(raw_maximum)
    ], np.float64)
    strict_axes = (0, 1) if dimensions == 2 else (0, 2)
    _require(all(minimum[index] < maximum[index] for index in strict_axes)
             and all(minimum[index] <= maximum[index] for index in range(dimensions)),
             f"{label} bounds must be strict in X/Z")
    return minimum, maximum


def _same_f64_array(left, right):
    left = np.ascontiguousarray(left, np.float64)
    right = np.ascontiguousarray(right, np.float64)
    return left.shape == right.shape and np.array_equal(
        left.view(np.uint64), right.view(np.uint64))


def _validate_regions(scene_id, regions, grid, walkability, lookahead):
    _require(type(regions) is dict
             and set(regions) == {"certified", "stress", "blocked"},
             f"{scene_id}: region keys are invalid")
    expected_class = {"blocked": 0, "certified": 1, "stress": 2}
    region_ids = set()
    for class_name in ("certified", "stress", "blocked"):
        entries = regions[class_name]
        _require(type(entries) is list,
                 f"{scene_id}: {class_name} regions must be a list")
        _require(len(entries) <= grid.nx * grid.nz,
                 f"{scene_id}: region count exceeds grid-derived bound")
        for entry in entries:
            _require(type(entry) is dict and set(entry) == {"id", "bounds_xz"}
                     and type(entry["id"]) is str
                     and _SAFE_ID.fullmatch(entry["id"]),
                     f"{scene_id}: invalid {class_name} region")
            _require(entry["id"] not in region_ids,
                     f"{scene_id}: duplicate region ID")
            region_ids.add(entry["id"])
            raw = entry["bounds_xz"]
            _require(type(raw) is list and len(raw) == 4,
                     f"{scene_id}: invalid {class_name} region bounds")
            bounds = tuple(_json_f32(
                value, f"{class_name} region bounds[{index}]")
                for index, value in enumerate(raw))
            _require(bounds[0] < bounds[1] and bounds[2] < bounds[3]
                     and lookahead[0] <= bounds[0] <= bounds[1] <= lookahead[1]
                     and lookahead[2] <= bounds[2] <= bounds[3] <= lookahead[3],
                     f"{scene_id}: invalid {class_name} region bounds")
            minimum_ix, minimum_iz = _walkability_indices(
                grid, bounds[0], bounds[2])
            maximum_ix, maximum_iz = _walkability_indices(
                grid, bounds[1], bounds[3])
            _require(minimum_ix <= maximum_ix and minimum_iz <= maximum_iz,
                     f"{scene_id}: region nearest-node cover is not monotone")
            observed = walkability[
                minimum_iz:maximum_iz + 1, minimum_ix:maximum_ix + 1]
            _require(observed.size > 0
                     and bool(np.all(observed == expected_class[class_name])),
                     f"{scene_id}: {class_name} region disagrees with G1WM")


def _validate_grail_scene(scene_id, scene, grid, walkability):
    base = GRAIL_EXPECTED_BASES[scene_id]
    _require(scene["label"] == GRAIL_LABELS[scene_id],
             f"{scene_id}: GRAIL label changed")
    provenance = scene["provenance"]
    terrain = GrailTerrain.from_base(base)
    measured = GRAIL_EXPECTED_HEIGHTS[scene_id]
    source_measured = float(terrain.footprint()["height"])
    _require(np.isfinite(source_measured)
             and abs(source_measured - measured) <= 1e-12,
             f"{scene_id}: canonical GRAIL source height changed")
    target = GRAIL_TARGETS[scene_id]
    expected_parameters = {
        "selection_rule": (
            "fixed-default" if target is None
            else "nearest-measured-maximum-then-lexical"),
        "target_height_m": target,
        "measured_maximum_height_m": measured,
        "route_source": "converted-holden-root-path",
    }
    _require(provenance["kind"] == "grail"
             and _json_exact(provenance["source_ids"], [base, base])
             and _json_exact(provenance["parameters"], expected_parameters),
             f"{scene_id}: GRAIL provenance or parameters changed")
    route_class = GRAIL_EXPECTED_CLASSES[scene_id]
    class_name = "certified" if route_class == 1 else "stress"
    expected_regions = {"certified": [], "stress": [], "blocked": []}
    expected_regions[class_name] = [{
        "id": "curb-route",
        "bounds_xz": [
            scene["bounds"]["playable_min_xz"][0],
            scene["bounds"]["playable_max_xz"][0],
            scene["bounds"]["playable_min_xz"][1],
            scene["bounds"]["playable_max_xz"][1],
        ],
    }]
    _require(_json_exact(scene["regions"], expected_regions),
             f"{scene_id}: GRAIL route region changed")
    _require(len(scene["routes"]) == 1
             and len(scene["routes"][0]["waypoints_xz"]) == 5
             and scene["routes"][0]["walkability_class"] == route_class,
             f"{scene_id}: GRAIL route class or cardinality changed")
    playable = (
        scene["bounds"]["playable_min_xz"][0],
        scene["bounds"]["playable_max_xz"][0],
        scene["bounds"]["playable_min_xz"][1],
        scene["bounds"]["playable_max_xz"][1],
    )
    classifier = (
        _f32_sub(playable[0], WALKABILITY_HALO, "GRAIL classifier xmin"),
        _f32_add(playable[1], WALKABILITY_HALO, "GRAIL classifier xmax"),
        _f32_sub(playable[2], WALKABILITY_HALO, "GRAIL classifier zmin"),
        _f32_add(playable[3], WALKABILITY_HALO, "GRAIL classifier zmax"),
    )
    expected = np.zeros((grid.nz, grid.nx), np.uint8)
    for iz in range(grid.nz):
        z = float(grid.origin_z) + iz * float(grid.cell_size)
        if not classifier[2] <= z <= classifier[3]:
            continue
        for ix in range(grid.nx):
            x = float(grid.origin_x) + ix * float(grid.cell_size)
            if classifier[0] <= x <= classifier[1]:
                expected[iz, ix] = route_class
    _require(np.array_equal(walkability, expected),
             f"{scene_id}: complete GRAIL classifier differs from playable halo")
    report = _independent_grail_surface_parity(terrain, grid)
    _require(report["node_error_m"] <= 1e-6,
             f"{scene_id}: node surface parity failed")
    _require(report["within_cell_error_m"] <= 1e-4,
             f"{scene_id}: within-cell parity failed")
    _require(report["source_away_edge_error_m"] <= 0.005,
             f"{scene_id}: source surface parity failed")
    _require(report["top_edge_movement_m"] <= 0.02 + 1e-9,
             f"{scene_id}: source edge moved by more than one cell")
    _require(type(report["away_edge_probe_count"]) is int
             and report["away_edge_probe_count"] > 0,
             f"{scene_id}: surface parity used no away-edge probes")


def _fixed_diagonal_height(grid, ix, iz, tx, tz):
    h00 = float(grid.heights[iz, ix])
    h10 = float(grid.heights[iz, ix + 1])
    h01 = float(grid.heights[iz + 1, ix])
    h11 = float(grid.heights[iz + 1, ix + 1])
    if tx >= tz:
        return h00 + tx * (h10 - h00) + tz * (h11 - h10)
    return h00 + tx * (h11 - h01) + tz * (h01 - h00)


def _independent_grail_surface_parity(terrain, grid):
    source_nodes = np.empty((grid.nz, grid.nx), np.float64)
    for iz in range(grid.nz):
        z = float(grid.origin_z) + iz * float(grid.cell_size)
        for ix in range(grid.nx):
            x = float(grid.origin_x) + ix * float(grid.cell_size)
            source_nodes[iz, ix] = terrain.height(x, z)
    node_error = float(np.max(np.abs(
        source_nodes - grid.heights.astype(np.float64))))

    cell_count = (grid.nx - 1) * (grid.nz - 1)
    sample_count = min(cell_count, 4096)
    linear_cells = np.unique(np.linspace(
        0, cell_count - 1, sample_count, dtype=np.int64))
    probes = ((0.25, 0.125), (0.75, 0.25),
              (0.25, 0.75), (0.75, 0.875))
    within_cell_error = 0.0
    source_away_error = 0.0
    away_edge_probe_count = 0
    for linear in linear_cells:
        iz, ix = divmod(int(linear), grid.nx - 1)
        local_source = [
            float(grid.heights[iz, ix]),
            float(grid.heights[iz, ix + 1]),
            float(grid.heights[iz + 1, ix]),
            float(grid.heights[iz + 1, ix + 1]),
        ]
        local_pairs = []
        for tx, tz in probes:
            x = float(grid.origin_x) + (ix + tx) * float(grid.cell_size)
            z = float(grid.origin_z) + (iz + tz) * float(grid.cell_size)
            encoded = _fixed_diagonal_height(grid, ix, iz, tx, tz)
            # Re-evaluate through an algebraically separate barycentric form.
            h00 = float(grid.heights[iz, ix])
            h10 = float(grid.heights[iz, ix + 1])
            h01 = float(grid.heights[iz + 1, ix])
            h11 = float(grid.heights[iz + 1, ix + 1])
            if tx >= tz:
                reconstructed = (
                    (1.0 - tx) * h00 + (tx - tz) * h10 + tz * h11)
            else:
                reconstructed = (
                    (1.0 - tz) * h00 + (tz - tx) * h01 + tx * h11)
            within_cell_error = max(
                within_cell_error, abs(encoded - reconstructed))
            source = float(terrain.height(x, z))
            local_pairs.append((encoded, source))
            local_source.append(source)
        if max(local_source) - min(local_source) <= 0.005:
            for encoded, source in local_pairs:
                source_away_error = max(
                    source_away_error, abs(encoded - source))
                away_edge_probe_count += 1

    footprint = terrain.footprint()
    threshold = float(footprint["height"]) - 0.02
    top_iz, top_ix = np.nonzero(grid.heights >= threshold)
    _require(len(top_ix) > 0, "GRAIL grid contains no measured top nodes")
    grid_bounds = (
        float(grid.origin_x) + int(top_ix.min()) * float(grid.cell_size),
        float(grid.origin_x) + int(top_ix.max()) * float(grid.cell_size),
        float(grid.origin_z) + int(top_iz.min()) * float(grid.cell_size),
        float(grid.origin_z) + int(top_iz.max()) * float(grid.cell_size),
    )
    source_bounds = (
        float(footprint["x"][0]), float(footprint["x"][1]),
        float(footprint["z"][0]), float(footprint["z"][1]),
    )
    edge_movement = max(
        abs(actual - expected)
        for actual, expected in zip(grid_bounds, source_bounds))
    return {
        "node_error_m": node_error,
        "within_cell_error_m": float(within_cell_error),
        "source_away_edge_error_m": float(source_away_error),
        "top_edge_movement_m": float(edge_movement),
        "away_edge_probe_count": int(away_edge_probe_count),
    }


def _validate_scene(
    root, scene_id, scene_path, scene_digest, manifest_signature,
):
    scene_root = os.path.join(root, "scenes", scene_id)
    scene = _read_json(scene_path, _MAX_SCENE_JSON_BYTES, f"{scene_id} scene JSON")
    _require(hashlib.sha256(_canonical_json_bytes(scene)).hexdigest()
             == scene_digest,
             f"{scene_id}: parsed scene JSON differs from authenticated bytes")
    _require(type(scene) is dict and set(scene) == SCENE_KEYS,
             f"{scene_id}: scene metadata keys are invalid")
    _require(scene["schema"] == "g1-terrain-scene/v1" and scene["id"] == scene_id,
             f"{scene_id}: scene schema or ID mismatch")
    _require(type(scene["label"]) is str and scene["label"],
             f"{scene_id}: label is invalid")
    _require(scene["coordinate_signature"] == COORDINATE_SIGNATURE,
             f"{scene_id}: coordinate signature mismatch")
    _require(scene["surface_signature"] == manifest_signature,
             f"{scene_id}: surface signature mismatch")
    _require(_json_exact(
        scene["terrain_feature_distances_m"], list(TERRAIN_DISTANCES)),
        f"{scene_id}: terrain distances mismatch")
    provenance = scene["provenance"]
    _require(type(provenance) is dict
             and set(provenance) == {"kind", "source_ids", "parameters"}
             and type(provenance["kind"]) is str
             and provenance["kind"] in ("grail", "procedural")
             and type(provenance["source_ids"]) is list
             and all(type(value) is str and value for value in provenance["source_ids"])
             and type(provenance["parameters"]) is dict
             and ((provenance["kind"] == "procedural"
                   and not provenance["source_ids"])
                  or (provenance["kind"] == "grail"
                      and len(provenance["source_ids"]) == 2)),
             f"{scene_id}: provenance is invalid")
    for key in ("heightfield", "mesh", "walkability"):
        _require(type(scene[key]) is dict,
                 f"{scene_id}: {key} descriptor must be an object")
    terrain_path = _regular_relative_file(
        scene_root, scene["heightfield"].get("path"), f"{scene_id} heightfield")
    mesh_path = _regular_relative_file(
        scene_root, scene["mesh"].get("path"), f"{scene_id} mesh")
    walkability_path = _regular_relative_file(
        scene_root, scene["walkability"].get("path"), f"{scene_id} walkability")
    grid = _parse_heightfield(terrain_path, scene["heightfield"])
    _validate_obj(mesh_path, scene["mesh"], grid)
    walkability = _parse_walkability(
        walkability_path, scene["walkability"], grid)
    bounds = scene["bounds"]
    _require(type(bounds) is dict and set(bounds) == {
        "mesh_min_xyz", "mesh_max_xyz", "heightfield_min_xyz",
        "heightfield_max_xyz", "playable_min_xz", "playable_max_xz",
        "lookahead_min_xz", "lookahead_max_xz",
    }, f"{scene_id}: bounds keys are invalid")
    mesh_min, mesh_max = _bounds_pair(
        bounds, "mesh_min_xyz", "mesh_max_xyz", 3, "mesh", True)
    height_min, height_max = _bounds_pair(
        bounds, "heightfield_min_xyz", "heightfield_max_xyz", 3,
        "heightfield", False)
    playable_min, playable_max = _bounds_pair(
        bounds, "playable_min_xz", "playable_max_xz", 2, "playable", True)
    lookahead_min, lookahead_max = _bounds_pair(
        bounds, "lookahead_min_xz", "lookahead_max_xz", 2, "lookahead", True)
    derived_min = np.array([
        grid.origin_x, float(grid.heights.min()), grid.origin_z], np.float64)
    derived_max = np.array([
        grid.origin_x + (grid.nx - 1) * grid.cell_size,
        float(grid.heights.max()),
        grid.origin_z + (grid.nz - 1) * grid.cell_size,
    ], np.float64)
    mesh_derived_min = np.array([
        float(np.float32(derived_min[0])), derived_min[1],
        float(np.float32(derived_min[2])),
    ], np.float64)
    mesh_derived_max = np.array([
        float(np.float32(derived_max[0])), derived_max[1],
        float(np.float32(derived_max[2])),
    ], np.float64)
    for label, actual, expected in (
        ("mesh minimum", mesh_min, mesh_derived_min),
        ("mesh maximum", mesh_max, mesh_derived_max),
        ("heightfield minimum", height_min, derived_min),
        ("heightfield maximum", height_max, derived_max),
    ):
        _require(_same_f64_array(actual, expected),
                 f"{scene_id}: {label} mismatch")
    grid_bounds = (derived_min[0], derived_max[0], derived_min[2], derived_max[2])
    playable = (
        playable_min[0], playable_max[0], playable_min[1], playable_max[1])
    lookahead = (
        lookahead_min[0], lookahead_max[0], lookahead_min[1], lookahead_max[1])
    _require(grid_bounds[0] <= lookahead[0] < lookahead[1] <= grid_bounds[1]
             and grid_bounds[2] <= lookahead[2] < lookahead[3] <= grid_bounds[3],
             f"{scene_id}: lookahead bounds leave heightfield")
    _require(lookahead[0] <= playable[0] < playable[1] <= lookahead[1]
             and lookahead[2] <= playable[2] < playable[3] <= lookahead[3],
             f"{scene_id}: playable bounds leave lookahead")
    spawn = scene["spawn"]
    _require(type(spawn) is dict and set(spawn) == {"position", "yaw_radians"},
             f"{scene_id}: spawn keys are invalid")
    _require(type(spawn["position"]) is list and len(spawn["position"]) == 3,
             f"{scene_id}: spawn position is invalid")
    position = tuple(_json_f32(
        value, f"spawn position[{index}]")
        for index, value in enumerate(spawn["position"]))
    _json_f32(spawn["yaw_radians"], "spawn yaw")
    _require(playable[0] <= position[0] <= playable[1]
             and playable[2] <= position[2] <= playable[3],
             f"{scene_id}: spawn lies outside playable bounds")
    region_container = scene["regions"]
    _require(type(region_container) is dict
             and set(region_container) == {"certified", "stress", "blocked"}
             and all(type(entries) is list
                     for entries in region_container.values()),
             f"{scene_id}: region container is invalid")
    expected_region_count = 3 if scene_id == "blocked-course" else 1
    _require(sum(len(entries) for entries in region_container.values())
             == expected_region_count,
             f"{scene_id}: deterministic region count changed")
    _validate_regions(scene_id, region_container, grid, walkability, lookahead)
    routes = scene["routes"]
    _require(type(routes) is list and routes,
             f"{scene_id}: routes must be non-empty")
    _require(len(routes) == len(EXPECTED_ROUTE_IDS[scene_id]),
             f"{scene_id}: deterministic route count changed")
    _require(all(
        type(route) is dict
        and type(route.get("waypoints_xz")) is list
        and len(route["waypoints_xz"]) == waypoint_count
        for route, waypoint_count in zip(
            routes, EXPECTED_WAYPOINT_COUNTS[scene_id])),
        f"{scene_id}: deterministic waypoint count changed")
    route_ids = []
    for route in routes:
        points = _validate_route(route, grid, walkability, lookahead)
        route_ids.append(route["id"])
        _require(points[0] == (position[0], position[2]),
                 f"{scene_id}: route does not start at scene spawn")
        if (provenance["kind"] == "procedural"
                and route["expected_outcome"] == "traverse"):
            for x, z in points:
                _require(
                    x - grid_bounds[0] >= 1.0 - 1e-9
                    and grid_bounds[1] - x >= 1.0 - 1e-9
                    and z - grid_bounds[2] >= 1.0 - 1e-9
                    and grid_bounds[3] - z >= 1.0 - 1e-9,
                    f"{scene_id}: certified route lacks 1 m lookahead margin")
    _require(len(route_ids) == len(set(route_ids)),
             f"{scene_id}: duplicate route ID")
    _require(tuple(route_ids) == EXPECTED_ROUTE_IDS[scene_id],
             f"{scene_id}: deterministic route IDs changed")
    if scene_id in GRAIL_EXPECTED_BASES:
        _validate_grail_scene(scene_id, scene, grid, walkability)
    else:
        _require(provenance["kind"] == "procedural",
                 f"{scene_id}: procedural provenance kind changed")
    return scene, grid, walkability


def _validate_scene_catalog(
    root, manifest, index_path, index_identity, index_digest,
):
    index = _read_json(index_path, _MAX_SCENE_JSON_BYTES, "scene index JSON")
    _require(hashlib.sha256(_canonical_json_bytes(index)).hexdigest()
             == index_digest,
             "parsed scene index differs from authenticated bytes")
    _require_file_identity(index_path, index_identity, "scene index")
    expected_header = {
        "schema": "g1-terrain-scene-index/v1",
        "default_scene_id": "grail-curb-default",
        "scene_ids": list(LOCKED_SCENE_IDS),
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
    }
    _require(type(index) is dict and set(index) == set(expected_header) | {"scenes"},
             "scene index key set changed")
    for key, expected in expected_header.items():
        _require(_json_exact(index[key], expected),
                 f"scene index {key} changed")
    descriptors = index["scenes"]
    _require(type(descriptors) is list
             and len(descriptors) == len(LOCKED_SCENE_IDS),
             "scene index descriptor count changed")
    scenes = {}
    for scene_id, descriptor in zip(LOCKED_SCENE_IDS, descriptors):
        _require(type(descriptor) is dict
                 and set(descriptor) == {"id", "path", "sha256"},
                 f"{scene_id}: scene index descriptor keys changed")
        expected_path = f"scenes/{scene_id}/scene.json"
        _require(_json_exact(descriptor["id"], scene_id)
                 and _json_exact(descriptor["path"], expected_path),
                 f"{scene_id}: scene index descriptor ID/path mismatch")
        digest = descriptor["sha256"]
        _require(type(digest) is str and _HEX_SHA256.fullmatch(digest),
                 f"{scene_id}: scene descriptor SHA-256 is invalid")
        scene_path = _regular_relative_file(root, expected_path, f"{scene_id} metadata")
        _require(os.lstat(scene_path).st_size <= _MAX_SCENE_JSON_BYTES,
                 f"{scene_id}: scene JSON exceeds its byte-size contract")
        actual_digest, scene_identity = _sha256_file(
            scene_path, _MAX_SCENE_JSON_BYTES)
        _require(actual_digest == digest,
                 f"{scene_id}: scene descriptor SHA-256 mismatch")
        scenes[scene_id] = _validate_scene(
            root, scene_id, scene_path, digest,
            manifest["surface"]["signature"])
        _require_file_identity(
            scene_path, scene_identity, f"{scene_id} scene JSON")
    for scene_id, expected in _expected_procedural_scenes():
        scene_root = os.path.join(root, "scenes", scene_id)
        for name, payload in (
            ("scene.json", expected.scene_json),
            ("terrain.bin", expected.terrain_bin),
            ("terrain.obj", expected.terrain_obj),
            ("walkability.bin", expected.walkability_bin),
        ):
            limits = {
                "scene.json": _MAX_SCENE_JSON_BYTES,
                "terrain.bin": _HEIGHTFIELD_HEADER.size + 4 * _MAX_GRID_CELLS,
                "terrain.obj": _MAX_OBJ_BYTES,
                "walkability.bin": _WALKABILITY_HEADER.size + _MAX_GRID_CELLS,
            }
            actual = _read_regular_bytes(
                os.path.join(scene_root, name), limits[name],
                f"{scene_id} deterministic {name}")
            _require(actual == payload,
                     f"{scene_id}: deterministic procedural bytes changed")
    _require_file_identity(index_path, index_identity, "scene index")
    return scenes


@lru_cache(maxsize=1)
def _expected_procedural_scenes():
    return tuple(
        (definition.scene_id, build_scene(definition))
        for definition in procedural_scene_definitions()
    )


def _read_legacy_json(path, maximum_size, label):
    def object_without_duplicates(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r}")
            output[key] = value
        return output

    try:
        payload = _read_regular_bytes(path, maximum_size, label)
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")),
        )
    except (
        OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError,
    ) as error:
        raise ValueError(f"{os.path.basename(path)} is invalid JSON: {error}") \
            from error
    _validate_json_complexity(value, label)
    return value


def _peek_manifest_schema(root):
    try:
        root_node = os.lstat(root)
    except OSError as error:
        raise ValueError("artifact directory does not exist") from error
    _require(stat.S_ISDIR(root_node.st_mode),
             "artifact directory does not exist or is a symlink")
    manifest = _read_legacy_json(
        os.path.join(root, "manifest.json"), _MAX_MANIFEST_BYTES,
        "manifest JSON")
    _require(type(manifest) is dict, "manifest.json root must be an object")
    schema = manifest.get("schema")
    _require(type(schema) is str, "manifest schema must be a string")
    return schema, manifest


def _legacy_int(value, label, minimum=0):
    _require(type(value) is int, f"{label} must be an integer")
    _require(value >= minimum, f"{label} must be at least {minimum}")
    return value


def _legacy_number(value, label):
    _require(type(value) in (int, float), f"{label} must be numeric")
    result = float(value)
    _require(np.isfinite(result), f"{label} must be finite")
    return result


def _legacy_validate_skeleton(manifest, database):
    skeleton = manifest.get("skeleton")
    _require(type(skeleton) is dict, "skeleton must be an object")
    names = skeleton.get("names")
    parents = skeleton.get("parents")
    _require(type(names) is list and len(names) == FEATURE_DIMENSIONS,
             "skeleton names must contain 31 bones")
    _require(all(type(name) is str and name for name in names),
             "skeleton names must be non-empty strings")
    _require(len(set(names)) == len(names), "skeleton names must be unique")
    _require(names[0] == "Simulation",
             "skeleton first bone must be Simulation")
    _require(type(parents) is list and len(parents) == len(names)
             and all(type(value) is int for value in parents),
             "skeleton parents must contain one integer per bone")
    _require(parents == np.asarray(database.parents, np.int64).tolist(),
             "skeleton parents do not match database.bin")
    _require(skeleton.get("signature") == _manifest_signature(names, parents),
             "skeleton signature does not match names and parents")
    _require(database.positions.shape[1] == len(names),
             "database.bin bone count does not match skeleton names")
    return names


def _legacy_validate_sources(manifest, database):
    sources = manifest.get("sources")
    _require(type(sources) is list and sources, "sources must be non-empty")
    total = _legacy_int(manifest.get("total_clips"), "total_clips", 1)
    grail = _legacy_int(manifest.get("grail_clips"), "grail_clips")
    skipped = _legacy_int(manifest.get("skipped_clips"), "skipped_clips")
    frames = _legacy_int(manifest.get("database_frames"), "database_frames", 1)
    _require(total <= _MAX_CLIPS and frames <= _MAX_FRAMES,
             "legacy corpus counts exceed canonical bounds")
    _require(skipped == 0, "skipped_clips must be zero")
    _require(total == len(sources), "total_clips does not match sources")
    _require(grail == total - 1,
             "grail_clips must count every source after Takara")
    _require(frames == len(database.positions),
             "database_frames does not match database.bin")
    _require(type(manifest.get("diagnostic_mode")) is bool,
             "diagnostic_mode must be boolean")
    _require(len(database.range_starts) == total,
             "database.bin range count does not match total_clips")
    cursor = 0
    seen = set()
    for index, source in enumerate(sources):
        label = f"sources[{index}]"
        _require(type(source) is dict, f"{label} must be an object")
        name = source.get("name")
        terrain_id = source.get("terrain_id")
        _require(type(name) is str and name, f"{label} name is invalid")
        _require(name not in seen, f"duplicate source name {name}")
        seen.add(name)
        _require(type(terrain_id) is str and terrain_id,
                 f"{label} terrain_id is invalid")
        if index == 0:
            _require(name == "takara_walk_50hz", "first source must be Takara")
            _require(terrain_id == "flat", "Takara terrain_id must be flat")
        else:
            _require(terrain_id == name, f"{label} terrain_id must match name")
        fps = _legacy_number(source.get("source_fps"), f"{label} source_fps")
        _require(fps > 0.0, f"{label} source_fps must be positive")
        source_frames = _legacy_int(
            source.get("source_frames"), f"{label} source_frames", 1)
        output_frames = _legacy_int(
            source.get("output_frames"), f"{label} output_frames", 1)
        start = _legacy_int(source.get("range_start"), f"{label} range_start")
        stop = _legacy_int(source.get("range_stop"), f"{label} range_stop", 1)
        _require(output_frames == _expected_output_frames(source_frames, fps),
                 f"{label} output_frames violates the 25 Hz duration contract")
        _require(start == cursor, f"{label} range_start is not contiguous")
        _require(stop == start + output_frames,
                 f"{label} range_stop does not match output_frames")
        _require(int(database.range_starts[index]) == start
                 and int(database.range_stops[index]) == stop,
                 f"{label} range does not match database.bin")
        source_map = source.get("source_frame_map")
        _require(type(source_map) is list
                 and len(source_map) == output_frames
                 and all(type(value) is int for value in source_map),
                 f"{label} source frame map must contain integer indices")
        expected = np.rint(
            np.arange(output_frames, dtype=np.float64) / OUTPUT_FPS * fps,
        ).astype(np.int64)
        expected = np.clip(expected, 0, source_frames - 1).tolist()
        _require(source_map == expected,
                 f"{label} source frame map content violates 25 Hz provenance")
        cursor = stop
    _require(cursor == len(database.positions),
             "source ranges do not cover database.bin frames")
    return sources


def _legacy_validate_parameters(manifest, clip_count):
    contact = manifest.get("contact")
    _require(type(contact) is dict, "contact must be an object")
    _require(contact.get("speed_threshold") == 0.15,
             "contact speed_threshold must be 0.15")
    _require(contact.get("height_threshold") == 0.06,
             "contact height_threshold must be 0.06")
    _require(contact.get("median_filter_frames") == 3,
             "contact median_filter_frames must be 3")
    terrain = manifest.get("terrain")
    _require(type(terrain) is dict, "terrain must be an object")
    _require(terrain.get("distances_m") == [0.25, 0.5, 0.75, 1.0],
             "terrain distances_m must be [0.25, 0.5, 0.75, 1.0]")
    _require(terrain.get("coordinate_mapping")
             == "mujoco_xyz_to_holden_x_z_neg_y",
             "terrain coordinate_mapping is unsupported")
    _require(type(terrain.get("runtime_base")) is str
             and terrain["runtime_base"],
             "terrain runtime_base must be a non-empty string")
    _require(terrain.get("cell_size_m") == 0.02,
             "terrain cell_size_m must be 0.02")
    _require(terrain.get("border_m") == 2.0,
             "terrain border_m must be 2.0")
    validation = manifest.get("validation")
    _require(type(validation) is dict, "validation must be an object")
    limits = {
        "fk_max_error_m": 0.001,
        "duration_error_s": 1.0 / OUTPUT_FPS + 1e-12,
        "quaternion_norm_max_error": 1e-4,
    }
    for name, limit in limits.items():
        values = validation.get(name)
        _require(type(values) is list and len(values) == clip_count,
                 f"validation {name} must contain one value per clip")
        numbers = [
            _legacy_number(value, f"validation {name}[{index}]")
            for index, value in enumerate(values)
        ]
        _require(all(0.0 <= value <= limit for value in numbers),
                 f"validation {name} exceeds {limit}")
    return terrain


def _legacy_parse_heightfield(path, metadata, schema_cell_size):
    payload = _read_regular_bytes(
        path, _HEIGHTFIELD_HEADER.size + 4 * _MAX_GRID_CELLS,
        "legacy terrain.bin")
    _require(len(payload) >= _HEIGHTFIELD_HEADER.size,
             "terrain.bin has a truncated G1HF header")
    magic, version, nx, nz, ox, oz, cell, exterior = \
        _HEIGHTFIELD_HEADER.unpack_from(payload)
    _require(magic == b"G1HF", "terrain.bin magic must be G1HF")
    _require(version == 1, "terrain.bin G1HF version must be 1")
    _require(2 <= nx <= _MAX_GRID_AXIS and 2 <= nz <= _MAX_GRID_AXIS,
             "terrain.bin G1HF dimensions must be >= 2")
    _require(nx * nz <= _MAX_GRID_CELLS,
             "terrain.bin G1HF dimensions exceed canonical bounds")
    _require(np.isfinite([ox, oz, cell, exterior]).all(),
             "terrain.bin G1HF header values must be finite")
    _require(cell > 0.0, "terrain.bin G1HF cell size must be positive")
    _require(len(payload) == _HEIGHTFIELD_HEADER.size + nx * nz * 4,
             "terrain.bin G1HF payload length does not match nx*nz float32 values")
    heights = np.frombuffer(payload, "<f4", nx * nz, 32)
    _require(np.isfinite(heights).all(),
             "terrain.bin G1HF heights must be finite")
    expected_keys = {
        "nx", "nz", "origin_x", "origin_z", "cell_size", "exterior_height",
    }
    _require(type(metadata) is dict and set(metadata) == expected_keys,
             "terrain heightfield metadata fields are incomplete")
    _require(metadata["nx"] == nx, "terrain heightfield nx mismatch")
    _require(metadata["nz"] == nz, "terrain heightfield nz mismatch")
    for key, actual in (
        ("origin_x", ox), ("origin_z", oz),
        ("cell_size", cell), ("exterior_height", exterior),
    ):
        expected = _legacy_number(metadata[key], f"terrain heightfield {key}")
        _require(np.isclose(expected, actual, rtol=1e-6, atol=1e-6),
                 f"terrain heightfield {key} mismatch")
    _require(metadata["cell_size"] == schema_cell_size
             and cell == float(np.float32(schema_cell_size)),
             "terrain heightfield cell size must match terrain cell_size_m")
    _require(metadata["exterior_height"] == 0.0 and exterior == 0.0,
             "terrain heightfield exterior height must be 0.0")
    return (
        float(ox), float(ox + (nx - 1) * cell),
        float(oz), float(oz + (nz - 1) * cell),
    )


def _legacy_parse_obj(path):
    payload = _read_regular_bytes(path, _MAX_OBJ_BYTES, "legacy terrain.obj")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError(f"terrain.obj cannot be read: {error}") from error
    vertices = []
    faces = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if fields[0] == "v":
            _require(len(fields) == 4,
                     f"terrain.obj line {number}: vertex must contain three values")
            try:
                vertex = tuple(float(value) for value in fields[1:])
            except ValueError as error:
                raise ValueError(
                    f"terrain.obj line {number}: invalid vertex") from error
            _require(np.isfinite(vertex).all(),
                     f"terrain.obj line {number}: vertex must be finite")
            vertices.append(vertex)
        elif fields[0] == "f":
            _require(len(fields) >= 4,
                     f"terrain.obj line {number}: face needs at least three vertices")
            _require(all(_LEGACY_OBJ_INDEX.fullmatch(value)
                         for value in fields[1:]),
                     f"terrain.obj line {number}: faces require one-based indices")
            face = tuple(int(value) for value in fields[1:])
            _require(len(set(face)) >= 3,
                     f"terrain.obj line {number}: face is degenerate")
            faces.append((number, face))
        else:
            raise ValueError(
                f"terrain.obj line {number}: malformed record {fields[0]!r}")
    _require(vertices, "terrain.obj must contain vertices")
    _require(faces, "terrain.obj must contain faces")
    for number, face in faces:
        _require(max(face) <= len(vertices),
                 f"terrain.obj line {number}: face index exceeds vertex count")
    positions = np.asarray(vertices, np.float64)
    return (
        float(positions[:, 0].min()), float(positions[:, 0].max()),
        float(positions[:, 2].min()), float(positions[:, 2].max()),
    )


def _legacy_validate_coverage(heightfield, obj, border):
    scale = max(1.0, abs(border),
                *(abs(value) for value in heightfield),
                *(abs(value) for value in obj))
    tolerance = 8.0 * float(np.finfo(np.float32).eps) * scale
    hxmin, hxmax, hzmin, hzmax = heightfield
    oxmin, oxmax, ozmin, ozmax = obj
    _require(hxmin <= oxmin - border + tolerance
             and hxmax >= oxmax + border - tolerance
             and hzmin <= ozmin - border + tolerance
             and hzmax >= ozmax + border - tolerance,
             "terrain.bin domain does not cover terrain.obj XZ bounds plus terrain.border_m")


def _validate_legacy_v1(root, manifest):
    required = (
        "database.bin", "terrain_features.bin", "manifest.json",
        "validation.json", "terrain.bin", "terrain.obj",
    )
    for name in required:
        path = os.path.join(root, name)
        try:
            node = os.lstat(path)
        except OSError as error:
            raise ValueError(f"missing {name}") from error
        _require(stat.S_ISREG(node.st_mode), f"missing {name}")
    _require(manifest.get("schema") == "g1-terrain-artifacts/v1",
             "schema must be g1-terrain-artifacts/v1")
    _require(manifest.get("output_fps") == OUTPUT_FPS,
             "output_fps must be 25.0")
    _require(manifest.get("feature_dimensions") == FEATURE_DIMENSIONS,
             "feature_dimensions must be 31")
    _require(manifest.get("terrain_dimensions") == TERRAIN_DIMENSIONS,
             "terrain_dimensions must be 4")
    validation_file = _read_legacy_json(
        os.path.join(root, "validation.json"), _MAX_VALIDATION_BYTES,
        "legacy validation JSON")
    _require(validation_file == manifest.get("validation"),
             "validation.json does not match manifest validation")
    database = _load_database(os.path.join(root, "database.bin"))
    features = _load_terrain_features(os.path.join(root, "terrain_features.bin"))
    database.terrain_features = features
    try:
        database.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"ArtifactSet validation failed: {error}") from error
    _require(len(database.positions) == len(features),
             "database.bin and terrain_features.bin frame counts differ")
    names = _legacy_validate_skeleton(manifest, database)
    sources = _legacy_validate_sources(manifest, database)
    terrain = _legacy_validate_parameters(manifest, len(sources))
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(database.rotations, axis=-1) - 1.0)))
    _require(quaternion_error <= 1e-4,
             f"database.bin quaternion norm error {quaternion_error} exceeds 0.0001")
    heightfield = _legacy_parse_heightfield(
        os.path.join(root, "terrain.bin"), terrain.get("heightfield"),
        float(terrain["cell_size_m"]))
    obj = _legacy_parse_obj(os.path.join(root, "terrain.obj"))
    _legacy_validate_coverage(heightfield, obj, float(terrain["border_m"]))
    return {
        "frames": len(database.positions),
        "clips": len(sources),
        "bones": len(names),
    }


def _validate_dispatched(
    artifact_dir, full_source_validation=False, source_options=None,
):
    _require(type(full_source_validation) is bool,
             "full_source_validation must be an exact boolean")
    root = os.path.abspath(os.fspath(artifact_dir))
    schema, peeked_manifest = _peek_manifest_schema(root)
    if schema == "g1-terrain-artifacts/v1":
        _require(not full_source_validation,
                 "full source validation requires v2 artifacts")
        # Temporary compatibility bridge. Task 11B migrates the production
        # builder to v2 and removes this dispatch with the publisher bridge.
        return schema, _validate_legacy_v1(root, peeked_manifest)
    _require(schema == SCHEMA, f"schema must be {SCHEMA}")
    if full_source_validation:
        raise ValueError(
            "full source validation is not implemented until Task 12B")
    _validate_exact_file_tree(root)
    manifest = _read_json(
        os.path.join(root, "manifest.json"), _MAX_MANIFEST_BYTES,
        "manifest JSON")
    _validate_manifest_header(manifest)
    database_entry, features_entry, support_entry, index_entry, validation_entry = \
        _validate_motion_descriptors(root, manifest)
    database_path, database_identity = database_entry
    features_path, features_identity = features_entry
    support_path, support_identity = support_entry
    index_path, index_identity = index_entry
    validation_path, validation_identity = validation_entry
    validation_file = _read_json(
        validation_path, _MAX_VALIDATION_BYTES, "validation JSON")
    _require(hashlib.sha256(_canonical_json_bytes(validation_file)).hexdigest()
             == manifest["validation_file"]["sha256"],
             "parsed validation JSON differs from authenticated bytes")
    _require_file_identity(
        validation_path, validation_identity, "validation file")
    _require(_json_exact(validation_file, manifest["validation"]),
             "validation.json does not match manifest validation")
    database_size, features_size, support_size = _motion_payload_sizes(
        manifest["database_frames"], manifest["total_clips"])
    with _authenticated_reader_path(
        database_path, manifest["database"]["sha256"],
        database_size, "database",
        lambda descriptor: _preflight_database_payload(
            descriptor, database_size, manifest["database_frames"],
            manifest["total_clips"]),
    ) as authenticated_path:
        database = _load_database(authenticated_path)
    _require_file_identity(database_path, database_identity, "database")
    with _authenticated_reader_path(
        features_path,
        manifest["sidecars"]["terrain_features"]["sha256"],
        features_size, "terrain features",
        lambda descriptor: _preflight_sidecar_payload(
            descriptor, features_size, manifest["database_frames"],
            b"G1TF", TERRAIN_DIMENSIONS, "terrain sidecar"),
    ) as authenticated_path:
        database.terrain_features = _load_terrain_features(authenticated_path)
    _require_file_identity(
        features_path, features_identity, "terrain features")
    with _authenticated_reader_path(
        support_path,
        manifest["sidecars"]["terrain_support"]["sha256"],
        support_size, "terrain support",
        lambda descriptor: _preflight_sidecar_payload(
            descriptor, support_size, manifest["database_frames"],
            b"G1SP", SUPPORT_DIMENSIONS, "support sidecar"),
    ) as authenticated_path:
        database.terrain_support = _load_terrain_support(authenticated_path)
    _require_file_identity(
        support_path, support_identity, "terrain support")
    try:
        database.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"ArtifactSet validation failed: {error}") from error
    frames = len(database.positions)
    _require(frames == len(database.terrain_features)
             and frames == len(database.terrain_support),
             "database and sidecar frame counts differ")
    _require(database.terrain_features.shape == (frames, TERRAIN_DIMENSIONS),
             "terrain_features.bin dimension count must be 4")
    _require(database.terrain_support.shape == (frames, SUPPORT_DIMENSIONS),
             "terrain_support.bin dimension count must be 3")
    names = _validate_skeleton(manifest, database)
    sources = _validate_sources(manifest, database)
    _validate_parameters(manifest, len(sources))
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(database.rotations, axis=-1) - 1.0)))
    _require(quaternion_error <= 1e-4,
             "database quaternion norm error exceeds 0.0001")
    scenes = _validate_scene_catalog(
        root, manifest, index_path, index_identity,
        manifest["scene_index"]["sha256"])
    for path, identity, label in (
        (database_path, database_identity, "database"),
        (features_path, features_identity, "terrain features"),
        (support_path, support_identity, "terrain support"),
        (validation_path, validation_identity, "validation file"),
    ):
        _require_file_identity(path, identity, label)
    _validate_exact_file_tree(root)
    return schema, {
        "frames": frames,
        "clips": len(sources),
        "bones": len(names),
        "scenes": len(scenes),
        "source_rows": 0,
    }


def validate_artifact_directory(
    artifact_dir, full_source_validation=False, source_options=None,
):
    return _validate_dispatched(
        artifact_dir, full_source_validation, source_options)[1]


def _parser():
    parser = argparse.ArgumentParser(
        description="Validate published Holden G1 terrain motion artifacts")
    parser.add_argument("artifact_directory")
    parser.add_argument("--full-source-validation", action="store_true")
    parser.add_argument("--grail-glob")
    parser.add_argument("--g1-xml")
    parser.add_argument("--takara")
    parser.add_argument("--remap")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    path = os.path.abspath(args.artifact_directory)
    source_options = {
        key: value for key, value in {
            "grail_glob": args.grail_glob,
            "g1_xml": args.g1_xml,
            "takara": args.takara,
            "remap": args.remap,
        }.items() if value is not None
    }
    try:
        reported_schema, summary = _validate_dispatched(
            path, args.full_source_validation, source_options)
    except Exception as error:
        print(f"INVALID {path}: {error}", file=sys.stderr)
        return 1
    print(
        f"VALID {reported_schema} frames={summary['frames']} clips={summary['clips']} "
        f"bones={summary['bones']} terrain_dims={TERRAIN_DIMENSIONS}"
        + (f" support_dims={SUPPORT_DIMENSIONS} scenes={summary['scenes']} "
           f"source_rows={summary['source_rows']}"
           if reported_schema == SCHEMA else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
