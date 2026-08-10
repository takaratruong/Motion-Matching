"""Authenticated, resumable full-PFNN lane for the walking terrain LMM.

The released PFNN database generator is deliberately parsed as data and is
never imported: importing it executes a module-level database build.  This
module keeps gait admission on the native 120 Hz clock, retargets each of the
80 released identities once, and converts each safe range independently on
the absolute 60 Hz clock.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import linalg
from scipy.spatial import distance

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import canonical_json_bytes, sha256_file
from resources.g1_terrain_builder.database import (
    combine_clips,
    forward_kinematics_arrays,
    read_holden_database,
    refresh_lmm_clip_dynamics,
    write_holden_database,
)
from resources.g1_terrain_builder.kinematics import G1Kinematics, convert_source_clip
from resources.g1_terrain_builder.resample import resample_map
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_PARENTS,
    ArtifactSet,
    HoldenClip,
    SourceClip,
    require_canonical_g1_skeleton,
)

from .full_walking_terrain_lmm_contracts import (
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    publish_lane_exclusive,
)
from .full_walking_terrain_lmm_inventory import load_inventory

PFNN_INVENTORY_SHA256 = (
    "7d1efd01abba6eea86280cb7720f999173fd8cedca89ddf56a61caad81a341e4"
)
PFNN_SPLIT_LEDGER_SHA256 = (
    "5185bd42c153518c80b67c3dd68f54e9122e2e81970893209e9235ff5cc2feb5"
)
PFNN_SOURCE_TABLE_SHA256 = (
    "41ddd2a84dc2fc4a39226bd18463e8704d2915335996cd832b51e14223167db6"
)
G1_XML_SHA256 = "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376"
EXPECTED_EXCLUDED_COUNTS = {
    "jog": 248_368,
    "run": 77_430,
    "crouch": 75_726,
    "jump": 30_598,
    "crawl": 6_644,
}
EXPECTED_NATIVE_ROWS = 493_214
EXPECTED_NATIVE_RUNS = 458
EXPECTED_OUTPUT_ROWS = 246_630
SOURCE_FPS = 120.0
TARGET_FPS = 60.0
BEAM_SEED = 1234
PFNN_POSITION_SCALE = 5.6444
PFNN_G1_SCALE = 0.875
PFNN_G1_Z_OFFSET_M = 0.05224985936713168
PFNN_PATCHES_SHA256 = "344ec49b3aab3c93cd3f618c613e7d8fd8dfaf07964583a7461f0fb7593aae79"
PFNN_PATCHES_SIZE = 620_692_714
TERRAIN_FITTER_VERSION = "full-pfnn-family-fit/v2"
PFNN_PIPELINE_VERSION = "g1-full-walking-pfnn-pipeline/v3"
PATCH_BATCH_SIZE = 256
MAX_OBJECTIVE_PROBES = 256
MAX_RBF_CENTERS = 256
RETARGET_PYTHON = Path("/home/ubuntu/.cache/native-g1-pfnn/venv/bin/python")
_FAMILIES = ("flat", "rocky", "beam", "jumpy")
_GAIT_LABELS = ("stand", "walk", "jog", "run", "crouch", "jump", "crawl", "bump")
_EXCLUDED_LABELS = _GAIT_LABELS[2:7]
_BVH_FRAMES = re.compile(r"(?m)^\s*Frames:\s*(\d+)\s*$")
_PATCH_CACHE: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
_PATCH_DIGEST_CACHE: dict[str, str] = {}
PFNN_PIPELINE_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "schema": PFNN_PIPELINE_VERSION,
            "absolute_clock_hz": TARGET_FPS,
            "minimum_range_rows": 4,
            "range_local_contacts": True,
            "four_frame_contact_endpoints": "initialize-last-before-first/v1",
            "terrain_fitter": TERRAIN_FITTER_VERSION,
            "maximum_objective_probes": MAX_OBJECTIVE_PROBES,
            "maximum_rbf_centers": MAX_RBF_CENTERS,
        }
    )
).hexdigest()


@dataclass(frozen=True)
class FileDescriptor:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PFNNDiscovery:
    sources: tuple[SourceRecord, ...]
    rest_pose: FileDescriptor
    inventory_sha256: str


@dataclass(frozen=True)
class PFNNGaitSummary:
    admitted_rows: int
    excluded_counts: dict[str, int]
    malformed_rows: int


@dataclass(frozen=True)
class PFNNRangeTerrain:
    family: str
    mode: str
    fit_id: str
    selected_patch_index: int
    fitting_error: float
    constant_height: float
    patch: np.ndarray | None
    contact_center_xz: np.ndarray
    patch_height_mean: float
    stance_height_mean: float
    rbf_centers_xz: np.ndarray
    rbf_epsilon: np.ndarray
    rbf_weights: np.ndarray
    source_contacts: np.ndarray
    receipt: dict[str, object]

    def height(self, query_xz: np.ndarray) -> np.ndarray:
        query = np.asarray(query_xz, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2 or not np.isfinite(query).all():
            raise ValueError("PFNN terrain queries must have finite shape [Q,2]")
        if self.patch is None:
            return np.full(len(query), self.constant_height, dtype=np.float64)
        base = _patch_height(self.patch, query - self.contact_center_xz)
        base = base - self.patch_height_mean + self.stance_height_mean
        distances = distance.cdist(query, self.rbf_centers_xz)
        residual = (self.rbf_weights @ (self.rbf_epsilon * distances).T).T[:, 0]
        result = base + residual
        if not np.isfinite(result).all():
            raise ValueError("PFNN terrain fit returned nonfinite heights")
        return result


@dataclass(frozen=True)
class _HoldenPFNNSurface:
    fit: PFNNRangeTerrain

    def heights(self, xz: np.ndarray) -> np.ndarray:
        query = np.asarray(xz, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2 or not np.isfinite(query).all():
            raise ValueError("placed PFNN queries must have finite shape [Q,2]")
        source_xz = np.column_stack(
            (
                100.0 * query[:, 0] / PFNN_G1_SCALE,
                -100.0 * query[:, 1] / PFNN_G1_SCALE,
            )
        )
        result = np.empty(len(query), dtype=np.float64)
        for start in range(0, len(query), 65_536):
            stop = min(start + 65_536, len(query))
            result[start:stop] = (
                PFNN_G1_SCALE * self.fit.height(source_xz[start:stop]) / 100.0
                + PFNN_G1_Z_OFFSET_M
            )
        if not np.isfinite(result).all():
            raise ValueError("placed PFNN terrain returned nonfinite heights")
        return result

    def height(self, x: float, z: float) -> float:
        return float(self.heights(np.array([[float(x), float(z)]]))[0])


@dataclass(frozen=True)
class _ProcessedIdentity:
    clips: tuple[HoldenClip, ...]
    ranges: tuple[RangeRecord, ...]
    receipts: tuple[dict[str, object], ...]
    terrain_grid: np.ndarray


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _descriptor(path: Path, *, root: Path | None = None) -> FileDescriptor:
    resolved = Path(path).resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"authority input must be a regular non-symlink file: {path}")
    stored = str(resolved) if root is None else resolved.relative_to(root).as_posix()
    return FileDescriptor(stored, resolved.stat().st_size, sha256_file(resolved))


def _source_table_paths(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "data_terrain"
            for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            if type(value) is not list or not all(type(item) is str for item in value):
                break
            return tuple(Path(item).as_posix().removeprefix("./") for item in value)
    raise ValueError("PFNN source table has no literal data_terrain list")


def classify_pfnn_terrain(stem: str) -> Literal["flat", "rocky", "beam", "jumpy"]:
    """Return the exact family selected by released ``generate_database.py``."""

    name = Path(stem).stem
    if "LocomotionFlat12_000" in name:
        return "jumpy"
    if any(value in name for value in ("NewCaptures01_000", "NewCaptures02_000")):
        return "flat"
    if any(
        value in name
        for value in (
            "NewCaptures03_000",
            "NewCaptures03_001",
            "NewCaptures03_002",
            "NewCaptures04_000",
        )
    ):
        return "jumpy"
    if "WalkingUpSteps06_000" in name:
        return "beam"
    if any(
        value in name
        for value in (
            "WalkingUpSteps09_000",
            "WalkingUpSteps10_000",
            "WalkingUpSteps11_000",
        )
    ):
        return "flat"
    return "flat" if "Flat" in name else "rocky"


def _authenticated_relative(
    root: Path, descriptor: object, expected_path: str, label: str
) -> Path:
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError(f"{label} descriptor has invalid fields")
    if descriptor["path"] != expected_path or Path(expected_path).is_absolute():
        raise ValueError(f"{label} descriptor path changed")
    path = (root / expected_path).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not a regular file under the PFNN root")
    if (
        path.stat().st_size != descriptor["size_bytes"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise ValueError(f"{label} size or SHA-256 mismatch")
    return path


def discover_full_pfnn_inventory(
    inventory: Path,
    *,
    pfnn_root: Path,
    expected_inventory_sha256: str = PFNN_INVENTORY_SHA256,
) -> PFNNDiscovery:
    """Authenticate all 80 released trainable identities and metadata rest pose."""

    inventory_path = Path(inventory).resolve(strict=True)
    actual_inventory_sha = sha256_file(inventory_path)
    if actual_inventory_sha != expected_inventory_sha256:
        raise ValueError("walking inventory SHA-256 mismatch")
    frozen = load_inventory(inventory_path)
    root = Path(pfnn_root).resolve(strict=True)
    table = root / "generate_database.py"
    if table.stat().st_size != 21_707 or sha256_file(table) != PFNN_SOURCE_TABLE_SHA256:
        raise ValueError("PFNN source table authority changed")
    relative_paths = _source_table_paths(table)
    if len(relative_paths) != 80 or len(set(relative_paths)) != 80:
        raise ValueError("PFNN source table must contain exactly 80 identities")
    records = {
        record.source_id: record
        for record in frozen.sources
        if record.authority.get("kind") == "pfnn"
    }
    if len(records) != 80:
        raise ValueError("walking inventory must contain exactly 80 PFNN identities")
    ordered: list[SourceRecord] = []
    frame_total = 0
    for relative in relative_paths:
        stem = Path(relative).stem
        source_id = f"pfnn:{stem}"
        if source_id not in records:
            raise ValueError(
                f"PFNN source table identity missing from inventory: {stem}"
            )
        record = records[source_id]
        inputs = record.authority.get("inputs")
        if type(inputs) is not dict or set(inputs) != {
            "motion",
            "gait",
            "phase",
            "footsteps",
        }:
            raise ValueError(f"{source_id}: sidecar inventory changed")
        motion = _authenticated_relative(root, inputs["motion"], relative, "PFNN BVH")
        companions = {
            "gait": motion.with_suffix(".gait"),
            "phase": motion.with_suffix(".phase"),
            "footsteps": motion.with_name(f"{stem}_footsteps.txt"),
        }
        for role, path in companions.items():
            _authenticated_relative(
                root, inputs[role], path.relative_to(root).as_posix(), f"PFNN {role}"
            )
        match = _BVH_FRAMES.findall(motion.read_text(encoding="utf-8"))
        if len(match) != 1:
            raise ValueError(f"{source_id}: BVH frame declaration is invalid")
        frame_count = int(match[0])
        gait = np.loadtxt(companions["gait"], dtype=np.float64)
        phase = np.loadtxt(companions["phase"], dtype=np.float64)
        if gait.shape != (frame_count, 8) or phase.shape != (frame_count,):
            raise ValueError(f"{source_id}: PFNN sidecar row count changed")
        summarize_pfnn_gait(gait)
        if (
            not np.isfinite(phase).all()
            or not companions["footsteps"].read_text().strip()
        ):
            raise ValueError(f"{source_id}: PFNN phase/footstep sidecar is invalid")
        if record.authority.get("terrain_semantics") != classify_pfnn_terrain(stem):
            raise ValueError(f"{source_id}: terrain family differs from released table")
        ordered.append(record)
        frame_total += frame_count
    if frame_total != 931_980:
        raise ValueError("PFNN native row inventory changed")
    by_id = {record.source_id: record for record in ordered}
    mirrors = 0
    for record in ordered:
        if record.mirror_of is None:
            continue
        mirrors += 1
        if (
            record.mirror_of not in by_id
            or record.canonical_source_id != by_id[record.mirror_of].source_id
        ):
            raise ValueError("PFNN official mirror linkage changed")
    if mirrors != 40:
        raise ValueError("PFNN inventory must have 40 official mirrors")
    rest = _descriptor(root / "data" / "animations" / "rest.bvh", root=root)
    if rest.path in relative_paths:
        raise ValueError("PFNN rest pose must remain metadata-only")
    return PFNNDiscovery(tuple(ordered), rest, actual_inventory_sha)


def _validated_gait(
    gait: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(gait)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("PFNN gait must have exact shape [T,8]")
    if values.dtype.kind not in "iuf":
        raise ValueError("PFNN gait rows must contain real values")
    values = values.astype(np.float64, copy=False)
    finite = np.isfinite(values).all(axis=1)
    bounded = ((values >= 0.0) & (values <= 1.0)).all(axis=1)
    base_sum = values[:, :7].sum(axis=1)
    valid = finite & bounded & np.isclose(base_sum, 1.0, rtol=0.0, atol=1.0e-6)
    safe = np.where(valid[:, None], values, 0.0)
    allowed = safe[:, 0] + safe[:, 1]
    disallowed = safe[:, 2:7].sum(axis=1)
    admitted = valid & (allowed > 0.0) & (allowed >= disallowed)
    return values, valid, admitted


def summarize_pfnn_gait(gait: np.ndarray) -> PFNNGaitSummary:
    values, valid, admitted = _validated_gait(gait)
    counts: Counter[str] = Counter()
    for index in np.flatnonzero(valid & ~admitted):
        counts[_EXCLUDED_LABELS[int(np.argmax(values[index, 2:7]))]] += 1
    return PFNNGaitSummary(
        int(admitted.sum()),
        dict(counts),
        int((~valid).sum()),
    )


def excluded_pfnn_blocks(
    gait: np.ndarray,
) -> tuple[tuple[int, int, str], ...]:
    """Return maximal same-reason gait exclusions on the native clock."""

    values, valid, admitted = _validated_gait(gait)
    reasons: list[str | None] = [None] * len(values)
    for index in np.flatnonzero(~valid):
        reasons[int(index)] = "malformed"
    for index in np.flatnonzero(valid & ~admitted):
        reasons[int(index)] = _EXCLUDED_LABELS[int(np.argmax(values[index, 2:7]))]
    blocks: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(reasons):
        reason = reasons[cursor]
        if reason is None:
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(reasons) and reasons[cursor] == reason:
            cursor += 1
        blocks.append((start, cursor, reason))
    return tuple(blocks)


def _validated_breaks(value: np.ndarray, frames: int) -> np.ndarray:
    breaks = np.asarray(value)
    if breaks.dtype != np.bool_ or breaks.shape != (max(0, frames - 1),):
        raise ValueError("continuity_breaks must be bool shape [T-1]")
    return breaks


def admitted_pfnn_ranges(
    gait: np.ndarray,
    continuity_breaks: np.ndarray,
    fit_intervals: tuple[tuple[int, int, str], ...],
) -> tuple[tuple[int, int, str], ...]:
    """Return maximal admitted native runs with zero boundary halo."""

    _values, _valid, admitted = _validated_gait(gait)
    breaks = _validated_breaks(continuity_breaks, len(admitted))
    previous_stop = 0
    ranges: list[tuple[int, int, str]] = []
    for interval in fit_intervals:
        if type(interval) is not tuple or len(interval) != 3:
            raise ValueError("fit intervals must be (start,stop,family) tuples")
        start, stop, family = interval
        if (
            type(start) is not int
            or type(stop) is not int
            or not (0 <= start < stop <= len(admitted))
            or start < previous_stop
            or family not in _FAMILIES
        ):
            raise ValueError(
                "PFNN fit intervals must be ordered valid half-open ranges"
            )
        cursor = start
        while cursor < stop:
            while cursor < stop and not admitted[cursor]:
                cursor += 1
            if cursor == stop:
                break
            run_start = cursor
            cursor += 1
            while cursor < stop and admitted[cursor] and not breaks[cursor - 1]:
                cursor += 1
            ranges.append((run_start, cursor, family))
        previous_stop = stop
    return tuple(ranges)


def safe_absolute_60hz_rows(
    total_frames: int,
    ranges: tuple[tuple[int, int], ...],
    *,
    minimum_rows: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select absolute 60 Hz rows whose interpolation brackets share one range."""

    if type(total_frames) is not int or total_frames < 1:
        raise ValueError("total_frames must be a positive integer")
    if type(minimum_rows) is not int or minimum_rows < 1:
        raise ValueError("minimum_rows must be a positive integer")
    left, right, alpha = resample_map(total_frames, SOURCE_FPS, TARGET_FPS)
    retained = np.zeros(len(left), dtype=np.bool_)
    previous_stop = 0
    for start, stop in ranges:
        if (
            type(start) is not int
            or type(stop) is not int
            or not (0 <= start < stop <= total_frames)
            or start < previous_stop
        ):
            raise ValueError(
                "safe source ranges must be ordered nonoverlapping intervals"
            )
        retained |= (left >= start) & (right < stop)
        previous_stop = stop
    selected_left = left[retained]
    selected_right = right[retained]
    if len(selected_left) < minimum_rows:
        retained[:] = False
        selected_left = left[retained]
        selected_right = right[retained]
    return selected_left.copy(), selected_left, selected_right, alpha[retained]


def range_local_pfnn_contacts(
    global_positions_60hz: np.ndarray,
    indices_60hz: np.ndarray,
) -> np.ndarray:
    """Derive PFNN foot contacts without consulting a neighboring range."""

    from .pfnn_terrain_fit import _pfnn_contacts

    positions = np.asarray(global_positions_60hz)
    indices = np.asarray(indices_60hz)
    if (
        positions.ndim != 3
        or positions.shape[1] <= 10
        or positions.shape[2] != 3
        or not np.isfinite(positions).all()
        or indices.dtype.kind not in "iu"
        or indices.ndim != 1
        or len(indices) < 2
        or indices[0] < 0
        or indices[-1] >= len(positions)
        or np.any(np.diff(indices) != 1)
    ):
        raise ValueError("PFNN contact range must be finite contiguous 60 Hz rows")
    return _pfnn_contacts(positions[indices])


def reverse_mirror_terrain_lanes(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid)
    flattened = values.ndim == 2 and values.shape[1] == 36
    if flattened:
        shaped = values.reshape(len(values), 12, 3)
    elif values.ndim == 3 and values.shape[1:] == (12, 3):
        shaped = values
    else:
        raise ValueError("PFNN terrain grid must have shape [T,36] or [T,12,3]")
    result = np.ascontiguousarray(shaped[..., ::-1])
    return result.reshape(values.shape)


def _patch_height(
    patch: np.ndarray,
    query_xz: np.ndarray,
    *,
    hscale: float = 3.937007874,
    vscale: float = 3.0,
) -> np.ndarray:
    values = np.asarray(patch, dtype=np.float64)
    query = np.asarray(query_xz, dtype=np.float64)
    coordinates = query / hscale + np.array(
        [values.shape[0] // 2, values.shape[1] // 2], dtype=np.float64
    )
    fraction = np.fmod(coordinates, 1.0)
    lower = np.clip(
        np.floor(coordinates).astype(np.int64), 0, np.array(values.shape) - 1
    )
    upper = np.clip(
        np.ceil(coordinates).astype(np.int64), 0, np.array(values.shape) - 1
    )
    h00 = values[lower[:, 0], lower[:, 1]]
    h01 = values[lower[:, 0], upper[:, 1]]
    h10 = values[upper[:, 0], lower[:, 1]]
    h11 = values[upper[:, 0], upper[:, 1]]
    left = (1.0 - fraction[:, 0]) * h00 + fraction[:, 0] * h10
    right = (1.0 - fraction[:, 0]) * h01 + fraction[:, 0] * h11
    return vscale * ((1.0 - fraction[:, 1]) * left + fraction[:, 1] * right)


def _patch_bank_height(patches: np.ndarray, query_xz: np.ndarray) -> np.ndarray:
    bank = np.asarray(patches, dtype=np.float64)
    query = np.asarray(query_xz, dtype=np.float64)
    if (
        bank.ndim != 3
        or min(bank.shape[1:]) < 2
        or query.ndim != 2
        or query.shape[1] != 2
        or not np.isfinite(bank).all()
        or not np.isfinite(query).all()
    ):
        raise ValueError("PFNN patches/queries have invalid shape or values")
    coordinates = query / 3.937007874 + np.array(
        [bank.shape[1] // 2, bank.shape[2] // 2], dtype=np.float64
    )
    fraction = np.fmod(coordinates, 1.0)
    lower = np.clip(
        np.floor(coordinates).astype(np.int64), 0, np.array(bank.shape[1:]) - 1
    )
    upper = np.clip(
        np.ceil(coordinates).astype(np.int64), 0, np.array(bank.shape[1:]) - 1
    )
    h00 = bank[:, lower[:, 0], lower[:, 1]]
    h01 = bank[:, lower[:, 0], upper[:, 1]]
    h10 = bank[:, upper[:, 0], lower[:, 1]]
    h11 = bank[:, upper[:, 0], upper[:, 1]]
    left = (1.0 - fraction[:, 0]) * h00 + fraction[:, 0] * h10
    right = (1.0 - fraction[:, 0]) * h01 + fraction[:, 0] * h11
    return 3.0 * ((1.0 - fraction[:, 1]) * left + fraction[:, 1] * right)


def _foot_samples(
    global_positions: np.ndarray, contacts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(global_positions, dtype=np.float64)
    values = np.asarray(contacts)
    if (
        positions.ndim != 3
        or positions.shape[1] <= 10
        or positions.shape[2] != 3
        or values.dtype != np.bool_
        or values.shape != (len(positions), 4)
        or not np.isfinite(positions).all()
    ):
        raise ValueError("PFNN fit motion/contact arrays are invalid")
    down, up = [], []
    for column, (joint, height) in enumerate(zip((4, 5, 9, 10), (5, 4, 5, 4))):
        offset = np.array([0.0, float(height), 0.0])
        down.append(positions[values[:, column], joint] - offset)
        up.append(positions[~values[:, column], joint] - offset)
    down_points = (
        np.concatenate(down)
        if any(len(value) for value in down)
        else np.empty((0, 3), dtype=np.float64)
    )
    up_points = (
        np.concatenate(up)
        if any(len(value) for value in up)
        else np.empty((0, 3), dtype=np.float64)
    )
    return (
        down_points[:, [0, 2]],
        down_points[:, 1:2],
        up_points[:, [0, 2]],
        up_points[:, 1:2],
    )


def _fit_linear_rbf(
    centers_xz: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances = distance.cdist(centers_xz, centers_xz)
    mean_distance = float(distances.mean())
    if not np.isfinite(mean_distance):
        raise ValueError("PFNN stance probes cannot fit a linear RBF")
    if mean_distance <= np.finfo(np.float64).eps:
        return (
            np.zeros(len(distances), dtype=np.float64),
            np.zeros((1, len(distances)), dtype=np.float64),
        )
    epsilon = np.ones(len(distances), dtype=np.float64) / mean_distance
    kernel = epsilon * distances
    weights = linalg.solve(
        kernel.T - np.eye(len(kernel)) * 0.1,
        residual[:, None],
        assume_a="gen",
    ).T
    if not np.isfinite(weights).all():
        raise ValueError("PFNN linear RBF weights are nonfinite")
    return epsilon, weights


def _bounded_probe_indices(rows: int, limit: int) -> np.ndarray:
    if rows <= limit:
        return np.arange(rows, dtype=np.int64)
    return np.linspace(0, rows - 1, limit, dtype=np.int64)


def fit_pfnn_range_terrain(
    *,
    family: str,
    global_positions: np.ndarray,
    contacts: np.ndarray,
    patches: np.ndarray,
    patch_coords: np.ndarray,
    source_start: int,
    source_sha256: str,
    patches_sha256: str,
    source_stop: int | None = None,
) -> PFNNRangeTerrain:
    """Fit one safe range with the released flat/rocky/jumpy/beam objective."""

    if family not in _FAMILIES:
        raise ValueError(f"unsupported PFNN terrain family {family!r}")
    positions = np.asarray(global_positions, dtype=np.float64)
    contact_values = np.asarray(contacts)
    bank = np.asarray(patches, dtype=np.float64)
    coords = np.asarray(patch_coords, dtype=np.float64)
    if coords.shape != (len(bank), 4) or not np.isfinite(coords).all():
        raise ValueError("PFNN patch coordinates must have shape [P,4]")
    down_xz, down_y, up_xz, up_y = _foot_samples(positions, contact_values)
    stop = (
        int(source_stop)
        if source_stop is not None
        else source_start + 2 * len(positions)
    )
    if type(source_start) is not int or not source_start >= 0 or stop <= source_start:
        raise ValueError("PFNN terrain fit source interval is invalid")
    if len(down_xz) == 0:
        mode = "no-contact-zero"
        selected = -1
        fitting_error = 0.0
        constant = 0.0
        patch = None
        center = np.zeros(2, dtype=np.float64)
        patch_mean = stance_mean = 0.0
        centers = np.zeros((0, 2), dtype=np.float64)
        epsilon = np.zeros(0, dtype=np.float64)
        weights = np.zeros((1, 0), dtype=np.float64)
        objective_down_count = objective_up_count = 0
        beam_seed = None
    elif family == "flat":
        mode = "flat-constant"
        selected = -1
        fitting_error = 0.0
        constant = float(down_y.mean())
        patch = None
        center = down_xz.mean(axis=0)
        patch_mean = 0.0
        stance_mean = constant
        centers = np.zeros((0, 2), dtype=np.float64)
        epsilon = np.zeros(0, dtype=np.float64)
        weights = np.zeros((1, 0), dtype=np.float64)
        objective_down_count = len(down_xz)
        objective_up_count = len(up_xz)
        beam_seed = None
    else:
        mode = f"{family}-patch-rbf"
        objective_down_indices = _bounded_probe_indices(
            len(down_xz), MAX_OBJECTIVE_PROBES
        )
        objective_up_indices = _bounded_probe_indices(len(up_xz), MAX_OBJECTIVE_PROBES)
        objective_down_xz = down_xz[objective_down_indices]
        objective_down_y = down_y[objective_down_indices]
        objective_up_xz = up_xz[objective_up_indices]
        objective_up_y = up_y[objective_up_indices]
        objective_down_count = len(objective_down_xz)
        objective_up_count = len(objective_up_xz)
        center = objective_down_xz.mean(axis=0)
        stance_mean = float(objective_down_y.mean())
        beam_center_xz = beam_outer_xz = None
        far = np.zeros(0, dtype=np.bool_)
        beam_seed = None
        if family == "beam":
            beam_seed = int.from_bytes(
                hashlib.sha256(
                    canonical_json_bytes([BEAM_SEED, source_sha256, source_start, stop])
                ).digest()[:4],
                "little",
            )
            rng = np.random.RandomState(beam_seed)
            beam_indices = _bounded_probe_indices(len(positions), MAX_OBJECTIVE_PROBES)
            beam_center = positions[beam_indices, 0]
            beam_center_xz = beam_center[:, [0, 2]]
            beam_outer = beam_center + np.array([50.0, 0.0, 50.0]) * rng.normal(
                size=beam_center.shape
            )
            beam_outer_xz = beam_outer[:, [0, 2]]
            pair_distance = np.linalg.norm(
                beam_outer[:, None] - beam_center[None, :], axis=-1
            )
            far = np.all(pair_distance > 15.0, axis=1)
        selected = -1
        fitting_error = float("inf")
        patch_mean = 0.0
        for batch_start in range(0, len(bank), PATCH_BATCH_SIZE):
            batch_stop = min(batch_start + PATCH_BATCH_SIZE, len(bank))
            batch = bank[batch_start:batch_stop]
            terrain_down = _patch_bank_height(batch, objective_down_xz - center)
            terrain_mean = terrain_down.mean(axis=1)
            down_error = 0.1 * np.mean(
                (
                    (terrain_down - terrain_mean[:, None])
                    - (objective_down_y[:, 0] - stance_mean)[None, :]
                )
                ** 2,
                axis=1,
            )
            up_error = np.zeros(len(batch), dtype=np.float64)
            jump_error = np.zeros(len(batch), dtype=np.float64)
            if len(objective_up_xz):
                terrain_up = _patch_bank_height(batch, objective_up_xz - center)
                up_delta = (terrain_up - terrain_mean[:, None]) - (
                    objective_up_y[:, 0] - stance_mean
                )[None, :]
                up_error = np.mean(np.maximum(up_delta, 0.0) ** 2, axis=1)
                if family == "jumpy":
                    over_delta = (
                        (objective_up_y[:, 0] - stance_mean)[None, :] - 5.0
                    ) - (terrain_up - terrain_mean[:, None])
                    jump_error = np.mean(np.maximum(over_delta, 0.0) ** 2, axis=1)
            beam_error = np.zeros(len(batch), dtype=np.float64)
            if family == "beam" and np.any(far):
                assert beam_center_xz is not None and beam_outer_xz is not None
                beam_center_y = _patch_bank_height(batch, beam_center_xz - center)
                beam_outer_y = _patch_bank_height(batch, beam_outer_xz - center)
                beam_error = np.mean(
                    np.maximum(
                        beam_outer_y[:, far] - (beam_center_y[:, far] - 40.0),
                        0.0,
                    )
                    ** 2,
                    axis=1,
                )
            total = down_error + up_error + jump_error + beam_error
            if not np.isfinite(total).all():
                raise ValueError(
                    "PFNN terrain family objective returned nonfinite errors"
                )
            local = int(np.argmin(total))
            error = float(total[local])
            if error < fitting_error:
                selected = batch_start + local
                fitting_error = error
                patch_mean = float(terrain_mean[local])
        if selected < 0:
            raise ValueError("PFNN patch bank is empty")
        constant = 0.0
        patch = np.array(bank[selected], dtype=np.float64, copy=True)
        selected_down = _patch_height(patch, down_xz - center)
        residual = down_y[:, 0] - (selected_down - patch_mean + stance_mean)
        if len(down_xz) > MAX_RBF_CENTERS:
            selected_probes = _bounded_probe_indices(len(down_xz), MAX_RBF_CENTERS)
        else:
            selected_probes = np.arange(len(down_xz), dtype=np.int64)
        centers = np.array(down_xz[selected_probes], dtype=np.float64, copy=True)
        epsilon, weights = _fit_linear_rbf(centers, residual[selected_probes])
    identity = {
        "schema": TERRAIN_FITTER_VERSION,
        "family": family,
        "mode": mode,
        "source_sha256": source_sha256,
        "patches_sha256": patches_sha256,
        "source_start": source_start,
        "source_stop": stop,
        "selected_patch_index": selected,
        "beam_seed": beam_seed,
        "maximum_objective_probes": MAX_OBJECTIVE_PROBES,
        "maximum_rbf_centers": MAX_RBF_CENTERS,
    }
    fit_id = _sha256_bytes(canonical_json_bytes(identity))
    receipt: dict[str, object] = {
        **identity,
        "fit_id": fit_id,
        "fitting_error": fitting_error,
        "stance_probe_count": len(down_xz),
        "swing_probe_count": len(up_xz),
        "objective_stance_probe_count": objective_down_count,
        "objective_swing_probe_count": objective_up_count,
        "rbf_probe_count": len(centers),
        "rbf_degenerate_zero_residual": bool(len(centers) and not np.any(epsilon)),
        "patch_batch_size": PATCH_BATCH_SIZE,
        "patch_coord": coords[selected].tolist() if selected >= 0 else None,
    }
    return PFNNRangeTerrain(
        family,
        mode,
        fit_id,
        selected,
        fitting_error,
        constant,
        patch,
        np.asarray(center),
        float(patch_mean),
        float(stance_mean),
        np.asarray(centers),
        np.asarray(epsilon),
        np.asarray(weights),
        contact_values.copy(),
        receipt,
    )


def _native_continuity_breaks(
    root: np.ndarray, quaternion_xyzw: np.ndarray, dof: np.ndarray
) -> np.ndarray:
    if not all(np.isfinite(value).all() for value in (root, quaternion_xyzw, dof)):
        raise ValueError("retargeted PFNN motion contains nonfinite values")
    planar = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) > 0.10
    joint = np.max(np.abs(np.diff(dof, axis=0)), axis=1) > 0.50
    quaternion = quaternion_xyzw.astype(np.float64, copy=True)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    dots = np.clip(np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=1)), 0.0, 1.0)
    rotation = 2.0 * np.arccos(dots) > 0.30
    return planar | joint | rotation


def _load_patches(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    resolved = Path(path).resolve(strict=True)
    key = str(resolved)
    if key not in _PATCH_CACHE:
        digest = _authenticate_patches(resolved)
        with np.load(resolved, allow_pickle=False) as archive:
            if set(archive.files) != {"X", "C"}:
                raise ValueError("PFNN patches archive must contain exactly X and C")
            patches = np.asarray(archive["X"], dtype=np.float64)
            coords = np.asarray(archive["C"], dtype=np.float64)
        _PATCH_CACHE[key] = (patches, coords, digest)
    return _PATCH_CACHE[key]


def _authenticate_patches(path: Path) -> str:
    resolved = Path(path).resolve(strict=True)
    key = str(resolved)
    if key not in _PATCH_DIGEST_CACHE:
        if resolved.stat().st_size != PFNN_PATCHES_SIZE:
            raise ValueError("PFNN patches archive size changed")
        digest = sha256_file(resolved)
        if digest != PFNN_PATCHES_SHA256:
            raise ValueError("PFNN patches archive SHA-256 changed")
        _PATCH_DIGEST_CACHE[key] = digest
    return _PATCH_DIGEST_CACHE[key]


def _load_retarget(
    path: Path, frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        root = np.asarray(archive["root_pos"], dtype=np.float64)
        quaternion = np.asarray(archive["root_quat"], dtype=np.float64)
        dof = np.asarray(archive["dof"], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).item())
    if (
        root.shape != (frames, 3)
        or quaternion.shape != (frames, 4)
        or dof.shape != (frames, 29)
        or fps != SOURCE_FPS
    ):
        raise ValueError("retargeted PFNN motion shape or rate changed")
    return root, quaternion, dof


def _valid_retarget_cache(
    motion: Path,
    *,
    prepared: Path,
    source_sha256: str,
    frames: int,
) -> bool:
    from .retarget_pfnn_bvh_g1 import GMR_COMMIT, RETARGET_PROJECT_COMMIT

    receipt_path = motion.with_suffix(".receipt.json")
    if not motion.is_file() or not receipt_path.is_file() or not prepared.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return (
            receipt.get("schema") == "native-g1-pfnn-sample-retarget/v1"
            and receipt.get("status") == "accepted"
            and receipt.get("source_sha256") == source_sha256
            and receipt.get("output_sha256") == sha256_file(motion)
            and receipt.get("prepared_sha256") == sha256_file(prepared)
            and receipt.get("gmr_commit") == GMR_COMMIT
            and receipt.get("retarget_project_commit") == RETARGET_PROJECT_COMMIT
            and receipt.get("fps") == SOURCE_FPS
            and receipt.get("frame_count") == frames
            and receipt.get("source_frame_count") == frames
            and receipt.get("start_frame") == 0
            and receipt.get("grounding") == "source"
            and receipt.get("warmup_frames") == 0
        )
    except (OSError, json.JSONDecodeError):
        return False


def authenticate_pfnn_runtime(
    *, gmr_root: Path, retarget_root: Path, g1_xml: Path
) -> dict[str, str]:
    """Authenticate clean pinned retarget repositories and the canonical G1 XML."""

    from .retarget_pfnn_bvh_g1 import (
        GMR_COMMIT,
        RETARGET_PROJECT_COMMIT,
        _git_commit,
    )

    authorities = (
        (Path(gmr_root), GMR_COMMIT, "GMR"),
        (Path(retarget_root), RETARGET_PROJECT_COMMIT, "retargeting_project"),
    )
    commits: dict[str, str] = {}
    for supplied, expected, label in authorities:
        if supplied.is_symlink():
            raise ValueError(f"{label} authority must not be a symlink")
        root = supplied.resolve(strict=True)
        commit = _git_commit(root)
        if commit != expected:
            raise ValueError(f"{label} must be pinned to {expected}, got {commit}")
        status = subprocess.run(
            ("git", "-C", str(root), "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise ValueError(f"{label} authority must have a clean worktree")
        commits[label] = commit
    supplied_xml = Path(g1_xml)
    if supplied_xml.is_symlink():
        raise ValueError("canonical G1 XML authority must not be a symlink")
    xml = supplied_xml.resolve(strict=True)
    xml_sha = sha256_file(xml)
    if xml.stat().st_size != 26_914 or xml_sha != G1_XML_SHA256:
        raise ValueError("canonical G1 XML authority changed")
    return {
        "gmr_commit": commits["GMR"],
        "retarget_project_commit": commits["retargeting_project"],
        "g1_xml_sha256": xml_sha,
    }


def retarget_subprocess_argv(
    *, source: Path, gmr_root: Path, retarget_root: Path, output: Path
) -> tuple[str, ...]:
    """Return the pinned-environment command for one full source retarget."""

    return (
        str(RETARGET_PYTHON),
        "-m",
        "mm_sonic.retarget_pfnn_bvh_g1",
        "--source",
        str(Path(source).resolve()),
        "--gmr-root",
        str(Path(gmr_root).resolve()),
        "--retarget-project-root",
        str(Path(retarget_root).resolve()),
        "--output",
        str(Path(output).resolve()),
        "--grounding",
        "source",
    )


def _run_retarget_subprocess(
    *, source: Path, gmr_root: Path, retarget_root: Path, output: Path
) -> None:
    if not RETARGET_PYTHON.is_file():
        raise ValueError(
            f"pinned PFNN retarget interpreter is missing: {RETARGET_PYTHON}"
        )
    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    python_paths = (
        str(repository),
        str(repository / "resources"),
        str(repository / "sonic" / "python"),
    )
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, *((existing,) if existing else ()))
    )
    subprocess.run(
        retarget_subprocess_argv(
            source=source,
            gmr_root=gmr_root,
            retarget_root=retarget_root,
            output=output,
        ),
        cwd=repository,
        env=environment,
        check=True,
    )


def _annotate_terrain(
    clip: HoldenClip, surface: _HoldenPFNNSurface, *, mirror: bool
) -> np.ndarray:
    from resources.g1_terrain_builder.terrain import LOOKAHEAD, build_facing_centerline

    clip.validate()
    global_positions, global_rotations = forward_kinematics_arrays(
        clip.positions, clip.rotations, np.asarray(G1_SKELETON_PARENTS, np.int32)
    )
    rows = len(global_positions)
    support_queries = global_positions[:, (0, 7, 13)][:, :, (0, 2)]
    support = surface.heights(support_queries.reshape(-1, 2)).reshape(rows, 3)
    feature_queries = np.empty((rows, 4, 2), dtype=np.float64)
    forward_axis = np.array([0.0, 0.0, 1.0])
    right_axis = np.array([1.0, 0.0, 0.0])
    stations = np.arange(-60, 60, 10, dtype=np.int32)
    for frame in range(rows):
        stop = min(frame + 121, rows)
        path = global_positions[frame:stop, 0][:, [0, 2]]
        headings = holden_quat.mul_vec(global_rotations[frame:stop, 0], forward_axis)[
            :, [0, 2]
        ]
        centerline = build_facing_centerline(path[0], headings, path)
        for column, lookahead in enumerate(LOOKAHEAD):
            feature_queries[frame, column] = _point_at_distance(
                centerline, float(lookahead)
            )
    feature_heights = surface.heights(feature_queries.reshape(-1, 2)).reshape(rows, 4)
    features = feature_heights - support[:, :1]
    frame_indices = np.arange(rows, dtype=np.int32)[:, None]
    indices = np.clip(frame_indices + stations[None], 0, rows - 1)
    centers = global_positions[indices, 0]
    right = holden_quat.mul_vec(global_rotations[indices, 0], right_axis)
    offsets = np.array([0.25, 0.0, -0.25], dtype=np.float64)
    grid_queries = (
        centers[:, :, None, :] + offsets[None, None, :, None] * right[:, :, None, :]
    )
    grid = surface.heights(grid_queries[..., (0, 2)].reshape(-1, 2)).reshape(
        rows, 12, 3
    )
    grid -= support[:, None, :1]
    clip.terrain_support = np.ascontiguousarray(support, dtype=np.float32)
    clip.terrain_features = np.ascontiguousarray(features, dtype=np.float32)
    clip.validate()
    if mirror:
        grid = reverse_mirror_terrain_lanes(grid)
    return np.ascontiguousarray(grid.reshape(rows, 36), dtype=np.float32)


def _point_at_distance(line: np.ndarray, distance_m: float) -> np.ndarray:
    remaining = float(distance_m)
    for start, stop in pairwise(line):
        length = float(np.linalg.norm(stop - start))
        if remaining <= length:
            return start + remaining / max(length, 1.0e-8) * (stop - start)
        remaining -= length
    return line[-1]


def _record_input(record: SourceRecord, role: str, root: Path) -> Path:
    inputs = record.authority.get("inputs")
    if type(inputs) is not dict or role not in inputs:
        raise ValueError(f"{record.source_id}: missing authenticated {role} input")
    expected = inputs[role].get("path") if type(inputs[role]) is dict else None
    if type(expected) is not str:
        raise ValueError(f"{record.source_id}: invalid {role} descriptor")
    return _authenticated_relative(root, inputs[role], expected, f"PFNN {role}")


def _identity_work_name(record: SourceRecord) -> str:
    stem = record.source_id.split(":", 1)[-1]
    digest = hashlib.sha256(record.source_id.encode()).hexdigest()[:12]
    return f"{stem}__{digest}"


def pfnn_identity_stage_name(record: SourceRecord) -> str:
    """Return the pipeline-versioned deterministic identity-stage name."""

    if not isinstance(record, SourceRecord):
        raise TypeError("PFNN identity stage requires a SourceRecord")
    return f"{_identity_work_name(record)}__stage-{PFNN_PIPELINE_SHA256[:12]}"


def pfnn_worker_start_method() -> str:
    """Use fresh SciPy/BLAS state instead of fork-inheriting parent locks."""

    return "spawn"


def resolve_pfnn_work_root(*, output: Path, work_root: Path | None = None) -> Path:
    """Resolve a deterministic default or an explicit authenticated stage root."""

    destination = Path(output).resolve()
    if work_root is None:
        return destination.parent / f".{destination.name}.work"
    candidate = Path(work_root)
    if candidate.is_symlink():
        raise ValueError("explicit PFNN work root cannot be a symlink")
    resolved = candidate.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("explicit PFNN work root must be a directory")
    return resolved


def _process_identity(
    record: SourceRecord,
    *,
    pfnn_root: Path,
    gmr_root: Path,
    retarget_root: Path,
    g1_xml: Path,
    work_root: Path,
    split_group_id: str,
    split: str,
) -> _ProcessedIdentity:
    from .pfnn_terrain_fit import _global_positions, _pfnn_modules
    from .retarget_pfnn_bvh_g1 import prepare_pfnn_bvh

    authenticate_pfnn_runtime(
        gmr_root=gmr_root,
        retarget_root=retarget_root,
        g1_xml=g1_xml,
    )

    root = Path(pfnn_root).resolve(strict=True)
    source = _record_input(record, "motion", root)
    gait_path = _record_input(record, "gait", root)
    _record_input(record, "phase", root)
    _record_input(record, "footsteps", root)
    gait = np.loadtxt(gait_path, dtype=np.float64)
    frames = len(gait)
    work = Path(work_root) / _identity_work_name(record)
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / "retarget" / "prepared.bvh"
    motion = work / "retarget" / "motion.npz"
    motion.parent.mkdir(parents=True, exist_ok=True)
    if not prepared.is_file():
        prepare_pfnn_bvh(source, prepared)
    source_sha = sha256_file(source)
    if not _valid_retarget_cache(
        motion,
        prepared=prepared,
        source_sha256=source_sha,
        frames=frames,
    ):
        if motion.exists() or motion.with_suffix(".receipt.json").exists():
            raise ValueError(f"{record.source_id}: incomplete/tampered retarget cache")
        _run_retarget_subprocess(
            source=source,
            gmr_root=gmr_root,
            retarget_root=retarget_root,
            output=motion,
        )
    root_position, quaternion_xyzw, dof = _load_retarget(motion, frames)
    continuity = _native_continuity_breaks(root_position, quaternion_xyzw, dof)
    family = classify_pfnn_terrain(source.stem)
    ranges = admitted_pfnn_ranges(gait, continuity, ((0, frames, family),))
    bvh, animation_module = _pfnn_modules(root)
    animation, _names, _frametime = bvh.load(str(source))
    animation.offsets *= PFNN_POSITION_SCALE
    animation.positions *= PFNN_POSITION_SCALE
    global_60 = _global_positions(animation_module, animation[::2])
    patches_path = root / "patches.npz"
    if family == "flat":
        patches_sha = _authenticate_patches(patches_path)
        patches = np.empty((0, 2, 2), dtype=np.float64)
        patch_coords = np.empty((0, 4), dtype=np.float64)
    else:
        patches, patch_coords, patches_sha = _load_patches(patches_path)
    kinematics = G1Kinematics(str(Path(g1_xml).resolve(strict=True)))
    clips: list[HoldenClip] = []
    range_records: list[RangeRecord] = []
    receipts: list[dict[str, object]] = [
        {
            "source_id": record.source_id,
            "source_start": start,
            "source_stop": stop,
            "native_rows": stop - start,
            "status": "excluded",
            "reason": reason,
        }
        for start, stop, reason in excluded_pfnn_blocks(gait)
    ]
    receipts.extend(
        {
            "source_id": record.source_id,
            "source_start": int(index + 1),
            "source_stop": int(index + 1),
            "native_rows": 0,
            "status": "range_boundary",
            "reason": "retarget_continuity",
        }
        for index in np.flatnonzero(continuity)
    )
    grids: list[np.ndarray] = []
    cursor = 0
    for ordinal, (start, stop, terrain_family) in enumerate(ranges):
        rows, left, right, alpha = safe_absolute_60hz_rows(
            frames,
            ((start, stop),),
            minimum_rows=4,
        )
        if not len(rows):
            receipts.append(
                {
                    "source_id": record.source_id,
                    "source_start": start,
                    "source_stop": stop,
                    "status": "short_fragment",
                }
            )
            continue
        if np.any(left != right) or np.any(alpha != 0.0):
            raise AssertionError(
                "120-to-60 PFNN conversion lost its exact global clock"
            )
        indices_60 = rows // 2
        contacts_60 = range_local_pfnn_contacts(global_60, indices_60)
        fit = fit_pfnn_range_terrain(
            family=terrain_family,
            global_positions=global_60[indices_60],
            contacts=contacts_60,
            patches=patches,
            patch_coords=patch_coords,
            source_start=start,
            source_stop=stop,
            source_sha256=source_sha,
            patches_sha256=patches_sha,
        )
        qpos = np.empty((len(rows), 36), dtype=np.float32)
        qpos[:, :3] = root_position[rows]
        qpos[:, 3:7] = quaternion_xyzw[rows][:, (3, 0, 1, 2)]
        qpos[:, 7:] = dof[rows]
        source_clip = SourceClip(
            f"{record.source_id}:{start}:{stop}",
            TARGET_FPS,
            qpos,
            rows.astype(np.int32),
            record.terrain_id,
        )
        clip, skeleton, report = convert_source_clip(
            source_clip, kinematics, target_fps=TARGET_FPS, root_filter_mode="interp"
        )
        require_canonical_g1_skeleton(skeleton, f"{record.source_id} PFNN range")
        grid = _annotate_terrain(
            clip, _HoldenPFNNSurface(fit), mirror=record.mirror_of is not None
        )
        clip = refresh_lmm_clip_dynamics(clip, skeleton, TARGET_FPS)
        range_stop = cursor + len(clip.positions)
        range_identity = {
            "source_id": record.source_id,
            "source_start": start,
            "source_stop": stop,
            "fit_id": fit.fit_id,
            "ordinal": ordinal,
        }
        range_id = _sha256_bytes(canonical_json_bytes(range_identity))
        authority = {
            "kind": "pfnn",
            "terrain_family": terrain_family,
            "source_interval_120hz": [start, stop],
            "output_rows_60hz": len(rows),
            "absolute_source_rows_sha256": hashlib.sha256(
                np.ascontiguousarray(rows, dtype="<i4").tobytes()
            ).hexdigest(),
            "terrain_fit": fit.receipt,
            "retarget": asdict(_descriptor(motion)),
            "retarget_receipt": asdict(
                _descriptor(motion.with_suffix(".receipt.json"))
            ),
            "prepared_bvh": asdict(_descriptor(prepared)),
            "conversion": report,
        }
        range_records.append(
            RangeRecord(
                range_id=range_id,
                canonical_source_id=record.canonical_source_id,
                terrain_id=record.terrain_id,
                mirror_of=record.mirror_of,
                family=record.family,
                split_group_id=split_group_id,
                split=split,
                start=cursor,
                stop=range_stop,
                quality="usable",
                authority=authority,
            )
        )
        receipts.append({**range_identity, "status": "accepted", **authority})
        clips.append(clip)
        grids.append(grid)
        cursor = range_stop
    if not clips:
        raise ValueError(f"{record.source_id}: no admitted PFNN output range")
    gait_summary = summarize_pfnn_gait(gait)
    terminal = {
        "schema": "g1-full-walking-pfnn-source/v1",
        "status": "accepted",
        "source_id": record.source_id,
        "native_rows": frames,
        "admitted_native_rows": sum(stop - start for start, stop, _ in ranges),
        "output_rows": cursor,
        "ranges": len(clips),
        "continuity_breaks": int(continuity.sum()),
        "gait_excluded_counts": gait_summary.excluded_counts,
        "malformed_gait_rows": gait_summary.malformed_rows,
        "terrain_family": family,
        "mirror_of": record.mirror_of,
    }
    receipts.append(terminal)
    return _ProcessedIdentity(
        tuple(clips),
        tuple(range_records),
        tuple(receipts),
        np.ascontiguousarray(np.concatenate(grids), dtype=np.float32),
    )


def process_pfnn_identity(
    record: SourceRecord,
    *,
    pfnn_root: Path,
    gmr_root: Path,
    retarget_root: Path,
    g1_xml: Path,
    work_root: Path | None = None,
    split_group_id: str | None = None,
    split: str = "train",
) -> tuple[list[HoldenClip], list[RangeRecord], list[dict[str, object]]]:
    """Process one identity; build callers should provide persistent ``work_root``."""

    if not isinstance(record, SourceRecord) or record.authority.get("kind") != "pfnn":
        raise TypeError("record must be an authenticated PFNN SourceRecord")
    if split not in ("train", "validation", "test"):
        raise ValueError("PFNN range split is invalid")
    group = (
        split_group_id
        or hashlib.sha256(
            canonical_json_bytes([record.canonical_source_id])
        ).hexdigest()
    )
    destination = work_root or Path(retarget_root) / "full-walking-terrain-lmm-cache"
    result = _process_identity(
        record,
        pfnn_root=pfnn_root,
        gmr_root=gmr_root,
        retarget_root=retarget_root,
        g1_xml=g1_xml,
        work_root=destination,
        split_group_id=group,
        split=split,
    )
    return list(result.clips), list(result.ranges), list(result.receipts)


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _stage_identity(
    record: SourceRecord,
    *,
    pfnn_root: Path,
    gmr_root: Path,
    retarget_root: Path,
    g1_xml: Path,
    work_root: Path,
    split_group_id: str,
    split: str,
) -> str:
    destination = Path(work_root) / "lanes" / pfnn_identity_stage_name(record)
    manifest = destination / "manifest.json"
    if manifest.is_file():
        _load_identity_stage(
            destination,
            expected_source_id=record.source_id,
            expected_split_group_id=split_group_id,
            expected_split=split,
        )
        return str(destination)
    result = _process_identity(
        record,
        pfnn_root=pfnn_root,
        gmr_root=gmr_root,
        retarget_root=retarget_root,
        g1_xml=g1_xml,
        work_root=work_root / "sources",
        split_group_id=split_group_id,
        split=split,
    )
    artifacts = combine_clips(list(result.clips), _canonical_skeleton())
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        write_holden_database(staging / "database.bin", artifacts)
        _write_npy(staging / "terrain_features.npy", artifacts.terrain_features)
        _write_npy(staging / "terrain_support.npy", artifacts.terrain_support)
        _write_npy(staging / "terrain_grid.npy", result.terrain_grid)
        _write_npy(
            staging / "source_left_indices.npy",
            np.concatenate([clip.source_left_indices for clip in result.clips]),
        )
        _write_npy(
            staging / "source_right_indices.npy",
            np.concatenate([clip.source_right_indices for clip in result.clips]),
        )
        _write_npy(
            staging / "source_alpha.npy",
            np.concatenate([clip.source_alpha for clip in result.clips]),
        )
        (staging / "ranges.json").write_bytes(
            canonical_json_bytes([asdict(value) for value in result.ranges])
        )
        (staging / "receipts.json").write_bytes(
            canonical_json_bytes(list(result.receipts))
        )
        members = {}
        for path in sorted(staging.iterdir()):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            members[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        payload = canonical_json_bytes(
            {
                "schema": "g1-full-walking-pfnn-identity-stage/v2",
                "pipeline_sha256": PFNN_PIPELINE_SHA256,
                "source_id": record.source_id,
                "canonical_source_id": record.canonical_source_id,
                "split_group_id": split_group_id,
                "split": split,
                "rows": len(artifacts.positions),
                "ranges": len(result.ranges),
                "members": members,
            }
        )
        (staging / "manifest.json").write_bytes(payload)
        with (staging / "manifest.json").open("rb") as stream:
            os.fsync(stream.fileno())
        _fsync_directory(staging)
        if destination.exists():
            raise FileExistsError(
                f"PFNN identity stage appeared concurrently: {destination}"
            )
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(destination)


def _canonical_skeleton():
    from resources.g1_terrain_builder.schema import G1_SKELETON_NAMES, SkeletonSpec

    return SkeletonSpec(
        G1_SKELETON_NAMES, np.asarray(G1_SKELETON_PARENTS, dtype=np.int32)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_array(
    path: Path, dtype: np.dtype, shape: tuple[int | None, ...]
) -> np.ndarray:
    with path.open("rb") as stream:
        value = np.load(stream, allow_pickle=False)
        if stream.read(1):
            raise ValueError(f"{path.name} has trailing bytes")
    if (
        value.dtype != np.dtype(dtype)
        or value.ndim != len(shape)
        or any(
            expected is not None and actual != expected
            for actual, expected in zip(value.shape, shape)
        )
    ):
        raise ValueError(f"{path.name} dtype or shape changed")
    return np.ascontiguousarray(value)


def bind_pfnn_source_frame_count(
    ranges: tuple[RangeRecord, ...],
    source_left_indices: np.ndarray,
    source_right_indices: np.ndarray,
    receipts: tuple[dict[str, object], ...],
    *,
    expected_source_id: str,
) -> tuple[RangeRecord, ...]:
    """Bind authenticated native source bounds to every staged range."""

    terminals = tuple(
        receipt
        for receipt in receipts
        if type(receipt) is dict
        and receipt.get("schema") == "g1-full-walking-pfnn-source/v1"
    )
    if len(terminals) != 1:
        raise ValueError("PFNN identity stage requires one terminal source receipt")
    terminal = terminals[0]
    source_frame_count = terminal.get("native_rows")
    if (
        terminal.get("status") != "accepted"
        or terminal.get("source_id") != expected_source_id
        or type(source_frame_count) is not int
        or source_frame_count < 1
        or terminal.get("ranges") != len(ranges)
    ):
        raise ValueError("PFNN terminal source receipt changed")
    left = np.asarray(source_left_indices)
    right = np.asarray(source_right_indices)
    expected_rows = ranges[-1].stop if ranges else 0
    if (
        left.ndim != 1
        or right.ndim != 1
        or len(left) != expected_rows
        or len(right) != expected_rows
        or np.any(left < 0)
        or np.any(right < left)
        or np.any(right >= source_frame_count)
    ):
        raise ValueError("PFNN source provenance exceeds its source frame count")
    result = []
    for record in ranges:
        authority = dict(record.authority)
        existing = authority.get("source_frame_count")
        if existing is not None and existing != source_frame_count:
            raise ValueError("PFNN range source frame count changed")
        authority["source_frame_count"] = source_frame_count
        result.append(replace(record, authority=authority))
    return tuple(result)


def _load_identity_stage(
    root: Path,
    *,
    expected_source_id: str | None = None,
    expected_split_group_id: str | None = None,
    expected_split: str | None = None,
) -> tuple[
    ArtifactSet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[RangeRecord, ...]
]:
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    members = value.get("members") if type(value) is dict else None
    expected = {
        "database.bin",
        "terrain_features.npy",
        "terrain_support.npy",
        "terrain_grid.npy",
        "source_left_indices.npy",
        "source_right_indices.npy",
        "source_alpha.npy",
        "ranges.json",
        "receipts.json",
    }
    fields = {
        "schema",
        "pipeline_sha256",
        "source_id",
        "canonical_source_id",
        "split_group_id",
        "split",
        "rows",
        "ranges",
        "members",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value.get("schema") != "g1-full-walking-pfnn-identity-stage/v2"
        or value.get("pipeline_sha256") != PFNN_PIPELINE_SHA256
        or type(members) is not dict
        or set(members) != expected
        or {path.name for path in root.iterdir()} != {*expected, "manifest.json"}
        or (
            expected_source_id is not None
            and value.get("source_id") != expected_source_id
        )
        or (
            expected_split_group_id is not None
            and value.get("split_group_id") != expected_split_group_id
        )
        or (expected_split is not None and value.get("split") != expected_split)
    ):
        raise ValueError(f"invalid PFNN identity stage: {root}")
    for name, descriptor in members.items():
        path = root / name
        if (
            type(descriptor) is not dict
            or set(descriptor) != {"size_bytes", "sha256"}
            or path.stat().st_size != descriptor.get("size_bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise ValueError(f"tampered PFNN identity stage member: {path}")
    artifacts = read_holden_database(root / "database.bin")
    rows = len(artifacts.positions)
    artifacts.terrain_features = _load_array(
        root / "terrain_features.npy", np.dtype("<f4"), (rows, 4)
    )
    artifacts.terrain_support = _load_array(
        root / "terrain_support.npy", np.dtype("<f4"), (rows, 3)
    )
    artifacts.validate()
    grid = _load_array(root / "terrain_grid.npy", np.dtype("<f4"), (rows, 36))
    left = _load_array(root / "source_left_indices.npy", np.dtype("<i4"), (rows,))
    right = _load_array(root / "source_right_indices.npy", np.dtype("<i4"), (rows,))
    alpha = _load_array(root / "source_alpha.npy", np.dtype("<f4"), (rows,))
    ranges_value = json.loads((root / "ranges.json").read_text(encoding="utf-8"))
    ranges = tuple(RangeRecord(**item) for item in ranges_value)
    if (
        len(ranges) != value.get("ranges")
        or rows != value.get("rows")
        or tuple(value.start for value in ranges)
        != tuple(int(item) for item in artifacts.range_starts)
        or tuple(value.stop for value in ranges)
        != tuple(int(item) for item in artifacts.range_stops)
        or any(
            record.canonical_source_id != value.get("canonical_source_id")
            or record.split_group_id != value.get("split_group_id")
            or record.split != value.get("split")
            for record in ranges
        )
    ):
        raise ValueError("PFNN identity stage counts changed")
    receipts_value = json.loads((root / "receipts.json").read_text(encoding="utf-8"))
    if type(receipts_value) is not list or not all(
        type(receipt) is dict for receipt in receipts_value
    ):
        raise ValueError("invalid PFNN identity receipts")
    ranges = bind_pfnn_source_frame_count(
        ranges,
        left,
        right,
        tuple(receipts_value),
        expected_source_id=value["source_id"],
    )
    return artifacts, grid, left, right, alpha, ranges


def _split_map(ledger: object) -> dict[str, tuple[str, str]]:
    assignments = getattr(ledger, "assignments", ledger)
    result: dict[str, tuple[str, str]] = {}
    for assignment in assignments:
        for source_id in assignment.source_ids:
            if source_id in result:
                raise ValueError("split ledger assigns one PFNN source twice")
            result[source_id] = (assignment.split_group_id, assignment.split)
    return result


def build_full_pfnn_lane(
    *,
    inventory: Path,
    split_ledger: Path,
    pfnn_root: Path,
    gmr_root: Path,
    retarget_root: Path,
    g1_xml: Path,
    workers: int,
    output: Path,
    publish: bool = True,
    work_root: Path | None = None,
) -> Path:
    """Build or resume all identities, optionally publishing the immutable lane."""

    if type(workers) is not int or workers < 1:
        raise ValueError("PFNN worker count must be a positive integer")
    xml = Path(g1_xml).resolve(strict=True)
    authenticate_pfnn_runtime(
        gmr_root=gmr_root,
        retarget_root=retarget_root,
        g1_xml=xml,
    )
    discovery = discover_full_pfnn_inventory(inventory, pfnn_root=pfnn_root)
    split_path = Path(split_ledger).resolve(strict=True)
    if sha256_file(split_path) != PFNN_SPLIT_LEDGER_SHA256:
        raise ValueError("walking split-ledger SHA-256 mismatch")
    frozen_inventory = load_inventory(inventory)
    from .full_walking_terrain_lmm_contracts import load_split_ledger

    ledger = load_split_ledger(
        split_path,
        inventory=frozen_inventory,
        expected_manifest_sha256=PFNN_SPLIT_LEDGER_SHA256,
    )
    assignments = _split_map(ledger)
    if any(record.source_id not in assignments for record in discovery.sources):
        raise ValueError("split ledger is missing a PFNN identity")
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable PFNN lane already exists: {destination}")
    stage_root = resolve_pfnn_work_root(output=destination, work_root=work_root)
    stage_root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Path] = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(discovery.sources)),
        mp_context=multiprocessing.get_context(pfnn_worker_start_method()),
    ) as pool:
        futures = {}
        for record in discovery.sources:
            group, split = assignments[record.source_id]
            future = pool.submit(
                _stage_identity,
                record,
                pfnn_root=Path(pfnn_root),
                gmr_root=Path(gmr_root),
                retarget_root=Path(retarget_root),
                g1_xml=xml,
                work_root=stage_root,
                split_group_id=group,
                split=split,
            )
            futures[future] = record.source_id
        for completed, future in enumerate(as_completed(futures), start=1):
            source_id = futures[future]
            stages[source_id] = Path(future.result())
            print(
                json.dumps(
                    {
                        "event": "pfnn_identity_complete",
                        "completed": completed,
                        "total": len(futures),
                        "source_id": source_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not publish:
        return stage_root
    artifacts_by_source = []
    grids = []
    lefts = []
    rights = []
    alphas = []
    source_ids: list[str] = []
    ranges: list[RangeRecord] = []
    cursor = 0
    for record in discovery.sources:
        artifacts, grid, left, right, alpha, identity_ranges = _load_identity_stage(
            stages[record.source_id],
            expected_source_id=record.source_id,
            expected_split_group_id=assignments[record.source_id][0],
            expected_split=assignments[record.source_id][1],
        )
        artifacts_by_source.append(artifacts)
        grids.append(grid)
        lefts.append(left)
        rights.append(right)
        alphas.append(alpha)
        for item in identity_ranges:
            ranges.append(
                replace(item, start=item.start + cursor, stop=item.stop + cursor)
            )
            source_ids.append(record.source_id)
        cursor += len(artifacts.positions)
    combined = ArtifactSet(
        positions=np.concatenate([value.positions for value in artifacts_by_source]),
        velocities=np.concatenate([value.velocities for value in artifacts_by_source]),
        rotations=np.concatenate([value.rotations for value in artifacts_by_source]),
        angular_velocities=np.concatenate(
            [value.angular_velocities for value in artifacts_by_source]
        ),
        parents=np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
        range_starts=np.asarray([value.start for value in ranges], dtype=np.int32),
        range_stops=np.asarray([value.stop for value in ranges], dtype=np.int32),
        contacts=np.concatenate([value.contacts for value in artifacts_by_source]),
        terrain_features=np.concatenate(
            [value.terrain_features for value in artifacts_by_source]
        ),
        terrain_support=np.concatenate(
            [value.terrain_support for value in artifacts_by_source]
        ),
    )
    combined.validate()
    if len(combined.positions) > EXPECTED_OUTPUT_ROWS:
        raise ValueError(
            "PFNN continuity processing exceeded gait-admitted row maximum"
        )
    lane = LaneArtifact(
        root=destination,
        manifest_sha256="",
        inventory_manifest_sha256=discovery.inventory_sha256,
        split_ledger_manifest_sha256=PFNN_SPLIT_LEDGER_SHA256,
        artifacts=combined,
        terrain_grid=np.ascontiguousarray(np.concatenate(grids), dtype=np.float32),
        source_ids=tuple(source_ids),
        source_left_indices=np.ascontiguousarray(np.concatenate(lefts), dtype=np.int32),
        source_right_indices=np.ascontiguousarray(
            np.concatenate(rights), dtype=np.int32
        ),
        source_alpha=np.ascontiguousarray(np.concatenate(alphas), dtype=np.float32),
        ranges=tuple(ranges),
    )
    publish_lane_exclusive(lane, destination)
    return destination


def audit_full_pfnn(*, inventory: Path, pfnn_root: Path) -> dict[str, object]:
    discovery = discover_full_pfnn_inventory(inventory, pfnn_root=pfnn_root)
    native_rows = output_rows = run_count = 0
    excluded: Counter[str] = Counter()
    families: Counter[str] = Counter()
    root = Path(pfnn_root).resolve(strict=True)
    for record in discovery.sources:
        gait = np.loadtxt(_record_input(record, "gait", root), dtype=np.float64)
        summary = summarize_pfnn_gait(gait)
        excluded.update(summary.excluded_counts)
        family = classify_pfnn_terrain(record.source_id)
        families[family] += 1
        ranges = admitted_pfnn_ranges(
            gait, np.zeros(len(gait) - 1, dtype=np.bool_), ((0, len(gait), family),)
        )
        native_rows += sum(stop - start for start, stop, _ in ranges)
        run_count += len(ranges)
        for start, stop, _ in ranges:
            rows, _left, _right, _alpha = safe_absolute_60hz_rows(
                len(gait), ((start, stop),)
            )
            output_rows += len(rows)
    receipt = {
        "schema": "g1-full-walking-pfnn-audit/v1",
        "status": "accepted",
        "identities": len(discovery.sources),
        "mirrors": sum(value.mirror_of is not None for value in discovery.sources),
        "admitted_native_rows": native_rows,
        "admitted_native_runs": run_count,
        "maximum_output_rows_60hz": output_rows,
        "excluded_counts": dict(excluded),
        "terrain_families": dict(sorted(families.items())),
        "inventory_sha256": discovery.inventory_sha256,
        "rest_pose": asdict(discovery.rest_pose),
    }
    expected = (
        native_rows == EXPECTED_NATIVE_ROWS
        and run_count == EXPECTED_NATIVE_RUNS
        and output_rows == EXPECTED_OUTPUT_ROWS
        and dict(excluded) == EXPECTED_EXCLUDED_COUNTS
    )
    if not expected:
        raise ValueError("full PFNN walking admission audit changed")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--inventory", type=Path, required=True)
    audit.add_argument("--pfnn-root", type=Path, required=True)
    for name in ("stage", "build"):
        command = commands.add_parser(name)
        command.add_argument("--inventory", type=Path, required=True)
        command.add_argument("--split-ledger", type=Path, required=True)
        command.add_argument("--pfnn-root", type=Path, required=True)
        command.add_argument("--gmr-root", type=Path, required=True)
        command.add_argument("--retarget-root", type=Path, required=True)
        command.add_argument("--g1-xml", type=Path, required=True)
        command.add_argument(
            "--workers", type=int, default=max(1, min(16, os.cpu_count() or 1))
        )
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--work-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "audit":
        result: object = audit_full_pfnn(
            inventory=arguments.inventory, pfnn_root=arguments.pfnn_root
        )
    else:
        publish = arguments.command == "build"
        result = {
            "status": "accepted" if publish else "staged",
            "output": str(
                build_full_pfnn_lane(
                    inventory=arguments.inventory,
                    split_ledger=arguments.split_ledger,
                    pfnn_root=arguments.pfnn_root,
                    gmr_root=arguments.gmr_root,
                    retarget_root=arguments.retarget_root,
                    g1_xml=arguments.g1_xml,
                    workers=arguments.workers,
                    output=arguments.output,
                    publish=publish,
                    work_root=arguments.work_root,
                )
            ),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_EXCLUDED_COUNTS",
    "PFNN_INVENTORY_SHA256",
    "PFNNDiscovery",
    "PFNNGaitSummary",
    "PFNNRangeTerrain",
    "admitted_pfnn_ranges",
    "audit_full_pfnn",
    "bind_pfnn_source_frame_count",
    "build_full_pfnn_lane",
    "classify_pfnn_terrain",
    "discover_full_pfnn_inventory",
    "fit_pfnn_range_terrain",
    "main",
    "process_pfnn_identity",
    "resolve_pfnn_work_root",
    "retarget_subprocess_argv",
    "reverse_mirror_terrain_lanes",
    "safe_absolute_60hz_rows",
    "summarize_pfnn_gait",
]
