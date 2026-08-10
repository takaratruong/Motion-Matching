"""Authenticated PFNN supplement for the overnight terrain learned matcher.

The bridge keeps every source range independent, converts the audited PFNN
core intervals to the overnight 25 Hz contract, and stores terrain heights in
the same support-relative representation used by the broad GRAIL/Takara bank.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from resources import quat as holden_quat
from resources.g1_terrain_builder.artifacts import (
    canonical_json_bytes,
    read_features,
    read_support_sidecar,
    read_terrain_sidecar,
    write_features,
    write_support_sidecar,
    write_terrain_sidecar,
)
from resources.g1_terrain_builder.database import (
    combine_clips,
    derive_velocities,
    forward_kinematics_arrays,
    read_holden_database,
    write_holden_database,
)
from resources.g1_terrain_builder.features import build_matching_features
from resources.g1_terrain_builder.resample import resample_map
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    ArtifactSet,
    FeatureSet,
    HoldenClip,
    SkeletonSpec,
    SourceClip,
    require_canonical_g1_skeleton,
)

SCHEMA = "pfnn-terrain-lmm-supplement/v1"
SOURCE_FPS = 120.0
REFERENCE_FPS = 60.0
TARGET_FPS = 25.0
SOURCE_INTERVAL_ROWS = 840
CORE_TRIM_ROWS = 120
CORE_ROWS_120HZ = 600
CORE_ROWS_60HZ = 300
OUTPUT_ROWS_25HZ = 125
HORIZONS = (8, 17, 25)
EXPECTED_TRAIN_STEM_FAMILIES = (
    ("LocomotionFlat02_000", "flat"),
    ("LocomotionFlat06_000", "flat"),
    ("LocomotionFlat08_000", "flat"),
    ("LocomotionFlat10_000", "flat"),
    ("LocomotionFlat11_000", "flat"),
    ("LocomotionFlat12_000", "flat"),
    ("WalkingUpSteps01_000", "stair"),
    ("WalkingUpSteps02_000", "stair"),
    ("WalkingUpSteps03_000", "stair"),
    ("WalkingUpSteps04_000", "stair"),
    ("WalkingUpSteps05_000", "stair"),
    ("WalkingUpSteps06_000", "stair"),
    ("WalkingUpSteps07_000", "stair"),
    ("WalkingUpSteps08_000", "stair"),
    ("WalkingUpSteps09_000", "stair"),
    ("WalkingUpSteps11_000", "stair"),
    ("WalkingUpSteps12_000", "stair"),
)
EXPECTED_COUNTS = {
    "excluded_clips": 4,
    "excluded_core_rows_120hz": 2_400,
    "excluded_output_rows_25hz": 500,
    "excluded_reference_rows_60hz": 1_200,
    "selected_clips": 17,
    "selected_core_rows_120hz": 10_200,
    "selected_flat_clips": 6,
    "selected_output_rows_25hz": 2_125,
    "selected_reference_rows_60hz": 5_100,
    "selected_source_rows_120hz": 14_280,
    "selected_vertical_clips": 11,
    "terrain_fit_artifacts_excluded": 3,
    "terrain_fit_artifacts_selected": 53,
    "terrain_fit_artifacts_total": 56,
}
_RECEIPT_CORE_FIELDS = {
    "schema",
    "status",
    "output_fps",
    "trajectory_horizons",
    "counts",
    "inputs",
    "skeleton",
    "ranges",
    "terrain",
    "validation",
}
_RECEIPT_PUBLISHED_FIELDS = _RECEIPT_CORE_FIELDS | {"artifacts"}
_RANGE_FIELDS = {
    "contact_source_rows_sha256",
    "core_rows_120hz",
    "core_start_120hz",
    "core_stop_120hz",
    "coverage",
    "duration_error_s",
    "family",
    "fk_max_error_m",
    "output_rows_25hz",
    "quaternion_norm_max_error",
    "reference_rows_60hz",
    "retarget",
    "retarget_receipt",
    "role",
    "source_bvh",
    "source_interval_start_120hz",
    "source_interval_stop_120hz",
    "source_rows_120hz",
    "start",
    "stem",
    "stop",
    "terrain_fits",
}
_DESCRIPTOR_FIELDS = {"path", "size_bytes", "sha256"}
_INPUT_FACTS = {
    "g1_xml": (
        26_914,
        "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
    ),
    "patches": (
        620_692_714,
        "344ec49b3aab3c93cd3f618c613e7d8fd8dfaf07964583a7461f0fb7593aae79",
    ),
    "retarget_manifest": (
        11_405,
        "6a477a223d38375124bc34e19d2fdd7eee795b3ef6339b44c21ce26b873d186b",
    ),
    "source_selection": (
        12_086,
        "a1babd74388c5b23593c562213d7116fc280c85bb1abd540bac7c6d28e0e3596",
    ),
    "vertical_dataset_manifest": (
        3_040,
        "eaea027337fa39c53464f15ad69e8946a381e44d08b28c1b027ef5add41311e9",
    ),
}
_EXPECTED_COVERAGE = {
    "LocomotionFlat02_000": ["left_turn", "right_turn"],
    "LocomotionFlat06_000": ["idle_transition", "straight"],
    "LocomotionFlat08_000": ["left_turn", "right_turn"],
    "LocomotionFlat10_000": ["left_turn", "right_turn"],
    "LocomotionFlat11_000": ["idle_transition", "straight"],
    "LocomotionFlat12_000": ["idle_transition", "straight"],
}
_EXPECTED_VERTICAL_FIT_COUNTS = (5, 4, 1, 5, 4, 5, 5, 6, 5, 7, 6)
_CONTACT_SOURCE_ROWS_SHA256 = (
    "3a60a0b2396a362ff97b4f9fc6c4c0db538ae1efdde58e0f9db94572e1063a39"
)
_FULL_TERRAIN_RECEIPT_SET_SHA256 = (
    "17ccb203f59a7cfc5440bdb1cc2813828a1b8e2c07fd0f37fc1d320846a002b3"
)
_TERRAIN_FIELDS = {
    "lookahead_m",
    "receipt_set_sha256",
    "semantics",
    "support_columns",
}
_VALIDATION_FIELDS = {
    "all_source_derivatives_range_local",
    "maximum_duration_error_s",
    "maximum_fk_error_m",
    "maximum_quaternion_norm_error",
    "range_local",
    "support_max_m",
    "support_min_m",
    "terrain_feature_max_abs_m",
    "terrain_feature_std_m",
}
_FIT_FIELDS = {
    "artifact",
    "receipt",
    "source_start_frame_120hz",
    "source_stop_frame_120hz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_claim_descriptor(value: object, label: str) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != _DESCRIPTOR_FIELDS
        or type(value.get("path")) is not str
        or not value["path"]
        or not Path(value["path"]).is_absolute()
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
        or not _valid_sha256(value.get("sha256"))
    ):
        raise ValueError(f"PFNN supplement {label} descriptor is invalid")
    return value


def _validate_nested_receipt(
    artifacts: ArtifactSet,
    normalized: dict[str, object],
) -> None:
    inputs = normalized.get("inputs")
    if type(inputs) is not dict or set(inputs) != set(_INPUT_FACTS):
        raise ValueError("PFNN supplement input authority inventory is invalid")
    for name, (expected_size, expected_sha256) in _INPUT_FACTS.items():
        descriptor = _validated_claim_descriptor(inputs[name], f"input {name}")
        if (
            descriptor["size_bytes"] != expected_size
            or descriptor["sha256"] != expected_sha256
        ):
            raise ValueError(f"PFNN supplement input {name} authority changed")

    ranges = normalized["ranges"]
    fit_total = 0
    vertical_index = 0
    duration_values: list[float] = []
    fk_values: list[float] = []
    quaternion_values: list[float] = []
    for value, (expected_stem, expected_family) in zip(
        ranges, EXPECTED_TRAIN_STEM_FAMILIES, strict=True
    ):
        source_start = value["source_interval_start_120hz"]
        source_stop = value["source_interval_stop_120hz"]
        core_start = value["core_start_120hz"]
        core_stop = value["core_stop_120hz"]
        coverage = (
            ["ascent", "descent"]
            if expected_family == "stair"
            else _EXPECTED_COVERAGE[expected_stem]
        )
        if (
            type(source_start) is not int
            or type(source_stop) is not int
            or type(core_start) is not int
            or type(core_stop) is not int
            or source_stop - source_start != SOURCE_INTERVAL_ROWS
            or core_start != source_start + CORE_TRIM_ROWS
            or core_stop != source_stop - CORE_TRIM_ROWS
            or value.get("coverage") != coverage
            or value.get("contact_source_rows_sha256")
            != _CONTACT_SOURCE_ROWS_SHA256
        ):
            raise ValueError("PFNN supplement range interval/coverage is invalid")
        retarget = _validated_claim_descriptor(
            value.get("retarget"), f"{expected_stem} retarget"
        )
        retarget_receipt = _validated_claim_descriptor(
            value.get("retarget_receipt"), f"{expected_stem} retarget receipt"
        )
        source_bvh = _validated_claim_descriptor(
            value.get("source_bvh"), f"{expected_stem} source BVH"
        )
        if (
            not str(retarget["path"]).endswith(".npz")
            or Path(retarget_receipt["path"])
            != Path(retarget["path"]).with_suffix(".receipt.json")
            or Path(source_bvh["path"]).name != f"{expected_stem}.bvh"
        ):
            raise ValueError("PFNN supplement range source companions are invalid")
        metrics = (
            ("duration_error_s", 0.04, duration_values),
            ("fk_max_error_m", 1.0e-5, fk_values),
            ("quaternion_norm_max_error", 1.0e-4, quaternion_values),
        )
        for name, limit, destination in metrics:
            metric = value.get(name)
            if (
                type(metric) not in (int, float)
                or not np.isfinite(metric)
                or metric < 0.0
                or metric > limit
            ):
                raise ValueError(f"PFNN supplement range {name} is invalid")
            destination.append(float(metric))

        fits = value.get("terrain_fits")
        if type(fits) is not list:
            raise ValueError("PFNN supplement terrain fit inventory is invalid")
        if expected_family == "flat":
            if fits:
                raise ValueError("PFNN supplement flat range has terrain fits")
            continue
        expected_fit_count = _EXPECTED_VERTICAL_FIT_COUNTS[vertical_index]
        vertical_index += 1
        if len(fits) != expected_fit_count:
            raise ValueError("PFNN supplement stair terrain fit count is invalid")
        previous_stop = core_start
        for fit in fits:
            if type(fit) is not dict or set(fit) != _FIT_FIELDS:
                raise ValueError("PFNN supplement terrain fit fields are invalid")
            fit_start = fit.get("source_start_frame_120hz")
            fit_stop = fit.get("source_stop_frame_120hz")
            artifact = _validated_claim_descriptor(
                fit.get("artifact"), "terrain fit artifact"
            )
            fit_receipt = _validated_claim_descriptor(
                fit.get("receipt"), "terrain fit receipt"
            )
            if (
                type(fit_start) is not int
                or type(fit_stop) is not int
                or fit_start != previous_stop
                or fit_stop <= fit_start
                or Path(fit_receipt["path"])
                != Path(artifact["path"]).with_suffix(".receipt.json")
            ):
                raise ValueError("PFNN supplement terrain fits do not tile the core")
            previous_stop = fit_stop
            fit_total += 1
        if previous_stop != core_stop:
            raise ValueError("PFNN supplement terrain fits do not tile the core")
    if fit_total != EXPECTED_COUNTS["terrain_fit_artifacts_selected"]:
        raise ValueError("PFNN supplement terrain fit total is invalid")

    terrain = normalized.get("terrain")
    if (
        type(terrain) is not dict
        or set(terrain) != _TERRAIN_FIELDS
        or terrain.get("semantics")
        != "PFNN fitted surface lookahead relative to current support"
        or terrain.get("lookahead_m") != [0.25, 0.5, 0.75, 1.0]
        or terrain.get("support_columns")
        != [
            "source_root_height_m",
            "source_left_toe_height_m",
            "source_right_toe_height_m",
        ]
    ):
        raise ValueError("PFNN supplement terrain contract is invalid")
    if terrain.get("receipt_set_sha256") != _FULL_TERRAIN_RECEIPT_SET_SHA256:
        raise ValueError("PFNN supplement terrain receipt set is invalid")

    validation = normalized.get("validation")
    expected_validation = {
        "range_local": True,
        "all_source_derivatives_range_local": True,
        "maximum_fk_error_m": max(fk_values),
        "maximum_duration_error_s": max(duration_values),
        "maximum_quaternion_norm_error": max(quaternion_values),
        "terrain_feature_max_abs_m": float(
            np.max(np.abs(artifacts.terrain_features))
        ),
        "terrain_feature_std_m": float(np.std(artifacts.terrain_features)),
        "support_min_m": float(np.min(artifacts.terrain_support)),
        "support_max_m": float(np.max(artifacts.terrain_support)),
    }
    if (
        type(validation) is not dict
        or set(validation) != _VALIDATION_FIELDS
        or validation != expected_validation
    ):
        raise ValueError("PFNN supplement validation aggregates are invalid")


def _validate_receipt_binary_contract(
    artifacts: ArtifactSet,
    features: FeatureSet,
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Bind the frozen receipt claims to the exact canonical binary payload."""

    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("artifacts must be an ArtifactSet")
    if not isinstance(features, FeatureSet):
        raise TypeError("features must be a FeatureSet")
    if not isinstance(receipt, Mapping):
        raise TypeError("receipt must be a mapping")
    artifacts.validate()
    features.validate()
    require_canonical_g1_skeleton(
        SkeletonSpec(G1_SKELETON_NAMES, np.asarray(artifacts.parents)),
        "PFNN supplement binary",
    )
    try:
        normalized = json.loads(canonical_json_bytes(dict(receipt)).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("PFNN supplement receipt is not stable JSON") from error
    fields = set(normalized)
    if fields != _RECEIPT_CORE_FIELDS and fields != _RECEIPT_PUBLISHED_FIELDS:
        raise ValueError("PFNN supplement receipt does not match binary fields")
    if (
        normalized["schema"] != SCHEMA
        or normalized["status"] != "accepted"
        or normalized["output_fps"] != TARGET_FPS
        or normalized["trajectory_horizons"] != list(HORIZONS)
    ):
        raise ValueError("PFNN supplement receipt contract is invalid")

    expected_skeleton = {
        "names": list(G1_SKELETON_NAMES),
        "parents": list(G1_SKELETON_PARENTS),
        "signature": G1_SKELETON_SIGNATURE,
    }
    if normalized["skeleton"] != expected_skeleton:
        raise ValueError(
            "PFNN supplement receipt does not match binary canonical G1 skeleton"
        )
    if normalized["counts"] != EXPECTED_COUNTS:
        raise ValueError("PFNN supplement receipt does not match binary counts")

    ranges = normalized["ranges"]
    if type(ranges) is not list or len(ranges) != len(EXPECTED_TRAIN_STEM_FAMILIES):
        raise ValueError("PFNN supplement receipt does not match binary ranges")
    if (
        len(artifacts.positions) != EXPECTED_COUNTS["selected_output_rows_25hz"]
        or len(features.values) != len(artifacts.positions)
        or not np.array_equal(
            artifacts.range_starts,
            np.arange(len(ranges), dtype=np.int32) * OUTPUT_ROWS_25HZ,
        )
        or not np.array_equal(
            artifacts.range_stops,
            np.arange(1, len(ranges) + 1, dtype=np.int32) * OUTPUT_ROWS_25HZ,
        )
    ):
        raise ValueError("PFNN supplement receipt does not match binary ranges")
    for index, ((expected_stem, expected_family), value) in enumerate(
        zip(EXPECTED_TRAIN_STEM_FAMILIES, ranges, strict=True)
    ):
        start = index * OUTPUT_ROWS_25HZ
        stop = start + OUTPUT_ROWS_25HZ
        if (
            type(value) is not dict
            or set(value) != _RANGE_FIELDS
            or value.get("stem") != expected_stem
            or value.get("family") != expected_family
            or value.get("role") != "train"
            or value.get("start") != start
            or value.get("stop") != stop
            or value.get("output_rows_25hz") != OUTPUT_ROWS_25HZ
            or value.get("source_rows_120hz") != SOURCE_INTERVAL_ROWS
            or value.get("core_rows_120hz") != CORE_ROWS_120HZ
            or value.get("reference_rows_60hz") != CORE_ROWS_60HZ
        ):
            raise ValueError("PFNN supplement receipt does not match binary ranges")

    _validate_nested_receipt(artifacts, normalized)
    recomputed = build_matching_features(artifacts, TARGET_FPS, HORIZONS)
    if any(
        not np.array_equal(getattr(features, field), getattr(recomputed, field))
        for field in ("values", "offset", "scale")
    ):
        raise ValueError("PFNN supplement binary differs from recomputed 31-D features")
    return normalized


@dataclass(frozen=True)
class PlaneSurface:
    """Small deterministic Holden-coordinate plane used for flat PFNN rows."""

    offset: float
    slope_x: float
    slope_z: float

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(value) for value in (self.offset, self.slope_x, self.slope_z)
        ):
            raise ValueError("plane surface coefficients must be finite")

    def height(self, x: float, z: float) -> float:
        value = self.offset + self.slope_x * float(x) + self.slope_z * float(z)
        if not np.isfinite(value):
            raise ValueError("plane surface produced a nonfinite height")
        return float(value)


@dataclass(frozen=True)
class PFNNSupplement:
    artifacts: ArtifactSet
    features: FeatureSet
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt",
            _validate_receipt_binary_contract(
                self.artifacts, self.features, self.receipt
            ),
        )


def summarize_admission(
    selection_items: Sequence[Mapping[str, object]], *, terrain_fit_count: int
) -> dict[str, int]:
    """Return exact counts for the frozen role=train PFNN admission policy."""

    if type(terrain_fit_count) is not int or terrain_fit_count < 0:
        raise ValueError("terrain fit count must be a nonnegative integer")
    if not selection_items:
        raise ValueError("PFNN selection must be nonempty")
    selected: list[Mapping[str, object]] = []
    excluded: list[Mapping[str, object]] = []
    stems: set[str] = set()
    for item in selection_items:
        stem = item.get("stem")
        role = item.get("role")
        start = item.get("start_frame_120hz")
        stop = item.get("stop_frame_120hz")
        coverage = item.get("coverage")
        if (
            type(stem) is not str
            or not stem
            or stem in stems
            or role not in ("train", "validation")
            or type(start) is not int
            or type(stop) is not int
            or stop - start != SOURCE_INTERVAL_ROWS
            or type(coverage) is not list
        ):
            raise ValueError("PFNN selection item is invalid")
        stems.add(stem)
        (selected if role == "train" else excluded).append(item)
    flat = sum(
        not ({"ascent", "descent"} <= set(item["coverage"])) for item in selected
    )
    vertical = len(selected) - flat
    return {
        "selected_clips": len(selected),
        "excluded_clips": len(excluded),
        "selected_flat_clips": flat,
        "selected_vertical_clips": vertical,
        "selected_source_rows_120hz": len(selected) * SOURCE_INTERVAL_ROWS,
        "selected_core_rows_120hz": len(selected) * CORE_ROWS_120HZ,
        "selected_reference_rows_60hz": len(selected) * CORE_ROWS_60HZ,
        "selected_output_rows_25hz": len(selected) * OUTPUT_ROWS_25HZ,
        "excluded_core_rows_120hz": len(excluded) * CORE_ROWS_120HZ,
        "excluded_reference_rows_60hz": len(excluded) * CORE_ROWS_60HZ,
        "excluded_output_rows_25hz": len(excluded) * OUTPUT_ROWS_25HZ,
        "terrain_fit_artifacts_total": terrain_fit_count,
    }


def select_half_open_segments(
    source_positions: np.ndarray,
    *,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Map source-frame positions to nonoverlapping half-open fit segments."""

    positions = np.asarray(source_positions, dtype=np.float64)
    starts64 = np.asarray(starts, dtype=np.int64)
    stops64 = np.asarray(stops, dtype=np.int64)
    if positions.ndim != 1 or not np.isfinite(positions).all():
        raise ValueError("source positions must be a finite vector")
    if (
        starts64.ndim != 1
        or starts64.shape != stops64.shape
        or len(starts64) < 1
        or np.any(starts64 >= stops64)
        or np.any(starts64[1:] < stops64[:-1])
    ):
        raise ValueError("terrain fit segments must be ordered and nonoverlapping")
    indices = np.searchsorted(stops64, positions, side="right")
    valid = indices < len(starts64)
    valid &= positions >= starts64[np.minimum(indices, len(starts64) - 1)]
    valid &= positions < stops64[np.minimum(indices, len(stops64) - 1)]
    if not np.all(valid):
        raise ValueError("terrain fit segment gap does not cover a source row")
    return indices.astype(np.int32)


def resample_contact_timeline(
    contacts: np.ndarray,
    *,
    source_fps: float = SOURCE_FPS,
    target_fps: float = TARGET_FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-resample one sealed four-probe contact range to bilateral rows."""

    values = np.asarray(contacts)
    if (
        values.dtype != np.bool_
        or values.ndim != 2
        or values.shape[1] != 4
        or len(values) < 1
    ):
        raise ValueError("PFNN contacts must have bool shape (T, 4)")
    left, right, alpha = resample_map(len(values), source_fps, target_fps)
    selected = np.where(alpha < np.float32(0.5), left, right).astype(np.int32)
    probes = values[selected]
    bilateral = np.column_stack(
        (np.any(probes[:, :2], axis=1), np.any(probes[:, 2:], axis=1))
    ).astype(np.uint8)
    return bilateral, selected


def annotate_holden_clip_terrain(
    clip: HoldenClip,
    *,
    surfaces: Sequence[object],
    surface_indices: np.ndarray,
    fps: float,
) -> None:
    """Fill support heights and four root-relative lookahead heights in place."""

    from resources.g1_terrain_builder.terrain import (
        build_facing_centerline,
        sample_terrain_features,
    )

    if not isinstance(clip, HoldenClip):
        raise TypeError("clip must be a HoldenClip")
    clip.validate()
    if float(fps) != TARGET_FPS:
        raise ValueError("PFNN supplement terrain annotation requires 25 Hz")
    if not surfaces or any(
        not callable(getattr(value, "height", None)) for value in surfaces
    ):
        raise TypeError("terrain surfaces must provide height(x, z)")
    indices = np.asarray(surface_indices)
    if (
        indices.shape != (len(clip.positions),)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(surfaces))
    ):
        raise ValueError("surface indices must select one surface per clip row")
    if clip.positions.shape[1] != len(G1_SKELETON_PARENTS):
        raise ValueError("PFNN terrain annotation requires the canonical G1 skeleton")
    global_positions, global_rotations = forward_kinematics_arrays(
        clip.positions,
        clip.rotations,
        np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
    )
    support_bones = (0, 7, 13)
    support = np.empty((len(global_positions), 3), dtype=np.float32)
    terrain = np.empty((len(global_positions), 4), dtype=np.float32)
    forward_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    future_rows = round(2.0 * fps)
    for frame in range(len(global_positions)):
        surface = surfaces[int(indices[frame])]
        for column, bone in enumerate(support_bones):
            point = global_positions[frame, bone]
            support[frame, column] = np.float32(
                surface.height(float(point[0]), float(point[2]))
            )
        stop = min(frame + future_rows + 1, len(global_positions))
        path = global_positions[frame:stop, 0][:, [0, 2]]
        headings3 = holden_quat.mul_vec(global_rotations[frame:stop, 0], forward_axis)
        centerline = build_facing_centerline(path[0], headings3[:, [0, 2]], path)
        terrain[frame] = sample_terrain_features(surface, centerline)
    if not np.isfinite(support).all() or not np.isfinite(terrain).all():
        raise ValueError("PFNN terrain annotation produced nonfinite channels")
    clip.terrain_support = support
    clip.terrain_features = terrain
    clip.validate()


def _descriptor(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    stored = (
        resolved.relative_to(relative_to).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": stored,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_and_sync(path: Path, writer, value: object) -> None:
    writer(path, value)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def publish_pfnn_supplement(supplement: PFNNSupplement, output: Path) -> Path:
    """Atomically publish the compact canonical ArtifactSet and its receipt."""

    if not isinstance(supplement, PFNNSupplement):
        raise TypeError("supplement must be a PFNNSupplement")
    validated_receipt = _validate_receipt_binary_contract(
        supplement.artifacts, supplement.features, supplement.receipt
    )
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite PFNN supplement: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        _write_and_sync(
            staging / "database.bin", write_holden_database, supplement.artifacts
        )
        _write_and_sync(
            staging / "terrain.bin",
            write_terrain_sidecar,
            supplement.artifacts.terrain_features,
        )
        _write_and_sync(
            staging / "support.bin",
            write_support_sidecar,
            supplement.artifacts.terrain_support,
        )
        _write_and_sync(staging / "features.bin", write_features, supplement.features)
        receipt = dict(validated_receipt)
        receipt["artifacts"] = {
            name: _descriptor(staging / name, relative_to=staging)
            for name in ("database.bin", "terrain.bin", "support.bin", "features.bin")
        }
        payload = canonical_json_bytes(receipt)
        manifest = staging / "manifest.json"
        with manifest.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(staging, destination)
        parent = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _checked_relative(root: Path, descriptor: object, label: str) -> Path:
    if type(descriptor) is not dict or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError(f"{label} descriptor is invalid")
    relative = descriptor["path"]
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} descriptor path is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} descriptor path is invalid")
    if (
        type(descriptor["size_bytes"]) is not int
        or path.stat().st_size != descriptor["size_bytes"]
        or _sha256(path) != descriptor["sha256"]
    ):
        raise ValueError(f"{label} digest mismatch")
    return path


def load_pfnn_supplement(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> PFNNSupplement:
    """Load and rehash a published PFNN supplement."""

    if expected_manifest_sha256 is None:
        raise TypeError("expected manifest SHA-256 is required")
    if not _valid_sha256(expected_manifest_sha256):
        raise ValueError("expected manifest SHA-256 is invalid")
    candidate = Path(path).expanduser().resolve(strict=True)
    manifest_path = candidate if candidate.is_file() else candidate / "manifest.json"
    root = manifest_path.parent
    try:
        members = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise ValueError("PFNN supplement publication is unreadable") from error
    expected_members = {
        "manifest.json",
        "database.bin",
        "terrain.bin",
        "support.bin",
        "features.bin",
    }
    if members != expected_members:
        raise ValueError("PFNN supplement publication contains unexpected files")
    try:
        payload = manifest_path.read_bytes()
        receipt = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("PFNN supplement manifest is invalid") from error
    if type(receipt) is not dict or canonical_json_bytes(receipt) != payload:
        raise ValueError("PFNN supplement manifest is not canonical JSON")
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("PFNN supplement differs from expected manifest SHA-256")
    records = receipt.get("artifacts")
    required = {"database.bin", "terrain.bin", "support.bin", "features.bin"}
    if type(records) is not dict or set(records) != required:
        raise ValueError("PFNN supplement artifact set is invalid")
    paths = {name: _checked_relative(root, records[name], name) for name in required}
    artifacts = read_holden_database(paths["database.bin"])
    artifacts.terrain_features = read_terrain_sidecar(paths["terrain.bin"])
    artifacts.terrain_support = read_support_sidecar(paths["support.bin"])
    artifacts.validate()
    features = read_features(paths["features.bin"])
    return PFNNSupplement(artifacts, features, receipt)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if type(value) is not dict:
        raise ValueError(f"{label} is invalid")
    return value


def _canonical_selection_digest(selection: Mapping[str, object]) -> str:
    if set(selection) != {"schema", "items", "sha256"}:
        raise ValueError("PFNN source selection fields are invalid")
    payload = {
        "schema": selection["schema"],
        "items": selection["items"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _absolute_descriptor(path: Path) -> dict[str, object]:
    return _descriptor(Path(path).expanduser().resolve(strict=True))


def _require_sha(path: Path, expected: object, label: str) -> None:
    if type(expected) is not str or _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")


def _flat_surface_digest(z_offset: float) -> str:
    return hashlib.sha256(
        json.dumps(
            {"schema": "g1-pfnn-flat-surface/v1", "z_offset_m": z_offset},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _FitRecord:
    stem: str
    path: Path
    receipt_path: Path
    fit: object
    artifact_sha256: str
    receipt_sha256: str
    start: int
    stop: int


@dataclass(frozen=True)
class _PlacedSurfaceAdapter:
    placed: object

    def height(self, x: float, z: float) -> float:
        result = np.asarray(
            self.placed.height_at(np.array([[float(x), -float(z)]], dtype=np.float64)),
            dtype=np.float64,
        )
        if result.shape != (1,) or not np.isfinite(result[0]):
            raise ValueError("placed PFNN terrain returned an invalid height")
        return float(result[0])


def _authenticate_source_documents(
    retarget_root: Path,
    pfnn_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, Mapping[str, object]],
    dict[str, Mapping[str, object]],
]:
    selection_path = retarget_root / "source-selection.json"
    manifest_path = retarget_root / "retarget-manifest.json"
    selection = _read_json(selection_path, "PFNN source selection")
    manifest = _read_json(manifest_path, "PFNN retarget manifest")
    digest = _canonical_selection_digest(selection)
    if (
        selection.get("schema") != "g1-pfnn-vertical-slice-selection/v1"
        or selection.get("sha256") != digest
        or type(selection.get("items")) is not list
        or len(selection["items"]) != 21
    ):
        raise ValueError("PFNN source selection contract is invalid")
    if (
        set(manifest) != {"schema", "status", "selection_sha256", "items"}
        or manifest.get("schema") != "g1-pfnn-vertical-slice-retarget/v1"
        or manifest.get("status") != "accepted"
        or manifest.get("selection_sha256") != digest
        or type(manifest.get("items")) is not list
        or len(manifest["items"]) != 21
    ):
        raise ValueError("PFNN retarget manifest contract is invalid")
    selection_fields = {
        "stem",
        "role",
        "start_frame_120hz",
        "stop_frame_120hz",
        "coverage",
        "bvh_sha256",
        "phase_sha256",
        "gait_sha256",
        "footsteps_sha256",
    }
    retarget_fields = {
        "stem",
        "role",
        "start_frame_120hz",
        "stop_frame_120hz",
        "coverage",
        "output",
        "output_sha256",
        "receipt",
        "receipt_sha256",
    }
    selected_by_stem: dict[str, Mapping[str, object]] = {}
    retarget_by_stem: dict[str, Mapping[str, object]] = {}
    for value in selection["items"]:
        if type(value) is not dict or set(value) != selection_fields:
            raise ValueError("PFNN source selection item fields are invalid")
        stem = value.get("stem")
        if type(stem) is not str or not stem or stem in selected_by_stem:
            raise ValueError("PFNN source selection stem is invalid")
        selected_by_stem[stem] = value
    for value in manifest["items"]:
        if type(value) is not dict or set(value) != retarget_fields:
            raise ValueError("PFNN retarget item fields are invalid")
        stem = value.get("stem")
        if type(stem) is not str or not stem or stem in retarget_by_stem:
            raise ValueError("PFNN retarget stem is invalid")
        retarget_by_stem[stem] = value
    if set(selected_by_stem) != set(retarget_by_stem):
        raise ValueError("PFNN selection and retarget stems differ")

    animation_root = pfnn_root / "data" / "animations"
    for stem in sorted(selected_by_stem):
        selected = selected_by_stem[stem]
        item = retarget_by_stem[stem]
        identity = (
            selected["role"],
            selected["start_frame_120hz"],
            selected["stop_frame_120hz"],
            selected["coverage"],
        )
        if identity != (
            item["role"],
            item["start_frame_120hz"],
            item["stop_frame_120hz"],
            item["coverage"],
        ):
            raise ValueError("PFNN selection and retarget identities differ")
        output_relative = item["output"]
        receipt_relative = item["receipt"]
        if (
            type(output_relative) is not str
            or type(receipt_relative) is not str
            or receipt_relative
            != str(Path(output_relative).with_suffix(".receipt.json"))
        ):
            raise ValueError("PFNN retarget receipt must be the output companion")
        output_path = (retarget_root / output_relative).resolve()
        receipt_path = (retarget_root / receipt_relative).resolve()
        if (
            not output_path.is_relative_to(retarget_root)
            or not receipt_path.is_relative_to(retarget_root)
            or not output_path.is_file()
            or not receipt_path.is_file()
        ):
            raise ValueError("PFNN retarget artifact path is invalid")
        if receipt_path != output_path.with_suffix(".receipt.json"):
            raise ValueError("PFNN retarget receipt must be the output companion")
        for path, field in (
            (animation_root / f"{stem}.bvh", "bvh_sha256"),
            (animation_root / f"{stem}.phase", "phase_sha256"),
            (animation_root / f"{stem}.gait", "gait_sha256"),
            (animation_root / f"{stem}_footsteps.txt", "footsteps_sha256"),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            _require_sha(path, selected[field], f"{stem} {field}")
        for path, digest_field in (
            (output_path, "output_sha256"),
            (receipt_path, "receipt_sha256"),
        ):
            _require_sha(path, item[digest_field], f"{stem} {digest_field}")
    return selection, manifest, selected_by_stem, retarget_by_stem


def _authenticate_terrain_inventory(
    *,
    retarget_root: Path,
    selected_by_stem: Mapping[str, Mapping[str, object]],
    patches_sha256: str,
) -> tuple[dict[str, tuple[_FitRecord, ...]], str]:
    from mm_sonic.pfnn_terrain_fit import SCHEMA as FIT_SCHEMA
    from mm_sonic.pfnn_terrain_fit import load_terrain_fit

    terrain_root = retarget_root / "vertical-corpus" / "terrain"
    paths = sorted(terrain_root.glob("*.npz"))
    receipts = sorted(terrain_root.glob("*.receipt.json"))
    if len(paths) != 56 or len(receipts) != 56:
        raise ValueError(
            "PFNN terrain inventory must contain 56 artifact/receipt pairs"
        )
    records: dict[str, list[_FitRecord]] = {}
    receipt_fields = {
        "schema",
        "status",
        "artifact_sha256",
        "source_sha256",
        "patches_sha256",
        "selected_patch_index",
        "patch_coord",
        "fitting_error",
        "source_start_frame",
        "source_frame_count",
        "cycle_start_frame",
        "cycle_stop_frame",
    }
    for path in paths:
        receipt_path = path.with_suffix(".receipt.json")
        if not receipt_path.is_file():
            raise ValueError(f"missing PFNN terrain receipt: {receipt_path}")
        name_parts = path.stem.split("__", 1)
        if len(name_parts) != 2 or name_parts[0] not in selected_by_stem:
            raise ValueError("PFNN terrain artifact stem is invalid")
        stem = name_parts[0]
        selected = selected_by_stem[stem]
        if not ({"ascent", "descent"} <= set(selected["coverage"])):
            raise ValueError("flat PFNN source unexpectedly owns a terrain fit")
        receipt = _read_json(receipt_path, "PFNN terrain receipt")
        artifact_sha = _sha256(path)
        if (
            set(receipt) != receipt_fields
            or receipt.get("schema") != FIT_SCHEMA
            or receipt.get("status") != "accepted"
            or receipt.get("artifact_sha256") != artifact_sha
            or receipt.get("source_sha256") != selected["bvh_sha256"]
            or receipt.get("patches_sha256") != patches_sha256
        ):
            raise ValueError("PFNN terrain receipt contract is invalid")
        fit = load_terrain_fit(path)
        expected = {
            "source_sha256": fit.source_sha256,
            "patches_sha256": fit.patches_sha256,
            "selected_patch_index": fit.selected_patch_index,
            "patch_coord": fit.patch_coord.tolist(),
            "fitting_error": fit.fitting_error,
            "source_start_frame": fit.source_start_frame,
            "source_frame_count": fit.source_frame_count,
            "cycle_start_frame": fit.cycle_start_frame,
            "cycle_stop_frame": fit.cycle_stop_frame,
        }
        if any(receipt[key] != value for key, value in expected.items()):
            raise ValueError("PFNN terrain receipt and artifact differ")
        record = _FitRecord(
            stem=stem,
            path=path.resolve(),
            receipt_path=receipt_path.resolve(),
            fit=fit,
            artifact_sha256=artifact_sha,
            receipt_sha256=_sha256(receipt_path),
            start=fit.source_start_frame,
            stop=fit.source_start_frame + fit.source_frame_count,
        )
        records.setdefault(stem, []).append(record)

    flat_digest = _flat_surface_digest(
        __import__(
            "mm_sonic.terrain_pfnn.pfnn_surface",
            fromlist=["PFNN_G1_Z_OFFSET_M"],
        ).PFNN_G1_Z_OFFSET_M
    )
    receipt_set: list[tuple[str, int, int, str]] = []
    result: dict[str, tuple[_FitRecord, ...]] = {}
    for stem, selected in sorted(selected_by_stem.items()):
        core_start = int(selected["start_frame_120hz"]) + CORE_TRIM_ROWS
        core_stop = int(selected["stop_frame_120hz"]) - CORE_TRIM_ROWS
        vertical = {"ascent", "descent"} <= set(selected["coverage"])
        if vertical:
            ordered = tuple(
                sorted(records.get(stem, ()), key=lambda value: value.start)
            )
            if (
                not ordered
                or ordered[0].start != core_start
                or ordered[-1].stop != core_stop
                or any(left.stop != right.start for left, right in pairwise(ordered))
            ):
                raise ValueError(
                    f"{stem}: PFNN terrain fit segments do not tile the core"
                )
            result[stem] = ordered
            receipt_set.extend(
                (stem, record.start, record.stop, record.artifact_sha256)
                for record in ordered
            )
        else:
            if stem in records:
                raise ValueError(f"{stem}: flat PFNN source has terrain artifacts")
            result[stem] = ()
            receipt_set.append((stem, core_start, core_stop, flat_digest))
    if set(records) != {stem for stem, values in result.items() if values}:
        raise ValueError("PFNN terrain inventory contains an unselected source")
    digest = hashlib.sha256(
        json.dumps(sorted(receipt_set), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result, digest


def _load_core_source(
    motion_path: Path,
    *,
    stem: str,
    source_start: int,
    terrain_id: str,
) -> SourceClip:
    with np.load(motion_path, allow_pickle=False) as archive:
        root = np.asarray(archive["root_pos"], dtype=np.float64)
        xyzw = np.asarray(archive["root_quat"], dtype=np.float64)
        dof = np.asarray(archive["dof"], dtype=np.float64)
    if (
        root.shape != (SOURCE_INTERVAL_ROWS, 3)
        or xyzw.shape != (SOURCE_INTERVAL_ROWS, 4)
        or dof.shape != (SOURCE_INTERVAL_ROWS, 29)
    ):
        raise ValueError(f"{stem}: authenticated PFNN retarget shapes changed")
    core = slice(CORE_TRIM_ROWS, SOURCE_INTERVAL_ROWS - CORE_TRIM_ROWS)
    qpos = np.empty((CORE_ROWS_120HZ, 36), dtype=np.float32)
    qpos[:, :3] = root[core]
    quaternion = xyzw[core][:, (3, 0, 1, 2)]
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError(f"{stem}: PFNN root quaternion is zero")
    qpos[:, 3:7] = quaternion / norms
    qpos[:, 7:] = dof[core]
    result = SourceClip(
        stem,
        SOURCE_FPS,
        qpos,
        np.arange(
            source_start + CORE_TRIM_ROWS,
            source_start + SOURCE_INTERVAL_ROWS - CORE_TRIM_ROWS,
            dtype=np.int32,
        ),
        terrain_id,
    )
    result.validate()
    return result


def build_pfnn_supplement(
    *,
    retarget_root: Path,
    pfnn_root: Path,
    g1_xml: Path,
) -> PFNNSupplement:
    """Build the authenticated 17-source PFNN train supplement at exact 25 Hz."""

    from mm_sonic.build_g1_pfnn_vertical_dataset import load_vertical_dataset
    from mm_sonic.grail_terrain_source import G1MujocoFK
    from mm_sonic.pfnn_terrain_fit import released_pfnn_contacts_for_interval
    from mm_sonic.terrain_pfnn.pfnn_surface import (
        PFNN_G1_Z_OFFSET_M,
        PlacedPFNNSurface,
    )
    from mm_sonic.terrain_pfnn.sources import load_pfnn_retarget_source
    from resources.g1_terrain_builder.kinematics import (
        G1Kinematics,
        convert_source_clip,
    )

    retarget = Path(retarget_root).expanduser().resolve(strict=True)
    released = Path(pfnn_root).expanduser().resolve(strict=True)
    model_path = Path(g1_xml).expanduser().resolve(strict=True)
    patches_path = (released / "patches.npz").resolve(strict=True)
    selection, _manifest, selected_by_stem, retarget_by_stem = (
        _authenticate_source_documents(retarget, released)
    )
    counts = summarize_admission(selection["items"], terrain_fit_count=56)
    frozen_counts = {
        "selected_clips": 17,
        "excluded_clips": 4,
        "selected_flat_clips": 6,
        "selected_vertical_clips": 11,
        "selected_reference_rows_60hz": 5_100,
        "selected_output_rows_25hz": 2_125,
    }
    if any(counts[key] != expected for key, expected in frozen_counts.items()):
        raise ValueError("PFNN overnight admission counts changed")

    retarget_manifest_path = retarget / "retarget-manifest.json"
    dataset_root = retarget / "vertical-corpus" / "dataset"
    dataset = load_vertical_dataset(dataset_root)
    if (
        dataset.selection_sha256 != selection["sha256"]
        or dataset.retarget_manifest_sha256 != _sha256(retarget_manifest_path)
        or dict(dataset.source_roles)
        != {stem: str(item["role"]) for stem, item in sorted(selected_by_stem.items())}
    ):
        raise ValueError(
            "PFNN vertical dataset provenance differs from retarget inputs"
        )
    patches_sha = _sha256(patches_path)
    fits_by_stem, terrain_receipt_set = _authenticate_terrain_inventory(
        retarget_root=retarget,
        selected_by_stem=selected_by_stem,
        patches_sha256=patches_sha,
    )
    if terrain_receipt_set != dataset.terrain_receipt_set_sha256:
        raise ValueError("PFNN terrain receipt set digest mismatch")
    selected_fit_count = sum(
        len(fits_by_stem[stem])
        for stem, item in selected_by_stem.items()
        if item["role"] == "train"
    )
    excluded_fit_count = 56 - selected_fit_count
    if (selected_fit_count, excluded_fit_count) != (53, 3):
        raise ValueError("PFNN admitted/excluded terrain fit counts changed")
    counts.update(
        {
            "terrain_fit_artifacts_selected": selected_fit_count,
            "terrain_fit_artifacts_excluded": excluded_fit_count,
        }
    )

    authentication_fk = G1MujocoFK(model_path)
    validated_motion: dict[str, Path] = {}
    for stem in sorted(selected_by_stem):
        selected = selected_by_stem[stem]
        item = retarget_by_stem[stem]
        motion_path = (retarget / str(item["output"])).resolve(strict=True)
        validated = load_pfnn_retarget_source(
            motion_path,
            authentication_fk,
            clip_id=stem,
            terrain_id=(
                "pfnn_vertical"
                if {"ascent", "descent"} <= set(selected["coverage"])
                else "flat"
            ),
            expected_source_sha256=str(selected["bvh_sha256"]),
            expected_start_frame=int(selected["start_frame_120hz"]),
        )
        if validated.frame_count != 210:
            raise ValueError(f"{stem}: authenticated PFNN 30 Hz row count changed")
        validated_motion[stem] = motion_path

    kinematics = G1Kinematics(str(model_path))
    canonical_skeleton = SkeletonSpec(
        G1_SKELETON_NAMES, np.asarray(G1_SKELETON_PARENTS, dtype=np.int32)
    )
    clips: list[HoldenClip] = []
    range_receipts: list[dict[str, object]] = []
    cursor = 0
    validation_reports: list[dict[str, float]] = []
    animation_root = released / "data" / "animations"
    for stem in sorted(
        name for name, item in selected_by_stem.items() if item["role"] == "train"
    ):
        selected = selected_by_stem[stem]
        item = retarget_by_stem[stem]
        vertical = {"ascent", "descent"} <= set(selected["coverage"])
        source_start = int(selected["start_frame_120hz"])
        core_start = source_start + CORE_TRIM_ROWS
        core_stop = int(selected["stop_frame_120hz"]) - CORE_TRIM_ROWS
        source = _load_core_source(
            validated_motion[stem],
            stem=stem,
            source_start=source_start,
            terrain_id="pfnn_vertical" if vertical else "flat",
        )
        clip, skeleton, report = convert_source_clip(
            source, kinematics, target_fps=TARGET_FPS, root_filter_mode="interp"
        )
        require_canonical_g1_skeleton(skeleton, f"{stem} PFNN skeleton")
        if len(clip.positions) != OUTPUT_ROWS_25HZ:
            raise ValueError(f"{stem}: PFNN 25 Hz row count changed")
        with np.errstate(divide="ignore", invalid="ignore"):
            clip.velocities, clip.angular_velocities = derive_velocities(
                clip.positions, clip.rotations, TARGET_FPS
            )

        fit_records = fits_by_stem[stem]
        if vertical:
            contacts_120 = np.concatenate(
                [np.asarray(record.fit.source_contacts) for record in fit_records],
                axis=0,
            )
            surfaces = tuple(
                _PlacedSurfaceAdapter(PlacedPFNNSurface(record.fit))
                for record in fit_records
            )
            source_positions = clip.source_left_indices.astype(np.float64) * (
                1.0 - clip.source_alpha.astype(np.float64)
            ) + clip.source_right_indices.astype(np.float64) * clip.source_alpha.astype(
                np.float64
            )
            surface_indices = select_half_open_segments(
                source_positions,
                starts=np.asarray([record.start for record in fit_records]),
                stops=np.asarray([record.stop for record in fit_records]),
            )
        else:
            contacts_120 = released_pfnn_contacts_for_interval(
                pfnn_root=released,
                source=animation_root / f"{stem}.bvh",
                display_start_frame=core_start,
                display_frame_count=CORE_ROWS_120HZ,
            )
            surfaces = (PlaneSurface(PFNN_G1_Z_OFFSET_M, 0.0, 0.0),)
            surface_indices = np.zeros(OUTPUT_ROWS_25HZ, dtype=np.int32)
        if contacts_120.shape != (CORE_ROWS_120HZ, 4):
            raise ValueError(f"{stem}: PFNN contact core does not have 600 rows")
        clip.contacts, contact_rows = resample_contact_timeline(contacts_120)
        annotate_holden_clip_terrain(
            clip,
            surfaces=surfaces,
            surface_indices=surface_indices,
            fps=TARGET_FPS,
        )
        stop = cursor + len(clip.positions)
        terrain_descriptors = [
            {
                "artifact": _absolute_descriptor(record.path),
                "receipt": _absolute_descriptor(record.receipt_path),
                "source_start_frame_120hz": record.start,
                "source_stop_frame_120hz": record.stop,
            }
            for record in fit_records
        ]
        range_receipts.append(
            {
                "stem": stem,
                "role": "train",
                "family": "stair" if vertical else "flat",
                "coverage": list(selected["coverage"]),
                "start": cursor,
                "stop": stop,
                "source_interval_start_120hz": source_start,
                "source_interval_stop_120hz": int(selected["stop_frame_120hz"]),
                "core_start_120hz": core_start,
                "core_stop_120hz": core_stop,
                "source_rows_120hz": SOURCE_INTERVAL_ROWS,
                "core_rows_120hz": CORE_ROWS_120HZ,
                "reference_rows_60hz": CORE_ROWS_60HZ,
                "output_rows_25hz": OUTPUT_ROWS_25HZ,
                "retarget": _absolute_descriptor(validated_motion[stem]),
                "retarget_receipt": _absolute_descriptor(
                    (retarget / str(item["receipt"])).resolve(strict=True)
                ),
                "source_bvh": _absolute_descriptor(animation_root / f"{stem}.bvh"),
                "terrain_fits": terrain_descriptors,
                "contact_source_rows_sha256": hashlib.sha256(
                    np.ascontiguousarray(contact_rows, dtype="<i4").tobytes()
                ).hexdigest(),
                "fk_max_error_m": float(report["fk_max_error_m"]),
                "duration_error_s": float(report["duration_error_s"]),
                "quaternion_norm_max_error": float(report["quaternion_norm_max_error"]),
            }
        )
        validation_reports.append(report)
        clips.append(clip)
        cursor = stop

    artifacts = combine_clips(clips, canonical_skeleton)
    features = build_matching_features(artifacts, TARGET_FPS, HORIZONS)
    if (
        cursor != counts["selected_output_rows_25hz"]
        or features.values.shape != (cursor, 31)
        or not np.isfinite(artifacts.terrain_support).all()
        or float(np.max(np.abs(artifacts.terrain_features))) <= 1.0e-3
    ):
        raise ValueError("PFNN supplement output validation failed")
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "accepted",
        "output_fps": TARGET_FPS,
        "trajectory_horizons": list(HORIZONS),
        "counts": counts,
        "inputs": {
            "source_selection": _absolute_descriptor(
                retarget / "source-selection.json"
            ),
            "retarget_manifest": _absolute_descriptor(retarget_manifest_path),
            "vertical_dataset_manifest": _absolute_descriptor(
                dataset_root / "manifest.json"
            ),
            "patches": _absolute_descriptor(patches_path),
            "g1_xml": _absolute_descriptor(model_path),
        },
        "skeleton": {
            "names": list(G1_SKELETON_NAMES),
            "parents": list(G1_SKELETON_PARENTS),
            "signature": G1_SKELETON_SIGNATURE,
        },
        "terrain": {
            "semantics": "PFNN fitted surface lookahead relative to current support",
            "lookahead_m": [0.25, 0.50, 0.75, 1.00],
            "support_columns": [
                "source_root_height_m",
                "source_left_toe_height_m",
                "source_right_toe_height_m",
            ],
            "receipt_set_sha256": terrain_receipt_set,
        },
        "ranges": range_receipts,
        "validation": {
            "range_local": True,
            "all_source_derivatives_range_local": True,
            "maximum_fk_error_m": max(
                float(value["fk_max_error_m"]) for value in validation_reports
            ),
            "maximum_duration_error_s": max(
                float(value["duration_error_s"]) for value in validation_reports
            ),
            "maximum_quaternion_norm_error": max(
                float(value["quaternion_norm_max_error"])
                for value in validation_reports
            ),
            "terrain_feature_max_abs_m": float(
                np.max(np.abs(artifacts.terrain_features))
            ),
            "terrain_feature_std_m": float(np.std(artifacts.terrain_features)),
            "support_min_m": float(np.min(artifacts.terrain_support)),
            "support_max_m": float(np.max(artifacts.terrain_support)),
        },
    }
    return PFNNSupplement(artifacts, features, receipt)


__all__ = [
    "CORE_ROWS_120HZ",
    "HORIZONS",
    "OUTPUT_ROWS_25HZ",
    "SCHEMA",
    "TARGET_FPS",
    "PFNNSupplement",
    "PlaneSurface",
    "annotate_holden_clip_terrain",
    "build_pfnn_supplement",
    "load_pfnn_supplement",
    "publish_pfnn_supplement",
    "resample_contact_timeline",
    "select_half_open_segments",
    "summarize_admission",
]
