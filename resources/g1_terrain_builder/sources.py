from dataclasses import dataclass
from collections import Counter
import os
import joblib
import numpy as np

from .schema import SourceClip


GRAIL_OBJECT_POSITION_STATIC_MAX_DEVIATION_M = 0.005
GRAIL_OBJECT_QUATERNION_STATIC_ANGLE_ATOL_RAD = 1e-4
GRAIL_OBJECT_QUATERNION_NORM_ATOL = 2e-4
GRAIL_MOVING_SLOPE_OBJECT_BASENAMES = (
    "terrain_slopes__slope_009__007",
    "terrain_slopes__slope_011__000",
    "terrain_slopes__slope_014__004",
    "terrain_slopes__slope_014__006",
    "terrain_slopes__slope_018__009",
    "terrain_slopes__slope_038__004",
    "terrain_slopes__slope_052__004",
    "terrain_slopes__slope_055__009",
    "terrain_slopes__slope_079__000",
    "terrain_slopes__slope_079__006",
    "terrain_slopes__slope_088__006",
    "terrain_slopes__slope_092__004",
    "terrain_slopes__slope_092__007",
    "terrain_slopes__slope_127__004",
    "terrain_slopes__slope_145__007",
    "terrain_slopes__slope_154__005",
    "terrain_slopes__slope_154__007",
    "terrain_slopes__slope_157__001",
    "terrain_slopes__slope_162__001",
    "terrain_slopes__slope_165__001",
    "terrain_slopes__slope_165__008",
    "terrain_slopes__slope_187__002",
    "terrain_slopes__slope_193__000",
)
GRAIL_STATIC_SLOPE_CLIP_COUNT = 1_857
GRAIL_STATIC_SLOPE_FRAME_COUNT = 464_250
GRAIL_PARTITION_SOURCE_COUNTS = {
    "curb": 1_769,
    "slope": 1_880,
    "stair_p1": 6_094,
    "stair_p2": 6_094,
}


@dataclass(frozen=True)
class GrailSourceAsset:
    name: str
    partition: str
    family: str
    robot_path: str
    object_path: str
    usd_path: str
    release_surface: bool


@dataclass(frozen=True)
class GrailSourceCorpus:
    all_sources: tuple[GrailSourceAsset, ...]
    motion_sources: tuple[GrailSourceAsset, ...]
    skipped_basenames: tuple[str, ...]
    diagnostic: bool


def _regular_stem_map(directory: str, suffix: str, label: str) -> dict:
    try:
        entries = tuple(os.scandir(directory))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing GRAIL {label} directory: {directory}") \
            from None
    result = {}
    for entry in entries:
        if not entry.name.endswith(suffix):
            continue
        if not entry.is_file(follow_symlinks=False):
            raise ValueError(f"GRAIL {label} must be a regular file: {entry.path}")
        stem = entry.name[:-len(suffix)]
        if not stem or stem in result:
            raise ValueError(f"duplicate GRAIL {label} basename: {stem}")
        result[stem] = os.path.abspath(entry.path)
    return result


def _partition_source_assets(
    dataset_root: str,
    partition: str,
    family: str,
    object_modality: str,
) -> tuple[GrailSourceAsset, ...]:
    root = os.path.join(os.path.abspath(os.fspath(dataset_root)), "data", partition)
    robot = _regular_stem_map(
        os.path.join(root, "robot"), ".pkl", f"{partition} robot")
    objects = _regular_stem_map(
        os.path.join(root, object_modality), ".pkl",
        f"{partition} {object_modality}")
    usd = _regular_stem_map(
        os.path.join(root, "object_usd"), ".usd", f"{partition} USD")
    stems = set(robot)
    if not stems or stems != set(objects) or stems != set(usd):
        raise ValueError(
            f"{partition} robot/object/USD basename coverage mismatch")
    return tuple(
        GrailSourceAsset(
            stem,
            partition,
            family,
            robot[stem],
            objects[stem],
            usd[stem],
            object_modality == "objects",
        )
        for stem in sorted(stems)
    )


def discover_grail_source_assets(
    dataset_root: str,
    source_limit_per_family: int | None = None,
    *,
    moving_slope_basenames=GRAIL_MOVING_SLOPE_OBJECT_BASENAMES,
    expected_partition_counts: dict | None = None,
) -> GrailSourceCorpus:
    """Audit exact local pairings, exclude locked moving slopes, then limit.

    The cap is semantic: stair-p1 and stair-p2 together form one ``stair``
    family.  Pairing and global duplicate audits intentionally happen before
    exclusion and before cap selection.
    """
    if source_limit_per_family is not None and (
        isinstance(source_limit_per_family, (bool, np.bool_))
        or not isinstance(source_limit_per_family, (int, np.integer))
        or source_limit_per_family < 0
    ):
        raise ValueError("source limit per family must be a non-negative integer")
    partitions = (
        ("curb", "curb", "recon"),
        ("slope", "slope", "objects"),
        ("stair_p1", "stair", "objects"),
        ("stair_p2", "stair", "objects"),
    )
    by_partition = {
        partition: _partition_source_assets(
            dataset_root, partition, family, object_modality)
        for partition, family, object_modality in partitions
    }
    if expected_partition_counts is not None:
        if type(expected_partition_counts) is not dict:
            raise TypeError("expected partition counts must be a dictionary")
        observed = {
            partition: len(by_partition[partition])
            for partition, _family, _modality in partitions
        }
        if observed != expected_partition_counts:
            raise ValueError(
                f"GRAIL partition source counts changed: {observed}")

    all_paired = tuple(
        item
        for partition, _family, _modality in partitions
        for item in by_partition[partition]
    )
    names = [item.name for item in all_paired]
    if len(names) != len(set(names)):
        duplicates = sorted(
            name for name, count in Counter(names).items() if count > 1)
        raise ValueError(f"duplicate source basename across families: {duplicates}")
    paths = [
        path
        for item in all_paired
        for path in (item.robot_path, item.object_path, item.usd_path)
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate source path across families")

    moving = tuple(moving_slope_basenames)
    if len(moving) != len(set(moving)) or tuple(sorted(moving)) != moving:
        raise ValueError("moving slope exclusions must be unique and lexical")
    slope_names = {item.name for item in by_partition["slope"]}
    missing_exclusions = sorted(set(moving) - slope_names)
    if missing_exclusions:
        raise ValueError(
            f"locked slope exclusion is missing: {missing_exclusions}")
    moving_set = set(moving)
    accepted_slope = tuple(
        item for item in by_partition["slope"]
        if item.name not in moving_set
    )
    accepted = (
        by_partition["curb"]
        + accepted_slope
        + by_partition["stair_p1"]
        + by_partition["stair_p2"]
    )
    if source_limit_per_family is None:
        selected = accepted
    else:
        limit = int(source_limit_per_family)
        selected = tuple(
            item for family in ("curb", "slope", "stair")
            for item in tuple(
                candidate for candidate in accepted
                if candidate.family == family
            )[:limit]
        )
    return GrailSourceCorpus(
        accepted,
        selected,
        moving,
        source_limit_per_family is not None,
    )


@dataclass(frozen=True)
class GrailObjectPose:
    root_pos: np.ndarray
    root_quat: np.ndarray
    fps: float
    scale: np.ndarray


def _grail_real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) \
            or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"GRAIL object {name} must be a real numeric array")
    array = np.array(array, np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"GRAIL object {name} must contain only finite values")
    return array


def load_grail_object_pose(path: str) -> GrailObjectPose:
    records = joblib.load(path)
    if type(records) is not dict or len(records) != 1:
        count = len(records) if isinstance(records, dict) else "non-dict"
        raise ValueError(f"{path}: expected one object record, got {count}")
    record = next(iter(records.values()))
    if type(record) is not dict:
        raise TypeError(f"{path}: object record must be a dictionary")
    for name in ("root_pos", "root_quat", "fps", "scale"):
        if name not in record:
            raise ValueError(f"{path}: object record is missing {name}")

    root_pos = _grail_real_array(record["root_pos"], "root_pos")
    root_quat_xyzw = _grail_real_array(
        record["root_quat"], "root_quat")
    scale = _grail_real_array(record["scale"], "scale")
    if root_pos.ndim != 3 or root_pos.shape[1:] != (1, 3) \
            or not len(root_pos):
        raise ValueError(
            f"{path}: root_pos must have non-empty shape (T, 1, 3), "
            f"got {root_pos.shape}")
    if root_quat_xyzw.shape != (len(root_pos), 1, 4):
        raise ValueError(
            f"{path}: root_quat must have shape ({len(root_pos)}, 1, 4), "
            f"got {root_quat_xyzw.shape}")
    if scale.shape != (3, 1):
        raise ValueError(f"{path}: scale must have shape (3, 1), got {scale.shape}")
    if np.any(scale <= 0.0):
        raise ValueError(f"{path}: scale must be finite and positive")

    fps_value = np.asarray(record["fps"])
    if fps_value.shape != () or not np.issubdtype(fps_value.dtype, np.number) \
            or np.issubdtype(fps_value.dtype, np.complexfloating):
        raise TypeError(f"{path}: fps must be a real numeric scalar")
    fps = float(fps_value)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: fps must be finite and positive")

    norms = np.linalg.norm(root_quat_xyzw[:, 0], axis=1)
    if np.any(np.abs(norms - 1.0) > GRAIL_OBJECT_QUATERNION_NORM_ATOL):
        raise ValueError(f"{path}: root_quat must contain normalized quaternions")
    position = np.median(root_pos[:, 0], axis=0)
    position_deviations = np.linalg.norm(
        root_pos[:, 0] - position, axis=1)
    if np.max(position_deviations) \
            > GRAIL_OBJECT_POSITION_STATIC_MAX_DEVIATION_M:
        raise ValueError(f"{path}: terrain object root_pos must be static")
    quaternions = root_quat_xyzw[:, 0] / norms[:, None]
    signs = np.where(
        np.sum(quaternions * quaternions[0], axis=1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    aligned = quaternions * signs
    quaternion = np.median(aligned, axis=0)
    quaternion_norm = np.linalg.norm(quaternion)
    if not np.isfinite(quaternion_norm) or quaternion_norm < 1e-12:
        raise ValueError(f"{path}: terrain object root_quat has no robust median")
    quaternion /= quaternion_norm
    canonical_signs = np.where(
        np.sum(quaternions * quaternion, axis=1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    canonical_aligned = quaternions * canonical_signs
    angular_deviations = 4.0 * np.arctan2(
        np.linalg.norm(canonical_aligned - quaternion, axis=1),
        np.linalg.norm(canonical_aligned + quaternion, axis=1),
    )
    if np.max(angular_deviations) \
            > GRAIL_OBJECT_QUATERNION_STATIC_ANGLE_ATOL_RAD:
        raise ValueError(f"{path}: terrain object root_quat must be static")

    position = np.array(position, np.float64, copy=True)
    quaternion_wxyz = np.array(
        quaternion[[3, 0, 1, 2]], np.float64, copy=True)
    object_scale = np.array(scale[:, 0], np.float64, copy=True)
    for array in (position, quaternion_wxyz, object_scale):
        array.setflags(write=False)
    return GrailObjectPose(position, quaternion_wxyz, fps, object_scale)


def _normalized_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("zero-length root quaternion")
    q = q / n
    if not np.isfinite(q).all():
        raise ValueError("non-finite root quaternion")
    return q.astype(np.float32)


def load_takara(path: str, remap_path: str) -> SourceClip:
    data = np.load(path)
    remap = np.load(remap_path)
    joints = np.asarray(data["joint_pos"], np.float32)
    root = np.asarray(data["body_pos_w"][:, 0], np.float32)
    quat = _normalized_wxyz(data["body_quat_w"][:, 0])
    qpos = np.zeros((len(joints), 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = quat
    qpos[:, 7:] = joints[:, remap]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    clip = SourceClip(
        "takara_walk_50hz", fps, qpos, np.arange(len(qpos)), "flat",
    )
    clip.validate()
    return clip


def load_grail(path: str) -> SourceClip:
    records = joblib.load(path)
    if len(records) != 1:
        raise ValueError(f"{path}: expected one robot record, got {len(records)}")
    record = next(iter(records.values()))
    dof = np.asarray(record["dof"], np.float32)
    root = np.asarray(record["root_trans_offset"], np.float32)
    xyzw = np.asarray(record["root_rot"], np.float32)
    quat = _normalized_wxyz(xyzw[:, [3, 0, 1, 2]])
    qpos = np.zeros((len(dof), 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = quat
    qpos[:, 7:] = dof
    fps = float(record["fps"])
    name = os.path.splitext(os.path.basename(path))[0]
    clip = SourceClip(name, fps, qpos, np.arange(len(qpos)), name)
    clip.validate()
    return clip
