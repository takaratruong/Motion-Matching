"""Pinned source inventory and strict GRAIL stair-record loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import joblib
import numpy as np


_STAIR_OBJECT = (
    "terrain_stairs__compact_freestanding_ash_hardwood_practice_stair_block_"
    "with_4_visible_steps_straight_single_flight_training_prop_largest_"
    "dimension_under_150_cm_no_handrails_no_balustrad_04d99a9e43"
)
STAIR_BASES: tuple[str, str, str, str] = (
    f"{_STAIR_OBJECT}__0000",
    f"{_STAIR_OBJECT}__0001",
    f"{_STAIR_OBJECT}__0002",
    f"{_STAIR_OBJECT}_updown__0000",
)
SOURCE_FPS = 25.0
SOURCE_FRAMES = 250


@dataclass(frozen=True)
class PinnedStairSource:
    base: str
    robot_path: Path
    object_path: Path
    usd_path: Path
    robot_qpos_mujoco: np.ndarray
    object_position_world: np.ndarray
    object_quaternion_world_xyzw: np.ndarray
    object_scale: np.ndarray
    source_fps: float
    source_sha256: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_single_record(path: Path, base: str, kind: str) -> tuple[str, dict]:
    try:
        value = joblib.load(path)
    except Exception as error:
        raise ValueError(f"{base}: cannot load {kind} record") from error
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"{base}: {kind} must contain exactly one outer record")
    identity, record = next(iter(value.items()))
    if not isinstance(identity, str) or not identity or not isinstance(record, dict):
        raise ValueError(f"{base}: {kind} outer record is invalid")
    return identity, record


def _finite_array(
    record: dict,
    field: str,
    shape: tuple[int, ...],
    base: str,
    kind: str,
) -> np.ndarray:
    if field not in record:
        raise ValueError(f"{base}: {kind} field {field} is missing")
    array = np.asarray(record[field])
    if array.shape != shape:
        raise ValueError(
            f"{base}: {kind} field {field} shape {array.shape} != {shape}"
        )
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{base}: {kind} field {field} must be finite numeric")
    return np.asarray(array, np.float64)


def _fps(record: dict, base: str, kind: str) -> float:
    if "fps" not in record:
        raise ValueError(f"{base}: {kind} field fps is missing")
    try:
        value = float(record["fps"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{base}: {kind} fps is invalid") from error
    if not np.isfinite(value) or value != SOURCE_FPS:
        raise ValueError(
            f"{base}: {kind} fps must equal 25, got {value}"
        )
    return value


def _normalized_unrolled_wxyz(xyzw: np.ndarray, base: str) -> np.ndarray:
    wxyz = np.asarray(xyzw[:, (3, 0, 1, 2)], np.float64).copy()
    norms = np.linalg.norm(wxyz, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError(f"{base}: robot root quaternion is invalid")
    wxyz /= norms
    for frame in range(1, len(wxyz)):
        if float(np.dot(wxyz[frame - 1], wxyz[frame])) < 0.0:
            wxyz[frame] *= -1.0
    return wxyz


def _constant_row(array: np.ndarray, field: str, base: str) -> np.ndarray:
    flat = array.reshape(SOURCE_FRAMES, -1)
    if not np.allclose(flat, flat[:1], rtol=0.0, atol=1e-7):
        raise ValueError(f"{base}: object field {field} must be constant")
    return flat[0].copy()


def _owned_readonly(value: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(value, dtype=np.float32)
    output.setflags(write=False)
    return output


def _load_source(root: Path, base: str) -> PinnedStairSource:
    robot_path = (root / "robot" / f"{base}.pkl").resolve()
    object_path = (root / "objects" / f"{base}.pkl").resolve()
    usd_path = (root / "object_usd" / f"{base}.usd").resolve()
    for path, label in (
        (robot_path, "robot"),
        (object_path, "object"),
        (usd_path, "usd"),
    ):
        if not path.is_file():
            raise ValueError(f"{base}: {label} source is missing: {path}")

    robot_identity, robot = _load_single_record(robot_path, base, "robot")
    object_identity, object_record = _load_single_record(
        object_path, base, "object"
    )
    if robot_identity != object_identity:
        raise ValueError(f"{base}: robot/object outer record identity differs")
    robot_fps = _fps(robot, base, "robot")
    object_fps = _fps(object_record, base, "object")
    if robot_fps != object_fps:
        raise ValueError(f"{base}: robot/object fps differs")

    root_position = _finite_array(
        robot, "root_trans_offset", (SOURCE_FRAMES, 3), base, "robot"
    )
    root_xyzw = _finite_array(
        robot, "root_rot", (SOURCE_FRAMES, 4), base, "robot"
    )
    joints = _finite_array(
        robot, "dof", (SOURCE_FRAMES, 29), base, "robot"
    )
    root_wxyz = _normalized_unrolled_wxyz(root_xyzw, base)
    qpos = np.concatenate((root_position, root_wxyz, joints), axis=1)

    object_position = _constant_row(
        _finite_array(
            object_record,
            "root_pos",
            (SOURCE_FRAMES, 1, 3),
            base,
            "object",
        ),
        "root_pos",
        base,
    )
    object_quaternion = _constant_row(
        _finite_array(
            object_record,
            "root_quat",
            (SOURCE_FRAMES, 1, 4),
            base,
            "object",
        ),
        "root_quat",
        base,
    )
    object_quaternion_norm = float(np.linalg.norm(object_quaternion))
    if abs(object_quaternion_norm - 1.0) > 1e-4:
        raise ValueError(f"{base}: object root_quat is not unit length")
    object_quaternion /= object_quaternion_norm

    if "scale" not in object_record:
        raise ValueError(f"{base}: object field scale is missing")
    scale = np.asarray(object_record["scale"])
    if scale.shape not in ((3,), (3, 1), (1, 3)):
        raise ValueError(f"{base}: object field scale shape {scale.shape} is invalid")
    scale = np.asarray(scale, np.float64).reshape(3)
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"{base}: object scale must be finite and positive")

    return PinnedStairSource(
        base=base,
        robot_path=robot_path,
        object_path=object_path,
        usd_path=usd_path,
        robot_qpos_mujoco=_owned_readonly(qpos),
        object_position_world=_owned_readonly(object_position),
        object_quaternion_world_xyzw=_owned_readonly(object_quaternion),
        object_scale=_owned_readonly(scale),
        source_fps=robot_fps,
        source_sha256=MappingProxyType(
            {
                "robot": _sha256(robot_path),
                "object": _sha256(object_path),
                "usd": _sha256(usd_path),
            }
        ),
    )


def load_pinned_sources(root: str | Path) -> tuple[PinnedStairSource, ...]:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise ValueError(f"GRAIL stair root is not a directory: {resolved}")
    return tuple(_load_source(resolved, base) for base in STAIR_BASES)

