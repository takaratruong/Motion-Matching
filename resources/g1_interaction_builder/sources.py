from collections import Counter
from pathlib import Path
import re
from typing import Sequence

import joblib
import numpy as np

from .schema import RawInteractionClip, SourcePaths, SourceValidationError


SEQUENCE_RE = re.compile(r"^(pickup_table|pickup_ground)__(.+)__([0-9]{3})$")
_MODALITIES = (
    ("robot", "robot", ".pkl"),
    ("objects", "objects", ".pkl"),
    ("meta", "meta", ".pkl"),
    ("object_usd", "object_usd", ".usd"),
)
_META_KEYS = frozenset(
    ("object_name", "table_pos", "table_quat", "table_size")
)
_GROUND_META_KEYS = frozenset(("object_name",))
GROUND_SUPPORT_POSITION = np.array([0.0, 0.0, -0.02], np.float32)
GROUND_SUPPORT_ROTATION_XYZW = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
GROUND_SUPPORT_SIZE = np.array([20.0, 20.0, 0.04], np.float32)


def sequence_parts(sequence_id: str) -> tuple[str, str]:
    match = SEQUENCE_RE.fullmatch(sequence_id)
    if match is None:
        raise ValueError(f"invalid pickup sequence id: {sequence_id}")
    return match.group(1), match.group(2)


def object_id_from_sequence(sequence_id: str) -> str:
    return sequence_parts(sequence_id)[1]


def _single_record(path: Path) -> dict:
    records = joblib.load(path)
    if not isinstance(records, dict) or len(records) != 1:
        raise ValueError(f"{path}: expected exactly one record")
    return next(iter(records.values()))


def _contacts(record: dict, key: str, frames: int) -> np.ndarray:
    points = record[key]
    return np.array([
        int(len(np.asarray(points.get(i, points.get(str(i), np.empty((0, 3)))))) > 0)
        for i in range(frames)
    ], dtype=np.uint8)


def discover_source_paths(root: Path) -> list[SourcePaths]:
    root = Path(root)
    indexes: dict[str, dict[str, Path]] = {}
    for modality, directory_name, suffix in _MODALITIES:
        directory = root / directory_name
        indexes[modality] = {
            path.stem: path
            for path in directory.glob(f"*{suffix}")
            if path.is_file()
        }

    sequence_ids = sorted(
        set().union(*(set(index) for index in indexes.values()))
    )
    missing_records: list[str] = []
    for sequence_id in sequence_ids:
        missing = [
            modality
            for modality, _, _ in _MODALITIES
            if sequence_id not in indexes[modality]
        ]
        if missing:
            missing_records.append(
                f"{sequence_id}: missing {', '.join(missing)}"
            )
    if missing_records:
        raise ValueError("; ".join(missing_records))

    return [
        SourcePaths(
            sequence_id=sequence_id,
            object_id=object_id_from_sequence(sequence_id),
            robot=indexes["robot"][sequence_id],
            objects=indexes["objects"][sequence_id],
            meta=indexes["meta"][sequence_id],
            object_usd=indexes["object_usd"][sequence_id],
        )
        for sequence_id in sequence_ids
    ]


def discover_source_paths_many(roots: Sequence[Path]) -> list[SourcePaths]:
    discovered = [
        item for root in roots for item in discover_source_paths(root)
    ]
    sequence_ids = [item.sequence_id for item in discovered]
    duplicates = sorted(
        sequence
        for sequence, count in Counter(sequence_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate sequence ids across roots: {duplicates}")
    return sorted(discovered, key=lambda item: item.sequence_id)


def _source_error(
    code: str, sequence_id: str, path: Path, message: str
) -> SourceValidationError:
    return SourceValidationError(code, f"{sequence_id}: {path}: {message}")


def _wrapped_record(
    path: Path, sequence_id: str, modality: str
) -> dict:
    try:
        record = _single_record(path)
    except ValueError as error:
        raise _source_error(
            "invalid_source_record", sequence_id, path, str(error)
        ) from error
    if not isinstance(record, dict):
        raise _source_error(
            "invalid_source_record",
            sequence_id,
            path,
            f"expected {modality} record dictionary",
        )
    return record


def _meta_record(path: Path, sequence_id: str) -> dict:
    records = joblib.load(path)
    category, _ = sequence_parts(sequence_id)
    if category == "pickup_ground":
        if isinstance(records, dict) and set(records) == _GROUND_META_KEYS:
            record = records
        else:
            if isinstance(records, dict) and set(records) & _META_KEYS:
                raise _source_error(
                    "invalid_source_record",
                    sequence_id,
                    path,
                    "ground meta record must contain only object_name",
                )
            record = _wrapped_record(path, sequence_id, "meta")
        if set(record) != _GROUND_META_KEYS:
            raise _source_error(
                "invalid_source_record",
                sequence_id,
                path,
                "ground meta record must contain only object_name",
            )
        return {
            "object_name": record["object_name"],
            "table_pos": GROUND_SUPPORT_POSITION.copy(),
            "table_quat": GROUND_SUPPORT_ROTATION_XYZW.copy(),
            "table_size": GROUND_SUPPORT_SIZE.copy(),
        }
    if isinstance(records, dict) and set(records) == _META_KEYS:
        return records
    if isinstance(records, dict) and set(records) & _META_KEYS:
        raise _source_error(
            "invalid_source_record",
            sequence_id,
            path,
            "partial or ambiguous native meta record",
        )
    record = _wrapped_record(path, sequence_id, "meta")
    if set(record) != _META_KEYS:
        raise _source_error(
            "invalid_source_record",
            sequence_id,
            path,
            "meta record must contain object_name, table_pos, table_quat, and table_size",
        )
    return record


def _array(
    record: dict, key: str, path: Path, sequence_id: str
) -> np.ndarray:
    if key not in record:
        raise _source_error(
            "missing_field", sequence_id, path, f"missing field {key}"
        )
    try:
        return np.asarray(record[key], np.float32)
    except (TypeError, ValueError) as error:
        raise _source_error(
            "invalid_source_record",
            sequence_id,
            path,
            f"field {key} is not a numeric array",
        ) from error


def _source_shape(
    sequence_id: str,
    path: Path,
    name: str,
    value: np.ndarray,
    expected: tuple[int, ...],
    frames: int,
) -> None:
    if value.shape == expected:
        return
    code = (
        "frame_count_mismatch"
        if value.ndim > 0 and value.shape[0] != frames
        else "invalid_shape"
    )
    raise _source_error(
        code,
        sequence_id,
        path,
        f"{name} shape must be {expected}, got {value.shape}",
    )


def _normalized_xyzw(
    value: np.ndarray, sequence_id: str, path: Path, name: str
) -> np.ndarray:
    if value.ndim == 0 or value.shape[-1] != 4:
        raise _source_error(
            "invalid_shape",
            sequence_id,
            path,
            f"{name} quaternion shape must end in 4, got {value.shape}",
        )
    quaternion = np.asarray(value, np.float64)
    norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if (
        not np.isfinite(quaternion).all()
        or not np.isfinite(norms).all()
        or np.any(norms < 1e-8)
    ):
        raise _source_error(
            "invalid_quaternion",
            sequence_id,
            path,
            f"invalid {name} quaternion",
        )
    quaternion = quaternion / norms
    return quaternion[..., [3, 0, 1, 2]].astype(np.float32)


def load_raw_interaction(
    paths: SourcePaths, object_dimensions: np.ndarray
) -> RawInteractionClip:
    sequence_id = paths.sequence_id
    derived_object_id = object_id_from_sequence(sequence_id)
    if paths.object_id != derived_object_id:
        raise _source_error(
            "object_identity_mismatch",
            sequence_id,
            paths.robot,
            f"expected object {derived_object_id}, got {paths.object_id}",
        )

    robot_record = _wrapped_record(paths.robot, sequence_id, "robot")
    object_record = _wrapped_record(paths.objects, sequence_id, "objects")
    meta_record = _meta_record(paths.meta, sequence_id)

    dof = _array(robot_record, "dof", paths.robot, sequence_id)
    if dof.ndim != 2 or dof.shape[1] != 29 or dof.shape[0] == 0:
        raise _source_error(
            "invalid_shape",
            sequence_id,
            paths.robot,
            f"dof shape must be (T, 29), got {dof.shape}",
        )
    frames = dof.shape[0]
    root_positions = _array(
        robot_record, "root_trans_offset", paths.robot, sequence_id
    )
    root_xyzw = _array(robot_record, "root_rot", paths.robot, sequence_id)
    hand_dof = _array(
        robot_record, "hand_dof_pos", paths.robot, sequence_id
    )
    object_positions = _array(
        object_record, "root_pos", paths.objects, sequence_id
    )
    object_xyzw = _array(
        object_record, "root_quat", paths.objects, sequence_id
    )
    for path, name, value, expected in (
        (paths.robot, "root position", root_positions, (frames, 3)),
        (paths.robot, "root rotation", root_xyzw, (frames, 4)),
        (paths.robot, "hand dof", hand_dof, (frames, 14)),
        (
            paths.objects,
            "object position",
            object_positions,
            (frames, 1, 3),
        ),
        (
            paths.objects,
            "object rotation",
            object_xyzw,
            (frames, 1, 4),
        ),
    ):
        _source_shape(sequence_id, path, name, value, expected, frames)

    try:
        robot_fps = float(robot_record["fps"])
        object_fps = float(object_record["fps"])
    except KeyError as error:
        path = paths.robot if "fps" not in robot_record else paths.objects
        raise _source_error(
            "missing_field", sequence_id, path, "missing field fps"
        ) from error
    except (TypeError, ValueError) as error:
        raise _source_error(
            "invalid_fps",
            sequence_id,
            paths.robot,
            "source fps is not numeric",
        ) from error
    if not np.isclose(robot_fps, object_fps, rtol=0.0, atol=1e-6):
        raise _source_error(
            "fps_mismatch",
            sequence_id,
            paths.objects,
            f"robot fps {robot_fps} does not match object fps {object_fps}",
        )

    for key in ("contact_points_left_hand", "contact_points_right_hand"):
        if key not in object_record:
            raise _source_error(
                "missing_field", sequence_id, paths.objects, f"missing field {key}"
            )
        if not isinstance(object_record[key], dict):
            raise _source_error(
                "invalid_contact",
                sequence_id,
                paths.objects,
                f"field {key} must be a dictionary",
            )
    try:
        left_contacts = _contacts(
            object_record, "contact_points_left_hand", frames
        )
        right_contacts = _contacts(
            object_record, "contact_points_right_hand", frames
        )
    except (TypeError, ValueError) as error:
        raise _source_error(
            "invalid_contact",
            sequence_id,
            paths.objects,
            "contact points must be array-like",
        ) from error

    table_position = _array(
        meta_record, "table_pos", paths.meta, sequence_id
    )
    table_xyzw = _array(
        meta_record, "table_quat", paths.meta, sequence_id
    )
    table_size = _array(
        meta_record, "table_size", paths.meta, sequence_id
    )

    qpos = np.zeros((frames, 36), np.float32)
    qpos[:, :3] = root_positions
    qpos[:, 3:7] = _normalized_xyzw(
        root_xyzw, sequence_id, paths.robot, "root"
    )
    qpos[:, 7:] = dof
    clip = RawInteractionClip(
        sequence_id=sequence_id,
        object_id=paths.object_id,
        fps=robot_fps,
        qpos=qpos,
        hand_dof=hand_dof,
        object_positions=object_positions[:, 0, :].copy(),
        object_rotations=_normalized_xyzw(
            object_xyzw[:, 0, :], sequence_id, paths.objects, "object"
        ),
        hand_contacts=np.column_stack(
            (left_contacts, right_contacts)
        ).astype(np.uint8, copy=False),
        table_position=table_position,
        table_rotation=_normalized_xyzw(
            table_xyzw, sequence_id, paths.meta, "table"
        ),
        table_size=table_size,
        object_dimensions=np.asarray(object_dimensions, np.float32).copy(),
        source_frames=np.arange(frames, dtype=np.int32),
    )
    clip.validate()
    return clip
