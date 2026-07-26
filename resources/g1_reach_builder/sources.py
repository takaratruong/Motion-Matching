from collections.abc import Mapping
import csv
import hashlib
from pathlib import Path
import pickle
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from .schema import GMRSource


_PREFIX = "g1_retargeted_motions/gmr_pkl/"
_EXCLUDED = {"walking", "carry_walking"}

SOMA_G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
SOMA_COLUMNS = (
    "Frame",
    "root_translateX", "root_translateY", "root_translateZ",
    "root_rotateX", "root_rotateY", "root_rotateZ",
    *tuple(f"{name}_dof" for name in SOMA_G1_JOINT_NAMES),
)


def load_soma_csv_directory(
    path: Path,
    fps: float = 100.0,
) -> list[GMRSource]:
    """Load strict SOMA tabletop CSV recordings as validated G1 sources.

    Root XYZ is parsed as centimeters, root Euler as intrinsic ZYX degrees
    stored in XYZ columns, and all 29 G1 joint DOFs as degrees. Sequence IDs
    use the ``tabletop_soma/<stem>`` namespace so they cannot collide with the
    original reach recordings.
    """
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid fps {fps}")
    result: list[GMRSource] = []
    for csv_path in sorted(Path(path).glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            header = tuple(next(reader))
            if header != SOMA_COLUMNS:
                raise ValueError(f"{csv_path.name}: SOMA header mismatch")
            values = np.asarray(
                [[float(value) for value in row] for row in reader],
                dtype=np.float64,
            )
        if values.ndim != 2 or values.shape[1] != len(SOMA_COLUMNS):
            raise ValueError(f"{csv_path.name}: SOMA row width mismatch")
        if not np.isfinite(values).all():
            raise ValueError(f"{csv_path.name}: non-finite value")
        frames = values[:, 0]
        if not np.array_equal(frames, np.arange(len(values))):
            raise ValueError(
                f"{csv_path.name}: source frames must be contiguous from zero"
            )
        root_xyzw = ScipyRotation.from_euler(
            "ZYX", values[:, 4:7][:, [2, 1, 0]], degrees=True
        ).as_quat()
        qpos = np.concatenate((
            values[:, 1:4] / 100.0,
            root_xyzw[:, [3, 0, 1, 2]],
            np.radians(values[:, 7:]),
        ), axis=1)
        source = GMRSource(
            sequence_id=f"tabletop_soma/{csv_path.stem}",
            archive_member=csv_path.name,
            archive_path=csv_path.resolve(),
            archive_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            fps=float(fps),
            fps_overridden=True,
            qpos=qpos,
            source_frames=np.arange(len(qpos), dtype=np.int32),
            disposition="included",
        )
        source.validate()
        result.append(source)
    if not result:
        raise ValueError(f"no SOMA CSV files in {path}")
    return result


def _disposition(sequence_id: str) -> str | None:
    if sequence_id in _EXCLUDED:
        return "excluded"
    if sequence_id == "drawers" or sequence_id.startswith(
        ("pickup_", "left_to_right_", "right_to_left_")
    ):
        return "included"
    return None


def _array(
    record: Mapping[str, object],
    key: str,
    columns: int,
    sequence_id: str,
) -> np.ndarray:
    if key not in record:
        raise ValueError(f"{sequence_id}: missing {key}")
    value = np.asarray(record[key])
    if value.ndim != 2 or value.shape[1] != columns or len(value) == 0:
        raise ValueError(
            f"{sequence_id}: {key} shape must be (T, {columns}), got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{sequence_id}: {key} must be numeric")
    value = value.astype(np.float64, copy=False)
    if not np.isfinite(value).all():
        raise ValueError(f"{sequence_id}: {key} contains non-finite values")
    return value


def _source_from_record(
    record: object,
    sequence_id: str,
    member: str,
    archive_path: Path,
    archive_sha256: str,
    disposition: str,
    fps_override: float | None,
) -> GMRSource:
    if not isinstance(record, Mapping):
        raise ValueError(f"{sequence_id}: GMR pickle must contain a dictionary")
    root_pos = _array(record, "root_pos", 3, sequence_id)
    root_xyzw = _array(record, "root_rot", 4, sequence_id)
    dof_pos = _array(record, "dof_pos", 29, sequence_id)
    if len(root_xyzw) != len(root_pos) or len(dof_pos) != len(root_pos):
        raise ValueError(f"{sequence_id}: GMR frame counts do not match")
    norms = np.linalg.norm(root_xyzw, axis=-1)
    if np.max(np.abs(norms - 1.0)) > 1e-4:
        raise ValueError(f"{sequence_id}: root quaternion norm error")

    embedded_fps = record.get("fps")
    overridden = embedded_fps is None
    fps_value = fps_override if overridden else embedded_fps
    if fps_value is None:
        raise ValueError(f"{sequence_id}: missing fps; provide fps_override")
    try:
        fps = float(fps_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{sequence_id}: invalid fps {fps_value!r}") from error
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{sequence_id}: invalid fps {fps}")

    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate((root_pos, root_wxyz, dof_pos), axis=1)
    source = GMRSource(
        sequence_id=sequence_id,
        archive_member=member,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        fps=fps,
        fps_overridden=overridden,
        qpos=qpos,
        source_frames=np.arange(len(qpos), dtype=np.int32),
        disposition=disposition,
    )
    source.validate()
    return source


def load_gmr_archive(
    path: Path,
    fps_override: float | None = None,
    *,
    include_excluded: bool = False,
) -> list[GMRSource]:
    """Load trusted user-authored GMR pickle members from an archive."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result: list[GMRSource] = []
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith(_PREFIX) and name.endswith(".pkl")
        )
        for member in members:
            sequence_id = Path(member).stem
            disposition = _disposition(sequence_id)
            if disposition is None:
                continue
            if sequence_id in seen:
                raise ValueError(f"duplicate GMR sequence id {sequence_id}")
            seen.add(sequence_id)
            if disposition == "excluded" and not include_excluded:
                continue
            record = pickle.loads(archive.read(member))
            result.append(_source_from_record(
                record,
                sequence_id,
                member,
                path.resolve(),
                digest,
                disposition,
                fps_override,
            ))
    return result

