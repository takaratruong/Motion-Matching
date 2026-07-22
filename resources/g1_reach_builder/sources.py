from collections.abc import Mapping
import hashlib
from pathlib import Path
import pickle
import zipfile

import numpy as np

from .schema import GMRSource


_PREFIX = "g1_retargeted_motions/gmr_pkl/"
_EXCLUDED = {"walking", "carry_walking"}


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

