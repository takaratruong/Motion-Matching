"""Build the compact released-PFNN train/validation corpus for native G1."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np

from mm_sonic.grail_terrain_source import G1MujocoFK, mujoco_to_isaaclab_joints
from mm_sonic.pfnn_terrain_fit import (
    PFNNTerrainFit,
    fit_terrain_cycle,
    released_pfnn_contacts_for_interval,
    save_terrain_fit,
)
from mm_sonic.retarget_pfnn_bvh_g1 import G1_JOINT_NAMES
from mm_sonic.terrain_oracle.canonical import ISAACLAB_JOINT_NAMES
from mm_sonic.terrain_pfnn.features import (
    PFNNTrainingWindow,
    build_clip_windows_with_audit,
    mirror_window,
    smoothed_body_facing_yaw_world,
)
from mm_sonic.terrain_pfnn.dataset import (
    mirror_classic_g1_joint_state,
    pack_classic_g1_input_v3,
)
from mm_sonic.terrain_pfnn.layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    CONTACT_ORDER,
    INPUT_LAYOUT,
    OUTPUT_LAYOUT,
    TRAJECTORY_TIMES_S,
)
from mm_sonic.terrain_pfnn.provenance import (
    MAXIMUM_JOINT_STEP_RAD,
    canonical_joint_state_receipt,
    validate_joint_state_receipt,
)
from mm_sonic.terrain_pfnn.pfnn_surface import (
    PFNN_G1_Z_OFFSET_M,
    PlacedPFNNSurface,
)
from mm_sonic.terrain_pfnn.phase import ContactPhaseTrack, released_pfnn_phase_track
from mm_sonic.terrain_pfnn.sources import (
    PFNNSourceClip,
    _backward_finite_difference,
    load_pfnn_retarget_source,
)


VERTICAL_DATASET_SCHEMA = "g1-pfnn-vertical-dataset/v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = ("train", "validation")
_SPLIT_FIELDS = {
    "x",
    "y",
    "phase",
    "clip_id",
    "sequence_lane",
    "center_frame_120hz",
    "root_world_xy",
    "root_world_yaw",
    "terrain_class",
    "terrain_sha256",
    "mirrored",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_string(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class VerticalSurfaceSegment:
    cycle_start_frame_120hz: int
    cycle_stop_frame_120hz: int
    terrain_sha256: str
    height_at: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self) -> None:
        if (
            type(self.cycle_start_frame_120hz) is not int
            or type(self.cycle_stop_frame_120hz) is not int
            or self.cycle_start_frame_120hz < 0
            or self.cycle_stop_frame_120hz <= self.cycle_start_frame_120hz
        ):
            raise ValueError("PFNN terrain cycle bounds are invalid")
        _digest_string(self.terrain_sha256, "terrain_sha256")
        if not callable(self.height_at):
            raise TypeError("PFNN surface height_at must be callable")


@dataclass(frozen=True)
class VerticalSliceSource:
    role: Literal["train", "validation"]
    stem: str
    source_start_frame_120hz: int
    clip: PFNNSourceClip
    phase_track: ContactPhaseTrack
    segments: tuple[VerticalSurfaceSegment, ...]
    selection_sha256: str
    retarget_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.role not in _SPLITS:
            raise ValueError("released PFNN source role is invalid")
        if type(self.stem) is not str or not self.stem:
            raise ValueError("released PFNN source stem is invalid")
        if type(self.source_start_frame_120hz) is not int or self.source_start_frame_120hz < 0:
            raise ValueError("released PFNN source start frame is invalid")
        if not isinstance(self.clip, PFNNSourceClip):
            raise TypeError("clip must be a PFNNSourceClip")
        if not isinstance(self.phase_track, ContactPhaseTrack):
            raise TypeError("phase_track must be a ContactPhaseTrack")
        if len(self.phase_track.phase) != self.clip.frame_count:
            raise ValueError("released PFNN phase track length mismatch")
        if not self.segments or not all(
            isinstance(value, VerticalSurfaceSegment) for value in self.segments
        ):
            raise ValueError("released PFNN source must contain terrain segments")
        ordered = tuple(
            sorted(self.segments, key=lambda value: value.cycle_start_frame_120hz)
        )
        for previous, current in zip(ordered, ordered[1:]):
            if previous.cycle_stop_frame_120hz > current.cycle_start_frame_120hz:
                raise ValueError("released PFNN terrain segments overlap")
        object.__setattr__(self, "segments", ordered)
        _digest_string(self.selection_sha256, "selection_sha256")
        _digest_string(self.retarget_manifest_sha256, "retarget_manifest_sha256")


@dataclass(frozen=True)
class VerticalSplitArrays:
    x: np.ndarray
    y: np.ndarray
    phase: np.ndarray
    clip_id: np.ndarray
    sequence_lane: np.ndarray
    center_frame_120hz: np.ndarray
    root_world_xy: np.ndarray
    root_world_yaw: np.ndarray
    terrain_class: np.ndarray
    terrain_sha256: np.ndarray
    mirrored: np.ndarray

    def __post_init__(self) -> None:
        count = len(np.asarray(self.phase))
        expected = {
            "x": ((count, CLASSIC_G1_INPUT_LAYOUT_V3.size), np.dtype(np.float32)),
            "y": ((count, OUTPUT_LAYOUT.size), np.dtype(np.float32)),
            "phase": ((count,), np.dtype(np.float32)),
            "clip_id": ((count,), np.dtype("<U128")),
            "sequence_lane": ((count,), np.dtype("<U16")),
            "center_frame_120hz": ((count,), np.dtype(np.int64)),
            "root_world_xy": ((count, 2), np.dtype(np.float32)),
            "root_world_yaw": ((count,), np.dtype(np.float32)),
            "terrain_class": ((count,), np.dtype("<U10")),
            "terrain_sha256": ((count,), np.dtype("<U64")),
            "mirrored": ((count,), np.dtype(np.bool_)),
        }
        for name, (shape, dtype) in expected.items():
            value = np.ascontiguousarray(np.asarray(getattr(self, name)), dtype=dtype)
            if value.shape != shape:
                raise ValueError(f"vertical split {name} has invalid shape")
            if value.dtype.kind in "f" and not np.isfinite(value).all():
                raise ValueError(f"vertical split {name} must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if count < 1:
            raise ValueError("vertical split must contain at least one row")
        if not np.isin(
            self.sequence_lane,
            ("motion", *(f"idle_phase_{index}" for index in range(8))),
        ).all():
            raise ValueError("vertical split sequence lane is invalid")
        if not np.isin(self.terrain_class, ("flat", "ascent", "descent", "transition")).all():
            raise ValueError("vertical split terrain_class is invalid")
        if any(_SHA256_RE.fullmatch(str(value)) is None for value in self.terrain_sha256):
            raise ValueError("vertical split terrain digest is invalid")


@dataclass(frozen=True)
class VerticalDataset:
    splits: Mapping[str, VerticalSplitArrays]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    selection_sha256: str
    retarget_manifest_sha256: str
    terrain_receipt_set_sha256: str
    source_roles: Mapping[str, str]
    joint_state_receipt: Mapping[str, object]
    dataset_sha256: str

    def __post_init__(self) -> None:
        if set(self.splits) != set(_SPLITS):
            raise ValueError("vertical dataset splits are invalid")
        checked_splits = {
            name: value
            for name, value in self.splits.items()
            if isinstance(value, VerticalSplitArrays)
        }
        if len(checked_splits) != len(_SPLITS):
            raise TypeError("vertical dataset split value is invalid")
        object.__setattr__(self, "splits", MappingProxyType(checked_splits))
        for name, width in (
            ("x_mean", CLASSIC_G1_INPUT_LAYOUT_V3.size),
            ("x_std", CLASSIC_G1_INPUT_LAYOUT_V3.size),
            ("y_mean", OUTPUT_LAYOUT.size),
            ("y_std", OUTPUT_LAYOUT.size),
        ):
            value = np.ascontiguousarray(np.asarray(getattr(self, name)), dtype=np.float32)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f"vertical normalization {name} is invalid")
            if name.endswith("_std") and np.any(value <= 0.0):
                raise ValueError(f"vertical normalization {name} must be positive")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in (
            "selection_sha256",
            "retarget_manifest_sha256",
            "terrain_receipt_set_sha256",
            "dataset_sha256",
        ):
            _digest_string(getattr(self, name), name)
        roles = dict(self.source_roles)
        if not roles or any(
            type(stem) is not str or role not in _SPLITS for stem, role in roles.items()
        ):
            raise ValueError("vertical source roles are invalid")
        object.__setattr__(self, "source_roles", MappingProxyType(dict(sorted(roles.items()))))
        receipt = validate_joint_state_receipt(self.joint_state_receipt)
        object.__setattr__(self, "joint_state_receipt", MappingProxyType(receipt))


def _array_digest_update(digest: object, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.dtype.kind in "iufb":
        dtype = array.dtype.newbyteorder("<")
        array = np.ascontiguousarray(array, dtype=dtype)
    else:
        array = np.ascontiguousarray(array)
    header = json.dumps(
        {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))


def _terrain_receipt_set_sha256(sources: tuple[VerticalSliceSource, ...]) -> str:
    payload = sorted(
        {
            (source.stem, segment.cycle_start_frame_120hz, segment.cycle_stop_frame_120hz, segment.terrain_sha256)
            for source in sources
            for segment in source.segments
        }
    )
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dataset_digest(
    splits: Mapping[str, VerticalSplitArrays],
    normalization: Mapping[str, np.ndarray],
    *,
    selection_sha256: str,
    retarget_manifest_sha256: str,
    terrain_receipt_set_sha256: str,
    source_roles: Mapping[str, str],
    joint_state_receipt: Mapping[str, object],
) -> str:
    digest = hashlib.sha256(b"g1-pfnn-vertical-dataset/v3\0")
    provenance = json.dumps(
        {
            "selection_sha256": selection_sha256,
            "retarget_manifest_sha256": retarget_manifest_sha256,
            "terrain_receipt_set_sha256": terrain_receipt_set_sha256,
            "source_roles": dict(sorted(source_roles.items())),
            "joint_state_receipt": dict(joint_state_receipt),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(provenance)
    for split in _SPLITS:
        values = splits[split]
        for name in sorted(_SPLIT_FIELDS):
            _array_digest_update(digest, f"{split}/{name}", getattr(values, name))
    for name in sorted(normalization):
        _array_digest_update(digest, f"normalization/{name}", normalization[name])
    return digest.hexdigest()


@dataclass(frozen=True)
class _VerticalRow:
    window: PFNNTrainingWindow
    x: np.ndarray
    center_frame_120hz: int
    terrain_sha256: str
    mirrored: bool
    root_world_xy: np.ndarray
    root_world_yaw: float


def _split_arrays(rows: list[_VerticalRow]) -> VerticalSplitArrays:
    return VerticalSplitArrays(
        x=np.stack([row.x for row in rows]).astype(np.float32),
        y=np.stack([row.window.y for row in rows]).astype(np.float32),
        phase=np.asarray(
            [row.window.phase for row in rows], dtype=np.float32
        ),
        clip_id=np.asarray(
            [row.window.clip_id for row in rows], dtype="<U128"
        ),
        sequence_lane=np.asarray(
            [row.window.sequence_lane for row in rows], dtype="<U16"
        ),
        center_frame_120hz=np.asarray(
            [row.center_frame_120hz for row in rows], dtype=np.int64
        ),
        root_world_xy=np.stack([row.root_world_xy for row in rows]).astype(np.float32),
        root_world_yaw=np.asarray(
            [row.root_world_yaw for row in rows], dtype=np.float32
        ),
        terrain_class=np.asarray(
            [row.window.terrain_class for row in rows], dtype="<U10"
        ),
        terrain_sha256=np.asarray(
            [row.terrain_sha256 for row in rows], dtype="<U64"
        ),
        mirrored=np.asarray(
            [row.mirrored for row in rows], dtype=np.bool_
        ),
    )


def build_vertical_dataset(sources: tuple[VerticalSliceSource, ...]) -> VerticalDataset:
    """Build deterministic cycle-local rows and train-only normalization."""

    if not sources or not all(isinstance(source, VerticalSliceSource) for source in sources):
        raise TypeError("sources must contain VerticalSliceSource values")
    ordered = tuple(sorted(sources, key=lambda value: (value.role, value.stem)))
    identities: dict[str, str] = {}
    for source in ordered:
        previous = identities.setdefault(source.stem, source.role)
        if previous != source.role:
            raise ValueError("train/validation source overlap")
    if len(identities) != len(ordered):
        raise ValueError("duplicate PFNN source")
    selection = {source.selection_sha256 for source in ordered}
    retarget = {source.retarget_manifest_sha256 for source in ordered}
    if len(selection) != 1 or len(retarget) != 1:
        raise ValueError("vertical source provenance mismatch")

    rows: dict[str, list[_VerticalRow]] = {name: [] for name in _SPLITS}
    keys: set[tuple[str, str, int, str, bool]] = set()
    accepted_rows: Counter[str] = Counter({source.stem: 0 for source in ordered})
    rejected_rows: defaultdict[str, Counter[str]] = defaultdict(Counter)
    rejected_joints: Counter[str] = Counter(
        {name: 0 for name in ISAACLAB_JOINT_NAMES}
    )
    state_sources = {
        source.stem: source.clip.joint_velocity_source for source in ordered
    }
    for source in ordered:
        facing_yaw_world = smoothed_body_facing_yaw_world(source.clip)
        if source.clip.joint_velocity_source == "unique_predecessor":
            expected_velocity = _backward_finite_difference(
                source.clip.joint_position, source.clip.fps
            )
            if not np.array_equal(
                source.clip.joint_velocity[1:], expected_velocity[1:]
            ):
                raise ValueError("predecessor-derived joint velocity is invalid")
        for segment in source.segments:
            result = build_clip_windows_with_audit(
                source.clip,
                source.phase_track,
                height_at=segment.height_at,
            )
            for window in result.windows:
                center = source.source_start_frame_120hz + 4 * window.center_frame
                if not (
                    segment.cycle_start_frame_120hz
                    <= center
                    < segment.cycle_stop_frame_120hz
                ):
                    continue
                if (
                    source.clip.joint_velocity_source == "unique_predecessor"
                    and window.center_frame < 1
                ):
                    rejected_rows[source.stem]["missing_unique_predecessor"] += 1
                    continue
                root_xy = np.asarray(
                    source.clip.root_position_world[window.center_frame, :2],
                    dtype=np.float64,
                )
                root_yaw = float(facing_yaw_world[window.center_frame])
                joint_position = source.clip.joint_position[window.center_frame]
                joint_velocity = source.clip.joint_velocity[window.center_frame]
                transition = np.abs(
                    window.y[OUTPUT_LAYOUT["joint_position"]] - joint_position
                )
                unsafe = np.flatnonzero(transition > MAXIMUM_JOINT_STEP_RAD)
                if len(unsafe):
                    rejected_rows[source.stem]["joint_step_exceeds_limit"] += 1
                    for joint_index in unsafe:
                        rejected_joints[ISAACLAB_JOINT_NAMES[int(joint_index)]] += 1
                    continue
                values = [
                    (
                        window,
                        pack_classic_g1_input_v3(
                            window.x, joint_position, joint_velocity
                        ),
                        False,
                    )
                ]
                if source.role == "train":
                    mirrored_window = mirror_window(window)
                    values.append(
                        (
                            mirrored_window,
                            pack_classic_g1_input_v3(
                                mirrored_window.x,
                                mirror_classic_g1_joint_state(joint_position),
                                mirror_classic_g1_joint_state(joint_velocity),
                            ),
                            True,
                        )
                    )
                for value, x_v3, mirrored in values:
                    key = (
                        source.role,
                        source.stem,
                        center,
                        value.sequence_lane,
                        mirrored,
                    )
                    if key in keys:
                        raise ValueError("duplicate PFNN window")
                    keys.add(key)
                    rows[source.role].append(
                        _VerticalRow(
                            window=value,
                            x=x_v3,
                            center_frame_120hz=center,
                            terrain_sha256=segment.terrain_sha256,
                            mirrored=mirrored,
                            root_world_xy=root_xy,
                            root_world_yaw=root_yaw,
                        )
                    )
                    accepted_rows[source.stem] += 1
    split_arrays = {name: _split_arrays(rows[name]) for name in _SPLITS}
    train = split_arrays["train"]
    x_mean = np.mean(train.x, axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.std(train.x, axis=0, dtype=np.float64).astype(np.float32)
    y_mean = np.mean(train.y, axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.std(train.y, axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < np.float32(1.0e-6)] = np.float32(1.0)
    y_std[y_std < np.float32(1.0e-6)] = np.float32(1.0)
    contact = OUTPUT_LAYOUT["contact_logit"]
    y_mean[contact] = np.float32(0.0)
    y_std[contact] = np.float32(1.0)
    normalization = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    terrain_receipt = _terrain_receipt_set_sha256(ordered)
    source_roles = dict(sorted(identities.items()))
    joint_state_receipt = canonical_joint_state_receipt(
        accepted_rows_by_source=dict(accepted_rows),
        rejected_rows_by_source={
            source: dict(counts) for source, counts in rejected_rows.items()
        },
        rejected_rows_by_joint=dict(rejected_joints),
        state_source_by_clip=state_sources,
    )
    dataset_sha256 = _dataset_digest(
        split_arrays,
        normalization,
        selection_sha256=next(iter(selection)),
        retarget_manifest_sha256=next(iter(retarget)),
        terrain_receipt_set_sha256=terrain_receipt,
        source_roles=source_roles,
        joint_state_receipt=joint_state_receipt,
    )
    return VerticalDataset(
        splits=split_arrays,
        **normalization,
        selection_sha256=next(iter(selection)),
        retarget_manifest_sha256=next(iter(retarget)),
        terrain_receipt_set_sha256=terrain_receipt,
        source_roles=source_roles,
        joint_state_receipt=joint_state_receipt,
        dataset_sha256=dataset_sha256,
    )


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_vertical_dataset(path: Path, dataset: VerticalDataset) -> Path:
    """Atomically write safe NPZ splits, normalization, and a bound manifest."""

    if not isinstance(dataset, VerticalDataset):
        raise TypeError("dataset must be a VerticalDataset")
    root = Path(path).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite vertical dataset: {root}")
    root.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for split in _SPLITS:
        relative = f"{split}.npz"
        destination = root / relative
        values = dataset.splits[split]
        _atomic_npz(destination, {name: getattr(values, name) for name in _SPLIT_FIELDS})
        records[split] = {
            "path": relative,
            "sha256": _sha256(destination),
            "count": len(values.phase),
        }
    normal_path = root / "normalization.npz"
    _atomic_npz(
        normal_path,
        {
            "x_mean": dataset.x_mean,
            "x_std": dataset.x_std,
            "y_mean": dataset.y_mean,
            "y_std": dataset.y_std,
        },
    )
    manifest = {
        "schema": VERTICAL_DATASET_SCHEMA,
        "status": "accepted",
        "fps": 30.0,
        "input_size": CLASSIC_G1_INPUT_LAYOUT_V3.size,
        "input_layout": [list(field) for field in CLASSIC_G1_INPUT_LAYOUT_V3.fields],
        "output_size": OUTPUT_LAYOUT.size,
        "joint_order": list(ISAACLAB_JOINT_NAMES),
        "trajectory_times_s": TRAJECTORY_TIMES_S.tolist(),
        "contact_order": list(CONTACT_ORDER),
        "selection_sha256": dataset.selection_sha256,
        "retarget_manifest_sha256": dataset.retarget_manifest_sha256,
        "terrain_receipt_set_sha256": dataset.terrain_receipt_set_sha256,
        "source_roles": dict(dataset.source_roles),
        "joint_state_receipt": dict(dataset.joint_state_receipt),
        "splits": records,
        "normalization": {
            "path": "normalization.npz",
            "sha256": _sha256(normal_path),
            "training_sample_count": len(dataset.splits["train"].phase),
        },
        "dataset_sha256": dataset.dataset_sha256,
    }
    _atomic_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def _artifact_path(root: Path, value: object) -> Path:
    if type(value) is not str or not value or Path(value).is_absolute():
        raise ValueError("vertical dataset artifact path is invalid")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("vertical dataset artifact path escapes root")
    return path


def load_vertical_dataset(path: Path) -> VerticalDataset:
    """Load and fully rehash a compact released-PFNN corpus."""

    root = Path(path).expanduser().resolve(strict=True)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("vertical dataset manifest is invalid") from error
    required = {
        "schema",
        "status",
        "fps",
        "input_size",
        "input_layout",
        "output_size",
        "joint_order",
        "trajectory_times_s",
        "contact_order",
        "selection_sha256",
        "retarget_manifest_sha256",
        "terrain_receipt_set_sha256",
        "source_roles",
        "joint_state_receipt",
        "splits",
        "normalization",
        "dataset_sha256",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != required
        or manifest["schema"] != VERTICAL_DATASET_SCHEMA
        or manifest["status"] != "accepted"
        or manifest["fps"] != 30.0
        or manifest["input_size"] != CLASSIC_G1_INPUT_LAYOUT_V3.size
        or manifest["input_layout"]
        != [list(field) for field in CLASSIC_G1_INPUT_LAYOUT_V3.fields]
        or manifest["output_size"] != OUTPUT_LAYOUT.size
        or manifest["joint_order"] != list(ISAACLAB_JOINT_NAMES)
        or manifest["trajectory_times_s"] != TRAJECTORY_TIMES_S.tolist()
        or manifest["contact_order"] != list(CONTACT_ORDER)
        or not isinstance(manifest["splits"], dict)
        or set(manifest["splits"]) != set(_SPLITS)
    ):
        raise ValueError("vertical dataset manifest contract is invalid")
    split_arrays: dict[str, VerticalSplitArrays] = {}
    for split in _SPLITS:
        record = manifest["splits"][split]
        if type(record) is not dict or set(record) != {"path", "sha256", "count"}:
            raise ValueError("vertical dataset split record is invalid")
        source = _artifact_path(root, record["path"])
        if _sha256(source) != record["sha256"]:
            raise ValueError("vertical dataset split digest mismatch")
        try:
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != _SPLIT_FIELDS:
                    raise ValueError("vertical split fields are invalid")
                values = {name: np.asarray(archive[name]).copy() for name in archive.files}
        except (OSError, ValueError, KeyError) as error:
            raise ValueError("vertical dataset split is invalid") from error
        split_arrays[split] = VerticalSplitArrays(**values)
        if len(split_arrays[split].phase) != record["count"]:
            raise ValueError("vertical dataset split count mismatch")
    normal = manifest["normalization"]
    if type(normal) is not dict or set(normal) != {"path", "sha256", "training_sample_count"}:
        raise ValueError("vertical normalization record is invalid")
    normal_path = _artifact_path(root, normal["path"])
    if _sha256(normal_path) != normal["sha256"]:
        raise ValueError("vertical normalization digest mismatch")
    try:
        with np.load(normal_path, allow_pickle=False) as archive:
            if set(archive.files) != {"x_mean", "x_std", "y_mean", "y_std"}:
                raise ValueError("vertical normalization fields are invalid")
            normalization = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("vertical normalization is invalid") from error
    if normal["training_sample_count"] != len(split_arrays["train"].phase):
        raise ValueError("vertical normalization training count mismatch")
    rebuilt_digest = _dataset_digest(
        split_arrays,
        normalization,
        selection_sha256=manifest["selection_sha256"],
        retarget_manifest_sha256=manifest["retarget_manifest_sha256"],
        terrain_receipt_set_sha256=manifest["terrain_receipt_set_sha256"],
        source_roles=manifest["source_roles"],
        joint_state_receipt=manifest["joint_state_receipt"],
    )
    if rebuilt_digest != manifest["dataset_sha256"]:
        raise ValueError("vertical dataset digest mismatch")
    return VerticalDataset(
        splits=split_arrays,
        **normalization,
        selection_sha256=manifest["selection_sha256"],
        retarget_manifest_sha256=manifest["retarget_manifest_sha256"],
        terrain_receipt_set_sha256=manifest["terrain_receipt_set_sha256"],
        source_roles=manifest["source_roles"],
        joint_state_receipt=manifest["joint_state_receipt"],
        dataset_sha256=rebuilt_digest,
    )


def _load_authenticated_vertical_v2(
    path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Read the sealed compact v2 corpus solely as a migration source."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest = _read_json(root / "manifest.json", "vertical v2 manifest")
    required = {
        "schema", "status", "fps", "input_size", "output_size", "joint_order",
        "trajectory_times_s", "contact_order", "selection_sha256",
        "retarget_manifest_sha256", "terrain_receipt_set_sha256", "source_roles",
        "splits", "normalization", "dataset_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema") != "g1-pfnn-vertical-dataset/v2"
        or manifest.get("status") != "accepted"
        or manifest.get("fps") != 30.0
        or manifest.get("input_size") != INPUT_LAYOUT.size
        or manifest.get("output_size") != OUTPUT_LAYOUT.size
        or manifest.get("joint_order") != list(ISAACLAB_JOINT_NAMES)
        or manifest.get("trajectory_times_s") != TRAJECTORY_TIMES_S.tolist()
        or manifest.get("contact_order") != list(CONTACT_ORDER)
        or not isinstance(manifest.get("splits"), dict)
        or set(manifest["splits"]) != set(_SPLITS)
    ):
        raise ValueError("vertical v2 migration manifest contract is invalid")
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in _SPLITS:
        record = manifest["splits"][split]
        if type(record) is not dict or set(record) != {"path", "sha256", "count"}:
            raise ValueError("vertical v2 migration split record is invalid")
        source = _artifact_path(root, record["path"])
        if _sha256(source) != record["sha256"]:
            raise ValueError("vertical v2 migration split digest mismatch")
        try:
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != _SPLIT_FIELDS:
                    raise ValueError("vertical v2 migration split fields are invalid")
                arrays = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
        except (OSError, ValueError, KeyError) as error:
            raise ValueError("vertical v2 migration split is invalid") from error
        count = record["count"]
        expected = {
            "x": ((count, INPUT_LAYOUT.size), np.dtype(np.float32)),
            "y": ((count, OUTPUT_LAYOUT.size), np.dtype(np.float32)),
            "phase": ((count,), np.dtype(np.float32)),
            "clip_id": ((count,), np.dtype("<U128")),
            "sequence_lane": ((count,), np.dtype("<U16")),
            "center_frame_120hz": ((count,), np.dtype(np.int64)),
            "root_world_xy": ((count, 2), np.dtype(np.float32)),
            "root_world_yaw": ((count,), np.dtype(np.float32)),
            "terrain_class": ((count,), np.dtype("<U10")),
            "terrain_sha256": ((count,), np.dtype("<U64")),
            "mirrored": ((count,), np.dtype(np.bool_)),
        }
        if type(count) is not int or count < 1 or any(
            arrays[name].shape != shape or arrays[name].dtype != dtype
            for name, (shape, dtype) in expected.items()
        ):
            raise ValueError("vertical v2 migration split shape is invalid")
        if not all(
            np.isfinite(arrays[name]).all()
            for name in ("x", "y", "phase", "root_world_xy", "root_world_yaw")
        ):
            raise ValueError("vertical v2 migration split contains nonfinite values")
        split_arrays[split] = arrays
    normal = manifest["normalization"]
    if type(normal) is not dict or set(normal) != {
        "path", "sha256", "training_sample_count"
    }:
        raise ValueError("vertical v2 migration normalization record is invalid")
    normal_path = _artifact_path(root, normal["path"])
    if _sha256(normal_path) != normal["sha256"]:
        raise ValueError("vertical v2 migration normalization digest mismatch")
    try:
        with np.load(normal_path, allow_pickle=False) as archive:
            if set(archive.files) != {"x_mean", "x_std", "y_mean", "y_std"}:
                raise ValueError("vertical v2 migration normalization fields are invalid")
            normalization = {
                name: np.asarray(archive[name]).copy() for name in archive.files
            }
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("vertical v2 migration normalization is invalid") from error
    for name, width in (
        ("x_mean", INPUT_LAYOUT.size), ("x_std", INPUT_LAYOUT.size),
        ("y_mean", OUTPUT_LAYOUT.size), ("y_std", OUTPUT_LAYOUT.size),
    ):
        if normalization[name].shape != (width,) or normalization[name].dtype != np.float32:
            raise ValueError("vertical v2 migration normalization shape is invalid")
    digest = hashlib.sha256(b"g1-pfnn-vertical-dataset/v2\0")
    digest.update(json.dumps(
        {
            "selection_sha256": manifest["selection_sha256"],
            "retarget_manifest_sha256": manifest["retarget_manifest_sha256"],
            "terrain_receipt_set_sha256": manifest["terrain_receipt_set_sha256"],
            "source_roles": dict(sorted(manifest["source_roles"].items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    for split in _SPLITS:
        for name in sorted(_SPLIT_FIELDS):
            _array_digest_update(digest, f"{split}/{name}", split_arrays[split][name])
    for name in sorted(normalization):
        _array_digest_update(digest, f"normalization/{name}", normalization[name])
    if digest.hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("vertical v2 migration dataset digest mismatch")
    return manifest, split_arrays, normalization


def _authenticated_retarget_joint_state(
    retarget_root: Path,
    expected_manifest_sha256: str,
) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    root = Path(retarget_root).expanduser().resolve(strict=True)
    manifest_path = root / "retarget-manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("vertical migration retarget manifest digest mismatch")
    manifest = _read_json(manifest_path, "vertical migration retarget manifest")
    if (
        set(manifest) != {"schema", "status", "selection_sha256", "items"}
        or manifest.get("schema") != "g1-pfnn-vertical-slice-retarget/v1"
        or manifest.get("status") != "accepted"
        or not isinstance(manifest.get("items"), list)
    ):
        raise ValueError("vertical migration retarget manifest is invalid")
    output: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    required = {
        "stem", "role", "start_frame_120hz", "stop_frame_120hz", "coverage",
        "output", "output_sha256", "receipt", "receipt_sha256",
    }
    for item in manifest["items"]:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("vertical migration retarget item is invalid")
        stem = item["stem"]
        start = item["start_frame_120hz"]
        stop = item["stop_frame_120hz"]
        if (
            type(stem) is not str or not stem or stem in output
            or item["role"] not in _SPLITS
            or type(start) is not int or type(stop) is not int or stop <= start
        ):
            raise ValueError("vertical migration retarget identity is invalid")
        motion = _relative_artifact(root, item["output"])
        receipt = _relative_artifact(root, item["receipt"])
        if _sha256(motion) != item["output_sha256"] or _sha256(receipt) != item["receipt_sha256"]:
            raise ValueError("vertical migration retarget artifact digest mismatch")
        try:
            with np.load(motion, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "root_pos", "root_quat", "dof", "fps", "engine",
                    "joint_names", "joint_limits",
                }:
                    raise ValueError("retarget fields are invalid")
                dof = np.asarray(archive["dof"], dtype=np.float64)
                fps = np.asarray(archive["fps"])
                engine = np.asarray(archive["engine"])
                joint_names = np.asarray(archive["joint_names"])
        except (OSError, ValueError, KeyError) as error:
            raise ValueError("vertical migration retarget archive is invalid") from error
        if (
            dof.shape != (stop - start, 29)
            or not np.isfinite(dof).all()
            or fps.shape != () or float(fps) != 120.0
            or engine.shape != () or str(engine) != "gmr"
            or joint_names.shape != (29,)
            or tuple(joint_names.astype(str).tolist()) != G1_JOINT_NAMES
        ):
            raise ValueError("vertical migration retarget joint contract is invalid")
        q = mujoco_to_isaaclab_joints(dof[::4])
        qdot = _backward_finite_difference(q, 30.0)
        output[stem] = (
            start,
            np.ascontiguousarray(q, dtype=np.float32),
            np.ascontiguousarray(qdot, dtype=np.float32),
        )
    return output


def migrate_vertical_dataset_v3(
    *,
    source_v2: Path,
    retarget_root: Path,
    output: Path | None = None,
) -> VerticalDataset:
    """Upgrade one authenticated compact v2 corpus using direct retarget q/qdot."""

    manifest, v2_splits, _ = _load_authenticated_vertical_v2(source_v2)
    retarget = _authenticated_retarget_joint_state(
        retarget_root, str(manifest["retarget_manifest_sha256"])
    )
    if set(retarget) != set(manifest["source_roles"]):
        raise ValueError("vertical migration retarget inventory mismatch")
    accepted: Counter[str] = Counter({stem: 0 for stem in retarget})
    rejected: defaultdict[str, Counter[str]] = defaultdict(Counter)
    rejected_joints: Counter[str] = Counter(
        {name: 0 for name in ISAACLAB_JOINT_NAMES}
    )
    migrated: dict[str, VerticalSplitArrays] = {}
    joint_target = OUTPUT_LAYOUT["joint_position"]
    for split in _SPLITS:
        source = v2_splits[split]
        keep: list[int] = []
        x_v3: list[np.ndarray] = []
        for index in range(len(source["phase"])):
            clip_id = str(source["clip_id"][index])
            mirrored = bool(source["mirrored"][index])
            suffix = "__mirror"
            stem = clip_id.removesuffix(suffix)
            if mirrored != clip_id.endswith(suffix) or stem not in retarget:
                raise ValueError("vertical migration row mirror/source identity is invalid")
            start, q_trace, qdot_trace = retarget[stem]
            offset = int(source["center_frame_120hz"][index]) - start
            if offset < 0 or offset % 4 or offset // 4 >= len(q_trace):
                raise ValueError("vertical migration row has no exact retarget frame")
            frame = offset // 4
            if frame < 1:
                rejected[stem]["missing_unique_predecessor"] += 1
                continue
            q = q_trace[frame]
            qdot = qdot_trace[frame]
            if mirrored:
                q = mirror_classic_g1_joint_state(q)
                qdot = mirror_classic_g1_joint_state(qdot)
            delta = np.asarray(source["y"][index, joint_target], dtype=np.float32) - q
            unsafe = np.flatnonzero(np.abs(delta) > MAXIMUM_JOINT_STEP_RAD)
            if len(unsafe):
                rejected[stem]["joint_step_exceeds_limit"] += 1
                for joint_index in unsafe:
                    rejected_joints[ISAACLAB_JOINT_NAMES[int(joint_index)]] += 1
                continue
            keep.append(index)
            x_v3.append(pack_classic_g1_input_v3(source["x"][index], q, qdot))
            accepted[stem] += 1
        if not keep:
            raise ValueError(f"vertical migration removed every {split} row")
        indices = np.asarray(keep, dtype=np.int64)
        migrated[split] = VerticalSplitArrays(
            x=np.stack(x_v3).astype(np.float32),
            **{
                name: np.asarray(source[name])[indices]
                for name in _SPLIT_FIELDS
                if name != "x"
            },
        )
    train = migrated["train"]
    x_mean = np.mean(train.x, axis=0, dtype=np.float64).astype(np.float32)
    x_std = np.std(train.x, axis=0, dtype=np.float64).astype(np.float32)
    y_mean = np.mean(train.y, axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.std(train.y, axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < np.float32(1.0e-6)] = np.float32(1.0)
    y_std[y_std < np.float32(1.0e-6)] = np.float32(1.0)
    y_mean[OUTPUT_LAYOUT["contact_logit"]] = np.float32(0.0)
    y_std[OUTPUT_LAYOUT["contact_logit"]] = np.float32(1.0)
    normalization = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    receipt = canonical_joint_state_receipt(
        accepted_rows_by_source=dict(accepted),
        rejected_rows_by_source={source: dict(counts) for source, counts in rejected.items()},
        rejected_rows_by_joint=dict(rejected_joints),
        state_source_by_clip={stem: "unique_predecessor" for stem in retarget},
        migration_provenance={
            "source_dataset_sha256": str(manifest["dataset_sha256"]),
            "retarget_manifest_sha256": str(manifest["retarget_manifest_sha256"]),
        },
    )
    dataset_sha = _dataset_digest(
        migrated,
        normalization,
        selection_sha256=str(manifest["selection_sha256"]),
        retarget_manifest_sha256=str(manifest["retarget_manifest_sha256"]),
        terrain_receipt_set_sha256=str(manifest["terrain_receipt_set_sha256"]),
        source_roles=manifest["source_roles"],
        joint_state_receipt=receipt,
    )
    dataset = VerticalDataset(
        splits=migrated,
        **normalization,
        selection_sha256=str(manifest["selection_sha256"]),
        retarget_manifest_sha256=str(manifest["retarget_manifest_sha256"]),
        terrain_receipt_set_sha256=str(manifest["terrain_receipt_set_sha256"]),
        source_roles=manifest["source_roles"],
        joint_state_receipt=receipt,
        dataset_sha256=dataset_sha,
    )
    if output is not None:
        save_vertical_dataset(Path(output), dataset)
    return dataset


def _canonical_selection_sha256(document: Mapping[str, object]) -> str:
    if set(document) != {"schema", "items", "sha256"}:
        raise ValueError("vertical source selection fields are invalid")
    payload = {"schema": document["schema"], "items": document["items"]}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if type(value) is not dict:
        raise ValueError(f"{label} is invalid")
    return value


def _relative_artifact(root: Path, value: object) -> Path:
    if type(value) is not str or not value or Path(value).is_absolute():
        raise ValueError("retarget artifact path is invalid")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("retarget artifact path escapes root")
    return path


def _footstep_cycles(path: Path) -> tuple[tuple[int, int], ...]:
    markers: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            markers.append(int(fields[0]))
        except ValueError as error:
            raise ValueError("PFNN footstep marker is invalid") from error
    if len(markers) < 2 or any(stop <= start for start, stop in zip(markers, markers[1:])):
        raise ValueError("PFNN footstep markers must be strictly increasing")
    return tuple(zip(markers, markers[1:]))


def _flat_height(points: np.ndarray) -> np.ndarray:
    query = np.asarray(points)
    if query.ndim < 1 or query.shape[-1] != 2:
        raise ValueError("flat PFNN terrain query must end in two coordinates")
    return np.full(query.shape[:-1], PFNN_G1_Z_OFFSET_M, dtype=np.float64)


def _flat_terrain_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "g1-pfnn-flat-surface/v1",
                "z_offset_m": PFNN_G1_Z_OFFSET_M,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_vertical_dataset_from_retarget(
    *,
    retarget_root: Path,
    pfnn_root: Path,
    patches_path: Path,
    model_path: Path,
    output: Path,
    fk: object | None = None,
    load_one: Callable[..., PFNNSourceClip] = load_pfnn_retarget_source,
    contacts_one: Callable[..., np.ndarray] = released_pfnn_contacts_for_interval,
    fit_one: Callable[..., PFNNTerrainFit] = fit_terrain_cycle,
) -> VerticalDataset:
    """Fit all selected PFNN cycles and atomically write the compact corpus."""

    retarget = Path(retarget_root).expanduser().resolve(strict=True)
    released = Path(pfnn_root).expanduser().resolve(strict=True)
    patches = Path(patches_path).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite vertical output: {destination}")
    selection_path = retarget / "source-selection.json"
    manifest_path = retarget / "retarget-manifest.json"
    selection = _read_json(selection_path, "vertical source selection")
    manifest = _read_json(manifest_path, "vertical retarget manifest")
    selection_sha = _canonical_selection_sha256(selection)
    if (
        selection.get("schema") != "g1-pfnn-vertical-slice-selection/v1"
        or selection.get("sha256") != selection_sha
        or not isinstance(selection.get("items"), list)
        or len(selection["items"]) < 4
    ):
        raise ValueError("vertical source selection contract is invalid")
    if (
        set(manifest) != {"schema", "status", "selection_sha256", "items"}
        or manifest.get("schema") != "g1-pfnn-vertical-slice-retarget/v1"
        or manifest.get("status") != "accepted"
        or manifest.get("selection_sha256") != selection_sha
        or not isinstance(manifest.get("items"), list)
        or len(manifest["items"]) != len(selection["items"])
    ):
        raise ValueError("vertical retarget manifest contract is invalid")
    selected_by_stem = {
        str(item.get("stem")): item
        for item in selection["items"]
        if isinstance(item, dict)
    }
    retarget_by_stem = {
        str(item.get("stem")): item
        for item in manifest["items"]
        if isinstance(item, dict)
    }
    if (
        len(selected_by_stem) != len(selection["items"])
        or len(retarget_by_stem) != len(manifest["items"])
        or set(selected_by_stem) != set(retarget_by_stem)
    ):
        raise ValueError("vertical selection and retarget identities differ")
    retarget_sha = _sha256(manifest_path)
    forward_kinematics = fk if fk is not None else G1MujocoFK(model_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
    )
    try:
        terrain_root = temporary / "terrain"
        terrain_root.mkdir()
        sources: list[VerticalSliceSource] = []
        for stem in sorted(selected_by_stem, key=lambda value: (selected_by_stem[value]["role"], value)):
            selected = selected_by_stem[stem]
            item = retarget_by_stem[stem]
            required_selection = {
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
            required_item = {
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
            if set(selected) != required_selection or set(item) != required_item:
                raise ValueError("vertical source item fields are invalid")
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
                raise ValueError("vertical retarget item conflicts with selection")
            start = selected["start_frame_120hz"]
            stop = selected["stop_frame_120hz"]
            if (
                selected["role"] not in _SPLITS
                or type(start) is not int
                or type(stop) is not int
                or stop - start < 8
                or not isinstance(selected["coverage"], list)
            ):
                raise ValueError("vertical source item metadata is invalid")
            source_path = released / "data" / "animations" / f"{stem}.bvh"
            phase_path = source_path.with_suffix(".phase")
            gait_path = source_path.with_suffix(".gait")
            footsteps_path = source_path.with_name(f"{stem}_footsteps.txt")
            for path, field in (
                (source_path, "bvh_sha256"),
                (phase_path, "phase_sha256"),
                (gait_path, "gait_sha256"),
                (footsteps_path, "footsteps_sha256"),
            ):
                if not path.is_file() or _sha256(path) != selected[field]:
                    raise ValueError(f"vertical source {field} mismatch")
            motion_path = _relative_artifact(retarget, item["output"])
            receipt_path = _relative_artifact(retarget, item["receipt"])
            if (
                _sha256(motion_path) != item["output_sha256"]
                or _sha256(receipt_path) != item["receipt_sha256"]
            ):
                raise ValueError("vertical retarget artifact digest mismatch")
            clip = load_one(
                motion_path,
                forward_kinematics,
                clip_id=stem,
                terrain_id="pfnn_vertical",
                expected_source_sha256=selected["bvh_sha256"],
                expected_start_frame=start,
            )
            if clip.frame_count != len(range(start, stop, 4)):
                raise ValueError("vertical retarget frame count mismatch")
            phase_values = np.asarray(np.loadtxt(phase_path), dtype=np.float64)
            if phase_values.ndim != 1 or len(phase_values) < stop:
                raise ValueError("released PFNN phase sidecar length mismatch")
            contacts = contacts_one(
                pfnn_root=released,
                source=source_path,
                display_start_frame=start,
                display_frame_count=stop - start,
            )
            track = released_pfnn_phase_track(phase_values[start:stop], contacts)
            core_start = start + 120
            core_stop = stop - 120
            segments: list[VerticalSurfaceSegment] = []
            if {"ascent", "descent"} <= set(selected["coverage"]):
                for cycle_start, cycle_stop in _footstep_cycles(footsteps_path):
                    bounded_start = max(core_start, cycle_start)
                    bounded_stop = min(core_stop, cycle_stop)
                    if bounded_stop <= bounded_start:
                        continue
                    fit = fit_one(
                        pfnn_root=released,
                        patches_path=patches,
                        source=source_path,
                        cycle_start=cycle_start,
                        cycle_stop=cycle_stop,
                        display_start=bounded_start,
                        display_count=bounded_stop - bounded_start,
                    )
                    if (
                        fit.source_sha256 != selected["bvh_sha256"]
                        or fit.patches_sha256 != _sha256(patches)
                    ):
                        raise ValueError("terrain fit digest mismatch")
                    fit_path = terrain_root / f"{stem}__{cycle_start:05d}_{cycle_stop:05d}.npz"
                    save_terrain_fit(fit_path, fit)
                    terrain_sha = _sha256(fit_path)
                    surface = PlacedPFNNSurface(fit)
                    segments.append(
                        VerticalSurfaceSegment(
                            cycle_start_frame_120hz=bounded_start,
                            cycle_stop_frame_120hz=bounded_stop,
                            terrain_sha256=terrain_sha,
                            height_at=surface.height_at,
                        )
                    )
            else:
                segments.append(
                    VerticalSurfaceSegment(
                        cycle_start_frame_120hz=core_start,
                        cycle_stop_frame_120hz=core_stop,
                        terrain_sha256=_flat_terrain_sha256(),
                        height_at=_flat_height,
                    )
                )
            if not segments:
                raise ValueError(f"{stem}: no PFNN terrain cycles overlap the selected core")
            sources.append(
                VerticalSliceSource(
                    role=selected["role"],
                    stem=stem,
                    source_start_frame_120hz=start,
                    clip=clip,
                    phase_track=track,
                    segments=tuple(segments),
                    selection_sha256=selection_sha,
                    retarget_manifest_sha256=retarget_sha,
                )
            )
        dataset = build_vertical_dataset(tuple(sources))
        save_vertical_dataset(temporary / "dataset", dataset)
        os.replace(temporary, destination)
        return dataset
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retarget-root", type=Path, required=True)
    parser.add_argument("--source-v2", type=Path)
    parser.add_argument("--pfnn-root", type=Path)
    parser.add_argument("--patches-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.source_v2 is not None:
        if any(
            value is not None
            for value in (
                arguments.pfnn_root,
                arguments.patches_path,
                arguments.model_path,
            )
        ):
            raise ValueError(
                "--source-v2 cannot be combined with terrain rebuild inputs"
            )
        dataset = migrate_vertical_dataset_v3(
            source_v2=arguments.source_v2,
            retarget_root=arguments.retarget_root,
            output=arguments.output,
        )
    else:
        if any(
            value is None
            for value in (
                arguments.pfnn_root,
                arguments.patches_path,
                arguments.model_path,
            )
        ):
            raise ValueError(
                "terrain rebuild requires --pfnn-root, --patches-path, and --model-path"
            )
        dataset = build_vertical_dataset_from_retarget(
            retarget_root=arguments.retarget_root,
            pfnn_root=arguments.pfnn_root,
            patches_path=arguments.patches_path,
            model_path=arguments.model_path,
            output=arguments.output,
        )
    print(
        json.dumps(
            {
                "status": "accepted",
                "dataset_sha256": dataset.dataset_sha256,
                "train_rows": len(dataset.splits["train"].phase),
                "validation_rows": len(dataset.splits["validation"].phase),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "VERTICAL_DATASET_SCHEMA",
    "VerticalDataset",
    "VerticalSliceSource",
    "VerticalSplitArrays",
    "VerticalSurfaceSegment",
    "build_vertical_dataset",
    "build_vertical_dataset_from_retarget",
    "load_vertical_dataset",
    "migrate_vertical_dataset_v3",
    "save_vertical_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
