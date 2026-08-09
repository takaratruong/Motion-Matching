"""Deterministic source inventory for a playable released-PFNN G1 slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Literal

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FRAMES_RE = re.compile(r"(?m)^\s*Frames:\s*(\d+)\s*$")
_FRAME_TIME_RE = re.compile(
    r"(?m)^\s*Frame\s+Time:\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*$"
)
_SOURCE_FPS = 120.0
_CONTEXT_FRAMES = 120
_TURN_THRESHOLD_RAD = 0.5
_GRADE_THRESHOLD_DEGREES = 3.0
_STRAIGHT_DISPLACEMENT_M = 0.5
_AUDIT_WINDOW_FRAMES = 600
_AUDIT_STRIDE_FRAMES = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(path: Path) -> tuple[int, float]:
    text = path.read_text(encoding="utf-8")
    frames = _FRAMES_RE.findall(text)
    frame_times = _FRAME_TIME_RE.findall(text)
    if len(frames) != 1 or len(frame_times) != 1:
        raise ValueError(f"{path.name}: PFNN BVH frame metadata is invalid")
    frame_count = int(frames[0])
    frame_time = float(frame_times[0])
    if (
        frame_count < 1
        or not math.isfinite(frame_time)
        or frame_time <= 0.0
        or not math.isclose(frame_time, 1.0 / _SOURCE_FPS, abs_tol=1.0e-5)
    ):
        raise ValueError(f"{path.name}: released PFNN source must be exactly 120 Hz")
    return frame_count, _SOURCE_FPS


@dataclass(frozen=True)
class ReleasedPFNNRecord:
    stem: str
    bvh_path: Path
    phase_path: Path
    gait_path: Path
    footsteps_path: Path
    bvh_sha256: str
    phase_sha256: str
    gait_sha256: str
    footsteps_sha256: str
    frame_count: int
    fps: float

    def __post_init__(self) -> None:
        if type(self.stem) is not str or not self.stem or self.stem.endswith("_mirror"):
            raise ValueError("released PFNN stem must be nonempty and non-mirrored")
        for name in ("bvh_path", "phase_path", "gait_path", "footsteps_path"):
            path = Path(getattr(self, name))
            if not path.is_file():
                raise FileNotFoundError(path)
            object.__setattr__(self, name, path)
        for name in (
            "bvh_sha256",
            "phase_sha256",
            "gait_sha256",
            "footsteps_sha256",
        ):
            if _SHA256_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.frame_count) is not int or self.frame_count < 1:
            raise ValueError("released PFNN frame count must be positive")
        if float(self.fps) != _SOURCE_FPS:
            raise ValueError("released PFNN record must be exactly 120 Hz")


@dataclass(frozen=True)
class PFNNIntervalMetrics:
    stem: str
    start_frame_120hz: int
    stop_frame_120hz: int
    root_displacement_m: float
    maximum_left_turn_rad: float
    maximum_right_turn_rad: float
    maximum_ascent_degrees: float
    minimum_descent_degrees: float
    has_idle_transition: bool

    def __post_init__(self) -> None:
        if type(self.stem) is not str or not self.stem:
            raise ValueError("PFNN interval stem must be nonempty")
        if (
            type(self.start_frame_120hz) is not int
            or type(self.stop_frame_120hz) is not int
            or self.start_frame_120hz < 0
            or self.stop_frame_120hz <= self.start_frame_120hz
        ):
            raise ValueError("PFNN interval bounds are invalid")
        for name in (
            "root_displacement_m",
            "maximum_left_turn_rad",
            "maximum_right_turn_rad",
            "maximum_ascent_degrees",
            "minimum_descent_degrees",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if type(self.has_idle_transition) is not bool:
            raise TypeError("has_idle_transition must be bool")


@dataclass(frozen=True)
class PFNNSliceRole:
    record: ReleasedPFNNRecord
    role: Literal["train", "validation"]
    start_frame_120hz: int
    stop_frame_120hz: int
    coverage: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.record, ReleasedPFNNRecord):
            raise TypeError("PFNN slice record is invalid")
        if self.role not in ("train", "validation"):
            raise ValueError("PFNN slice role must be train or validation")
        if (
            type(self.start_frame_120hz) is not int
            or type(self.stop_frame_120hz) is not int
            or self.start_frame_120hz < 0
            or self.stop_frame_120hz > self.record.frame_count
            or self.stop_frame_120hz <= self.start_frame_120hz
        ):
            raise ValueError("PFNN slice interval is outside its source")
        allowed = {
            "idle_transition",
            "straight",
            "left_turn",
            "right_turn",
            "ascent",
            "descent",
        }
        if not self.coverage or set(self.coverage) - allowed:
            raise ValueError("PFNN slice coverage is invalid")


def discover_released_pfnn_records(root: Path) -> tuple[ReleasedPFNNRecord, ...]:
    """Discover non-mirrored released PFNN records with exact sidecars."""

    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    output: list[ReleasedPFNNRecord] = []
    seen: set[str] = set()
    for bvh_path in sorted(source_root.glob("*.bvh"), key=lambda path: path.name):
        stem = bvh_path.stem
        if stem == "rest" or stem.endswith("_mirror"):
            continue
        canonical = stem.casefold()
        if canonical in seen:
            raise ValueError(f"duplicate PFNN stem: {stem}")
        seen.add(canonical)
        phase_path = source_root / f"{stem}.phase"
        gait_path = source_root / f"{stem}.gait"
        footsteps_path = source_root / f"{stem}_footsteps.txt"
        missing = tuple(
            path
            for path in (phase_path, gait_path, footsteps_path)
            if not path.is_file()
        )
        if missing:
            raise ValueError(
                f"{stem}: missing PFNN sidecar: "
                + ", ".join(path.name for path in missing)
            )
        frame_count, fps = _source_metadata(bvh_path)
        output.append(
            ReleasedPFNNRecord(
                stem=stem,
                bvh_path=bvh_path,
                phase_path=phase_path,
                gait_path=gait_path,
                footsteps_path=footsteps_path,
                bvh_sha256=_sha256(bvh_path),
                phase_sha256=_sha256(phase_path),
                gait_sha256=_sha256(gait_path),
                footsteps_sha256=_sha256(footsteps_path),
                frame_count=frame_count,
                fps=fps,
            )
        )
    if not output:
        raise ValueError("released PFNN inventory is empty")
    return tuple(output)


def _coverage(record: ReleasedPFNNRecord, metric: PFNNIntervalMetrics) -> frozenset[str]:
    names: set[str] = set()
    if record.stem.startswith("LocomotionFlat"):
        if (
            metric.has_idle_transition
            and metric.root_displacement_m >= _STRAIGHT_DISPLACEMENT_M
            and metric.maximum_left_turn_rad < _TURN_THRESHOLD_RAD
            and metric.maximum_right_turn_rad < _TURN_THRESHOLD_RAD
        ):
            names.update(("idle_transition", "straight"))
        if metric.maximum_left_turn_rad >= _TURN_THRESHOLD_RAD:
            names.add("left_turn")
        if metric.maximum_right_turn_rad >= _TURN_THRESHOLD_RAD:
            names.add("right_turn")
    if record.stem.startswith("WalkingUpSteps"):
        if metric.maximum_ascent_degrees >= _GRADE_THRESHOLD_DEGREES:
            names.add("ascent")
        if metric.minimum_descent_degrees <= -_GRADE_THRESHOLD_DEGREES:
            names.add("descent")
    return frozenset(names)


def interval_metrics_from_tracks(
    stem: str,
    root_position_world: object,
    facing_yaw: object,
    gait: object,
) -> tuple[PFNNIntervalMetrics, ...]:
    """Return the best bounded walk-only coverage intervals for one source."""

    from scipy.ndimage import gaussian_filter1d

    root = np.asarray(root_position_world, dtype=np.float64)
    yaw = np.asarray(facing_yaw, dtype=np.float64)
    gait_array = np.asarray(gait, dtype=np.float64)
    frames = len(root)
    if (
        type(stem) is not str
        or not stem
        or root.shape != (frames, 3)
        or yaw.shape != (frames,)
        or gait_array.shape != (frames, 8)
        or frames < _AUDIT_WINDOW_FRAMES + 2 * _CONTEXT_FRAMES
        or not np.isfinite(root).all()
        or not np.isfinite(yaw).all()
        or not np.isfinite(gait_array).all()
        or np.any(gait_array < 0.0)
        or np.any(gait_array > 1.0)
    ):
        raise ValueError("released PFNN audit tracks are invalid")
    smoothed_root = gaussian_filter1d(root, 30.0, axis=0, mode="nearest")
    unwrapped_yaw = np.unwrap(yaw)
    candidates: list[PFNNIntervalMetrics] = []
    last_start = frames - _CONTEXT_FRAMES - _AUDIT_WINDOW_FRAMES
    for start in range(_CONTEXT_FRAMES, last_start + 1, _AUDIT_STRIDE_FRAMES):
        stop = start + _AUDIT_WINDOW_FRAMES
        gait_window = gait_array[start:stop]
        if float(np.max(gait_window[:, 2:])) > 0.05:
            continue
        root_window = smoothed_root[start:stop]
        planar_delta = np.diff(root_window[:, :2], axis=0)
        vertical_delta = np.diff(root_window[:, 2])
        horizontal_step = np.linalg.norm(planar_delta, axis=1)
        grade = np.degrees(
            np.arctan2(vertical_delta, np.maximum(horizontal_step, 1.0e-6))
        )
        relative_yaw = unwrapped_yaw[start:stop] - unwrapped_yaw[start]
        candidates.append(
            PFNNIntervalMetrics(
                stem=stem,
                start_frame_120hz=start,
                stop_frame_120hz=stop,
                root_displacement_m=float(
                    np.linalg.norm(root_window[-1, :2] - root_window[0, :2])
                ),
                maximum_left_turn_rad=float(np.max(relative_yaw)),
                maximum_right_turn_rad=float(-np.min(relative_yaw)),
                maximum_ascent_degrees=float(np.quantile(grade, 0.95)),
                minimum_descent_degrees=float(np.quantile(grade, 0.05)),
                has_idle_transition=bool(
                    np.max(gait_window[:, 0]) >= 0.8
                    and np.max(gait_window[:, 1]) >= 0.8
                ),
            )
        )
    if not candidates:
        return ()
    output: list[PFNNIntervalMetrics] = []
    if stem.startswith("LocomotionFlat"):
        straight = [
            metric
            for metric in candidates
            if metric.has_idle_transition
            and metric.root_displacement_m >= _STRAIGHT_DISPLACEMENT_M
            and metric.maximum_left_turn_rad < _TURN_THRESHOLD_RAD
            and metric.maximum_right_turn_rad < _TURN_THRESHOLD_RAD
        ]
        turning = [
            metric
            for metric in candidates
            if metric.maximum_left_turn_rad >= _TURN_THRESHOLD_RAD
            and metric.maximum_right_turn_rad >= _TURN_THRESHOLD_RAD
        ]
        if straight:
            output.append(
                max(
                    straight,
                    key=lambda value: (
                        value.root_displacement_m,
                        -value.start_frame_120hz,
                    ),
                )
            )
        if turning:
            output.append(
                max(
                    turning,
                    key=lambda value: (
                        min(
                            value.maximum_left_turn_rad,
                            value.maximum_right_turn_rad,
                        ),
                        value.root_displacement_m,
                        -value.start_frame_120hz,
                    ),
                )
            )
    elif stem.startswith("WalkingUpSteps"):
        terrain = [
            metric
            for metric in candidates
            if metric.maximum_ascent_degrees >= _GRADE_THRESHOLD_DEGREES
            and metric.minimum_descent_degrees <= -_GRADE_THRESHOLD_DEGREES
        ]
        if terrain:
            output.append(
                max(
                    terrain,
                    key=lambda value: (
                        min(
                            value.maximum_ascent_degrees,
                            -value.minimum_descent_degrees,
                        ),
                        value.root_displacement_m,
                        -value.start_frame_120hz,
                    ),
                )
            )
    return tuple(sorted(output, key=lambda value: value.start_frame_120hz))


def audit_released_pfnn_metrics(
    records: tuple[ReleasedPFNNRecord, ...], pfnn_root: Path
) -> tuple[PFNNIntervalMetrics, ...]:
    """Measure deterministic interval coverage from the real PFNN skeleton."""

    from scipy.ndimage import gaussian_filter1d

    from mm_sonic.pfnn_terrain_fit import _pfnn_modules

    root = Path(pfnn_root).expanduser().resolve(strict=True)
    bvh, animation_module = _pfnn_modules(root)
    output: list[PFNNIntervalMetrics] = []
    for record in sorted(records, key=lambda value: value.stem):
        if not (
            record.stem.startswith("LocomotionFlat")
            or record.stem.startswith("WalkingUpSteps")
        ):
            continue
        animation, _, _ = bvh.load(str(record.bvh_path))
        transforms = animation_module.transforms_global(animation)
        positions = transforms[:, :, :3, 3] / transforms[:, :, 3:, 3]
        if len(positions) != record.frame_count or positions.shape[1] <= 25:
            raise ValueError(f"{record.stem}: PFNN skeleton topology is invalid")
        across = (positions[:, 18] - positions[:, 25]) + (
            positions[:, 2] - positions[:, 7]
        )
        norms = np.linalg.norm(across, axis=1)
        if np.any(norms < 1.0e-12):
            raise ValueError(f"{record.stem}: PFNN facing axis is invalid")
        across /= norms[:, None]
        source_forward = gaussian_filter1d(
            np.cross(across, np.array([[0.0, 1.0, 0.0]])),
            20.0,
            axis=0,
            mode="nearest",
        )
        source_forward /= np.linalg.norm(source_forward, axis=1)[:, None]
        root_g1 = np.column_stack(
            (
                positions[:, 0, 0],
                -positions[:, 0, 2],
                positions[:, 0, 1],
            )
        ) * 0.056444
        yaw_g1 = np.unwrap(np.arctan2(-source_forward[:, 2], source_forward[:, 0]))
        gait = np.asarray(np.loadtxt(record.gait_path), dtype=np.float64)
        output.extend(
            interval_metrics_from_tracks(record.stem, root_g1, yaw_g1, gait)
        )
    return tuple(
        sorted(output, key=lambda value: (value.stem, value.start_frame_120hz))
    )


def _slice(
    record: ReleasedPFNNRecord,
    metric: PFNNIntervalMetrics,
    *,
    role: Literal["train", "validation"],
    coverage: frozenset[str],
) -> PFNNSliceRole:
    if (
        metric.start_frame_120hz < _CONTEXT_FRAMES
        or metric.stop_frame_120hz + _CONTEXT_FRAMES > record.frame_count
    ):
        raise ValueError(f"{record.stem}: interval lacks one-second context")
    return PFNNSliceRole(
        record=record,
        role=role,
        start_frame_120hz=metric.start_frame_120hz - _CONTEXT_FRAMES,
        stop_frame_120hz=metric.stop_frame_120hz + _CONTEXT_FRAMES,
        coverage=tuple(sorted(coverage)),
    )


def select_vertical_slice(
    records: tuple[ReleasedPFNNRecord, ...],
    metrics: Iterable[PFNNIntervalMetrics],
) -> tuple[PFNNSliceRole, ...]:
    """Select three train clips and one terrain validation clip deterministically."""

    record_by_stem: dict[str, ReleasedPFNNRecord] = {}
    for record in records:
        if not isinstance(record, ReleasedPFNNRecord):
            raise TypeError("records must contain ReleasedPFNNRecord values")
        if record.stem in record_by_stem:
            raise ValueError(f"duplicate PFNN stem: {record.stem}")
        record_by_stem[record.stem] = record
    candidates: list[
        tuple[ReleasedPFNNRecord, PFNNIntervalMetrics, frozenset[str]]
    ] = []
    ordered_metrics = sorted(
        metrics,
        key=lambda value: (
            getattr(value, "stem", ""),
            getattr(value, "start_frame_120hz", -1),
            getattr(value, "stop_frame_120hz", -1),
        ),
    )
    seen_metrics: set[tuple[str, int, int]] = set()
    for metric in ordered_metrics:
        if not isinstance(metric, PFNNIntervalMetrics):
            raise TypeError("metrics must contain PFNNIntervalMetrics values")
        stem = metric.stem
        record = record_by_stem.get(stem)
        if record is None or not isinstance(metric, PFNNIntervalMetrics):
            raise ValueError(f"{stem}: PFNN interval metric has no source record")
        key = (stem, metric.start_frame_120hz, metric.stop_frame_120hz)
        if key in seen_metrics:
            raise ValueError(f"{stem}: duplicate PFNN interval metric")
        seen_metrics.add(key)
        candidates.append((record, metric, _coverage(record, metric)))

    def first(
        predicate,
        *,
        excluded: frozenset[str] = frozenset(),
        missing: str,
    ) -> tuple[ReleasedPFNNRecord, PFNNIntervalMetrics, frozenset[str]]:
        match = next(
            (
                item
                for item in candidates
                if item[0].stem not in excluded and predicate(item[2])
            ),
            None,
        )
        if match is None:
            raise ValueError(f"vertical slice lacks {missing}")
        return match

    straight = first(
        lambda value: {"idle_transition", "straight"} <= value,
        missing="idle_transition/straight",
    )
    turning = first(
        lambda value: {"left_turn", "right_turn"} <= value,
        excluded=frozenset((straight[0].stem,)),
        missing="right_turn or left_turn",
    )
    terrain_train = first(
        lambda value: {"ascent", "descent"} <= value,
        excluded=frozenset((straight[0].stem, turning[0].stem)),
        missing="ascent or descent",
    )
    terrain_validation = first(
        lambda value: {"ascent", "descent"} <= value,
        excluded=frozenset(
            (straight[0].stem, turning[0].stem, terrain_train[0].stem)
        ),
        missing="validation ascent/descent",
    )
    selected = (
        _slice(*straight[:2], role="train", coverage=straight[2]),
        _slice(*turning[:2], role="train", coverage=turning[2]),
        _slice(*terrain_train[:2], role="train", coverage=terrain_train[2]),
        _slice(
            *terrain_validation[:2],
            role="validation",
            coverage=terrain_validation[2],
        ),
    )
    required = {
        "idle_transition",
        "straight",
        "left_turn",
        "right_turn",
        "ascent",
        "descent",
    }
    missing = required - required_coverage(selected)
    if missing:
        raise ValueError("vertical slice lacks " + ", ".join(sorted(missing)))
    return selected


def required_coverage(selection: tuple[PFNNSliceRole, ...]) -> frozenset[str]:
    values: set[str] = set()
    for item in selection:
        if not isinstance(item, PFNNSliceRole):
            raise TypeError("selection must contain PFNNSliceRole values")
        values.update(item.coverage)
    return frozenset(values)


def vertical_slice_receipt(selection: tuple[PFNNSliceRole, ...]) -> dict[str, object]:
    """Return canonical, path-independent selection provenance."""

    if len(selection) != 4:
        raise ValueError("vertical slice receipt requires exactly four items")
    items = []
    for item in sorted(selection, key=lambda value: (value.role, value.record.stem)):
        record = item.record
        items.append(
            {
                "stem": record.stem,
                "role": item.role,
                "start_frame_120hz": item.start_frame_120hz,
                "stop_frame_120hz": item.stop_frame_120hz,
                "coverage": list(item.coverage),
                "bvh_sha256": record.bvh_sha256,
                "phase_sha256": record.phase_sha256,
                "gait_sha256": record.gait_sha256,
                "footsteps_sha256": record.footsteps_sha256,
            }
        )
    payload = {"schema": "g1-pfnn-vertical-slice-selection/v1", "items": items}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "sha256": digest}
