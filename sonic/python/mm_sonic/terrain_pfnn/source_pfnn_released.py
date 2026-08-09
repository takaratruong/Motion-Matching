"""Deterministic source inventory for a playable released-PFNN G1 slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Literal, Mapping


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
    metrics: Mapping[str, PFNNIntervalMetrics],
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
    for stem in sorted(metrics):
        metric = metrics[stem]
        record = record_by_stem.get(stem)
        if record is None or not isinstance(metric, PFNNIntervalMetrics):
            raise ValueError(f"{stem}: PFNN interval metric has no source record")
        if metric.stem != stem:
            raise ValueError(f"{stem}: PFNN interval metric stem mismatch")
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
