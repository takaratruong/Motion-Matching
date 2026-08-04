"""Authenticated artifacts for offline horizontal staircase traversals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping

from .joints import ContractError
from .torch_terrain_omni_routes import StairFrame


_SHA256_HEX_LENGTH = 64


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


@dataclass(frozen=True, order=True)
class HorizontalGridLine:
    line_index: int
    stair_u_m: float
    direction: int

    def __post_init__(self) -> None:
        if (
            type(self.line_index) is not int
            or self.line_index < 0
            or not _is_finite_number(self.stair_u_m)
            or self.direction not in (-1, 1)
        ):
            raise ContractError("horizontal grid line is invalid")

    @property
    def line_id(self) -> str:
        return _canonical_sha256(
            {
                "schema": "g1-horizontal-grid-line/v1",
                **asdict(self),
            }
        )


@dataclass(frozen=True, order=True)
class TraversalArtifact:
    line_id: str
    route_sha256: str
    route_path: str
    source_family: str
    source_clip: str
    phase_boundaries: tuple[int, ...]
    minimum_sole_clearance_m: float
    maximum_planted_error_m: float
    maximum_joint_speed_rad_s: float
    maximum_joint_acceleration_rad_s2: float

    def __post_init__(self) -> None:
        finite_metrics = (
            self.minimum_sole_clearance_m,
            self.maximum_planted_error_m,
            self.maximum_joint_speed_rad_s,
            self.maximum_joint_acceleration_rad_s2,
        )
        if (
            not _is_sha256(self.line_id)
            or not _is_sha256(self.route_sha256)
            or not isinstance(self.route_path, str)
            or not self.route_path
            or self.source_family not in ("curb", "stair")
            or not isinstance(self.source_clip, str)
            or not self.source_clip
            or len(self.phase_boundaries) != 6
            or any(type(boundary) is not int for boundary in self.phase_boundaries)
            or tuple(sorted(self.phase_boundaries)) != self.phase_boundaries
            or self.phase_boundaries[0] < 0
            or self.phase_boundaries[-1] <= self.phase_boundaries[0]
            or any(not _is_finite_number(metric) for metric in finite_metrics)
            or self.maximum_planted_error_m < 0.0
            or self.maximum_joint_speed_rad_s < 0.0
            or self.maximum_joint_acceleration_rad_s2 < 0.0
        ):
            raise ContractError("horizontal traversal artifact is invalid")


@dataclass(frozen=True, order=True)
class TraversalFailure:
    line_id: str
    failed_phase: str
    rejection_histogram: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.line_id)
            or not isinstance(self.failed_phase, str)
            or not self.failed_phase
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count < 1
                for reason, count in self.rejection_histogram
            )
            or tuple(sorted(self.rejection_histogram))
            != self.rejection_histogram
        ):
            raise ContractError("horizontal traversal failure is invalid")


def _validate_stair_frame(frame: object) -> None:
    if (
        not isinstance(frame, StairFrame)
        or len(frame.origin_world_xy) != 2
        or any(not _is_finite_number(value) for value in frame.origin_world_xy)
        or not _is_finite_number(frame.ascent_world_yaw)
        or not _is_finite_number(frame.width_m)
        or float(frame.width_m) <= 0.0
        or not _is_finite_number(frame.tread_depth_m)
        or float(frame.tread_depth_m) <= 0.0
        or not _is_finite_number(frame.riser_height_m)
        or float(frame.riser_height_m) <= 0.0
        or type(frame.tread_count) is not int
        or frame.tread_count < 1
    ):
        raise ContractError("horizontal traversal stair frame is invalid")


@dataclass(frozen=True)
class HorizontalTraversalLibrary:
    dataset_manifest_sha256: str
    stair_frame: StairFrame
    spacing_m: float
    lines: tuple[HorizontalGridLine, ...]
    artifacts: tuple[TraversalArtifact, ...]
    failures: tuple[TraversalFailure, ...]

    def __post_init__(self) -> None:
        _validate_stair_frame(self.stair_frame)
        if (
            not _is_sha256(self.dataset_manifest_sha256)
            or not _is_finite_number(self.spacing_m)
            or float(self.spacing_m) <= 0.0
            or any(not isinstance(line, HorizontalGridLine) for line in self.lines)
            or any(
                not isinstance(artifact, TraversalArtifact)
                for artifact in self.artifacts
            )
            or any(
                not isinstance(failure, TraversalFailure)
                for failure in self.failures
            )
        ):
            raise ContractError("horizontal traversal library is invalid")

        line_ids = tuple(line.line_id for line in self.lines)
        if len(set(line_ids)) != len(line_ids):
            raise ContractError("horizontal traversal library has duplicate lines")
        known = set(line_ids)
        artifact_lines = {artifact.line_id for artifact in self.artifacts}
        failure_lines = {failure.line_id for failure in self.failures}
        if artifact_lines & failure_lines:
            raise ContractError(
                "horizontal traversal library has duplicate outcome"
            )
        if not artifact_lines | failure_lines <= known:
            raise ContractError("horizontal traversal outcome line is unknown")
        if artifact_lines | failure_lines != known:
            raise ContractError("horizontal traversal line has no outcome")
        if len(failure_lines) != len(self.failures):
            raise ContractError(
                "horizontal traversal library has duplicate failure"
            )

    @property
    def library_id(self) -> str:
        payload = self.to_dict()
        payload.pop("library_id", None)
        return _canonical_sha256(payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "g1-horizontal-traversal-library/v1",
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "stair_frame": asdict(self.stair_frame),
            "spacing_m": self.spacing_m,
            "lines": [
                {**asdict(line), "line_id": line.line_id}
                for line in self.lines
            ],
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "failures": [asdict(failure) for failure in self.failures],
            "library_id": _canonical_sha256(
                {
                    "schema": "g1-horizontal-traversal-library/v1",
                    "dataset_manifest_sha256": self.dataset_manifest_sha256,
                    "stair_frame": asdict(self.stair_frame),
                    "spacing_m": self.spacing_m,
                    "lines": [
                        {**asdict(line), "line_id": line.line_id}
                        for line in self.lines
                    ],
                    "artifacts": [
                        asdict(artifact) for artifact in self.artifacts
                    ],
                    "failures": [
                        asdict(failure) for failure in self.failures
                    ],
                }
            ),
        }


def horizontal_grid_lines(
    frame: StairFrame, *, spacing_m: float = 0.10
) -> tuple[HorizontalGridLine, ...]:
    _validate_stair_frame(frame)
    if not _is_finite_number(spacing_m) or float(spacing_m) <= 0.0:
        raise ContractError("horizontal traversal grid spacing is invalid")
    length = float(frame.tread_depth_m) * frame.tread_count
    centers = []
    value = float(spacing_m) / 2.0
    while value < length:
        centers.append(value)
        value += float(spacing_m)
    return tuple(
        HorizontalGridLine(index, center, direction)
        for index, center in enumerate(centers)
        for direction in (-1, 1)
    )


def horizontal_library_from_dict(
    payload: object,
) -> HorizontalTraversalLibrary:
    """Reconstruct and authenticate a serialized traversal library."""

    if (
        not isinstance(payload, dict)
        or payload.get("schema")
        != "g1-horizontal-traversal-library/v1"
        or not isinstance(payload.get("lines"), list)
        or not isinstance(payload.get("artifacts"), list)
        or not isinstance(payload.get("failures"), list)
    ):
        raise ContractError("serialized horizontal traversal library is invalid")
    try:
        lines = []
        for raw in payload["lines"]:
            fields = dict(raw)
            claimed_id = fields.pop("line_id")
            line = HorizontalGridLine(**fields)
            if line.line_id != claimed_id:
                raise ContractError(
                    "serialized horizontal traversal line identity is invalid"
                )
            lines.append(line)
        artifacts = tuple(
            TraversalArtifact(
                **{
                    **dict(raw),
                    "phase_boundaries": tuple(raw["phase_boundaries"]),
                }
            )
            for raw in payload["artifacts"]
        )
        failures = tuple(
            TraversalFailure(
                **{
                    **dict(raw),
                    "rejection_histogram": tuple(
                        tuple(item) for item in raw["rejection_histogram"]
                    ),
                }
            )
            for raw in payload["failures"]
        )
        library = HorizontalTraversalLibrary(
            dataset_manifest_sha256=payload["dataset_manifest_sha256"],
            stair_frame=StairFrame(**payload["stair_frame"]),
            spacing_m=payload["spacing_m"],
            lines=tuple(lines),
            artifacts=artifacts,
            failures=failures,
        )
        if payload.get("library_id") != library.library_id:
            raise ContractError(
                "serialized horizontal traversal library identity is invalid"
            )
        return library
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(
            "serialized horizontal traversal library is invalid"
        ) from error
