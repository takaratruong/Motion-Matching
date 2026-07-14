from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import subprocess

import numpy as np

from resources.g1_terrain_builder.kinematics import G1Kinematics
from resources.g1_terrain_builder.schema import SkeletonSpec

from .artifacts import (
    DB_MAGIC,
    FEATURE_MAGIC,
    prepare_artifacts,
)
from .conversion import convert_interaction
from .features import FEATURE_NAMES, serialized_feature_groups
from .metadata import DEPENDENCY_VERSION_KEYS, GRAIL_DATASET_ID
from .object_geometry import read_usd_dimensions
from .phases import derive_interaction_labels
from .schema import (
    ConversionValidationError,
    EvaluationSplit,
    FeatureSet,
    G1_SKELETON,
    InteractionArtifact,
    InteractionValidationError,
    LabeledInteractionClip,
    PhaseConfig,
    SourcePaths,
    SourceValidationError,
)
from .sources import load_raw_interaction
from .splits import split_objects


EXPECTED_CLIP_ERRORS = (
    SourceValidationError,
    ConversionValidationError,
    InteractionValidationError,
)

NUMERIC_BOUND_NAMES = (
    ("fk_max_error_m", "fk_max_error_m"),
    (
        "fk_rotation_max_error_degrees",
        "fk_rotation_max_error_degrees",
    ),
    ("duration_max_error_s", "duration_error_s"),
    ("quaternion_norm_max_error", "quaternion_norm_max_error"),
)


@dataclass(frozen=True)
class Rejection:
    sequence_id: str
    object_id: str
    stage: str
    code: str
    message: str


def build_object_dimensions(
    sources: Sequence[SourcePaths],
) -> dict[str, np.ndarray]:
    """Read one deterministic USD bound per selected object identity."""
    representatives: dict[str, SourcePaths] = {}
    for source in sorted(sources, key=lambda item: item.sequence_id):
        representatives.setdefault(source.object_id, source)

    dimensions = {}
    for object_id in sorted(representatives):
        value = np.asarray(
            read_usd_dimensions(representatives[object_id].object_usd),
            np.float32,
        )
        if (
            value.shape != (3,)
            or not np.isfinite(value).all()
            or np.any(value <= 0)
        ):
            raise ValueError(
                f"invalid object bounds for {object_id}: {value}"
            )
        dimensions[object_id] = value.copy()
    return dimensions


def build_labeled_clips(
    sources: Sequence[SourcePaths],
    dimensions: Mapping[str, np.ndarray],
    kinematics: G1Kinematics,
) -> tuple[list[LabeledInteractionClip], list[Rejection], list[dict]]:
    included: list[LabeledInteractionClip] = []
    rejected: list[Rejection] = []
    numeric_reports: list[dict] = []
    for source in sorted(sources, key=lambda item: item.sequence_id):
        try:
            raw = load_raw_interaction(
                source, dimensions[source.object_id]
            )
            canonical, skeleton, numeric = convert_interaction(
                raw, kinematics, 25.0
            )
            if skeleton.signature() != G1_SKELETON.signature():
                raise ConversionValidationError(
                    "skeleton_mismatch", skeleton.signature()
                )
            included.append(derive_interaction_labels(canonical))
            numeric_reports.append(numeric)
        except EXPECTED_CLIP_ERRORS as error:
            stage = (
                "source"
                if isinstance(error, SourceValidationError)
                else "conversion"
                if isinstance(error, ConversionValidationError)
                else "interaction"
            )
            rejected.append(
                Rejection(
                    source.sequence_id,
                    source.object_id,
                    stage,
                    error.code,
                    str(error),
                )
            )
    rejected.sort(
        key=lambda item: (item.sequence_id, item.stage, item.code)
    )
    return included, rejected, numeric_reports


def prepare_database(
    clips: Sequence[LabeledInteractionClip],
    heldout_count: int,
    seed: int,
    skeleton: SkeletonSpec,
) -> tuple[
    tuple[LabeledInteractionClip, ...],
    InteractionArtifact,
    FeatureSet,
    EvaluationSplit,
]:
    split = split_objects(
        clips, heldout_count=heldout_count, seed=seed
    )
    database, artifact, features = prepare_artifacts(
        clips, split, skeleton
    )
    return database, artifact, features, split


def build_validation_report(
    source_count: int,
    included: Sequence[LabeledInteractionClip],
    rejections: Sequence[Rejection],
    numeric_reports: Sequence[Mapping[str, object]],
    included_frames: int,
) -> dict:
    ordered_rejections = sorted(
        rejections,
        key=lambda item: (item.sequence_id, item.stage, item.code),
    )
    histogram = Counter(item.code for item in ordered_rejections)
    numeric_bounds = {}
    for output_name, report_name in NUMERIC_BOUND_NAMES:
        numeric_bounds[output_name] = max(
            (float(report[report_name]) for report in numeric_reports),
            default=0.0,
        )
    return {
        "schema_version": 1,
        "source_clips": int(source_count),
        "included_clips": len(included),
        "rejected_clips": len(ordered_rejections),
        "included_frames": int(included_frames),
        "rejections_by_code": {
            code: histogram[code] for code in sorted(histogram)
        },
        "rejections": [
            dataclasses.asdict(item) for item in ordered_rejections
        ],
        "numeric_bounds": numeric_bounds,
    }


def _dependency_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in DEPENDENCY_VERSION_KEYS
    }


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _source_date_epoch() -> int | None:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            "SOURCE_DATE_EPOCH must be an integer"
        ) from error


def build_manifest(
    *,
    source_root: Path,
    source_count: int,
    included: Sequence[LabeledInteractionClip],
    rejections: Sequence[Rejection],
    database: Sequence[LabeledInteractionClip],
    artifact: InteractionArtifact,
    split: EvaluationSplit,
    target_fps: float,
    diagnostic_limit: int | None,
) -> dict:
    clips = []
    for labeled, start, stop in zip(
        database,
        artifact.range_starts,
        artifact.range_stops,
        strict=True,
    ):
        clips.append(
            {
                "sequence_id": labeled.motion.sequence_id,
                "object_id": labeled.motion.object_id,
                "active_hand": int(labeled.active_hand),
                "range_start": int(start),
                "range_stop": int(stop),
            }
        )
    return {
        "schema_version": 1,
        "database_magic": DB_MAGIC.decode("ascii"),
        "feature_magic": FEATURE_MAGIC.decode("ascii"),
        "skeleton_names": list(G1_SKELETON.names),
        "skeleton_parents": G1_SKELETON.parents.astype(int).tolist(),
        "skeleton_signature": G1_SKELETON.signature(),
        "source_root": str(Path(source_root).resolve()),
        "dataset_id": GRAIL_DATASET_ID,
        "source_clips": int(source_count),
        "included_clips": len(included),
        "rejected_clips": len(rejections),
        "target_fps": float(target_fps),
        "clips": clips,
        "phase_config": dataclasses.asdict(PhaseConfig()),
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": serialized_feature_groups(),
        "split": {
            "seed": int(split.seed),
            "database_object_count": len(split.database_objects),
            "heldout_object_count": len(split.heldout_objects),
        },
        "dependency_versions": _dependency_versions(),
        "git_commit": _git_commit(),
        "diagnostic_limit": diagnostic_limit,
        "source_date_epoch": _source_date_epoch(),
    }
