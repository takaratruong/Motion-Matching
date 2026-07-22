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
from .sources import load_raw_interaction, sequence_parts
from .splits import split_objects


EXPECTED_CLIP_ERRORS = (
    SourceValidationError,
    ConversionValidationError,
    InteractionValidationError,
)

_SCHEMA_V1_REJECTION_CODES_BY_ERROR = {
    SourceValidationError: frozenset(
        {
            "fps_mismatch",
            "frame_count_mismatch",
            "invalid_contact",
            "invalid_dimensions",
            "invalid_fps",
            "invalid_quaternion",
            "invalid_shape",
            "invalid_source_frames",
            "invalid_source_record",
            "missing_field",
            "non_finite",
            "object_identity_mismatch",
        }
    ),
    ConversionValidationError: frozenset(
        {
            "duration_error",
            "fk_error",
            "fk_rotation_error",
            "fps_mismatch",
            "frame_count_mismatch",
            "invalid_contact",
            "invalid_dimensions",
            "invalid_fps",
            "invalid_quaternion",
            "invalid_shape",
            "invalid_source_frames",
            "joint_limit_violation",
            "non_finite",
            "skeleton_mismatch",
        }
    ),
    InteractionValidationError: frozenset(
        {
            "ambiguous_active_hand",
            "contact_lost_before_hold",
            "invalid_approach",
            "invalid_grasp",
            "no_distinct_lift_phase",
            "no_five_centimeter_lift",
            "no_stable_contact",
            "no_stable_hold",
        }
    ),
}
_SCHEMA_V1_REJECTION_CODES = frozenset().union(
    *_SCHEMA_V1_REJECTION_CODES_BY_ERROR.values()
)
_SCHEMA_V1_REJECTION_ERROR_BY_STAGE = {
    "source": SourceValidationError,
    "conversion": ConversionValidationError,
    "interaction": InteractionValidationError,
}
_SCHEMA_V1_REJECTION_STAGE_BY_ERROR = {
    error_type: stage
    for stage, error_type in _SCHEMA_V1_REJECTION_ERROR_BY_STAGE.items()
}

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


def all_schema_v1_rejection_codes() -> tuple[str, ...]:
    """Return the reviewed clip-level exclusions allowed in schema v1."""
    return tuple(sorted(_SCHEMA_V1_REJECTION_CODES))


def is_schema_v1_rejection(stage: str, code: str) -> bool:
    """Return whether a stage/code pair is a reviewed schema-v1 exclusion."""
    if not isinstance(stage, str) or not isinstance(code, str):
        return False
    error_type = _SCHEMA_V1_REJECTION_ERROR_BY_STAGE.get(stage)
    if error_type is None:
        return False
    return code in _SCHEMA_V1_REJECTION_CODES_BY_ERROR[error_type]


def build_object_dimensions(
    sources: Sequence[SourcePaths],
) -> dict[str, np.ndarray]:
    """Read consistent deterministic USD bounds for selected objects."""
    representatives: dict[tuple[str, str], SourcePaths] = {}
    for source in sorted(sources, key=lambda item: item.sequence_id):
        category = sequence_parts(source.sequence_id)[0]
        representatives.setdefault((source.object_id, category), source)

    category_dimensions: dict[tuple[str, str], np.ndarray] = {}
    for key in sorted(representatives):
        object_id, category = key
        value = np.asarray(
            read_usd_dimensions(representatives[key].object_usd),
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
        category_dimensions[(object_id, category)] = value.copy()

    dimensions = {}
    for object_id in sorted({key[0] for key in category_dimensions}):
        values = [
            category_dimensions[(object_id, category)]
            for category in sorted(
                category
                for candidate_object, category in category_dimensions
                if candidate_object == object_id
            )
        ]
        reference = values[0]
        if not all(
            np.allclose(
                np.sort(reference), np.sort(value), rtol=0.0, atol=1e-3
            )
            for value in values[1:]
        ):
            raise ValueError(
                "inconsistent cross-category USD bounds "
                f"for {object_id}: {values}"
            )
        dimensions[object_id] = category_dimensions.get(
            (object_id, "pickup_table"), reference
        ).copy()
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
            stage = _SCHEMA_V1_REJECTION_STAGE_BY_ERROR.get(type(error))
            if stage is None or not is_schema_v1_rejection(
                stage, error.code
            ):
                raise
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
    source_roots: Sequence[Path],
    source_count: int,
    included: Sequence[LabeledInteractionClip],
    rejections: Sequence[Rejection],
    database: Sequence[LabeledInteractionClip],
    artifact: InteractionArtifact,
    split: EvaluationSplit,
    target_fps: float,
    diagnostic_limit: int | None,
) -> dict:
    roots = tuple(
        sorted({Path(root).resolve() for root in source_roots}, key=str)
    )
    if not roots:
        raise ValueError("at least one source root is required")
    multi_root = len(roots) > 1
    clips = []
    for labeled, start, stop in zip(
        database,
        artifact.range_starts,
        artifact.range_stops,
        strict=True,
    ):
        clip = {
            "sequence_id": labeled.motion.sequence_id,
            "object_id": labeled.motion.object_id,
            "active_hand": int(labeled.active_hand),
            "range_start": int(start),
            "range_stop": int(stop),
        }
        if multi_root:
            clip["source_category"] = sequence_parts(
                labeled.motion.sequence_id
            )[0]
        clips.append(clip)
    manifest = {
        "schema_version": 1,
        "database_magic": DB_MAGIC.decode("ascii"),
        "feature_magic": FEATURE_MAGIC.decode("ascii"),
        "skeleton_names": list(G1_SKELETON.names),
        "skeleton_parents": G1_SKELETON.parents.astype(int).tolist(),
        "skeleton_signature": G1_SKELETON.signature(),
        "source_root": str(
            Path(os.path.commonpath([str(root) for root in roots]))
        ),
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
    if multi_root:
        manifest["source_roots"] = [str(root) for root in roots]
    return manifest
