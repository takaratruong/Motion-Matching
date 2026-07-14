from collections.abc import Sequence

import numpy as np

from .schema import EvaluationSplit, LabeledInteractionClip


def split_objects(
    clips: Sequence[LabeledInteractionClip],
    heldout_count: int = 20,
    seed: int = 20260714,
) -> EvaluationSplit:
    objects = sorted({clip.motion.object_id for clip in clips})
    if heldout_count <= 0:
        raise ValueError(
            f"heldout_count must be positive, got {heldout_count}"
        )
    if heldout_count >= len(objects):
        raise ValueError(
            f"heldout_count {heldout_count} must leave at least one "
            "database object; "
            f"got {len(objects)}"
            " unique objects"
        )
    permutation = np.random.default_rng(seed).permutation(len(objects))
    heldout = tuple(
        sorted(objects[index] for index in permutation[:heldout_count])
    )
    database = tuple(sorted(set(objects) - set(heldout)))
    split = EvaluationSplit(seed, database, heldout)
    validate_split(clips, split)
    return split


def validate_split(
    clips: Sequence[LabeledInteractionClip],
    split: EvaluationSplit,
) -> None:
    partitions = (
        ("database", split.database_objects),
        ("heldout", split.heldout_objects),
    )
    for category, object_ids in partitions:
        if not object_ids:
            raise ValueError(f"{category} partition is empty")
        duplicates = sorted(
            {
                object_id
                for object_id in object_ids
                if object_ids.count(object_id) > 1
            }
        )
        if duplicates:
            raise ValueError(
                f"duplicate {category} object IDs: {duplicates!r}"
            )
        if tuple(object_ids) != tuple(sorted(object_ids)):
            raise ValueError(
                f"{category} object IDs must be sorted; "
                f"got {tuple(object_ids)!r}"
            )

    database = set(split.database_objects)
    heldout = set(split.heldout_objects)
    overlap = sorted(database & heldout)
    if overlap:
        raise ValueError(
            "overlapping object IDs in database and heldout partitions: "
            f"{overlap!r}"
        )
    expected = {clip.motion.object_id for clip in clips}
    actual = database | heldout
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            f"missing source object IDs from split: {missing!r}"
        )
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(
            "unknown split object IDs not present in clips: "
            f"{unknown!r}"
        )


def partition_clips(
    clips: Sequence[LabeledInteractionClip],
    split: EvaluationSplit,
) -> tuple[
    tuple[LabeledInteractionClip, ...],
    tuple[LabeledInteractionClip, ...],
]:
    validate_split(clips, split)
    database = set(split.database_objects)
    database_clips = tuple(
        clip for clip in clips if clip.motion.object_id in database
    )
    heldout_clips = tuple(
        clip for clip in clips if clip.motion.object_id not in database
    )
    return database_clips, heldout_clips
