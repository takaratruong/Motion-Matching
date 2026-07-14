from collections.abc import Sequence

import numpy as np

from .schema import EvaluationSplit, LabeledInteractionClip


def split_objects(
    clips: Sequence[LabeledInteractionClip],
    heldout_count: int = 20,
    seed: int = 20260714,
) -> EvaluationSplit:
    objects = sorted({clip.motion.object_id for clip in clips})
    if not 0 < heldout_count < len(objects):
        raise ValueError(
            f"need at least {heldout_count + 1} unique objects, "
            f"got {len(objects)}"
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
    database = set(split.database_objects)
    heldout = set(split.heldout_objects)
    if database & heldout:
        raise ValueError(
            "object identity leak between database and heldout split"
        )
    expected = {clip.motion.object_id for clip in clips}
    if database | heldout != expected:
        raise ValueError(
            "split does not cover every source object exactly once"
        )
