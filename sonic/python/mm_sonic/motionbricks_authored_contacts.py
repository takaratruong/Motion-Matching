"""Authored MotionBricks contacts paired with displayed qpos frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AuthoredFootContacts:
    channels: tuple[bool, bool, bool, bool]
    stance: tuple[bool, bool]
    valid: bool
    reason: str


def collapse_authored_contacts(values: object) -> tuple[bool, bool]:
    channels = np.asarray(values)
    if (
        channels.shape != (4,)
        or channels.dtype.kind not in "biuf"
        or not np.isfinite(channels.astype(np.float64)).all()
    ):
        raise ValueError("authored contacts must contain four finite channels")
    active = channels.astype(np.float64) > 0.5
    return (
        bool(active[0] or active[1]),
        bool(active[2] or active[3]),
    )


def _unavailable(reason: str) -> AuthoredFootContacts:
    return AuthoredFootContacts(
        channels=(False, False, False, False),
        stance=(False, False),
        valid=False,
        reason=reason,
    )


def _as_numpy(values: object) -> np.ndarray:
    detached = (
        values.detach()
        if callable(getattr(values, "detach", None))
        else values
    )
    cpu = (
        detached.cpu()
        if callable(getattr(detached, "cpu", None))
        else detached
    )
    return np.asarray(cpu)


def sample_authored_contacts(
    full_agent: object,
    frame_index: int,
) -> AuthoredFootContacts:
    try:
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame index must be a nonnegative integer")
        frames = getattr(full_agent, "frames", None)
        if not isinstance(frames, dict) or "model_features" not in frames:
            raise ValueError("model_features are unavailable")
        features = frames["model_features"]
        if len(features.shape) != 3 or frame_index >= int(features.shape[1]):
            raise ValueError("contact frame index is outside model_features")
        motion_rep = getattr(full_agent, "_motion_rep", None)
        extractor = getattr(motion_rep, "extract_foot_contacts", None)
        if not callable(extractor):
            raise ValueError("MotionBricks contact extractor is unavailable")
        extracted = extractor(
            features[:, frame_index : frame_index + 1],
            is_normalized=False,
            contact_thresh=0.5,
        )
        array = _as_numpy(extracted)
        if array.shape != (1, 1, 4):
            raise ValueError("contact extractor returned an unexpected shape")
        channels = tuple(bool(value) for value in array[0, 0])
        stance = collapse_authored_contacts(channels)
        return AuthoredFootContacts(channels, stance, True, "ok")
    except Exception as error:
        return _unavailable(str(error))
