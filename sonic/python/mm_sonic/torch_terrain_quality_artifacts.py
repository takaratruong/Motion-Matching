"""Authenticate source-conditioned quality evidence from saved route arrays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np

from .torch_contact_segments import source_support_mask
from .torch_terrain_contact_quality import (
    ContactQualityMetrics,
    evaluate_contact_quality,
)
from .torch_terrain_features import TerrainDataset


_TIMING_ARRAYS = frozenset(("step_time_ns", "search_time_ns"))
_REQUIRED_ARRAYS = frozenset(
    (
        "foot_position_world",
        "foot_surface_height_m",
        "root_position_world",
        "command_velocity_world_xy",
        "selected_clip_path",
        "selected_source_frame",
    )
)


def _readonly(value: np.ndarray, *, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SavedRouteQuality:
    route_name: str
    metrics: ContactQualityMetrics
    source_support_mask: np.ndarray
    transition_start_mask: np.ndarray
    deterministic_sha256: str

    def __post_init__(self) -> None:
        support = np.asarray(self.source_support_mask)
        transitions = np.asarray(self.transition_start_mask)
        if not isinstance(self.route_name, str) or not self.route_name:
            raise ValueError("saved quality route name is invalid")
        if not isinstance(self.metrics, ContactQualityMetrics):
            raise ValueError("saved quality metrics are invalid")
        if (
            support.dtype != np.bool_
            or support.ndim != 2
            or support.shape[1:] != (2,)
            or transitions.dtype != np.bool_
            or transitions.shape != (support.shape[0],)
        ):
            raise ValueError("saved quality masks are invalid")
        digest = self.deterministic_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("saved quality hash is invalid")
        object.__setattr__(
            self, "source_support_mask", _readonly(support, dtype=np.bool_)
        )
        object.__setattr__(
            self, "transition_start_mask", _readonly(transitions, dtype=np.bool_)
        )


def _validate_identities(
    selected_clip_path: np.ndarray,
    selected_source_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    paths = np.asarray(selected_clip_path)
    frames = np.asarray(selected_source_frame)
    if paths.ndim != 1 or paths.dtype.kind not in "US" or any(
        not str(value) for value in paths.tolist()
    ):
        raise ValueError("selected clip paths must be a nonempty string vector")
    if (
        frames.shape != paths.shape
        or frames.dtype.kind not in "iu"
        or bool((frames < 0).any())
    ):
        raise ValueError("selected source frames must be aligned non-negative integers")
    return paths.astype(str, copy=False), frames.astype(np.int64, copy=False)


def reconstruct_source_support(
    dataset: TerrainDataset,
    selected_clip_path: np.ndarray,
    selected_source_frame: np.ndarray,
) -> np.ndarray:
    """Resolve every saved clip/frame identity to authenticated source support."""

    if not isinstance(dataset, TerrainDataset):
        raise ValueError("source support reconstruction requires a TerrainDataset")
    paths, frames = _validate_identities(
        selected_clip_path, selected_source_frame
    )
    by_path: dict[str, int] = {}
    for clip_index, clip in enumerate(dataset.folder.clips):
        path = str(clip.relative_path)
        if path in by_path:
            raise ValueError(f"duplicate dataset clip identity: {path}")
        by_path[path] = clip_index

    cached: dict[int, np.ndarray] = {}
    output = np.empty((paths.shape[0], 2), dtype=bool)
    for row, (raw_path, raw_frame) in enumerate(zip(paths, frames)):
        path = str(raw_path)
        clip_index = by_path.get(path)
        if clip_index is None:
            raise ValueError(f"unknown selected clip: {path}")
        if clip_index not in cached:
            cached[clip_index] = (
                source_support_mask(dataset, clip_index)
                .detach()
                .cpu()
                .numpy()
                .astype(bool, copy=False)
            )
        frame = int(raw_frame)
        if not 0 <= frame < cached[clip_index].shape[0]:
            raise ValueError(
                f"selected source frame {frame} is outside clip {path}"
            )
        output[row] = cached[clip_index][frame]
    return _readonly(output, dtype=np.bool_)


def transition_start_mask(
    selected_clip_path: np.ndarray,
    selected_source_frame: np.ndarray,
) -> np.ndarray:
    """Mark frame zero and every non-consecutive source transition."""

    paths, frames = _validate_identities(
        selected_clip_path, selected_source_frame
    )
    output = np.zeros(paths.shape[0], dtype=bool)
    if output.size:
        output[0] = True
        output[1:] = (paths[1:] != paths[:-1]) | (
            frames[1:] != frames[:-1] + 1
        )
    return _readonly(output, dtype=np.bool_)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _metrics_json(metrics: ContactQualityMetrics) -> dict:
    return {
        "contact_agreement_fraction": metrics.contact_agreement_fraction,
        "expected_stance_floating_fraction": list(
            metrics.expected_stance_floating_fraction
        ),
        "maximum_no_contact_frames": metrics.maximum_no_contact_frames,
        "unload_count": list(metrics.unload_count),
        "touchdown_count": list(metrics.touchdown_count),
        "complete_step_count": metrics.complete_step_count,
        "complete_steps_per_m": metrics.complete_steps_per_m,
        "source_stance_drift_m": list(metrics.source_stance_drift_m),
        "transition_source_stance_drift_m": (
            metrics.transition_source_stance_drift_m
        ),
        "steady_source_stance_drift_m": metrics.steady_source_stance_drift_m,
        "command_to_unload_frames": list(metrics.command_to_unload_frames),
        "command_to_touchdown_frames": list(
            metrics.command_to_touchdown_frames
        ),
    }


def _quality_hash(
    route_name: str,
    arrays: Mapping[str, np.ndarray],
    support: np.ndarray,
    transitions: np.ndarray,
    metrics: ContactQualityMetrics,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-terrain-motion-quality-route/v1\0")
    digest.update(route_name.encode("utf-8"))
    for name in sorted(set(arrays) - _TIMING_ARRAYS):
        value = np.asarray(arrays[name])
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(_canonical_json(value.shape))
        digest.update(value.tobytes(order="C"))
    for name, value in (
        ("source_support_mask", support),
        ("transition_start_mask", transitions),
    ):
        digest.update(b"\0" + name.encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    digest.update(b"\0metrics\0")
    digest.update(_canonical_json(_metrics_json(metrics)))
    return digest.hexdigest()


def analyze_saved_route(
    dataset: TerrainDataset,
    route_name: str,
    arrays: Mapping[str, np.ndarray],
) -> SavedRouteQuality:
    """Attach truthful source-conditioned contact metrics to one saved route."""

    if not isinstance(route_name, str) or not route_name:
        raise ValueError("saved quality route name is invalid")
    if not isinstance(arrays, Mapping):
        raise ValueError("saved route arrays must be a mapping")
    missing = sorted(_REQUIRED_ARRAYS - set(arrays))
    if missing:
        raise ValueError("saved route arrays are missing: " + ", ".join(missing))
    support = reconstruct_source_support(
        dataset,
        arrays["selected_clip_path"],
        arrays["selected_source_frame"],
    )
    transitions = transition_start_mask(
        arrays["selected_clip_path"], arrays["selected_source_frame"]
    )
    metrics = evaluate_contact_quality(
        foot_position_world=arrays["foot_position_world"],
        foot_surface_height_m=arrays["foot_surface_height_m"],
        source_support_mask=support,
        root_position_world=arrays["root_position_world"],
        command_velocity_world_xy=arrays["command_velocity_world_xy"],
        transition_start_mask=transitions,
    )
    return SavedRouteQuality(
        route_name=route_name,
        metrics=metrics,
        source_support_mask=support,
        transition_start_mask=transitions,
        deterministic_sha256=_quality_hash(
            route_name, arrays, support, transitions, metrics
        ),
    )
