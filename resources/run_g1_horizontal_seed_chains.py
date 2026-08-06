#!/usr/bin/env python3
"""Package terrain-valid horizontal motion seeds for MotionBricks."""

from __future__ import annotations

import math

import numpy as np

from mm_sonic.joints import ContractError


_KINDS = ("mount", "interior", "dismount")
_ARRAY_KEYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
)


def _chain_boundaries(
    seeds: tuple[dict[str, object], ...],
    *,
    maximum_gap_m: float,
) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(seeds, tuple)
        or tuple(seed.get("kind") for seed in seeds) != _KINDS
    ):
        raise ContractError("horizontal chain seed order is invalid")
    if not math.isfinite(float(maximum_gap_m)) or maximum_gap_m < 0.0:
        raise ContractError("horizontal chain maximum gap is invalid")
    intervals = []
    for seed in seeds:
        try:
            start, stop = (
                float(value) for value in seed["path_interval_m"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "horizontal chain seed interval is invalid"
            ) from error
        if (
            not math.isfinite(start)
            or not math.isfinite(stop)
            or start < 0.0
            or stop <= start
        ):
            raise ContractError("horizontal chain seed interval is invalid")
        intervals.append((start, stop))
    boundaries = []
    for index, ((_, first_stop), (second_start, _)) in enumerate(
        zip(intervals[:-1], intervals[1:])
    ):
        uncovered = max(0.0, second_start - first_stop)
        overlap = max(0.0, first_stop - second_start)
        if uncovered > maximum_gap_m + 1.0e-9:
            raise ContractError("horizontal chain has an uncovered boundary")
        boundaries.append(
            {
                "from_kind": _KINDS[index],
                "to_kind": _KINDS[index + 1],
                "uncovered_m": uncovered,
                "overlap_m": overlap,
            }
        )
    return tuple(boundaries)


def _connector(value: object) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or not set(_ARRAY_KEYS).issubset(value):
        raise ContractError("horizontal seed connector is invalid")
    arrays = {
        name: np.asarray(value[name], dtype=np.float64)
        for name in _ARRAY_KEYS
    }
    frames = len(arrays["joint_position"])
    if (
        frames < 2
        or arrays["joint_position"].shape != (frames, 29)
        or arrays["root_position_world"].shape != (frames, 3)
        or arrays["root_orientation_world_wxyz"].shape != (frames, 4)
        or not all(np.isfinite(array).all() for array in arrays.values())
        or np.any(
            np.abs(
                np.linalg.norm(
                    arrays["root_orientation_world_wxyz"], axis=1
                )
                - 1.0
            )
            > 1.0e-4
        )
    ):
        raise ContractError("horizontal seed connector is invalid")
    return arrays


def _build_playlist(
    chains: tuple[dict[str, object], ...],
    *,
    hold_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if (
        not isinstance(chains, tuple)
        or not chains
        or type(hold_frames) is not int
        or hold_frames < 1
    ):
        raise ContractError("horizontal seed playlist input is invalid")
    chunks = {name: [] for name in _ARRAY_KEYS}
    segments = []
    frame = 0
    lane_ids = set()
    for chain in chains:
        try:
            lane_id = str(chain["lane_id"])
            seeds = tuple(chain["seeds"])
        except (KeyError, TypeError) as error:
            raise ContractError(
                "horizontal seed playlist input is invalid"
            ) from error
        if (
            not lane_id
            or lane_id in lane_ids
            or tuple(seed.get("kind") for seed in seeds) != _KINDS
        ):
            raise ContractError("horizontal seed playlist input is invalid")
        lane_ids.add(lane_id)
        for seed in seeds:
            arrays = _connector(seed.get("arrays"))
            motion_frames = len(arrays["joint_position"])
            start = frame
            motion_start = start + hold_frames
            motion_stop = motion_start + motion_frames
            stop = motion_stop + hold_frames
            for name in _ARRAY_KEYS:
                chunks[name].extend(
                    (
                        np.repeat(
                            arrays[name][0:1], hold_frames, axis=0
                        ),
                        arrays[name],
                        np.repeat(
                            arrays[name][-1:], hold_frames, axis=0
                        ),
                    )
                )
            segments.append(
                {
                    "lane_id": lane_id,
                    "kind": str(seed["kind"]),
                    "source_clip": str(seed["source_clip"]),
                    "path_interval_m": [
                        float(value)
                        for value in seed["path_interval_m"]
                    ],
                    "segment_frames": [start, stop],
                    "motion_frames": [motion_start, motion_stop],
                }
            )
            frame = stop
    playlist = {
        name: np.ascontiguousarray(np.concatenate(values, axis=0))
        for name, values in chunks.items()
    }
    metadata = {
        "schema": "g1-horizontal-seed-playlist/v1",
        "frame_count": frame,
        "hold_frames": hold_frames,
        "segments": segments,
        "teleport_boundaries": [
            segment["segment_frames"][0] for segment in segments[1:]
        ],
    }
    return playlist, metadata
