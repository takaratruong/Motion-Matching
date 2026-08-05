#!/usr/bin/env python3
"""Package certified complete horizontal lanes into one viewer playlist."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mm_sonic.joints import ContractError


_CONNECTOR_KEYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-summary", type=Path, required=True)
    parser.add_argument(
        "--composite",
        action="append",
        default=[],
        metavar="LANE_ID=COMPOSITE_NPZ",
    )
    parser.add_argument("--hold-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _connector(value: object, lane_id: str) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or set(value) != set(_CONNECTOR_KEYS):
        raise ContractError(f"complete grid connector is invalid: {lane_id}")
    arrays = {
        name: np.asarray(value[name], dtype=np.float64)
        for name in _CONNECTOR_KEYS
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
        raise ContractError(f"complete grid connector is invalid: {lane_id}")
    return arrays


def _build_playlist(
    *,
    lanes: list[dict[str, object]],
    connectors: dict[str, dict[str, np.ndarray]],
    hold_frames: int,
) -> tuple[
    dict[str, np.ndarray], dict[str, object], list[dict[str, object]]
]:
    if len(lanes) != 11 or type(hold_frames) is not int or hold_frames < 1:
        raise ContractError("complete grid playlist options are invalid")
    lane_ids = []
    results = []
    chunks = {name: [] for name in _CONNECTOR_KEYS}
    segments = []
    frame = 0
    for expected_index, lane in enumerate(lanes):
        try:
            lane_id = str(lane["lane_id"])
            lane_index = int(lane["lane_index"])
            center_y = float(lane["center_y_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("complete grid lane is invalid") from error
        if (
            not lane_id
            or lane_id in lane_ids
            or lane_index != expected_index
            or not math.isfinite(center_y)
        ):
            raise ContractError("complete grid lane is invalid")
        lane_ids.append(lane_id)
        connector = connectors.get(lane_id)
        if connector is None:
            results.append(
                {
                    "lane_id": lane_id,
                    "lane_index": lane_index,
                    "center_y_m": center_y,
                    "classification": "unresolved",
                    "reason": "no-certified-complete-composite",
                }
            )
            continue
        arrays = _connector(connector, lane_id)
        motion_frames = len(arrays["joint_position"])
        start = frame
        motion_start = start + hold_frames
        motion_stop = motion_start + motion_frames
        stop = motion_stop + hold_frames
        for name in _CONNECTOR_KEYS:
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
                "lane_index": lane_index,
                "center_y_m": center_y,
                "classification": "full",
                "segment_frames": [start, stop],
                "motion_frames": [motion_start, motion_stop],
            }
        )
        results.append(
            {
                "lane_id": lane_id,
                "lane_index": lane_index,
                "center_y_m": center_y,
                "classification": "complete",
                "reason": None,
                "motion_frame_count": motion_frames,
            }
        )
        frame = stop
    if not segments:
        raise ContractError("complete grid has no certified traversal")
    arrays = {
        name: np.ascontiguousarray(np.concatenate(chunks[name], axis=0))
        for name in _CONNECTOR_KEYS
    }
    metadata = {
        "schema": "g1-horizontal-grid-playlist/v1",
        "phase": "complete-traversal",
        "frame_count": frame,
        "hold_frames": hold_frames,
        "segments": segments,
        "teleport_boundaries": [
            segment["segment_frames"][0] for segment in segments[1:]
        ],
    }
    return arrays, metadata, results


def _load_connector(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, KeyError) as error:
        raise ContractError(f"failed to load composite: {path}") from error


def _parse_composites(values: list[str]) -> dict[str, dict[str, np.ndarray]]:
    output = {}
    for value in values:
        lane_id, separator, raw_path = value.partition("=")
        path = Path(raw_path)
        if not separator or not lane_id or lane_id in output:
            raise ContractError(f"invalid composite argument: {value}")
        metrics_path = path.with_name("metrics.json")
        try:
            metrics = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(
                f"composite metrics are unavailable: {metrics_path}"
            ) from error
        if metrics.get("accepted") is not True:
            raise ContractError(f"composite is not accepted: {path}")
        output[lane_id] = _load_connector(path)
    return output


def main() -> int:
    args = _parser().parse_args()
    try:
        summary = json.loads(args.phase_summary.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("phase summary is unavailable") from error
    lanes = summary.get("lanes")
    if (
        summary.get("schema") != "g1-horizontal-grid-phase-coverage/v1"
        or not isinstance(lanes, list)
    ):
        raise ContractError("phase summary is invalid")
    connectors = _parse_composites(args.composite)
    arrays, metadata, results = _build_playlist(
        lanes=lanes,
        connectors=connectors,
        hold_frames=args.hold_frames,
    )
    metadata["source_summary"] = str(args.phase_summary)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "complete-grid-playlist.npz", **arrays)
    (output / "complete-grid-playlist.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "schema": "g1-complete-horizontal-grid-attempts/v1",
        "lane_attempt_count": len(results),
        "complete_lane_count": sum(
            row["classification"] == "complete" for row in results
        ),
        "infeasible_lane_count": sum(
            row["classification"] == "infeasible" for row in results
        ),
        "unresolved_lane_count": sum(
            row["classification"] == "unresolved" for row in results
        ),
        "lanes": results,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
