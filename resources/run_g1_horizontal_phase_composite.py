#!/usr/bin/env python3
"""Compose and certify one retrieved horizontal path traversal."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_path_phase_composite import compose_path_phases
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)
from resources.run_g1_path_motion_placement import _source_profiles


_CONNECTOR_KEYS = (
    "joint_position",
    "root_position_world",
    "root_orientation_world_wxyz",
)
_SOLE_CLEARANCE_MIN_M = -0.03
_STANCE_ERROR_MAX_M = 0.03


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend-frames", type=int, default=20)
    return parser


def _phase_source_slice(
    lane: dict[str, object], kind: str
) -> tuple[str, int, int]:
    phases = lane.get("phases")
    if not isinstance(phases, list):
        raise ContractError("phase lane is missing phases")
    rows = [row for row in phases if row.get("kind") == kind]
    if len(rows) != 1 or not isinstance(rows[0].get("selected"), dict):
        raise ContractError(f"phase lane has no selected {kind}")
    selected = rows[0]["selected"]
    clip = selected.get("source_clip")
    start = selected.get("start_frame")
    stop = selected.get("stop_frame")
    if (
        not isinstance(clip, str)
        or type(start) is not int
        or type(stop) is not int
        or start < 0
        or stop <= start
    ):
        raise ContractError(f"phase lane has invalid selected {kind}")
    return clip, start, stop


def _accepted_metrics(metrics: dict[str, object]) -> bool:
    try:
        minimum = float(metrics["minimum_sole_clearance_m"])
        maximum = float(metrics["maximum_stance_error_m"])
        unsupported = int(metrics["unsupported_frame_count"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        math.isfinite(minimum)
        and math.isfinite(maximum)
        and minimum >= _SOLE_CLEARANCE_MIN_M
        and maximum <= _STANCE_ERROR_MAX_M
        and unsupported == 0
    )


def _oriented_sole_contact_metrics(
    sole_clearance_m: object,
) -> tuple[np.ndarray, dict[str, object]]:
    clearance = np.asarray(sole_clearance_m, dtype=np.float64)
    if (
        clearance.ndim != 3
        or clearance.shape[1:] != (2, 7)
        or len(clearance) < 1
        or not np.isfinite(clearance).all()
    ):
        raise ContractError("oriented sole clearance is invalid")
    contact_error = np.min(np.abs(clearance), axis=2)
    support = contact_error <= _STANCE_ERROR_MAX_M
    unsupported = ~support.any(axis=1)
    supported_error = contact_error[support]
    metrics = {
        "minimum_sole_clearance_m": float(clearance.min()),
        "maximum_stance_error_m": float(
            supported_error.max() if len(supported_error) else math.inf
        ),
        "unsupported_frame_count": int(unsupported.sum()),
        "unsupported_frames": np.flatnonzero(unsupported).tolist(),
    }
    return np.asarray(support, dtype=np.bool_), metrics


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"failed to load JSON: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is invalid: {path}")
    return value


def _load_connector(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(_CONNECTOR_KEYS):
                raise ContractError(f"connector keys are invalid: {path}")
            return {
                name: np.asarray(archive[name], dtype=np.float64)
                for name in _CONNECTOR_KEYS
            }
    except OSError as error:
        raise ContractError(f"failed to load connector: {path}") from error


def _source_descriptors(root: Path) -> dict[str, dict[str, object]]:
    manifest = _load_json(root / "manifest.json")
    clips = manifest.get("accepted_clips")
    if not isinstance(clips, list):
        raise ContractError("source manifest has no accepted clips")
    output = {}
    for row in clips:
        if isinstance(row, dict) and isinstance(row.get("logical_name"), str):
            output[row["logical_name"]] = row
    return output


def main() -> int:
    args = _parser().parse_args()
    summary = _load_json(args.phase_root / "phase-grid-summary.json")
    lanes = summary.get("lanes")
    if not isinstance(lanes, list):
        raise ContractError("phase summary has no lanes")
    lane_rows = [row for row in lanes if row.get("lane_id") == args.lane_id]
    if len(lane_rows) != 1:
        raise ContractError(f"phase lane is unavailable: {args.lane_id}")
    lane = lane_rows[0]
    descriptors = _source_descriptors(args.source_dataset)

    connectors: dict[str, dict[str, np.ndarray]] = {}
    source_support: dict[str, np.ndarray] = {}
    lineage = {}
    for kind in ("mount", "interior", "dismount"):
        connectors[kind] = _load_connector(
            args.phase_root
            / "lanes"
            / args.lane_id
            / kind
            / "traversal.npz"
        )
        clip, start, stop = _phase_source_slice(lane, kind)
        if clip not in descriptors:
            raise ContractError(f"source clip is unavailable: {clip}")
        support = _source_profiles(
            args.source_dataset, descriptors[clip]
        )["support"][start:stop]
        if len(support) != len(connectors[kind]["joint_position"]):
            raise ContractError(f"source support length differs for {kind}")
        source_support[kind] = np.asarray(support, dtype=np.bool_)
        lineage[kind] = {
            "source_clip": clip,
            "start_frame": start,
            "stop_frame": stop,
        }

    arrays, composed_source_support, metrics = compose_path_phases(
        mount=connectors["mount"],
        interior=connectors["interior"],
        dismount=connectors["dismount"],
        mount_support=source_support["mount"],
        interior_support=source_support["interior"],
        dismount_support=source_support["dismount"],
        blend_frames=args.blend_frames,
    )
    resolved = resolve_stair_config(
        args.phase_root / "dataset",
        load_experiment_config(args.config),
        device="cpu",
    )
    extension = resolved.measurement_extension
    soles = MujocoG1SoleKinematics(args.g1_xml).sole_points(
        arrays["joint_position"],
        arrays["root_position_world"],
        arrays["root_orientation_world_wxyz"],
    )
    sole_scene_xy = extension.alignment.matcher_to_scene_xy(
        torch.tensor(soles[..., :2], dtype=torch.float32)
    )
    sole_surface = (
        extension.query_grid.sample_xy(sole_scene_xy).cpu().numpy()
    )
    sole_clearance = soles[..., 2] - sole_surface
    target_support, contact_metrics = _oriented_sole_contact_metrics(
        sole_clearance
    )
    metrics.update(contact_metrics)
    metrics.update(
        {
            "lane_id": args.lane_id,
            "lineage": lineage,
            "source_support_unsupported_frame_count": int(
                (~composed_source_support.any(axis=1)).sum()
            ),
            "contact_definition": (
                "minimum absolute oriented-sole sample clearance <= 0.03 m"
            ),
        }
    )
    metrics["accepted"] = _accepted_metrics(metrics)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "candidate.npz",
        **{name: arrays[name] for name in _CONNECTOR_KEYS},
    )
    np.savez(
        args.output / "contact-support.npz",
        target_support=target_support,
        source_support=composed_source_support,
    )
    if metrics["accepted"]:
        np.savez(
            args.output / "composite.npz",
            **{name: arrays[name] for name in _CONNECTOR_KEYS},
        )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
