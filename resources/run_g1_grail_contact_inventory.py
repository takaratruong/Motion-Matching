#!/usr/bin/env python3
"""Inventory GRAIL windows matching a sustained split-height gait."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import (
    ANKLE_ORIGIN_SOLE_M,
    STANCE_CLEARANCE_TOLERANCE_M,
    STANCE_VERTICAL_SPEED_MAX_MPS,
)
from mm_sonic.torch_grail_contact_window_search import (
    ContactWindow,
    contact_window_cost,
    contact_windows_from_source,
)


_FEET = (18, 19)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-split-height-m", type=float, default=0.174)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--maximum-results", type=int, default=200)
    parser.add_argument("--maximum-clips", type=int, default=0)
    return parser


def _sample_height_grid(
    *,
    origin_xy: np.ndarray,
    cell_size_m: float,
    height_z: np.ndarray,
    points_xy: np.ndarray,
) -> np.ndarray:
    coordinate = (points_xy.reshape(-1, 2) - origin_xy) / cell_size_m
    ny, nx = height_z.shape
    if (
        (coordinate[:, 0] < -1.0e-5).any()
        or (coordinate[:, 1] < -1.0e-5).any()
        or (coordinate[:, 0] > nx - 1 + 1.0e-5).any()
        or (coordinate[:, 1] > ny - 1 + 1.0e-5).any()
    ):
        raise ContractError("GRAIL foot query is outside terrain")
    x = np.clip(coordinate[:, 0], 0.0, nx - 1.0)
    y = np.clip(coordinate[:, 1], 0.0, ny - 1.0)
    ix = np.minimum(np.floor(x).astype(np.int64), nx - 2)
    iy = np.minimum(np.floor(y).astype(np.int64), ny - 2)
    tx = x - ix
    ty = y - iy
    h00 = height_z[iy, ix]
    h10 = height_z[iy, ix + 1]
    h01 = height_z[iy + 1, ix]
    h11 = height_z[iy + 1, ix + 1]
    first = h00 + tx * (h10 - h00) + ty * (h11 - h10)
    second = h00 + tx * (h11 - h01) + ty * (h01 - h00)
    sampled = np.where(tx >= ty, first, second)
    return sampled.reshape(points_xy.shape[:-1])


def _one_descriptor(
    arguments: tuple[str, dict, float],
) -> tuple[ContactWindow, ...]:
    root_text, descriptor, target_split_height_m = arguments
    root = Path(root_text)
    try:
        with np.load(
            root / descriptor["relative_motion_path"], allow_pickle=False
        ) as archive:
            joint_position = np.asarray(
                archive["joint_pos"], dtype=np.float64
            )
            joint_velocity = np.asarray(
                archive["joint_vel"], dtype=np.float64
            )
            body_position = np.asarray(
                archive["body_pos_w"], dtype=np.float64
            )
            body_quaternion = np.asarray(
                archive["body_quat_w"], dtype=np.float64
            )
            body_velocity = np.asarray(
                archive["body_lin_vel_w"], dtype=np.float64
            )
        with np.load(
            root / descriptor["terrain"]["path"], allow_pickle=False
        ) as archive:
            origin = np.asarray(archive["origin_xy"], dtype=np.float64)
            cell = float(np.asarray(archive["cell_size_m"]).reshape(-1)[0])
            height = np.asarray(archive["height_z"], dtype=np.float64)
        tx, ty, yaw = (
            float(value)
            for value in descriptor["terrain"][
                "motion_to_terrain_xy_yaw"
            ]
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotation = np.array(
            ((cosine, -sine), (sine, cosine)), dtype=np.float64
        )
        feet = body_position[:, _FEET]
        feet_scene = feet[..., :2] @ rotation.T + np.array((tx, ty))
        surface = _sample_height_grid(
            origin_xy=origin,
            cell_size_m=cell,
            height_z=height,
            points_xy=feet_scene,
        )
        support = (
            np.abs(
                feet[..., 2] - surface - float(ANKLE_ORIGIN_SOLE_M)
            )
            <= float(STANCE_CLEARANCE_TOLERANCE_M)
        ) & (
            np.abs(body_velocity[:, _FEET, 2])
            <= float(STANCE_VERTICAL_SPEED_MAX_MPS)
        )
        return contact_windows_from_source(
            source_clip=descriptor["logical_name"],
            joint_position=joint_position,
            root_position_world=body_position[:, 0],
            root_orientation_world_wxyz=body_quaternion[:, 0],
            generalized_velocity=joint_velocity,
            foot_position_world=feet,
            support_mask=np.asarray(support, dtype=np.bool_),
            foot_surface_height_m=surface,
            target_split_height_m=target_split_height_m,
        )
    except (ContractError, KeyError, OSError, ValueError):
        return ()


def _rank_windows(
    windows: Iterable[ContactWindow], *, maximum_results: int
) -> tuple[ContactWindow, ...]:
    owned = tuple(windows)
    if (
        type(maximum_results) is not int
        or maximum_results < 1
        or any(not isinstance(item, ContactWindow) for item in owned)
    ):
        raise ContractError("GRAIL contact inventory input is invalid")
    ranked = []
    for item in owned:
        try:
            cost = contact_window_cost(item)
        except ContractError:
            continue
        ranked.append((cost, item.window_id, item))
    if not ranked:
        raise ContractError("no GRAIL contact windows matched")
    return tuple(
        item
        for _, _, item in sorted(ranked, key=lambda row: (row[0], row[1]))[
            :maximum_results
        ]
    )


def main() -> int:
    args = _parser().parse_args()
    if (
        args.workers < 1
        or args.maximum_results < 1
        or args.maximum_clips < 0
    ):
        raise ContractError("GRAIL contact inventory options are invalid")
    manifest_path = args.source_dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    descriptors = [
        item
        for item in manifest["clips"]
        if item.get("kind") == "terrain"
        and str(item.get("logical_name", "")).startswith(
            ("grail-curb-", "grail-stair-", "grail-stair_p")
        )
    ]
    descriptors.sort(key=lambda item: item["logical_name"])
    if args.maximum_clips:
        descriptors = descriptors[: args.maximum_clips]
    jobs = (
        (
            str(args.source_dataset.resolve()),
            descriptor,
            float(args.target_split_height_m),
        )
        for descriptor in descriptors
    )
    windows: list[ContactWindow] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(_one_descriptor, jobs, chunksize=8):
            windows.extend(result)
    ranked_by_role = {}
    for role in ("uneven_walk", "exit"):
        try:
            ranked_by_role[role] = _rank_windows(
                (item for item in windows if item.role == role),
                maximum_results=args.maximum_results,
            )
        except ContractError as error:
            if "no GRAIL contact windows matched" not in str(error):
                raise
            ranked_by_role[role] = ()
    ranked = tuple(
        item
        for role in ("uneven_walk", "exit")
        for item in ranked_by_role[role]
    )
    payload = {
        "schema": "g1-grail-contact-inventory/v1",
        "source_manifest": str(manifest_path),
        "scanned_clip_count": len(descriptors),
        "matched_window_count": len(windows),
        "target_split_height_m": float(args.target_split_height_m),
        "windows_by_role": {
            role: [
                {
                    **asdict(item),
                    "intrinsic_cost": contact_window_cost(item),
                }
                for item in ranked_by_role[role]
            ]
            for role in ("uneven_walk", "exit")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "scanned": len(descriptors),
                "matched": len(windows),
                "retained": {
                    role: len(ranked_by_role[role])
                    for role in ("uneven_walk", "exit")
                },
                "best": {
                    role: (
                        ranked_by_role[role][0].window_id
                        if ranked_by_role[role]
                        else None
                    )
                    for role in ("uneven_walk", "exit")
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
