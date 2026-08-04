#!/usr/bin/env python3
"""Extract a stair-relative contact graph from one validated route artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_stair_motion_graph import (
    merge_stair_motion_graphs,
)
from mm_sonic.torch_stair_motion_graph_extract import (
    aligned_source_support_mask,
    extract_stair_motion_graph,
    landing_decision_frames,
)
from mm_sonic.torch_terrain_features import TerrainDataset
from mm_sonic.torch_terrain_omni_routes import StairFrame


class _AppendPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None) or ()
        setattr(namespace, self.dest, (*current, Path(values)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--g1-xml", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--route-directory", type=Path)
    source.add_argument(
        "--matrix-directory",
        action=_AppendPathAction,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--origin-x-m", type=float, default=0.0)
    parser.add_argument("--origin-y-m", type=float, default=0.0)
    parser.add_argument("--ascent-yaw-rad", type=float, default=0.0)
    parser.add_argument("--width-m", type=float, default=0.6223)
    parser.add_argument("--tread-depth-m", type=float, default=0.3302)
    parser.add_argument("--riser-height-m", type=float, default=0.1778)
    parser.add_argument("--tread-count", type=int, default=3)
    parser.add_argument("--lateral-cell-m", type=float, default=0.10)
    return parser


def _load_metrics(route_directory: Path) -> dict:
    path = route_directory / "metrics.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ContractError(f"cannot read route metrics: {path}") from error
    if not isinstance(value, dict):
        raise ContractError("route metrics must be an object")
    metrics = value.get("metrics")
    penetration = (
        metrics.get("penetration_depth_m")
        if isinstance(metrics, dict)
        else None
    )
    support_error = (
        metrics.get("support_height_error_m")
        if isinstance(metrics, dict)
        else None
    )
    identity = value.get("deterministic_sha256")
    try:
        maximum_penetration = float(
            penetration.get("maximum", math.inf)
        )
        p95_support_error = float(
            support_error.get("p95", math.inf)
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ContractError(
            "route artifact has no contact-safe prefix"
        ) from error
    safe_prefix = (
        type(value.get("completed_frames")) is int
        and value["completed_frames"] > 0
        and isinstance(identity, str)
        and len(identity) == 64
        and isinstance(penetration, dict)
        and math.isfinite(maximum_penetration)
        and maximum_penetration <= 1e-6
        and isinstance(support_error, dict)
        and math.isfinite(p95_support_error)
        and p95_support_error <= 0.02
    )
    if not safe_prefix:
        raise ContractError(
            "route artifact has no contact-safe prefix"
        )
    return value


def _load_arrays(route_directory: Path) -> dict[str, np.ndarray]:
    path = route_directory / "arrays.npz"
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}
    except Exception as error:
        raise ContractError(f"cannot read route arrays: {path}") from error


def _sole_inputs(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joints = arrays.get("joint_position")
    roots = arrays.get("root_position_world")
    qpos = arrays.get("qpos")
    if (
        not isinstance(joints, np.ndarray)
        or joints.ndim != 2
        or joints.shape[1] != 29
        or len(joints) < 1
        or not np.isfinite(joints).all()
        or not isinstance(roots, np.ndarray)
        or roots.shape != (len(joints), 3)
        or not np.isfinite(roots).all()
        or not isinstance(qpos, np.ndarray)
        or qpos.ndim != 2
        or qpos.shape[0] != len(joints)
        or qpos.shape[1] < 7
        or not np.isfinite(qpos[:, 3:7]).all()
    ):
        raise ContractError("route sole kinematics arrays are invalid")
    return joints, roots, qpos[:, 3:7]


def main() -> int:
    args = _parser().parse_args()
    dataset = TerrainDataset.load(args.dataset, device=args.device)
    from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
    from mm_sonic.torch_terrain_rollout import (
        load_experiment_config,
        resolve_stair_config,
    )

    resolved = resolve_stair_config(
        args.dataset,
        load_experiment_config(args.config),
        device=args.device,
    )
    sole_kinematics = MujocoG1SoleKinematics(args.g1_xml)
    stair_frame = StairFrame(
        origin_world_xy=(args.origin_x_m, args.origin_y_m),
        ascent_world_yaw=args.ascent_yaw_rad,
        width_m=args.width_m,
        tread_depth_m=args.tread_depth_m,
        riser_height_m=args.riser_height_m,
        tread_count=args.tread_count,
    )
    if args.route_directory is not None:
        route_directories = (args.route_directory,)
    else:
        route_directories = tuple(
            route
            for matrix in args.matrix_directory
            for route in sorted((matrix / "routes").iterdir())
        )
    graphs = []
    accepted_sources = []
    rejected_sources = []
    for route_directory in route_directories:
        try:
            metrics = _load_metrics(route_directory)
            arrays = _load_arrays(route_directory)
            support = aligned_source_support_mask(
                dataset,
                arrays.get("selected_clip_path"),
                arrays.get("selected_source_frame"),
            )
            sole_clearance = None
            import torch

            joints, roots, orientations = _sole_inputs(arrays)
            sole_points = sole_kinematics.sole_points(
                joints,
                roots,
                orientations,
            )
            points = torch.as_tensor(
                sole_points[..., :2],
                dtype=torch.float32,
                device=resolved.device,
            )
            surface = (
                resolved.measurement_extension.query_grid.sample_xy(
                    resolved.measurement_extension.alignment.matcher_to_scene_xy(
                        points
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            sole_clearance = (
                sole_points[..., 2] - surface
            ).min(axis=(1, 2))
            graph = extract_stair_motion_graph(
                arrays,
                support_mask=support,
                stair_frame=stair_frame,
                exact_contact_valid=True,
                source_artifact_sha256=str(
                    metrics.get("deterministic_sha256")
                ),
                lateral_cell_m=args.lateral_cell_m,
                minimum_sole_clearance_by_frame=sole_clearance,
                minimum_sole_clearance_m=-0.025,
            )
        except ContractError as error:
            rejected_sources.append(
                {
                    "route_directory": str(route_directory),
                    "reason": str(error),
                }
            )
            continue
        graphs.append(graph)
        candidate_edge_count = max(
            0,
            len(landing_decision_frames(support)) - 1,
        )
        accepted_sources.append(
            {
                "route": metrics.get("route"),
                "route_deterministic_sha256": metrics.get(
                    "deterministic_sha256"
                ),
                "route_directory": str(route_directory.resolve()),
                "candidate_landing_edge_count": candidate_edge_count,
                "validated_edge_count": len(graph.edges),
                "rejected_edge_count": (
                    candidate_edge_count - len(graph.edges)
                ),
            }
        )
    if not graphs:
        raise ContractError("no exact-contact route artifacts were accepted")
    graph = merge_stair_motion_graphs(tuple(graphs))
    payload = {
        **graph.to_dict(),
        "source": {
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "stair_frame": {
                "origin_world_xy": list(stair_frame.origin_world_xy),
                "ascent_world_yaw": stair_frame.ascent_world_yaw,
                "width_m": stair_frame.width_m,
                "tread_depth_m": stair_frame.tread_depth_m,
                "riser_height_m": stair_frame.riser_height_m,
                "tread_count": stair_frame.tread_count,
                "lateral_cell_m": args.lateral_cell_m,
            },
            "accepted_routes": accepted_sources,
            "rejected_routes": rejected_sources,
            "minimum_sole_clearance_m": -0.025,
        },
        "coverage": graph.coverage_summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
