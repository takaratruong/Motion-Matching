#!/usr/bin/env python3
"""Optimize and save one terrain-conformal swing from a frozen route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--arrays", required=True)
    parser.add_argument("--foot", required=True, type=int, choices=(0, 1))
    parser.add_argument("--start-frame", required=True, type=int)
    parser.add_argument("--end-frame-exclusive", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--reference-weight", type=float, default=0.5)
    parser.add_argument("--smoothness-weight", type=float, default=2.0)
    parser.add_argument("--clearance-weight", type=float, default=500.0)
    parser.add_argument("--edge-weight", type=float, default=100.0)
    parser.add_argument("--project-landing", action="store_true")
    parser.add_argument("--foothold-search-radius", type=float, default=0.12)
    parser.add_argument("--foothold-search-step", type=float, default=0.01)
    parser.add_argument(
        "--maximum-foothold-height-range", type=float, default=0.025
    )
    return parser


def resolve_frame_interval(
    frame_count: int, start: int, end_exclusive: int
) -> tuple[int, int]:
    if (
        type(frame_count) is not int
        or type(start) is not int
        or type(end_exclusive) is not int
        or frame_count < 3
        or start < 0
        or end_exclusive > frame_count
        or end_exclusive - start < 3
    ):
        raise ValueError("terrain-conformal swing frame interval is invalid")
    return start, end_exclusive


def require_new_output(path: str | Path) -> Path:
    output = Path(os.path.abspath(os.fspath(path)))
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"terrain-conformal swing output already exists: {output}"
        )
    return output


def _cost_values(cost) -> dict[str, float]:
    return {
        name: float(getattr(cost, name)[0].detach().cpu().item())
        for name in (
            "reference",
            "smoothness",
            "clearance",
            "edge",
            "endpoint",
            "total",
        )
    }


def _quality(path, toe, heel, sample_surface) -> dict[str, float | int]:
    import torch

    zero = torch.zeros(
        (path.shape[0], 1), dtype=path.dtype, device=path.device
    )
    toe_xyz = torch.cat((toe, zero), dim=1)
    heel_xyz = torch.cat((heel, zero), dim=1)
    points = torch.stack((path, path + toe_xyz, path + heel_xyz), dim=1)
    surface = sample_surface(points[..., :2])
    clearance = points[..., 2] - surface
    acceleration = torch.diff(path, n=2, dim=0) * (50.0**2)
    return {
        "clearance_violation_count": int(
            (clearance < 0.025).sum().detach().cpu().item()
        ),
        "minimum_point_clearance_m": float(
            clearance.min().detach().cpu().item()
        ),
        "mean_midfoot_acceleration_m_s2": float(
            torch.linalg.vector_norm(acceleration, dim=1)
            .mean()
            .detach()
            .cpu()
            .item()
        ),
    }


def _render_path_plot(path: Path, raw, optimized, toe, heel, sample_surface) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    raw_surface = sample_surface(raw[:, :2]).detach().cpu().numpy()
    optimized_surface = sample_surface(optimized[:, :2]).detach().cpu().numpy()
    raw_numpy = raw.detach().cpu().numpy()
    optimized_numpy = optimized.detach().cpu().numpy()
    toe_numpy = toe.detach().cpu().numpy()
    heel_numpy = heel.detach().cpu().numpy()
    frames = list(range(raw.shape[0]))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frames, raw_numpy[:, 2], label="raw mid-foot", linewidth=2)
    axes[0].plot(
        frames, optimized_numpy[:, 2], label="optimized mid-foot", linewidth=2
    )
    axes[0].plot(frames, raw_surface + 0.025, "--", label="terrain + 2.5 cm")
    axes[0].plot(
        frames, optimized_surface + 0.025, ":", label="optimized terrain + margin"
    )
    axes[0].set_xlabel("route-frame offset")
    axes[0].set_ylabel("world z (m)")
    axes[0].set_title("Swing height")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(raw_numpy[:, 0], raw_numpy[:, 1], label="raw mid-foot")
    axes[1].plot(
        optimized_numpy[:, 0], optimized_numpy[:, 1], label="optimized mid-foot"
    )
    axes[1].plot(
        raw_numpy[:, 0] + toe_numpy[:, 0],
        raw_numpy[:, 1] + toe_numpy[:, 1],
        "--",
        label="raw toe",
    )
    axes[1].plot(
        raw_numpy[:, 0] + heel_numpy[:, 0],
        raw_numpy[:, 1] + heel_numpy[:, 1],
        ":",
        label="raw heel",
    )
    axes[1].axis("equal")
    axes[1].set_xlabel("matcher x (m)")
    axes[1].set_ylabel("matcher y (m)")
    axes[1].set_title("Planar foot geometry")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = require_new_output(args.output)

    import mujoco
    import numpy as np
    import torch

    from mm_sonic.torch_terrain_conformal_swing import (
        TerrainConformalSwingConfig,
        optimize_terrain_conformal_swing,
        project_landing_to_stable_foothold,
    )
    from mm_sonic.torch_terrain_live_viewer import (
        apply_kinematic_state,
        build_kinematic_scene,
    )
    from mm_sonic.torch_terrain_rollout import (
        load_experiment_config,
        resolve_stair_config,
    )

    arrays_path = Path(args.arrays).resolve()
    if not arrays_path.is_file() or arrays_path.is_symlink():
        raise ValueError("terrain-conformal swing route arrays are missing")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise ValueError("terrain-conformal swing route qpos is missing")
        qpos = np.asarray(archive["qpos"], np.float64)
    if qpos.ndim != 2 or qpos.shape[1:] != (36,) or not np.isfinite(qpos).all():
        raise ValueError("terrain-conformal swing route qpos is invalid")
    start, end = resolve_frame_interval(
        qpos.shape[0], args.start_frame, args.end_frame_exclusive
    )
    resolved = resolve_stair_config(
        args.dataset,
        load_experiment_config(args.config),
        device=args.device,
    )
    model, data = build_kinematic_scene(args.g1_xml, resolved)
    body_name = (
        "left_ankle_roll_link" if args.foot == 0 else "right_ankle_roll_link"
    )
    body_id = int(model.body(body_name).id)
    local = np.array(
        (
            (0.035, 0.0, -0.05),
            (0.120, 0.0, -0.05),
            (-0.050, 0.0, -0.05),
        ),
        np.float64,
    )
    world = []
    for frame in range(start, end):
        apply_kinematic_state(mujoco, model, data, qpos[frame])
        rotation = data.xmat[body_id].reshape(3, 3)
        world.append(data.xpos[body_id] + local @ rotation.T)
    world = np.asarray(world, np.float32)
    raw = torch.tensor(world[:, 0], dtype=torch.float32, device=resolved.device)
    toe = torch.tensor(
        world[:, 1, :2] - world[:, 0, :2],
        dtype=torch.float32,
        device=resolved.device,
    )
    heel = torch.tensor(
        world[:, 2, :2] - world[:, 0, :2],
        dtype=torch.float32,
        device=resolved.device,
    )

    def sample_surface(points_xy: torch.Tensor) -> torch.Tensor:
        scene = resolved.measurement_extension.alignment.matcher_to_scene_xy(
            points_xy
        )
        return resolved.measurement_extension.query_grid.sample_xy(scene)

    config = TerrainConformalSwingConfig(
        sample_count=512,
        iteration_count=16,
        reference_weight=args.reference_weight,
        smoothness_weight=args.smoothness_weight,
        clearance_weight=args.clearance_weight,
        edge_weight=args.edge_weight,
    )
    projection = None
    landing_target = None
    if args.project_landing:
        projection = project_landing_to_stable_foothold(
            raw[-1],
            toe_offset_xy=toe[-1],
            heel_offset_xy=heel[-1],
            sample_surface=sample_surface,
            search_radius_m=args.foothold_search_radius,
            search_step_m=args.foothold_search_step,
            edge_probe_m=config.edge_probe_m,
            maximum_height_range_m=args.maximum_foothold_height_range,
        )
        landing_target = projection.position_world
    result = optimize_terrain_conformal_swing(
        raw,
        torch.ones(raw.shape[0], dtype=torch.bool, device=raw.device),
        sample_surface=sample_surface,
        toe_offset_xy=toe,
        heel_offset_xy=heel,
        config=config,
        seed=args.seed,
        landing_target_world=landing_target,
    )
    expected_endpoints = raw[[0, -1]].clone()
    if landing_target is not None:
        expected_endpoints[-1] = landing_target
    endpoint_error = torch.linalg.vector_norm(
        result.path[[0, -1]] - expected_endpoints, dim=1
    ).max()
    payload = {
        "schema": "g1-terrain-conformal-swing-evidence/v1",
        "arrays": str(arrays_path),
        "foot": args.foot,
        "frame_interval": [start, end],
        "seed": args.seed,
        "weights": {
            "reference": args.reference_weight,
            "smoothness": args.smoothness_weight,
            "clearance": args.clearance_weight,
            "edge": args.edge_weight,
        },
        "improved": result.improved,
        "endpoint_error_m": float(endpoint_error.detach().cpu().item()),
        "landing_projection": (
            None
            if projection is None
            else {
                "original_world": raw[-1].detach().cpu().tolist(),
                "target_world": projection.position_world.detach().cpu().tolist(),
                "displacement_m": projection.displacement_m,
                "maximum_height_range_m": projection.maximum_height_range_m,
                "projected": projection.projected,
            }
        ),
        "raw_cost": _cost_values(result.raw_cost),
        "optimized_cost": _cost_values(result.optimized_cost),
        "raw_quality": _quality(raw, toe, heel, sample_surface),
        "optimized_quality": _quality(result.path, toe, heel, sample_surface),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["deterministic_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        (staging / "metrics.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            staging / "paths.npz",
            frame_index=np.arange(start, end, dtype=np.int64),
            raw_midfoot_world=raw.detach().cpu().numpy(),
            optimized_midfoot_world=result.path.detach().cpu().numpy(),
            toe_offset_world_xy=toe.detach().cpu().numpy(),
            heel_offset_world_xy=heel.detach().cpu().numpy(),
            landing_target_world=(
                raw[-1].detach().cpu().numpy()
                if landing_target is None
                else landing_target.detach().cpu().numpy()
            ),
        )
        _render_path_plot(
            staging / "trajectory.png",
            raw,
            result.path,
            toe,
            heel,
            sample_surface,
        )
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if result.improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
