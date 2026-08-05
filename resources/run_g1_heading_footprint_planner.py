#!/usr/bin/env python3
"""Plan and validate one constant-heading G1 terrain traversal offline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import ContactSegmentIndex
from mm_sonic.torch_foothold_actions import (
    FootholdActionIndex,
    FootholdTransitionGraph,
)
from mm_sonic.torch_g1_fk import MujocoG1FootKinematics
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    NominalFootprintPath,
    heading_basis,
    nominal_footprint_path,
)
from mm_sonic.torch_heading_footprint_realizer import (
    RealizationFailure,
    realize_heading_footprint_plan,
)
from mm_sonic.torch_heading_footprint_search import (
    HeadingPlanningFailure,
    search_heading_footprint_plan,
)
from mm_sonic.torch_horizontal_terrain_retarget import (
    WideBoundG1TerrainRetargeter,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    FootprintCandidateFailure,
    terrain_footprint_layers,
)
from mm_sonic.torch_terrain_rollout import (
    load_experiment_config,
    resolve_stair_config,
)


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser()
    output.add_argument("--dataset", type=Path, required=True)
    output.add_argument("--config", type=Path, required=True)
    output.add_argument("--g1-xml", type=Path, required=True)
    output.add_argument("--start-state", type=Path, required=True)
    output.add_argument("--heading-degrees", type=float, required=True)
    output.add_argument("--distance-m", type=float, required=True)
    output.add_argument("--speed-mps", type=float, required=True)
    output.add_argument("--output", type=Path, required=True)
    output.add_argument("--device", default="cpu")
    output.add_argument("--forward-search-m", type=float, default=0.10)
    output.add_argument("--lateral-search-m", type=float, default=0.10)
    output.add_argument("--search-resolution-m", type=float, default=0.025)
    output.add_argument("--beam-width", type=int, default=16)
    output.add_argument("--xy-tolerance-m", type=float, default=0.15)
    output.add_argument("--height-tolerance-m", type=float, default=0.06)
    output.add_argument("--timing-tolerance-frames", type=int, default=15)
    output.add_argument(
        "--maximum-transition-position-error-rad",
        type=float,
        default=2.0,
    )
    output.add_argument(
        "--maximum-transition-velocity-error-rad-s",
        type=float,
        default=8.0,
    )
    output.add_argument("--skip-validation", action="store_true")
    return output


def failure_record(
    *,
    code: str,
    heading_degrees: float,
    step_index: int,
    attempted_footprints: int,
    attempted_actions: int,
    reasons: tuple[str, ...],
) -> dict[str, object]:
    if (
        not isinstance(code, str)
        or not code
        or not math.isfinite(float(heading_degrees))
        or type(step_index) is not int
        or type(attempted_footprints) is not int
        or attempted_footprints < 0
        or type(attempted_actions) is not int
        or attempted_actions < 0
        or not isinstance(reasons, tuple)
        or not reasons
    ):
        raise ValueError("invalid heading-footprint failure record")
    return {
        "schema": "g1-heading-footprint-failure/v1",
        "code": code,
        "heading_degrees": float(heading_degrees),
        "step_index": step_index,
        "attempted_footprints": attempted_footprints,
        "attempted_actions": attempted_actions,
        "reasons": list(reasons),
    }


def _rotate_xy(value: np.ndarray, yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.array(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    return np.asarray(value, dtype=np.float64) @ rotation.T


class _TerrainAdapter:
    def __init__(self, alignment: object, grid: object) -> None:
        self.alignment = alignment
        self.grid = grid

    def scene_to_world_xy(self, points: object) -> np.ndarray:
        scene = np.asarray(points, dtype=np.float64)
        translation = (
            self.alignment.translation_scene_xy.detach().cpu().numpy()
        )
        yaw = float(self.alignment.yaw_scene_from_matcher.item())
        return _rotate_xy(scene - translation, -yaw)

    def scene_heading_to_world(self, heading: object) -> np.ndarray:
        yaw = float(self.alignment.yaw_scene_from_matcher.item())
        return _rotate_xy(np.asarray(heading, dtype=np.float64), -yaw)

    def sample_surface(self, world_points: object) -> np.ndarray:
        world = np.asarray(world_points, dtype=np.float64)
        tensor = torch.tensor(
            world, dtype=torch.float32, device=self.grid.height_z.device
        )
        scene = self.alignment.matcher_to_scene_xy(tensor)
        return self.grid.sample_xy(scene).detach().cpu().numpy().astype(
            np.float64
        )


class _RealizerSource:
    def __init__(
        self,
        *,
        clips: object,
        root_body_index: int,
        foot_body_indices: tuple[int, int],
        action_index: FootholdActionIndex,
        contact_index: ContactSegmentIndex,
        start_root_position_world: np.ndarray,
    ) -> None:
        self.clips = clips
        self.root_body_index = root_body_index
        self.foot_body_indices = foot_body_indices
        self.action_index = action_index
        self.contact_index = contact_index
        self.start_root_position_world = start_root_position_world

    def support_mask(self, clip_index: int) -> torch.Tensor:
        return self.contact_index.support_mask(clip_index)


def _load_start_state(path: Path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            joints = np.asarray(archive["joint_position"], dtype=np.float64)
            roots = np.asarray(
                archive["root_position_world"], dtype=np.float64
            )
            quaternions = np.asarray(
                archive["root_orientation_world_wxyz"], dtype=np.float64
            )
            support = (
                np.asarray(archive["source_support_mask"])
                if "source_support_mask" in archive
                else None
            )
    except (OSError, KeyError, ValueError) as error:
        raise ContractError("start-state artifact is invalid") from error
    if joints.ndim == 2:
        joints = joints[-1]
    if roots.ndim == 2:
        roots = roots[-1]
    if quaternions.ndim == 2:
        quaternions = quaternions[-1]
    if support is not None and support.ndim == 2:
        support = support[-1]
    if (
        joints.shape != (29,)
        or roots.shape != (3,)
        or quaternions.shape != (4,)
        or not all(
            np.isfinite(value).all()
            for value in (joints, roots, quaternions)
        )
        or (
            support is not None
            and (
                support.shape != (2,)
                or support.dtype != np.bool_
                or not bool(support.any())
            )
        )
    ):
        raise ContractError("start-state artifact arrays are invalid")
    return joints, roots, quaternions, support


def _search_values(limit: float, resolution: float) -> tuple[float, ...]:
    if (
        not math.isfinite(limit)
        or not math.isfinite(resolution)
        or limit < 0.0
        or resolution <= 0.0
    ):
        raise ContractError("footprint search extent is invalid")
    count = int(math.floor(limit / resolution + 1.0e-9))
    return tuple(index * resolution for index in range(-count, count + 1))


def _motion_geometry(
    action_index: FootholdActionIndex,
) -> tuple[float, float]:
    actions = [
        action
        for action in action_index.actions
        if abs(float(action.root_yaw_delta_rad[-1].item())) <= math.pi / 6.0
    ]
    if not actions:
        raise ContractError("motion corpus has no constant-heading actions")
    stride = np.median(
        [
            abs(
                float(
                    action.landing_xy_start_frame_m[1, 0].item()
                    - action.landing_xy_start_frame_m[0, 0].item()
                )
            )
            for action in actions
        ]
    )
    width = np.median(
        [
            abs(
                float(
                    action.landing_xy_start_frame_m[0, 1].item()
                    - action.landing_xy_start_frame_m[1, 1].item()
                )
            )
            for action in actions
        ]
    )
    if not 0.10 <= stride <= 0.60 or not 0.10 <= width <= 0.50:
        raise ContractError("motion corpus gait geometry is implausible")
    return float(stride), float(width)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _footprint_records(plan) -> list[dict[str, object]]:
    return [
        {
            "step_index": item.step_index,
            "foot": item.foot,
            "center_scene_xy": [
                float(value) for value in item.center_scene_xy.tolist()
            ],
            "center_heading_xy": [
                float(value) for value in item.center_heading_xy.tolist()
            ],
            "surface_height_m": item.surface_height_m,
            "source_action_key": list(
                plan.action_keys[item.step_index // 2]
            ),
        }
        for item in plan.footprints
    ]


def run(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    action_count = 0
    attempted_footprints = 0
    try:
        if (
            not math.isfinite(args.heading_degrees)
            or not math.isfinite(args.distance_m)
            or not math.isfinite(args.speed_mps)
            or args.distance_m <= 0.0
            or args.speed_mps <= 0.0
            or args.beam_width < 1
        ):
            raise ContractError("planner command values are invalid")
        resolved = resolve_stair_config(
            args.dataset,
            load_experiment_config(args.config),
            device=args.device,
        )
        dataset = resolved.dataset
        extension = resolved.measurement_extension
        terrain = _TerrainAdapter(extension.alignment, extension.query_grid)
        contact_index = ContactSegmentIndex.from_dataset(
            dataset, maximum_action_frames=160
        )
        action_index = FootholdActionIndex.from_dataset(
            dataset, contact_index
        )
        action_count = len(action_index.actions)
        transition_graph = FootholdTransitionGraph.from_dataset(
            action_index, dataset
        )
        stride, width = _motion_geometry(action_index)
        joints, root, quaternion, support = _load_start_state(
            args.start_state
        )
        foot_kinematics = MujocoG1FootKinematics(args.g1_xml)
        feet_world = foot_kinematics.foot_positions(
            joints[None], root[None], quaternion[None]
        )[0]
        feet_scene = extension.alignment.matcher_to_scene_xy(
            torch.tensor(
                feet_world[:, :2],
                dtype=torch.float32,
                device=resolved.device,
            )
        )
        if support is None:
            clearance = (
                feet_world[:, 2]
                - terrain.sample_surface(feet_world[:, :2])
                - 0.035
            )
            support = np.abs(clearance) <= 0.025
        heading_radians = math.radians(args.heading_degrees)
        heading = torch.tensor(
            (math.cos(heading_radians), math.sin(heading_radians)),
            dtype=torch.float32,
            device=resolved.device,
        )
        path = nominal_footprint_path(
            ConstantHeadingRequest(
                start_foot_scene_xy=feet_scene,
                start_support=torch.tensor(
                    support, dtype=torch.bool, device=resolved.device
                ),
                heading_scene_xy=heading,
                distance_m=args.distance_m,
                speed_mps=args.speed_mps,
                stride_m=stride,
                step_width_m=width,
                frames_per_second=50.0,
            )
        )
        complete_count = len(path.footprints) // 2 * 2
        if complete_count < 2:
            raise ContractError(
                "requested distance contains no complete two-contact action"
            )
        if complete_count != len(path.footprints):
            path = NominalFootprintPath(
                origin_scene_xy=path.origin_scene_xy,
                heading_scene_xy=path.heading_scene_xy,
                footprints=path.footprints[:complete_count],
                progress_m=float(
                    path.footprints[complete_count - 1]
                    .center_heading_xy[0]
                    .item()
                ),
            )
        sole_world = MujocoG1SoleKinematics(args.g1_xml).sole_points(
            joints[None], root[None], quaternion[None]
        )[0]
        sole_scene = extension.alignment.matcher_to_scene_xy(
            torch.tensor(
                sole_world[..., :2],
                dtype=torch.float32,
                device=resolved.device,
            )
        )
        basis = heading_basis(heading)
        sole_templates = tuple(
            (sole_scene[foot] - feet_scene[foot]) @ basis
            for foot in range(2)
        )
        offsets_forward = _search_values(
            args.forward_search_m, args.search_resolution_m
        )
        offsets_lateral = _search_values(
            args.lateral_search_m, args.search_resolution_m
        )
        layers = terrain_footprint_layers(
            path=path,
            sole_offsets_by_foot=sole_templates,
            sample_surface=extension.query_grid.sample_xy,
            forward_offsets_m=offsets_forward,
            lateral_offsets_m=offsets_lateral,
            maximum_surface_variation_m=0.025,
            edge_safety_margin_m=0.01,
        )
        attempted_footprints = sum(
            len(layer.candidates) for layer in layers
        )
        start_surface = extension.query_grid.sample_xy(feet_scene)
        plan = search_heading_footprint_plan(
            layers=layers,
            heading_scene_xy=heading,
            action_index=action_index,
            transition_graph=transition_graph,
            beam_width=args.beam_width,
            xy_tolerance_m=args.xy_tolerance_m,
            height_tolerance_m=args.height_tolerance_m,
            timing_tolerance_frames=args.timing_tolerance_frames,
            maximum_yaw_delta_rad=math.pi / 6.0,
            maximum_transition_position_error_rad=(
                args.maximum_transition_position_error_rad
            ),
            maximum_transition_velocity_error_rad_s=(
                args.maximum_transition_velocity_error_rad_s
            ),
            start_foot_surface_height_m=start_surface,
            current_joint_position=torch.tensor(
                joints,
                dtype=torch.float32,
                device=resolved.device,
            ),
            current_joint_velocity=torch.zeros(
                29, dtype=torch.float32, device=resolved.device
            ),
        )
        source = _RealizerSource(
            clips=dataset.folder.clips,
            root_body_index=dataset.folder.layout.root_body_index,
            foot_body_indices=(
                dataset.folder.layout.left_foot_body_index,
                dataset.folder.layout.right_foot_body_index,
            ),
            action_index=action_index,
            contact_index=contact_index,
            start_root_position_world=root,
        )
        traversal = realize_heading_footprint_plan(
            plan=plan,
            source=source,
            terrain=terrain,
            kinematics=foot_kinematics,
            retargeter=WideBoundG1TerrainRetargeter(
                args.g1_xml,
                maximum_root_horizontal_deviation_m=0.05,
            ),
        )
        traversal_path = args.output / "traversal.npz"
        np.savez_compressed(
            traversal_path,
            joint_position=traversal.joint_position,
            root_position_world=traversal.root_position_world,
            root_orientation_world_wxyz=(
                traversal.root_orientation_world_wxyz
            ),
            source_support_mask=traversal.source_support_mask,
            source_frame_provenance=traversal.source_frame_provenance,
        )
        _write_json(args.output / "footprints.json", _footprint_records(plan))
        metrics = {
            "schema": "g1-heading-footprint-planner-metrics/v1",
            "heading_degrees": args.heading_degrees,
            "frame_count": len(traversal.joint_position),
            "action_count": len(plan.action_keys),
            "total_search_cost": plan.total_cost,
            "maximum_stance_error_m": (
                traversal.maximum_stance_error_m
            ),
            "minimum_sole_clearance_m": (
                traversal.minimum_sole_clearance_m
            ),
            "terminal_support": (
                traversal.source_support_mask[-1].tolist()
            ),
            "provenance_coverage": 1.0,
            "validated": bool(args.skip_validation),
        }
        _write_json(args.output / "metrics.json", metrics)
        if not args.skip_validation:
            validator = (
                Path(__file__).resolve().parent
                / "run_g1_validate_traversal.py"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    "--input",
                    str(traversal_path),
                    "--target-dataset",
                    str(args.dataset),
                    "--config",
                    str(args.config),
                    "--g1-xml",
                    str(args.g1_xml),
                    "--expected-heading-degrees",
                    str(args.heading_degrees),
                    "--planned-footprints",
                    str(args.output / "footprints.json"),
                    "--output",
                    str(args.output / "validation.json"),
                ],
                check=False,
            )
            if result.returncode != 0:
                raise ContractError(
                    f"independent validator exited {result.returncode}"
                )
            metrics["validated"] = True
            _write_json(args.output / "metrics.json", metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return 0
    except (
        FootprintCandidateFailure,
        HeadingPlanningFailure,
        RealizationFailure,
        ContractError,
    ) as error:
        code = getattr(error, "code", "validation_failed")
        step_index = getattr(
            error, "step_index", 2 * getattr(error, "edge_index", -1)
        )
        reasons = tuple(getattr(error, "reasons", (str(error),)))
        record = failure_record(
            code=code,
            heading_degrees=args.heading_degrees,
            step_index=int(step_index),
            attempted_footprints=attempted_footprints,
            attempted_actions=action_count,
            reasons=reasons,
        )
        _write_json(args.output / "failure.json", record)
        print(json.dumps(record, sort_keys=True), flush=True)
        return 2


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
