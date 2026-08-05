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

from mm_sonic.joints import (
    ContractError,
    PINNED_TARGET_TO_SOURCE_PERMUTATION,
)
from mm_sonic.torch_contact_segments import ContactSegmentIndex
from mm_sonic.torch_foothold_actions import (
    FootholdActionIndex,
    FootholdTransitionGraph,
)
from mm_sonic.torch_g1_fk import (
    MujocoG1FootKinematics,
    target_state_qpos,
)
from mm_sonic.torch_g1_sole_kinematics import MujocoG1SoleKinematics
from mm_sonic.torch_heading_footprint_path import (
    ConstantHeadingRequest,
    NominalFootprint,
    NominalFootprintPath,
    heading_basis,
    heading_local_to_scene,
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
from mm_sonic.torch_motionbricks_drop_fallback import (
    pad_motionbricks_exact_endpoints,
    project_motionbricks_flight_over_terrain,
    solve_motionbricks_landing_qpos,
)
from mm_sonic.torch_terrain_footprint_candidates import (
    FootprintCandidateFailure,
    TerrainFootprintLayer,
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
    output.add_argument(
        "--maximum-footprint-candidates", type=int, default=32
    )
    output.add_argument(
        "--edge-safety-margin-m", type=float, default=0.03
    )
    output.add_argument("--beam-width", type=int, default=16)
    output.add_argument("--xy-tolerance-m", type=float, default=0.15)
    output.add_argument("--height-tolerance-m", type=float, default=0.06)
    output.add_argument("--timing-tolerance-frames", type=int, default=40)
    output.add_argument(
        "--maximum-entry-foot-error-m", type=float, default=0.10
    )
    output.add_argument(
        "--allow-retargeted-entry",
        action="store_true",
        help="Let bounded realization bridge an authored start pose.",
    )
    output.add_argument(
        "--maximum-retarget-target-error-m",
        type=float,
        default=0.015,
    )
    output.add_argument(
        "--action-time-scale",
        type=float,
        default=1.0,
        help="Offline source-action duration multiplier at the fixed 50 Hz.",
    )
    output.add_argument("--motionbricks", type=Path)
    output.add_argument("--motionbricks-num-tokens", type=int, default=8)
    output.add_argument("--edge-approach-m", type=float, default=0.15)
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
    output.add_argument(
        "--forbid-action",
        action="append",
        default=[],
        help="Diagnostic clip:start action key to exclude from retrieval.",
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


def prune_footprint_layers(
    layers: tuple[TerrainFootprintLayer, ...],
    *,
    maximum_candidates: int,
) -> tuple[TerrainFootprintLayer, ...]:
    """Bound retrieval branching after complete-sole terrain filtering."""

    if (
        not isinstance(layers, tuple)
        or any(
            not isinstance(layer, TerrainFootprintLayer)
            for layer in layers
        )
        or type(maximum_candidates) is not int
        or maximum_candidates < 1
    ):
        raise ContractError("footprint candidate pruning input is invalid")
    output = []
    for layer in layers:
        strata: dict[int, list[object]] = {}
        for candidate in layer.candidates:
            level = round(candidate.surface_height_m / 0.025)
            strata.setdefault(level, []).append(candidate)
        ordered_strata = sorted(
            (
                sorted(
                    candidates,
                    key=lambda item: item.placement_cost,
                )
                for candidates in strata.values()
            ),
            key=lambda candidates: candidates[0].placement_cost,
        )
        selected = []
        rank = 0
        while len(selected) < maximum_candidates:
            added = False
            for candidates in ordered_strata:
                if rank < len(candidates):
                    selected.append(candidates[rank])
                    added = True
                    if len(selected) == maximum_candidates:
                        break
            if not added:
                break
            rank += 1
        output.append(
            TerrainFootprintLayer(
                step_index=layer.step_index,
                nominal=layer.nominal,
                candidates=tuple(selected),
            )
        )
    return tuple(output)


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
        start_foot_position_world: np.ndarray,
    ) -> None:
        self.clips = clips
        self.root_body_index = root_body_index
        self.foot_body_indices = foot_body_indices
        self.action_index = action_index
        self.contact_index = contact_index
        self.start_root_position_world = start_root_position_world
        self.start_foot_position_world = start_foot_position_world

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


def edge_approach_path(
    *,
    path: NominalFootprintPath,
    prefix_footprints: tuple[object, ...],
    advance_m: float,
    speed_mps: float,
    frames_per_second: float,
) -> NominalFootprintPath:
    """Insert one same-level contact per foot before a terminal drop."""

    if (
        not isinstance(path, NominalFootprintPath)
        or len(path.footprints) < 2
        or len(prefix_footprints) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (advance_m, speed_mps, frames_per_second)
        )
    ):
        raise ContractError("edge approach path input is invalid")
    last_by_foot = {}
    for footprint in prefix_footprints:
        foot = getattr(footprint, "foot", None)
        center = getattr(footprint, "center_heading_xy", None)
        if (
            foot not in (0, 1)
            or not isinstance(center, torch.Tensor)
            or tuple(center.shape) != (2,)
        ):
            raise ContractError(
                "edge approach prefix footprints are invalid"
            )
        last_by_foot[foot] = footprint
    if set(last_by_foot) != {0, 1}:
        raise ContractError(
            "edge approach needs a contact from each foot"
        )
    nominal_base = tuple(
        path.footprints[: len(prefix_footprints)]
    )
    if len(nominal_base) != len(prefix_footprints):
        raise ContractError("edge approach prefix exceeds nominal path")
    base = tuple(
        NominalFootprint(
            step_index=index,
            foot=footprint.foot,
            center_scene_xy=footprint.center_scene_xy,
            center_heading_xy=footprint.center_heading_xy,
            yaw_scene_rad=footprint.yaw_scene_rad,
            contact_frame=nominal_base[index].contact_frame,
        )
        for index, footprint in enumerate(prefix_footprints)
    )
    last_contact_frame = base[-1].contact_frame
    frame_increment = max(
        1,
        round(float(advance_m) / float(speed_mps) * frames_per_second),
    )
    yaw = math.atan2(
        float(path.heading_scene_xy[1].item()),
        float(path.heading_scene_xy[0].item()),
    )
    inserted = []
    next_feet = (
        base[-2].foot,
        base[-1].foot,
    )
    for offset, foot in enumerate(next_feet, start=1):
        previous = last_by_foot[foot]
        local = previous.center_heading_xy + torch.tensor(
            (float(advance_m), 0.0),
            dtype=previous.center_heading_xy.dtype,
            device=previous.center_heading_xy.device,
        )
        inserted.append(
            NominalFootprint(
                step_index=len(base) + offset - 1,
                foot=foot,
                center_scene_xy=heading_local_to_scene(
                    local,
                    path.origin_scene_xy,
                    path.heading_scene_xy,
                ),
                center_heading_xy=local,
                yaw_scene_rad=yaw,
                contact_frame=(
                    last_contact_frame + offset * frame_increment
                ),
            )
        )
    footprints = (*base, *inserted)
    return NominalFootprintPath(
        origin_scene_xy=path.origin_scene_xy,
        heading_scene_xy=path.heading_scene_xy,
        footprints=footprints,
        progress_m=float(
            max(
                item.center_heading_xy[0].item()
                for item in inserted
            )
        ),
    )


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


def _departure_action_index(
    action_index: FootholdActionIndex,
    contact_index: ContactSegmentIndex,
    required_swing_foot: int,
) -> FootholdActionIndex:
    """Keep double-support actions with four pre-landing swing frames."""

    if required_swing_foot not in (0, 1):
        raise ContractError("departure swing foot is invalid")
    actions = []
    for action in action_index.actions:
        if (
            action.start_support != (True, True)
            or action.landing_feet[0] != required_swing_foot
            or action.landing_frame_offsets[0] < 4
        ):
            continue
        support = contact_index.support_mask(action.clip_index)
        landing_frame = (
            action.start_frame + action.landing_frame_offsets[0]
        )
        context_start = landing_frame - 4
        context_support = support[context_start:landing_frame]
        if (
            context_start < action.start_frame
            or len(context_support) != 4
            or not bool(context_support.any(dim=1).all().item())
            or bool(context_support.all(dim=1).any().item())
            or not bool(
                (context_support == context_support[:1]).all().item()
            )
            or bool(context_support[0, required_swing_foot].item())
        ):
            continue
        actions.append(action)
    if not actions:
        raise ContractError(
            "motion corpus has no authenticated departure actions"
        )
    selected = tuple(actions)
    return FootholdActionIndex(
        actions=selected,
        _entries={
            (action.clip_index, action.start_frame): action
            for action in selected
        },
    )


def terminal_swing_relative_heading(
    *,
    body_position_world: object,
    body_quaternion_world_wxyz: object,
    start_frame: int,
    terminal_frame: int,
    root_body_index: int,
    foot_body_indices: tuple[int, int],
    swing_foot: int,
) -> np.ndarray:
    """Measure the pre-touchdown swing pose in the source heading frame."""

    position = np.asarray(body_position_world, dtype=np.float64)
    quaternion = np.asarray(
        body_quaternion_world_wxyz, dtype=np.float64
    )
    if (
        position.ndim != 3
        or position.shape[-1] != 3
        or quaternion.shape != (*position.shape[:2], 4)
        or not 0 <= start_frame < len(position)
        or not 0 <= terminal_frame < len(position)
        or root_body_index not in range(position.shape[1])
        or len(foot_body_indices) != 2
        or any(index not in range(position.shape[1]) for index in foot_body_indices)
        or swing_foot not in (0, 1)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ContractError("departure swing source geometry is invalid")
    root_quaternion = quaternion[start_frame, root_body_index]
    w, x, y, z = root_quaternion
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    stance_foot = 1 - swing_foot
    relative = (
        position[terminal_frame, foot_body_indices[swing_foot]]
        - position[terminal_frame, foot_body_indices[stance_foot]]
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local_xy = np.array(
        (
            cosine * relative[0] + sine * relative[1],
            -sine * relative[0] + cosine * relative[1],
        ),
        dtype=np.float64,
    )
    return np.ascontiguousarray(
        np.array((local_xy[0], local_xy[1], relative[2]))
    )


def _select_departure_action(
    *,
    plan: object,
    action_index: FootholdActionIndex,
    contact_index: ContactSegmentIndex,
    transition_graph: FootholdTransitionGraph,
    clips: tuple[object, ...],
    root_body_index: int,
    foot_body_indices: tuple[int, int],
    desired_terminal_swing_relative_heading_xyz: np.ndarray,
    drop_footprints: tuple[object, object],
    required_swing_foot: int,
    maximum_position_error_rad: float,
    maximum_velocity_error_rad_s: float,
) -> tuple[int, int]:
    inventory = _departure_action_index(
        action_index, contact_index, required_swing_foot
    )
    graph_rows = {
        key: index
        for index, key in enumerate(transition_graph.action_keys)
    }
    previous_key = plan.action_keys[-1]
    previous_row = graph_rows[previous_key]
    current_root = torch.zeros_like(
        inventory.actions[0].root_displacement_m[-1]
    )
    current_surface = {}
    for edge_key in plan.action_keys:
        action = action_index.entry(*edge_key)
        assert action is not None
        current_root = current_root + action.root_displacement_m[-1]
    for footprint in plan.footprints:
        current_surface[footprint.foot] = footprint.surface_height_m
    target = next(
        footprint
        for footprint in drop_footprints
        if footprint.foot == required_swing_foot
    )
    desired_terminal = np.asarray(
        desired_terminal_swing_relative_heading_xyz,
        dtype=np.float64,
    )
    if desired_terminal.shape != (3,) or not np.isfinite(
        desired_terminal
    ).all():
        raise ContractError("departure swing target geometry is invalid")
    desired_xy = target.center_heading_xy - current_root
    desired_height = (
        target.surface_height_m
        - current_surface[required_swing_foot]
    )
    ranked = []
    for action in inventory.actions:
        key = (action.clip_index, action.start_frame)
        row = graph_rows[key]
        position = float(
            torch.linalg.vector_norm(
                transition_graph.entry_joint_position[row]
                - transition_graph.terminal_joint_position[
                    previous_row
                ]
            ).item()
        )
        velocity = float(
            torch.linalg.vector_norm(
                transition_graph.entry_joint_velocity[row]
                - transition_graph.terminal_joint_velocity[
                    previous_row
                ]
            ).item()
        )
        if (
            position > maximum_position_error_rad
            or velocity > maximum_velocity_error_rad_s
        ):
            continue
        xy = float(
            torch.linalg.vector_norm(
                action.landing_xy_start_frame_m[0] - desired_xy
            ).item()
        )
        height = abs(
            float(action.landing_height_delta_m[0].item())
            - desired_height
        )
        yaw = abs(float(action.root_yaw_delta_rad[0].item()))
        terminal_frame = (
            action.start_frame + action.landing_frame_offsets[0] - 1
        )
        clip = clips[action.clip_index]
        terminal_relative = terminal_swing_relative_heading(
            body_position_world=clip.body_position_world,
            body_quaternion_world_wxyz=(
                clip.body_quaternion_world_wxyz
            ),
            start_frame=action.start_frame,
            terminal_frame=terminal_frame,
            root_body_index=root_body_index,
            foot_body_indices=foot_body_indices,
            swing_foot=required_swing_foot,
        )
        terminal_geometry = float(
            np.sum(
                (
                    (terminal_relative - desired_terminal)
                    / np.array((0.15, 0.10, 0.15))
                )
                ** 2
            )
        )
        score = (
            (position / maximum_position_error_rad) ** 2
            + (velocity / maximum_velocity_error_rad_s) ** 2
            + (xy / 0.25) ** 2
            + (height / 0.30) ** 2
            + (yaw / (math.pi / 6.0)) ** 2
            + terminal_geometry
        )
        ranked.append((score, key))
    if not ranked:
        raise ContractError(
            "no pose-compatible authenticated departure action"
        )
    ranked.sort()
    return ranked[0][1]


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


def action_has_exact_declared_contacts(
    action: object, support_mask: object
) -> bool:
    """Reject action windows containing landings absent from the descriptor."""

    support = np.asarray(support_mask)
    try:
        start = int(action.start_frame)
        end = int(action.end_frame)
        expected = tuple(
            (int(offset), int(foot))
            for offset, foot in zip(
                action.landing_frame_offsets,
                action.landing_feet,
            )
        )
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        support.ndim != 2
        or support.shape[1] != 2
        or support.dtype != np.bool_
        or not 0 <= start < end <= len(support)
        or len(expected) != 2
    ):
        return False
    window = support[start:end]
    if not bool(window[-1].all()):
        return False
    onset = (~window[:-1]) & window[1:]
    observed = tuple(
        (int(frame) + 1, int(foot))
        for frame, foot in np.argwhere(onset)
    )
    return observed == expected


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
            or not math.isfinite(args.maximum_entry_foot_error_m)
            or args.maximum_entry_foot_error_m <= 0.0
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
        forbidden_actions = set()
        for encoded in args.forbid_action:
            try:
                clip_text, frame_text = encoded.split(":", maxsplit=1)
                key = (int(clip_text), int(frame_text))
            except (AttributeError, TypeError, ValueError) as error:
                raise ContractError(
                    "forbidden action must use clip:start"
                ) from error
            if min(key) < 0:
                raise ContractError(
                    "forbidden action must use clip:start"
                )
            forbidden_actions.add(key)
        if forbidden_actions:
            allowed_actions = tuple(
                action
                for action in action_index.actions
                if (action.clip_index, action.start_frame)
                not in forbidden_actions
            )
            action_index = FootholdActionIndex(
                actions=allowed_actions,
                _entries={
                    (action.clip_index, action.start_frame): action
                    for action in allowed_actions
                },
            )
        faithful_actions = tuple(
            action
            for action in action_index.actions
            if action_has_exact_declared_contacts(
                action,
                np.asarray(
                    contact_index.support_mask(action.clip_index)
                    .detach()
                    .cpu(),
                    dtype=np.bool_,
                ),
            )
        )
        if not faithful_actions:
            raise ContractError(
                "motion corpus has no exact two-contact actions"
            )
        action_index = FootholdActionIndex(
            actions=faithful_actions,
            _entries={
                (action.clip_index, action.start_frame): action
                for action in faithful_actions
            },
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
        heading_world = terrain.scene_heading_to_world(
            heading.detach().cpu().numpy()
        )
        lateral_world = np.array(
            (-heading_world[1], heading_world[0]), dtype=np.float64
        )
        current_relative_world = feet_world[1] - feet_world[0]
        current_relative_heading = np.array(
            (
                float(current_relative_world[:2] @ heading_world),
                float(current_relative_world[:2] @ lateral_world),
                float(current_relative_world[2]),
            ),
            dtype=np.float64,
        )
        entry_geometry_values = []
        foot_body_indices = (
            dataset.folder.layout.left_foot_body_index,
            dataset.folder.layout.right_foot_body_index,
        )
        for action in action_index.actions:
            clip = dataset.folder.clips[action.clip_index]
            source_relative_heading = (
                terminal_swing_relative_heading(
                    body_position_world=clip.body_position_world,
                    body_quaternion_world_wxyz=(
                        clip.body_quaternion_world_wxyz
                    ),
                    start_frame=action.start_frame,
                    terminal_frame=action.start_frame,
                    root_body_index=(
                        dataset.folder.layout.root_body_index
                    ),
                    foot_body_indices=foot_body_indices,
                    swing_foot=1,
                )
            )
            per_foot_error = 0.5 * float(
                np.linalg.norm(
                    source_relative_heading
                    - current_relative_heading
                )
            )
            entry_geometry_values.append(
                math.inf
                if per_foot_error > args.maximum_entry_foot_error_m
                else (
                    per_foot_error
                    / args.maximum_entry_foot_error_m
                )
                ** 2
            )
        entry_geometry_cost = torch.tensor(
            entry_geometry_values,
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
            edge_safety_margin_m=args.edge_safety_margin_m,
        )
        attempted_footprints = sum(
            len(layer.candidates) for layer in layers
        )
        layers = prune_footprint_layers(
            layers,
            maximum_candidates=args.maximum_footprint_candidates,
        )
        start_surface = extension.query_grid.sample_xy(feet_scene)
        def search(candidate_layers, inventory=action_index):
            return search_heading_footprint_plan(
                layers=candidate_layers,
                heading_scene_xy=heading,
                action_index=inventory,
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
                current_joint_position=(
                    None
                    if args.allow_retargeted_entry
                    else torch.tensor(
                        joints,
                        dtype=torch.float32,
                        device=resolved.device,
                    )
                ),
                current_joint_velocity=(
                    None
                    if args.allow_retargeted_entry
                    else torch.zeros(
                        29,
                        dtype=torch.float32,
                        device=resolved.device,
                    )
                ),
                entry_action_geometry_cost=entry_geometry_cost,
            )

        drop_footprints = ()
        departure_action_key = None
        departure_swing_target_xy = None
        try:
            plan = search(layers)
        except HeadingPlanningFailure as search_failure:
            fallback_step = search_failure.step_index
            if (
                args.motionbricks is None
                or search_failure.code != "no_motion_coverage"
                or fallback_step < 2
                or fallback_step % 2
                or fallback_step + 2 != len(layers)
            ):
                raise
            prefix_plan = search(layers[:fallback_step])
            approach_path = edge_approach_path(
                path=path,
                prefix_footprints=prefix_plan.footprints,
                advance_m=args.edge_approach_m,
                speed_mps=args.speed_mps,
                frames_per_second=50.0,
            )
            approach_extent = min(
                args.edge_approach_m,
                args.forward_search_m,
                args.lateral_search_m,
            )
            approach_offsets = _search_values(
                approach_extent, args.search_resolution_m
            )
            approach_layers = terrain_footprint_layers(
                path=approach_path,
                sole_offsets_by_foot=sole_templates,
                sample_surface=extension.query_grid.sample_xy,
                forward_offsets_m=approach_offsets,
                lateral_offsets_m=approach_offsets,
                maximum_surface_variation_m=0.025,
                edge_safety_margin_m=args.edge_safety_margin_m,
            )
            approach_layers = (
                *tuple(
                    TerrainFootprintLayer(
                        step_index=index,
                        nominal=approach_layers[index].nominal,
                        candidates=(prefix_plan.footprints[index],),
                    )
                    for index in range(fallback_step)
                ),
                *approach_layers[fallback_step:],
            )
            approach_layers = prune_footprint_layers(
                tuple(approach_layers),
                maximum_candidates=args.maximum_footprint_candidates,
            )
            required_swing_foot = max(
                (
                    layers[fallback_step].nominal,
                    layers[fallback_step + 1].nominal,
                ),
                key=lambda item: float(
                    item.center_heading_xy[0].item()
                ),
            ).foot
            plan = search(approach_layers)
            drop_footprints = (
                layers[fallback_step].candidates[0],
                layers[fallback_step + 1].candidates[0],
            )
            swing_footprint = next(
                footprint
                for footprint in drop_footprints
                if footprint.foot == required_swing_foot
            )
            stance_footprint = next(
                footprint
                for footprint in reversed(plan.footprints)
                if footprint.foot != required_swing_foot
            )
            departure_swing_target_xy = (
                terrain.scene_to_world_xy(
                    swing_footprint.center_scene_xy.detach()
                    .cpu()
                    .numpy()
                )
                - 0.12
                * terrain.scene_heading_to_world(
                    heading.detach().cpu().numpy()
                )
            )
            departure_surface = float(
                terrain.sample_surface(
                    departure_swing_target_xy[None]
                )[0]
            )
            desired_terminal_swing = np.array(
                (
                    float(
                        swing_footprint.center_heading_xy[0].item()
                        - 0.12
                        - stance_footprint.center_heading_xy[0].item()
                    ),
                    float(
                        swing_footprint.center_heading_xy[1].item()
                        - stance_footprint.center_heading_xy[1].item()
                    ),
                    departure_surface
                    + 0.24
                    - stance_footprint.surface_height_m,
                ),
                dtype=np.float64,
            )
            departure_action_key = _select_departure_action(
                plan=plan,
                action_index=action_index,
                contact_index=contact_index,
                transition_graph=transition_graph,
                clips=dataset.folder.clips,
                root_body_index=dataset.folder.layout.root_body_index,
                foot_body_indices=(
                    dataset.folder.layout.left_foot_body_index,
                    dataset.folder.layout.right_foot_body_index,
                ),
                desired_terminal_swing_relative_heading_xyz=(
                    desired_terminal_swing
                ),
                drop_footprints=drop_footprints,
                required_swing_foot=required_swing_foot,
                maximum_position_error_rad=(
                    args.maximum_transition_position_error_rad
                ),
                maximum_velocity_error_rad_s=(
                    args.maximum_transition_velocity_error_rad_s
                ),
            )
        footprint_records = _footprint_records(plan)
        for footprint in drop_footprints:
            footprint_records.append(
                {
                    "step_index": footprint.step_index,
                    "foot": footprint.foot,
                    "center_scene_xy": [
                        float(value)
                        for value in footprint.center_scene_xy.tolist()
                    ],
                    "center_heading_xy": [
                        float(value)
                        for value in footprint.center_heading_xy.tolist()
                    ],
                    "surface_height_m": footprint.surface_height_m,
                    "source_action_key": None,
                    "source_primitive": "motionbricks-conditioned",
                }
            )
        _write_json(
            args.output / "plan.json",
            {
                "schema": "g1-heading-footprint-plan/v1",
                "heading_degrees": args.heading_degrees,
                "stride_m": stride,
                "step_width_m": width,
                "action_keys": [list(key) for key in plan.action_keys],
                "action_landing_frame_offsets": [
                    list(
                        action_index.entry(*key).landing_frame_offsets
                    )
                    for key in plan.action_keys
                ],
                "action_landing_xy_start_frame_m": [
                    action_index.entry(*key)
                    .landing_xy_start_frame_m.detach()
                    .cpu()
                    .tolist()
                    for key in plan.action_keys
                ],
                "action_landing_height_delta_m": [
                    action_index.entry(*key)
                    .landing_height_delta_m.detach()
                    .cpu()
                    .tolist()
                    for key in plan.action_keys
                ],
                "action_root_displacement_m": [
                    action_index.entry(*key)
                    .root_displacement_m.detach()
                    .cpu()
                    .tolist()
                    for key in plan.action_keys
                ],
                "nominal_contact_frames": [
                    layer.nominal.contact_frame for layer in layers
                ],
                "footprints": footprint_records,
                "total_cost": plan.total_cost,
                "terminal_primitive": (
                    "motionbricks-conditioned"
                    if drop_footprints
                    else None
                ),
                "departure_action_key": (
                    list(departure_action_key)
                    if departure_action_key is not None
                    else None
                ),
            },
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
            start_foot_position_world=feet_world,
        )
        traversal = realize_heading_footprint_plan(
            plan=plan,
            source=source,
            terrain=terrain,
            kinematics=foot_kinematics,
            departure_action_key=departure_action_key,
            departure_swing_target_xy_world=(
                departure_swing_target_xy
            ),
            ankle_origin_sole_m=0.05,
            maximum_stance_error_m=(
                args.maximum_retarget_target_error_m
            ),
            swing_sample_uncertainty_m=args.edge_safety_margin_m,
            action_time_scale=args.action_time_scale,
            retargeter=WideBoundG1TerrainRetargeter(
                args.g1_xml,
                maximum_root_horizontal_deviation_m=0.05,
                maximum_target_error_m=(
                    args.maximum_retarget_target_error_m
                ),
                maximum_warm_joint_step_rad=0.30,
                foot_position_scale=200.0,
                foot_orientation_scale=60.0,
            ),
        )
        output_joints = traversal.joint_position
        output_root = traversal.root_position_world
        output_quaternion = traversal.root_orientation_world_wxyz
        output_support = traversal.source_support_mask
        output_provenance = traversal.source_frame_provenance
        selected_motionbricks_tokens = None
        if drop_footprints:
            if (
                args.motionbricks is None
                or not (args.motionbricks / "motionbricks").is_dir()
            ):
                raise ContractError(
                    "MotionBricks checkout is missing"
                )
            target_feet = np.empty((2, 3), dtype=np.float64)
            for footprint in drop_footprints:
                target_feet[footprint.foot, :2] = (
                    terrain.scene_to_world_xy(
                        footprint.center_scene_xy.detach().cpu().numpy()
                    )
                )
                target_feet[footprint.foot, 2] = (
                    footprint.surface_height_m + 0.05
                )
            resource_directory = str(Path(__file__).resolve().parent)
            if resource_directory not in sys.path:
                sys.path.insert(0, resource_directory)
            from run_g1_motionbricks_contact_exit import (
                _load_motionbricks_agent,
                endpoint_warp_and_resample,
                generate_route_conditioned_qpos,
            )

            agent = _load_motionbricks_agent(args.motionbricks.resolve())
            released_landing = agent._clip_holder.mujoco_qpos[0, 0]
            if isinstance(released_landing, torch.Tensor):
                released_landing = (
                    released_landing.detach().cpu().numpy()
                )
            heading_world = terrain.scene_heading_to_world(
                heading.detach().cpu().numpy()
            )
            landing_qpos = solve_motionbricks_landing_qpos(
                reference_qpos=released_landing,
                target_foot_position_world=target_feet,
                target_heading_world_xy=heading_world,
                kinematics=foot_kinematics,
                retargeter=WideBoundG1TerrainRetargeter(
                    args.g1_xml,
                    maximum_root_horizontal_deviation_m=0.20,
                    maximum_root_height_deviation_m=0.30,
                    maximum_target_error_m=0.040,
                ),
            )
            prefix_qpos = np.stack(
                [
                    target_state_qpos(joint, root_item, quaternion_item)
                    for joint, root_item, quaternion_item in zip(
                        output_joints[-4:],
                        output_root[-4:],
                        output_quaternion[-4:],
                    )
                ]
            )
            target_qpos = np.repeat(landing_qpos[None], 4, axis=0)
            minimum_tokens = int(
                agent._inferencer._args["min_tokens"]
            )
            maximum_tokens = int(
                agent._inferencer._args["max_tokens"]
            )
            token_records = []
            candidates = []
            for token_count in range(
                minimum_tokens, maximum_tokens + 1
            ):
                try:
                    candidate_generated = (
                        generate_route_conditioned_qpos(
                            agent,
                            context_qpos=prefix_qpos,
                            target_qpos=target_qpos,
                            token_count=token_count,
                        )
                    )
                    transition = candidate_generated[
                        3 : len(candidate_generated) - 3
                    ]
                    raw_aligned = endpoint_warp_and_resample(
                        transition,
                        exact_start_qpos=prefix_qpos[-1],
                        exact_stop_qpos=landing_qpos,
                        source_frames_per_second=30.0,
                        target_frames_per_second=50.0,
                    )
                    raw_aligned = pad_motionbricks_exact_endpoints(
                        raw_aligned, hold_frames=32
                    )
                    raw_joints = raw_aligned[
                        :,
                        7
                        + np.asarray(
                            PINNED_TARGET_TO_SOURCE_PERMUTATION
                        ),
                    ]
                    raw_sole = MujocoG1SoleKinematics(
                        args.g1_xml
                    ).sole_points(
                        raw_joints,
                        raw_aligned[:, :3],
                        raw_aligned[:, 3:7],
                    )
                    raw_clearance = float(
                        (
                            raw_sole[..., 2]
                            - terrain.sample_surface(
                                raw_sole[..., :2]
                            )
                        ).min()
                    )
                    candidate_aligned = (
                        project_motionbricks_flight_over_terrain(
                            qpos=raw_aligned,
                            sample_surface=terrain.sample_surface,
                            kinematics=foot_kinematics,
                            sole_kinematics=MujocoG1SoleKinematics(
                                args.g1_xml
                            ),
                            retargeter=WideBoundG1TerrainRetargeter(
                                args.g1_xml,
                                maximum_root_horizontal_deviation_m=0.05,
                                maximum_root_height_deviation_m=0.15,
                                maximum_target_error_m=0.015,
                            ),
                        )
                    )
                    candidate_joints = candidate_aligned[
                        :,
                        7
                        + np.asarray(
                            PINNED_TARGET_TO_SOURCE_PERMUTATION
                        ),
                    ]
                    candidate_sole = MujocoG1SoleKinematics(
                        args.g1_xml
                    ).sole_points(
                        candidate_joints,
                        candidate_aligned[:, :3],
                        candidate_aligned[:, 3:7],
                    )
                    minimum_clearance = float(
                        (
                            candidate_sole[..., 2]
                            - terrain.sample_surface(
                                candidate_sole[..., :2]
                            )
                        ).min()
                    )
                    maximum_joint_step = float(
                        np.abs(np.diff(candidate_joints, axis=0)).max()
                    )
                    maximum_root_step = float(
                        np.linalg.norm(
                            np.diff(candidate_aligned[:, :3], axis=0),
                            axis=1,
                        ).max()
                    )
                    accepted = (
                        minimum_clearance >= -0.025
                        and maximum_joint_step <= 0.30
                        and maximum_root_step <= 0.05
                    )
                    record = {
                        "token_count": token_count,
                        "frame_count": len(candidate_aligned),
                        "raw_minimum_sole_clearance_m": raw_clearance,
                        "minimum_sole_clearance_m": minimum_clearance,
                        "maximum_joint_step_rad": maximum_joint_step,
                        "maximum_root_step_m": maximum_root_step,
                        "accepted": accepted,
                    }
                    token_records.append(record)
                    candidates.append(
                        (
                            (
                                not accepted,
                                abs(
                                    token_count
                                    - args.motionbricks_num_tokens
                                ),
                                -raw_clearance,
                                maximum_joint_step,
                                maximum_root_step,
                            ),
                            token_count,
                            candidate_generated,
                            candidate_aligned,
                        )
                    )
                except (ContractError, ValueError, RuntimeError) as error:
                    token_records.append(
                        {
                            "token_count": token_count,
                            "accepted": False,
                            "failure": str(error),
                        }
                    )
            if not candidates:
                _write_json(
                    args.output / "motionbricks-token-sweep.json",
                    {
                        "schema": "g1-motionbricks-token-sweep/v1",
                        "selected_token_count": None,
                        "candidates": token_records,
                    },
                )
                raise ContractError(
                    "all MotionBricks terrain candidates failed"
                )
            (
                _,
                selected_motionbricks_tokens,
                generated,
                aligned,
            ) = min(candidates, key=lambda item: item[0])
            _write_json(
                args.output / "motionbricks-token-sweep.json",
                {
                    "schema": "g1-motionbricks-token-sweep/v1",
                    "selected_token_count": selected_motionbricks_tokens,
                    "candidates": token_records,
                },
            )
            generated_joints = aligned[
                1:, 7 + np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
            ]
            generated_sole = MujocoG1SoleKinematics(
                args.g1_xml
            ).sole_points(
                generated_joints,
                aligned[1:, :3],
                aligned[1:, 3:7],
            )
            generated_clearance = (
                generated_sole[..., 2]
                - terrain.sample_surface(generated_sole[..., :2])
            )
            generated_support = (
                np.abs(generated_clearance) <= 0.025
            ).all(axis=2)
            output_joints = np.concatenate(
                (output_joints, generated_joints), axis=0
            )
            output_root = np.concatenate(
                (output_root, aligned[1:, :3]), axis=0
            )
            output_quaternion = np.concatenate(
                (output_quaternion, aligned[1:, 3:7]),
                axis=0,
            )
            output_support = np.concatenate(
                (output_support, generated_support), axis=0
            )
            generated_provenance = np.stack(
                (
                    np.full(len(aligned) - 1, -1, dtype=np.int64),
                    np.arange(1, len(aligned), dtype=np.int64),
                ),
                axis=1,
            )
            output_provenance = np.concatenate(
                (output_provenance, generated_provenance), axis=0
            )
            np.save(args.output / "motionbricks-generated-qpos.npy", generated)
            np.save(args.output / "motionbricks-landing-qpos.npy", target_qpos)
        sole_output = MujocoG1SoleKinematics(args.g1_xml).sole_points(
            output_joints, output_root, output_quaternion
        )
        sole_surface = terrain.sample_surface(sole_output[..., :2])
        minimum_sole_clearance = float(
            (sole_output[..., 2] - sole_surface).min()
        )
        unsupported_frames = int((~output_support.any(axis=1)).sum())
        traversal_path = args.output / "traversal.npz"
        np.savez_compressed(
            traversal_path,
            joint_position=output_joints,
            root_position_world=output_root,
            root_orientation_world_wxyz=output_quaternion,
            source_support_mask=output_support,
            source_frame_provenance=output_provenance,
        )
        _write_json(args.output / "footprints.json", footprint_records)
        metrics = {
            "schema": "g1-heading-footprint-planner-metrics/v1",
            "heading_degrees": args.heading_degrees,
            "frame_count": len(output_joints),
            "action_count": (
                len(plan.action_keys)
                + int(departure_action_key is not None)
                + int(bool(drop_footprints))
            ),
            "total_search_cost": plan.total_cost,
            "maximum_stance_error_m": (
                traversal.maximum_stance_error_m
            ),
            "minimum_sole_clearance_m": minimum_sole_clearance,
            "unsupported_frame_count": unsupported_frames,
            "terminal_support": output_support[-1].tolist(),
            "terminal_primitive": (
                "motionbricks-conditioned" if drop_footprints else None
            ),
            "motionbricks_token_count": selected_motionbricks_tokens,
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
                    "--maximum-unsupported-frames",
                    str(unsupported_frames),
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
