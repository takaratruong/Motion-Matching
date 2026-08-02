"""Live native-MuJoCo rendering for Torch stair motion-matched kinematics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .joints import ContractError
from .torch_g1_fk import MujocoG1FootKinematics, target_state_qpos
from .operator_x11 import KEYSYMS, X11KeyStateProvider
from .torch_motion_data import MotionFolder
from .torch_motion_features import CommandTrajectory, TorchMotionDatabase
from .torch_motion_matcher import (
    MotionMatchResult,
    TorchMotionMatcher,
    predict_command_trajectory,
)
from .torch_terrain_features import DENSE_FORWARD_M, dense_query_points
from .torch_terrain_rollout import (
    ResolvedStairConfig,
    load_experiment_config,
    matcher_config_from_resolved,
    resolve_stair_config,
    terrain_transition_validator_from_resolved,
)
from .torch_terrain_skill_horizon_rollout import TerrainSkillHorizonMatcher
from .torch_terrain_skill_horizon_search import (
    horizon_search_config_from_experiment,
)
from .torch_terrain_skill_horizons import build_horizon_inventory
from .torch_terrain_skills import build_terrain_skill_inventory


_DT_S = 0.02
_CONTROL_KEYS = frozenset(KEYSYMS)


def _finite_vector(
    value: object, shape: tuple[int, ...], label: str
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().to("cpu").numpy()
    else:
        array = np.asarray(value)
    output = np.asarray(array, np.float64)
    if output.shape != shape or not np.isfinite(output).all():
        raise ContractError(f"{label} must be finite with shape {shape}")
    return np.ascontiguousarray(output)


@dataclass(frozen=True)
class LiveControlCommand:
    velocity_world_xy: np.ndarray
    heading_world_yaw: float

    def __post_init__(self) -> None:
        velocity = _finite_vector(
            self.velocity_world_xy, (2,), "live command velocity"
        )
        heading = float(self.heading_world_yaw)
        if not math.isfinite(heading):
            raise ContractError("live command heading must be finite")
        velocity.setflags(write=False)
        object.__setattr__(self, "velocity_world_xy", velocity)
        object.__setattr__(self, "heading_world_yaw", heading)

    @property
    def advance_matcher(self) -> bool:
        """Whether this operator command consumes another motion frame."""

        return float(np.linalg.norm(self.velocity_world_xy)) > 1e-10


def command_from_keys(
    pressed: frozenset,
    *,
    reference_direction_xy: object,
    speed_mps: float,
    previous_heading_rad: float,
) -> LiveControlCommand:
    """Map focused WASD levels into the experiment's matcher-world frame."""

    if type(pressed) is not frozenset or any(
        key not in _CONTROL_KEYS for key in pressed
    ):
        raise ContractError("pressed keys must be a valid frozenset")
    reference = _finite_vector(
        reference_direction_xy, (2,), "reference direction"
    )
    norm = float(np.linalg.norm(reference))
    speed = float(speed_mps)
    previous_heading = float(previous_heading_rad)
    if (
        not math.isfinite(norm)
        or norm <= 1e-8
        or not math.isfinite(speed)
        or speed <= 0.0
        or not math.isfinite(previous_heading)
    ):
        raise ContractError(
            "reference direction, speed, and previous heading are invalid"
        )
    forward = reference / norm
    right = np.array((forward[1], -forward[0]), np.float64)
    if "SPACE" in pressed:
        local_forward = 0.0
        local_right = 0.0
    else:
        local_forward = float("W" in pressed) - float("S" in pressed)
        local_right = float("D" in pressed) - float("A" in pressed)
    velocity = local_forward * forward + local_right * right
    magnitude = float(np.linalg.norm(velocity))
    if magnitude > 1.0:
        velocity /= magnitude
    velocity *= speed
    heading = (
        previous_heading
        if float(np.linalg.norm(velocity)) <= 1e-10
        else math.atan2(float(velocity[1]), float(velocity[0]))
    )
    return LiveControlCommand(velocity, heading)


@dataclass(frozen=True)
class ControlEdges:
    reset_requested: bool
    exit_requested: bool


class ControlEdgeLatch:
    """Turn held reset/exit levels into one event per physical press."""

    def __init__(self) -> None:
        self._previous = frozenset()

    def update(self, pressed: frozenset) -> ControlEdges:
        if type(pressed) is not frozenset or any(
            key not in _CONTROL_KEYS for key in pressed
        ):
            raise ContractError("edge latch requires valid pressed keys")
        edges = ControlEdges(
            reset_requested=(
                "BACKSPACE" in pressed and "BACKSPACE" not in self._previous
            ),
            exit_requested=("X" in pressed and "X" not in self._previous),
        )
        self._previous = pressed
        return edges


class LiveActionChunkLatch:
    """Finish one selected terrain chunk after movement input is released."""

    def __init__(self) -> None:
        self._moving_command: LiveControlCommand | None = None

    def reset(self) -> None:
        self._moving_command = None

    def resolve(
        self,
        command: LiveControlCommand,
        *,
        selected_frame: int,
        endpoint: int | None,
    ) -> LiveControlCommand | None:
        if command.advance_matcher:
            self._moving_command = command
            return command
        if self._moving_command is None or endpoint is None:
            return None
        if int(selected_frame) >= int(endpoint) - 1:
            self._moving_command = None
            return None
        return self._moving_command


def matcher_result_qpos(result: object) -> np.ndarray:
    """Convert one target-ordered matcher result into native G1 qpos."""

    try:
        root_position = _finite_vector(
            result.root_position_world, (3,), "matcher root position"
        )
        root_orientation = _finite_vector(
            result.root_orientation_world_wxyz,
            (4,),
            "matcher root orientation",
        )
        target_joints = _finite_vector(
            result.joint_position, (29,), "matcher joint position"
        )
    except AttributeError as error:
        raise ContractError("matcher result is missing kinematic state") from error
    qpos = target_state_qpos(
        target_joints, root_position, root_orientation
    )
    if qpos.shape != (36,) or not np.isfinite(qpos).all():
        raise ContractError("converted matcher qpos is invalid")
    return qpos


def _scene_to_matcher_xy(
    scene_xy: np.ndarray,
    *,
    translation_scene_xy: np.ndarray,
    yaw_scene_from_matcher: float,
) -> np.ndarray:
    relative = np.asarray(scene_xy, np.float64) - translation_scene_xy
    cosine = math.cos(yaw_scene_from_matcher)
    sine = math.sin(yaw_scene_from_matcher)
    x = relative[..., 0]
    y = relative[..., 1]
    return np.stack(
        (cosine * x + sine * y, -sine * x + cosine * y),
        axis=-1,
    )


def _make_compiler_paths_absolute(root: ET.Element, g1_xml: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    for attribute in ("meshdir", "texturedir", "assetdir"):
        raw = compiler.get(attribute)
        if raw and not Path(raw).is_absolute():
            compiler.set(attribute, str((g1_xml.parent / raw).resolve()))


def _remove_flat_floor(worldbody: ET.Element) -> None:
    for geom in tuple(worldbody.findall("geom")):
        if (
            geom.get("name") == "floor"
            or geom.get("type", "sphere") == "plane"
        ):
            worldbody.remove(geom)


def build_kinematic_scene(
    g1_xml: str | Path,
    resolved: ResolvedStairConfig,
):
    """Compile the G1 plus the authenticated query grid as one MuJoCo scene."""

    if not isinstance(resolved, ResolvedStairConfig):
        raise ContractError("live scene requires a resolved stair config")
    g1_xml = Path(g1_xml).resolve()
    if not g1_xml.is_file() or g1_xml.is_symlink():
        raise ContractError(f"G1 XML is missing or invalid: {g1_xml}")
    try:
        import mujoco
    except ImportError as error:
        raise ContractError("live terrain viewer requires mujoco") from error
    try:
        tree = ET.parse(g1_xml)
    except Exception as error:
        raise ContractError(f"cannot parse G1 XML: {g1_xml}") from error
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ContractError("G1 XML root must be mujoco")
    _make_compiler_paths_absolute(root, g1_xml)
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = next(
            (
                index
                for index, child in enumerate(root)
                if child.tag == "worldbody"
            ),
            len(root),
        )
        root.insert(worldbody_index, asset)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ContractError("G1 XML is missing worldbody")
    _remove_flat_floor(worldbody)

    measurement = resolved.measurement_extension
    grid = measurement.query_grid
    height = (
        grid.height_z.detach().to("cpu", torch.float32).numpy().astype(
            np.float64
        )
    )
    origin = _finite_vector(
        grid.origin_xy.detach().to("cpu").numpy(),
        (2,),
        "terrain grid origin",
    )
    if (
        height.ndim != 2
        or min(height.shape) < 2
        or not np.isfinite(height).all()
    ):
        raise ContractError("terrain grid heights are invalid")
    cell = float(grid.cell_size_m)
    if not math.isfinite(cell) or cell <= 0.0:
        raise ContractError("terrain grid cell size is invalid")
    ny, nx = height.shape
    extent_scene = np.array(((nx - 1) * cell, (ny - 1) * cell))
    center_scene = origin + 0.5 * extent_scene
    translation = _finite_vector(
        measurement.alignment.translation_scene_xy.detach().to("cpu").numpy(),
        (2,),
        "terrain alignment translation",
    )
    yaw = float(
        measurement.alignment.yaw_scene_from_matcher.detach()
        .to("cpu")
        .item()
    )
    if not math.isfinite(yaw):
        raise ContractError("terrain alignment yaw is invalid")
    center_matcher = _scene_to_matcher_xy(
        center_scene,
        translation_scene_xy=translation,
        yaw_scene_from_matcher=yaw,
    )
    minimum = float(height.min())
    maximum = float(height.max())
    vertical_scale = max(maximum - minimum, 1e-6)
    normalized = np.ascontiguousarray(
        np.clip((height - minimum) / vertical_scale, 0.0, 1.0),
        np.float32,
    )
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "torch_stair_heightfield",
            "nrow": str(ny),
            "ncol": str(nx),
            "size": (
                f"{0.5 * extent_scene[0]:.9g} "
                f"{0.5 * extent_scene[1]:.9g} "
                f"{vertical_scale:.9g} 0.01"
            ),
        },
    )
    half_yaw = -0.5 * yaw
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "torch_stair_terrain",
            "type": "hfield",
            "hfield": "torch_stair_heightfield",
            "pos": (
                f"{center_matcher[0]:.9g} "
                f"{center_matcher[1]:.9g} {minimum:.9g}"
            ),
            "quat": (
                f"{math.cos(half_yaw):.9g} 0 0 "
                f"{math.sin(half_yaw):.9g}"
            ),
            "rgba": "0.52 0.42 0.29 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="torch-stair-live-scene-"
        ) as temporary:
            scene_xml = Path(temporary) / "scene.xml"
            tree.write(scene_xml, encoding="utf-8", xml_declaration=True)
            model = mujoco.MjModel.from_xml_path(str(scene_xml))
    except Exception as error:
        raise ContractError("cannot compile live kinematic MuJoCo scene") from error
    if model.nq != 36 or model.nhfield != 1:
        raise ContractError(
            f"live G1 scene must have nq=36 and one heightfield, got "
            f"nq={model.nq}, nhfield={model.nhfield}"
        )
    if model.hfield_data.size != normalized.size:
        raise ContractError("compiled heightfield size does not match grid")
    model.hfield_data[:] = normalized.reshape(-1)
    data = mujoco.MjData(model)
    return model, data


def apply_kinematic_state(
    mujoco_module, model, data, qpos: object
) -> None:
    """Write one complete state and run MuJoCo forward kinematics."""

    state = _finite_vector(qpos, (36,), "kinematic qpos")
    if getattr(model, "nq", None) != 36 or data.qpos.shape != (36,):
        raise ContractError("kinematic scene qpos contract is invalid")
    data.qpos[:] = state
    mujoco_module.mj_forward(model, data)


def dense_patch_positions(
    result: MotionMatchResult,
    measurement,
    trajectory: CommandTrajectory,
) -> np.ndarray:
    """Return the exact 95 body-and-path samples used by terrain search."""

    root = result.root_position_world
    quaternion = result.root_orientation_world_wxyz
    if (
        not isinstance(root, torch.Tensor)
        or tuple(root.shape) != (3,)
        or not isinstance(quaternion, torch.Tensor)
        or tuple(quaternion.shape) != (4,)
    ):
        raise ContractError("dense patch requires a valid matcher result")
    w, x, y, z = quaternion
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    matcher_xy = dense_query_points(root[:2], yaw, trajectory)
    scene_xy = measurement.alignment.matcher_to_scene_xy(matcher_xy)
    try:
        height = measurement.query_grid.sample_xy(scene_xy)
    except ContractError as error:
        if "outside the authoritative terrain domain" not in str(error):
            raise
        grid = measurement.query_grid
        try:
            origin = grid.origin_xy
            ny, nx = grid.height_z.shape
            margin = max(1.0e-6, float(grid.cell_size_m) * 1.0e-4)
            minimum = origin + margin
            maximum = origin + torch.tensor(
                ((nx - 1) * grid.cell_size_m, (ny - 1) * grid.cell_size_m),
                dtype=scene_xy.dtype,
                device=scene_xy.device,
            ) - margin
            clamped = torch.maximum(
                torch.minimum(scene_xy, maximum), minimum
            )
            height = grid.sample_xy(clamped)
        except (AttributeError, TypeError) as clamp_error:
            raise ContractError(
                "cannot clamp visualization markers to terrain domain"
            ) from clamp_error
    points = torch.cat((matcher_xy, height[:, None]), dim=1)
    output = points.detach().to("cpu").numpy().astype(np.float64)
    if output.shape != (95, 3) or not np.isfinite(output).all():
        raise ContractError("dense terrain marker positions are invalid")
    return output


def _set_patch_markers(mujoco_module, viewer, positions: np.ndarray) -> None:
    scene = viewer.user_scn
    if scene is None or scene.maxgeom < len(positions):
        raise ContractError("MuJoCo user scene cannot hold dense markers")
    scene.ngeom = 0
    identity = np.eye(3, dtype=np.float64).reshape(-1)
    for index, position in enumerate(positions):
        marker = np.array(position, copy=True)
        marker[2] += 0.012
        mujoco_module.mjv_initGeom(
            scene.geoms[index],
            type=mujoco_module.mjtGeom.mjGEOM_SPHERE,
            size=np.array((0.012, 0.0, 0.0), np.float64),
            pos=marker,
            mat=identity,
            rgba=np.array((1.0, 0.72, 0.05, 0.85), np.float32),
        )
        scene.ngeom += 1


def _diagnostic_overlay(
    result: object,
    command: LiveControlCommand,
    *,
    focused: bool,
    horizon_event: object | None = None,
    fault: str | None = None,
    multi_horizon: bool = False,
) -> tuple[None, None, str, str]:
    diagnostic = result.diagnostics
    horizon_mode = multi_horizon or horizon_event is not None
    left = (
        (
            "MULTI-HORIZON KINEMATIC / NO PHYSICS / NO SONIC\n"
            if horizon_mode
            else "KINEMATIC ONLY / NO PHYSICS / NO SONIC\n"
        )
        + "WASD move  Space stop  Backspace reset  X exit\n"
        f"focus={'LIVE' if focused else 'click MuJoCo window'}\n"
        f"command=({command.velocity_world_xy[0]:+.3f}, "
        f"{command.velocity_world_xy[1]:+.3f}) m/s"
    )
    if fault is not None:
        right = (
            f"SEARCH FAILURE: {fault}\n"
            "change command or Backspace reset"
        )
    elif horizon_event is not None:
        right = (
            f"{diagnostic.selected_clip_path}:{diagnostic.selected_frame}\n"
            f"horizon={horizon_event.target_frames}  "
            f"endpoint={horizon_event.endpoint_frame_exclusive}\n"
            f"entry={horizon_event.cost.entry:.2f}  "
            f"outcome={horizon_event.cost.outcome:.2f}\n"
            f"total={horizon_event.cost.total:.2f}  "
            f"matcher={diagnostic.step_time_ns / 1e6:.2f} ms"
        )
    else:
        right = (
            f"{diagnostic.selected_clip_path}:{diagnostic.selected_frame}\n"
            f"motion={diagnostic.motion_feature_cost:.2f}  "
            f"terrain={diagnostic.extension_feature_cost:.2f}\n"
            f"total={diagnostic.selected_total_cost:.2f}\n"
            f"matcher={diagnostic.step_time_ns / 1e6:.2f} ms"
        )
    return (None, None, left, right)


def _validate_live_mode(
    *,
    multi_horizon: bool,
    contact_segments: bool,
    foothold_arm: str | None,
    foot_lock: bool = False,
    contact_phase_gate: bool = False,
    swing_clearance_margin_m: float | None = None,
    foot_correction_halflife_s: float = 0.04,
    swing_plan_sigma_frames: float | None = None,
    maximum_source_contact_p95_m: float | None = None,
    normalization_source: str | None = None,
    maximum_translation_warp_m: float = 0.0,
    maximum_yaw_warp_rad: float = 0.0,
    minimum_endpoint_warp_yaw_rad: float = 0.0,
    maximum_endpoint_warp_yaw_rad: float = math.pi,
    maximum_endpoint_warp_terrain_delta_m: float = math.inf,
) -> None:
    if multi_horizon and (contact_segments or foothold_arm is not None):
        raise ContractError(
            "multi-horizon and contact/foothold modes are mutually exclusive"
        )
    if foot_lock and not multi_horizon:
        raise ContractError("terrain foot lock requires multi-horizon mode")
    if contact_phase_gate and not multi_horizon:
        raise ContractError("contact phase gate requires multi-horizon mode")
    if swing_clearance_margin_m is not None and not foot_lock:
        raise ContractError("swing clearance requires terrain foot lock")
    if swing_plan_sigma_frames is not None and swing_clearance_margin_m is None:
        raise ContractError("swing plan requires swing clearance")
    if maximum_source_contact_p95_m is not None and not multi_horizon:
        raise ContractError("source contact quality gate requires multi-horizon mode")
    if normalization_source is not None and not multi_horizon:
        raise ContractError("normalization source requires multi-horizon mode")
    if (
        maximum_translation_warp_m != 0.0
        or maximum_yaw_warp_rad != 0.0
        or minimum_endpoint_warp_yaw_rad != 0.0
        or maximum_endpoint_warp_yaw_rad != math.pi
        or maximum_endpoint_warp_terrain_delta_m != math.inf
    ) and not multi_horizon:
        raise ContractError("endpoint warp requires multi-horizon mode")


def _build_live_matcher(
    resolved: ResolvedStairConfig,
    g1_xml: str | Path,
    *,
    multi_horizon: bool,
    contact_segment_policy: Any,
    foothold_action_policy: Any,
    foot_lock: bool = False,
    contact_phase_gate: bool = False,
    swing_clearance_margin_m: float | None = None,
    foot_correction_halflife_s: float = 0.04,
    swing_plan_sigma_frames: float | None = None,
    maximum_source_contact_p95_m: float | None = None,
    normalization_source: str | None = None,
    maximum_translation_warp_m: float = 0.0,
    maximum_yaw_warp_rad: float = 0.0,
    minimum_endpoint_warp_yaw_rad: float = 0.0,
    maximum_endpoint_warp_yaw_rad: float = math.pi,
    maximum_endpoint_warp_terrain_delta_m: float = math.inf,
):
    matcher_config = matcher_config_from_resolved(resolved.resolved_config)
    if multi_horizon:
        normalization_override = None
        if normalization_source is not None:
            source_folder = MotionFolder.load(normalization_source)
            source_database = TorchMotionDatabase.from_folder(
                source_folder,
                device=resolved.device,
                reset_clip_path=resolved.resolved_config["reset_clip"],
            )
            normalization_override = source_database.normalization
            del source_database, source_folder
        base = TorchMotionMatcher.from_folder(
            resolved.dataset.root,
            device=str(resolved.device),
            config=matcher_config,
            reset_clip_path=resolved.resolved_config["reset_clip"],
            normalization_override=normalization_override,
        )
        skills = build_terrain_skill_inventory(
            resolved.dataset,
            base.database,
            maximum_source_contact_p95_m=maximum_source_contact_p95_m,
        )
        horizons = build_horizon_inventory(
            resolved.dataset, base.database, skills
        )
        foot_kinematics = MujocoG1FootKinematics(str(g1_xml))
        matcher_kwargs = {}
        if foot_lock:
            from .torch_terrain_foot_lock import build_terrain_foot_lock

            matcher_kwargs["result_filter"] = build_terrain_foot_lock(
                resolved,
                foot_kinematics,
                swing_clearance_margin_m=swing_clearance_margin_m,
                correction_halflife_s=foot_correction_halflife_s,
                swing_plan_sigma_frames=swing_plan_sigma_frames,
            )
        if contact_phase_gate:
            matcher_kwargs["contact_phase_gate"] = True
        return TerrainSkillHorizonMatcher(
            base_matcher=base,
            skill_inventory=skills,
            horizon_inventory=horizons,
            dataset=resolved.dataset,
            query_terrain=resolved.measurement_extension,
            foot_kinematics=foot_kinematics,
            config=matcher_config,
            search_config=horizon_search_config_from_experiment(
                resolved.resolved_config
            ),
            maximum_translation_warp_m=maximum_translation_warp_m,
            maximum_yaw_warp_rad=maximum_yaw_warp_rad,
            minimum_endpoint_warp_yaw_rad=minimum_endpoint_warp_yaw_rad,
            maximum_endpoint_warp_yaw_rad=maximum_endpoint_warp_yaw_rad,
            maximum_endpoint_warp_terrain_delta_m=(
                maximum_endpoint_warp_terrain_delta_m
            ),
            **matcher_kwargs,
        )
    return TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved.device),
        config=matcher_config,
        extension=resolved.measurement_extension,
        reset_clip_path=resolved.resolved_config["reset_clip"],
        emitted_window_validator=terrain_transition_validator_from_resolved(
            resolved
        ),
        contact_segment_policy=contact_segment_policy,
        foothold_action_policy=foothold_action_policy,
    )


def run_live_viewer(
    *,
    dataset: str | Path,
    config: str | Path,
    g1_xml: str | Path,
    device: str,
    multi_horizon: bool = False,
    contact_segments: bool = False,
    foothold_arm: str | None = None,
    foothold_height_tolerance_m: float = 0.04,
    turn_root_lateral_cost_weight: float = 100.0,
    turn_root_progress_cost_weight: float = 10.0,
    turn_sequence_candidate_count: int = 64,
    heading_maintenance_yaw_cost_weight: float = 0.3,
    turn_lateral_root_warp_gain: float = 0.606,
    small_turn_lateral_root_warp_gain: float = 0.25,
    reversal_lateral_root_warp_gain: float = 0.25,
    strafe_action_gate: bool = False,
    foot_lock: bool = False,
    contact_phase_gate: bool = False,
    swing_clearance_margin_m: float | None = None,
    foot_correction_halflife_s: float = 0.04,
    swing_plan_sigma_frames: float | None = None,
    maximum_source_contact_p95_m: float | None = None,
    normalization_source: str | None = None,
    maximum_translation_warp_m: float = 0.0,
    maximum_yaw_warp_rad: float = 0.0,
    minimum_endpoint_warp_yaw_rad: float = 0.0,
    maximum_endpoint_warp_yaw_rad: float = math.pi,
    maximum_endpoint_warp_terrain_delta_m: float = math.inf,
) -> None:
    """Run the dense 50 Hz matcher and display each committed state."""

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise ContractError("live terrain viewer requires mujoco viewer") from error
    _validate_live_mode(
        multi_horizon=multi_horizon,
        contact_segments=contact_segments,
        foothold_arm=foothold_arm,
        foot_lock=foot_lock,
        contact_phase_gate=contact_phase_gate,
        swing_clearance_margin_m=swing_clearance_margin_m,
        foot_correction_halflife_s=foot_correction_halflife_s,
        swing_plan_sigma_frames=swing_plan_sigma_frames,
        maximum_source_contact_p95_m=maximum_source_contact_p95_m,
        normalization_source=normalization_source,
        maximum_translation_warp_m=maximum_translation_warp_m,
        maximum_yaw_warp_rad=maximum_yaw_warp_rad,
        minimum_endpoint_warp_yaw_rad=minimum_endpoint_warp_yaw_rad,
        maximum_endpoint_warp_yaw_rad=maximum_endpoint_warp_yaw_rad,
        maximum_endpoint_warp_terrain_delta_m=(
            maximum_endpoint_warp_terrain_delta_m
        ),
    )
    if contact_segments:
        from .torch_contact_segment_rollout import (
            build_contact_segment_policy,
            load_contact_segment_config,
            resolve_contact_segment_config,
        )

        raw = load_contact_segment_config(config)
        resolved = resolve_contact_segment_config(
            dataset, raw, device=device
        )
    else:
        raw = load_experiment_config(config)
        resolved = resolve_stair_config(dataset, raw, device=device)
    descriptor = resolved.resolved_config["conditions"]["dense"]
    if (
        descriptor["encoder"] != "dense"
        or float(descriptor["weight"]) <= 0.0
    ):
        raise ContractError("live terrain viewer requires the dense condition")
    contact_segment_policy = None
    foothold_action_policy = None
    if contact_segments:
        contact_segment_policy = build_contact_segment_policy(
            resolved, g1_xml
        )
    if foothold_arm is not None:
        if contact_segment_policy is None:
            raise ContractError(
                "foothold arm requires contact-segment terrain actions"
            )
        if not 0.0 < float(foothold_height_tolerance_m) < 0.0889:
            raise ContractError(
                "foothold height tolerance must be positive and below half a riser"
            )
        from .torch_foothold_actions import (
            FootholdActionIndex,
            FootholdActionPolicy,
            FootholdSelectionArm,
            FootholdTransitionGraph,
        )

        action_index = FootholdActionIndex.from_dataset(
            resolved.dataset, contact_segment_policy.index
        )
        arm = FootholdSelectionArm(foothold_arm)
        foothold_action_policy = FootholdActionPolicy(
            index=action_index,
            extension=resolved.measurement_extension,
            arm=arm,
            transition_graph=(
                FootholdTransitionGraph.from_dataset(
                    action_index, resolved.dataset
                )
                if arm is FootholdSelectionArm.LAYERED_GRAPH_HYBRID
                else None
            ),
            turn_sequence_lookahead=(
                arm is FootholdSelectionArm.LAYERED_GRAPH_HYBRID
            ),
            height_tolerance_m=float(foothold_height_tolerance_m),
            turn_root_lateral_cost_weight=float(
                turn_root_lateral_cost_weight
            ),
            turn_root_progress_cost_weight=float(
                turn_root_progress_cost_weight
            ),
            turn_sequence_candidate_count=int(
                turn_sequence_candidate_count
            ),
            heading_maintenance_yaw_cost_weight=float(
                heading_maintenance_yaw_cost_weight
            ),
            turn_lateral_root_warp_gain=float(turn_lateral_root_warp_gain),
            small_turn_lateral_root_warp_gain=float(
                small_turn_lateral_root_warp_gain
            ),
            reversal_lateral_root_warp_gain=float(
                reversal_lateral_root_warp_gain
            ),
            strafe_action_gate_enabled=bool(strafe_action_gate),
        )
    matcher_config = matcher_config_from_resolved(resolved.resolved_config)
    matcher = _build_live_matcher(
        resolved,
        g1_xml,
        multi_horizon=multi_horizon,
        contact_segment_policy=contact_segment_policy,
        foothold_action_policy=foothold_action_policy,
        foot_lock=foot_lock,
        contact_phase_gate=contact_phase_gate,
        swing_clearance_margin_m=swing_clearance_margin_m,
        foot_correction_halflife_s=foot_correction_halflife_s,
        swing_plan_sigma_frames=swing_plan_sigma_frames,
        maximum_source_contact_p95_m=maximum_source_contact_p95_m,
        normalization_source=normalization_source,
        maximum_translation_warp_m=maximum_translation_warp_m,
        maximum_yaw_warp_rad=maximum_yaw_warp_rad,
        minimum_endpoint_warp_yaw_rad=minimum_endpoint_warp_yaw_rad,
        maximum_endpoint_warp_yaw_rad=maximum_endpoint_warp_yaw_rad,
        maximum_endpoint_warp_terrain_delta_m=(
            maximum_endpoint_warp_terrain_delta_m
        ),
    )
    from .torch_terrain_omni_rollout import resolved_stair_reset_position

    reset_position = resolved_stair_reset_position(resolved)
    model, data = build_kinematic_scene(g1_xml, resolved)
    result = matcher.reset(root_position_world_xy=reset_position)
    apply_kinematic_state(
        mujoco, model, data, matcher_result_qpos(result)
    )
    reference = np.asarray(
        resolved.resolved_config["reference_direction_matcher_xy"],
        np.float64,
    )
    speed = float(resolved.resolved_config["command_speed_mps"])
    heading = float(
        resolved.resolved_config["command_heading_matcher_yaw"]
    )
    command = command_from_keys(
        frozenset(),
        reference_direction_xy=reference,
        speed_mps=speed,
        previous_heading_rad=heading,
    )
    shaped_velocity = torch.zeros(
        2, dtype=result.root_position_world.dtype, device=resolved.device
    )
    shaped_heading = torch.zeros(
        (), dtype=result.root_position_world.dtype, device=resolved.device
    )
    marker_result = result
    marker_trajectory = predict_command_trajectory(
        result.root_position_world[:2],
        shaped_velocity,
        shaped_heading,
        shaped_velocity,
        shaped_heading,
        config=matcher_config,
    ).trajectory
    provider: X11KeyStateProvider | None = None
    latch = ControlEdgeLatch()
    action_latch = LiveActionChunkLatch()
    horizon_fault: str | None = None
    fault_command: tuple[tuple[float, float], float] | None = None
    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.distance = 3.2
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -22.0
            viewer.cam.lookat[:] = data.qpos[:3]
            provider = X11KeyStateProvider()
            print(
                "LIVE TORCH STAIR KINEMATICS: focus MuJoCo; "
                "WASD move, Space stop, Backspace reset, X exit.",
                flush=True,
            )
            next_tick = time.perf_counter()
            while viewer.is_running():
                levels = provider.sample()
                edges = latch.update(levels.pressed)
                if edges.exit_requested:
                    break
                command = command_from_keys(
                    levels.pressed,
                    reference_direction_xy=reference,
                    speed_mps=speed,
                    previous_heading_rad=heading,
                )
                if edges.reset_requested:
                    result = matcher.reset(
                        root_position_world_xy=reset_position
                    )
                    action_latch.reset()
                    shaped_velocity.zero_()
                    shaped_heading.zero_()
                    heading = float(
                        resolved.resolved_config[
                            "command_heading_matcher_yaw"
                        ]
                    )
                    marker_result = result
                    marker_trajectory = predict_command_trajectory(
                        result.root_position_world[:2],
                        shaped_velocity,
                        shaped_heading,
                        shaped_velocity,
                        shaped_heading,
                        config=matcher_config,
                    ).trajectory
                    horizon_fault = None
                    fault_command = None
                else:
                    marker_result = result
                    requested_velocity = torch.tensor(
                        command.velocity_world_xy,
                        dtype=result.root_position_world.dtype,
                        device=resolved.device,
                    )
                    requested_heading = torch.tensor(
                        command.heading_world_yaw,
                        dtype=result.root_position_world.dtype,
                        device=resolved.device,
                    )
                    shaped = predict_command_trajectory(
                        result.root_position_world[:2],
                        shaped_velocity,
                        shaped_heading,
                        requested_velocity,
                        requested_heading,
                        config=matcher_config,
                    )
                    marker_trajectory = shaped.trajectory
                command_identity = (
                    (
                        float(command.velocity_world_xy[0]),
                        float(command.velocity_world_xy[1]),
                    ),
                    float(command.heading_world_yaw),
                )
                if fault_command is not None and command_identity != fault_command:
                    horizon_fault = None
                    fault_command = None
                horizon_event = (
                    matcher.chunk_events[-1]
                    if multi_horizon and matcher.chunk_events
                    else None
                )
                playback_command = None
                if not edges.reset_requested:
                    playback_command = (
                        action_latch.resolve(
                            command,
                            selected_frame=int(result.diagnostics.selected_frame),
                            endpoint=(
                                horizon_event.endpoint_frame_exclusive
                                if horizon_event is not None
                                else None
                            ),
                        )
                        if multi_horizon
                        else (command if command.advance_matcher else None)
                    )
                if playback_command is not None:
                    if fault_command is None:
                        heading = playback_command.heading_world_yaw
                        try:
                            prepared = matcher.prepare_step(
                                tuple(playback_command.velocity_world_xy),
                                heading,
                                dt=_DT_S,
                            )
                            result = matcher.commit(prepared)
                            shaped_velocity = shaped.velocity_world_xy.clone()
                            shaped_heading = shaped.heading_world_yaw.clone()
                            horizon_fault = None
                        except ContractError as error:
                            if not multi_horizon:
                                raise
                            horizon_fault = str(error)
                            fault_command = command_identity
                            print(
                                f"MULTI-HORIZON SEARCH FAILURE: {error}",
                                flush=True,
                            )
                qpos = matcher_result_qpos(result)
                patch = dense_patch_positions(
                    marker_result,
                    resolved.measurement_extension,
                    marker_trajectory,
                )
                with viewer.lock():
                    apply_kinematic_state(mujoco, model, data, qpos)
                    _set_patch_markers(mujoco, viewer, patch)
                    viewer.cam.lookat[:] = qpos[:3]
                viewer.set_texts(
                    _diagnostic_overlay(
                        result,
                        command,
                        focused=levels.focused,
                        horizon_event=(
                            matcher.chunk_events[-1]
                            if multi_horizon and matcher.chunk_events
                            else None
                        ),
                        fault=horizon_fault,
                        multi_horizon=multi_horizon,
                    )
                )
                viewer.sync()
                next_tick += _DT_S
                delay = next_tick - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                elif delay < -_DT_S:
                    next_tick = time.perf_counter()
    finally:
        if provider is not None:
            provider.close()


def build_live_viewer_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive dense Torch stair kinematics in native MuJoCo "
            "without a tracking controller or physics integration."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--g1-xml", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--multi-horizon",
        action="store_true",
        help="Use the qualified stable-endpoint multi-horizon matcher.",
    )
    parser.add_argument(
        "--foot-lock",
        action="store_true",
        help="Pin source-supported feet to query terrain during multi-horizon playback.",
    )
    parser.add_argument(
        "--contact-phase-gate",
        action="store_true",
        help="Prefer skill entries with the current source support pattern.",
    )
    parser.add_argument(
        "--swing-clearance-margin-m",
        type=float,
        default=None,
        help="Lift unsupported ankles above query terrain by this margin.",
    )
    parser.add_argument(
        "--foot-correction-halflife-s",
        type=float,
        default=0.04,
        help="Inertialization halflife for terrain foot corrections.",
    )
    parser.add_argument(
        "--swing-plan-sigma-frames",
        type=float,
        default=None,
        help="Spread future source-path clearance backward by this sigma.",
    )
    parser.add_argument(
        "--maximum-source-contact-p95-m",
        type=float,
        default=None,
        help="Exclude terrain clips with worse authenticated contact fit.",
    )
    parser.add_argument(
        "--normalization-source",
        default=None,
        help="Freeze feature normalization to another motion corpus.",
    )
    parser.add_argument(
        "--maximum-translation-warp-m", type=float, default=0.0
    )
    parser.add_argument("--maximum-yaw-warp-rad", type=float, default=0.0)
    parser.add_argument(
        "--minimum-endpoint-warp-yaw-rad", type=float, default=0.0
    )
    parser.add_argument(
        "--maximum-endpoint-warp-yaw-rad", type=float, default=math.pi
    )
    parser.add_argument(
        "--maximum-endpoint-warp-terrain-delta-m",
        type=float,
        default=math.inf,
    )
    parser.add_argument(
        "--contact-segments",
        action="store_true",
        help="Use committed authoritative-FK terrain contact segments.",
    )
    parser.add_argument(
        "--foothold-arm",
        choices=(
            "first-contact",
            "two-contact",
            "hybrid",
            "layered",
            "layered-hybrid",
            "layered-graph-hybrid",
            "continuous-control",
        ),
        help="Condition terrain entries with the selected foothold policy.",
    )
    parser.add_argument(
        "--foothold-height-tolerance-m",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--turn-root-lateral-cost-weight",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--turn-root-progress-cost-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--turn-lateral-root-warp-gain",
        type=float,
        default=0.606,
    )
    parser.add_argument(
        "--small-turn-lateral-root-warp-gain",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--reversal-lateral-root-warp-gain",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--strafe-action-gate",
        action="store_true",
    )
    parser.add_argument(
        "--turn-sequence-candidate-count",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--heading-maintenance-yaw-cost-weight",
        type=float,
        default=0.3,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_live_viewer_argument_parser().parse_args(argv)
    run_live_viewer(
        dataset=args.dataset,
        config=args.config,
        g1_xml=args.g1_xml,
        device=args.device,
        multi_horizon=args.multi_horizon,
        contact_segments=args.contact_segments,
        foothold_arm=args.foothold_arm,
        foothold_height_tolerance_m=args.foothold_height_tolerance_m,
        turn_root_lateral_cost_weight=args.turn_root_lateral_cost_weight,
        turn_root_progress_cost_weight=args.turn_root_progress_cost_weight,
        turn_sequence_candidate_count=args.turn_sequence_candidate_count,
        heading_maintenance_yaw_cost_weight=(
            args.heading_maintenance_yaw_cost_weight
        ),
        turn_lateral_root_warp_gain=args.turn_lateral_root_warp_gain,
        small_turn_lateral_root_warp_gain=(
            args.small_turn_lateral_root_warp_gain
        ),
        reversal_lateral_root_warp_gain=(
            args.reversal_lateral_root_warp_gain
        ),
        strafe_action_gate=args.strafe_action_gate,
        foot_lock=args.foot_lock,
        contact_phase_gate=args.contact_phase_gate,
        swing_clearance_margin_m=args.swing_clearance_margin_m,
        foot_correction_halflife_s=args.foot_correction_halflife_s,
        swing_plan_sigma_frames=args.swing_plan_sigma_frames,
        maximum_source_contact_p95_m=(
            args.maximum_source_contact_p95_m
        ),
        normalization_source=args.normalization_source,
        maximum_translation_warp_m=args.maximum_translation_warp_m,
        maximum_yaw_warp_rad=args.maximum_yaw_warp_rad,
        minimum_endpoint_warp_yaw_rad=(
            args.minimum_endpoint_warp_yaw_rad
        ),
        maximum_endpoint_warp_yaw_rad=(
            args.maximum_endpoint_warp_yaw_rad
        ),
        maximum_endpoint_warp_terrain_delta_m=(
            args.maximum_endpoint_warp_terrain_delta_m
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
