"""Live native-MuJoCo rendering for Torch stair motion-matched kinematics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import time
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .joints import ContractError
from .torch_g1_fk import target_state_qpos
from .operator_x11 import KEYSYMS, X11KeyStateProvider
from .torch_motion_matcher import MotionMatchResult, TorchMotionMatcher
from .torch_terrain_features import DENSE_FORWARD_M, DENSE_LATERAL_M
from .torch_terrain_rollout import (
    ResolvedStairConfig,
    load_experiment_config,
    matcher_config_from_resolved,
    resolve_stair_config,
    terrain_transition_validator_from_resolved,
)


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
    result: MotionMatchResult, measurement
) -> np.ndarray:
    """Return the 91 current dense height samples in matcher coordinates."""

    root = result.root_position_world
    quaternion = result.root_orientation_world_wxyz
    if (
        not isinstance(root, torch.Tensor)
        or tuple(root.shape) != (3,)
        or not isinstance(quaternion, torch.Tensor)
        or tuple(quaternion.shape) != (4,)
    ):
        raise ContractError("dense patch requires a valid matcher result")
    device = root.device
    dtype = root.dtype
    w, x, y, z = quaternion
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    forward, lateral = torch.meshgrid(
        torch.tensor(DENSE_FORWARD_M, device=device, dtype=dtype),
        torch.tensor(DENSE_LATERAL_M, device=device, dtype=dtype),
        indexing="ij",
    )
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    matcher_xy = torch.stack(
        (
            cosine * forward - sine * lateral,
            sine * forward + cosine * lateral,
        ),
        dim=-1,
    ).reshape(-1, 2) + root[:2]
    scene_xy = measurement.alignment.matcher_to_scene_xy(matcher_xy)
    height = measurement.query_grid.sample_xy(scene_xy)
    points = torch.cat((matcher_xy, height[:, None]), dim=1)
    output = points.detach().to("cpu").numpy().astype(np.float64)
    if output.shape != (91, 3) or not np.isfinite(output).all():
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
    result: MotionMatchResult,
    command: LiveControlCommand,
    *,
    focused: bool,
) -> tuple[None, None, str, str]:
    diagnostic = result.diagnostics
    left = (
        "KINEMATIC ONLY / NO PHYSICS / NO SONIC\n"
        "WASD move  Space stop  Backspace reset  X exit\n"
        f"focus={'LIVE' if focused else 'click MuJoCo window'}\n"
        f"command=({command.velocity_world_xy[0]:+.3f}, "
        f"{command.velocity_world_xy[1]:+.3f}) m/s"
    )
    right = (
        f"{diagnostic.selected_clip_path}:{diagnostic.selected_frame}\n"
        f"motion={diagnostic.motion_feature_cost:.2f}  "
        f"terrain={diagnostic.extension_feature_cost:.2f}\n"
        f"total={diagnostic.selected_total_cost:.2f}\n"
        f"matcher={diagnostic.step_time_ns / 1e6:.2f} ms"
    )
    return (None, None, left, right)


def run_live_viewer(
    *,
    dataset: str | Path,
    config: str | Path,
    g1_xml: str | Path,
    device: str,
    contact_segments: bool = False,
) -> None:
    """Run the dense 50 Hz matcher and display each committed state."""

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as error:
        raise ContractError("live terrain viewer requires mujoco viewer") from error
    if contact_segments:
        from .torch_contact_segment_rollout import (
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
    if contact_segments:
        from .torch_contact_segments import (
            ContactSegmentIndex,
            TerrainContactSegmentPolicy,
        )
        from .torch_g1_fk import MujocoG1FootKinematics

        bounds = resolved.resolved_config["contact_segments"]
        contact_segment_policy = TerrainContactSegmentPolicy(
            index=ContactSegmentIndex.from_dataset(
                resolved.dataset,
                minimum_frames=int(bounds["minimum_frames"]),
                maximum_frames=int(bounds["maximum_frames"]),
            ),
            extension=resolved.measurement_extension,
            foot_kinematics=MujocoG1FootKinematics(g1_xml),
        )
    matcher = TorchMotionMatcher.from_folder(
        resolved.dataset.root,
        device=str(resolved.device),
        config=matcher_config_from_resolved(resolved.resolved_config),
        extension=resolved.measurement_extension,
        reset_clip_path=resolved.resolved_config["reset_clip"],
        emitted_window_validator=(
            terrain_transition_validator_from_resolved(resolved)
        ),
        contact_segment_policy=contact_segment_policy,
    )
    model, data = build_kinematic_scene(g1_xml, resolved)
    result = matcher.reset()
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
    provider: X11KeyStateProvider | None = None
    latch = ControlEdgeLatch()
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
                    result = matcher.reset()
                    heading = float(
                        resolved.resolved_config[
                            "command_heading_matcher_yaw"
                        ]
                    )
                elif command.advance_matcher:
                    heading = command.heading_world_yaw
                    prepared = matcher.prepare_step(
                        (
                            float(command.velocity_world_xy[0]),
                            float(command.velocity_world_xy[1]),
                        ),
                        heading,
                        dt=_DT_S,
                    )
                    result = matcher.commit(prepared)
                qpos = matcher_result_qpos(result)
                patch = dense_patch_positions(
                    result, resolved.measurement_extension
                )
                with viewer.lock():
                    apply_kinematic_state(mujoco, model, data, qpos)
                    _set_patch_markers(mujoco, viewer, patch)
                    viewer.cam.lookat[:] = qpos[:3]
                viewer.set_texts(
                    _diagnostic_overlay(
                        result, command, focused=levels.focused
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
        "--contact-segments",
        action="store_true",
        help="Use committed authoritative-FK terrain contact segments.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_live_viewer_argument_parser().parse_args(argv)
    run_live_viewer(
        dataset=args.dataset,
        config=args.config,
        g1_xml=args.g1_xml,
        device=args.device,
        contact_segments=args.contact_segments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
