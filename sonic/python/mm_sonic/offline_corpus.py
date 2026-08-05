"""Offline two-stick command -> motion-matching corpus generator.

The output is a multi-clip zarr that Justin's ``sonic_port.pipeline`` can
consume directly as a ``motion_zarr`` source.  Every generated base clip is
followed by an analytically exact sagittal mirror; mirrored kinematics are not
re-searched, so command and motion counts are exactly balanced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

import numpy as np

from .human_command_curriculum import (
    BUTTON_CHANNELS,
    COMMAND_EVENT_ABRUPT,
    COMMAND_EVENT_DUAL_STICK,
    COMMAND_EVENT_HEADING_JUMP,
    COMMAND_EVENT_STOP_OR_RESTART,
    COMMAND_EVENT_VELOCITY_REVERSE,
    CURRICULUM_SCHEMA,
    RAW_STICK_CHANNELS,
    CommandTrace,
    build_abrupt_curriculum,
    build_base_curriculum,
    build_omnidirectional_curriculum,
    command_event_mask,
)
from .joints import ContractError
from .torch_motion_matcher import (
    MatcherConfig,
    TorchMotionMatcher,
)


CORPUS_SCHEMA = "takara-motion-matching-corpus/v2"
TASK4_SCHEMA = (
    "velocity_local_forward",
    "velocity_local_lateral",
    "cos_facing_delta",
    "sin_facing_delta",
)
TASK12_KNOT_OFFSETS = (6, 12, 18, 24)
TASK12_SCHEMA = tuple(
    f"knot_{offset}_{feature}"
    for offset in TASK12_KNOT_OFFSETS
    for feature in ("velocity_local_forward", "velocity_local_lateral", "yaw_rate")
)
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _mirror_permutation(names: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(names)}
    output = np.arange(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        if name.startswith("left_"):
            output[i] = index["right_" + name[len("left_") :]]
        elif name.startswith("right_"):
            output[i] = index["left_" + name[len("right_") :]]
    return output


JOINT_MIRROR_PERMUTATION = _mirror_permutation(JOINT_NAMES)
BODY_MIRROR_PERMUTATION = _mirror_permutation(BODY_NAMES)
JOINT_MIRROR_SIGNS = np.asarray(
    [-1.0 if ("roll" in name or "yaw" in name) else 1.0 for name in JOINT_NAMES],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _world_to_local_xy(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.stack(
        (
            c * vector[..., 0] + s * vector[..., 1],
            -s * vector[..., 0] + c * vector[..., 1],
        ),
        axis=-1,
    )


def _local_to_world_xy(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    value = np.asarray(vector)
    angle = np.asarray(yaw)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.stack(
        (
            c * value[..., 0] - s * value[..., 1],
            s * value[..., 0] + c * value[..., 1],
        ),
        axis=-1,
    )


def _task4(
    velocity_world_xy: np.ndarray,
    facing_world_yaw: np.ndarray,
    root_world_yaw: np.ndarray,
) -> np.ndarray:
    local_velocity = _world_to_local_xy(velocity_world_xy, root_world_yaw)
    delta = _wrap(facing_world_yaw - root_world_yaw)
    return np.concatenate(
        (
            local_velocity,
            np.cos(delta)[:, None],
            np.sin(delta)[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def _realized_task4(
    body_quat_wxyz: np.ndarray,
    body_linear_velocity_world: np.ndarray,
    *,
    horizon_frames: int,
) -> np.ndarray:
    yaw = _yaw_from_wxyz(body_quat_wxyz[:, 0])
    future = np.minimum(
        np.arange(len(yaw), dtype=np.int64) + int(horizon_frames),
        len(yaw) - 1,
    )
    return _task4(body_linear_velocity_world[:, 0, :2], yaw[future], yaw)


def _task12_intended(
    ball_velocity_world_xy: np.ndarray,
    ball_heading_world_yaw: np.ndarray,
    ball_heading_velocity_rad_s: np.ndarray,
) -> np.ndarray:
    """Encode exact future filtered-ball commands at +6/+12/+18/+24 frames."""
    velocity = np.asarray(ball_velocity_world_xy, dtype=np.float64)
    heading = np.asarray(ball_heading_world_yaw, dtype=np.float64)
    heading_velocity = np.asarray(
        ball_heading_velocity_rad_s, dtype=np.float64
    )
    if (
        velocity.ndim != 2
        or velocity.shape[1] != 2
        or heading.shape != velocity.shape[:1]
        or heading_velocity.shape != velocity.shape[:1]
    ):
        raise ContractError("ball velocity/heading arrays have incompatible shapes")
    rows = np.arange(len(velocity), dtype=np.int64)[:, None]
    future = np.minimum(
        rows + np.asarray(TASK12_KNOT_OFFSETS, dtype=np.int64)[None, :],
        len(velocity) - 1,
    )
    future_velocity = velocity[future]
    future_heading = heading[future]
    local_velocity = _world_to_local_xy(future_velocity, future_heading)
    knots = np.concatenate(
        (local_velocity, heading_velocity[future, None]), axis=-1
    )
    task = knots.reshape(len(velocity), len(TASK12_SCHEMA))
    if not np.isfinite(task).all():
        raise ContractError("Task12 intended command contains non-finite values")
    return np.ascontiguousarray(task, dtype=np.float32)


def generate_clip(
    matcher: TorchMotionMatcher,
    trace: CommandTrace,
    *,
    realized_horizon_frames: int = 24,
) -> dict[str, np.ndarray]:
    """Generate one base clip and retain both command and matcher diagnostics."""
    matcher.reset()
    frames = trace.frame_count
    arrays: dict[str, np.ndarray] = {
        "joint_pos": np.empty((frames, 29), dtype=np.float32),
        "joint_vel": np.empty((frames, 29), dtype=np.float32),
        "body_pos_w": np.empty((frames, 30, 3), dtype=np.float32),
        "body_quat_w": np.empty((frames, 30, 4), dtype=np.float32),
        "body_lin_vel_w": np.empty((frames, 30, 3), dtype=np.float32),
        "body_ang_vel_w": np.empty((frames, 30, 3), dtype=np.float32),
        "command_raw_sticks": trace.raw_sticks.copy(),
        "command_buttons": trace.buttons.copy(),
        "command_event_mask": command_event_mask(trace.raw_sticks),
        "command_requested_velocity_local_xy": (
            trace.requested_velocity_local_xy.copy()
        ),
        "command_requested_velocity_world_xy": np.empty(
            (frames, 2), dtype=np.float32
        ),
        "command_requested_heading_world_yaw": (
            trace.requested_heading_world_yaw.copy()
        ),
        "command_shaped_velocity_world_xy": np.empty((frames, 2), dtype=np.float32),
        "command_shaped_heading_world_yaw": np.empty(frames, dtype=np.float32),
        "command_requested_path_world_xy": np.empty((frames, 2), dtype=np.float32),
        "command_path_world_xy": np.empty((frames, 2), dtype=np.float32),
        "command_ball_position_world_xy": np.empty((frames, 2), dtype=np.float32),
        "command_ball_velocity_world_xy": np.empty((frames, 2), dtype=np.float32),
        "command_ball_heading_world_yaw": np.empty(frames, dtype=np.float32),
        "command_ball_heading_velocity_rad_s": np.empty(
            frames, dtype=np.float32
        ),
        "matcher_root_adjustment_distance_m": np.empty(frames, dtype=np.float32),
        "matcher_root_adjustment_angle_rad": np.empty(frames, dtype=np.float32),
        "matcher_root_clamp_distance_m": np.empty(frames, dtype=np.float32),
        "matcher_root_clamp_angle_rad": np.empty(frames, dtype=np.float32),
        "matcher_selected_clip_index": np.empty(frames, dtype=np.int32),
        "matcher_selected_frame": np.empty(frames, dtype=np.int32),
        "matcher_feature_cost": np.empty(frames, dtype=np.float32),
        "matcher_total_cost": np.empty(frames, dtype=np.float32),
        "matcher_searched": np.empty(frames, dtype=np.uint8),
        "matcher_transitioned": np.empty(frames, dtype=np.uint8),
        "matcher_force_reason": np.empty(frames, dtype=np.int8),
    }
    source_clip_index = {
        clip.relative_path: index
        for index, clip in enumerate(matcher.folder.clips)
    }
    requested_path = np.zeros(2, dtype=np.float64)
    reason_codes = {None: 0, "clip_end": 1, "command_transition": 2}
    for frame in range(frames):
        local_velocity = trace.requested_velocity_local_xy[frame]
        root_yaw_before_step = matcher.current_root_world_yaw
        requested_velocity_world = _local_to_world_xy(
            local_velocity, root_yaw_before_step
        ).astype(np.float32)
        arrays["command_requested_velocity_world_xy"][frame] = (
            requested_velocity_world
        )
        result = matcher.step(
            tuple(float(value) for value in requested_velocity_world),
            float(trace.requested_heading_world_yaw[frame]),
        )
        arrays["joint_pos"][frame] = result.joint_position.detach().cpu().numpy()
        arrays["joint_vel"][frame] = result.joint_velocity.detach().cpu().numpy()
        arrays["body_pos_w"][frame] = result.body_position_world.detach().cpu().numpy()
        arrays["body_quat_w"][frame] = (
            result.body_orientation_world_wxyz.detach().cpu().numpy()
        )
        arrays["body_lin_vel_w"][frame] = (
            result.body_linear_velocity_world.detach().cpu().numpy()
        )
        arrays["body_ang_vel_w"][frame] = (
            result.body_angular_velocity_world.detach().cpu().numpy()
        )
        shaped_v_cpu = (
            result.applied_command_velocity_world_xy.detach().cpu().numpy()
        )
        shaped_h_cpu = float(
            result.applied_command_heading_world_yaw.detach().cpu().item()
        )
        arrays["command_shaped_velocity_world_xy"][frame] = shaped_v_cpu
        arrays["command_shaped_heading_world_yaw"][frame] = shaped_h_cpu
        requested_path += (
            requested_velocity_world.astype(np.float64)
            * matcher.config.dt
        )
        arrays["command_requested_path_world_xy"][frame] = requested_path
        ball_position = (
            result.simulation_ball_position_world_xy.detach().cpu().numpy()
        )
        ball_velocity = (
            result.simulation_ball_velocity_world_xy.detach().cpu().numpy()
        )
        ball_heading = float(
            result.simulation_ball_heading_world_yaw.detach().cpu().item()
        )
        arrays["command_path_world_xy"][frame] = ball_position
        arrays["command_ball_position_world_xy"][frame] = ball_position
        arrays["command_ball_velocity_world_xy"][frame] = ball_velocity
        arrays["command_ball_heading_world_yaw"][frame] = ball_heading
        arrays["command_ball_heading_velocity_rad_s"][frame] = float(
            result.simulation_ball_heading_velocity_rad_s.detach().cpu().item()
        )
        arrays["matcher_root_adjustment_distance_m"][frame] = float(
            result.root_adjustment_distance_m.detach().cpu().item()
        )
        arrays["matcher_root_adjustment_angle_rad"][frame] = float(
            result.root_adjustment_angle_rad.detach().cpu().item()
        )
        arrays["matcher_root_clamp_distance_m"][frame] = float(
            result.root_clamp_distance_m.detach().cpu().item()
        )
        arrays["matcher_root_clamp_angle_rad"][frame] = float(
            result.root_clamp_angle_rad.detach().cpu().item()
        )
        diag = result.diagnostics
        arrays["matcher_selected_clip_index"][frame] = source_clip_index[
            diag.selected_clip_path
        ]
        arrays["matcher_selected_frame"][frame] = diag.selected_frame
        arrays["matcher_feature_cost"][frame] = diag.selected_feature_cost
        arrays["matcher_total_cost"][frame] = diag.selected_total_cost
        arrays["matcher_searched"][frame] = diag.searched
        arrays["matcher_transitioned"][frame] = diag.transitioned
        if diag.force_search_reason not in reason_codes:
            raise ContractError(
                f"unknown matcher force reason {diag.force_search_reason!r}"
            )
        arrays["matcher_force_reason"][frame] = reason_codes[diag.force_search_reason]

    root_yaw = _yaw_from_wxyz(arrays["body_quat_w"][:, 0])
    arrays["task4_intended"] = _task4(
        arrays["command_shaped_velocity_world_xy"],
        arrays["command_shaped_heading_world_yaw"],
        root_yaw,
    )
    arrays["task4_realized_h24"] = _realized_task4(
        arrays["body_quat_w"],
        arrays["body_lin_vel_w"],
        horizon_frames=realized_horizon_frames,
    )
    arrays["task12_intended"] = _task12_intended(
        arrays["command_ball_velocity_world_xy"],
        arrays["command_ball_heading_world_yaw"],
        arrays["command_ball_heading_velocity_rad_s"],
    )
    return arrays


def mirror_clip(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Analytically mirror every kinematic and command field."""
    output = {name: value.copy() for name, value in arrays.items()}
    for name in ("joint_pos", "joint_vel"):
        output[name] = (
            arrays[name][:, JOINT_MIRROR_PERMUTATION] * JOINT_MIRROR_SIGNS
        ).astype(np.float32)
    for name in ("body_pos_w", "body_lin_vel_w"):
        value = arrays[name][:, BODY_MIRROR_PERMUTATION].copy()
        value[..., 1] *= -1.0
        output[name] = value
    angular = arrays["body_ang_vel_w"][:, BODY_MIRROR_PERMUTATION].copy()
    angular[..., 0] *= -1.0
    angular[..., 2] *= -1.0
    output["body_ang_vel_w"] = angular
    quaternion = arrays["body_quat_w"][:, BODY_MIRROR_PERMUTATION].copy()
    quaternion[..., 1] *= -1.0
    quaternion[..., 3] *= -1.0
    output["body_quat_w"] = quaternion

    output["command_raw_sticks"][:, (0, 2)] *= -1.0
    for name in (
        "command_requested_velocity_world_xy",
        "command_shaped_velocity_world_xy",
        "command_requested_path_world_xy",
        "command_path_world_xy",
        "command_ball_position_world_xy",
        "command_ball_velocity_world_xy",
    ):
        output[name][:, 1] *= -1.0
    output["command_requested_velocity_local_xy"][:, 1] *= -1.0
    for name in (
        "command_requested_heading_world_yaw",
        "command_shaped_heading_world_yaw",
        "command_ball_heading_world_yaw",
    ):
        output[name] *= -1.0
    output["command_ball_heading_velocity_rad_s"] *= -1.0
    for name in ("task4_intended", "task4_realized_h24"):
        output[name][:, 1] *= -1.0
        output[name][:, 3] *= -1.0
    task12 = output["task12_intended"].reshape(
        len(output["task12_intended"]), len(TASK12_KNOT_OFFSETS), 3
    )
    task12[:, :, (1, 2)] *= -1.0
    return output


def validate_mirror_pair(
    original: dict[str, np.ndarray],
    mirrored: dict[str, np.ndarray],
) -> None:
    """Fail if a supposedly paired clip is not the exact implemented reflection."""
    expected = mirror_clip(original)
    failures = []
    for name in sorted(expected):
        if expected[name].dtype.kind == "f":
            same = np.array_equal(expected[name], mirrored[name])
        else:
            same = np.array_equal(expected[name], mirrored[name])
        if not same:
            failures.append(name)
    if failures:
        raise ContractError(f"mirror pair mismatch for fields {failures}")
    if not np.allclose(
        np.linalg.norm(mirrored["body_quat_w"], axis=-1),
        1.0,
        atol=1e-5,
    ):
        raise ContractError("mirrored body quaternion is not unit length")


def _chunks(value: np.ndarray) -> tuple[int, ...]:
    return (min(600, len(value)),) + tuple(value.shape[1:])


def write_corpus(
    output: Path,
    clips: Iterable[tuple[CommandTrace, dict[str, np.ndarray], str | None]],
    *,
    source_npz: Path | None = None,
    source_motion_dir: Path | None = None,
    matcher: TorchMotionMatcher,
    matcher_config: MatcherConfig,
    curriculum_name: str = "base",
    source_quaternion_convention: str = "wxyz",
) -> None:
    import zarr

    if (source_npz is None) == (source_motion_dir is None):
        raise ContractError(
            "write_corpus requires exactly one source NPZ or source motion folder"
        )
    source_path = source_npz if source_npz is not None else source_motion_dir
    assert source_path is not None
    source_kind = "npz" if source_npz is not None else "motion_folder"

    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite corpus: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    values = list(clips)
    if not values:
        raise ContractError("cannot write an empty corpus")
    names = [trace.trace_id for trace, _arrays, _mirror_of in values]
    if len(names) != len(set(names)):
        raise ContractError("corpus clip IDs must be unique")
    keys = tuple(values[0][1])
    if any(tuple(arrays) != keys for _trace, arrays, _mirror in values):
        raise ContractError("corpus clip array schemas differ")
    episode_ends = np.cumsum(
        [trace.frame_count for trace, _arrays, _mirror in values],
        dtype=np.int64,
    )
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
    root = zarr.open_group(str(output), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    for key in keys:
        combined = np.concatenate([arrays[key] for _trace, arrays, _mirror in values])
        data.array(
            key,
            combined,
            chunks=_chunks(combined),
            compressor=compressor,
            overwrite=False,
        )
    string_width = max(len(name) for name in names)
    categories = [trace.category for trace, _arrays, _mirror in values]
    category_width = max(len(value) for value in categories)
    mirror_ids = [mirror_of or "" for _trace, _arrays, mirror_of in values]
    mirror_width = max(1, max(len(value) for value in mirror_ids))
    meta.array("episode_ends", episode_ends, overwrite=False)
    meta.array(
        "episode_clip",
        np.asarray(names, dtype=f"<U{string_width}"),
        overwrite=False,
    )
    meta.array(
        "episode_category",
        np.asarray(categories, dtype=f"<U{category_width}"),
        overwrite=False,
    )
    meta.array(
        "episode_mirror_of",
        np.asarray(mirror_ids, dtype=f"<U{mirror_width}"),
        overwrite=False,
    )
    meta.array(
        "episode_command_seed",
        np.asarray(
            [
                -1 if trace.seed is None else int(trace.seed)
                for trace, _arrays, _mirror in values
            ],
            dtype=np.int64,
        ),
        overwrite=False,
    )
    meta.array("fps", np.asarray([50], dtype=np.int64), overwrite=False)
    meta.array(
        "joint_names",
        np.asarray(JOINT_NAMES, dtype=f"<U{max(map(len, JOINT_NAMES))}"),
        overwrite=False,
    )
    meta.array(
        "body_names",
        np.asarray(BODY_NAMES, dtype=f"<U{max(map(len, BODY_NAMES))}"),
        overwrite=False,
    )
    source_motion_clips = tuple(
        clip.relative_path for clip in matcher.folder.clips
    )
    source_clip_width = max(len(path) for path in source_motion_clips)
    meta.array(
        "source_motion_clip",
        np.asarray(source_motion_clips, dtype=f"<U{source_clip_width}"),
        overwrite=False,
    )
    provenance = {
        "schema": CORPUS_SCHEMA,
        "curriculum_schema": CURRICULUM_SCHEMA,
        "curriculum_name": curriculum_name,
        "task4_schema": TASK4_SCHEMA,
        "task12_schema": TASK12_SCHEMA,
        "task12_knot_offsets": TASK12_KNOT_OFFSETS,
        "operator_velocity_frame": (
            "robot root yaw at each matcher step; no global position"
        ),
        "raw_stick_channels": RAW_STICK_CHANNELS,
        "button_channels": BUTTON_CHANNELS,
        "source_type": source_kind,
        "source_path": str(source_path),
        "source_quaternion_convention": source_quaternion_convention,
        "corpus_quaternion_convention": "wxyz",
        "source_inventory_sha256": matcher.motion_inventory_sha256,
        "motion_matching_commit": _git_head(Path(__file__).resolve().parents[3]),
        "matcher_config": asdict(matcher_config),
        "mirror_definition": {
            "plane": "world y=0",
            "linear_vector": [1, -1, 1],
            "pseudovector": [-1, 1, -1],
            "quaternion_wxyz": [1, -1, 1, -1],
            "joint_sign_rule": "roll/yaw negate; pitch unchanged; left/right swap",
        },
        "clip_count": len(values),
        "frame_count": int(episode_ends[-1]),
    }
    if "command_event_mask" in keys:
        event_mask = np.concatenate(
            [arrays["command_event_mask"] for _trace, arrays, _mirror in values]
        )
        event_bits = {
            "abrupt": COMMAND_EVENT_ABRUPT,
            "stop_or_restart": COMMAND_EVENT_STOP_OR_RESTART,
            "velocity_reverse": COMMAND_EVENT_VELOCITY_REVERSE,
            "heading_jump_ge_90deg": COMMAND_EVENT_HEADING_JUMP,
            "dual_stick": COMMAND_EVENT_DUAL_STICK,
        }
        provenance["command_event_bits"] = event_bits
        provenance["command_event_counts"] = {
            name: int(np.count_nonzero(event_mask & bit))
            for name, bit in event_bits.items()
        }
    if source_npz is not None:
        # Retain the historical keys for existing artifact readers.
        provenance["source_npz"] = str(source_npz)
        provenance["source_npz_sha256"] = _sha256(source_npz)
    else:
        provenance["source_motion_dir"] = str(source_motion_dir)
        source_manifest = source_motion_dir / "manifest.json"  # type: ignore[operator]
        if source_manifest.is_file():
            provenance["source_manifest_sha256"] = _sha256(source_manifest)
    root.attrs.update(
        {
            "schema": CORPUS_SCHEMA,
            "quaternion_convention": "wxyz",
            "provenance_json": json.dumps(provenance, sort_keys=True),
        }
    )


def _motion_folder_for_npz(
    source_npz: Path,
    *,
    source_quaternion_convention: str,
):
    if source_quaternion_convention not in {"wxyz", "xyzw"}:
        raise ContractError(
            "source quaternion convention must be 'wxyz' or 'xyzw'"
        )
    temporary = tempfile.TemporaryDirectory(prefix="takara-mm-source-")
    clip_dir = Path(temporary.name) / "takara"
    clip_dir.mkdir(parents=True)
    # Takara's canonical file is a legacy exception: its field is named
    # ``body_quat_w`` but its values match the verified Isaac-5 CSV in XYZW
    # order. MotionFolder has an intentionally strict WXYZ contract, so make
    # the convention conversion explicit at this one ingestion boundary.
    with np.load(source_npz, allow_pickle=False) as source:
        values = {name: np.asarray(source[name]) for name in source.files}
    if "body_quat_w" not in values:
        raise ContractError("source NPZ is missing body_quat_w")
    if source_quaternion_convention == "xyzw":
        values["body_quat_w"] = np.ascontiguousarray(
            values["body_quat_w"][..., (3, 0, 1, 2)]
        )
    np.savez(clip_dir / "motion.npz", **values)
    return temporary


def build_corpus(
    *,
    source_npz: Path | None = None,
    source_motion_dir: Path | None = None,
    output: Path,
    device: str,
    frames: int,
    base_start: int,
    base_stop: int | None,
    base_indices: tuple[int, ...] | None = None,
    curriculum: str = "base",
    pelvis_velocity_weight: float = 1.0,
    trajectory_position_weight: float = 1.0,
    trajectory_facing_weight: float = 1.5,
    trajectory_model: str = "takara_ball",
    max_source_joint_step_rad: float | None = 0.35,
    source_quaternion_convention: str = "xyzw",
) -> None:
    if (source_npz is None) == (source_motion_dir is None):
        raise ContractError(
            "build_corpus requires exactly one source NPZ or source motion folder"
        )
    resolved_npz: Path | None = None
    resolved_motion_dir: Path | None = None
    if source_npz is not None:
        resolved_npz = source_npz.expanduser().resolve()
        if not resolved_npz.is_file():
            raise FileNotFoundError(resolved_npz)
    else:
        assert source_motion_dir is not None
        resolved_motion_dir = source_motion_dir.expanduser().resolve()
        if not resolved_motion_dir.is_dir():
            raise FileNotFoundError(resolved_motion_dir)
        if source_quaternion_convention != "wxyz":
            raise ContractError(
                "native source motion folders must use WXYZ quaternions"
            )
    if curriculum == "base":
        traces = build_base_curriculum(frames=frames)
    elif curriculum == "omnidirectional":
        traces = build_omnidirectional_curriculum(frames=frames)
    elif curriculum == "abrupt":
        traces = build_abrupt_curriculum(frames=frames)
    else:
        raise ContractError(f"unknown command curriculum: {curriculum}")
    if base_indices is not None:
        if (
            not base_indices
            or len(base_indices) != len(set(base_indices))
            or min(base_indices) < 0
            or max(base_indices) >= len(traces)
        ):
            raise ContractError("base indices must be unique curriculum indices")
        selected = tuple(traces[index] for index in base_indices)
    else:
        stop = len(traces) if base_stop is None else int(base_stop)
        if not 0 <= base_start < stop <= len(traces):
            raise ContractError(
                f"base range [{base_start},{stop}) outside [0,{len(traces)})"
            )
        selected = traces[base_start:stop]
    config = MatcherConfig(
        trajectory_model=trajectory_model,
        max_source_joint_step_rad=max_source_joint_step_rad,
        feature_weight_overrides=(
            ("pelvis_velocity", float(pelvis_velocity_weight)),
            ("trajectory_position", float(trajectory_position_weight)),
            ("trajectory_facing", float(trajectory_facing_weight)),
        )
    )
    temporary = None
    if resolved_npz is not None:
        temporary = _motion_folder_for_npz(
            resolved_npz,
            source_quaternion_convention=source_quaternion_convention,
        )
        matcher_root = Path(temporary.name)
    else:
        assert resolved_motion_dir is not None
        matcher_root = resolved_motion_dir
    try:
        matcher = TorchMotionMatcher.from_folder(
            matcher_root, device=device, config=config
        )
        clips = []
        for trace in selected:
            arrays = generate_clip(matcher, trace)
            clips.append((trace, arrays, None))
            mirrored_arrays = mirror_clip(arrays)
            validate_mirror_pair(arrays, mirrored_arrays)
            mirrored_trace = CommandTrace(
                trace_id=f"{trace.trace_id}__mirror",
                category=trace.category,
                raw_sticks=mirrored_arrays["command_raw_sticks"],
                buttons=mirrored_arrays["command_buttons"],
                requested_velocity_local_xy=mirrored_arrays[
                    "command_requested_velocity_local_xy"
                ],
                requested_heading_world_yaw=mirrored_arrays[
                    "command_requested_heading_world_yaw"
                ],
                seed=trace.seed,
            )
            clips.append((mirrored_trace, mirrored_arrays, trace.trace_id))
        write_corpus(
            output,
            clips,
            source_npz=resolved_npz,
            source_motion_dir=resolved_motion_dir,
            matcher=matcher,
            matcher_config=config,
            curriculum_name=curriculum,
            source_quaternion_convention=source_quaternion_convention,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion-npz", type=Path)
    source.add_argument(
        "--motions-dir",
        type=Path,
        help="native WXYZ motion folder containing one or more */motion.npz clips",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument(
        "--curriculum",
        choices=("base", "omnidirectional", "abrupt"),
        default="base",
        help="command curriculum to synthesize before exact mirror pairing",
    )
    parser.add_argument("--base-start", type=int, default=0)
    parser.add_argument("--base-stop", type=int)
    parser.add_argument(
        "--base-indices",
        help="comma-separated base curriculum indices; overrides --base-start/--base-stop",
    )
    parser.add_argument("--pelvis-velocity-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-position-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-facing-weight", type=float, default=1.5)
    parser.add_argument(
        "--trajectory-model",
        choices=("legacy", "takara_ball"),
        default="takara_ball",
        help="persistent Takara simulation ball or the legacy bounded integrator",
    )
    parser.add_argument(
        "--max-source-joint-step-rad",
        type=float,
        default=0.35,
        help=(
            "reject candidate windows containing a larger 20 ms source-joint "
            "step; set to 0 to disable"
        ),
    )
    parser.add_argument(
        "--source-quaternion-convention",
        choices=("xyzw", "wxyz"),
        default=None,
        help=(
            "storage order of source body_quat_w; Takara's legacy NPZ is XYZW "
            "despite the field name; native motion folders are WXYZ"
        ),
    )
    args = parser.parse_args()
    base_indices = (
        tuple(int(value) for value in args.base_indices.split(","))
        if args.base_indices
        else None
    )
    build_corpus(
        source_npz=args.motion_npz,
        source_motion_dir=args.motions_dir,
        output=args.output,
        device=args.device,
        frames=args.frames,
        base_start=args.base_start,
        base_stop=args.base_stop,
        base_indices=base_indices,
        curriculum=args.curriculum,
        pelvis_velocity_weight=args.pelvis_velocity_weight,
        trajectory_position_weight=args.trajectory_position_weight,
        trajectory_facing_weight=args.trajectory_facing_weight,
        trajectory_model=args.trajectory_model,
        max_source_joint_step_rad=(
            None
            if args.max_source_joint_step_rad == 0.0
            else args.max_source_joint_step_rad
        ),
        source_quaternion_convention=(
            args.source_quaternion_convention
            or ("wxyz" if args.motions_dir is not None else "xyzw")
        ),
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
