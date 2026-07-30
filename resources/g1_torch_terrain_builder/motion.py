"""Strict native-motion loading and endpoint-preserving 50 Hz conversion."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from mm_sonic.joints import TARGET_JOINT_ORDER
from resources.g1_terrain_builder.database import derive_velocities
from resources.g1_terrain_builder.resample import (
    resample_quaternions_wxyz,
    resample_vectors,
)
from resources.g1_torch_stair_builder.conversion import NativeMotionArrays


TARGET_FPS = 50
LAYOUT = "g1-29dof-isaaclab-v1"
_REQUIRED = frozenset(
    (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
)


def _finite_array(
    fields: dict[str, np.ndarray],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(fields[name])
    if value.shape != shape or value.dtype.kind not in "iuf":
        raise ValueError(f"{name} shape must be {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(value, np.float64)


def _rate(fields: dict[str, np.ndarray], expected: float | None) -> float:
    raw = np.asarray(fields["fps"])
    if raw.size != 1 or raw.dtype.kind not in "iuf":
        raise ValueError("motion fps must contain one numeric value")
    value = float(raw.reshape(-1)[0])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("motion fps must be finite and positive")
    if expected is not None:
        expected_value = float(expected)
        if (
            not math.isfinite(expected_value)
            or expected_value <= 0.0
            or abs(expected_value - value) > 1e-6
        ):
            raise ValueError(
                f"source_fps {expected!r} does not match archive fps {value}"
            )
    return value


def _prove_layout(
    fields: dict[str, np.ndarray], layout_provenance: str | None
) -> None:
    if layout_provenance is not None and layout_provenance != LAYOUT:
        raise ValueError(f"layout provenance must equal {LAYOUT}")
    if "joint_names" in fields:
        names = np.asarray(fields["joint_names"])
        if names.shape != (29,) or tuple(str(name) for name in names) != tuple(
            TARGET_JOINT_ORDER
        ):
            raise ValueError("joint_names do not match the pinned G1 layout")
    elif layout_provenance != LAYOUT:
        raise ValueError(
            "layout provenance is required when joint_names are absent"
        )


def _endpoint_resample_count(frames: int, source_fps: float) -> int:
    return int(round((frames - 1) * TARGET_FPS / source_fps)) + 1


def _endpoint_vectors(
    value: np.ndarray, output_frames: int
) -> np.ndarray:
    if len(value) == output_frames:
        return np.asarray(value, np.float64).copy()
    return resample_vectors(
        value,
        float(len(value) - 1),
        float(output_frames - 1),
    )


def _endpoint_quaternions(
    value: np.ndarray, output_frames: int
) -> np.ndarray:
    return resample_quaternions_wxyz(
        value,
        float(len(value) - 1),
        float(output_frames - 1),
    )


def load_native_motion_50hz(
    path: str | Path,
    *,
    source_fps: float | None = None,
    layout_provenance: str | None = None,
) -> NativeMotionArrays:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"motion source must be a real file: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = _REQUIRED - set(archive.files)
            if missing:
                raise ValueError(
                    f"motion archive is missing fields: {sorted(missing)}"
                )
            fields = {name: archive[name] for name in archive.files}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"cannot load motion archive: {source}") from error

    fps = _rate(fields, source_fps)
    _prove_layout(fields, layout_provenance)
    joint_raw = np.asarray(fields["joint_pos"])
    if joint_raw.ndim != 2 or joint_raw.shape[1:] != (29,):
        raise ValueError("joint_pos shape must be (frames, 29)")
    frames = len(joint_raw)
    if frames < 3:
        raise ValueError("motion source must contain at least three frames")
    joint = _finite_array(fields, "joint_pos", (frames, 29))
    body = _finite_array(fields, "body_pos_w", (frames, 30, 3))
    quaternion = _finite_array(
        fields, "body_quat_w", (frames, 30, 4)
    )
    for name, shape in (
        ("joint_vel", (frames, 29)),
        ("body_lin_vel_w", (frames, 30, 3)),
        ("body_ang_vel_w", (frames, 30, 3)),
    ):
        _finite_array(fields, name, shape)
    norms = np.linalg.norm(quaternion, axis=-1)
    if np.any(norms < 1e-12) or np.max(np.abs(norms - 1.0)) > 1e-3:
        raise ValueError("source body quaternion norm is invalid")

    output_frames = _endpoint_resample_count(frames, fps)
    if output_frames < 46:
        raise ValueError("native 50 Hz motion must contain at least 46 frames")
    joint_out = _endpoint_vectors(joint, output_frames)
    body_out = _endpoint_vectors(body, output_frames)
    quaternion_out = _endpoint_quaternions(quaternion, output_frames)
    joint_velocity = (
        np.gradient(joint_out, axis=0, edge_order=2) * TARGET_FPS
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        body_linear_velocity, body_angular_velocity = derive_velocities(
            body_out, quaternion_out, TARGET_FPS
        )
    output = NativeMotionArrays(
        fps=TARGET_FPS,
        joint_position=np.ascontiguousarray(joint_out, np.float32),
        joint_velocity=np.ascontiguousarray(joint_velocity, np.float32),
        body_position_world=np.ascontiguousarray(body_out, np.float32),
        body_quaternion_world_wxyz=np.ascontiguousarray(
            quaternion_out, np.float32
        ),
        body_linear_velocity_world=np.ascontiguousarray(
            body_linear_velocity, np.float32
        ),
        body_angular_velocity_world=np.ascontiguousarray(
            body_angular_velocity, np.float32
        ),
    )
    output.validate()
    return output
