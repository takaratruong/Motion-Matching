"""Shared exact Torch feature extractor and immutable motion database.

This module owns the single 27-value search feature definition used by both the
database-row path and the runtime query path. There is no separate approximate
online implementation: ``TorchMotionDatabase.from_folder`` and
``extract_query_features`` call the same ``_feature_row`` math.

Everything stays in the native Takara convention: Z-up world, wxyz quaternions,
and the G1/MuJoCo positive-X forward axis. Positions are translated by the
current root position and rotated by the inverse current root yaw; velocities
are rotated but not translated; future trajectory groups are expressed relative
to the current root in the same heading frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .joints import ContractError
from .torch_motion_data import MotionClip, MotionFolder

# The three future horizons, in frames, at 50 Hz: 0.3 / 0.6 / 0.9 seconds.
FEATURE_HORIZON_FRAMES = (15, 30, 45)

# Frozen 27-component group order: (name, flat slice, initial weight).
FEATURE_GROUPS = (
    ("left_foot_position", slice(0, 3), 0.75),
    ("right_foot_position", slice(3, 6), 0.75),
    ("left_foot_velocity", slice(6, 9), 1.0),
    ("right_foot_velocity", slice(9, 12), 1.0),
    ("pelvis_velocity", slice(12, 15), 1.0),
    ("trajectory_position", slice(15, 21), 1.0),
    ("trajectory_facing", slice(21, 27), 1.5),
)

FEATURE_DIM = 27


@dataclass(frozen=True)
class GeneratedFeatureState:
    root_position_world: torch.Tensor
    root_orientation_world_wxyz: torch.Tensor
    root_linear_velocity_world: torch.Tensor
    left_foot_position_world: torch.Tensor
    right_foot_position_world: torch.Tensor
    left_foot_velocity_world: torch.Tensor
    right_foot_velocity_world: torch.Tensor


@dataclass(frozen=True)
class CommandTrajectory:
    position_world_xy: torch.Tensor
    facing_world_xy: torch.Tensor


@dataclass(frozen=True)
class FeatureNormalization:
    _component_mean: torch.Tensor
    _component_scale: torch.Tensor

    def normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return (raw - self._component_mean) / self._component_scale

    def parameters_copy(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._component_mean.clone(), self._component_scale.clone()


@dataclass(frozen=True)
class TorchMotionDatabase:
    folder: MotionFolder
    device: torch.device
    normalization: FeatureNormalization
    reset_row: int
    _search_features: torch.Tensor
    _search_clip_index: torch.Tensor
    _search_frame_index: torch.Tensor
    # Private ``(clip_index, frame_index) -> global row`` provenance mapping.
    _source_row_map: "dict[tuple[int, int], int]"

    @property
    def feature_shape(self) -> tuple[int, int]:
        return tuple(self._search_features.shape)

    def normalized_features_copy(self) -> torch.Tensor:
        return self._search_features.clone()

    def row_for_source(self, clip_index: int, frame_index: int) -> "int | None":
        return self._source_row_map.get((int(clip_index), int(frame_index)))

    @staticmethod
    def from_folder(
        folder: MotionFolder, *, device: "str | torch.device"
    ) -> "TorchMotionDatabase":
        return _build_database(folder, device)


# ---------------------------------------------------------------------------
# Quaternion and per-sample feature primitives (Z-up, wxyz).
# ---------------------------------------------------------------------------


def _yaw_from_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Z-up yaw of one or many wxyz quaternions."""
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _inverse_yaw_rotate_xy(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate trailing-dim-2 planar vectors by ``-yaw`` into the heading frame."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x = xy[..., 0]
    y = xy[..., 1]
    rx = c * x + s * y
    ry = -s * x + c * y
    return torch.stack((rx, ry), dim=-1)


def _feature_row(state: GeneratedFeatureState, trajectory: CommandTrajectory) -> torch.Tensor:
    """Compute the flat 27-value feature vector for one sample.

    This is the single shared feature math for both database rows and runtime
    queries.
    """
    root_pos = state.root_position_world
    yaw = _yaw_from_wxyz(state.root_orientation_world_wxyz)
    device = root_pos.device
    dtype = root_pos.dtype

    parts: list[torch.Tensor] = []

    # Groups 0-1: foot positions relative to root, planar-rotated into heading.
    for foot in (state.left_foot_position_world, state.right_foot_position_world):
        rel = foot - root_pos
        rel_xy = _inverse_yaw_rotate_xy(rel[..., :2], yaw)
        parts.append(torch.cat((rel_xy, rel[..., 2:3]), dim=-1))

    # Groups 2-4: foot and pelvis world velocities rotated (not translated).
    for vel in (
        state.left_foot_velocity_world,
        state.right_foot_velocity_world,
        state.root_linear_velocity_world,
    ):
        vel_xy = _inverse_yaw_rotate_xy(vel[..., :2], yaw)
        parts.append(torch.cat((vel_xy, vel[..., 2:3]), dim=-1))

    # Group 5: future root XY positions relative to current root in heading frame.
    rel_traj = trajectory.position_world_xy - root_pos[..., :2].unsqueeze(-2)
    rot_traj = _inverse_yaw_rotate_xy(rel_traj, yaw.unsqueeze(-1))
    parts.append(rot_traj.flatten(start_dim=-2))

    # Group 6: future facing XY directions rotated into the heading frame.
    rot_face = _inverse_yaw_rotate_xy(
        trajectory.facing_world_xy, yaw.unsqueeze(-1)
    )
    parts.append(rot_face.flatten(start_dim=-2))

    row = torch.cat(parts, dim=-1)
    return row.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Query entry point with strict validation.
# ---------------------------------------------------------------------------


def _require_vec3(tensor: torch.Tensor, name: str, ref: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ContractError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != (3,):
        raise ContractError(f"{name} must have shape (3,), got {tuple(tensor.shape)}")
    _require_agree(tensor, name, ref)


def _require_agree(tensor: torch.Tensor, name: str, ref: torch.Tensor) -> None:
    if tensor.dtype != ref.dtype:
        raise ContractError(f"{name} dtype {tensor.dtype} != {ref.dtype}")
    if tensor.device != ref.device:
        raise ContractError(f"{name} device {tensor.device} != {ref.device}")
    if not torch.isfinite(tensor).all():
        raise ContractError(f"{name} contains non-finite values")


def extract_query_features(
    state: GeneratedFeatureState, trajectory: CommandTrajectory
) -> torch.Tensor:
    """Return the shared 27-value feature vector for a runtime query."""
    ref = state.root_position_world
    if not isinstance(ref, torch.Tensor) or tuple(ref.shape) != (3,):
        raise ContractError("root_position_world must have shape (3,)")
    if not torch.isfinite(ref).all():
        raise ContractError("root_position_world contains non-finite values")

    if tuple(state.root_orientation_world_wxyz.shape) != (4,):
        raise ContractError("root_orientation_world_wxyz must have shape (4,)")
    _require_agree(state.root_orientation_world_wxyz, "root_orientation_world_wxyz", ref)

    for name in (
        "root_linear_velocity_world",
        "left_foot_position_world",
        "right_foot_position_world",
        "left_foot_velocity_world",
        "right_foot_velocity_world",
    ):
        _require_vec3(getattr(state, name), name, ref)

    for name in ("position_world_xy", "facing_world_xy"):
        tensor = getattr(trajectory, name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (3, 2):
            raise ContractError(f"{name} must have shape (3, 2)")
        _require_agree(tensor, name, ref)

    return _feature_row(state, trajectory)


# ---------------------------------------------------------------------------
# Database construction.
# ---------------------------------------------------------------------------

def _clip_feature_rows(
    clip: MotionClip, device: torch.device, layout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer one clip once and extract all searchable rows as one batch."""
    # The loader owns read-only NumPy arrays. These four calls make one owned
    # device copy per source array, independent of the clip's frame count.
    body_pos = torch.tensor(
        clip.body_position_world, dtype=torch.float32, device=device
    )
    body_quat = torch.tensor(
        clip.body_quaternion_world_wxyz, dtype=torch.float32, device=device
    )
    body_lin = torch.tensor(
        clip.body_linear_velocity_world, dtype=torch.float32, device=device
    )
    joint_vel = torch.tensor(
        clip.joint_velocity, dtype=torch.float32, device=device
    )
    root = layout.root_body_index
    lf = layout.left_foot_body_index
    rf = layout.right_foot_body_index
    frames = torch.arange(clip.valid_frame_stop, device=device)

    state = GeneratedFeatureState(
        root_position_world=body_pos[frames, root],
        root_orientation_world_wxyz=body_quat[frames, root],
        root_linear_velocity_world=body_lin[frames, root],
        left_foot_position_world=body_pos[frames, lf],
        right_foot_position_world=body_pos[frames, rf],
        left_foot_velocity_world=body_lin[frames, lf],
        right_foot_velocity_world=body_lin[frames, rf],
    )

    offsets = torch.tensor(FEATURE_HORIZON_FRAMES, device=device)
    future_frames = frames[:, None] + offsets[None, :]
    future_root = body_pos[future_frames, root]
    future_quat = body_quat[future_frames, root]
    future_yaw = _yaw_from_wxyz(future_quat)
    facing_xy = torch.stack((torch.cos(future_yaw), torch.sin(future_yaw)), dim=-1)
    trajectory = CommandTrajectory(
        position_world_xy=future_root[..., :2].contiguous(),
        facing_world_xy=facing_xy.contiguous(),
    )
    joint_velocity_sq = (joint_vel[frames] ** 2).sum(dim=-1)
    return _feature_row(state, trajectory), joint_velocity_sq


def _build_database(
    folder: MotionFolder, device: "str | torch.device"
) -> TorchMotionDatabase:
    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    layout = folder.layout

    rows: list[torch.Tensor] = []
    clip_indices: list[int] = []
    frame_indices: list[int] = []
    joint_velocity_sq: list[torch.Tensor] = []
    source_row_map: dict[tuple[int, int], int] = {}

    for clip_index, clip in enumerate(folder.clips):
        clip_rows, clip_joint_velocity_sq = _clip_feature_rows(
            clip, resolved, layout
        )
        rows.append(clip_rows)
        joint_velocity_sq.append(clip_joint_velocity_sq)
        first_row = sum(item.shape[0] for item in rows[:-1])
        for frame in range(clip.valid_frame_stop):
            global_row = first_row + frame
            source_row_map[(clip_index, frame)] = global_row
            clip_indices.append(clip_index)
            frame_indices.append(frame)

    if not rows:
        raise ContractError("motion folder produced no searchable feature rows")

    features = torch.cat(rows, dim=0).to(device=resolved, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ContractError("search features contain non-finite values")

    component_mean = features.mean(dim=0)
    component_std = features.std(dim=0, unbiased=False)
    scale = torch.empty_like(component_mean)
    for name, group_slice, weight in FEATURE_GROUPS:
        group_std = component_std[group_slice].mean()
        group_scale = group_std / weight
        if not torch.isfinite(group_scale) or group_scale <= 0.0:
            raise ContractError(
                f"feature group {name!r} has non-positive or non-finite scale "
                f"{float(group_scale)}"
            )
        scale[group_slice] = group_scale

    normalized = (features - component_mean) / scale
    normalized.requires_grad_(False)

    # reset_row: minimum joint-velocity squared norm, smallest global row tie break.
    jv = torch.cat(joint_velocity_sq)
    reset_row = int(torch.argmin(jv).item())

    normalization = FeatureNormalization(
        _component_mean=component_mean.detach(),
        _component_scale=scale.detach(),
    )
    database = TorchMotionDatabase(
        folder=folder,
        device=resolved,
        normalization=normalization,
        reset_row=reset_row,
        _search_features=normalized.detach(),
        _search_clip_index=torch.tensor(clip_indices, device=resolved),
        _search_frame_index=torch.tensor(frame_indices, device=resolved),
        _source_row_map=source_row_map,
    )
    return database
