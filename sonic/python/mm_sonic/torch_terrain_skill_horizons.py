"""Fixed-horizon outcome descriptors for coherent terrain skill entries."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from .joints import ContractError
from .torch_motion_features import TorchMotionDatabase
from .torch_terrain_features import TerrainDataset
from .torch_terrain_skills import TerrainSkillInventory


HORIZON_TARGET_FRAMES = (25, 50, 100)
MAXIMUM_ENDPOINT_LATENESS_FRAMES = 25
STALL_SPEED_MPS = 0.03
DT_S = 0.02


@dataclass(frozen=True)
class TerrainSkillHorizonRecord:
    root_displacement_local_xy: torch.Tensor
    yaw_delta_rad: float
    root_height_delta_m: float
    surface_height_delta_m: torch.Tensor
    maximum_stall_frames: int
    duration_frames: int


@dataclass(frozen=True)
class TerrainSkillHorizonInventory:
    """Flat, device-resident outcomes owned by terrain-skill entry rows."""

    entry_row: torch.Tensor
    skill_index: torch.Tensor
    clip_index: torch.Tensor
    entry_frame: torch.Tensor
    target_frames: torch.Tensor
    endpoint_frame_exclusive: torch.Tensor
    root_displacement_local_xy: torch.Tensor
    yaw_delta_rad: torch.Tensor
    root_height_delta_m: torch.Tensor
    surface_height_delta_m: torch.Tensor
    maximum_stall_frames: torch.Tensor
    duration_frames: torch.Tensor
    rejected_by_reason: Mapping[str, int]

    @property
    def record_count(self) -> int:
        return int(self.entry_row.shape[0])


def stable_horizon_endpoints(
    support_mask: torch.Tensor,
    *,
    entry_frame: int,
    playback_stop: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(target frames, exclusive stable endpoint)`` pairs."""

    if (
        not isinstance(support_mask, torch.Tensor)
        or support_mask.dtype != torch.bool
        or support_mask.ndim != 2
        or support_mask.shape[1] != 2
    ):
        raise ContractError("horizon support mask must have boolean shape (T, 2)")
    frame_count = int(support_mask.shape[0])
    if (
        type(entry_frame) is not int
        or type(playback_stop) is not int
        or not 0 <= entry_frame < playback_stop <= frame_count
    ):
        raise ContractError("horizon source bounds are invalid")
    stable_frames = frozenset(
        int(frame)
        for frame in torch.nonzero(
            support_mask.all(dim=1), as_tuple=False
        ).flatten().cpu().tolist()
    )
    output: list[tuple[int, int]] = []
    for target in HORIZON_TARGET_FRAMES:
        first = entry_frame + target
        final = min(
            playback_stop - 1,
            first + MAXIMUM_ENDPOINT_LATENESS_FRAMES,
        )
        endpoint = next(
            (frame for frame in range(first, final + 1) if frame in stable_frames),
            None,
        )
        if endpoint is not None:
            output.append((target, endpoint + 1))
    return tuple(output)


def next_sequential_horizon_endpoint(
    support_mask: torch.Tensor,
    *,
    current_endpoint_frame_exclusive: int,
    playback_stop: int,
    target_frames: int = 25,
) -> int | None:
    """Return the first later stable endpoint for the same placed skill."""

    if (
        not isinstance(support_mask, torch.Tensor)
        or support_mask.dtype != torch.bool
        or support_mask.ndim != 2
        or support_mask.shape[1] != 2
    ):
        raise ContractError("sequential horizon support mask must have boolean shape (T, 2)")
    frame_count = int(support_mask.shape[0])
    if (
        type(current_endpoint_frame_exclusive) is not int
        or type(playback_stop) is not int
        or type(target_frames) is not int
        or target_frames < 1
        or not 0 < current_endpoint_frame_exclusive < playback_stop <= frame_count
    ):
        raise ContractError("sequential horizon source bounds are invalid")
    first = current_endpoint_frame_exclusive + target_frames - 1
    final = min(
        playback_stop - 1,
        first + MAXIMUM_ENDPOINT_LATENESS_FRAMES,
    )
    for frame in range(first, final + 1):
        if bool(support_mask[frame].all().item()):
            return frame + 1
    return None


def remaining_stall_profile(root_position_world_xy: torch.Tensor) -> torch.Tensor:
    """Longest consecutive future low-speed run beginning at each frame."""

    if (
        not isinstance(root_position_world_xy, torch.Tensor)
        or not root_position_world_xy.dtype.is_floating_point
        or root_position_world_xy.ndim != 2
        or root_position_world_xy.shape[1] != 2
        or root_position_world_xy.shape[0] < 1
        or not torch.isfinite(root_position_world_xy).all()
    ):
        raise ContractError("horizon root positions must have finite shape (T, 2)")
    frame_count = int(root_position_world_xy.shape[0])
    low = torch.ones(
        frame_count,
        dtype=torch.bool,
        device=root_position_world_xy.device,
    )
    if frame_count > 1:
        speed = torch.linalg.vector_norm(
            root_position_world_xy[1:] - root_position_world_xy[:-1], dim=1
        ) / DT_S
        low[:-1] = speed <= STALL_SPEED_MPS
    low_values = low.detach().cpu().tolist()
    profile = [0] * frame_count
    run = 0
    for frame in range(frame_count - 1, -1, -1):
        run = run + 1 if low_values[frame] else 0
        profile[frame] = run
    return torch.tensor(
        profile,
        dtype=torch.long,
        device=root_position_world_xy.device,
    )


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def describe_horizon(
    *,
    root_position_world: torch.Tensor,
    root_orientation_world_wxyz: torch.Tensor,
    foot_surface_height_m: torch.Tensor,
    entry_frame: int,
    endpoint_frame_exclusive: int,
) -> TerrainSkillHorizonRecord:
    """Describe one bounded source outcome in its entry-heading frame."""

    if (
        not isinstance(root_position_world, torch.Tensor)
        or root_position_world.ndim != 2
        or root_position_world.shape[1] != 3
        or not root_position_world.dtype.is_floating_point
        or not torch.isfinite(root_position_world).all()
    ):
        raise ContractError("horizon roots must have finite shape (T, 3)")
    frame_count = int(root_position_world.shape[0])
    if (
        not isinstance(root_orientation_world_wxyz, torch.Tensor)
        or tuple(root_orientation_world_wxyz.shape) != (frame_count, 4)
        or root_orientation_world_wxyz.device != root_position_world.device
        or root_orientation_world_wxyz.dtype != root_position_world.dtype
        or not torch.isfinite(root_orientation_world_wxyz).all()
        or not isinstance(foot_surface_height_m, torch.Tensor)
        or tuple(foot_surface_height_m.shape) != (frame_count, 2)
        or foot_surface_height_m.device != root_position_world.device
        or foot_surface_height_m.dtype != root_position_world.dtype
        or not torch.isfinite(foot_surface_height_m).all()
    ):
        raise ContractError("horizon orientation or surface profile is invalid")
    if (
        type(entry_frame) is not int
        or type(endpoint_frame_exclusive) is not int
        or not 0 <= entry_frame < endpoint_frame_exclusive <= frame_count
    ):
        raise ContractError("horizon descriptor bounds are invalid")

    final = endpoint_frame_exclusive - 1
    yaw = _yaw_from_wxyz(
        root_orientation_world_wxyz[entry_frame:endpoint_frame_exclusive]
    )
    entry_yaw = yaw[0]
    displacement = root_position_world[final, :2] - root_position_world[entry_frame, :2]
    c, s = torch.cos(entry_yaw), torch.sin(entry_yaw)
    local = torch.stack(
        (
            c * displacement[0] + s * displacement[1],
            -s * displacement[0] + c * displacement[1],
        )
    )
    yaw_delta = torch.atan2(
        torch.sin(yaw[1:] - yaw[:-1]),
        torch.cos(yaw[1:] - yaw[:-1]),
    ).sum()
    stall = remaining_stall_profile(
        root_position_world[entry_frame:endpoint_frame_exclusive, :2]
    )
    return TerrainSkillHorizonRecord(
        root_displacement_local_xy=local,
        yaw_delta_rad=float(yaw_delta.item()),
        root_height_delta_m=float(
            (root_position_world[final, 2] - root_position_world[entry_frame, 2]).item()
        ),
        surface_height_delta_m=(
            foot_surface_height_m[final] - foot_surface_height_m[entry_frame]
        ).clone(),
        maximum_stall_frames=int(stall.max().item()),
        duration_frames=endpoint_frame_exclusive - entry_frame,
    )


def _numpy_yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _maximum_stall_frames(root_xy: np.ndarray) -> int:
    if root_xy.shape[0] == 1:
        return 1
    speed = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) / DT_S
    low = np.concatenate((speed <= STALL_SPEED_MPS, np.ones(1, dtype=bool)))
    longest = run = 0
    for value in low:
        run = run + 1 if bool(value) else 0
        longest = max(longest, run)
    return longest


def build_horizon_inventory(
    dataset: TerrainDataset,
    database: TorchMotionDatabase,
    skill_inventory: TerrainSkillInventory,
) -> TerrainSkillHorizonInventory:
    """Build deterministic fixed-horizon outcomes without per-row GPU syncs."""

    if not isinstance(dataset, TerrainDataset):
        raise ContractError("horizon inventory requires a TerrainDataset")
    if not isinstance(database, TorchMotionDatabase):
        raise ContractError("horizon inventory requires a TorchMotionDatabase")
    if not isinstance(skill_inventory, TerrainSkillInventory):
        raise ContractError("horizon inventory requires a TerrainSkillInventory")
    if database.folder is not dataset.folder and (
        database.folder.inventory_sha256 != dataset.folder.inventory_sha256
    ):
        raise ContractError("horizon database does not match terrain dataset")

    integer_values: dict[str, list[int]] = {
        "entry_row": [],
        "skill_index": [],
        "clip_index": [],
        "entry_frame": [],
        "target_frames": [],
        "endpoint_frame_exclusive": [],
        "maximum_stall_frames": [],
        "duration_frames": [],
    }
    displacement_values: list[list[float]] = []
    yaw_values: list[float] = []
    root_height_values: list[float] = []
    surface_values: list[list[float]] = []
    rejected: dict[str, int] = {}
    seen_pairs: set[tuple[int, int]] = set()

    row_clip = database._search_clip_index.detach().cpu().tolist()
    row_frame = database._search_frame_index.detach().cpu().tolist()
    row_count = len(row_clip)
    root_body = dataset.folder.layout.root_body_index

    for skill in skill_inventory.skills:
        clip_index = int(skill.clip_index)
        if not 0 <= clip_index < len(dataset.folder.clips):
            rejected["invalid_source"] = rejected.get("invalid_source", 0) + len(
                skill.entry_rows
            )
            continue
        clip = dataset.folder.clips[clip_index]
        roots = np.asarray(
            clip.body_position_world[:, root_body], dtype=np.float64
        )
        yaw = np.unwrap(
            _numpy_yaw_from_wxyz(
                np.asarray(
                    clip.body_quaternion_world_wxyz[:, root_body],
                    dtype=np.float64,
                )
            )
        )
        surface = (
            skill.foot_surface_height_m.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
        )
        stable_frames = np.flatnonzero(
            skill.support_mask.detach().cpu().numpy().all(axis=1)
        )
        if (
            roots.shape != (clip.frame_count, 3)
            or yaw.shape != (clip.frame_count,)
            or surface.shape != (clip.frame_count, 2)
        ):
            raise ContractError("horizon source arrays have invalid shapes")

        for raw_row in skill.entry_rows:
            row = int(raw_row)
            if (
                not 0 <= row < row_count
                or skill_inventory.row_to_skill.get(row) != skill.skill_index
                or int(row_clip[row]) != clip_index
            ):
                rejected["invalid_source"] = rejected.get("invalid_source", 0) + 1
                continue
            entry = int(row_frame[row])
            if not skill.interval.entry_start <= entry < skill.interval.playback_stop:
                rejected["invalid_source"] = rejected.get("invalid_source", 0) + 1
                continue
            endpoints: list[tuple[int, int]] = []
            for target in HORIZON_TARGET_FRAMES:
                first = entry + target
                if first >= skill.interval.playback_stop:
                    rejected["too_short"] = rejected.get("too_short", 0) + 1
                    continue
                final = min(
                    skill.interval.playback_stop - 1,
                    first + MAXIMUM_ENDPOINT_LATENESS_FRAMES,
                )
                stable_index = int(np.searchsorted(stable_frames, first))
                if (
                    stable_index < stable_frames.shape[0]
                    and int(stable_frames[stable_index]) <= final
                ):
                    endpoints.append(
                        (target, int(stable_frames[stable_index]) + 1)
                    )
                else:
                    rejected["no_endpoint"] = rejected.get("no_endpoint", 0) + 1
            for target, endpoint in endpoints:
                pair = (row, target)
                if pair in seen_pairs:
                    raise ContractError("duplicate horizon entry ownership")
                seen_pairs.add(pair)
                final = endpoint - 1
                delta_xy = roots[final, :2] - roots[entry, :2]
                c, s = np.cos(yaw[entry]), np.sin(yaw[entry])
                displacement_values.append(
                    [
                        float(c * delta_xy[0] + s * delta_xy[1]),
                        float(-s * delta_xy[0] + c * delta_xy[1]),
                    ]
                )
                yaw_values.append(float(yaw[final] - yaw[entry]))
                root_height_values.append(float(roots[final, 2] - roots[entry, 2]))
                surface_values.append((surface[final] - surface[entry]).tolist())
                integer_values["entry_row"].append(row)
                integer_values["skill_index"].append(int(skill.skill_index))
                integer_values["clip_index"].append(clip_index)
                integer_values["entry_frame"].append(entry)
                integer_values["target_frames"].append(target)
                integer_values["endpoint_frame_exclusive"].append(endpoint)
                integer_values["maximum_stall_frames"].append(
                    _maximum_stall_frames(roots[entry:endpoint, :2])
                )
                integer_values["duration_frames"].append(endpoint - entry)

    rejected = {key: value for key, value in rejected.items() if value}
    device = database.device

    def integer(name: str) -> torch.Tensor:
        return torch.tensor(integer_values[name], dtype=torch.long, device=device)

    def floating(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=device)

    return TerrainSkillHorizonInventory(
        entry_row=integer("entry_row"),
        skill_index=integer("skill_index"),
        clip_index=integer("clip_index"),
        entry_frame=integer("entry_frame"),
        target_frames=integer("target_frames"),
        endpoint_frame_exclusive=integer("endpoint_frame_exclusive"),
        root_displacement_local_xy=torch.tensor(
            displacement_values, dtype=torch.float32, device=device
        ).reshape(-1, 2),
        yaw_delta_rad=floating(yaw_values),
        root_height_delta_m=floating(root_height_values),
        surface_height_delta_m=torch.tensor(
            surface_values, dtype=torch.float32, device=device
        ).reshape(-1, 2),
        maximum_stall_frames=integer("maximum_stall_frames"),
        duration_frames=integer("duration_frames"),
        rejected_by_reason=MappingProxyType(rejected),
    )
