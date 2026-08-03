"""Deterministic PHP-style entry windows for coherent terrain motion skills."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch

from .joints import ContractError
from .torch_contact_segments import source_support_mask
from .torch_motion_features import TorchMotionDatabase
from .torch_terrain_features import TerrainDataset


@dataclass(frozen=True, order=True)
class SkillInterval:
    entry_start: int
    playback_start: int
    playback_stop: int


@dataclass(frozen=True)
class TerrainSkill:
    skill_index: int
    clip_index: int
    interval: SkillInterval
    entry_rows: tuple[int, ...]
    support_mask: torch.Tensor
    foot_surface_height_m: torch.Tensor


@dataclass(frozen=True)
class TerrainSkillInventory:
    skills: tuple[TerrainSkill, ...]
    rejected_by_reason: Mapping[str, int]
    row_to_skill: Mapping[int, int]


def _source_contact_height_p95_descriptor(descriptor: object) -> float | None:
    try:
        if descriptor.get("kind") == "flat":
            return None
        value = descriptor["admission"]["contact_height_error_m"]["p95"]
    except (AttributeError, KeyError, TypeError) as error:
        raise ContractError("terrain source contact quality is unavailable") from error
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ContractError("terrain source contact quality is invalid")
    return float(value)


def source_contact_height_p95_m(
    dataset: TerrainDataset, clip_index: int
) -> float | None:
    """Return the authenticated source contact-fit p95 for one terrain clip."""

    try:
        descriptor = dataset.manifest["clips"][clip_index]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise ContractError("terrain source contact quality is unavailable") from error
    return _source_contact_height_p95_descriptor(descriptor)


def _validate_profiles(
    support_mask: torch.Tensor, foot_surface_height_m: torch.Tensor
) -> int:
    if not isinstance(support_mask, torch.Tensor) or support_mask.dtype != torch.bool:
        raise ContractError("skill support mask must be a boolean torch.Tensor")
    if (
        not isinstance(foot_surface_height_m, torch.Tensor)
        or not foot_surface_height_m.dtype.is_floating_point
    ):
        raise ContractError("skill surface heights must be floating-point torch.Tensor")
    if (
        support_mask.ndim != 2
        or support_mask.shape[1] != 2
        or tuple(support_mask.shape) != tuple(foot_surface_height_m.shape)
    ):
        raise ContractError("skill support and height profiles must have matching shape (T, 2)")
    if support_mask.device != foot_surface_height_m.device:
        raise ContractError("skill support and height profiles must share a device")
    if not torch.isfinite(foot_surface_height_m).all():
        raise ContractError("skill surface heights must be finite")
    return int(support_mask.shape[0])


def extract_skill_intervals(
    support_mask: torch.Tensor,
    foot_surface_height_m: torch.Tensor,
    *,
    valid_frame_stop: int,
    entry_window_frames: int = 50,
    minimum_surface_change_m: float = 0.08,
    stable_gap_frames: int = 50,
) -> tuple[SkillInterval, ...]:
    """Extract changed-height episodes bounded by stable double support.

    ``playback_stop`` is exclusive. Entry frames precede the maneuver and must
    remain inside the database's searchable feature horizon.
    """

    frame_count = _validate_profiles(support_mask, foot_surface_height_m)
    if type(valid_frame_stop) is not int or not 1 <= valid_frame_stop <= frame_count:
        raise ContractError("skill valid frame stop is invalid")
    if type(entry_window_frames) is not int or entry_window_frames < 1:
        raise ContractError("skill entry window must be positive")
    if type(stable_gap_frames) is not int or stable_gap_frames < 1:
        raise ContractError("skill stable gap must be positive")
    if not isinstance(minimum_surface_change_m, (int, float)) or not (
        0.0 < float(minimum_surface_change_m) < float("inf")
    ):
        raise ContractError("skill surface-change threshold must be positive")

    stable = support_mask.all(dim=1)
    stable_indices = torch.nonzero(stable, as_tuple=False).flatten()
    if stable_indices.numel() == 0:
        return ()
    baseline_frame = int(stable_indices[0].item())
    baseline = foot_surface_height_m[baseline_frame]
    changed_supported = support_mask & (
        torch.abs(foot_surface_height_m - baseline.unsqueeze(0))
        >= float(minimum_surface_change_m)
    )
    active_indices = torch.nonzero(changed_supported.any(dim=1), as_tuple=False).flatten().cpu().tolist()
    if not active_indices:
        return ()

    groups: list[tuple[int, int]] = []
    group_start = group_end = int(active_indices[0])
    for raw_index in active_indices[1:]:
        index = int(raw_index)
        if index - group_end - 1 < stable_gap_frames:
            group_end = index
        else:
            groups.append((group_start, group_end))
            group_start = group_end = index
    groups.append((group_start, group_end))

    stable_list = [int(value) for value in stable_indices.cpu().tolist()]
    intervals: list[SkillInterval] = []
    for first_active, last_active in groups:
        starts = [index for index in stable_list if index < first_active]
        ends = [index for index in stable_list if index > last_active]
        if not starts:
            continue
        playback_start = starts[-1]
        # A skill ending on an elevated platform is valid; use its final stable
        # state when no return-to-baseline state follows the episode.
        if ends:
            final_frame = ends[0]
        else:
            tail_stable = [index for index in stable_list if index >= last_active]
            if not tail_stable:
                continue
            final_frame = tail_stable[-1]
        if playback_start >= valid_frame_stop or final_frame <= playback_start:
            continue
        entry_start = max(0, playback_start - entry_window_frames)
        if entry_start == playback_start:
            continue
        intervals.append(
            SkillInterval(entry_start, playback_start, final_frame + 1)
        )
    return tuple(intervals)


def _surface_height_profile(dataset: TerrainDataset, clip_index: int) -> torch.Tensor:
    clip = dataset.folder.clips[clip_index]
    layout = dataset.folder.layout
    feet = torch.as_tensor(
        clip.body_position_world[
            :, (layout.left_foot_body_index, layout.right_foot_body_index), :2
        ],
        dtype=torch.float32,
        device=dataset.device,
    )
    grid = dataset.clip_grids[clip_index]
    if grid is None:
        return torch.zeros(feet.shape[:2], dtype=torch.float32, device=dataset.device)
    alignment = dataset.clip_alignments[clip_index]
    if alignment is None:
        raise ContractError("terrain skill clip has no scene alignment")
    return grid.sample_xy(alignment.matcher_to_scene_xy(feet))


def build_terrain_skill_inventory(
    dataset: TerrainDataset,
    database: TorchMotionDatabase,
    *,
    entry_window_frames: int = 50,
    minimum_surface_change_m: float = 0.08,
    stable_gap_frames: int = 50,
    minimum_remaining_frames: int = 50,
    maximum_source_contact_p95_m: float | None = None,
) -> TerrainSkillInventory:
    """Build an owned, deterministic terrain skill inventory."""

    if not isinstance(dataset, TerrainDataset):
        raise ContractError("terrain skill inventory requires a TerrainDataset")
    if not isinstance(database, TorchMotionDatabase):
        raise ContractError("terrain skill inventory requires a TorchMotionDatabase")
    if type(minimum_remaining_frames) is not int or minimum_remaining_frames < 1:
        raise ContractError("terrain skill minimum remaining frames must be positive")
    if maximum_source_contact_p95_m is not None and (
        not isinstance(maximum_source_contact_p95_m, (int, float))
        or isinstance(maximum_source_contact_p95_m, bool)
        or not math.isfinite(float(maximum_source_contact_p95_m))
        or float(maximum_source_contact_p95_m) <= 0.0
    ):
        raise ContractError("maximum source contact p95 must be positive")
    if database.folder is not dataset.folder and (
        database.folder.inventory_sha256 != dataset.folder.inventory_sha256
    ):
        raise ContractError("terrain skill database does not match dataset")

    source_quality: tuple[float | None, ...] | None = None
    if maximum_source_contact_p95_m is not None:
        try:
            descriptors = dataset.manifest["clips"]
        except (AttributeError, KeyError, TypeError) as error:
            raise ContractError(
                "terrain source contact quality is unavailable"
            ) from error
        if not isinstance(descriptors, list) or len(descriptors) != len(
            dataset.folder.clips
        ):
            raise ContractError("terrain source contact quality is misaligned")
        source_quality = tuple(
            _source_contact_height_p95_descriptor(descriptor)
            for descriptor in descriptors
        )

    skills: list[TerrainSkill] = []
    row_to_skill: dict[int, int] = {}
    rejected = {
        "flat": 0,
        "no_episode": 0,
        "no_entry_rows": 0,
        "overlapping_entry_rows": 0,
        "source_quality": 0,
    }
    for clip_index, clip in enumerate(dataset.folder.clips):
        if dataset.clip_grids[clip_index] is None:
            support = source_support_mask(dataset, clip_index).detach().clone()
            heights = (
                _surface_height_profile(dataset, clip_index).detach().clone()
            )
            interval = SkillInterval(0, 0, clip.valid_frame_stop)
            candidate_rows = tuple(
                row
                for frame in range(
                    0,
                    clip.valid_frame_stop - minimum_remaining_frames + 1,
                )
                if bool(support[frame].all().item())
                and (
                    row := database.row_for_source(clip_index, frame)
                )
                is not None
            )
            rows = tuple(
                row for row in candidate_rows if row not in row_to_skill
            )
            rejected["overlapping_entry_rows"] += len(candidate_rows) - len(rows)
            if not rows:
                rejected["no_entry_rows"] += 1
                continue
            skill_index = len(skills)
            skills.append(
                TerrainSkill(
                    skill_index=skill_index,
                    clip_index=clip_index,
                    interval=interval,
                    entry_rows=rows,
                    support_mask=support,
                    foot_surface_height_m=heights,
                )
            )
            for row in rows:
                row_to_skill[row] = skill_index
            continue
        clip_quality = None if source_quality is None else source_quality[clip_index]
        if (
            clip_quality is not None
            and clip_quality > float(maximum_source_contact_p95_m)
        ):
            rejected["source_quality"] += 1
            continue
        support = source_support_mask(dataset, clip_index).detach().clone()
        heights = _surface_height_profile(dataset, clip_index).detach().clone()
        intervals = extract_skill_intervals(
            support,
            heights,
            valid_frame_stop=clip.valid_frame_stop,
            entry_window_frames=entry_window_frames,
            minimum_surface_change_m=minimum_surface_change_m,
            stable_gap_frames=stable_gap_frames,
        )
        if not intervals:
            rejected["no_episode"] += 1
            continue
        stable_frames = frozenset(
            int(frame)
            for frame in torch.nonzero(
                support.all(dim=1), as_tuple=False
            ).flatten().cpu().tolist()
        )
        for interval in intervals:
            candidate_rows = tuple(
                row
                for frame in range(
                    interval.entry_start,
                    min(
                        interval.playback_stop - minimum_remaining_frames + 1,
                        clip.valid_frame_stop,
                    ),
                )
                if (
                    frame < interval.playback_start
                    or frame in stable_frames
                )
                and (row := database.row_for_source(clip_index, frame)) is not None
            )
            rows = tuple(row for row in candidate_rows if row not in row_to_skill)
            rejected["overlapping_entry_rows"] += len(candidate_rows) - len(rows)
            if not rows:
                rejected["no_entry_rows"] += 1
                continue
            skill_index = len(skills)
            skill = TerrainSkill(
                skill_index=skill_index,
                clip_index=clip_index,
                interval=interval,
                entry_rows=rows,
                support_mask=support,
                foot_surface_height_m=heights,
            )
            skills.append(skill)
            for row in rows:
                row_to_skill[row] = skill_index

    return TerrainSkillInventory(
        skills=tuple(skills),
        rejected_by_reason=MappingProxyType(dict(rejected)),
        row_to_skill=MappingProxyType(dict(row_to_skill)),
    )
