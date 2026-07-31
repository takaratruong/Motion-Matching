"""Authenticated support-to-support segments for terrain motion matching."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .joints import ContractError
from .torch_terrain_features import TerrainDataset


ANKLE_ORIGIN_SOLE_M = 0.035
STANCE_CLEARANCE_TOLERANCE_M = 0.020
STANCE_VERTICAL_SPEED_MAX_MPS = 0.12


@dataclass(frozen=True, order=True)
class ContactSegment:
    clip_index: int
    start_frame: int
    end_frame: int
    entering_foot: int

    def __post_init__(self) -> None:
        if (
            type(self.clip_index) is not int
            or self.clip_index < 0
            or type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.end_frame) is not int
            or self.end_frame <= self.start_frame
            or self.entering_foot not in (0, 1)
        ):
            raise ContractError("contact segment fields are invalid")

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


def _validated_bounds(minimum_frames: int, maximum_frames: int) -> None:
    if type(minimum_frames) is not int or minimum_frames < 1:
        raise ContractError("contact segment minimum must be a positive integer")
    if type(maximum_frames) is not int or maximum_frames < minimum_frames:
        raise ContractError(
            "contact segment maximum must be an integer not below minimum"
        )


def segments_from_support_mask(
    clip_index: int,
    support_mask: torch.Tensor,
    minimum_frames: int,
    maximum_frames: int,
) -> tuple[ContactSegment, ...]:
    """Return bounded onset-to-next-opposite-onset half-open segments."""

    if type(clip_index) is not int or clip_index < 0:
        raise ContractError("contact segment clip index is invalid")
    if not isinstance(support_mask, torch.Tensor):
        raise ContractError("contact support mask must be a tensor")
    if support_mask.dtype != torch.bool:
        raise ContractError("contact support mask must be boolean")
    if support_mask.ndim != 2 or support_mask.shape[1] != 2:
        raise ContractError("contact support mask must have shape (frames, 2)")
    _validated_bounds(minimum_frames, maximum_frames)
    if support_mask.shape[0] < 2:
        return ()

    onset = (~support_mask[:-1]) & support_mask[1:]
    raw_events = torch.nonzero(onset, as_tuple=False).detach().cpu().tolist()
    events = sorted((int(frame) + 1, int(foot)) for frame, foot in raw_events)
    output: list[ContactSegment] = []
    for event_index, (start, foot) in enumerate(events):
        end = next(
            (
                later_frame
                for later_frame, later_foot in events[event_index + 1 :]
                if later_foot != foot
            ),
            None,
        )
        if end is None:
            continue
        frame_count = end - start
        if minimum_frames <= frame_count <= maximum_frames:
            output.append(ContactSegment(clip_index, start, end, foot))
    return tuple(output)


def source_support_mask(
    dataset: TerrainDataset, clip_index: int
) -> torch.Tensor:
    """Derive source ankle support from its authenticated paired terrain."""

    if not isinstance(dataset, TerrainDataset):
        raise ContractError("contact segments require a terrain dataset")
    if type(clip_index) is not int or not 0 <= clip_index < len(dataset.folder.clips):
        raise ContractError("contact support clip index is invalid")
    clip = dataset.folder.clips[clip_index]
    layout = dataset.folder.layout
    foot_indices = (layout.left_foot_body_index, layout.right_foot_body_index)
    feet = torch.tensor(
        clip.body_position_world[:, foot_indices],
        dtype=torch.float32,
        device=dataset.device,
    )
    vertical_speed = torch.tensor(
        clip.body_linear_velocity_world[:, foot_indices, 2],
        dtype=torch.float32,
        device=dataset.device,
    )
    grid = dataset.clip_grids[clip_index]
    if grid is None:
        surface = torch.zeros_like(feet[..., 2])
    else:
        alignment = dataset.clip_alignments[clip_index]
        if alignment is None:
            raise ContractError("terrain contact clip has no scene alignment")
        surface = grid.sample_xy(
            alignment.matcher_to_scene_xy(feet[..., :2])
        )
    clearance = feet[..., 2] - surface
    return (
        (
            torch.abs(clearance - ANKLE_ORIGIN_SOLE_M)
            <= STANCE_CLEARANCE_TOLERANCE_M
        )
        & (torch.abs(vertical_speed) <= STANCE_VERTICAL_SPEED_MAX_MPS)
    ).detach()


@dataclass(frozen=True)
class ContactSegmentIndex:
    segments: tuple[ContactSegment, ...]
    terrain_clip_indices: frozenset[int]
    _support_masks: tuple[torch.Tensor, ...]
    _entries: Mapping[tuple[int, int], ContactSegment]

    @classmethod
    def from_support_masks(
        cls,
        support_masks: Sequence[torch.Tensor],
        *,
        terrain_clip_indices: Sequence[int],
        minimum_frames: int = 5,
        maximum_frames: int = 60,
    ) -> "ContactSegmentIndex":
        _validated_bounds(minimum_frames, maximum_frames)
        masks = tuple(support_masks)
        if not masks:
            raise ContractError("contact segment support inventory is empty")
        owned: list[torch.Tensor] = []
        for mask in masks:
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.ndim != 2
                or mask.shape[1] != 2
            ):
                raise ContractError(
                    "contact support inventory masks must be boolean (frames, 2)"
                )
            owned.append(mask.detach().clone())
        terrain = tuple(terrain_clip_indices)
        if (
            any(type(index) is not int or not 0 <= index < len(owned) for index in terrain)
            or len(terrain) != len(set(terrain))
        ):
            raise ContractError("terrain contact clip inventory is invalid")
        segments = tuple(
            segment
            for clip_index in terrain
            for segment in segments_from_support_mask(
                clip_index,
                owned[clip_index],
                minimum_frames,
                maximum_frames,
            )
        )
        entries = {
            (segment.clip_index, segment.start_frame): segment
            for segment in segments
        }
        if len(entries) != len(segments):
            raise ContractError("contact segment entries are not unique")
        return cls(
            segments=segments,
            terrain_clip_indices=frozenset(terrain),
            _support_masks=tuple(owned),
            _entries=MappingProxyType(entries),
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: TerrainDataset,
        *,
        minimum_frames: int = 5,
        maximum_frames: int = 60,
    ) -> "ContactSegmentIndex":
        if not isinstance(dataset, TerrainDataset):
            raise ContractError("contact segments require a terrain dataset")
        terrain = tuple(
            index
            for index, grid in enumerate(dataset.clip_grids)
            if grid is not None
        )
        masks = tuple(
            source_support_mask(dataset, index)
            for index in range(len(dataset.folder.clips))
        )
        return cls.from_support_masks(
            masks,
            terrain_clip_indices=terrain,
            minimum_frames=minimum_frames,
            maximum_frames=maximum_frames,
        )

    def support_mask(self, clip_index: int) -> torch.Tensor:
        if type(clip_index) is not int or not 0 <= clip_index < len(self._support_masks):
            raise ContractError("contact support clip index is invalid")
        return self._support_masks[clip_index].clone()

    def entry(self, clip_index: int, frame_index: int) -> ContactSegment | None:
        if type(clip_index) is not int or type(frame_index) is not int:
            raise ContractError("contact segment entry indices must be integers")
        return self._entries.get((clip_index, frame_index))

    def terrain_entry_eligibility(self, database: object) -> torch.Tensor:
        try:
            clip = database._search_clip_index
            frame = database._search_frame_index
            device = database.device
        except AttributeError as error:
            raise ContractError("contact segment database mapping is invalid") from error
        if (
            not isinstance(clip, torch.Tensor)
            or not isinstance(frame, torch.Tensor)
            or clip.ndim != 1
            or frame.ndim != 1
            or clip.shape != frame.shape
            or clip.device != frame.device
            or clip.device != torch.device(device)
        ):
            raise ContractError("contact segment database rows are invalid")
        eligible = torch.ones(clip.shape, dtype=torch.bool, device=clip.device)
        for row, (clip_index, frame_index) in enumerate(
            zip(clip.detach().cpu().tolist(), frame.detach().cpu().tolist())
        ):
            if int(clip_index) in self.terrain_clip_indices:
                eligible[row] = (
                    int(clip_index), int(frame_index)
                ) in self._entries
        return eligible
