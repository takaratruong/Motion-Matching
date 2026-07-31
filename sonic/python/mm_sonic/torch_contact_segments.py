"""Authenticated support-to-support segments for terrain motion matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
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
            source_support_mask(dataset, index)[
                : dataset.folder.clips[index].valid_frame_stop
            ]
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


@dataclass(frozen=True)
class SegmentPlacement:
    segment: ContactSegment
    vertical_offset_m: float
    source_support_mask: torch.Tensor
    required_entry_support_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        offset = float(self.vertical_offset_m)
        support = self.source_support_mask
        if not isinstance(self.segment, ContactSegment):
            raise ContractError("segment placement requires a contact segment")
        if not torch.isfinite(torch.tensor(offset)):
            raise ContractError("segment placement vertical offset must be finite")
        if (
            not isinstance(support, torch.Tensor)
            or support.dtype != torch.bool
            or tuple(support.shape) != (self.segment.frame_count, 2)
        ):
            raise ContractError(
                "segment placement source support must match the segment"
            )
        required = self.required_entry_support_mask
        if required is not None and (
            not isinstance(required, torch.Tensor)
            or required.dtype != torch.bool
            or tuple(required.shape) != (2,)
            or required.device != support.device
        ):
            raise ContractError(
                "segment placement required entry support must be boolean "
                "shape-(2,) on the source support device"
            )
        owned = support.detach().clone()
        object.__setattr__(self, "vertical_offset_m", offset)
        object.__setattr__(self, "source_support_mask", owned)
        if required is not None:
            object.__setattr__(
                self,
                "required_entry_support_mask",
                required.detach().clone(),
            )


def _owned_readonly(array: object, dtype=None) -> np.ndarray:
    output = np.array(array, dtype=dtype, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class SegmentValidation:
    accepted: bool
    reason: str | None
    foot_position_world: np.ndarray
    foot_clearance_m: np.ndarray
    emitted_support_mask: np.ndarray
    lost_source_support_fraction: float
    longest_unsupported_frames: int

    def __post_init__(self) -> None:
        feet = _owned_readonly(self.foot_position_world, np.float64)
        clearance = _owned_readonly(self.foot_clearance_m, np.float64)
        support = _owned_readonly(self.emitted_support_mask, np.bool_)
        frames = feet.shape[0] if feet.ndim == 3 else -1
        if (
            type(self.accepted) is not bool
            or (self.reason is not None and not isinstance(self.reason, str))
            or feet.shape != (frames, 2, 3)
            or clearance.shape != (frames, 2)
            or support.shape != (frames, 2)
            or not np.isfinite(feet).all()
            or not math.isfinite(float(self.lost_source_support_fraction))
            or not 0.0 <= float(self.lost_source_support_fraction) <= 1.0
            or type(self.longest_unsupported_frames) is not int
            or self.longest_unsupported_frames < 0
        ):
            raise ContractError("segment validation fields are invalid")
        object.__setattr__(self, "foot_position_world", feet)
        object.__setattr__(self, "foot_clearance_m", clearance)
        object.__setattr__(self, "emitted_support_mask", support)
        object.__setattr__(
            self,
            "lost_source_support_fraction",
            float(self.lost_source_support_fraction),
        )


@dataclass(frozen=True)
class TerrainContactSegmentPolicy:
    """Resolve contact-compatible terrain entries and rigid vertical placement."""

    index: ContactSegmentIndex
    extension: object
    foot_kinematics: object | None = None
    maximum_scene_xy_mismatch_m: float | None = None
    entry_inertialization_halflife_s: float | None = None
    flat_support_transition_cost_weight: float = 0.0

    def __post_init__(self) -> None:
        try:
            dataset = self.extension.dataset
            query_grid = self.extension.query_grid
            query_alignment = self.extension.alignment
            clip_count = len(dataset.folder.clips)
        except AttributeError as error:
            raise ContractError("contact segment terrain extension is invalid") from error
        if not isinstance(self.index, ContactSegmentIndex):
            raise ContractError("contact segment policy index is invalid")
        if clip_count != len(self.index._support_masks):
            raise ContractError(
                "contact segment policy motion inventory does not match"
            )
        if not callable(getattr(query_grid, "sample_xy", None)) or not callable(
            getattr(query_alignment, "matcher_to_scene_xy", None)
        ):
            raise ContractError("contact segment query terrain is invalid")
        mismatch = self.maximum_scene_xy_mismatch_m
        if mismatch is not None and (
            isinstance(mismatch, bool)
            or not isinstance(mismatch, (int, float))
            or not math.isfinite(float(mismatch))
            or float(mismatch) <= 0.0
        ):
            raise ContractError(
                "contact segment scene mismatch limit must be finite and positive"
            )
        if mismatch is not None:
            object.__setattr__(
                self, "maximum_scene_xy_mismatch_m", float(mismatch)
            )
        halflife = self.entry_inertialization_halflife_s
        if halflife is not None and (
            isinstance(halflife, bool)
            or not isinstance(halflife, (int, float))
            or not math.isfinite(float(halflife))
            or float(halflife) <= 0.0
        ):
            raise ContractError(
                "contact entry inertialization halflife must be finite and positive"
            )
        if halflife is not None:
            object.__setattr__(
                self,
                "entry_inertialization_halflife_s",
                float(halflife),
            )
        weight = self.flat_support_transition_cost_weight
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ContractError(
                "flat support transition cost weight must be finite and non-negative"
            )
        object.__setattr__(
            self, "flat_support_transition_cost_weight", float(weight)
        )
    @property
    def dataset(self):
        return self.extension.dataset

    def entry_eligibility(self, database: object) -> torch.Tensor:
        return self.index.terrain_entry_eligibility(database)

    def registered_yaw_offset(self, clip_index: int) -> torch.Tensor:
        if (
            type(clip_index) is not int
            or clip_index not in self.index.terrain_clip_indices
        ):
            raise ContractError("registered terrain yaw clip is invalid")
        source_alignment = self.dataset.clip_alignments[clip_index]
        if source_alignment is None:
            raise ContractError("registered terrain yaw has no source alignment")
        try:
            source_yaw = source_alignment.yaw_scene_from_matcher
            query_yaw = self.extension.alignment.yaw_scene_from_matcher
        except AttributeError as error:
            raise ContractError("registered terrain yaw alignment is invalid") from error
        if (
            not isinstance(source_yaw, torch.Tensor)
            or source_yaw.numel() != 1
            or not isinstance(query_yaw, torch.Tensor)
            or query_yaw.numel() != 1
            or source_yaw.device != query_yaw.device
            or source_yaw.dtype != query_yaw.dtype
            or not torch.isfinite(source_yaw).all()
            or not torch.isfinite(query_yaw).all()
        ):
            raise ContractError("registered terrain yaw tensors are invalid")
        delta = source_yaw.reshape(()) - query_yaw.reshape(())
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    @staticmethod
    def _longest_false_run(supported: torch.Tensor) -> int:
        longest = current = 0
        for value in supported.detach().cpu().tolist():
            if bool(value):
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return longest

    def emitted_foot_positions(
        self,
        joint_position: torch.Tensor,
        root_position: torch.Tensor,
        root_orientation_wxyz: torch.Tensor,
    ) -> np.ndarray:
        adapter = self.foot_kinematics
        if adapter is None or not callable(
            getattr(adapter, "foot_positions", None)
        ):
            raise ContractError(
                "contact segment validation requires foot kinematics"
            )
        try:
            feet = np.asarray(
                adapter.foot_positions(
                    joint_position, root_position, root_orientation_wxyz
                ),
                np.float64,
            )
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("contact segment foot kinematics failed") from error
        expected = (joint_position.shape[0], 2, 3)
        if feet.shape != expected or not np.isfinite(feet).all():
            raise ContractError(
                "contact segment foot kinematics returned invalid positions"
            )
        return np.ascontiguousarray(feet)

    def validate_emitted(
        self,
        placement: SegmentPlacement,
        *,
        joint_position: torch.Tensor,
        root_position: torch.Tensor,
        root_orientation_wxyz: torch.Tensor,
    ) -> SegmentValidation:
        if not isinstance(placement, SegmentPlacement):
            raise ContractError("emitted validation requires segment placement")
        frames = placement.segment.frame_count
        if (
            not isinstance(joint_position, torch.Tensor)
            or tuple(joint_position.shape) != (frames, 29)
            or not isinstance(root_position, torch.Tensor)
            or tuple(root_position.shape) != (frames, 3)
            or not isinstance(root_orientation_wxyz, torch.Tensor)
            or tuple(root_orientation_wxyz.shape) != (frames, 4)
            or joint_position.device != self.dataset.device
            or root_position.device != joint_position.device
            or root_orientation_wxyz.device != joint_position.device
            or not torch.isfinite(joint_position).all()
            or not torch.isfinite(root_position).all()
            or not torch.isfinite(root_orientation_wxyz).all()
        ):
            raise ContractError(
                "emitted segment state shapes must exactly match its frame count"
            )
        feet_np = self.emitted_foot_positions(
            joint_position, root_position, root_orientation_wxyz
        )
        feet = torch.tensor(
            feet_np, dtype=torch.float32, device=self.dataset.device
        )
        try:
            scene_xy = self.extension.alignment.matcher_to_scene_xy(
                feet[..., :2]
            )
            surface = self.extension.query_grid.sample_xy(scene_xy)
        except ContractError:
            return SegmentValidation(
                accepted=False,
                reason="out_of_domain",
                foot_position_world=feet_np,
                foot_clearance_m=np.full((frames, 2), np.nan),
                emitted_support_mask=np.zeros((frames, 2), np.bool_),
                lost_source_support_fraction=1.0,
                longest_unsupported_frames=frames,
            )
        clearance = feet[..., 2] - surface
        vertical_speed = torch.zeros_like(clearance)
        if frames > 1:
            vertical_speed[1:] = (feet[1:, :, 2] - feet[:-1, :, 2]) / 0.02
        emitted_support = (
            (
                torch.abs(clearance - ANKLE_ORIGIN_SOLE_M)
                <= STANCE_CLEARANCE_TOLERANCE_M
            )
            & (torch.abs(vertical_speed) <= STANCE_VERTICAL_SPEED_MAX_MPS)
        )
        supported = emitted_support.any(dim=1)
        longest = self._longest_false_run(supported)
        source_supported = placement.source_support_mask.any(dim=1)
        source_count = int(source_supported.sum().item())
        lost = int((source_supported & ~supported).sum().item())
        lost_fraction = 0.0 if source_count == 0 else lost / source_count

        if float(clearance.min().item()) < -0.03:
            reason = "penetration"
        elif not bool(
            emitted_support[0, placement.segment.entering_foot].item()
        ):
            reason = "entering_support"
        elif (
            placement.required_entry_support_mask is not None
            and bool(placement.required_entry_support_mask.any().item())
            and not bool(
                (
                    emitted_support[0]
                    & placement.required_entry_support_mask
                ).any().item()
            )
        ):
            reason = "support_side_switch"
        elif longest > 10:
            reason = "unsupported_run"
        elif lost_fraction > 0.05:
            reason = "source_support_lost"
        else:
            reason = None
        return SegmentValidation(
            accepted=reason is None,
            reason=reason,
            foot_position_world=feet_np,
            foot_clearance_m=clearance.detach().cpu().numpy(),
            emitted_support_mask=emitted_support.detach().cpu().numpy(),
            lost_source_support_fraction=lost_fraction,
            longest_unsupported_frames=longest,
        )

    @staticmethod
    def _validated_pose(
        yaw_offset: torch.Tensor,
        translation_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not isinstance(yaw_offset, torch.Tensor)
            or yaw_offset.numel() != 1
            or not yaw_offset.dtype.is_floating_point
            or not torch.isfinite(yaw_offset).all()
        ):
            raise ContractError("contact entry yaw must be a finite scalar tensor")
        if (
            not isinstance(translation_xy, torch.Tensor)
            or tuple(translation_xy.shape) != (2,)
            or translation_xy.device != yaw_offset.device
            or translation_xy.dtype != yaw_offset.dtype
            or not torch.isfinite(translation_xy).all()
        ):
            raise ContractError(
                "contact entry translation must be a matching finite shape-(2,) tensor"
            )
        return yaw_offset.reshape(()), translation_xy

    def query_support_mask(
        self,
        feature_body_position: torch.Tensor,
        feature_body_velocity: torch.Tensor,
    ) -> torch.Tensor:
        position = feature_body_position
        velocity = feature_body_velocity
        if (
            not isinstance(position, torch.Tensor)
            or tuple(position.shape) != (3, 3)
            or position.dtype != torch.float32
            or position.device != self.dataset.device
            or not torch.isfinite(position).all()
            or not isinstance(velocity, torch.Tensor)
            or tuple(velocity.shape) != (3, 3)
            or velocity.dtype != position.dtype
            or velocity.device != position.device
            or not torch.isfinite(velocity).all()
        ):
            raise ContractError(
                "contact query bodies must be finite float32 shape-(3,3) tensors"
            )
        feet = position[1:]
        scene_xy = self.extension.alignment.matcher_to_scene_xy(feet[:, :2])
        surface = self.extension.query_grid.sample_xy(scene_xy)
        clearance = feet[:, 2] - surface
        return (
            (
                torch.abs(clearance - ANKLE_ORIGIN_SOLE_M)
                <= STANCE_CLEARANCE_TOLERANCE_M
            )
            & (
                torch.abs(velocity[1:, 2])
                <= STANCE_VERTICAL_SPEED_MAX_MPS
            )
        ).detach()

    def resolve_entry(
        self,
        *,
        clip_index: int,
        frame_index: int,
        yaw_offset: torch.Tensor,
        translation_xy: torch.Tensor,
        current_support_mask: torch.Tensor,
    ) -> SegmentPlacement | None:
        segment = self.index.entry(clip_index, frame_index)
        if segment is None:
            return None
        yaw, translation = self._validated_pose(yaw_offset, translation_xy)
        support = current_support_mask
        if (
            not isinstance(support, torch.Tensor)
            or support.dtype != torch.bool
            or tuple(support.shape) != (2,)
            or support.device != yaw.device
        ):
            raise ContractError(
                "current support mask must be boolean shape-(2,) on the policy device"
            )
        dataset = self.dataset
        layout = dataset.folder.layout
        foot_body = (
            layout.left_foot_body_index
            if segment.entering_foot == 0
            else layout.right_foot_body_index
        )
        source_ankle = torch.tensor(
            dataset.folder.clips[clip_index].body_position_world[
                frame_index, foot_body
            ],
            dtype=torch.float32,
            device=dataset.device,
        )
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        transformed_xy = torch.stack(
            (
                cosine * source_ankle[0] - sine * source_ankle[1],
                sine * source_ankle[0] + cosine * source_ankle[1],
            )
        ) + translation
        source_grid = dataset.clip_grids[clip_index]
        source_alignment = dataset.clip_alignments[clip_index]
        if source_grid is None or source_alignment is None:
            raise ContractError("terrain contact entry has no source terrain")
        source_scene_xy = source_alignment.matcher_to_scene_xy(
            source_ankle[:2]
        )
        mismatch_limit = self.maximum_scene_xy_mismatch_m
        if mismatch_limit is not None:
            try:
                query_translation = self.extension.alignment.translation_scene_xy
                query_yaw = self.extension.alignment.yaw_scene_from_matcher
            except AttributeError as error:
                raise ContractError(
                    "contact scene mismatch gate requires rigid query alignment"
                ) from error
            relative = source_scene_xy - query_translation
            query_cosine = torch.cos(query_yaw)
            query_sine = torch.sin(query_yaw)
            registered_xy = torch.stack(
                (
                    query_cosine * relative[0]
                    + query_sine * relative[1],
                    -query_sine * relative[0]
                    + query_cosine * relative[1],
                )
            )
            mismatch = transformed_xy - registered_xy
            if float(torch.linalg.vector_norm(mismatch).item()) > mismatch_limit:
                return None
            source_start_ankle = torch.tensor(
                dataset.folder.clips[clip_index].body_position_world[
                    0, foot_body, :2
                ],
                dtype=torch.float32,
                device=dataset.device,
            )
            start_scene_xy = source_alignment.matcher_to_scene_xy(
                source_start_ankle
            )
            start_relative = start_scene_xy - query_translation
            registered_start_xy = torch.stack(
                (
                    query_cosine * start_relative[0]
                    + query_sine * start_relative[1],
                    -query_sine * start_relative[0]
                    + query_cosine * start_relative[1],
                )
            )
            phase_direction = registered_xy - registered_start_xy
            phase_length = torch.linalg.vector_norm(phase_direction)
            if (
                float(phase_length.item()) > 1e-6
                and float(
                    torch.dot(mismatch, phase_direction / phase_length).item()
                )
                < 0.0
            ):
                return None
        query_scene_xy = self.extension.alignment.matcher_to_scene_xy(
            transformed_xy
        )
        target_surface = self.extension.query_grid.sample_xy(query_scene_xy)
        source_surface = source_grid.sample_xy(source_scene_xy)
        return SegmentPlacement(
            segment=segment,
            vertical_offset_m=float((target_surface - source_surface).item()),
            source_support_mask=self.index.support_mask(clip_index)[
                segment.start_frame : segment.end_frame
            ],
            required_entry_support_mask=support,
        )
