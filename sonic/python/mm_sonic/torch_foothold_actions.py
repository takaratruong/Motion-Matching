"""Authenticated short-horizon foothold actions for terrain matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch

from .joints import ContractError


def _owned_float_tensor(
    value: torch.Tensor, shape: tuple[int, ...], label: str
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not value.dtype.is_floating_point
        or not torch.isfinite(value).all()
    ):
        raise ContractError(f"foothold action {label} is invalid")
    return value.detach().clone()


@dataclass(frozen=True)
class FootholdAction:
    """Two alternating source landings expressed in the start-root frame."""

    clip_index: int
    start_frame: int
    end_frame: int
    start_support: tuple[bool, bool]
    landing_feet: tuple[int, int]
    landing_frame_offsets: tuple[int, int]
    landing_xy_start_frame_m: torch.Tensor
    landing_height_delta_m: torch.Tensor
    root_displacement_m: torch.Tensor
    root_yaw_delta_rad: torch.Tensor
    minimum_swing_clearance_m: float
    maximum_unsupported_frames: int

    def __post_init__(self) -> None:
        if (
            type(self.clip_index) is not int
            or self.clip_index < 0
            or type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.end_frame) is not int
            or self.end_frame <= self.start_frame
            or not isinstance(self.start_support, tuple)
            or len(self.start_support) != 2
            or any(type(value) is not bool for value in self.start_support)
            or not any(self.start_support)
        ):
            raise ContractError("foothold action bounds or support are invalid")
        if (
            not isinstance(self.landing_feet, tuple)
            or len(self.landing_feet) != 2
            or any(foot not in (0, 1) for foot in self.landing_feet)
            or self.landing_feet[0] == self.landing_feet[1]
        ):
            raise ContractError("foothold action landings must alternate")
        offsets = self.landing_frame_offsets
        if (
            not isinstance(offsets, tuple)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or not 0 < offsets[0] < offsets[1] < self.end_frame - self.start_frame
        ):
            raise ContractError("foothold action landing offsets are invalid")
        xy = _owned_float_tensor(
            self.landing_xy_start_frame_m, (2, 2), "landing positions"
        )
        heights = _owned_float_tensor(
            self.landing_height_delta_m, (2,), "landing heights"
        )
        root = _owned_float_tensor(
            self.root_displacement_m, (2, 2), "root displacements"
        )
        yaw = _owned_float_tensor(
            self.root_yaw_delta_rad, (2,), "root yaw changes"
        )
        clearance = self.minimum_swing_clearance_m
        unsupported = self.maximum_unsupported_frames
        if (
            isinstance(clearance, bool)
            or not isinstance(clearance, (int, float))
            or not math.isfinite(float(clearance))
            or type(unsupported) is not int
            or unsupported < 0
        ):
            raise ContractError("foothold action clearance metadata is invalid")
        object.__setattr__(self, "landing_xy_start_frame_m", xy)
        object.__setattr__(self, "landing_height_delta_m", heights)
        object.__setattr__(self, "root_displacement_m", root)
        object.__setattr__(self, "root_yaw_delta_rad", yaw)
        object.__setattr__(self, "minimum_swing_clearance_m", float(clearance))


def _world_to_local(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x = cosine * vector[..., 0] + sine * vector[..., 1]
    y = -sine * vector[..., 0] + cosine * vector[..., 1]
    return torch.stack((x, y), dim=-1)


def _longest_unsupported(support_mask: torch.Tensor) -> int:
    longest = current = 0
    for supported in support_mask.any(dim=1).detach().cpu().tolist():
        if bool(supported):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def action_from_profiles(
    clip_index: int,
    start_frame: int,
    support_mask: torch.Tensor,
    foot_xy_m: torch.Tensor,
    foot_surface_height_m: torch.Tensor,
    root_xy_m: torch.Tensor,
    root_yaw_rad: torch.Tensor,
) -> FootholdAction | None:
    """Extract the first two alternating landings after a source boundary."""

    if type(clip_index) is not int or clip_index < 0:
        raise ContractError("foothold action clip index is invalid")
    if (
        not isinstance(support_mask, torch.Tensor)
        or support_mask.dtype != torch.bool
        or support_mask.ndim != 2
        or support_mask.shape[1] != 2
    ):
        raise ContractError("foothold action support profile is invalid")
    frames = support_mask.shape[0]
    float_profiles = (
        (foot_xy_m, (frames, 2, 2)),
        (foot_surface_height_m, (frames, 2)),
        (root_xy_m, (frames, 2)),
        (root_yaw_rad, (frames,)),
    )
    if (
        type(start_frame) is not int
        or not 0 <= start_frame < frames - 1
        or any(
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not value.dtype.is_floating_point
            or value.device != support_mask.device
            or not torch.isfinite(value).all()
            for value, shape in float_profiles
        )
    ):
        raise ContractError("foothold action source profiles are invalid")
    if not bool(support_mask[start_frame].any().item()):
        raise ContractError("foothold action must start from support")

    onset = (~support_mask[:-1]) & support_mask[1:]
    events = [
        (int(frame) + 1, int(foot))
        for frame, foot in torch.nonzero(onset, as_tuple=False)
        .detach()
        .cpu()
        .tolist()
        if int(frame) + 1 > start_frame
    ]
    pairs = [
        (first, second)
        for event_index, first in enumerate(events)
        for second in events[event_index + 1 :]
        if second[0] > first[0] and second[1] != first[1]
    ]
    if not pairs:
        return None
    first, second = min(
        pairs,
        key=lambda pair: (
            pair[1][0], pair[0][0], pair[0][1], pair[1][1]
        ),
    )

    landing_frames = torch.tensor(
        (first[0], second[0]), device=support_mask.device
    )
    landing_feet = torch.tensor(
        (first[1], second[1]), device=support_mask.device
    )
    start_root = root_xy_m[start_frame]
    start_yaw = root_yaw_rad[start_frame]
    landing_xy = foot_xy_m[landing_frames, landing_feet]
    local_landing = _world_to_local(landing_xy - start_root, start_yaw)
    local_root = _world_to_local(
        root_xy_m[landing_frames] - start_root, start_yaw
    )
    landing_height = (
        foot_surface_height_m[landing_frames, landing_feet]
        - foot_surface_height_m[start_frame, landing_feet]
    )
    yaw_delta = root_yaw_rad[landing_frames] - start_yaw
    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
    end_frame = second[0] + 1
    return FootholdAction(
        clip_index=clip_index,
        start_frame=start_frame,
        end_frame=end_frame,
        start_support=tuple(
            bool(value) for value in support_mask[start_frame].tolist()
        ),
        landing_feet=(first[1], second[1]),
        landing_frame_offsets=(
            first[0] - start_frame,
            second[0] - start_frame,
        ),
        landing_xy_start_frame_m=local_landing,
        landing_height_delta_m=landing_height,
        root_displacement_m=local_root,
        root_yaw_delta_rad=yaw_delta,
        minimum_swing_clearance_m=0.0,
        maximum_unsupported_frames=_longest_unsupported(
            support_mask[start_frame:end_frame]
        ),
    )


@dataclass(frozen=True)
class FootholdActionIndex:
    """Exact source-entry mapping for authenticated two-contact actions."""

    actions: tuple[FootholdAction, ...]
    _entries: Mapping[tuple[int, int], FootholdAction]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.actions, tuple)
            or any(not isinstance(action, FootholdAction) for action in self.actions)
        ):
            raise ContractError("foothold action inventory is invalid")
        entries = dict(self._entries)
        if (
            len(entries) != len(self.actions)
            or any(
                not any(value is action for action in self.actions)
                for value in entries.values()
            )
            or any(
                key != (action.clip_index, action.start_frame)
                for key, action in entries.items()
            )
        ):
            raise ContractError("foothold action entries are invalid")
        object.__setattr__(self, "_entries", MappingProxyType(entries))

    @classmethod
    def from_dataset(
        cls, dataset: object, segment_index: object
    ) -> "FootholdActionIndex":
        try:
            clips = dataset.folder.clips
            layout = dataset.folder.layout
            grids = dataset.clip_grids
            alignments = dataset.clip_alignments
            device = torch.device(dataset.device)
            segments = segment_index.segments
            support_for_clip = segment_index.support_mask
            root_index = int(layout.root_body_index)
            foot_indices = (
                int(layout.left_foot_body_index),
                int(layout.right_foot_body_index),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError("foothold action dataset is invalid") from error
        if (
            len(clips) != len(grids)
            or len(clips) != len(alignments)
            or not callable(support_for_clip)
        ):
            raise ContractError("foothold action dataset inventory is invalid")

        starts_by_clip: dict[int, list[int]] = {}
        for segment in segments:
            clip_index = int(segment.clip_index)
            start_frame = int(segment.start_frame)
            starts_by_clip.setdefault(clip_index, []).append(start_frame)
        output: list[FootholdAction] = []
        for clip_index in sorted(starts_by_clip):
            if not 0 <= clip_index < len(clips):
                raise ContractError("foothold action segment clip is invalid")
            grid = grids[clip_index]
            alignment = alignments[clip_index]
            if grid is None or alignment is None:
                continue
            clip = clips[clip_index]
            try:
                body_position = torch.tensor(
                    clip.body_position_world,
                    dtype=torch.float32,
                    device=device,
                )
                body_quaternion = torch.tensor(
                    clip.body_quaternion_world_wxyz,
                    dtype=torch.float32,
                    device=device,
                )
                support = support_for_clip(clip_index).to(device=device)
            except (AttributeError, TypeError, RuntimeError) as error:
                raise ContractError(
                    "foothold action clip profiles are invalid"
                ) from error
            frames = body_position.shape[0]
            if (
                body_position.ndim != 3
                or body_position.shape[-1] != 3
                or body_quaternion.shape != (*body_position.shape[:2], 4)
                or support.shape != (frames, 2)
                or support.dtype != torch.bool
                or not torch.isfinite(body_position).all()
                or not torch.isfinite(body_quaternion).all()
            ):
                raise ContractError("foothold action clip profiles are invalid")
            feet = body_position[:, foot_indices]
            root = body_position[:, root_index]
            root_yaw = _yaw_from_wxyz(body_quaternion[:, root_index])
            try:
                foot_scene_xy = alignment.matcher_to_scene_xy(feet[..., :2])
                surface = grid.sample_xy(foot_scene_xy)
            except ContractError:
                raise
            except Exception as error:
                raise ContractError(
                    "foothold action terrain sampling failed"
                ) from error
            for start_frame in sorted(set(starts_by_clip[clip_index])):
                action = action_from_profiles(
                    clip_index=clip_index,
                    start_frame=start_frame,
                    support_mask=support,
                    foot_xy_m=feet[..., :2],
                    foot_surface_height_m=surface,
                    root_xy_m=root[..., :2],
                    root_yaw_rad=root_yaw,
                )
                if action is not None:
                    output.append(action)
        actions = tuple(sorted(output, key=lambda item: (
            item.clip_index, item.start_frame
        )))
        entries = {
            (action.clip_index, action.start_frame): action
            for action in actions
        }
        return cls(actions=actions, _entries=entries)

    def rows_for_database(
        self, database: object
    ) -> tuple[FootholdAction | None, ...]:
        try:
            clips = database._search_clip_index
            frames = database._search_frame_index
            device = torch.device(database.device)
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError("foothold action database is invalid") from error
        if (
            not isinstance(clips, torch.Tensor)
            or not isinstance(frames, torch.Tensor)
            or clips.ndim != 1
            or frames.shape != clips.shape
            or clips.device != frames.device
            or clips.device != device
        ):
            raise ContractError("foothold action database rows are invalid")
        return tuple(
            self._entries.get((int(clip), int(frame)))
            for clip, frame in zip(
                clips.detach().cpu().tolist(),
                frames.detach().cpu().tolist(),
            )
        )
