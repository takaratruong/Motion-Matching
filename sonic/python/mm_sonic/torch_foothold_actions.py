"""Authenticated short-horizon foothold actions for terrain matching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

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


@dataclass(frozen=True)
class FootholdPlan:
    """Deterministic beam of two alternating query-terrain landings."""

    landing_feet: torch.Tensor
    landing_xy_world_m: torch.Tensor
    landing_xy_command_frame_m: torch.Tensor
    landing_height_delta_m: torch.Tensor
    landing_frame_offsets: torch.Tensor
    score: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            (self.landing_feet, (None, 2), False),
            (self.landing_xy_world_m, (None, 2, 2), True),
            (self.landing_xy_command_frame_m, (None, 2, 2), True),
            (self.landing_height_delta_m, (None, 2), True),
            (self.landing_frame_offsets, (None, 2), False),
            (self.score, (None,), True),
        )
        count = None
        owned: list[torch.Tensor] = []
        device = None
        for value, shape, floating in tensors:
            if not isinstance(value, torch.Tensor):
                raise ContractError("foothold plan tensors are invalid")
            if count is None:
                count = value.shape[0] if value.ndim else -1
                device = value.device
            expected = (count, *shape[1:])
            if (
                tuple(value.shape) != expected
                or value.device != device
                or floating != value.dtype.is_floating_point
                or (floating and not torch.isfinite(value).all())
            ):
                raise ContractError("foothold plan tensors are invalid")
            owned.append(value.detach().clone())
        assert count is not None
        if (
            self.landing_feet.dtype != torch.int64
            or self.landing_frame_offsets.dtype != torch.int64
            or (count and not bool(
                ((self.landing_feet == 0) | (self.landing_feet == 1))
                .all()
                .item()
            ))
            or (count and not bool(
                (self.landing_feet[:, 0] != self.landing_feet[:, 1])
                .all()
                .item()
            ))
        ):
            raise ContractError("foothold plan contact identities are invalid")
        for (name, _value), replacement in zip(
            self.__dict__.items(), owned
        ):
            object.__setattr__(self, name, replacement)


def _empty_plan(*, device: torch.device) -> FootholdPlan:
    return FootholdPlan(
        landing_feet=torch.empty((0, 2), dtype=torch.int64, device=device),
        landing_xy_world_m=torch.empty(
            (0, 2, 2), dtype=torch.float32, device=device
        ),
        landing_xy_command_frame_m=torch.empty(
            (0, 2, 2), dtype=torch.float32, device=device
        ),
        landing_height_delta_m=torch.empty(
            (0, 2), dtype=torch.float32, device=device
        ),
        landing_frame_offsets=torch.empty(
            (0, 2), dtype=torch.int64, device=device
        ),
        score=torch.empty((0,), dtype=torch.float32, device=device),
    )


def _stable_surface_height(
    point_xy: torch.Tensor,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    edge_margin_m: float,
) -> torch.Tensor | None:
    offsets = torch.tensor(
        (
            (0.0, 0.0),
            (edge_margin_m, 0.0),
            (-edge_margin_m, 0.0),
            (0.0, edge_margin_m),
            (0.0, -edge_margin_m),
        ),
        dtype=point_xy.dtype,
        device=point_xy.device,
    )
    values = sample_surface(point_xy[None, :] + offsets)
    if (
        not isinstance(values, torch.Tensor)
        or tuple(values.shape) != (5,)
        or values.device != point_xy.device
        or values.dtype != point_xy.dtype
        or not torch.isfinite(values).all()
    ):
        raise ContractError("foothold surface sampler returned invalid heights")
    if float((values.max() - values.min()).item()) > 0.025:
        return None
    return values[0]


def plan_footholds(
    *,
    foot_xy_m: torch.Tensor,
    support_mask: torch.Tensor,
    command_xy: torch.Tensor,
    sample_surface: Callable[[torch.Tensor], torch.Tensor],
    reachable_forward_m: Sequence[float],
    lateral_samples_m: Sequence[float],
    edge_margin_m: float,
    beam_width: int = 8,
) -> FootholdPlan:
    """Enumerate stable two-step placements along the current command."""

    if (
        not isinstance(foot_xy_m, torch.Tensor)
        or tuple(foot_xy_m.shape) != (2, 2)
        or not foot_xy_m.dtype.is_floating_point
        or not torch.isfinite(foot_xy_m).all()
        or not isinstance(support_mask, torch.Tensor)
        or tuple(support_mask.shape) != (2,)
        or support_mask.dtype != torch.bool
        or support_mask.device != foot_xy_m.device
        or not isinstance(command_xy, torch.Tensor)
        or tuple(command_xy.shape) != (2,)
        or command_xy.dtype != foot_xy_m.dtype
        or command_xy.device != foot_xy_m.device
        or not torch.isfinite(command_xy).all()
        or not callable(sample_surface)
        or type(beam_width) is not int
        or beam_width < 1
        or isinstance(edge_margin_m, bool)
        or not isinstance(edge_margin_m, (int, float))
        or not math.isfinite(float(edge_margin_m))
        or float(edge_margin_m) <= 0.0
    ):
        raise ContractError("foothold planning inputs are invalid")
    forward_samples = tuple(float(value) for value in reachable_forward_m)
    lateral_samples = tuple(float(value) for value in lateral_samples_m)
    if (
        not forward_samples
        or not lateral_samples
        or any(not math.isfinite(value) or value <= 0.0 for value in forward_samples)
        or any(not math.isfinite(value) for value in lateral_samples)
    ):
        raise ContractError("foothold reach samples are invalid")
    speed = torch.linalg.vector_norm(command_xy)
    if float(speed.item()) <= 1e-6 or not bool(support_mask.any().item()):
        return _empty_plan(device=foot_xy_m.device)
    forward = command_xy / speed
    right = torch.stack((forward[1], -forward[0]))
    anchor = foot_xy_m.mean(dim=0)
    base_surface = sample_surface(foot_xy_m)
    if (
        not isinstance(base_surface, torch.Tensor)
        or tuple(base_surface.shape) != (2,)
        or base_surface.device != foot_xy_m.device
        or base_surface.dtype != foot_xy_m.dtype
        or not torch.isfinite(base_surface).all()
    ):
        raise ContractError("foothold current surface heights are invalid")
    moving_feet = (
        tuple(range(2))
        if bool(support_mask.all().item())
        else (int(torch.nonzero(~support_mask, as_tuple=False)[0].item()),)
    )
    candidates: list[tuple] = []
    command_yaw = torch.atan2(forward[1], forward[0])
    for moving in moving_feet:
        other = 1 - moving
        for distance in sorted(set(forward_samples)):
            for lateral in sorted(set(lateral_samples)):
                first_xy = (
                    foot_xy_m[moving]
                    + forward * distance
                    + right * lateral
                )
                first_height = _stable_surface_height(
                    first_xy, sample_surface, float(edge_margin_m)
                )
                if first_height is None:
                    continue
                second_xy = (
                    foot_xy_m[other]
                    + forward * (2.0 * distance)
                    + right * lateral
                )
                second_height = _stable_surface_height(
                    second_xy, sample_surface, float(edge_margin_m)
                )
                if second_height is None:
                    continue
                world = torch.stack((first_xy, second_xy))
                local = _world_to_local(world - anchor, command_yaw)
                heights = torch.stack(
                    (
                        first_height - base_surface[moving],
                        second_height - base_surface[other],
                    )
                )
                times = torch.tensor(
                    (
                        max(1, round(distance / float(speed.item()) * 50.0)),
                        max(2, round(2.0 * distance / float(speed.item()) * 50.0)),
                    ),
                    dtype=torch.int64,
                    device=foot_xy_m.device,
                )
                score = -3.0 * distance + abs(lateral)
                candidates.append(
                    (
                        score,
                        moving,
                        distance,
                        lateral,
                        torch.tensor((moving, other), device=foot_xy_m.device),
                        world,
                        local,
                        heights,
                        times,
                    )
                )
    candidates.sort(key=lambda item: item[:4])
    retained = candidates[:beam_width]
    if not retained:
        return _empty_plan(device=foot_xy_m.device)
    return FootholdPlan(
        landing_feet=torch.stack([item[4] for item in retained]).to(torch.int64),
        landing_xy_world_m=torch.stack([item[5] for item in retained]),
        landing_xy_command_frame_m=torch.stack([item[6] for item in retained]),
        landing_height_delta_m=torch.stack([item[7] for item in retained]),
        landing_frame_offsets=torch.stack([item[8] for item in retained]),
        score=torch.tensor(
            [item[0] for item in retained],
            dtype=foot_xy_m.dtype,
            device=foot_xy_m.device,
        ),
    )


def first_contact_eligibility(
    *,
    plan: FootholdPlan,
    actions: Sequence[FootholdAction],
    height_tolerance_m: float,
    xy_tolerance_m: float,
    timing_tolerance_frames: int,
) -> torch.Tensor:
    """Hard-gate actions against at least one planned first landing."""

    if not isinstance(plan, FootholdPlan):
        raise ContractError("first-contact filter requires a foothold plan")
    tolerances = (height_tolerance_m, xy_tolerance_m)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in tolerances
        )
        or type(timing_tolerance_frames) is not int
        or timing_tolerance_frames < 0
        or any(not isinstance(action, FootholdAction) for action in actions)
    ):
        raise ContractError("first-contact filter inputs are invalid")
    output = torch.zeros(
        len(actions), dtype=torch.bool, device=plan.score.device
    )
    for action_index, action in enumerate(actions):
        if action.landing_height_delta_m.device != plan.score.device:
            raise ContractError("first-contact action device is invalid")
        feet = plan.landing_feet[:, 0] == action.landing_feet[0]
        height = (
            plan.landing_height_delta_m[:, 0]
            - action.landing_height_delta_m[0]
        ).abs() <= float(height_tolerance_m)
        xy = torch.linalg.vector_norm(
            plan.landing_xy_command_frame_m[:, 0]
            - action.landing_xy_start_frame_m[0],
            dim=-1,
        ) <= float(xy_tolerance_m)
        timing = (
            plan.landing_frame_offsets[:, 0]
            - action.landing_frame_offsets[0]
        ).abs() <= timing_tolerance_frames
        output[action_index] = bool((feet & height & xy & timing).any().item())
    return output


class FootholdSelectionArm(Enum):
    FIRST_CONTACT = "first-contact"
    TWO_CONTACT = "two-contact"
    HYBRID = "hybrid"
    CONTINUOUS = "continuous-control"


@dataclass(frozen=True)
class FootholdRanking:
    eligible: torch.Tensor
    additional_cost: torch.Tensor
    selected_action: int | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.eligible, torch.Tensor)
            or self.eligible.ndim != 1
            or self.eligible.dtype != torch.bool
            or not isinstance(self.additional_cost, torch.Tensor)
            or self.additional_cost.shape != self.eligible.shape
            or not self.additional_cost.dtype.is_floating_point
            or self.additional_cost.device != self.eligible.device
            or not torch.isfinite(self.additional_cost).all()
        ):
            raise ContractError("foothold ranking tensors are invalid")
        selected = self.selected_action
        if selected is not None and (
            type(selected) is not int
            or not 0 <= selected < self.eligible.shape[0]
            or not bool(self.eligible[selected].item())
        ):
            raise ContractError("foothold ranking selection is invalid")
        object.__setattr__(self, "eligible", self.eligible.detach().clone())
        object.__setattr__(
            self, "additional_cost", self.additional_cost.detach().clone()
        )


def rank_foothold_actions(
    *,
    arm: FootholdSelectionArm,
    plan: FootholdPlan,
    actions: Sequence[FootholdAction],
    motion_cost: torch.Tensor,
    height_tolerance_m: float,
    xy_tolerance_m: float,
    timing_tolerance_frames: int,
) -> FootholdRanking:
    """Compare hard first/two-contact gates with a continuous control arm."""

    if (
        not isinstance(arm, FootholdSelectionArm)
        or not isinstance(plan, FootholdPlan)
        or not isinstance(motion_cost, torch.Tensor)
        or tuple(motion_cost.shape) != (len(actions),)
        or not motion_cost.dtype.is_floating_point
        or motion_cost.device != plan.score.device
        or not torch.isfinite(motion_cost).all()
        or any(not isinstance(action, FootholdAction) for action in actions)
    ):
        raise ContractError("foothold ranking inputs are invalid")
    tolerances = (height_tolerance_m, xy_tolerance_m)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in tolerances
        )
        or type(timing_tolerance_frames) is not int
        or timing_tolerance_frames < 0
    ):
        raise ContractError("foothold ranking tolerances are invalid")
    count = len(actions)
    eligible = torch.zeros(count, dtype=torch.bool, device=motion_cost.device)
    descriptor_cost = torch.zeros_like(motion_cost)
    if plan.score.shape[0] == 0:
        return FootholdRanking(eligible, descriptor_cost, None)

    for action_index, action in enumerate(actions):
        tensors = (
            action.landing_xy_start_frame_m,
            action.landing_height_delta_m,
        )
        if any(value.device != motion_cost.device for value in tensors):
            raise ContractError("foothold ranking action device is invalid")
        action_feet = torch.tensor(
            action.landing_feet,
            dtype=torch.int64,
            device=motion_cost.device,
        )
        feet = plan.landing_feet == action_feet[None, :]
        height_error = torch.abs(
            plan.landing_height_delta_m
            - action.landing_height_delta_m[None, :]
        )
        xy_error = torch.linalg.vector_norm(
            plan.landing_xy_command_frame_m
            - action.landing_xy_start_frame_m[None, :, :],
            dim=-1,
        )
        action_timing = torch.tensor(
            action.landing_frame_offsets,
            dtype=torch.int64,
            device=motion_cost.device,
        )
        timing_error = torch.abs(
            plan.landing_frame_offsets - action_timing[None, :]
        )
        first_valid = (
            feet[:, 0]
            & (height_error[:, 0] <= float(height_tolerance_m))
            & (xy_error[:, 0] <= float(xy_tolerance_m))
            & (timing_error[:, 0] <= timing_tolerance_frames)
        )
        both_valid = (
            feet.all(dim=1)
            & (height_error <= float(height_tolerance_m)).all(dim=1)
            & (xy_error <= float(xy_tolerance_m)).all(dim=1)
            & (timing_error <= timing_tolerance_frames).all(dim=1)
        )
        normalized = (
            torch.square(height_error / float(height_tolerance_m)).sum(dim=1)
            + torch.square(xy_error / float(xy_tolerance_m)).sum(dim=1)
            + torch.square(
                timing_error.to(motion_cost.dtype)
                / float(max(1, timing_tolerance_frames))
            ).sum(dim=1)
            + (~feet).to(motion_cost.dtype).sum(dim=1) * 100.0
            + plan.score
        )
        descriptor_cost[action_index] = normalized.min().clamp_min(0.0)
        if arm is FootholdSelectionArm.FIRST_CONTACT:
            eligible[action_index] = bool(first_valid.any().item())
        elif arm in (
            FootholdSelectionArm.TWO_CONTACT,
            FootholdSelectionArm.HYBRID,
        ):
            eligible[action_index] = bool(both_valid.any().item())
        else:
            eligible[action_index] = True

    total = motion_cost + descriptor_cost
    selected = None
    if bool(eligible.any().item()):
        masked = torch.where(
            eligible,
            total,
            torch.full_like(total, torch.inf),
        )
        selected = int(torch.argmin(masked).item())
    return FootholdRanking(eligible, descriptor_cost, selected)


@dataclass(frozen=True)
class FootholdPolicyResult:
    row_eligibility: torch.Tensor
    additional_row_cost: torch.Tensor
    plan: FootholdPlan
    candidate_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.row_eligibility, torch.Tensor)
            or self.row_eligibility.ndim != 1
            or self.row_eligibility.dtype != torch.bool
            or not isinstance(self.additional_row_cost, torch.Tensor)
            or self.additional_row_cost.shape != self.row_eligibility.shape
            or not self.additional_row_cost.dtype.is_floating_point
            or self.additional_row_cost.device != self.row_eligibility.device
            or not torch.isfinite(self.additional_row_cost).all()
            or bool((self.additional_row_cost < 0.0).any().item())
            or not isinstance(self.plan, FootholdPlan)
            or self.plan.score.device != self.row_eligibility.device
            or type(self.candidate_count) is not int
            or self.candidate_count < 0
        ):
            raise ContractError("foothold policy result is invalid")
        object.__setattr__(
            self, "row_eligibility", self.row_eligibility.detach().clone()
        )
        object.__setattr__(
            self,
            "additional_row_cost",
            self.additional_row_cost.detach().clone(),
        )


@dataclass
class FootholdActionPolicy:
    """Map a query heightmap plan into exact motion-database row gates."""

    index: FootholdActionIndex
    extension: object
    arm: FootholdSelectionArm
    height_tolerance_m: float = 0.06
    xy_tolerance_m: float = 0.25
    timing_tolerance_frames: int = 20
    edge_margin_m: float = 0.04
    beam_width: int = 16

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index, FootholdActionIndex)
            or not isinstance(self.arm, FootholdSelectionArm)
            or not callable(
                getattr(getattr(self.extension, "query_grid", None),
                        "sample_xy", None)
            )
            or not callable(
                getattr(getattr(self.extension, "alignment", None),
                        "matcher_to_scene_xy", None)
            )
        ):
            raise ContractError("foothold action policy inputs are invalid")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in (
                    self.height_tolerance_m,
                    self.xy_tolerance_m,
                    self.edge_margin_m,
                )
            )
            or type(self.timing_tolerance_frames) is not int
            or self.timing_tolerance_frames < 0
            or type(self.beam_width) is not int
            or self.beam_width < 1
        ):
            raise ContractError("foothold action policy tolerances are invalid")
        self._terrain_clip_indices = frozenset(
            action.clip_index for action in self.index.actions
        )
        self._action_indices = {
            (action.clip_index, action.start_frame): action_index
            for action_index, action in enumerate(self.index.actions)
        }
        self._cached_database_id: int | None = None
        self._cached_rows: tuple[FootholdAction | None, ...] | None = None

    def _sample_matcher_surface(self, points: torch.Tensor) -> torch.Tensor:
        try:
            scene = self.extension.alignment.matcher_to_scene_xy(points)
            return self.extension.query_grid.sample_xy(scene)
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("foothold query terrain sampling failed") from error

    def _rows(
        self, database: object
    ) -> tuple[FootholdAction | None, ...]:
        token = id(database)
        if self._cached_database_id != token or self._cached_rows is None:
            self._cached_rows = self.index.rows_for_database(database)
            self._cached_database_id = token
        return self._cached_rows

    def prepare(
        self, state: object, shaped: object, database: object
    ) -> FootholdPolicyResult:
        try:
            body = state.feature_body_position
            command = shaped.velocity_world_xy
            device = torch.device(database.device)
            database_clips = database._search_clip_index
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError("foothold policy state is invalid") from error
        if (
            not isinstance(body, torch.Tensor)
            or tuple(body.shape) != (3, 3)
            or not body.dtype.is_floating_point
            or body.device != device
            or not torch.isfinite(body).all()
            or not isinstance(command, torch.Tensor)
            or tuple(command.shape) != (2,)
            or command.dtype != body.dtype
            or command.device != device
            or not torch.isfinite(command).all()
        ):
            raise ContractError("foothold policy state tensors are invalid")
        feet = body[1:, :]
        surface = self._sample_matcher_surface(feet[:, :2])
        if (
            not isinstance(surface, torch.Tensor)
            or tuple(surface.shape) != (2,)
            or surface.device != device
            or surface.dtype != body.dtype
            or not torch.isfinite(surface).all()
        ):
            raise ContractError("foothold policy support surface is invalid")
        clearance = feet[:, 2] - surface
        support = torch.abs(clearance - 0.035) <= 0.020
        plan = plan_footholds(
            foot_xy_m=feet[:, :2],
            support_mask=support,
            command_xy=command,
            sample_surface=self._sample_matcher_surface,
            reachable_forward_m=(0.20, 0.25, 0.30, 0.35, 0.40),
            lateral_samples_m=(-0.15, -0.075, 0.0, 0.075, 0.15),
            edge_margin_m=self.edge_margin_m,
            beam_width=self.beam_width,
        )
        actions = self.index.actions
        ranking = rank_foothold_actions(
            arm=self.arm,
            plan=plan,
            actions=actions,
            motion_cost=torch.zeros(
                len(actions), dtype=torch.float32, device=device
            ),
            height_tolerance_m=self.height_tolerance_m,
            xy_tolerance_m=self.xy_tolerance_m,
            timing_tolerance_frames=self.timing_tolerance_frames,
        )
        rows = self._rows(database)
        row_count = len(rows)
        eligible = torch.ones(row_count, dtype=torch.bool, device=device)
        additional = torch.zeros(row_count, dtype=torch.float32, device=device)
        for row, action in enumerate(rows):
            clip_index = int(database_clips[row].item())
            if clip_index not in self._terrain_clip_indices:
                continue
            eligible[row] = False
            if action is None:
                continue
            action_index = self._action_indices[
                (action.clip_index, action.start_frame)
            ]
            eligible[row] = ranking.eligible[action_index]
            additional[row] = ranking.additional_cost[action_index]
        return FootholdPolicyResult(
            row_eligibility=eligible,
            additional_row_cost=additional,
            plan=plan,
            candidate_count=int(ranking.eligible.sum().item()),
        )
