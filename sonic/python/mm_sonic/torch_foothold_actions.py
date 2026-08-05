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


def transition_graph_required(
    *,
    root_quaternion_world_wxyz: torch.Tensor,
    future_facing_world_xy: torch.Tensor,
    maximum_heading_error_rad: float,
) -> bool:
    """Use pose-compatible edges only after facing has joined the command."""

    if (
        not isinstance(root_quaternion_world_wxyz, torch.Tensor)
        or tuple(root_quaternion_world_wxyz.shape) != (4,)
        or not root_quaternion_world_wxyz.dtype.is_floating_point
        or not torch.isfinite(root_quaternion_world_wxyz).all()
        or not isinstance(future_facing_world_xy, torch.Tensor)
        or tuple(future_facing_world_xy.shape) != (2,)
        or future_facing_world_xy.dtype
        != root_quaternion_world_wxyz.dtype
        or future_facing_world_xy.device
        != root_quaternion_world_wxyz.device
        or not torch.isfinite(future_facing_world_xy).all()
        or isinstance(maximum_heading_error_rad, bool)
        or not isinstance(maximum_heading_error_rad, (int, float))
        or not math.isfinite(float(maximum_heading_error_rad))
        or not 0.0 < float(maximum_heading_error_rad) <= math.pi
    ):
        raise ContractError("foothold transition heading query is invalid")
    facing_norm = torch.linalg.vector_norm(future_facing_world_xy)
    if float(facing_norm.item()) <= 1e-6:
        raise ContractError("foothold transition future facing is zero")
    root_yaw = _yaw_from_wxyz(root_quaternion_world_wxyz)
    future = future_facing_world_xy / facing_norm
    future_yaw = torch.atan2(future[1], future[0])
    error = torch.atan2(
        torch.sin(future_yaw - root_yaw),
        torch.cos(future_yaw - root_yaw),
    )
    return abs(float(error.item())) <= float(maximum_heading_error_rad)


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
                entries.get((action.clip_index, action.start_frame)) is not action
                for action in self.actions
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

    def entry(
        self, clip_index: int, frame_index: int
    ) -> FootholdAction | None:
        if type(clip_index) is not int or type(frame_index) is not int:
            raise ContractError("foothold action entry indices must be integers")
        return self._entries.get((clip_index, frame_index))

    def next_entry_frame(
        self, clip_index: int, frame_index: int
    ) -> int | None:
        if type(clip_index) is not int or type(frame_index) is not int:
            raise ContractError("foothold action entry indices must be integers")
        return min(
            (
                action.start_frame
                for action in self.actions
                if action.clip_index == clip_index
                and action.start_frame > frame_index
            ),
            default=None,
        )


@dataclass(frozen=True)
class FootholdTransitionGraph:
    """Source action entries used to validate terminal-to-entry edges."""

    action_keys: tuple[tuple[int, int], ...]
    entry_joint_position: torch.Tensor
    entry_joint_velocity: torch.Tensor
    terminal_joint_position: torch.Tensor | None = None
    terminal_joint_velocity: torch.Tensor | None = None

    def __post_init__(self) -> None:
        count = len(self.action_keys)
        if (
            not isinstance(self.action_keys, tuple)
            or any(
                not isinstance(key, tuple)
                or len(key) != 2
                or any(type(value) is not int or value < 0 for value in key)
                for key in self.action_keys
            )
            or len(set(self.action_keys)) != count
        ):
            raise ContractError("foothold transition graph keys are invalid")
        owned = []
        device = None
        terminal_position = (
            self.entry_joint_position
            if self.terminal_joint_position is None
            else self.terminal_joint_position
        )
        terminal_velocity = (
            self.entry_joint_velocity
            if self.terminal_joint_velocity is None
            else self.terminal_joint_velocity
        )
        for value in (
            self.entry_joint_position,
            self.entry_joint_velocity,
            terminal_position,
            terminal_velocity,
        ):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (count, 29)
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
                or (device is not None and value.device != device)
            ):
                raise ContractError(
                    "foothold transition graph states are invalid"
                )
            device = value.device
            owned.append(value.detach().clone())
        object.__setattr__(self, "entry_joint_position", owned[0])
        object.__setattr__(self, "entry_joint_velocity", owned[1])
        object.__setattr__(self, "terminal_joint_position", owned[2])
        object.__setattr__(self, "terminal_joint_velocity", owned[3])

    @classmethod
    def from_dataset(
        cls, index: FootholdActionIndex, dataset: object
    ) -> "FootholdTransitionGraph":
        if not isinstance(index, FootholdActionIndex):
            raise ContractError("foothold transition graph index is invalid")
        try:
            clips = dataset.folder.clips
            device = torch.device(dataset.device)
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError(
                "foothold transition graph dataset is invalid"
            ) from error
        positions = []
        velocities = []
        terminal_positions = []
        terminal_velocities = []
        for action in index.actions:
            try:
                clip = clips[action.clip_index]
                positions.append(
                    torch.tensor(
                        clip.joint_position[action.start_frame],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                velocities.append(
                    torch.tensor(
                        clip.joint_velocity[action.start_frame],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                terminal_frame = action.end_frame - 1
                terminal_positions.append(
                    torch.tensor(
                        clip.joint_position[terminal_frame],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                terminal_velocities.append(
                    torch.tensor(
                        clip.joint_velocity[terminal_frame],
                        dtype=torch.float32,
                        device=device,
                    )
                )
            except (AttributeError, IndexError, TypeError, RuntimeError) as error:
                raise ContractError(
                    "foothold transition graph source state is invalid"
                ) from error
        if positions:
            position = torch.stack(positions)
            velocity = torch.stack(velocities)
            terminal_position = torch.stack(terminal_positions)
            terminal_velocity = torch.stack(terminal_velocities)
        else:
            position = torch.empty((0, 29), dtype=torch.float32, device=device)
            velocity = torch.empty((0, 29), dtype=torch.float32, device=device)
            terminal_position = torch.empty(
                (0, 29), dtype=torch.float32, device=device
            )
            terminal_velocity = torch.empty(
                (0, 29), dtype=torch.float32, device=device
            )
        return cls(
            action_keys=tuple(
                (action.clip_index, action.start_frame)
                for action in index.actions
            ),
            entry_joint_position=position,
            entry_joint_velocity=velocity,
            terminal_joint_position=terminal_position,
            terminal_joint_velocity=terminal_velocity,
        )

    def pair_eligibility(
        self,
        *,
        maximum_position_error_rad: float,
        maximum_velocity_error_rad_s: float,
    ) -> torch.Tensor:
        limits = (maximum_position_error_rad, maximum_velocity_error_rad_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in limits
        ):
            raise ContractError("foothold transition pair limits are invalid")
        assert self.terminal_joint_position is not None
        assert self.terminal_joint_velocity is not None
        position = torch.cdist(
            self.terminal_joint_position,
            self.entry_joint_position,
        )
        velocity = torch.cdist(
            self.terminal_joint_velocity,
            self.entry_joint_velocity,
        )
        return (
            (position <= float(maximum_position_error_rad))
            & (velocity <= float(maximum_velocity_error_rad_s))
        )

    def entry_eligibility(
        self,
        *,
        current_joint_position: torch.Tensor,
        current_joint_velocity: torch.Tensor,
        maximum_position_error_rad: float,
        maximum_velocity_error_rad_s: float,
    ) -> torch.Tensor:
        for value, label in (
            (current_joint_position, "position"),
            (current_joint_velocity, "velocity"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (29,)
                or value.dtype != torch.float32
                or value.device != self.entry_joint_position.device
                or not torch.isfinite(value).all()
            ):
                raise ContractError(
                    f"foothold transition current {label} is invalid"
                )
        limits = (maximum_position_error_rad, maximum_velocity_error_rad_s)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in limits
        ):
            raise ContractError("foothold transition limits are invalid")
        position_error = torch.linalg.vector_norm(
            self.entry_joint_position - current_joint_position[None, :], dim=1
        )
        velocity_error = torch.linalg.vector_norm(
            self.entry_joint_velocity - current_joint_velocity[None, :], dim=1
        )
        return (
            (position_error <= float(maximum_position_error_rad))
            & (velocity_error <= float(maximum_velocity_error_rad_s))
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
    proposals: list[tuple] = []
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
                second_xy = (
                    foot_xy_m[other]
                    + forward * (2.0 * distance)
                    + right * lateral
                )
                proposals.append(
                    (
                        moving,
                        other,
                        distance,
                        lateral,
                        torch.stack((first_xy, second_xy)),
                    )
                )
    worlds = torch.stack([item[4] for item in proposals])
    sole_offsets = torch.tensor(
        (
            (0.0, 0.0),
            (edge_margin_m, 0.0),
            (-edge_margin_m, 0.0),
            (0.0, edge_margin_m),
            (0.0, -edge_margin_m),
        ),
        dtype=foot_xy_m.dtype,
        device=foot_xy_m.device,
    )
    sampled = sample_surface(
        worlds[:, :, None, :] + sole_offsets[None, None, :, :]
    )
    expected_surface_shape = (len(proposals), 2, 5)
    if (
        not isinstance(sampled, torch.Tensor)
        or tuple(sampled.shape) != expected_surface_shape
        or sampled.device != foot_xy_m.device
        or sampled.dtype != foot_xy_m.dtype
        or not torch.isfinite(sampled).all()
    ):
        raise ContractError("foothold surface sampler returned invalid heights")
    stable = (sampled.amax(dim=-1) - sampled.amin(dim=-1)) <= 0.025
    stable_pairs = stable.all(dim=-1).detach().cpu().tolist()
    center_heights = sampled[:, :, 0]
    speed_value = float(speed.item())
    candidates: list[tuple] = []
    for proposal_index, proposal in enumerate(proposals):
        if stable_pairs[proposal_index]:
            moving, other, distance, lateral, world = proposal
            local = _world_to_local(world - anchor, command_yaw)
            heights = torch.stack(
                (
                    center_heights[proposal_index, 0]
                    - base_surface[moving],
                    center_heights[proposal_index, 1]
                    - base_surface[other],
                )
            )
            times = torch.tensor(
                (
                    max(1, round(distance / speed_value * 50.0)),
                    max(2, round(2.0 * distance / speed_value * 50.0)),
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


def foothold_action_descriptor_cost(
    *,
    landing_feet: torch.Tensor,
    landing_xy_heading_m: torch.Tensor,
    landing_height_delta_m: torch.Tensor,
    landing_frame_offsets: torch.Tensor,
    action: FootholdAction,
    xy_tolerance_m: float,
    height_tolerance_m: float,
    timing_tolerance_frames: int,
) -> torch.Tensor:
    """Hard-gate and score one planned pair against one source action."""

    tensors = (
        (landing_feet, (2,), False),
        (landing_xy_heading_m, (2, 2), True),
        (landing_height_delta_m, (2,), True),
        (landing_frame_offsets, (2,), False),
    )
    device = None
    dtype = None
    for value, shape, floating in tensors:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype.is_floating_point != floating
            or (device is not None and value.device != device)
            or (floating and not torch.isfinite(value).all())
        ):
            raise ContractError(
                "foothold action descriptor query is invalid"
            )
        device = value.device
        if floating:
            if dtype is not None and value.dtype != dtype:
                raise ContractError(
                    "foothold action descriptor dtypes differ"
                )
            dtype = value.dtype
    if (
        landing_feet.dtype != torch.int64
        or landing_frame_offsets.dtype != torch.int64
        or not isinstance(action, FootholdAction)
        or action.landing_xy_start_frame_m.device != device
        or action.landing_xy_start_frame_m.dtype != dtype
        or action.landing_height_delta_m.device != device
        or action.landing_height_delta_m.dtype != dtype
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (xy_tolerance_m, height_tolerance_m)
        )
        or type(timing_tolerance_frames) is not int
        or timing_tolerance_frames < 0
    ):
        raise ContractError("foothold action descriptor query is invalid")
    assert dtype is not None

    rejected = torch.full((), torch.inf, dtype=dtype, device=device)
    action_feet = torch.tensor(
        action.landing_feet, dtype=torch.int64, device=device
    )
    if not bool((landing_feet == action_feet).all().item()):
        return rejected
    xy_error = torch.linalg.vector_norm(
        landing_xy_heading_m - action.landing_xy_start_frame_m,
        dim=-1,
    )
    height_error = torch.abs(
        landing_height_delta_m - action.landing_height_delta_m
    )
    action_timing = torch.tensor(
        action.landing_frame_offsets,
        dtype=torch.int64,
        device=device,
    )
    timing_error = torch.abs(landing_frame_offsets - action_timing)
    if bool(
        (xy_error > float(xy_tolerance_m)).any().item()
        or (height_error > float(height_tolerance_m)).any().item()
        or (timing_error > timing_tolerance_frames).any().item()
    ):
        return rejected
    return (
        torch.square(xy_error / float(xy_tolerance_m)).sum()
        + torch.square(
            height_error / float(height_tolerance_m)
        ).sum()
        + torch.square(
            timing_error.to(dtype)
            / float(max(1, timing_tolerance_frames))
        ).sum()
    )


class FootholdSelectionArm(Enum):
    FIRST_CONTACT = "first-contact"
    TWO_CONTACT = "two-contact"
    HYBRID = "hybrid"
    LAYERED = "layered"
    LAYERED_HYBRID = "layered-hybrid"
    LAYERED_GRAPH_HYBRID = "layered-graph-hybrid"
    CONTINUOUS = "continuous-control"


@dataclass(frozen=True)
class FootholdRanking:
    eligible: torch.Tensor
    additional_cost: torch.Tensor
    selected_action: int | None
    selected_successor_action: int | None = None

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
        successor = self.selected_successor_action
        if successor is not None and (
            selected is None
            or type(successor) is not int
            or not 0 <= successor < self.eligible.shape[0]
        ):
            raise ContractError("foothold ranking successor is invalid")
        object.__setattr__(self, "eligible", self.eligible.detach().clone())
        object.__setattr__(
            self, "additional_cost", self.additional_cost.detach().clone()
        )


def _two_action_plan_tensors(
    *,
    terminal_yaw: torch.Tensor,
    terminal_root_displacement: torch.Tensor,
    terminal_frame_offset: torch.Tensor,
    desired_yaw_delta_rad: float,
    command_speed_mps: float,
    yaw_tolerance_rad: float,
    xy_tolerance_m: float,
    yaw_cost_weight: float,
    root_lateral_cost_weight: float,
    root_progress_cost_weight: float,
    pair_eligibility: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = terminal_yaw.shape[0]
    output = torch.full_like(terminal_yaw, torch.inf)
    successor = torch.full(
        (count,), -1, dtype=torch.int64, device=terminal_yaw.device
    )
    desired = torch.as_tensor(
        desired_yaw_delta_rad,
        dtype=terminal_yaw.dtype,
        device=terminal_yaw.device,
    )
    second_yaw = terminal_yaw[None, :]
    second_root = terminal_root_displacement[None, :, :]
    second_time = terminal_frame_offset[None, :].to(terminal_yaw.dtype)
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        first_yaw = terminal_yaw[start:stop, None]
        cosine = torch.cos(first_yaw)
        sine = torch.sin(first_yaw)
        rotated_second = torch.stack(
            (
                cosine * second_root[..., 0]
                - sine * second_root[..., 1],
                sine * second_root[..., 0]
                + cosine * second_root[..., 1],
            ),
            dim=-1,
        )
        cumulative_root = (
            terminal_root_displacement[start:stop, None, :]
            + rotated_second
        )
        command_root = _world_to_local(cumulative_root, desired)
        yaw_error = torch.atan2(
            torch.sin(first_yaw + second_yaw - desired),
            torch.cos(first_yaw + second_yaw - desired),
        )
        expected_progress = (
            terminal_frame_offset[start:stop, None].to(terminal_yaw.dtype)
            + second_time
        ) * (float(command_speed_mps) / 50.0)
        pair_cost = (
            float(yaw_cost_weight)
            * torch.square(yaw_error / float(yaw_tolerance_rad))
            + float(root_lateral_cost_weight)
            * torch.square(command_root[..., 1] / float(xy_tolerance_m))
            + float(root_progress_cost_weight)
            * torch.square(
                (command_root[..., 0] - expected_progress)
                / float(xy_tolerance_m)
            )
        )
        feasible = torch.abs(yaw_error) <= float(yaw_tolerance_rad)
        if pair_eligibility is not None:
            feasible &= pair_eligibility[start:stop]
        pair_cost = torch.where(
            feasible, pair_cost, torch.full_like(pair_cost, torch.inf)
        )
        best = pair_cost.min(dim=1)
        output[start:stop] = best.values
        successor[start:stop] = torch.where(
            torch.isfinite(best.values),
            best.indices,
            torch.full_like(best.indices, -1),
        )
    return output, successor


def _two_action_cost_tensors(**kwargs) -> torch.Tensor:
    return _two_action_plan_tensors(**kwargs)[0]


def two_action_lookahead_cost(
    *,
    actions: Sequence[FootholdAction],
    desired_yaw_delta_rad: float,
    command_speed_mps: float,
    yaw_tolerance_rad: float,
    xy_tolerance_m: float,
    yaw_cost_weight: float,
    root_lateral_cost_weight: float,
    root_progress_cost_weight: float,
) -> torch.Tensor:
    """Return the best cumulative terminal cost for each first action."""

    if not actions or any(
        not isinstance(action, FootholdAction) for action in actions
    ):
        raise ContractError("two-action lookahead actions are invalid")
    values = (
        desired_yaw_delta_rad,
        command_speed_mps,
        yaw_tolerance_rad,
        xy_tolerance_m,
        yaw_cost_weight,
        root_lateral_cost_weight,
        root_progress_cost_weight,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ) or (
        float(command_speed_mps) < 0.0
        or not 0.0 < float(yaw_tolerance_rad) <= math.pi
        or float(xy_tolerance_m) <= 0.0
        or any(float(value) < 0.0 for value in values[4:])
    ):
        raise ContractError("two-action lookahead parameters are invalid")
    device = actions[0].root_displacement_m.device
    if any(action.root_displacement_m.device != device for action in actions):
        raise ContractError("two-action lookahead devices are invalid")
    return _two_action_cost_tensors(
        terminal_yaw=torch.stack(
            [action.root_yaw_delta_rad[-1] for action in actions]
        ),
        terminal_root_displacement=torch.stack(
            [action.root_displacement_m[-1] for action in actions]
        ),
        terminal_frame_offset=torch.tensor(
            [action.landing_frame_offsets[-1] for action in actions],
            dtype=torch.int64,
            device=device,
        ),
        desired_yaw_delta_rad=desired_yaw_delta_rad,
        command_speed_mps=command_speed_mps,
        yaw_tolerance_rad=yaw_tolerance_rad,
        xy_tolerance_m=xy_tolerance_m,
        yaw_cost_weight=yaw_cost_weight,
        root_lateral_cost_weight=root_lateral_cost_weight,
        root_progress_cost_weight=root_progress_cost_weight,
    )


def two_action_terrain_eligibility(
    *,
    first_plan: FootholdPlan,
    second_plans: Sequence[FootholdPlan],
    actions: Sequence[FootholdAction],
    height_tolerance_m: float,
) -> torch.Tensor:
    """Gate action pairs against four query-terrain contact heights."""

    if (
        not isinstance(first_plan, FootholdPlan)
        or len(second_plans) != first_plan.score.shape[0]
        or any(not isinstance(plan, FootholdPlan) for plan in second_plans)
        or any(not isinstance(action, FootholdAction) for action in actions)
        or isinstance(height_tolerance_m, bool)
        or not isinstance(height_tolerance_m, (int, float))
        or not math.isfinite(float(height_tolerance_m))
        or float(height_tolerance_m) <= 0.0
    ):
        raise ContractError("two-action terrain inputs are invalid")
    count = len(actions)
    device = first_plan.score.device
    output = torch.zeros((count, count), dtype=torch.bool, device=device)
    if not count or first_plan.score.shape[0] == 0:
        return output
    action_feet = torch.tensor(
        [action.landing_feet for action in actions],
        dtype=torch.int64,
        device=device,
    )
    action_height = torch.stack(
        [action.landing_height_delta_m for action in actions]
    )
    if action_height.device != device or any(
        plan.score.device != device for plan in second_plans
    ):
        raise ContractError("two-action terrain devices are invalid")
    for plan_index, second_plan in enumerate(second_plans):
        first_valid = (
            (action_feet == first_plan.landing_feet[plan_index]).all(dim=1)
            & (
                torch.abs(
                    action_height
                    - first_plan.landing_height_delta_m[plan_index]
                )
                <= float(height_tolerance_m)
            ).all(dim=1)
        )
        second_valid = torch.zeros(count, dtype=torch.bool, device=device)
        for second_index in range(second_plan.score.shape[0]):
            second_valid |= (
                (action_feet == second_plan.landing_feet[second_index]).all(
                    dim=1
                )
                & (
                    torch.abs(
                        action_height
                        - second_plan.landing_height_delta_m[second_index]
                    )
                    <= float(height_tolerance_m)
                ).all(dim=1)
            )
        output |= first_valid[:, None] & second_valid[None, :]
    return output


def rank_foothold_actions(
    *,
    arm: FootholdSelectionArm,
    plan: FootholdPlan,
    actions: Sequence[FootholdAction],
    motion_cost: torch.Tensor,
    height_tolerance_m: float,
    xy_tolerance_m: float,
    timing_tolerance_frames: int,
    desired_yaw_delta_rad: float | None = None,
    command_frame_yaw_delta_rad: float | None = None,
    command_speed_mps: float | None = None,
    hard_yaw_gate: bool = True,
    yaw_tolerance_rad: float = math.pi / 6.0,
    yaw_cost_weight: float = 100.0,
    turn_xy_cost_weight: float = 10.0,
    turn_root_lateral_cost_weight: float = 100.0,
    turn_root_progress_cost_weight: float = 10.0,
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
    if type(hard_yaw_gate) is not bool or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (
            yaw_cost_weight,
            turn_xy_cost_weight,
            turn_root_lateral_cost_weight,
            turn_root_progress_cost_weight,
        )
    ) or (command_frame_yaw_delta_rad is not None and (
        isinstance(command_frame_yaw_delta_rad, bool)
        or not isinstance(command_frame_yaw_delta_rad, (int, float))
        or not math.isfinite(float(command_frame_yaw_delta_rad))
    )) or (command_speed_mps is not None and (
        isinstance(command_speed_mps, bool)
        or not isinstance(command_speed_mps, (int, float))
        or not math.isfinite(float(command_speed_mps))
        or float(command_speed_mps) < 0.0
    )) or (desired_yaw_delta_rad is not None and (
        isinstance(desired_yaw_delta_rad, bool)
        or not isinstance(desired_yaw_delta_rad, (int, float))
        or not math.isfinite(float(desired_yaw_delta_rad))
        or isinstance(yaw_tolerance_rad, bool)
        or not isinstance(yaw_tolerance_rad, (int, float))
        or not math.isfinite(float(yaw_tolerance_rad))
        or not 0.0 < float(yaw_tolerance_rad) <= math.pi
    )):
        raise ContractError("foothold ranking yaw query is invalid")
    count = len(actions)
    eligible = torch.zeros(count, dtype=torch.bool, device=motion_cost.device)
    descriptor_cost = torch.zeros_like(motion_cost)
    yaw_cost = torch.zeros_like(motion_cost)
    turn_xy_cost = torch.zeros_like(motion_cost)
    turn_root_cost = torch.zeros_like(motion_cost)
    if plan.score.shape[0] == 0:
        return FootholdRanking(eligible, descriptor_cost, None)
    command_frame_yaw = (
        desired_yaw_delta_rad
        if command_frame_yaw_delta_rad is None
        else command_frame_yaw_delta_rad
    )

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
        action_xy = action.landing_xy_start_frame_m
        if command_frame_yaw is not None and hard_yaw_gate:
            action_xy = _world_to_local(
                action_xy,
                torch.as_tensor(
                    command_frame_yaw,
                    dtype=action_xy.dtype,
                    device=action_xy.device,
                ),
            )
        xy_error = torch.linalg.vector_norm(
            plan.landing_xy_command_frame_m
            - action_xy[None, :, :],
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
        contact_valid = (
            feet.all(dim=1)
            & (height_error <= float(height_tolerance_m)).all(dim=1)
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
        if command_frame_yaw is None or not hard_yaw_gate:
            public_cost = torch.stack(
                [
                    foothold_action_descriptor_cost(
                        landing_feet=plan.landing_feet[plan_index],
                        landing_xy_heading_m=(
                            plan.landing_xy_command_frame_m[plan_index]
                        ),
                        landing_height_delta_m=(
                            plan.landing_height_delta_m[plan_index]
                        ),
                        landing_frame_offsets=(
                            plan.landing_frame_offsets[plan_index]
                        ),
                        action=action,
                        xy_tolerance_m=xy_tolerance_m,
                        height_tolerance_m=height_tolerance_m,
                        timing_tolerance_frames=timing_tolerance_frames,
                    )
                    for plan_index in range(plan.score.shape[0])
                ]
            )
            public_valid = torch.isfinite(public_cost)
            both_valid = public_valid
            normalized = torch.where(
                public_valid,
                public_cost + plan.score,
                normalized,
            )
        descriptor_cost[action_index] = normalized.min().clamp_min(0.0)
        if arm is FootholdSelectionArm.FIRST_CONTACT:
            eligible[action_index] = bool(first_valid.any().item())
        elif arm in (
            FootholdSelectionArm.TWO_CONTACT,
            FootholdSelectionArm.HYBRID,
        ):
            eligible[action_index] = bool(both_valid.any().item())
        elif arm in (
            FootholdSelectionArm.LAYERED,
            FootholdSelectionArm.LAYERED_HYBRID,
            FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
        ):
            eligible[action_index] = bool(contact_valid.any().item())
        else:
            eligible[action_index] = True
        if desired_yaw_delta_rad is not None:
            yaw_error = math.atan2(
                math.sin(
                    float(action.root_yaw_delta_rad[-1].item())
                    - float(desired_yaw_delta_rad)
                ),
                math.cos(
                    float(action.root_yaw_delta_rad[-1].item())
                    - float(desired_yaw_delta_rad)
                ),
            )
            if hard_yaw_gate:
                eligible[action_index] &= (
                    abs(yaw_error) <= float(yaw_tolerance_rad)
                )
            yaw_cost[action_index] = float(yaw_cost_weight) * (
                yaw_error / float(yaw_tolerance_rad)
            ) ** 2
            turn_xy_cost[action_index] = float(turn_xy_cost_weight) * (
                torch.square(xy_error / float(xy_tolerance_m))
                .sum(dim=1)
                .min()
            )
            command_frame_root = _world_to_local(
                action.root_displacement_m,
                torch.as_tensor(
                    command_frame_yaw,
                    dtype=action.root_displacement_m.dtype,
                    device=action.root_displacement_m.device,
                ),
            )
            turn_root_cost[action_index] = float(
                turn_root_lateral_cost_weight
            ) * torch.square(
                command_frame_root[:, 1] / float(xy_tolerance_m)
            ).sum()
            if command_speed_mps is not None:
                expected_progress = action_timing.to(
                    command_frame_root.dtype
                ) * (float(command_speed_mps) / 50.0)
                turn_root_cost[action_index] += float(
                    turn_root_progress_cost_weight
                ) * torch.square(
                    (
                        command_frame_root[:, 0] - expected_progress
                    ) / float(xy_tolerance_m)
                ).sum()

    if arm in (
        FootholdSelectionArm.FIRST_CONTACT,
        FootholdSelectionArm.HYBRID,
        FootholdSelectionArm.LAYERED_HYBRID,
        FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
    ):
        descriptor_cost.zero_()
    if desired_yaw_delta_rad is not None:
        descriptor_cost += yaw_cost + turn_xy_cost + turn_root_cost
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
    terrain_action_required: bool
    fallback_row_eligibility: torch.Tensor | None = None

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
            or type(self.terrain_action_required) is not bool
        ):
            raise ContractError("foothold policy result is invalid")
        fallback = self.fallback_row_eligibility
        if fallback is not None and (
            not isinstance(fallback, torch.Tensor)
            or fallback.shape != self.row_eligibility.shape
            or fallback.dtype != torch.bool
            or fallback.device != self.row_eligibility.device
            or bool((self.row_eligibility & ~fallback).any().item())
        ):
            raise ContractError("foothold policy fallback is invalid")
        object.__setattr__(
            self, "row_eligibility", self.row_eligibility.detach().clone()
        )
        object.__setattr__(
            self,
            "additional_row_cost",
            self.additional_row_cost.detach().clone(),
        )
        if fallback is not None:
            object.__setattr__(
                self,
                "fallback_row_eligibility",
                fallback.detach().clone(),
            )


@dataclass
class FootholdActionPolicy:
    """Map a query heightmap plan into exact motion-database row gates."""

    accepts_requested_heading = True

    index: FootholdActionIndex
    extension: object
    arm: FootholdSelectionArm
    height_tolerance_m: float = 0.06
    xy_tolerance_m: float = 0.25
    timing_tolerance_frames: int = 20
    edge_margin_m: float = 0.04
    beam_width: int = 50
    fallback_candidate_count: int = 1024
    terrain_activation_height_m: float = 0.05
    command_velocity_query_blend: float = 1.0
    transition_graph: FootholdTransitionGraph | None = None
    graph_maximum_position_error_rad: float = 0.90
    graph_maximum_velocity_error_rad_s: float = 5.0
    graph_fallback_maximum_position_error_rad: float = 1.50
    graph_fallback_maximum_velocity_error_rad_s: float = 8.0
    graph_bypass_heading_error_rad: float = math.pi / 4.0
    turn_yaw_tolerance_rad: float = math.pi / 6.0
    turn_hard_gate_activation_rad: float = math.pi / 4.0
    turn_action_chunk_max_rad: float = math.pi / 2.0
    turn_yaw_cost_weight: float = 100.0
    heading_maintenance_yaw_cost_weight: float = 10.0
    heading_maintenance_min_elevation_m: float = 0.30
    heading_maintenance_elevated_dwell_frames: int = 10
    heading_maintenance_command_cooldown_frames: int = 100
    turn_xy_cost_weight: float = 10.0
    turn_root_lateral_cost_weight: float = 100.0
    turn_root_progress_cost_weight: float = 10.0
    turn_lateral_root_warp_gain: float = 0.606
    small_turn_lateral_root_warp_gain: float = 0.25
    reversal_lateral_root_warp_gain: float = 0.25
    strafe_gate_activation_rad: float = math.pi / 4.0
    strafe_direction_tolerance_rad: float = math.pi / 4.0
    strafe_minimum_progress_m: float = 0.03
    strafe_action_gate_enabled: bool = False
    turn_sequence_candidate_count: int = 1
    turn_sequence_lookahead: bool = False

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
                    self.terrain_activation_height_m,
                )
            )
            or type(self.timing_tolerance_frames) is not int
            or self.timing_tolerance_frames < 0
            or type(self.beam_width) is not int
            or self.beam_width < 1
            or type(self.fallback_candidate_count) is not int
            or self.fallback_candidate_count < 1
            or type(self.turn_sequence_candidate_count) is not int
            or self.turn_sequence_candidate_count < 1
            or type(self.turn_sequence_lookahead) is not bool
            or type(self.heading_maintenance_elevated_dwell_frames) is not int
            or self.heading_maintenance_elevated_dwell_frames < 1
            or type(self.heading_maintenance_command_cooldown_frames) is not int
            or self.heading_maintenance_command_cooldown_frames < 1
            or type(self.strafe_action_gate_enabled) is not bool
            or isinstance(self.command_velocity_query_blend, bool)
            or not isinstance(self.command_velocity_query_blend, (int, float))
            or not math.isfinite(float(self.command_velocity_query_blend))
            or not 0.0 <= float(self.command_velocity_query_blend) <= 1.0
        ):
            raise ContractError("foothold action policy tolerances are invalid")
        graph_arm = self.arm is FootholdSelectionArm.LAYERED_GRAPH_HYBRID
        if graph_arm and not isinstance(
            self.transition_graph, FootholdTransitionGraph
        ):
            raise ContractError("foothold graph arm requires a transition graph")
        if self.transition_graph is not None:
            expected_keys = tuple(
                (action.clip_index, action.start_frame)
                for action in self.index.actions
            )
            if self.transition_graph.action_keys != expected_keys:
                raise ContractError(
                    "foothold transition graph does not match the action index"
                )
        graph_limits = (
            self.graph_maximum_position_error_rad,
            self.graph_maximum_velocity_error_rad_s,
            self.graph_fallback_maximum_position_error_rad,
            self.graph_fallback_maximum_velocity_error_rad_s,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in graph_limits
        ) or (
            self.graph_fallback_maximum_position_error_rad
            < self.graph_maximum_position_error_rad
            or self.graph_fallback_maximum_velocity_error_rad_s
            < self.graph_maximum_velocity_error_rad_s
        ):
            raise ContractError("foothold transition graph limits are invalid")
        if (
            isinstance(self.graph_bypass_heading_error_rad, bool)
            or not isinstance(
                self.graph_bypass_heading_error_rad, (int, float)
            )
            or not math.isfinite(float(self.graph_bypass_heading_error_rad))
            or not 0.0 < float(self.graph_bypass_heading_error_rad) <= math.pi
        ):
            raise ContractError(
                "foothold transition graph heading limit is invalid"
            )
        for value in (
            self.turn_yaw_tolerance_rad,
            self.turn_hard_gate_activation_rad,
            self.turn_action_chunk_max_rad,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= math.pi
            ):
                raise ContractError("foothold turn yaw limits are invalid")
        for value in (
            self.turn_yaw_cost_weight,
            self.heading_maintenance_yaw_cost_weight,
            self.heading_maintenance_min_elevation_m,
            self.turn_xy_cost_weight,
            self.turn_root_lateral_cost_weight,
            self.turn_root_progress_cost_weight,
            self.turn_lateral_root_warp_gain,
            self.small_turn_lateral_root_warp_gain,
            self.reversal_lateral_root_warp_gain,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ContractError("foothold turn cost is invalid")
        if (
            self.turn_lateral_root_warp_gain > 1.0
            or self.small_turn_lateral_root_warp_gain > 1.0
            or self.reversal_lateral_root_warp_gain > 1.0
        ):
            raise ContractError("foothold turn warp gain is invalid")
        strafe_values = (
            self.strafe_gate_activation_rad,
            self.strafe_direction_tolerance_rad,
            self.strafe_minimum_progress_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in strafe_values
        ) or (
            not 0.0 < float(self.strafe_gate_activation_rad) <= math.pi
            or not 0.0 < float(self.strafe_direction_tolerance_rad)
            < math.pi / 2.0
            or float(self.strafe_minimum_progress_m) < 0.0
        ):
            raise ContractError("foothold strafe gate is invalid")
        self._terrain_clip_indices = frozenset(
            action.clip_index for action in self.index.actions
        )
        self._action_indices = {
            (action.clip_index, action.start_frame): action_index
            for action_index, action in enumerate(self.index.actions)
        }
        self._cached_database_id: int | None = None
        self._cached_rows: tuple[FootholdAction | None, ...] | None = None
        self._cached_row_action_indices: torch.Tensor | None = None
        self._cached_terrain_rows: torch.Tensor | None = None
        if self.index.actions:
            self._action_feet = torch.tensor(
                [action.landing_feet for action in self.index.actions],
                dtype=torch.int64,
                device=self.index.actions[0].landing_height_delta_m.device,
            )
            self._action_xy = torch.stack(
                [
                    action.landing_xy_start_frame_m
                    for action in self.index.actions
                ]
            )
            self._action_height = torch.stack(
                [
                    action.landing_height_delta_m
                    for action in self.index.actions
                ]
            )
            self._action_timing = torch.tensor(
                [
                    action.landing_frame_offsets
                    for action in self.index.actions
                ],
                dtype=torch.int64,
                device=self._action_feet.device,
            )
            self._action_terminal_yaw = torch.stack(
                [action.root_yaw_delta_rad[-1] for action in self.index.actions]
            )
            self._action_root_displacement = torch.stack(
                [action.root_displacement_m for action in self.index.actions]
            )
        else:
            self._action_feet = torch.empty((0, 2), dtype=torch.int64)
            self._action_xy = torch.empty((0, 2, 2), dtype=torch.float32)
            self._action_height = torch.empty((0, 2), dtype=torch.float32)
            self._action_timing = torch.empty((0, 2), dtype=torch.int64)
            self._action_terminal_yaw = torch.empty(
                (0,), dtype=torch.float32
            )
            self._action_root_displacement = torch.empty(
                (0, 2, 2), dtype=torch.float32
            )
        self._turn_episode_facing: torch.Tensor | None = None
        self._turn_episode_last_sequence = -1
        self._beam_pair_eligibility: torch.Tensor | None = None
        self._pending_turn_pair: tuple[int, int] | None = None
        self._pending_turn_facing: torch.Tensor | None = None
        self._pending_turn_sequence = -1
        self._support_base_height: torch.Tensor | None = None
        self._support_base_last_sequence = -1
        self._maintenance_elevated_frames = 0
        self._maintenance_last_facing: torch.Tensor | None = None
        self._maintenance_last_heading_change_sequence = 0
        self._maintenance_last_sequence = -1
        self._requested_heading_world_yaw: torch.Tensor | None = None
        self._requested_turn_sequence_active = False
        self._requested_turn_active = False
        self._requested_turn_delta_rad = 0.0
        self._requested_heading_last_sequence = -1

    def _action_turn_target(self, desired_yaw_delta_rad: float) -> float:
        if (
            isinstance(desired_yaw_delta_rad, bool)
            or not isinstance(desired_yaw_delta_rad, (int, float))
            or not math.isfinite(float(desired_yaw_delta_rad))
        ):
            raise ContractError("foothold action turn target is invalid")
        return math.copysign(
            min(
                abs(float(desired_yaw_delta_rad)),
                float(self.turn_action_chunk_max_rad),
            ),
            float(desired_yaw_delta_rad),
        )

    def _turn_episode_query(
        self,
        *,
        sequence: int,
        root_quaternion: torch.Tensor,
        future_facing: torch.Tensor,
    ) -> tuple[float | None, bool]:
        """Condition yaw only during and after an explicit large turn."""

        if (
            type(sequence) is not int
            or sequence < 0
            or not isinstance(root_quaternion, torch.Tensor)
            or tuple(root_quaternion.shape) != (4,)
            or not root_quaternion.dtype.is_floating_point
            or not torch.isfinite(root_quaternion).all()
            or not isinstance(future_facing, torch.Tensor)
            or tuple(future_facing.shape) != (2,)
            or future_facing.dtype != root_quaternion.dtype
            or future_facing.device != root_quaternion.device
            or not torch.isfinite(future_facing).all()
        ):
            raise ContractError("foothold turn episode query is invalid")
        norm = torch.linalg.vector_norm(future_facing)
        if float(norm.item()) <= 1e-6:
            raise ContractError("foothold turn episode facing is zero")
        facing = future_facing / norm
        if sequence < self._turn_episode_last_sequence:
            self._turn_episode_facing = None
        self._turn_episode_last_sequence = sequence
        root_yaw = _yaw_from_wxyz(root_quaternion)
        future_yaw = torch.atan2(facing[1], facing[0])
        desired = torch.atan2(
            torch.sin(future_yaw - root_yaw),
            torch.cos(future_yaw - root_yaw),
        )
        desired_value = float(desired.item())
        hard = abs(desired_value) >= float(
            self.turn_hard_gate_activation_rad
        )
        if hard:
            self._turn_episode_facing = facing.detach().clone()
            return desired_value, True
        active = self._turn_episode_facing
        if active is not None and float(torch.dot(active, facing).item()) >= (
            math.cos(math.radians(15.0))
        ):
            return desired_value, False
        self._turn_episode_facing = None
        return None, False

    @staticmethod
    def _heading_maintenance_query(
        *,
        root_quaternion: torch.Tensor,
        future_facing: torch.Tensor,
    ) -> float:
        if (
            not isinstance(root_quaternion, torch.Tensor)
            or tuple(root_quaternion.shape) != (4,)
            or not root_quaternion.dtype.is_floating_point
            or not torch.isfinite(root_quaternion).all()
            or not isinstance(future_facing, torch.Tensor)
            or tuple(future_facing.shape) != (2,)
            or future_facing.dtype != root_quaternion.dtype
            or future_facing.device != root_quaternion.device
            or not torch.isfinite(future_facing).all()
        ):
            raise ContractError("foothold heading maintenance state is invalid")
        norm = torch.linalg.vector_norm(future_facing)
        if float(norm.item()) <= 1e-6:
            raise ContractError("foothold heading maintenance facing is zero")
        root_yaw = _yaw_from_wxyz(root_quaternion)
        future_yaw = torch.atan2(future_facing[1], future_facing[0])
        return float(
            torch.atan2(
                torch.sin(future_yaw - root_yaw),
                torch.cos(future_yaw - root_yaw),
            ).item()
        )

    def _elevated_heading_maintenance_required(
        self,
        *,
        surface: torch.Tensor,
        base_height: torch.Tensor,
    ) -> bool:
        if (
            not isinstance(surface, torch.Tensor)
            or tuple(surface.shape) != (2,)
            or not surface.dtype.is_floating_point
            or not torch.isfinite(surface).all()
            or not isinstance(base_height, torch.Tensor)
            or base_height.numel() != 1
            or base_height.dtype != surface.dtype
            or base_height.device != surface.device
            or not torch.isfinite(base_height).all()
        ):
            raise ContractError("foothold maintenance surface is invalid")
        return bool(
            (
                surface - base_height
                >= float(self.heading_maintenance_min_elevation_m)
            ).all().item()
        )

    def _observed_support_base_height(
        self, *, sequence: int, surface: torch.Tensor
    ) -> torch.Tensor:
        if (
            type(sequence) is not int
            or sequence < 0
            or not isinstance(surface, torch.Tensor)
            or tuple(surface.shape) != (2,)
            or not surface.dtype.is_floating_point
            or not torch.isfinite(surface).all()
        ):
            raise ContractError("foothold support baseline is invalid")
        value = surface.min().detach()
        if (
            self._support_base_height is None
            or sequence < self._support_base_last_sequence
        ):
            self._support_base_height = value.clone()
        else:
            self._support_base_height = torch.minimum(
                self._support_base_height, value
            )
        self._support_base_last_sequence = sequence
        return self._support_base_height.clone()

    def _heading_maintenance_activated(
        self,
        *,
        sequence: int,
        surface: torch.Tensor,
        base_height: torch.Tensor,
        future_facing: torch.Tensor,
    ) -> bool:
        if (
            type(sequence) is not int
            or sequence < 0
            or not isinstance(future_facing, torch.Tensor)
            or tuple(future_facing.shape) != (2,)
            or future_facing.dtype != surface.dtype
            or future_facing.device != surface.device
            or not torch.isfinite(future_facing).all()
        ):
            raise ContractError("foothold maintenance activation is invalid")
        norm = torch.linalg.vector_norm(future_facing)
        if float(norm.item()) <= 1e-6:
            raise ContractError("foothold maintenance facing is zero")
        facing = future_facing / norm
        if sequence < self._maintenance_last_sequence:
            self._maintenance_elevated_frames = 0
            self._maintenance_last_facing = None
            self._maintenance_last_heading_change_sequence = sequence
        previous = self._maintenance_last_facing
        if previous is None or float(torch.dot(previous, facing).item()) < (
            math.cos(math.radians(1.0))
        ):
            self._maintenance_last_heading_change_sequence = sequence
        self._maintenance_last_facing = facing.detach().clone()
        self._maintenance_last_sequence = sequence
        elevated = self._elevated_heading_maintenance_required(
            surface=surface,
            base_height=base_height,
        )
        self._maintenance_elevated_frames = (
            self._maintenance_elevated_frames + 1 if elevated else 0
        )
        return (
            self._maintenance_elevated_frames
            >= self.heading_maintenance_elevated_dwell_frames
            and sequence - self._maintenance_last_heading_change_sequence
            >= self.heading_maintenance_command_cooldown_frames
        )

    def _requested_turn_sequence_query(
        self,
        *,
        sequence: int,
        requested_heading_world_yaw: torch.Tensor,
    ) -> bool:
        if (
            type(sequence) is not int
            or sequence < 0
            or not isinstance(requested_heading_world_yaw, torch.Tensor)
            or requested_heading_world_yaw.numel() != 1
            or not requested_heading_world_yaw.dtype.is_floating_point
            or not torch.isfinite(requested_heading_world_yaw).all()
        ):
            raise ContractError("foothold requested heading is invalid")
        value = requested_heading_world_yaw.reshape(()).detach()
        previous = self._requested_heading_world_yaw
        if sequence < self._requested_heading_last_sequence:
            previous = None
            self._requested_turn_sequence_active = False
            self._requested_turn_active = False
            self._requested_turn_delta_rad = 0.0
        if previous is not None:
            delta = torch.atan2(
                torch.sin(value - previous),
                torch.cos(value - previous),
            )
            if abs(float(delta.item())) >= math.radians(1.0):
                self._requested_turn_delta_rad = abs(float(delta.item()))
                self._requested_turn_active = abs(
                    float(delta.item())
                ) >= math.radians(30.0)
                self._requested_turn_sequence_active = abs(
                    float(delta.item())
                ) > float(self.turn_action_chunk_max_rad) + math.radians(5.0)
        self._requested_heading_world_yaw = value.clone()
        self._requested_heading_last_sequence = sequence
        return self._requested_turn_sequence_active

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
            device = torch.device(database.device)
            self._cached_row_action_indices = torch.tensor(
                [
                    -1
                    if action is None
                    else self._action_indices[
                        (action.clip_index, action.start_frame)
                    ]
                    for action in self._cached_rows
                ],
                dtype=torch.int64,
                device=device,
            )
            terrain_clips = torch.tensor(
                sorted(self._terrain_clip_indices),
                dtype=database._search_clip_index.dtype,
                device=device,
            )
            self._cached_terrain_rows = torch.isin(
                database._search_clip_index, terrain_clips
            )
        return self._cached_rows

    def _turn_action_eligibility(
        self, desired_yaw_delta_rad: float
    ) -> torch.Tensor:
        desired = torch.as_tensor(
            desired_yaw_delta_rad,
            dtype=self._action_terminal_yaw.dtype,
            device=self._action_terminal_yaw.device,
        )
        error = torch.atan2(
            torch.sin(self._action_terminal_yaw - desired),
            torch.cos(self._action_terminal_yaw - desired),
        )
        return torch.abs(error) <= float(self.turn_yaw_tolerance_rad)

    def _turn_action_cost(
        self,
        desired_yaw_delta_rad: float,
        *,
        weight: float | None = None,
    ) -> torch.Tensor:
        desired = torch.as_tensor(
            desired_yaw_delta_rad,
            dtype=self._action_terminal_yaw.dtype,
            device=self._action_terminal_yaw.device,
        )
        error = torch.atan2(
            torch.sin(self._action_terminal_yaw - desired),
            torch.cos(self._action_terminal_yaw - desired),
        )
        return float(
            self.turn_yaw_cost_weight if weight is None else weight
        ) * torch.square(
            error / float(self.turn_yaw_tolerance_rad)
        )

    def _sequence_candidate_eligibility(
        self, ranking: FootholdRanking
    ) -> torch.Tensor:
        eligible = ranking.eligible
        count = min(
            self.turn_sequence_candidate_count,
            int(eligible.sum().item()),
        )
        output = torch.zeros_like(eligible)
        if count:
            masked = torch.where(
                eligible,
                ranking.additional_cost,
                torch.full_like(ranking.additional_cost, torch.inf),
            )
            output[torch.topk(masked, count, largest=False).indices] = True
        return output

    def _remember_turn_pair(
        self,
        ranking: FootholdRanking,
        *,
        sequence: int,
        future_facing: torch.Tensor,
    ) -> None:
        first = ranking.selected_action
        second = ranking.selected_successor_action
        if first is None or second is None:
            return
        if (
            type(sequence) is not int
            or sequence < 0
            or not isinstance(future_facing, torch.Tensor)
            or tuple(future_facing.shape) != (2,)
            or not future_facing.dtype.is_floating_point
            or not torch.isfinite(future_facing).all()
        ):
            raise ContractError("foothold turn pair state is invalid")
        norm = torch.linalg.vector_norm(future_facing)
        if float(norm.item()) <= 1e-6:
            raise ContractError("foothold turn pair facing is zero")
        self._pending_turn_pair = (first, second)
        self._pending_turn_facing = (future_facing / norm).detach().clone()
        self._pending_turn_sequence = sequence

    def _pending_turn_pair_eligibility(
        self,
        *,
        state: object,
        future_facing: torch.Tensor,
        ranking: FootholdRanking,
    ) -> torch.Tensor | None:
        pair = self._pending_turn_pair
        if pair is None:
            return None
        try:
            sequence = state.sequence
            clip_index = state.clip_index
            frame_index = state.frame_index
        except AttributeError as error:
            raise ContractError("foothold pending turn state is invalid") from error
        if (
            type(sequence) is not int
            or type(clip_index) is not int
            or type(frame_index) is not int
            or not isinstance(future_facing, torch.Tensor)
            or tuple(future_facing.shape) != (2,)
            or future_facing.dtype != ranking.additional_cost.dtype
            or future_facing.device != ranking.additional_cost.device
            or not torch.isfinite(future_facing).all()
        ):
            raise ContractError("foothold pending turn state is invalid")
        if sequence <= self._pending_turn_sequence:
            if sequence < self._pending_turn_sequence:
                self._pending_turn_pair = None
                self._pending_turn_facing = None
            return None
        norm = torch.linalg.vector_norm(future_facing)
        remembered = self._pending_turn_facing
        if (
            float(norm.item()) <= 1e-6
            or remembered is None
            or float(torch.dot(future_facing / norm, remembered).item())
            < math.cos(math.radians(15.0))
        ):
            self._pending_turn_pair = None
            self._pending_turn_facing = None
            return None
        first, second = pair
        action = self.index.actions[first]
        matches_first = (
            clip_index == action.clip_index
            and action.start_frame <= frame_index < action.end_frame
        )
        self._pending_turn_pair = None
        self._pending_turn_facing = None
        if not matches_first or not bool(ranking.eligible[second].item()):
            return None
        output = torch.zeros_like(ranking.eligible)
        output[second] = True
        return output

    def _hard_entry_graph_gate_required(
        self,
        *,
        graph_required: bool,
        hard_yaw_gate: bool,
        heading_maintenance: bool = False,
        sequence_active: bool = False,
    ) -> bool:
        if (
            type(graph_required) is not bool
            or type(hard_yaw_gate) is not bool
            or type(heading_maintenance) is not bool
            or type(sequence_active) is not bool
        ):
            raise ContractError("foothold graph gate state is invalid")
        return graph_required and not sequence_active

    @staticmethod
    def _fallback_hard_yaw_gate_required(
        *,
        hard_yaw_gate: bool,
        sequence_action_eligibility: torch.Tensor | None,
    ) -> bool:
        if type(hard_yaw_gate) is not bool or (
            sequence_action_eligibility is not None
            and (
                not isinstance(sequence_action_eligibility, torch.Tensor)
                or sequence_action_eligibility.dtype != torch.bool
                or sequence_action_eligibility.ndim != 1
            )
        ):
            raise ContractError("foothold fallback yaw gate state is invalid")
        return hard_yaw_gate and sequence_action_eligibility is None

    def _second_foothold_plans(
        self,
        *,
        current_foot_xy_m: torch.Tensor,
        command_xy: torch.Tensor,
        first_plan: FootholdPlan,
    ) -> tuple[FootholdPlan, ...]:
        output = []
        for plan_index in range(first_plan.score.shape[0]):
            terminal_feet = current_foot_xy_m.clone()
            for landing_index in range(2):
                foot = int(
                    first_plan.landing_feet[
                        plan_index, landing_index
                    ].item()
                )
                terminal_feet[foot] = first_plan.landing_xy_world_m[
                    plan_index, landing_index
                ]
            output.append(
                plan_footholds(
                    foot_xy_m=terminal_feet,
                    support_mask=torch.ones(
                        2,
                        dtype=torch.bool,
                        device=terminal_feet.device,
                    ),
                    command_xy=command_xy,
                    sample_surface=self._sample_matcher_surface,
                    reachable_forward_m=(0.20, 0.25, 0.30, 0.35, 0.40),
                    lateral_samples_m=(-0.15, -0.075, 0.0, 0.075, 0.15),
                    edge_margin_m=self.edge_margin_m,
                    beam_width=self.beam_width,
                )
            )
        return tuple(output)

    def _rank_all_actions(
        self,
        plan: FootholdPlan,
        desired_yaw_delta_rad: float | None = None,
        hard_yaw_gate: bool = False,
        command_speed_mps: float | None = None,
        command_frame_yaw_delta_rad: float | None = None,
        sequence_pair_eligibility: torch.Tensor | None = None,
        yaw_cost_weight: float | None = None,
        sequence_lookahead_override: bool | None = None,
        velocity_facing_yaw_delta_rad: float | None = None,
    ) -> FootholdRanking:
        count = len(self.index.actions)
        device = plan.score.device
        eligible = torch.zeros(count, dtype=torch.bool, device=device)
        cost = torch.zeros(count, dtype=torch.float32, device=device)
        sequence_successor = None
        if count == 0 or plan.score.shape[0] == 0:
            return FootholdRanking(eligible, cost, None)
        feet = plan.landing_feet[:, None, :] == self._action_feet[None, :, :]
        height_error = torch.abs(
            plan.landing_height_delta_m[:, None, :]
            - self._action_height[None, :, :]
        )
        action_xy = self._action_xy
        command_frame_yaw = (
            desired_yaw_delta_rad
            if command_frame_yaw_delta_rad is None
            else command_frame_yaw_delta_rad
        )
        sequence_lookahead = (
            bool(sequence_lookahead_override)
            if sequence_lookahead_override is not None
            else (
                self.turn_sequence_lookahead
                and command_speed_mps is not None
                and command_frame_yaw is not None
                and abs(float(command_frame_yaw))
                > float(self.turn_action_chunk_max_rad) + 1e-6
            )
        )
        if command_frame_yaw is not None:
            action_xy = _world_to_local(
                action_xy,
                torch.as_tensor(
                    command_frame_yaw,
                    dtype=action_xy.dtype,
                    device=action_xy.device,
                ),
            )
        xy_error = torch.linalg.vector_norm(
            plan.landing_xy_command_frame_m[:, None, :, :]
            - action_xy[None, :, :, :],
            dim=-1,
        )
        timing_error = torch.abs(
            plan.landing_frame_offsets[:, None, :]
            - self._action_timing[None, :, :]
        )
        first_valid = (
            feet[..., 0]
            & (height_error[..., 0] <= self.height_tolerance_m)
            & (xy_error[..., 0] <= self.xy_tolerance_m)
            & (timing_error[..., 0] <= self.timing_tolerance_frames)
        )
        both_valid = (
            feet.all(dim=-1)
            & (height_error <= self.height_tolerance_m).all(dim=-1)
            & (xy_error <= self.xy_tolerance_m).all(dim=-1)
            & (timing_error <= self.timing_tolerance_frames).all(dim=-1)
        )
        contact_valid = (
            feet.all(dim=-1)
            & (height_error <= self.height_tolerance_m).all(dim=-1)
        )
        normalized = (
            torch.square(height_error / self.height_tolerance_m).sum(dim=-1)
            + torch.square(xy_error / self.xy_tolerance_m).sum(dim=-1)
            + torch.square(
                timing_error.to(torch.float32)
                / float(max(1, self.timing_tolerance_frames))
            ).sum(dim=-1)
            + (~feet).to(torch.float32).sum(dim=-1) * 100.0
            + plan.score[:, None]
        )
        cost = normalized.min(dim=0).values.clamp_min(0.0)
        if self.arm is FootholdSelectionArm.FIRST_CONTACT:
            eligible = first_valid.any(dim=0)
        elif self.arm in (
            FootholdSelectionArm.TWO_CONTACT,
            FootholdSelectionArm.HYBRID,
        ):
            eligible = both_valid.any(dim=0)
        elif self.arm in (
            FootholdSelectionArm.LAYERED,
            FootholdSelectionArm.LAYERED_HYBRID,
            FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
        ):
            eligible = contact_valid.any(dim=0)
        else:
            eligible.fill_(True)
        if velocity_facing_yaw_delta_rad is not None:
            delta = float(velocity_facing_yaw_delta_rad)
            if not math.isfinite(delta):
                raise ContractError("foothold strafe query is invalid")
            if abs(delta) >= float(self.strafe_gate_activation_rad):
                command_frame_root = _world_to_local(
                    self._action_root_displacement,
                    torch.as_tensor(
                        delta,
                        dtype=self._action_root_displacement.dtype,
                        device=self._action_root_displacement.device,
                    ),
                )
                norm = torch.linalg.vector_norm(
                    command_frame_root[:, -1], dim=-1
                )
                direction_cosine = (
                    command_frame_root[:, -1, 0] / norm.clamp_min(1e-6)
                )
                eligible &= (
                    (norm >= float(self.strafe_minimum_progress_m))
                    & (
                        direction_cosine
                        >= math.cos(float(self.strafe_direction_tolerance_rad))
                    )
                )
        if (
            desired_yaw_delta_rad is not None
            and hard_yaw_gate
            and not sequence_lookahead
        ):
            eligible &= self._turn_action_eligibility(
                desired_yaw_delta_rad
            )
        if self.arm in (
            FootholdSelectionArm.FIRST_CONTACT,
            FootholdSelectionArm.HYBRID,
            FootholdSelectionArm.LAYERED_HYBRID,
            FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
        ):
            cost.zero_()
        if desired_yaw_delta_rad is not None:
            if not sequence_lookahead:
                cost += self._turn_action_cost(
                    desired_yaw_delta_rad,
                    weight=yaw_cost_weight,
                )
            if hard_yaw_gate:
                cost += float(self.turn_xy_cost_weight) * torch.square(
                    xy_error / float(self.xy_tolerance_m)
                ).sum(dim=-1).min(dim=0).values
                if sequence_lookahead:
                    if (
                        self._beam_pair_eligibility is None
                        and self.transition_graph is not None
                    ):
                        self._beam_pair_eligibility = (
                            self.transition_graph.pair_eligibility(
                                maximum_position_error_rad=(
                                    self.graph_fallback_maximum_position_error_rad
                                ),
                                maximum_velocity_error_rad_s=(
                                    self.graph_fallback_maximum_velocity_error_rad_s
                                ),
                            )
                        )
                    pair_eligibility = self._beam_pair_eligibility
                    if sequence_pair_eligibility is not None:
                        pair_eligibility = (
                            sequence_pair_eligibility
                            if pair_eligibility is None
                            else pair_eligibility & sequence_pair_eligibility
                        )
                    lookahead, sequence_successor = _two_action_plan_tensors(
                        terminal_yaw=self._action_terminal_yaw,
                        terminal_root_displacement=(
                            self._action_root_displacement[:, -1]
                        ),
                        terminal_frame_offset=self._action_timing[:, -1],
                        desired_yaw_delta_rad=float(command_frame_yaw),
                        command_speed_mps=float(command_speed_mps),
                        yaw_tolerance_rad=self.turn_yaw_tolerance_rad,
                        xy_tolerance_m=self.xy_tolerance_m,
                        yaw_cost_weight=self.turn_yaw_cost_weight,
                        root_lateral_cost_weight=(
                            self.turn_root_lateral_cost_weight
                        ),
                        root_progress_cost_weight=(
                            self.turn_root_progress_cost_weight
                        ),
                        pair_eligibility=pair_eligibility,
                    )
                    lookahead_finite = torch.isfinite(lookahead)
                    eligible &= lookahead_finite
                    cost += torch.where(
                        lookahead_finite,
                        lookahead,
                        torch.zeros_like(lookahead),
                    )
                else:
                    command_frame_root = _world_to_local(
                        self._action_root_displacement,
                        torch.as_tensor(
                            command_frame_yaw,
                            dtype=self._action_root_displacement.dtype,
                            device=self._action_root_displacement.device,
                        ),
                    )
                    expected_progress = self._action_timing.to(
                        command_frame_root.dtype
                    ) * (float(command_speed_mps) / 50.0)
                    cost += (
                        float(self.turn_root_lateral_cost_weight)
                        * torch.square(
                            command_frame_root[..., 1]
                            / float(self.xy_tolerance_m)
                        ).sum(dim=-1)
                        + float(self.turn_root_progress_cost_weight)
                        * torch.square(
                            (
                                command_frame_root[..., 0]
                                - expected_progress
                            ) / float(self.xy_tolerance_m)
                        ).sum(dim=-1)
                    )
        selected = None
        if bool(eligible.any().item()):
            selected = int(
                torch.argmin(
                    torch.where(
                        eligible, cost, torch.full_like(cost, torch.inf)
                    )
                ).item()
            )
        selected_successor = None
        if selected is not None and sequence_successor is not None:
            value = int(sequence_successor[selected].item())
            if value >= 0:
                selected_successor = value
        return FootholdRanking(
            eligible,
            cost,
            selected,
            selected_successor_action=selected_successor,
        )

    def prepare(
        self,
        state: object,
        shaped: object,
        database: object,
        *,
        requested_heading_world_yaw: torch.Tensor | None = None,
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
        desired_yaw_delta_rad = None
        velocity_facing_yaw_delta_rad = None
        hard_yaw_gate = False
        heading_maintenance = False
        requested_sequence = False
        if self.arm in (
            FootholdSelectionArm.LAYERED_HYBRID,
            FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
        ):
            try:
                root_quaternion = state.root_quaternion
                future_facing = shaped.trajectory.facing_world_xy[-1]
            except (AttributeError, IndexError, TypeError) as error:
                raise ContractError(
                    "foothold turn heading state is invalid"
                ) from error
            desired_yaw_delta_rad, hard_yaw_gate = (
                self._turn_episode_query(
                    sequence=int(state.sequence),
                    root_quaternion=root_quaternion,
                    future_facing=future_facing,
                )
            )
            if requested_heading_world_yaw is not None:
                requested_sequence = self._requested_turn_sequence_query(
                    sequence=int(state.sequence),
                    requested_heading_world_yaw=(
                        requested_heading_world_yaw
                    ),
                )
                command_speed = torch.linalg.vector_norm(command)
                if float(command_speed.item()) > 1e-6:
                    command_yaw = torch.atan2(command[1], command[0])
                    velocity_facing_yaw_delta_rad = float(
                        torch.atan2(
                            torch.sin(
                                command_yaw
                                - requested_heading_world_yaw.reshape(())
                            ),
                            torch.cos(
                                command_yaw
                                - requested_heading_world_yaw.reshape(())
                            ),
                        ).item()
                    )
            if self._requested_turn_active:
                if desired_yaw_delta_rad is None:
                    desired_yaw_delta_rad = self._heading_maintenance_query(
                        root_quaternion=root_quaternion,
                        future_facing=future_facing,
                    )
                hard_yaw_gate = abs(desired_yaw_delta_rad) >= math.radians(
                    10.0
                )
                if not hard_yaw_gate:
                    self._requested_turn_active = False
            origin_surface = self._observed_support_base_height(
                sequence=int(state.sequence),
                surface=surface,
            )
            self._heading_maintenance_activated(
                sequence=int(state.sequence),
                surface=surface,
                base_height=origin_surface,
                future_facing=future_facing,
            )
            if desired_yaw_delta_rad is None:
                desired_yaw_delta_rad = self._heading_maintenance_query(
                    root_quaternion=root_quaternion,
                    future_facing=future_facing,
                )
                heading_maintenance = True
        if not self.strafe_action_gate_enabled:
            velocity_facing_yaw_delta_rad = None
        sequence_active = (
            self.turn_sequence_lookahead
            and requested_sequence
            and desired_yaw_delta_rad is not None
            and abs(desired_yaw_delta_rad)
            > float(self.turn_action_chunk_max_rad) + 1e-6
        )
        action_yaw_delta_rad = (
            None
            if desired_yaw_delta_rad is None
            else (
                self._action_turn_target(desired_yaw_delta_rad)
                if sequence_active
                else desired_yaw_delta_rad
            )
        )
        sequence_pair_eligibility = None
        if (
            sequence_active
            and hard_yaw_gate
        ):
            second_plans = self._second_foothold_plans(
                current_foot_xy_m=feet[:, :2],
                command_xy=command,
                first_plan=plan,
            )
            sequence_pair_eligibility = two_action_terrain_eligibility(
                first_plan=plan,
                second_plans=second_plans,
                actions=self.index.actions,
                height_tolerance_m=self.height_tolerance_m,
            )
        ranking = self._rank_all_actions(
            plan,
            action_yaw_delta_rad,
            hard_yaw_gate=hard_yaw_gate,
            command_speed_mps=float(
                torch.linalg.vector_norm(command).item()
            ),
            command_frame_yaw_delta_rad=desired_yaw_delta_rad,
            sequence_pair_eligibility=sequence_pair_eligibility,
            yaw_cost_weight=(
                self.heading_maintenance_yaw_cost_weight
                if heading_maintenance
                else None
            ),
            sequence_lookahead_override=(
                sequence_active
            ),
            velocity_facing_yaw_delta_rad=(
                velocity_facing_yaw_delta_rad
            ),
        )
        sequence_action_eligibility = None
        if self.turn_sequence_lookahead and requested_sequence:
            sequence_action_eligibility = (
                self._pending_turn_pair_eligibility(
                    state=state,
                    future_facing=shaped.trajectory.facing_world_xy[-1],
                    ranking=ranking,
                )
            )
        if (
            sequence_action_eligibility is None
            and
            sequence_active
            and hard_yaw_gate
        ):
            sequence_action_eligibility = (
                self._sequence_candidate_eligibility(ranking)
            )
            self._remember_turn_pair(
                ranking,
                sequence=int(state.sequence),
                future_facing=shaped.trajectory.facing_world_xy[-1],
            )
        rows = self._rows(database)
        row_count = len(rows)
        assert self._cached_row_action_indices is not None
        assert self._cached_terrain_rows is not None
        action_indices = self._cached_row_action_indices
        action_rows = action_indices >= 0
        terrain_required = bool(
            (torch.abs(surface[0] - surface[1])
             >= self.terrain_activation_height_m).item()
            or (
                plan.landing_height_delta_m.numel() > 0
                and bool(
                    (torch.abs(plan.landing_height_delta_m)
                     >= self.terrain_activation_height_m).any().item()
                )
            )
        )
        eligible = (~self._cached_terrain_rows).clone()
        graph_primary = None
        graph_fallback = None
        graph_required = False
        if self.arm is FootholdSelectionArm.LAYERED_GRAPH_HYBRID:
            try:
                future_facing = shaped.trajectory.facing_world_xy[-1]
                root_quaternion = state.root_quaternion
            except (AttributeError, IndexError, TypeError) as error:
                raise ContractError(
                    "foothold transition graph heading state is invalid"
                ) from error
            graph_required = transition_graph_required(
                root_quaternion_world_wxyz=root_quaternion,
                future_facing_world_xy=future_facing,
                maximum_heading_error_rad=(
                    self.graph_bypass_heading_error_rad
                ),
            )
            graph_required = self._hard_entry_graph_gate_required(
                graph_required=graph_required,
                hard_yaw_gate=hard_yaw_gate,
                heading_maintenance=heading_maintenance,
                sequence_active=sequence_active,
            )
        if (
            terrain_required
            and self.arm is FootholdSelectionArm.LAYERED_GRAPH_HYBRID
            and graph_required
        ):
            assert self.transition_graph is not None
            graph_primary = self.transition_graph.entry_eligibility(
                current_joint_position=state.joint_position,
                current_joint_velocity=state.joint_velocity,
                maximum_position_error_rad=(
                    self.graph_maximum_position_error_rad
                ),
                maximum_velocity_error_rad_s=(
                    self.graph_maximum_velocity_error_rad_s
                ),
            )
            graph_fallback = self.transition_graph.entry_eligibility(
                current_joint_position=state.joint_position,
                current_joint_velocity=state.joint_velocity,
                maximum_position_error_rad=(
                    self.graph_fallback_maximum_position_error_rad
                ),
                maximum_velocity_error_rad_s=(
                    self.graph_fallback_maximum_velocity_error_rad_s
                ),
            )
        if terrain_required:
            eligible.zero_()
            action_eligible = ranking.eligible
            if sequence_action_eligibility is not None:
                action_eligible = (
                    action_eligible & sequence_action_eligibility
                )
            if graph_primary is not None:
                action_eligible = action_eligible & graph_primary
            eligible[action_rows] = action_eligible[action_indices[action_rows]]
        additional = torch.zeros(row_count, dtype=torch.float32, device=device)
        if terrain_required:
            additional[action_rows] = ranking.additional_cost[
                action_indices[action_rows]
            ]
        fallback = None
        if self.arm in (
            FootholdSelectionArm.LAYERED,
            FootholdSelectionArm.LAYERED_HYBRID,
            FootholdSelectionArm.LAYERED_GRAPH_HYBRID,
        ):
            if terrain_required:
                if graph_fallback is None:
                    fallback = action_rows.clone()
                else:
                    fallback = torch.zeros_like(action_rows)
                    fallback[action_rows] = graph_fallback[
                        action_indices[action_rows]
                    ]
                if self._fallback_hard_yaw_gate_required(
                    hard_yaw_gate=hard_yaw_gate,
                    sequence_action_eligibility=(
                        sequence_action_eligibility
                    ),
                ):
                    turn_eligible = self._turn_action_eligibility(
                        action_yaw_delta_rad
                    )
                    fallback[action_rows] &= turn_eligible[
                        action_indices[action_rows]
                    ]
                if sequence_action_eligibility is not None:
                    fallback[action_rows] &= sequence_action_eligibility[
                        action_indices[action_rows]
                    ]
            else:
                fallback = (~self._cached_terrain_rows).clone()
        return FootholdPolicyResult(
            row_eligibility=eligible,
            additional_row_cost=additional,
            plan=plan,
            candidate_count=(
                int(
                    (
                        ranking.eligible
                        if graph_primary is None
                        else ranking.eligible & graph_primary
                    ).sum().item()
                )
                if terrain_required
                else 0
            ),
            terrain_action_required=terrain_required,
            fallback_row_eligibility=fallback,
        )
