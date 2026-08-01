"""Immutable one-landing actions for the offline terrain contact oracle."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch

from .joints import ContractError
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M


def _owned_tensor(
    value: torch.Tensor,
    shape: tuple[int, ...],
    label: str,
    *,
    boolean: bool = False,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or (value.dtype != torch.bool if boolean else not value.dtype.is_floating_point)
        or (not boolean and not torch.isfinite(value).all())
    ):
        raise ContractError(f"contact oracle action {label} is invalid")
    return value.detach().clone()


@dataclass(frozen=True)
class ContactPhaseAction:
    clip_index: int
    start_frame: int
    end_frame: int
    swing_foot: int
    entry_support: torch.Tensor
    exit_support: torch.Tensor
    support_mask: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position_local: torch.Tensor
    root_yaw_local: torch.Tensor
    foot_position_local: torch.Tensor
    foot_surface_delta_m: torch.Tensor
    minimum_swing_clearance_m: float

    def __post_init__(self) -> None:
        if (
            type(self.clip_index) is not int
            or self.clip_index < 0
            or type(self.start_frame) is not int
            or self.start_frame < 0
            or type(self.end_frame) is not int
            or self.end_frame <= self.start_frame
            or self.swing_foot not in (0, 1)
        ):
            raise ContractError("contact oracle action bounds are invalid")
        frames = self.end_frame - self.start_frame
        fields = (
            ("entry_support", (2,), True),
            ("exit_support", (2,), True),
            ("support_mask", (frames, 2), True),
            ("joint_position", (frames, 29), False),
            ("joint_velocity", (frames, 29), False),
            ("root_position_local", (frames, 3), False),
            ("root_yaw_local", (frames,), False),
            ("foot_position_local", (frames, 2, 3), False),
            ("foot_surface_delta_m", (frames, 2), False),
        )
        owned = []
        device = None
        for name, shape, boolean in fields:
            value = _owned_tensor(
                getattr(self, name), shape, name.replace("_", " "), boolean=boolean
            )
            if device is not None and value.device != device:
                raise ContractError("contact oracle action devices do not match")
            device = value.device
            owned.append((name, value))
        clearance = self.minimum_swing_clearance_m
        if (
            isinstance(clearance, bool)
            or not isinstance(clearance, (int, float))
            or not math.isfinite(float(clearance))
        ):
            raise ContractError("contact oracle swing clearance is invalid")
        for name, value in owned:
            object.__setattr__(self, name, value)
        object.__setattr__(self, "minimum_swing_clearance_m", float(clearance))

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def source_key(self) -> tuple[int, int, int]:
        return self.clip_index, self.start_frame, self.end_frame


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _world_to_local_xy(values: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * values[..., 0] + sine * values[..., 1],
            -sine * values[..., 0] + cosine * values[..., 1],
        ),
        dim=-1,
    )


def action_from_profiles(
    *,
    clip_index: int,
    start_frame: int,
    landing_frame: int,
    support_mask: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    root_position_world: torch.Tensor,
    root_orientation_world_wxyz: torch.Tensor,
    foot_position_world: torch.Tensor,
    foot_surface_height_m: torch.Tensor,
) -> ContactPhaseAction:
    """Extract one source phase including its terminal landing frame."""

    if (
        type(clip_index) is not int
        or clip_index < 0
        or type(start_frame) is not int
        or type(landing_frame) is not int
        or not 0 <= start_frame < landing_frame
        or not isinstance(support_mask, torch.Tensor)
        or support_mask.dtype != torch.bool
        or support_mask.ndim != 2
        or support_mask.shape[1] != 2
        or landing_frame >= support_mask.shape[0]
    ):
        raise ContractError("contact oracle source bounds are invalid")
    total_frames = support_mask.shape[0]
    float_profiles = (
        (joint_position, (total_frames, 29)),
        (joint_velocity, (total_frames, 29)),
        (root_position_world, (total_frames, 3)),
        (root_orientation_world_wxyz, (total_frames, 4)),
        (foot_position_world, (total_frames, 2, 3)),
        (foot_surface_height_m, (total_frames, 2)),
    )
    device = support_mask.device
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not value.dtype.is_floating_point
        or value.device != device
        or not torch.isfinite(value).all()
        for value, shape in float_profiles
    ):
        raise ContractError("contact oracle source profiles are invalid")
    onset = (~support_mask[landing_frame - 1]) & support_mask[landing_frame]
    landing_feet = torch.nonzero(onset, as_tuple=False).flatten()
    if landing_feet.numel() != 1:
        raise ContractError("contact oracle terminal contact is ambiguous")
    swing_foot = int(landing_feet[0].item())
    start_root = root_position_world[start_frame]
    start_yaw = _yaw_from_wxyz(root_orientation_world_wxyz[start_frame])
    selection = slice(start_frame, landing_frame + 1)
    root = root_position_world[selection] - start_root
    root_xy = _world_to_local_xy(root[:, :2], start_yaw)
    root_local = torch.cat((root_xy, root[:, 2:]), dim=1)
    foot = foot_position_world[selection] - start_root
    foot_xy = _world_to_local_xy(foot[..., :2], start_yaw)
    foot_local = torch.cat((foot_xy, foot[..., 2:]), dim=2)
    yaw = _yaw_from_wxyz(root_orientation_world_wxyz[selection]) - start_yaw
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    surface = (
        foot_surface_height_m[selection]
        - foot_surface_height_m[start_frame][None, :]
    )
    source_clearance = (
        foot_position_world[..., 2]
        - foot_surface_height_m
        - float(ANKLE_ORIGIN_SOLE_M)
    )
    swing_clearance = source_clearance[selection, swing_foot]
    return ContactPhaseAction(
        clip_index=clip_index,
        start_frame=start_frame,
        end_frame=landing_frame + 1,
        swing_foot=swing_foot,
        entry_support=support_mask[start_frame],
        exit_support=support_mask[landing_frame],
        support_mask=support_mask[selection],
        joint_position=joint_position[selection],
        joint_velocity=joint_velocity[selection],
        root_position_local=root_local,
        root_yaw_local=yaw,
        foot_position_local=foot_local,
        foot_surface_delta_m=surface,
        minimum_swing_clearance_m=float(swing_clearance.min().item()),
    )


@dataclass(frozen=True)
class ContactPhaseInventory:
    retained_count: int
    rejected_by_reason: Mapping[str, int]

    def __post_init__(self) -> None:
        rejected = dict(self.rejected_by_reason)
        if (
            type(self.retained_count) is not int
            or self.retained_count < 0
            or any(
                not isinstance(reason, str)
                or not reason
                or type(count) is not int
                or count < 1
                for reason, count in rejected.items()
            )
        ):
            raise ContractError("contact oracle inventory is invalid")
        object.__setattr__(
            self, "rejected_by_reason", MappingProxyType(rejected)
        )


@dataclass(frozen=True)
class ContactPhaseActionIndex:
    actions: tuple[ContactPhaseAction, ...]
    inventory: ContactPhaseInventory
    exact_successor_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.actions, tuple)
            or any(not isinstance(action, ContactPhaseAction) for action in self.actions)
            or tuple(sorted(action.source_key for action in self.actions))
            != tuple(action.source_key for action in self.actions)
            or not isinstance(self.inventory, ContactPhaseInventory)
            or self.inventory.retained_count != len(self.actions)
            or not isinstance(self.exact_successor_indices, tuple)
            or len(self.exact_successor_indices) != len(self.actions)
            or any(
                value is not None
                and (type(value) is not int or not 0 <= value < len(self.actions))
                for value in self.exact_successor_indices
            )
        ):
            raise ContractError("contact oracle action index is invalid")

    @classmethod
    def from_dataset(
        cls, dataset: object, segment_index: object, foot_kinematics: object
    ) -> "ContactPhaseActionIndex":
        try:
            clips = dataset.folder.clips
            layout = dataset.folder.layout
            grids = dataset.clip_grids
            alignments = dataset.clip_alignments
            device = torch.device(dataset.device)
            segments = tuple(segment_index.segments)
            support_for_clip = segment_index.support_mask
            root_index = int(layout.root_body_index)
        except (AttributeError, TypeError, ValueError) as error:
            raise ContractError("contact oracle dataset is invalid") from error
        if (
            len(clips) != len(grids)
            or len(clips) != len(alignments)
            or not callable(support_for_clip)
            or not callable(getattr(foot_kinematics, "foot_positions", None))
        ):
            raise ContractError("contact oracle dataset inventory is invalid")

        rejected: Counter[str] = Counter()
        actions: list[ContactPhaseAction] = []
        profiles: dict[int, tuple[torch.Tensor, ...] | None] = {}
        for segment in sorted(segments):
            clip_index = int(segment.clip_index)
            if not 0 <= clip_index < len(clips):
                raise ContractError("contact oracle segment clip is invalid")
            clip = clips[clip_index]
            frame_count = int(np.asarray(clip.joint_position).shape[0])
            landing_frame = int(segment.end_frame)
            if landing_frame >= frame_count:
                rejected["no-next-opposite-landing"] += 1
                continue
            if clip_index not in profiles:
                try:
                    joint_position = torch.tensor(
                        clip.joint_position, dtype=torch.float32, device=device
                    )
                    joint_velocity = torch.tensor(
                        clip.joint_velocity, dtype=torch.float32, device=device
                    )
                    root_position = torch.tensor(
                        clip.body_position_world[:, root_index],
                        dtype=torch.float32,
                        device=device,
                    )
                    root_quaternion = torch.tensor(
                        clip.body_quaternion_world_wxyz[:, root_index],
                        dtype=torch.float32,
                        device=device,
                    )
                    feet_np = foot_kinematics.foot_positions(
                        joint_position, root_position, root_quaternion
                    )
                    feet = torch.tensor(
                        feet_np, dtype=torch.float32, device=device
                    )
                    support = support_for_clip(clip_index).to(device=device)
                    grid = grids[clip_index]
                    alignment = alignments[clip_index]
                    if grid is None or alignment is None:
                        raise ValueError("missing terrain")
                    surface = grid.sample_xy(
                        alignment.matcher_to_scene_xy(feet[..., :2])
                    )
                    profiles[clip_index] = (
                        joint_position,
                        joint_velocity,
                        root_position,
                        root_quaternion,
                        feet,
                        support,
                        surface,
                    )
                except Exception:
                    profiles[clip_index] = None
            clip_profiles = profiles[clip_index]
            if clip_profiles is None:
                rejected["invalid-fk"] += 1
                continue
            (
                joint_position,
                joint_velocity,
                root_position,
                root_quaternion,
                feet,
                support,
                surface,
            ) = clip_profiles
            start = int(segment.start_frame)
            entering = int(segment.entering_foot)
            if (
                start <= 0
                or start >= support.shape[0]
                or bool(support[start - 1, entering].item())
                or not bool(support[start, entering].item())
            ):
                rejected["inconsistent-contact-order"] += 1
                continue
            onset = (~support[landing_frame - 1]) & support[landing_frame]
            landing = torch.nonzero(onset, as_tuple=False).flatten()
            if landing.numel() != 1 or not bool(support[landing_frame].any().item()):
                rejected["unstable-terminal-support"] += 1
                continue
            if int(landing[0].item()) == entering:
                rejected["inconsistent-contact-order"] += 1
                continue
            try:
                actions.append(
                    action_from_profiles(
                        clip_index=clip_index,
                        start_frame=start,
                        landing_frame=landing_frame,
                        support_mask=support,
                        joint_position=joint_position,
                        joint_velocity=joint_velocity,
                        root_position_world=root_position,
                        root_orientation_world_wxyz=root_quaternion,
                        foot_position_world=feet,
                        foot_surface_height_m=surface,
                    )
                )
            except ContractError:
                rejected["inconsistent-contact-order"] += 1

        ordered = tuple(sorted(actions, key=lambda action: action.source_key))
        by_entry = {
            (action.clip_index, action.start_frame): index
            for index, action in enumerate(ordered)
        }
        successors = tuple(
            by_entry.get((action.clip_index, action.end_frame - 1))
            for action in ordered
        )
        return cls(
            actions=ordered,
            inventory=ContactPhaseInventory(
                retained_count=len(ordered),
                rejected_by_reason=rejected,
            ),
            exact_successor_indices=successors,
        )

    def action(self, index: int) -> ContactPhaseAction:
        if type(index) is not int or not 0 <= index < len(self.actions):
            raise ContractError("contact oracle action index is outside inventory")
        return self.actions[index]

    def exact_successor(self, index: int) -> ContactPhaseAction | None:
        self.action(index)
        successor = self.exact_successor_indices[index]
        return None if successor is None else self.actions[successor]
