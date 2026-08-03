"""Stateful terrain-aware foot locking for emitted kinematic references."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import torch

from .joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M


_LEG_JOINT_INDICES = (
    torch.tensor(
        tuple(
            target
            for target, source in enumerate(
                PINNED_TARGET_TO_SOURCE_PERMUTATION
            )
            if 0 <= source < 6
        ),
        dtype=torch.long,
    ),
    torch.tensor(
        tuple(
            target
            for target, source in enumerate(
                PINNED_TARGET_TO_SOURCE_PERMUTATION
            )
            if 6 <= source < 12
        ),
        dtype=torch.long,
    ),
)
_TOUCHDOWN_PROJECTION_LEAD_FRAMES = 3


class TerrainFootLockFilter:
    """Pin source-supported feet while preserving the matcher's root path."""

    def __init__(
        self,
        *,
        clip_paths: Sequence[str],
        support_masks: Sequence[torch.Tensor],
        source_foot_positions: Sequence[torch.Tensor] | None = None,
        source_root_yaws: Sequence[torch.Tensor] | None = None,
        foot_kinematics: object,
        sole_kinematics: object | None = None,
        sample_surface: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dt_s: float = 0.02,
        unlock_radius_m: float = 0.25,
        correction_halflife_s: float = 0.04,
        root_height_correction_halflife_s: float | None = None,
        terrain_base_m: float | None = None,
        maximum_joint_correction_rad: float = 0.35,
        maximum_output_joint_speed_rad_s: float = 12.0,
        swing_clearance_margin_m: float | None = None,
        swing_plan_sigma_frames: float | None = None,
        touchdown_projection_max_shift_m: float | None = None,
        anticipatory_touchdown_projection: bool = False,
    ) -> None:
        paths = tuple(clip_paths)
        masks = tuple(support_masks)
        source_feet = (
            None
            if source_foot_positions is None
            else tuple(source_foot_positions)
        )
        source_yaws = (
            None if source_root_yaws is None else tuple(source_root_yaws)
        )
        if (
            not paths
            or len(paths) != len(masks)
            or len(set(paths)) != len(paths)
            or any(not isinstance(path, str) or not path for path in paths)
            or any(
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.ndim != 2
                or mask.shape[1:] != (2,)
                for mask in masks
            )
            or not callable(sample_surface)
            or not isinstance(device, torch.device)
            or isinstance(dt_s, bool)
            or not isinstance(dt_s, (int, float))
            or not np.isfinite(float(dt_s))
            or float(dt_s) <= 0.0
            or not math.isfinite(float(unlock_radius_m))
            or float(unlock_radius_m) <= 0.0
            or not math.isfinite(float(correction_halflife_s))
            or float(correction_halflife_s) <= 0.0
            or (
                root_height_correction_halflife_s is not None
                and (
                    not math.isfinite(
                        float(root_height_correction_halflife_s)
                    )
                    or float(root_height_correction_halflife_s) <= 0.0
                )
            )
            or (
                terrain_base_m is not None
                and not math.isfinite(float(terrain_base_m))
            )
            or not math.isfinite(float(maximum_joint_correction_rad))
            or float(maximum_joint_correction_rad) <= 0.0
            or not math.isfinite(float(maximum_output_joint_speed_rad_s))
            or float(maximum_output_joint_speed_rad_s) <= 0.0
            or (
                swing_clearance_margin_m is not None
                and (
                    isinstance(swing_clearance_margin_m, bool)
                    or not isinstance(swing_clearance_margin_m, (int, float))
                    or not math.isfinite(float(swing_clearance_margin_m))
                    or not 0.0 <= float(swing_clearance_margin_m) <= 0.20
                )
            )
            or (
                swing_plan_sigma_frames is not None
                and (
                    source_feet is None
                    or source_yaws is None
                    or len(source_feet) != len(paths)
                    or len(source_yaws) != len(paths)
                    or isinstance(swing_plan_sigma_frames, bool)
                    or not isinstance(swing_plan_sigma_frames, (int, float))
                    or not math.isfinite(float(swing_plan_sigma_frames))
                    or not 0.25 <= float(swing_plan_sigma_frames) <= 20.0
                    or swing_clearance_margin_m is None
                )
            )
            or ((source_feet is None) != (source_yaws is None))
            or (
                source_feet is not None
                and (
                    len(source_feet) != len(paths)
                    or len(source_yaws) != len(paths)
                )
            )
            or not callable(getattr(foot_kinematics, "foot_positions", None))
            or not callable(
                getattr(foot_kinematics, "solve_leg_positions", None)
            )
            or (
                touchdown_projection_max_shift_m is not None
                and (
                    sole_kinematics is None
                    or not callable(
                        getattr(sole_kinematics, "sole_points", None)
                    )
                    or isinstance(touchdown_projection_max_shift_m, bool)
                    or not isinstance(
                        touchdown_projection_max_shift_m, (int, float)
                    )
                    or not math.isfinite(
                        float(touchdown_projection_max_shift_m)
                    )
                    or not 0.01
                    <= float(touchdown_projection_max_shift_m)
                    <= 0.10
                )
            )
            or type(anticipatory_touchdown_projection) is not bool
            or (
                anticipatory_touchdown_projection
                and touchdown_projection_max_shift_m is None
            )
        ):
            raise ValueError("terrain foot lock configuration is invalid")
        self._path_to_index = {path: index for index, path in enumerate(paths)}
        self._support_masks = tuple(
            mask.detach().to(device=device).clone() for mask in masks
        )
        self._source_foot_positions = None if source_feet is None else tuple(
            value.detach().to(device=device).clone() for value in source_feet
        )
        self._source_root_yaws = (
            None
            if source_yaws is None
            else tuple(
                value.detach().to(device=device).clone()
                for value in source_yaws
            )
        )
        if self._source_foot_positions is not None:
            for mask, feet, yaw in zip(
                self._support_masks,
                self._source_foot_positions,
                self._source_root_yaws,
            ):
                if (
                    tuple(feet.shape) != (mask.shape[0], 2, 3)
                    or tuple(yaw.shape) != (mask.shape[0],)
                    or not feet.dtype.is_floating_point
                    or not yaw.dtype.is_floating_point
                    or not torch.isfinite(feet).all()
                    or not torch.isfinite(yaw).all()
                ):
                    raise ValueError("terrain foot lock source paths are invalid")
        self._foot_kinematics = foot_kinematics
        self._sole_kinematics = sole_kinematics
        self._sample_surface = sample_surface
        self._device = device
        self._dt_s = float(dt_s)
        self._unlock_radius_m = float(unlock_radius_m)
        self._correction_alpha = 1.0 - math.exp(
            -math.log(2.0) * self._dt_s / float(correction_halflife_s)
        )
        self._root_height_correction_alpha = (
            None
            if root_height_correction_halflife_s is None
            else 1.0
            - math.exp(
                -math.log(2.0)
                * self._dt_s
                / float(root_height_correction_halflife_s)
            )
        )
        self._terrain_base_m = (
            None
            if terrain_base_m is None
            else float(terrain_base_m)
        )
        self._maximum_joint_correction_rad = float(
            maximum_joint_correction_rad
        )
        self._maximum_output_joint_speed_rad_s = float(
            maximum_output_joint_speed_rad_s
        )
        self._swing_clearance_margin_m = (
            None
            if swing_clearance_margin_m is None
            else float(swing_clearance_margin_m)
        )
        self._swing_plan_sigma_frames = (
            None
            if swing_plan_sigma_frames is None
            else float(swing_plan_sigma_frames)
        )
        self._touchdown_projection_max_shift_m = (
            None
            if touchdown_projection_max_shift_m is None
            else float(touchdown_projection_max_shift_m)
        )
        self._anticipatory_touchdown_projection = (
            anticipatory_touchdown_projection
        )
        self._touchdown_projection_offsets = None
        if self._touchdown_projection_max_shift_m is not None:
            limit = self._touchdown_projection_max_shift_m
            steps = int(math.ceil(limit / 0.01))
            axis = torch.arange(
                -steps,
                steps + 1,
                dtype=torch.float32,
                device=device,
            ) * 0.01
            grid = torch.cartesian_prod(axis, axis)
            keep = torch.linalg.vector_norm(grid, dim=1) <= limit + 1e-6
            self._touchdown_projection_offsets = grid[keep]
        self._lock_position = torch.full(
            (2, 3), float("nan"), dtype=torch.float32, device=device
        )
        self._previous_support = torch.zeros(2, dtype=torch.bool, device=device)
        self._locked = torch.zeros(2, dtype=torch.bool, device=device)
        self._projected = torch.zeros(2, dtype=torch.bool, device=device)
        self._anticipated = torch.zeros(2, dtype=torch.bool, device=device)
        self._swing_projected = torch.zeros(
            2, dtype=torch.bool, device=device
        )
        self._swing_projection_shift = torch.zeros(
            (2, 3), dtype=torch.float32, device=device
        )
        self._swing_projection_touchdown = torch.full(
            (2,), -1, dtype=torch.long, device=device
        )
        self._swing_projection_clip = torch.full(
            (2,), -1, dtype=torch.long, device=device
        )
        self._joint_offset = torch.zeros(29, dtype=torch.float32, device=device)
        self._previous_joint_position: torch.Tensor | None = None
        self._root_height_offset_z: torch.Tensor | None = None
        self._failure_count = 0

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def reset(self) -> None:
        self._lock_position.fill_(float("nan"))
        self._previous_support.zero_()
        self._locked.zero_()
        self._projected.zero_()
        self._anticipated.zero_()
        self._swing_projected.zero_()
        self._swing_projection_shift.zero_()
        self._swing_projection_touchdown.fill_(-1)
        self._swing_projection_clip.fill_(-1)
        self._joint_offset.zero_()
        self._previous_joint_position = None
        self._root_height_offset_z = None
        self._failure_count = 0

    def _source_support(
        self, result: object
    ) -> tuple[torch.Tensor, int, int]:
        try:
            path = str(result.diagnostics.selected_clip_path)
            frame = int(result.diagnostics.selected_frame)
            clip_index = self._path_to_index[path]
            support = self._support_masks[clip_index]
        except Exception as error:
            raise ValueError("terrain foot lock source is invalid") from error
        if not 0 <= frame < support.shape[0]:
            raise ValueError("terrain foot lock source frame is invalid")
        return support[frame], clip_index, frame

    @staticmethod
    def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quaternion.unbind()
        return torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def _planned_swing_lift(
        self,
        result: object,
        native_feet: torch.Tensor,
        clip_index: int,
        frame: int,
        foot: int,
    ) -> torch.Tensor:
        support = self._support_masks[clip_index][:, foot]
        stop = frame + 1
        while stop < support.shape[0] and not bool(support[stop].item()):
            stop += 1
        source = self._source_foot_positions[clip_index][frame:stop, foot]
        delta = source - source[0]
        output_yaw = self._yaw_from_wxyz(
            result.root_orientation_world_wxyz
        )
        yaw_delta = output_yaw - self._source_root_yaws[clip_index][frame]
        cosine = torch.cos(yaw_delta)
        sine = torch.sin(yaw_delta)
        placed_xy = native_feet[foot, :2] + torch.stack(
            (
                cosine * delta[:, 0] - sine * delta[:, 1],
                sine * delta[:, 0] + cosine * delta[:, 1],
            ),
            dim=1,
        )
        placed_z = native_feet[foot, 2] + delta[:, 2]
        surface = self._sample_surface(placed_xy).to(placed_z.dtype)
        required = torch.clamp(
            surface
            + float(ANKLE_ORIGIN_SOLE_M)
            + self._swing_clearance_margin_m
            - placed_z,
            min=0.0,
        )
        distance = torch.arange(
            required.shape[0], dtype=required.dtype, device=self._device
        )
        weight = torch.exp(
            -0.5 * torch.square(distance / self._swing_plan_sigma_frames)
        )
        return torch.max(required * weight)

    def _feet(self, result: object) -> torch.Tensor:
        try:
            value = self._foot_kinematics.foot_positions(
                result.joint_position.detach().cpu().numpy()[None],
                result.root_position_world.detach().cpu().numpy()[None],
                result.root_orientation_world_wxyz.detach().cpu().numpy()[None],
            )
        except Exception as error:
            raise ValueError("terrain foot lock forward kinematics failed") from error
        feet = torch.as_tensor(
            np.asarray(value, dtype=np.float64)[0],
            dtype=result.joint_position.dtype,
            device=self._device,
        )
        if tuple(feet.shape) != (2, 3) or not torch.isfinite(feet).all():
            raise ValueError("terrain foot lock forward kinematics is invalid")
        return feet

    def _sole_points(self, result: object) -> torch.Tensor:
        try:
            value = self._sole_kinematics.sole_points(
                result.joint_position.detach().cpu().numpy()[None],
                result.root_position_world.detach().cpu().numpy()[None],
                result.root_orientation_world_wxyz.detach().cpu().numpy()[None],
            )
        except Exception as error:
            raise ValueError("terrain touchdown sole kinematics failed") from error
        points = torch.as_tensor(
            np.asarray(value, dtype=np.float64),
            dtype=result.joint_position.dtype,
            device=self._device,
        )
        if (
            points.ndim != 4
            or tuple(points.shape[:2]) != (1, 2)
            or points.shape[2] < 3
            or points.shape[3] != 3
            or not torch.isfinite(points).all()
        ):
            raise ValueError("terrain touchdown sole points are invalid")
        return points[0]

    def _project_touchdown(
        self,
        ankle: torch.Tensor,
        sole: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        offsets = self._touchdown_projection_offsets.to(dtype=ankle.dtype)
        footprint_xy = sole[None, :, :2] + offsets[:, None, :]
        surface = self._sample_surface(footprint_xy.reshape(-1, 2)).reshape(
            offsets.shape[0], sole.shape[0]
        ).to(dtype=ankle.dtype)
        reference_height = self._sample_surface(ankle[:2].reshape(1, 2))[0].to(
            dtype=ankle.dtype
        )
        spread = surface.max(dim=1).values - surface.min(dim=1).values
        zero = torch.linalg.vector_norm(offsets, dim=1) < 1e-8
        if bool((spread[zero] < 0.01).any().item()):
            return ankle.clone(), False
        shifted_ankle_xy = ankle[:2] + offsets
        ankle_surface = self._sample_surface(shifted_ankle_xy).to(
            dtype=ankle.dtype
        )
        distance = torch.linalg.vector_norm(offsets, dim=1)
        score = (
            (spread >= 0.01).to(ankle.dtype) * 1000.0
            + spread * 100.0
            + torch.abs(ankle_surface - reference_height) * 10.0
            + distance
        )
        selected = int(torch.argmin(score).item())
        if float(spread[selected].item()) >= 0.01:
            return ankle.clone(), False
        relative_sole_z = sole[:, 2] - ankle[2]
        target_z = torch.max(surface[selected] - relative_sole_z) + 0.002
        target = torch.cat((shifted_ankle_xy[selected], target_z.reshape(1)))
        return target, True

    def _plan_swing_touchdown(
        self,
        result: object,
        native_feet: torch.Tensor,
        sole_points: torch.Tensor,
        clip_index: int,
        frame: int,
        foot: int,
    ) -> None:
        if self._source_foot_positions is None:
            return
        support = self._support_masks[clip_index][:, foot]
        future = torch.nonzero(
            support[frame + 1 :], as_tuple=False
        ).flatten()
        if future.numel() == 0:
            return
        touchdown = frame + 1 + int(future[0].item())
        source_delta = (
            self._source_foot_positions[clip_index][touchdown, foot]
            - self._source_foot_positions[clip_index][frame, foot]
        )
        output_yaw = self._yaw_from_wxyz(
            result.root_orientation_world_wxyz
        )
        yaw_delta = output_yaw - self._source_root_yaws[clip_index][frame]
        cosine = torch.cos(yaw_delta)
        sine = torch.sin(yaw_delta)
        placed_delta = torch.stack(
            (
                cosine * source_delta[0] - sine * source_delta[1],
                sine * source_delta[0] + cosine * source_delta[1],
                source_delta[2],
            )
        )
        predicted_ankle = native_feet[foot] + placed_delta
        predicted_sole = sole_points[foot] + placed_delta
        target, projected = self._project_touchdown(
            predicted_ankle, predicted_sole
        )
        self._swing_projected[foot] = projected
        self._swing_projection_shift[foot] = target - predicted_ankle
        self._swing_projection_touchdown[foot] = touchdown
        self._swing_projection_clip[foot] = clip_index

    def _apply_swing_touchdown_plan(
        self,
        *,
        targets: torch.Tensor,
        correction_mask: torch.Tensor,
        native_feet: torch.Tensor,
        support: torch.Tensor,
        clip_index: int,
        frame: int,
    ) -> None:
        source_support = self._support_masks[clip_index]
        for foot in torch.nonzero(~support, as_tuple=False).flatten().tolist():
            if (
                not bool(self._swing_projected[foot].item())
                or int(self._swing_projection_clip[foot].item()) != clip_index
                or frame
                >= int(self._swing_projection_touchdown[foot].item())
            ):
                continue
            start = frame
            while start > 0 and not bool(
                source_support[start - 1, foot].item()
            ):
                start -= 1
            touchdown = int(self._swing_projection_touchdown[foot].item())
            blend_start = max(
                start, touchdown - _TOUCHDOWN_PROJECTION_LEAD_FRAMES
            )
            if frame < blend_start:
                continue
            final_swing = max(blend_start, touchdown - 1)
            phase = float(frame - blend_start) / float(
                max(1, final_swing - blend_start)
            )
            phase = min(1.0, max(0.0, phase))
            blend = phase * phase * (3.0 - 2.0 * phase)
            shifted = native_feet[foot] + (
                blend * self._swing_projection_shift[foot]
            )
            targets[foot, :2] = shifted[:2]
            targets[foot, 2] = torch.maximum(targets[foot, 2], shifted[2])
            correction_mask[foot] = True

    @staticmethod
    def _copy_result(
        result: object,
        *,
        joints: torch.Tensor,
        velocity: torch.Tensor,
        root_position: torch.Tensor,
        root_velocity: torch.Tensor | None,
    ):
        try:
            values = vars(result).copy()
        except TypeError as error:
            raise ValueError("terrain foot lock result is not copyable") from error
        values.update(
            joint_position=joints,
            joint_velocity=velocity,
            root_position_world=root_position,
        )
        if root_velocity is not None:
            values["root_linear_velocity_world"] = root_velocity
        return SimpleNamespace(**values)

    def apply(self, result: object):
        """Return one corrected result and update persistent contact locks."""

        required = (
            "joint_position",
            "joint_velocity",
            "root_position_world",
            "root_orientation_world_wxyz",
            "diagnostics",
        )
        if any(not hasattr(result, name) for name in required):
            raise ValueError("terrain foot lock result is invalid")
        support, clip_index, frame = self._source_support(result)
        if (
            self._root_height_correction_alpha is not None
            and bool(
                getattr(result.diagnostics, "terrain_chunk_start", False)
            )
        ):
            self._root_height_offset_z = None
        native_feet = self._feet(result)
        sole_points = (
            None
            if self._touchdown_projection_offsets is None
            else self._sole_points(result)
        )
        if sole_points is not None and self._anticipatory_touchdown_projection:
            for foot in torch.nonzero(
                ~support, as_tuple=False
            ).flatten().tolist():
                plan_valid = (
                    bool(self._swing_projected[foot].item())
                    and int(self._swing_projection_clip[foot].item())
                    == clip_index
                    and frame
                    < int(self._swing_projection_touchdown[foot].item())
                )
                if not plan_valid:
                    self._swing_projected[foot] = False
                    self._plan_swing_touchdown(
                        result,
                        native_feet,
                        sole_points,
                        clip_index,
                        frame,
                        foot,
                    )
        onset = support & ~self._previous_support
        self._lock_position[onset] = native_feet[onset]
        self._anticipated[onset] = False
        if sole_points is not None:
            for foot in torch.nonzero(onset, as_tuple=False).flatten().tolist():
                anticipated = (
                    bool(self._swing_projected[foot].item())
                    and int(self._swing_projection_clip[foot].item())
                    == clip_index
                    and int(self._swing_projection_touchdown[foot].item())
                    == frame
                )
                if anticipated:
                    target = (
                        native_feet[foot]
                        + self._swing_projection_shift[foot]
                    )
                    projected = True
                else:
                    target, projected = self._project_touchdown(
                        native_feet[foot], sole_points[foot]
                    )
                self._lock_position[foot] = target
                self._projected[foot] = projected
                self._anticipated[foot] = anticipated
        self._locked[onset] = True
        self._lock_position[~support] = float("nan")
        self._locked[~support] = False
        self._projected[~support] = False
        self._anticipated[~support] = False
        self._swing_projected[support] = False
        self._swing_projection_touchdown[support] = -1
        self._swing_projection_clip[support] = -1
        active = support & self._locked
        if bool(active.any()):
            deviation = torch.linalg.vector_norm(
                self._lock_position - native_feet, dim=1
            )
            unlock = active & (deviation > self._unlock_radius_m)
            self._locked[unlock] = False
            active = support & self._locked
        targets = native_feet.clone()
        desired = result.joint_position.clone()
        correction_mask = active.clone()
        if bool(active.any()):
            locked = self._lock_position[active]
            try:
                surface = self._sample_surface(locked[:, :2])
            except Exception as error:
                    raise ValueError(
                        "terrain foot lock surface sampling failed"
                    ) from error
            if (
                not isinstance(surface, torch.Tensor)
                or tuple(surface.shape) != (int(active.sum().item()),)
                or surface.device != self._device
                or not torch.isfinite(surface).all()
            ):
                raise ValueError("terrain foot lock surface samples are invalid")
            targets[active] = locked
            targets[active, 2] = surface.to(targets.dtype) + float(
                ANKLE_ORIGIN_SOLE_M
            )
            if sole_points is not None:
                projected_active = active & self._projected
                for foot in torch.nonzero(
                    projected_active, as_tuple=False
                ).flatten().tolist():
                    shift = (
                        self._lock_position[foot, :2]
                        - native_feet[foot, :2]
                    )
                    footprint_xy = sole_points[foot, :, :2] + shift
                    footprint_surface = self._sample_surface(
                        footprint_xy
                    ).to(dtype=targets.dtype)
                    relative_sole_z = (
                        sole_points[foot, :, 2] - native_feet[foot, 2]
                    )
                    targets[foot, 2] = torch.max(
                        footprint_surface - relative_sole_z
                    ) + 0.002
        if self._swing_clearance_margin_m is not None:
            swing_indices = torch.nonzero(~support, as_tuple=False).flatten()
            if swing_indices.numel():
                swing_xy = native_feet[swing_indices, :2]
                try:
                    swing_surface = self._sample_surface(swing_xy)
                except Exception as error:
                    raise ValueError(
                        "terrain swing-clearance surface sampling failed"
                    ) from error
                if (
                    not isinstance(swing_surface, torch.Tensor)
                    or tuple(swing_surface.shape) != (swing_indices.numel(),)
                    or swing_surface.device != self._device
                    or not torch.isfinite(swing_surface).all()
                ):
                    raise ValueError(
                        "terrain swing-clearance surface samples are invalid"
                    )
                minimum_z = (
                    swing_surface.to(targets.dtype)
                    + float(ANKLE_ORIGIN_SOLE_M)
                    + self._swing_clearance_margin_m
                )
                if self._swing_plan_sigma_frames is not None:
                    for local, foot in enumerate(swing_indices.tolist()):
                        try:
                            planned_lift = self._planned_swing_lift(
                                result,
                                native_feet,
                                clip_index,
                                frame,
                                int(foot),
                            )
                        except Exception:
                            self._failure_count += 1
                            continue
                        minimum_z[local] = torch.maximum(
                            minimum_z[local],
                            native_feet[foot, 2] + planned_lift,
                        )
                needs_lift = native_feet[swing_indices, 2] < minimum_z
                lifted = swing_indices[needs_lift]
                correction_mask[lifted] = True
                targets[lifted, 2] = minimum_z[needs_lift]
        if sole_points is not None:
            self._apply_swing_touchdown_plan(
                targets=targets,
                correction_mask=correction_mask,
                native_feet=native_feet,
                support=support,
                clip_index=clip_index,
                frame=frame,
            )
        previous_root_offset = self._root_height_offset_z
        root_position = result.root_position_world.clone()
        root_velocity = getattr(result, "root_linear_velocity_world", None)
        if root_velocity is not None:
            root_velocity = root_velocity.clone()
        terrain_engaged = bool(active.any()) and (
            self._terrain_base_m is None
            or float(torch.max(surface).item()) > self._terrain_base_m + 0.05
            or float((torch.max(surface) - torch.min(surface)).item()) > 0.05
        )
        if self._root_height_correction_alpha is not None:
            if bool(active.any()) and terrain_engaged:
                desired_root_offset = (
                    targets[active, 2] - native_feet[active, 2]
                ).mean().clamp(-0.25, 0.25)
            else:
                desired_root_offset = torch.zeros(
                    (), dtype=targets.dtype, device=self._device
                )
            if previous_root_offset is None:
                self._root_height_offset_z = torch.zeros_like(
                    desired_root_offset
                )
            else:
                requested_step = self._root_height_correction_alpha * (
                    desired_root_offset - previous_root_offset
                )
                self._root_height_offset_z = previous_root_offset + (
                    requested_step.clamp(-0.02, 0.02)
                )
            root_position[2] += self._root_height_offset_z
            native_feet[:, 2] += self._root_height_offset_z
            if root_velocity is not None and previous_root_offset is not None:
                root_velocity[2] += (
                    self._root_height_offset_z - previous_root_offset
                ) / self._dt_s
        if bool(correction_mask.any()):
            try:
                solved_numpy = self._foot_kinematics.solve_leg_positions(
                    result.joint_position.detach().cpu().numpy(),
                    root_position.detach().cpu().numpy(),
                    result.root_orientation_world_wxyz.detach().cpu().numpy(),
                    correction_mask.detach().cpu().numpy(),
                    targets.detach().cpu().numpy(),
                )
                desired = torch.as_tensor(
                    np.asarray(solved_numpy, dtype=np.float64),
                    dtype=result.joint_position.dtype,
                    device=self._device,
                )
                if tuple(desired.shape) != (29,) or not torch.isfinite(
                    desired
                ).all():
                    raise ValueError("invalid solve")
                if float(
                    torch.max(torch.abs(desired - result.joint_position)).item()
                ) > self._maximum_joint_correction_rad:
                    self._locked[active] = False
                    desired = result.joint_position.clone()
            except Exception:
                self._failure_count += 1
                self._locked[active] = False
                desired = result.joint_position.clone()
        desired_offset = desired - result.joint_position
        self._joint_offset = self._joint_offset + self._correction_alpha * (
            desired_offset - self._joint_offset
        )
        projected_support = active & self._projected
        for foot in torch.nonzero(
            projected_support, as_tuple=False
        ).flatten().tolist():
            indices = _LEG_JOINT_INDICES[foot].to(device=self._device)
            self._joint_offset[indices] = desired_offset[indices]
        self._joint_offset.clamp_(
            -self._maximum_joint_correction_rad,
            self._maximum_joint_correction_rad,
        )
        solved = result.joint_position + self._joint_offset
        velocity = result.joint_velocity.clone()
        if self._previous_joint_position is not None:
            maximum_step = self._maximum_output_joint_speed_rad_s * self._dt_s
            solved = self._previous_joint_position + torch.clamp(
                solved - self._previous_joint_position,
                min=-maximum_step,
                max=maximum_step,
            )
            self._joint_offset = solved - result.joint_position
            velocity = (solved - self._previous_joint_position) / self._dt_s
        self._previous_joint_position = solved.clone()
        self._previous_support = support.clone()
        return self._copy_result(
            result,
            joints=solved,
            velocity=velocity,
            root_position=root_position,
            root_velocity=root_velocity,
        )

    def preview(self, results: Sequence[object]) -> tuple[object, ...]:
        """Apply an isolated sequence without changing persistent lock state."""

        snapshot = (
            self._lock_position.clone(),
            self._previous_support.clone(),
            self._locked.clone(),
            self._projected.clone(),
            self._anticipated.clone(),
            self._swing_projected.clone(),
            self._swing_projection_shift.clone(),
            self._swing_projection_touchdown.clone(),
            self._swing_projection_clip.clone(),
            self._joint_offset.clone(),
            (
                None
                if self._previous_joint_position is None
                else self._previous_joint_position.clone()
            ),
            self._failure_count,
            (
                None
                if self._root_height_offset_z is None
                else self._root_height_offset_z.clone()
            ),
        )
        try:
            return tuple(self.apply(result) for result in results)
        finally:
            (
                lock_position,
                previous_support,
                locked,
                projected,
                anticipated,
                swing_projected,
                swing_projection_shift,
                swing_projection_touchdown,
                swing_projection_clip,
                joint_offset,
                previous_joint_position,
                failure_count,
                root_height_offset_z,
            ) = snapshot
            self._lock_position.copy_(lock_position)
            self._previous_support.copy_(previous_support)
            self._locked.copy_(locked)
            self._projected.copy_(projected)
            self._anticipated.copy_(anticipated)
            self._swing_projected.copy_(swing_projected)
            self._swing_projection_shift.copy_(swing_projection_shift)
            self._swing_projection_touchdown.copy_(
                swing_projection_touchdown
            )
            self._swing_projection_clip.copy_(swing_projection_clip)
            self._joint_offset.copy_(joint_offset)
            self._previous_joint_position = previous_joint_position
            self._failure_count = failure_count
            self._root_height_offset_z = root_height_offset_z


def build_terrain_foot_lock(
    resolved: object,
    foot_kinematics: object,
    sole_kinematics: object | None = None,
    *,
    swing_clearance_margin_m: float | None = None,
    correction_halflife_s: float = 0.04,
    root_height_correction_halflife_s: float | None = None,
    swing_plan_sigma_frames: float | None = None,
    touchdown_projection_max_shift_m: float | None = None,
    anticipatory_touchdown_projection: bool = False,
):
    """Build a source-contact foot lock against the resolved query terrain."""

    from .torch_contact_segments import source_support_mask

    try:
        clip_paths = tuple(
            clip.relative_path for clip in resolved.dataset.folder.clips
        )
        support_masks = tuple(
            source_support_mask(resolved.dataset, index)
            for index in range(len(clip_paths))
        )
        device = resolved.device
        source_foot_positions = None
        source_root_yaws = None
        if swing_plan_sigma_frames is not None:
            feet_indices = (
                resolved.dataset.folder.layout.left_foot_body_index,
                resolved.dataset.folder.layout.right_foot_body_index,
            )
            root_index = resolved.dataset.folder.layout.root_body_index
            source_foot_positions = tuple(
                torch.tensor(
                    clip.body_position_world[:, feet_indices],
                    dtype=torch.float32,
                    device=device,
                )
                for clip in resolved.dataset.folder.clips
            )
            source_root_yaws = tuple(
                torch.atan2(
                    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
                )
                for clip in resolved.dataset.folder.clips
                for q in (
                    torch.tensor(
                        clip.body_quaternion_world_wxyz[:, root_index],
                        dtype=torch.float32,
                        device=device,
                    ),
                )
            )
        measurement = resolved.measurement_extension
    except Exception as error:
        raise ValueError("cannot build terrain foot lock") from error

    def sample_surface(points: torch.Tensor) -> torch.Tensor:
        return measurement.query_grid.sample_xy(
            measurement.alignment.matcher_to_scene_xy(points)
        )

    return TerrainFootLockFilter(
        clip_paths=clip_paths,
        support_masks=support_masks,
        source_foot_positions=source_foot_positions,
        source_root_yaws=source_root_yaws,
        foot_kinematics=foot_kinematics,
        sole_kinematics=sole_kinematics,
        sample_surface=sample_surface,
        device=device,
        swing_clearance_margin_m=swing_clearance_margin_m,
        correction_halflife_s=correction_halflife_s,
        root_height_correction_halflife_s=(
            root_height_correction_halflife_s
        ),
        terrain_base_m=float(measurement.query_grid.height_z.min().item()),
        swing_plan_sigma_frames=swing_plan_sigma_frames,
        touchdown_projection_max_shift_m=(
            touchdown_projection_max_shift_m
        ),
        anticipatory_touchdown_projection=(
            anticipatory_touchdown_projection
        ),
    )
