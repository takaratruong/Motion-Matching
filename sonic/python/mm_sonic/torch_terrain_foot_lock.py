"""Stateful terrain-aware foot locking for emitted kinematic references."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import torch

from .torch_contact_segments import ANKLE_ORIGIN_SOLE_M


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
        sample_surface: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dt_s: float = 0.02,
        unlock_radius_m: float = 0.25,
        correction_halflife_s: float = 0.04,
        maximum_joint_correction_rad: float = 0.35,
        maximum_output_joint_speed_rad_s: float = 12.0,
        swing_clearance_margin_m: float | None = None,
        swing_plan_sigma_frames: float | None = None,
        support_root_height: bool = False,
        maximum_root_correction_m: float = 0.25,
        maximum_root_correction_speed_mps: float = 1.5,
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
            or not math.isfinite(float(maximum_joint_correction_rad))
            or float(maximum_joint_correction_rad) <= 0.0
            or not math.isfinite(float(maximum_output_joint_speed_rad_s))
            or float(maximum_output_joint_speed_rad_s) <= 0.0
            or type(support_root_height) is not bool
            or not math.isfinite(float(maximum_root_correction_m))
            or float(maximum_root_correction_m) <= 0.0
            or not math.isfinite(float(maximum_root_correction_speed_mps))
            or float(maximum_root_correction_speed_mps) <= 0.0
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
        self._sample_surface = sample_surface
        self._device = device
        self._dt_s = float(dt_s)
        self._unlock_radius_m = float(unlock_radius_m)
        self._correction_alpha = 1.0 - math.exp(
            -math.log(2.0) * self._dt_s / float(correction_halflife_s)
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
        self._support_root_height = support_root_height
        self._maximum_root_correction_m = float(maximum_root_correction_m)
        self._maximum_root_correction_speed_mps = float(
            maximum_root_correction_speed_mps
        )
        self._lock_position = torch.full(
            (2, 3), float("nan"), dtype=torch.float32, device=device
        )
        self._previous_support = torch.zeros(2, dtype=torch.bool, device=device)
        self._locked = torch.zeros(2, dtype=torch.bool, device=device)
        self._joint_offset = torch.zeros(29, dtype=torch.float32, device=device)
        self._root_height_offset = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        self._previous_joint_position: torch.Tensor | None = None
        self._failure_count = 0

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def reset(self) -> None:
        self._lock_position.fill_(float("nan"))
        self._previous_support.zero_()
        self._locked.zero_()
        self._joint_offset.zero_()
        self._root_height_offset.zero_()
        self._previous_joint_position = None
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
        native_feet = self._feet(result)
        onset = support & ~self._previous_support
        self._lock_position[onset] = native_feet[onset]
        self._locked[onset] = True
        self._lock_position[~support] = float("nan")
        self._locked[~support] = False
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
        previous_root_offset = self._root_height_offset.clone()
        target_root_offset = torch.zeros_like(self._root_height_offset)
        if self._support_root_height and bool(active.any()):
            target_root_offset = torch.mean(
                targets[active, 2] - native_feet[active, 2]
            ).clamp(
                -self._maximum_root_correction_m,
                self._maximum_root_correction_m,
            )
        candidate_root_offset = self._root_height_offset + self._correction_alpha * (
            target_root_offset - self._root_height_offset
        )
        maximum_root_step = (
            self._maximum_root_correction_speed_mps * self._dt_s
        )
        self._root_height_offset = self._root_height_offset + torch.clamp(
            candidate_root_offset - self._root_height_offset,
            min=-maximum_root_step,
            max=maximum_root_step,
        )
        corrected_root = result.root_position_world.clone()
        corrected_root[2] += self._root_height_offset
        root_velocity = None
        if hasattr(result, "root_linear_velocity_world"):
            root_velocity = result.root_linear_velocity_world.clone()
            root_velocity[2] += (
                self._root_height_offset - previous_root_offset
            ) / self._dt_s
        correction_failed = False
        if bool(correction_mask.any()):
            try:
                solved_numpy = self._foot_kinematics.solve_leg_positions(
                    result.joint_position.detach().cpu().numpy(),
                    corrected_root.detach().cpu().numpy(),
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
                    correction_failed = True
            except Exception:
                self._failure_count += 1
                self._locked[active] = False
                desired = result.joint_position.clone()
                correction_failed = True
        if correction_failed:
            self._root_height_offset = previous_root_offset
            corrected_root = result.root_position_world.clone()
            corrected_root[2] += self._root_height_offset
            root_velocity = (
                result.root_linear_velocity_world.clone()
                if hasattr(result, "root_linear_velocity_world")
                else None
            )
        desired_offset = desired - result.joint_position
        self._joint_offset = self._joint_offset + self._correction_alpha * (
            desired_offset - self._joint_offset
        )
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
            root_position=corrected_root,
            root_velocity=root_velocity,
        )


def build_terrain_foot_lock(
    resolved: object,
    foot_kinematics: object,
    *,
    swing_clearance_margin_m: float | None = None,
    correction_halflife_s: float = 0.04,
    swing_plan_sigma_frames: float | None = None,
    support_root_height: bool = False,
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
        sample_surface=sample_surface,
        device=device,
        swing_clearance_margin_m=swing_clearance_margin_m,
        correction_halflife_s=correction_halflife_s,
        swing_plan_sigma_frames=swing_plan_sigma_frames,
        support_root_height=support_root_height,
    )
