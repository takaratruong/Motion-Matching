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
        foot_kinematics: object,
        sample_surface: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dt_s: float = 0.02,
        unlock_radius_m: float = 0.25,
        correction_halflife_s: float = 0.04,
        maximum_joint_correction_rad: float = 0.35,
        maximum_output_joint_speed_rad_s: float = 12.0,
        swing_clearance_margin_m: float | None = None,
    ) -> None:
        paths = tuple(clip_paths)
        masks = tuple(support_masks)
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
            or (
                swing_clearance_margin_m is not None
                and (
                    isinstance(swing_clearance_margin_m, bool)
                    or not isinstance(swing_clearance_margin_m, (int, float))
                    or not math.isfinite(float(swing_clearance_margin_m))
                    or not 0.0 <= float(swing_clearance_margin_m) <= 0.20
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
        self._lock_position = torch.full(
            (2, 3), float("nan"), dtype=torch.float32, device=device
        )
        self._previous_support = torch.zeros(2, dtype=torch.bool, device=device)
        self._locked = torch.zeros(2, dtype=torch.bool, device=device)
        self._joint_offset = torch.zeros(29, dtype=torch.float32, device=device)
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
        self._previous_joint_position = None
        self._failure_count = 0

    def _source_support(self, result: object) -> torch.Tensor:
        try:
            path = str(result.diagnostics.selected_clip_path)
            frame = int(result.diagnostics.selected_frame)
            clip_index = self._path_to_index[path]
            support = self._support_masks[clip_index]
        except Exception as error:
            raise ValueError("terrain foot lock source is invalid") from error
        if not 0 <= frame < support.shape[0]:
            raise ValueError("terrain foot lock source frame is invalid")
        return support[frame]

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
        result: object, *, joints: torch.Tensor, velocity: torch.Tensor
    ):
        try:
            values = vars(result).copy()
        except TypeError as error:
            raise ValueError("terrain foot lock result is not copyable") from error
        values.update(joint_position=joints, joint_velocity=velocity)
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
        support = self._source_support(result)
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
                needs_lift = native_feet[swing_indices, 2] < minimum_z
                lifted = swing_indices[needs_lift]
                correction_mask[lifted] = True
                targets[lifted, 2] = minimum_z[needs_lift]
        if bool(correction_mask.any()):
            try:
                solved_numpy = self._foot_kinematics.solve_leg_positions(
                    result.joint_position.detach().cpu().numpy(),
                    result.root_position_world.detach().cpu().numpy(),
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
        return self._copy_result(result, joints=solved, velocity=velocity)


def build_terrain_foot_lock(
    resolved: object,
    foot_kinematics: object,
    *,
    swing_clearance_margin_m: float | None = None,
    correction_halflife_s: float = 0.04,
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
        measurement = resolved.measurement_extension
        device = resolved.device
    except Exception as error:
        raise ValueError("cannot build terrain foot lock") from error

    def sample_surface(points: torch.Tensor) -> torch.Tensor:
        return measurement.query_grid.sample_xy(
            measurement.alignment.matcher_to_scene_xy(points)
        )

    return TerrainFootLockFilter(
        clip_paths=clip_paths,
        support_masks=support_masks,
        foot_kinematics=foot_kinematics,
        sample_surface=sample_surface,
        device=device,
        swing_clearance_margin_m=swing_clearance_margin_m,
        correction_halflife_s=correction_halflife_s,
    )
