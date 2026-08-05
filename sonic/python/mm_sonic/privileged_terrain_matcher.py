"""Privileged online G1 motion matching over flat and stair motion together.

This is the deliberately easy, global-information bootstrap requested for the
terrain project.  It is a conventional rolling motion matcher, not a route or
fragment player: every search considers pose, future command trajectory,
contact phase, current location relative to the known staircase, and staircase
geometry.  Stair samples are always placed by mapping their recorded terrain
frame to the target terrain frame, so a transition cannot silently move the
stairs underneath the character.

The observation boundary is intentionally privileged for now.  A later stage
can replace :class:`PrivilegedStairScene` with a scene reconstructed from an ego
height map without changing the database or transition machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .joints import ContractError
from .terrain_oracle.math3d import (
    angular_velocity_world_wxyz,
    unroll_quaternions_wxyz,
)
from .build_grail_terrain_archive import DEFAULT_G1_MJCF
from .grail_terrain_source import (
    G1MujocoFK,
    MUJOCO_TO_ISAACLAB_JOINT,
    derive_clip_velocity,
)
from .terrain_oracle.stair_geometry_warp import (
    rank_geometry_compatible_sources,
    warp_archive_clip_to_stair_geometry,
)
from .stair_mesh_profile import measure_archive_stair_profile
from .torch_motion_data import G1_TAKARA_LAYOUT, MotionClip, MotionFolder
from .torch_motion_features import (
    FEATURE_HORIZON_FRAMES,
    GeneratedFeatureState,
    extract_query_features,
)
from .torch_motion_matcher import (
    ForcedMotionAlignment,
    MatcherConfig,
    MotionMatchResult,
    TorchMotionMatcher,
    _quat_yaw,
    _rotate_z,
    _wrapped_angle,
    predict_command_trajectory,
    predict_takara_ball_trajectory,
    search_is_due,
)


DEFAULT_FLAT_MOTIONS = Path(
    "/move/data/terrain-aware/motion-matching/"
    "takara_bones_walk_support_v2_startstop"
)
DEFAULT_STAIRS_ARCHIVE = Path(
    "/move/data/terrain-aware/motion-matching/"
    "grail-stairs500-clean-50hz-v1.zarr"
)


def _readonly_float32(value: object) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    result.setflags(write=False)
    return result


def _yaw_wxyz_numpy(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _rotate_z_numpy(value: np.ndarray, yaw: float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    cosine = math.cos(float(yaw))
    sine = math.sin(float(yaw))
    output = source.copy()
    output[..., 0] = cosine * source[..., 0] - sine * source[..., 1]
    output[..., 1] = sine * source[..., 0] + cosine * source[..., 1]
    return output


@dataclass(frozen=True)
class StairClipMetadata:
    unified_clip_index: int
    archive_clip_index: int
    name: str
    traversal: str
    rise_m: float
    tread_m: float
    step_count: int
    travel_yaw_rad: float
    terrain_position_world_xyz: tuple[float, float, float]
    terrain_usd_path: str
    compatible_target_archive_index: int
    physical_rise_m: float
    physical_tread_m: float
    physical_level_count: int
    physical_tread_edges_m: tuple[float, ...] = ()
    physical_tread_heights_m: tuple[float, ...] = ()


@dataclass(frozen=True)
class PrivilegedMotionLibrary:
    folder: MotionFolder
    flat_clip_count: int
    clip_metadata: tuple[StairClipMetadata | None, ...]


@dataclass(frozen=True)
class WarpedLibraryBuildReport:
    target_archive_clip_index: int
    accepted_source_indices: tuple[int, ...]
    attempted_source_indices: tuple[int, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MultiTargetWarpedLibraryBuildReport:
    """Mechanically audited warp reports for one shared multi-scene bank."""

    targets: tuple[WarpedLibraryBuildReport, ...]


@dataclass(frozen=True)
class PrivilegedStairScene:
    """Exact target staircase known in world coordinates."""

    traversal: str
    rise_m: float
    tread_m: float
    step_count: int
    travel_yaw_rad: float
    terrain_position_world_xyz: tuple[float, float, float]
    width_m: float = 1.2
    label: str = "target_stair"
    archive_clip_index: int | None = None
    physical_run_m: float | None = None
    physical_height_m: float | None = None
    physical_rise_m: float | None = None
    physical_tread_m: float | None = None
    physical_level_count: int | None = None
    physical_tread_edges_m: tuple[float, ...] = ()
    physical_tread_heights_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.traversal not in {"up", "down"}:
            raise ContractError("stair traversal must be 'up' or 'down'")
        values = (
            self.rise_m,
            self.tread_m,
            float(self.step_count),
            self.travel_yaw_rad,
            *self.terrain_position_world_xyz,
            self.width_m,
            self.tread_m * self.step_count
            if self.physical_run_m is None
            else self.physical_run_m,
            self.rise_m * self.step_count
            if self.physical_height_m is None
            else self.physical_height_m,
            self.rise_m if self.physical_rise_m is None else self.physical_rise_m,
            self.tread_m if self.physical_tread_m is None else self.physical_tread_m,
            float(
                self.step_count
                if self.physical_level_count is None
                else self.physical_level_count
            ),
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ContractError("stair scene values must be finite")
        if (
            self.rise_m <= 0.0
            or self.tread_m <= 0.0
            or type(self.step_count) is not int
            or self.step_count <= 0
            or self.width_m <= 0.0
            or (
                self.physical_run_m is not None
                and self.physical_run_m <= 0.0
            )
            or (
                self.physical_height_m is not None
                and self.physical_height_m <= 0.0
            )
            or (
                self.physical_rise_m is not None
                and self.physical_rise_m <= 0.0
            )
            or (
                self.physical_tread_m is not None
                and self.physical_tread_m <= 0.0
            )
            or (
                self.physical_level_count is not None
                and (
                    type(self.physical_level_count) is not int
                    or self.physical_level_count <= 0
                )
            )
        ):
            raise ContractError("stair geometry must be positive")
        if bool(self.physical_tread_edges_m) != bool(
            self.physical_tread_heights_m
        ):
            raise ContractError(
                "stair edges and heights must be supplied together"
            )
        if self.physical_tread_edges_m:
            edges = np.asarray(
                self.physical_tread_edges_m, dtype=np.float64
            )
            heights = np.asarray(
                self.physical_tread_heights_m, dtype=np.float64
            )
            if (
                edges.shape != (self.surface_level_count + 1,)
                or heights.shape != (self.surface_level_count,)
                or not np.isfinite(edges).all()
                or not np.isfinite(heights).all()
                or abs(float(edges[0])) > 1.0e-6
                or np.any(np.diff(edges) <= 0.0)
                or np.any(heights <= 0.0)
            ):
                raise ContractError("piecewise stair surface is invalid")

    @property
    def run_m(self) -> float:
        return float(
            self.surface_edges_m[-1]
            if self.physical_tread_edges_m
            else (
                self.tread_m * self.step_count
                if self.physical_run_m is None
                else self.physical_run_m
            )
        )

    @property
    def height_m(self) -> float:
        return float(
            max(self.surface_heights_m)
            if self.physical_tread_heights_m
            else (
                self.rise_m * self.step_count
                if self.physical_height_m is None
                else self.physical_height_m
            )
        )

    @property
    def surface_rise_m(self) -> float:
        return float(
            self.rise_m
            if self.physical_rise_m is None
            else self.physical_rise_m
        )

    @property
    def surface_tread_m(self) -> float:
        return float(
            self.tread_m
            if self.physical_tread_m is None
            else self.physical_tread_m
        )

    @property
    def surface_level_count(self) -> int:
        return int(
            self.step_count
            if self.physical_level_count is None
            else self.physical_level_count
        )

    @property
    def surface_edges_m(self) -> tuple[float, ...]:
        if self.physical_tread_edges_m:
            return tuple(float(value) for value in self.physical_tread_edges_m)
        total_run = (
            self.surface_tread_m * self.surface_level_count
            if self.physical_run_m is None
            else float(self.physical_run_m)
        )
        return tuple(
            float(value)
            for value in np.linspace(
                0.0,
                total_run,
                self.surface_level_count + 1,
            )
        )

    @property
    def surface_heights_m(self) -> tuple[float, ...]:
        if self.physical_tread_heights_m:
            return tuple(
                float(value) for value in self.physical_tread_heights_m
            )
        if self.traversal == "up":
            levels = range(1, self.surface_level_count + 1)
        else:
            levels = range(self.surface_level_count, 0, -1)
        rise = (
            self.surface_rise_m
            if self.physical_height_m is None
            else float(self.physical_height_m) / self.surface_level_count
        )
        return tuple(float(rise * level) for level in levels)

    def world_to_terrain(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        origin = np.asarray(self.terrain_position_world_xyz, dtype=np.float64)
        return _rotate_z_numpy(points - origin, -self.travel_yaw_rad)

    def surface_height_world(self, points_world_xy: np.ndarray) -> np.ndarray:
        """Idealized tread height for search/evaluation (infinite side ground)."""

        xy = np.asarray(points_world_xy, dtype=np.float64)
        points = np.zeros(xy.shape[:-1] + (3,), dtype=np.float64)
        points[..., :2] = xy
        local = self.world_to_terrain(points)
        x = local[..., 0]
        base = float(self.terrain_position_world_xyz[2])
        edges = np.asarray(self.surface_edges_m, dtype=np.float64)
        heights = np.asarray(self.surface_heights_m, dtype=np.float64)
        index = np.searchsorted(edges[1:-1], x, side="right")
        surface = heights[np.clip(index, 0, len(heights) - 1)]
        before = 0.0 if self.traversal == "up" else self.height_m
        after = self.height_m if self.traversal == "up" else 0.0
        surface = np.where(x < 0.0, before, surface)
        surface = np.where(x >= edges[-1], after, surface)
        return base + surface


@dataclass(frozen=True)
class PrivilegedMatcherConfig:
    search_interval_steps: int = 5
    transition_penalty: float = 0.65
    stage_weight: float = 5.0
    lateral_weight: float = 2.0
    height_weight: float = 2.0
    relative_facing_weight: float = 1.25
    rise_weight: float = 1.0
    tread_weight: float = 0.75
    step_count_weight: float = 0.35
    active_entry_margin_m: float = 1.55
    active_exit_margin_m: float = 0.75
    contact_speed_threshold_mps: float = 0.24
    # Until per-frame geometry warping/IK is enabled, only combine clips whose
    # authored stair actually fits the target.  The 500-clip bank still gives
    # multiple matches for most geometries; loose matching visibly hovers over
    # short target steps or penetrates taller ones.
    maximum_rise_error_m: float = 0.008
    maximum_tread_error_m: float = 0.015
    maximum_step_count_error: int = 0
    # Match route phase tightly in the travel direction, but allow the
    # authored descent preparation to lower the pelvis before the first step.
    # The clean down-stair clips begin in a deliberately crouched pose about
    # 7--10 cm below the ordinary flat gait.  Treating forward phase and root
    # height as one Euclidean distance made every safe oblique descent entry
    # impossible even when its feet, facing, and forward phase agreed.
    maximum_transition_forward_offset_m: float = 0.06
    maximum_transition_vertical_offset_m: float = 0.11
    maximum_transition_facing_error_rad: float = math.radians(15.0)
    maximum_transition_lateral_shift_m: float = 0.35
    # A target-registered stair clip is already mechanically audited against
    # the complete route.  Preserve its successor sequence through the route;
    # repeated nearest-neighbour re-entry into an earlier, similar double-
    # support phase creates a visible loop.  The separate terrain-height
    # matcher performs true cross-clip online matching at support boundaries.
    minimum_stair_clip_dwell_steps: int = 500
    # Entry anchors are the mechanically qualified start frames from the
    # retained zero-seam course.  Starting the same 10-second traversal at an
    # arbitrary nearest-neighbour frame can exhaust the source before the
    # character reaches the far side, leaving it cycling inside a riser.
    qualified_registered_entry_frames: tuple[tuple[int, int], ...] = (
        (449, 3),
        (197, 80),
    )
    # Registered assets outside the original two-stair demo do not need a
    # hand-authored entry table.  Their coherent clips all contain a flat
    # approach.  Pick a moving, terrain-aligned row while the pelvis is still
    # on the entry support surface, then protect its successors through the
    # traversal.  This turns the 50 accepted assets into a usable runtime bank
    # instead of making every new staircase another hard-coded special case.
    automatic_registered_entry_height_tolerance_m: float = 0.09
    automatic_registered_entry_max_frame: int = 200
    automatic_registered_entry_target_speed_mps: float = 0.38
    maximum_registered_entry_progress_error_m: float = 0.16
    maximum_registered_entry_lateral_m: float = 0.12
    maximum_registered_exit_height_error_m: float = 0.06
    registered_entry_progress_gain: float = 1.5
    registered_entry_lateral_gain: float = 1.5
    maximum_registered_entry_staging_speed_mps: float = 0.38
    # A game controller does not reselect a new animation every 10 Hz query.
    # Once a compatible flat clip has been chosen, preserve its authored
    # successor for a short phrase; turning clips get a longer phrase so a
    # planted step cannot be restarted at the same contact frame forever.
    minimum_flat_clip_dwell_steps: int = 25
    minimum_turn_clip_dwell_steps: int = 50
    joint_pose_weight: float = 0.8


@dataclass(frozen=True)
class PrivilegedMotionStep:
    result: MotionMatchResult
    selected_row: int
    mode: str
    searched: bool
    terrain_local_root_xyz: tuple[float, float, float] | None


def load_privileged_motion_library(
    flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
    stairs_archive: str | Path = DEFAULT_STAIRS_ARCHIVE,
    *,
    stair_clip_limit: int | None = None,
    stair_clip_indices: Sequence[int] | None = None,
    stair_clip_exclude_indices: Sequence[int] = (),
) -> PrivilegedMotionLibrary:
    """Load flat G1 clips and clean stair G1 clips into one motion folder."""

    flat = MotionFolder.load(flat_motions)
    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("zarr is required for the clean stair archive") from error
    archive_path = Path(stairs_archive).expanduser().resolve()
    root = zarr.open_group(str(archive_path), mode="r")
    if (
        root.attrs.get("quaternion_convention") != "xyzw"
        or root.attrs.get("terrain_rotation_convention") != "wxyz"
    ):
        raise ContractError("unsupported clean stair quaternion convention")
    names = tuple(str(value) for value in root["clip_names"][:])
    total_clip_count = len(names)
    if stair_clip_limit is not None and stair_clip_indices is not None:
        raise ContractError("use either stair_clip_limit or stair_clip_indices")
    excluded_indices = frozenset(int(value) for value in stair_clip_exclude_indices)
    if (
        len(excluded_indices) != len(tuple(stair_clip_exclude_indices))
        or any(value < 0 or value >= total_clip_count for value in excluded_indices)
    ):
        raise ContractError("stair_clip_exclude_indices are invalid")
    if stair_clip_indices is not None:
        selected_indices = tuple(int(value) for value in stair_clip_indices)
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or any(value < 0 or value >= total_clip_count for value in selected_indices)
        ):
            raise ContractError("stair_clip_indices are invalid")
    elif stair_clip_limit is not None:
        if type(stair_clip_limit) is not int or stair_clip_limit <= 0:
            raise ContractError("stair_clip_limit must be a positive integer")
        selected_indices = tuple(range(min(total_clip_count, stair_clip_limit)))
    else:
        selected_indices = tuple(range(total_clip_count))
    selected_indices = tuple(
        value for value in selected_indices if value not in excluded_indices
    )
    if not selected_indices:
        raise ContractError("stair clip selection is empty after exclusions")
    clips = list(flat.clips)
    metadata: list[StairClipMetadata | None] = [None] * len(clips)
    digest = hashlib.sha256()
    digest.update(flat.inventory_sha256.encode("ascii"))
    digest.update(str(archive_path).encode("utf-8"))

    for archive_index in selected_indices:
        start = int(root["clip_start_idx"][archive_index])
        stop = int(root["clip_end_idx"][archive_index])
        frame_slice = slice(int(start), int(stop))
        body_xyzw = np.asarray(root["body_quat_w"][frame_slice], dtype=np.float32)
        body_wxyz = unroll_quaternions_wxyz(body_xyzw[..., (3, 0, 1, 2)])
        body_angular = angular_velocity_world_wxyz(body_wxyz, 50.0)
        relative_path = f"stairs500/{archive_index:04d}_{names[archive_index]}/motion.npz"
        clip = MotionClip(
            relative_path=relative_path,
            fps=50,
            joint_position=_readonly_float32(root["joint_pos"][frame_slice]),
            joint_velocity=_readonly_float32(root["joint_vel"][frame_slice]),
            body_position_world=_readonly_float32(root["body_pos_w"][frame_slice]),
            body_quaternion_world_wxyz=_readonly_float32(body_wxyz),
            body_linear_velocity_world=_readonly_float32(
                root["body_lin_vel_w"][frame_slice]
            ),
            body_angular_velocity_world=_readonly_float32(body_angular),
        )
        unified_index = len(clips)
        clips.append(clip)
        traversal = str(root["clip_traversal"][archive_index])
        physical_profile = measure_archive_stair_profile(root, archive_index)
        terrain_position = tuple(
            float(value) for value in root["terrain_position_env"][archive_index]
        )
        item = StairClipMetadata(
            unified_clip_index=unified_index,
            archive_clip_index=archive_index,
            name=names[archive_index],
            traversal=traversal,
            rise_m=float(root["stair_rise_m"][archive_index]),
            tread_m=float(root["stair_tread_m"][archive_index]),
            step_count=int(root["stair_n_steps"][archive_index]),
            travel_yaw_rad=float(root["travel_yaw_rad"][archive_index]),
            terrain_position_world_xyz=terrain_position,  # type: ignore[arg-type]
            terrain_usd_path=str(root["terrain_usd_path"][archive_index]),
            compatible_target_archive_index=archive_index,
            physical_rise_m=physical_profile.rise_m,
            physical_tread_m=physical_profile.tread_m,
            physical_level_count=physical_profile.level_count,
            physical_tread_edges_m=physical_profile.tread_edges_m,
            physical_tread_heights_m=physical_profile.tread_heights_m,
        )
        metadata.append(item)
        digest.update(relative_path.encode("utf-8"))
        digest.update(np.asarray((start, stop), dtype=np.int64).tobytes())
        digest.update(
            np.asarray(
                (item.rise_m, item.tread_m, item.step_count, item.travel_yaw_rad),
                dtype=np.float64,
            ).tobytes()
        )

    folder = MotionFolder(
        root=flat.root,
        layout=G1_TAKARA_LAYOUT,
        clips=tuple(clips),
        inventory_sha256=digest.hexdigest(),
    )
    return PrivilegedMotionLibrary(
        folder=folder,
        flat_clip_count=len(flat.clips),
        clip_metadata=tuple(metadata),
    )


def load_registered_source_library(
    target_archive_clip_indices: Sequence[int],
    registered_source_root: str | Path,
    *,
    flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
    stairs_archive: str | Path = DEFAULT_STAIRS_ARCHIVE,
    model_path: str | Path = DEFAULT_G1_MJCF,
) -> PrivilegedMotionLibrary:
    """Replace raw stair clips with mechanically accepted registered assets."""

    targets = tuple(int(value) for value in target_archive_clip_indices)
    base = load_privileged_motion_library(
        flat_motions,
        stairs_archive,
        stair_clip_indices=targets,
    )
    root = Path(registered_source_root).expanduser().resolve()
    clips = list(base.folder.clips)
    metadata = list(base.clip_metadata)
    fk = G1MujocoFK(model_path)
    digest = hashlib.sha256(base.folder.inventory_sha256.encode("ascii"))
    for target in targets:
        asset_root = root / f"target_{target}"
        report_path = asset_root / "summary.json"
        motion_path = asset_root / "motion.npz"
        if not report_path.is_file() or not motion_path.is_file():
            raise ContractError(
                f"registered source asset is incomplete for target {target}"
            )
        report = json.loads(report_path.read_text())
        if (
            report.get("status") != "accepted"
            or int(report.get("internal_animation_seams", -1)) != 0
            or int(report.get("flat_bridge_count", -1)) != 0
        ):
            raise ContractError(
                f"registered source asset is not accepted for target {target}"
            )
        with np.load(motion_path, allow_pickle=False) as motion:
            joint_position = np.asarray(
                motion["joint_position"], dtype=np.float32
            )
            root_position = np.asarray(
                motion["root_position_world"], dtype=np.float32
            )
            root_wxyz = np.asarray(
                motion["root_quaternion_world_wxyz"], dtype=np.float32
            )
            seam_indices = np.asarray(motion["seam_indices"])
        if (
            joint_position.ndim != 2
            or joint_position.shape[1] != 29
            or root_position.shape != (len(joint_position), 3)
            or root_wxyz.shape != (len(joint_position), 4)
            or seam_indices.size != 0
        ):
            raise ContractError(
                f"registered source asset shape is invalid for target {target}"
            )
        # Search features require a 0.9-second future window, which normally
        # removes the last 45 source frames from the executable database.  A
        # committed coherent stair phrase still needs those authored landing
        # frames (especially the trailing foot on a descent).  Append a
        # private held-pose horizon so every *original* frame 0..N-1 owns a
        # valid dense window.  The padding itself is not searchable and is
        # never published once the destination-support egress gate succeeds.
        tail = int(FEATURE_HORIZON_FRAMES[-1])
        joint_position = np.concatenate(
            (joint_position, np.repeat(joint_position[-1:], tail, axis=0)),
            axis=0,
        )
        root_position = np.concatenate(
            (root_position, np.repeat(root_position[-1:], tail, axis=0)),
            axis=0,
        )
        root_wxyz = np.concatenate(
            (root_wxyz, np.repeat(root_wxyz[-1:], tail, axis=0)),
            axis=0,
        )
        dof_mujoco = np.empty_like(joint_position)
        dof_mujoco[..., MUJOCO_TO_ISAACLAB_JOINT] = joint_position
        forward = fk.forward(
            root_position,
            root_wxyz[..., (1, 2, 3, 0)],
            dof_mujoco,
        )
        body_xyzw = np.asarray(
            forward.body_quaternion_world_xyzw, dtype=np.float32
        )
        body_wxyz = unroll_quaternions_wxyz(
            body_xyzw[..., (3, 0, 1, 2)]
        )
        unified_index = next(
            index
            for index, item in enumerate(metadata)
            if item is not None and item.archive_clip_index == target
        )
        clips[unified_index] = MotionClip(
            relative_path=(
                f"registered_stairs500/target{target:04d}/motion.npz"
            ),
            fps=50,
            joint_position=_readonly_float32(joint_position),
            joint_velocity=_readonly_float32(
                derive_clip_velocity(joint_position, fps=50.0)
            ),
            body_position_world=_readonly_float32(
                forward.body_position_world
            ),
            body_quaternion_world_wxyz=_readonly_float32(body_wxyz),
            body_linear_velocity_world=_readonly_float32(
                derive_clip_velocity(forward.body_position_world, fps=50.0)
            ),
            body_angular_velocity_world=_readonly_float32(
                angular_velocity_world_wxyz(body_wxyz, 50.0)
            ),
        )
        digest.update(str(target).encode("ascii"))
        digest.update(motion_path.read_bytes())
        digest.update(report_path.read_bytes())
    return PrivilegedMotionLibrary(
        folder=MotionFolder(
            root=base.folder.root,
            layout=base.folder.layout,
            clips=tuple(clips),
            inventory_sha256=digest.hexdigest(),
        ),
        flat_clip_count=base.flat_clip_count,
        clip_metadata=tuple(metadata),
    )


def build_target_specific_warped_library(
    target_archive_clip_index: int,
    *,
    flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
    stairs_archive: str | Path = DEFAULT_STAIRS_ARCHIVE,
    model_path: str | Path = DEFAULT_G1_MJCF,
    successful_warp_count: int = 1,
    candidate_limit: int = 12,
    include_target_clip: bool = True,
    reverse_target_traversal: bool = False,
    geometry_ranking: str = "route",
) -> tuple[PrivilegedMotionLibrary, WarpedLibraryBuildReport]:
    """Compile mechanically audited source styles onto one exact target mesh.

    This is the global-information analogue of a game's environment-specific
    animation cache.  It expands a target's one clean authored traverse with
    other G1 traverses after route-coordinate warping and bounded stance-foot
    IK.  Failed mechanical fits are skipped, never admitted to online search.
    """

    try:
        import zarr
    except ImportError as error:
        raise RuntimeError("zarr is required for target-specific warping") from error
    archive_path = Path(stairs_archive).expanduser().resolve()
    archive = zarr.open_group(str(archive_path), mode="r")
    target = int(target_archive_clip_index)
    archive_count = len(archive["clip_names"])
    if not 0 <= target < archive_count:
        raise ContractError("target archive clip index is out of range")
    if successful_warp_count <= 0 or candidate_limit <= 0:
        raise ContractError("warp counts must be positive")
    if reverse_target_traversal and include_target_clip:
        raise ContractError(
            "a reversed target geometry cannot include its forward target motion"
        )

    base = load_privileged_motion_library(
        flat_motions,
        archive_path,
        stair_clip_indices=(target,),
    )
    target_start = int(archive["clip_start_idx"][target])
    target_stop = int(archive["clip_end_idx"][target])
    target_root = np.asarray(
        archive["body_pos_w"][target_start:target_stop, 0],
        dtype=np.float32,
    )
    authored_target_traversal = str(archive["clip_traversal"][target])
    target_traversal = (
        (
            "down"
            if authored_target_traversal == "up"
            else "up"
        )
        if reverse_target_traversal
        else authored_target_traversal
    )
    target_steps = int(archive["stair_n_steps"][target])
    target_rise = float(archive["stair_rise_m"][target])
    target_tread = float(archive["stair_tread_m"][target])
    target_physical_profile = measure_archive_stair_profile(archive, target)
    target_route_start = target_root[0, :2]
    target_route_end = target_root[-1, :2]
    target_yaw = float(archive["travel_yaw_rad"][target])
    target_terrain_position = np.asarray(
        archive["terrain_position_env"][target], dtype=np.float64
    )
    target_profile_edges = target_physical_profile.tread_edges_m
    target_profile_heights = target_physical_profile.tread_heights_m
    if reverse_target_traversal:
        target_route_start, target_route_end = (
            target_route_end.copy(),
            target_route_start.copy(),
        )
        target_yaw = math.remainder(target_yaw + math.pi, 2.0 * math.pi)
        direction = np.asarray(
            (
                math.cos(float(archive["travel_yaw_rad"][target])),
                math.sin(float(archive["travel_yaw_rad"][target])),
            ),
            dtype=np.float64,
        )
        target_terrain_position = target_terrain_position.copy()
        target_terrain_position[:2] += (
            target_physical_profile.run_m * direction
        )
        target_profile_edges = tuple(
            float(target_physical_profile.run_m - value)
            for value in reversed(target_physical_profile.tread_edges_m)
        )
        target_profile_heights = tuple(
            reversed(target_physical_profile.tread_heights_m)
        )
    proposals = rank_geometry_compatible_sources(
        archive,
        target,
        ranking=geometry_ranking,
        reverse_target_traversal=reverse_target_traversal,
    )

    if include_target_clip:
        clips = list(base.folder.clips)
        metadata = list(base.clip_metadata)
    else:
        # Leave-one-target-out evaluation: retain the common flat bank but
        # require every terrain frame to come from a different G1 traverse
        # warped onto the globally known target staircase.
        clips = list(base.folder.clips[: base.flat_clip_count])
        metadata = [None] * base.flat_clip_count
    warped_start = len(clips)
    accepted: list[int] = []
    attempted: list[int] = []
    rejected: list[str] = []
    fk = G1MujocoFK(model_path)
    for _geometry_cost, source in proposals[:candidate_limit]:
        attempted.append(source)
        source_frames = int(archive["clip_end_idx"][source]) - int(
            archive["clip_start_idx"][source]
        )
        try:
            warped = warp_archive_clip_to_stair_geometry(
                archive_path,
                source_clip_index=source,
                source_start_frame=0,
                source_stop_frame=source_frames,
                target_clip_index=target,
                target_route_start_xy=target_route_start,
                target_route_end_xy=target_route_end,
                model_path=model_path,
                maximum_joint_correction_rad=0.50,
                maximum_foot_target_error_m=0.003,
                maximum_sole_penetration_m=0.006,
                maximum_root_clearance_lift_m=0.025,
                minimum_stance_support_points=1,
            )
        except ValueError as error:
            rejected.append(f"{source}: {error}")
            continue

        motion = warped.motion
        joint_position = np.asarray(motion.joint_position, dtype=np.float32)
        dof_mujoco = np.empty_like(joint_position)
        dof_mujoco[..., MUJOCO_TO_ISAACLAB_JOINT] = joint_position
        root_wxyz = np.asarray(
            motion.root_quaternion_world_wxyz, dtype=np.float32
        )
        forward = fk.forward(
            motion.root_position_world,
            root_wxyz[..., (1, 2, 3, 0)],
            dof_mujoco,
        )
        body_xyzw = np.asarray(
            forward.body_quaternion_world_xyzw, dtype=np.float32
        )
        body_wxyz = unroll_quaternions_wxyz(
            body_xyzw[..., (3, 0, 1, 2)]
        )
        source_name = str(archive["clip_names"][source])
        target_name = str(archive["clip_names"][target])
        relative_path = (
            f"warped_stairs500/src{source:04d}_to_"
            f"{'reverse_' if reverse_target_traversal else ''}"
            f"target{target:04d}_"
            f"{source_name}/motion.npz"
        )
        clip = MotionClip(
            relative_path=relative_path,
            fps=50,
            joint_position=_readonly_float32(joint_position),
            joint_velocity=_readonly_float32(
                derive_clip_velocity(joint_position, fps=50.0)
            ),
            body_position_world=_readonly_float32(
                forward.body_position_world
            ),
            body_quaternion_world_wxyz=_readonly_float32(body_wxyz),
            body_linear_velocity_world=_readonly_float32(
                derive_clip_velocity(forward.body_position_world, fps=50.0)
            ),
            body_angular_velocity_world=_readonly_float32(
                angular_velocity_world_wxyz(body_wxyz, 50.0)
            ),
        )
        unified_index = len(clips)
        clips.append(clip)
        terrain_position = tuple(float(value) for value in target_terrain_position)
        metadata.append(
            StairClipMetadata(
                unified_clip_index=unified_index,
                archive_clip_index=source,
                name=f"warp:{source_name}->{target_name}",
                traversal=target_traversal,
                rise_m=target_rise,
                tread_m=target_tread,
                step_count=target_steps,
                travel_yaw_rad=target_yaw,
                terrain_position_world_xyz=terrain_position,  # type: ignore[arg-type]
                terrain_usd_path=str(archive["terrain_usd_path"][target]),
                compatible_target_archive_index=target,
                physical_rise_m=target_physical_profile.rise_m,
                physical_tread_m=target_physical_profile.tread_m,
                physical_level_count=target_physical_profile.level_count,
                physical_tread_edges_m=(
                    target_profile_edges
                ),
                physical_tread_heights_m=(
                    target_profile_heights
                ),
            )
        )
        accepted.append(source)
        if len(accepted) >= successful_warp_count:
            break

    digest = hashlib.sha256()
    digest.update(base.folder.inventory_sha256.encode("ascii"))
    digest.update(str(target).encode("ascii"))
    for clip in clips[warped_start:]:
        digest.update(clip.relative_path.encode("utf-8"))
        digest.update(np.asarray(clip.joint_position).tobytes())
    library = PrivilegedMotionLibrary(
        folder=MotionFolder(
            root=base.folder.root,
            layout=base.folder.layout,
            clips=tuple(clips),
            inventory_sha256=digest.hexdigest(),
        ),
        flat_clip_count=base.flat_clip_count,
        clip_metadata=tuple(metadata),
    )
    report = WarpedLibraryBuildReport(
        target_archive_clip_index=target,
        accepted_source_indices=tuple(accepted),
        attempted_source_indices=tuple(attempted),
        rejection_reasons=tuple(rejected),
    )
    return library, report


def build_multi_target_warped_library(
    target_archive_clip_indices: Sequence[int],
    *,
    flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
    stairs_archive: str | Path = DEFAULT_STAIRS_ARCHIVE,
    model_path: str | Path = DEFAULT_G1_MJCF,
    successful_warp_count_per_target: int = 1,
    candidate_limit_per_target: int = 30,
    include_target_clips: bool = False,
) -> tuple[
    PrivilegedMotionLibrary,
    MultiTargetWarpedLibraryBuildReport,
]:
    """Compile several exact terrain portals into one shared search bank.

    Each terrain target is warped and mechanically audited independently.  The
    common flat bank appears only once; accepted terrain clips retain their
    compatible target identity so runtime search cannot use an animation on
    the wrong staircase.  This is the database layout needed by a game-style
    scene containing several globally known traversable objects.
    """

    targets = tuple(int(value) for value in target_archive_clip_indices)
    if not targets or len(set(targets)) != len(targets):
        raise ContractError("multi-target warp indices must be unique and non-empty")
    libraries: list[PrivilegedMotionLibrary] = []
    reports: list[WarpedLibraryBuildReport] = []
    for target in targets:
        library, report = build_target_specific_warped_library(
            target,
            flat_motions=flat_motions,
            stairs_archive=stairs_archive,
            model_path=model_path,
            successful_warp_count=successful_warp_count_per_target,
            candidate_limit=candidate_limit_per_target,
            include_target_clip=include_target_clips,
        )
        if len(report.accepted_source_indices) < successful_warp_count_per_target:
            raise ContractError(
                f"target {target} produced only "
                f"{len(report.accepted_source_indices)} accepted warps"
            )
        libraries.append(library)
        reports.append(report)

    first = libraries[0]
    flat_count = first.flat_clip_count
    clips = list(first.folder.clips[:flat_count])
    metadata: list[StairClipMetadata | None] = [None] * flat_count
    digest = hashlib.sha256()
    digest.update(first.folder.inventory_sha256.encode("ascii"))
    for target, library in zip(targets, libraries, strict=True):
        if library.flat_clip_count != flat_count:
            raise ContractError("multi-target libraries disagree on flat clip count")
        digest.update(str(target).encode("ascii"))
        for clip, item in zip(
            library.folder.clips[flat_count:],
            library.clip_metadata[flat_count:],
            strict=True,
        ):
            if item is None:
                raise ContractError("warped terrain clip has no metadata")
            unified_index = len(clips)
            clips.append(clip)
            metadata.append(replace(item, unified_clip_index=unified_index))
            digest.update(clip.relative_path.encode("utf-8"))
            digest.update(np.asarray(clip.joint_position).tobytes())

    return (
        PrivilegedMotionLibrary(
            folder=MotionFolder(
                root=first.folder.root,
                layout=first.folder.layout,
                clips=tuple(clips),
                inventory_sha256=digest.hexdigest(),
            ),
            flat_clip_count=flat_count,
            clip_metadata=tuple(metadata),
        ),
        MultiTargetWarpedLibraryBuildReport(tuple(reports)),
    )


def build_bidirectional_target_geometry_library(
    target_archive_clip_index: int,
    *,
    flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
    stairs_archive: str | Path = DEFAULT_STAIRS_ARCHIVE,
    model_path: str | Path = DEFAULT_G1_MJCF,
    successful_warp_count_per_direction: int = 1,
    candidate_limit_per_direction: int = 120,
    include_forward_target_clip: bool = True,
) -> tuple[
    PrivilegedMotionLibrary,
    MultiTargetWarpedLibraryBuildReport,
]:
    """Compile ascent and synthesized descent on one exact target mesh.

    GRAIL contains many non-uniform ascent meshes without a separately
    authored descent on the same geometry.  The geometry warper needs no
    target animation: reverse the target route, fit an independently authored
    descent to that exact USD, and retain it only if the full sole/mesh audit
    passes.
    """

    forward, forward_report = build_target_specific_warped_library(
        target_archive_clip_index,
        flat_motions=flat_motions,
        stairs_archive=stairs_archive,
        model_path=model_path,
        successful_warp_count=successful_warp_count_per_direction,
        candidate_limit=candidate_limit_per_direction,
        include_target_clip=include_forward_target_clip,
    )
    reverse, reverse_report = build_target_specific_warped_library(
        target_archive_clip_index,
        flat_motions=flat_motions,
        stairs_archive=stairs_archive,
        model_path=model_path,
        successful_warp_count=successful_warp_count_per_direction,
        candidate_limit=candidate_limit_per_direction,
        include_target_clip=False,
        reverse_target_traversal=True,
    )
    required = int(successful_warp_count_per_direction)
    forward_covered = bool(
        include_forward_target_clip
        or len(forward_report.accepted_source_indices) >= required
    )
    if (
        not forward_covered
        or len(reverse_report.accepted_source_indices) < required
    ):
        raise ContractError(
            "bidirectional target geometry did not produce enough audited "
            "warps in both directions; forward="
            f"{forward_report.accepted_source_indices}, reverse="
            f"{reverse_report.accepted_source_indices}, reverse_tail="
            f"{reverse_report.rejection_reasons[-3:]}"
        )
    flat_count = forward.flat_clip_count
    clips = list(forward.folder.clips[:flat_count])
    metadata: list[StairClipMetadata | None] = [None] * flat_count
    digest = hashlib.sha256()
    digest.update(forward.folder.inventory_sha256.encode("ascii"))
    digest.update(b"bidirectional-target-geometry-v1")
    for library in (forward, reverse):
        for clip, item in zip(
            library.folder.clips[flat_count:],
            library.clip_metadata[flat_count:],
            strict=True,
        ):
            if item is None:
                raise ContractError("bidirectional terrain clip has no metadata")
            unified_index = len(clips)
            clips.append(clip)
            metadata.append(replace(item, unified_clip_index=unified_index))
            digest.update(clip.relative_path.encode("utf-8"))
            digest.update(np.asarray(clip.joint_position).tobytes())
    return (
        PrivilegedMotionLibrary(
            folder=MotionFolder(
                root=forward.folder.root,
                layout=forward.folder.layout,
                clips=tuple(clips),
                inventory_sha256=digest.hexdigest(),
            ),
            flat_clip_count=flat_count,
            clip_metadata=tuple(metadata),
        ),
        MultiTargetWarpedLibraryBuildReport(
            (forward_report, reverse_report)
        ),
    )


class PrivilegedTerrainMotionMatcher:
    """Unified rolling matcher with an exact global staircase observation."""

    def __init__(
        self,
        library: PrivilegedMotionLibrary,
        *,
        device: str | torch.device = "auto",
        matcher_config: MatcherConfig = MatcherConfig(
            trajectory_model="takara_ball",
            max_source_joint_step_rad=0.25,
            # A rigid yaw correction rotates the whole character about its
            # pelvis even when both source feet are planted.  That is useful
            # for keeping a generic animation near a game trajectory, but it
            # creates an unmistakable foot swivel in this clean-kinematic
            # viewer.  Let selected turn motion provide character yaw; the
            # simulation heading remains the query/command target.
            adjustment_rotation_max_ratio=0.0,
            clamping_max_angle_rad=math.pi,
        ),
        terrain_config: PrivilegedMatcherConfig = PrivilegedMatcherConfig(),
    ) -> None:
        self.library = library
        self.config = terrain_config
        self.matcher = TorchMotionMatcher.from_motion_folder(
            library.folder,
            device=device,
            config=matcher_config,
        )
        self.device = self.matcher.device
        self._scene_key: tuple[object, ...] | None = None
        self._scene_completed = False
        self._stair_clip_age = 0
        self._last_stair_clip: int | None = None
        self._flat_clip_age = 0
        self._last_flat_clip: int | None = None
        self._terrain_lateral_offset_m = 0.0
        self._automatic_registered_entry_rows: dict[int, int] = {}
        self._build_row_metadata()

    @classmethod
    def from_paths(
        cls,
        flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
        stairs_archive: str | Path = DEFAULT_STAIRS_ARCHIVE,
        *,
        stair_clip_limit: int | None = None,
        stair_clip_indices: Sequence[int] | None = None,
        stair_clip_exclude_indices: Sequence[int] = (),
        device: str | torch.device = "auto",
        matcher_config: MatcherConfig = MatcherConfig(
            trajectory_model="takara_ball",
            max_source_joint_step_rad=0.25,
            adjustment_rotation_max_ratio=0.0,
            clamping_max_angle_rad=math.pi,
        ),
        terrain_config: PrivilegedMatcherConfig = PrivilegedMatcherConfig(),
    ) -> "PrivilegedTerrainMotionMatcher":
        return cls(
            load_privileged_motion_library(
                flat_motions,
                stairs_archive,
                stair_clip_limit=stair_clip_limit,
                stair_clip_indices=stair_clip_indices,
                stair_clip_exclude_indices=stair_clip_exclude_indices,
            ),
            device=device,
            matcher_config=matcher_config,
            terrain_config=terrain_config,
        )

    def _build_row_metadata(self) -> None:
        database = self.matcher.database
        row_clip = database._search_clip_index.detach().cpu().numpy().astype(np.int64)
        row_frame = database._search_frame_index.detach().cpu().numpy().astype(np.int64)
        count = len(row_clip)
        stair = np.zeros(count, dtype=bool)
        traversal = np.zeros(count, dtype=np.int8)
        rise = np.zeros(count, dtype=np.float32)
        tread = np.ones(count, dtype=np.float32)
        steps = np.zeros(count, dtype=np.float32)
        physical_rise = np.zeros(count, dtype=np.float32)
        physical_tread = np.ones(count, dtype=np.float32)
        physical_levels = np.zeros(count, dtype=np.float32)
        local_root = np.zeros((count, 3), dtype=np.float32)
        local_root_velocity = np.zeros((count, 3), dtype=np.float32)
        relative_facing = np.zeros(count, dtype=np.float32)
        contact_phase = np.zeros(count, dtype=np.int8)
        compatible_target = np.full(count, -1, dtype=np.int32)
        joint_position = np.zeros(
            (count, self.library.folder.layout.joint_count), dtype=np.float32
        )
        source_turn_delta = np.zeros(count, dtype=np.float32)
        source_swing_excursion = np.zeros(count, dtype=np.float32)
        source_planar_displacement = np.zeros(count, dtype=np.float32)
        clip_turn_delta = np.zeros(count, dtype=np.float32)
        clip_planar_displacement = np.zeros(count, dtype=np.float32)

        # A terrain mesh can extend beyond the part actually traversed by an
        # authored clip (for example, a four-step phrase registered to one
        # section of a longer staircase).  Completion therefore belongs to
        # the phrase's destination support, not blindly to the far edge of the
        # whole mesh.  Keep one local-frame destination reference per source
        # clip; the exact mesh remains authoritative for collision/repair.
        self._registered_exit_reference_by_clip: dict[
            int, tuple[float, float, float, float]
        ] = {}
        self._registered_authored_last_frame_by_clip: dict[int, int] = {}

        left = self.library.folder.layout.left_foot_body_index
        right = self.library.folder.layout.right_foot_body_index
        speed_threshold = self.config.contact_speed_threshold_mps
        for clip_index, clip in enumerate(self.library.folder.clips):
            rows = np.flatnonzero(row_clip == clip_index)
            if rows.size == 0:
                continue
            frames = row_frame[rows]
            joint_position[rows] = clip.joint_position[frames]
            # Long BONES "idle_turn" recordings contain both genuine planted
            # stepping turns and long idle tails.  Clip-level labels cannot
            # distinguish them.  Measure the actual 0.6-second future window
            # for every searchable row so a stationary yaw request can admit
            # only rows that rotate *and* lift a swing foot.
            turn_horizon = 30
            root = self.library.folder.layout.root_body_index
            clip_frames = int(clip.joint_position.shape[0])
            future_frames = np.minimum(frames + turn_horizon, clip_frames - 1)
            root_quaternion_all = clip.body_quaternion_world_wxyz[:, root]
            root_yaw_all = _yaw_wxyz_numpy(root_quaternion_all)
            root_yaw_unwrapped = np.unwrap(root_yaw_all)
            source_turn_delta[rows] = np.arctan2(
                np.sin(root_yaw_all[future_frames] - root_yaw_all[frames]),
                np.cos(root_yaw_all[future_frames] - root_yaw_all[frames]),
            ).astype(np.float32)
            root_position_all = np.asarray(
                clip.body_position_world[:, root, :2], dtype=np.float32
            )
            source_planar_displacement[rows] = np.linalg.norm(
                root_position_all[future_frames] - root_position_all[frames],
                axis=1,
            ).astype(np.float32)
            clip_turn_delta[rows] = np.float32(
                root_yaw_unwrapped[-1] - root_yaw_unwrapped[0]
            )
            clip_planar_displacement[rows] = np.float32(
                np.linalg.norm(root_position_all[-1] - root_position_all[0])
            )
            feet = (
                self.library.folder.layout.left_foot_body_index,
                self.library.folder.layout.right_foot_body_index,
            )
            foot_height = np.asarray(
                clip.body_position_world[:, feet, 2], dtype=np.float32
            )
            future_max = foot_height.copy()
            future_min = foot_height.copy()
            for offset in range(1, turn_horizon + 1):
                shifted = np.concatenate(
                    (foot_height[offset:], np.repeat(foot_height[-1:], offset, axis=0)),
                    axis=0,
                )
                future_max = np.maximum(future_max, shifted)
                future_min = np.minimum(future_min, shifted)
            source_swing_excursion[rows] = np.max(
                future_max[frames] - future_min[frames], axis=1
            ).astype(np.float32)
            speed = np.linalg.norm(
                clip.body_linear_velocity_world[frames][:, (left, right)], axis=-1
            )
            contacts = speed <= speed_threshold
            contact_phase[rows] = (
                contacts[:, 0].astype(np.int8)
                + 2 * contacts[:, 1].astype(np.int8)
            )
            item = self.library.clip_metadata[clip_index]
            if item is None:
                continue
            stair[rows] = True
            traversal[rows] = 1 if item.traversal == "up" else -1
            rise[rows] = item.rise_m
            tread[rows] = item.tread_m
            steps[rows] = float(item.step_count)
            physical_rise[rows] = item.physical_rise_m
            physical_tread[rows] = item.physical_tread_m
            physical_levels[rows] = float(item.physical_level_count)
            compatible_target[rows] = item.compatible_target_archive_index
            origin = np.asarray(item.terrain_position_world_xyz, dtype=np.float64)
            position = np.asarray(
                clip.body_position_world[frames, self.library.folder.layout.root_body_index],
                dtype=np.float64,
            )
            local_root[rows] = _rotate_z_numpy(
                position - origin,
                -item.travel_yaw_rad,
            ).astype(np.float32)
            root_velocity = np.asarray(
                clip.body_linear_velocity_world[
                    frames, self.library.folder.layout.root_body_index
                ],
                dtype=np.float64,
            )
            local_root_velocity[rows] = _rotate_z_numpy(
                root_velocity,
                -item.travel_yaw_rad,
            ).astype(np.float32)
            root_quaternion = clip.body_quaternion_world_wxyz[
                frames, self.library.folder.layout.root_body_index
            ]
            relative_facing[rows] = np.arctan2(
                np.sin(_yaw_wxyz_numpy(root_quaternion) - item.travel_yaw_rad),
                np.cos(_yaw_wxyz_numpy(root_quaternion) - item.travel_yaw_rad),
            ).astype(np.float32)

            # Registered sources own a private held-pose future horizon so
            # their last authored frames remain executable.  Exclude that
            # private horizon when estimating the actual destination.  A
            # short median is insensitive to final-frame mocap jitter while
            # retaining the support level and progress the phrase authored.
            authored_stop = int(clip.joint_position.shape[0])
            if clip.relative_path.startswith("registered_stairs500/"):
                authored_stop = max(
                    1, authored_stop - int(FEATURE_HORIZON_FRAMES[-1])
                )
                self._registered_authored_last_frame_by_clip[clip_index] = (
                    authored_stop - 1
                )
            reference_start = max(0, authored_stop - 20)
            reference_frames = slice(reference_start, authored_stop)
            origin = np.asarray(
                item.terrain_position_world_xyz, dtype=np.float64
            )
            root_reference_world = np.asarray(
                clip.body_position_world[
                    reference_frames,
                    self.library.folder.layout.root_body_index,
                ],
                dtype=np.float64,
            )
            root_reference_local = _rotate_z_numpy(
                root_reference_world - origin,
                -item.travel_yaw_rad,
            )
            feet_reference_world = np.asarray(
                clip.body_position_world[
                    reference_frames,
                    (left, right),
                ],
                dtype=np.float64,
            )
            feet_reference_local = _rotate_z_numpy(
                feet_reference_world - origin,
                -item.travel_yaw_rad,
            )
            root_exit = np.median(root_reference_local, axis=0)
            feet_exit = np.median(feet_reference_local, axis=0)
            self._registered_exit_reference_by_clip[clip_index] = (
                float(root_exit[0]),
                float(root_exit[2]),
                float(np.median(feet_exit[:, 2] - 0.035)),
                float(np.min(feet_exit[:, 0])),
            )

        tensor = lambda value, dtype=None: torch.as_tensor(
            value,
            dtype=dtype,
            device=self.device,
        )
        self._row_clip = tensor(row_clip, torch.long)
        self._row_frame = tensor(row_frame, torch.long)
        self._row_stair = tensor(stair, torch.bool)
        self._row_traversal = tensor(traversal, torch.int8)
        self._row_rise = tensor(rise)
        self._row_tread = tensor(tread)
        self._row_steps = tensor(steps)
        self._row_physical_rise = tensor(physical_rise)
        self._row_physical_tread = tensor(physical_tread)
        self._row_physical_levels = tensor(physical_levels)
        self._row_local_root = tensor(local_root)
        self._row_local_root_velocity = tensor(local_root_velocity)
        self._row_relative_facing = tensor(relative_facing)
        self._row_contact_phase = tensor(contact_phase, torch.int8)
        self._row_compatible_target = tensor(compatible_target, torch.int32)
        self._row_joint_position = tensor(joint_position)
        self._row_source_turn_delta = tensor(source_turn_delta)
        self._row_source_swing_excursion = tensor(source_swing_excursion)
        self._row_source_planar_displacement = tensor(
            source_planar_displacement
        )
        self._row_clip_turn_delta = tensor(clip_turn_delta)
        self._row_clip_planar_displacement = tensor(
            clip_planar_displacement
        )

    def scene_from_archive_clip(
        self,
        archive_clip_index: int,
        *,
        terrain_position_world_xyz: tuple[float, float, float] | None = None,
        travel_yaw_rad: float | None = None,
    ) -> PrivilegedStairScene:
        items = tuple(
            item
            for item in self.library.clip_metadata
            if item is not None and item.archive_clip_index == archive_clip_index
        )
        if len(items) != 1:
            raise ContractError("archive clip is not loaded in this motion library")
        item = items[0]
        return PrivilegedStairScene(
            traversal=item.traversal,
            rise_m=item.rise_m,
            tread_m=item.tread_m,
            step_count=item.step_count,
            travel_yaw_rad=(
                item.travel_yaw_rad
                if travel_yaw_rad is None
                else float(travel_yaw_rad)
            ),
            terrain_position_world_xyz=(
                item.terrain_position_world_xyz
                if terrain_position_world_xyz is None
                else terrain_position_world_xyz
            ),
            label=item.name,
            archive_clip_index=archive_clip_index,
            physical_run_m=(
                item.physical_tread_edges_m[-1]
                if item.physical_tread_edges_m
                else item.physical_tread_m * item.physical_level_count
            ),
            physical_height_m=(
                max(item.physical_tread_heights_m)
                if item.physical_tread_heights_m
                else item.physical_rise_m * item.physical_level_count
            ),
            physical_rise_m=item.physical_rise_m,
            physical_tread_m=item.physical_tread_m,
            physical_level_count=item.physical_level_count,
            physical_tread_edges_m=item.physical_tread_edges_m,
            physical_tread_heights_m=item.physical_tread_heights_m,
        )

    def reset_flat_at(
        self,
        root_position_world_xyz: tuple[float, float, float] = (0.0, 0.0, 0.78),
        root_yaw_world_rad: float = 0.0,
    ) -> MotionMatchResult:
        row = int(self.matcher.database.reset_row)
        clip_index, frame_index = self.matcher._source_for_row(row)
        if self.library.clip_metadata[clip_index] is not None:
            flat_rows = torch.nonzero(
                ~self._row_stair, as_tuple=False
            ).flatten()
            if flat_rows.numel() == 0:
                raise ContractError("unified library contains no flat reset row")
            row = int(flat_rows[0].item())
            clip_index, frame_index = self.matcher._source_for_row(row)
        clip = self.matcher._clips[clip_index]
        root = self.library.folder.layout.root_body_index
        source_position = clip.body_position[frame_index, root]
        source_yaw = _quat_yaw(clip.body_quaternion[frame_index, root])
        yaw_offset = float(
            _wrapped_angle(
                torch.tensor(root_yaw_world_rad, device=self.device)
                - source_yaw
            ).item()
        )
        rotated = _rotate_z(source_position, torch.tensor(yaw_offset, device=self.device))
        target = torch.tensor(
            root_position_world_xyz,
            dtype=torch.float32,
            device=self.device,
        )
        translation = target - rotated
        self._scene_key = None
        self._scene_completed = False
        self._stair_clip_age = 0
        self._last_stair_clip = None
        self._flat_clip_age = 0
        self._last_flat_clip = None
        self._terrain_lateral_offset_m = 0.0
        return self.matcher.reset_to_row(
            row,
            alignment=ForcedMotionAlignment(
                yaw_offset_rad=yaw_offset,
                translation_world_xyz=tuple(float(value) for value in translation.tolist()),
            ),
        )

    def _preview_query(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
    ) -> tuple[torch.Tensor, object, int | None]:
        state = self.matcher._state
        if state is None:
            raise ContractError("reset must be called before step")
        requested_v = torch.tensor(
            velocity_world_xy, dtype=torch.float32, device=self.device
        )
        requested_h = torch.tensor(
            heading_world_yaw, dtype=torch.float32, device=self.device
        )
        successor = self.matcher.database.row_for_source(
            state.clip_index, state.frame_index + 1
        )
        if self.matcher.config.trajectory_model == "takara_ball":
            prediction = predict_takara_ball_trajectory(
                state.simulation_position,
                state.simulation_velocity,
                state.simulation_acceleration,
                state.simulation_heading,
                state.simulation_heading_velocity,
                requested_v,
                requested_h,
                has_valid_successor=successor is not None,
                config=self.matcher.config,
            )
            shaped = prediction.command
        else:
            shaped = predict_command_trajectory(
                state.root_position[:2],
                state.shaped_velocity,
                state.shaped_heading,
                requested_v,
                requested_h,
                has_valid_successor=successor is not None,
                config=self.matcher.config,
            )
        layout = self.library.folder.layout
        feature_state = GeneratedFeatureState(
            root_position_world=state.root_position,
            root_orientation_world_wxyz=state.root_quaternion,
            root_linear_velocity_world=state.root_linear_velocity,
            left_foot_position_world=state.feature_body_position[
                layout.left_foot_body_index
            ],
            right_foot_position_world=state.feature_body_position[
                layout.right_foot_body_index
            ],
            left_foot_velocity_world=state.feature_body_velocity[
                layout.left_foot_body_index
            ],
            right_foot_velocity_world=state.feature_body_velocity[
                layout.right_foot_body_index
            ],
        )
        query = self.matcher.database.normalization.normalize(
            extract_query_features(feature_state, shaped.trajectory)
        )
        return query, shaped, successor

    def _terrain_local_state(
        self, scene: PrivilegedStairScene
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.matcher._state
        assert state is not None
        origin = torch.tensor(
            scene.terrain_position_world_xyz,
            dtype=torch.float32,
            device=self.device,
        )
        local_root = _rotate_z(
            state.root_position - origin,
            torch.tensor(-scene.travel_yaw_rad, device=self.device),
        )
        relative_facing = _wrapped_angle(
            _quat_yaw(state.root_quaternion)
            - torch.tensor(scene.travel_yaw_rad, device=self.device)
        )
        return local_root, relative_facing

    def _qualified_registered_entry_row(
        self, scene: PrivilegedStairScene
    ) -> int | None:
        if scene.archive_clip_index is None:
            return None
        entry_frames = dict(self.config.qualified_registered_entry_frames)
        frame = entry_frames.get(int(scene.archive_clip_index))
        target = int(scene.archive_clip_index)
        if frame is not None:
            candidates = (
                (self._row_compatible_target == target)
                & self._row_stair
                & (self._row_frame == int(frame))
            )
            rows = torch.nonzero(candidates, as_tuple=False).flatten()
            if rows.numel() != 1:
                raise ContractError(
                    "qualified registered stair entry must identify one source row"
                )
            return int(rows[0].item())

        cached = self._automatic_registered_entry_rows.get(target)
        if cached is not None:
            return cached

        # A globally registered source is a complete coherent phrase.  Its
        # usable entry is the moving part of the flat approach, before the
        # pelvis changes support level.  Restricting by support height is what
        # prevents an apparently similar mid-stair row being selected.
        expected_entry_root_z = 0.76 + (
            scene.height_m if scene.traversal == "down" else 0.0
        )
        candidates = (
            (self._row_compatible_target == target)
            & self._row_stair
            & (self._row_frame >= 3)
            & (
                self._row_frame
                <= int(self.config.automatic_registered_entry_max_frame)
            )
            & (
                torch.abs(
                    self._row_local_root[:, 2] - expected_entry_root_z
                )
                <= self.config.automatic_registered_entry_height_tolerance_m
            )
            & (
                torch.abs(self._row_relative_facing)
                <= self.config.maximum_transition_facing_error_rad
            )
            & (self._row_local_root_velocity[:, 0] > 0.05)
        )
        if scene.traversal == "up":
            candidates &= self._row_local_root[:, 0] <= 0.10
            preferred_progress = -0.25
        else:
            first_tread = float(scene.surface_edges_m[1])
            maximum_progress = min(0.60, max(0.35, first_tread + 0.15))
            candidates &= self._row_local_root[:, 0] <= maximum_progress
            preferred_progress = 0.25

        # ``torch.flatnonzero`` is not available in the cloc3 PyTorch build.
        # The mask is one-dimensional, so the first component of ``nonzero``
        # is the portable equivalent.
        rows = torch.nonzero(candidates, as_tuple=False).flatten()
        if rows.numel() == 0:
            raise ContractError(
                f"registered source {target} has no safe automatic entry row"
            )
        target_speed = float(
            self.config.automatic_registered_entry_target_speed_mps
        )
        score = torch.square(
            (
                self._row_local_root_velocity[rows, 0]
                - target_speed
            )
            / 0.25
        )
        score += torch.square(
            self._row_relative_facing[rows] / math.radians(15.0)
        )
        # Portal position must remain competitive with speed/facing for a
        # descent.  With a near-zero weight the shared bank chose a fast row
        # almost at the ledge; the flat collision capsule correctly stopped
        # ~19 cm earlier, so the two gates could never meet.  Ascents do not
        # have that ledge deadlock and retain their lower weight: forcing them
        # later moved tall target 431 away from its collision-clean portal.
        progress_weight = 0.10 if scene.traversal == "up" else 1.0
        score += progress_weight * torch.square(
            (self._row_local_root[rows, 0] - preferred_progress) / 0.40
        )
        selected = int(rows[int(torch.argmin(score).item())].item())
        self._automatic_registered_entry_rows[target] = selected
        return selected

    def registered_stair_entry_ready(
        self, scene: PrivilegedStairScene
    ) -> bool:
        """Whether the current flat pose can enter a retained stair phrase."""

        state = self.matcher._state
        if state is None:
            raise ContractError("reset must be called before stair entry query")
        if self.library.clip_metadata[state.clip_index] is not None:
            return True
        row = self._qualified_registered_entry_row(scene)
        if row is None:
            return True
        local_root, relative_facing = self._terrain_local_state(scene)
        source_relative_facing = self._row_relative_facing[row]
        facing_error = _wrapped_angle(
            relative_facing - source_relative_facing
        )
        return bool(
            abs(
                float(local_root[0].item())
                - float(self._row_local_root[row, 0].item())
            )
            <= self.config.maximum_registered_entry_progress_error_m
            and abs(float(local_root[1].item()))
            <= self.config.maximum_registered_entry_lateral_m
            and abs(float(facing_error.item()))
            <= self.config.maximum_transition_facing_error_rad
        )

    def shape_registered_stair_entry_velocity(
        self,
        scene: PrivilegedStairScene | None,
        velocity_world_xy: np.ndarray,
    ) -> np.ndarray:
        """Stage a moving flat pose at the registered stair phrase anchor.

        A generic centreline correction can reach the riser before its
        lateral correction is complete.  This controller targets progress
        and lateral position together in stair-local coordinates, then hands
        the original command back as soon as the exact entry gate is ready.
        It never moves a centered stick on the operator's behalf.
        """

        requested = np.asarray(velocity_world_xy, dtype=np.float64)
        if scene is None or float(np.linalg.norm(requested)) <= 0.06:
            return requested
        state = self.matcher._state
        if state is None:
            raise ContractError("reset must be called before stair entry shaping")
        if self.library.clip_metadata[state.clip_index] is not None:
            return requested
        row = self._qualified_registered_entry_row(scene)
        if row is None or self.registered_stair_entry_ready(scene):
            return requested

        local_root, relative_facing = self._terrain_local_state(scene)
        progress_error = float(
            self._row_local_root[row, 0].item() - local_root[0].item()
        )
        lateral_error = -float(local_root[1].item())
        limit = float(self.config.maximum_registered_entry_staging_speed_mps)
        progress_velocity = float(
            np.clip(
                self.config.registered_entry_progress_gain * progress_error,
                -limit,
                limit,
            )
        )
        if abs(float(relative_facing.item())) > (
            self.config.maximum_transition_facing_error_rad
        ):
            # Never walk farther into the riser while the body is side-on.
            # Backing up toward an overshot anchor remains allowed; forward
            # progress resumes after the authored stepping turn aligns yaw.
            progress_velocity = min(progress_velocity, 0.0)
        local_velocity = np.asarray(
            (
                progress_velocity,
                np.clip(
                    self.config.registered_entry_lateral_gain * lateral_error,
                    -limit,
                    limit,
                ),
                0.0,
            ),
            dtype=np.float64,
        )
        speed = float(np.linalg.norm(local_velocity[:2]))
        requested_speed = float(np.linalg.norm(requested))
        if speed > requested_speed > 0.0:
            local_velocity[:2] *= requested_speed / speed
        return _rotate_z_numpy(local_velocity, scene.travel_yaw_rad)[:2]

    def shape_registered_stair_entry_heading(
        self,
        scene: PrivilegedStairScene | None,
        heading_world_yaw: float,
    ) -> float:
        """Face a selected registered portal while staging its exact entry."""

        if scene is None:
            return float(heading_world_yaw)
        state = self.matcher._state
        if state is None:
            raise ContractError("reset must be called before stair entry shaping")
        if self.library.clip_metadata[state.clip_index] is not None:
            return float(heading_world_yaw)
        if self._qualified_registered_entry_row(scene) is None:
            return float(heading_world_yaw)
        if self.registered_stair_entry_ready(scene):
            return float(heading_world_yaw)
        row = self._qualified_registered_entry_row(scene)
        assert row is not None
        return float(
            math.remainder(
                scene.travel_yaw_rad
                + float(self._row_relative_facing[row].item()),
                2.0 * math.pi,
            )
        )

    def _stair_alignment(
        self,
        row: int,
        scene: PrivilegedStairScene,
        *,
        lateral_offset_m: float = 0.0,
    ) -> ForcedMotionAlignment:
        clip_index = int(self._row_clip[row].item())
        item = self.library.clip_metadata[clip_index]
        if item is None:
            raise ContractError("stair alignment requested for a flat row")
        yaw_offset = math.atan2(
            math.sin(scene.travel_yaw_rad - item.travel_yaw_rad),
            math.cos(scene.travel_yaw_rad - item.travel_yaw_rad),
        )
        source_origin = np.asarray(item.terrain_position_world_xyz, dtype=np.float64)
        target_origin = np.asarray(scene.terrain_position_world_xyz, dtype=np.float64)
        translation = target_origin - _rotate_z_numpy(source_origin, yaw_offset)
        translation += _rotate_z_numpy(
            np.asarray((0.0, float(lateral_offset_m), 0.0), dtype=np.float64),
            scene.travel_yaw_rad,
        )
        return ForcedMotionAlignment(
            yaw_offset_rad=yaw_offset,
            translation_world_xyz=tuple(float(value) for value in translation),
        )

    def _flat_transition_alignment(self, row: int) -> ForcedMotionAlignment:
        state = self.matcher._state
        assert state is not None
        clip_index, frame_index = self.matcher._source_for_row(row)
        clip = self.matcher._clips[clip_index]
        root = self.library.folder.layout.root_body_index
        source_position = clip.body_position[frame_index, root]
        source_yaw = _quat_yaw(clip.body_quaternion[frame_index, root])
        yaw_offset_tensor = _wrapped_angle(
            _quat_yaw(state.root_quaternion) - source_yaw
        )
        rotated = _rotate_z(source_position, yaw_offset_tensor)
        translation = state.root_position - rotated
        return ForcedMotionAlignment(
            yaw_offset_rad=float(yaw_offset_tensor.item()),
            translation_world_xyz=tuple(float(value) for value in translation.tolist()),
        )

    def _registered_scene_completed_now(
        self,
        scene: PrivilegedStairScene,
        local_root: torch.Tensor,
        *,
        ignore_authored_tail: bool = False,
    ) -> bool:
        """Test support-surface egress for a registered coherent phrase."""

        state = self.matcher._state
        assert state is not None
        source_reference = self._registered_exit_reference_by_clip.get(
            int(state.clip_index)
        )
        authored_last_frame = self._registered_authored_last_frame_by_clip.get(
            int(state.clip_index)
        )
        if (
            not ignore_authored_tail
            and
            authored_last_frame is not None
            and int(state.frame_index) < authored_last_frame - 5
        ):
            # Do not truncate a coherent source just because it first reaches
            # the destination support.  Its authored landing/settling tail is
            # part of the animation and is often what makes the handoff look
            # physical rather than like a clipped game animation.
            return False
        expected_exit_root_z = (
            (scene.height_m if scene.traversal == "up" else 0.0) + 0.76
        )
        expected_exit_sole_z = scene.height_m if scene.traversal == "up" else 0.0
        expected_exit_foot_x = scene.run_m
        scene_rows = self._row_compatible_target == (
            -1
            if scene.archive_clip_index is None
            else scene.archive_clip_index
        )
        source_exit_x = scene.run_m
        if bool(torch.any(scene_rows).item()):
            source_exit_x = float(
                torch.max(self._row_local_root[scene_rows, 0]).item()
            )
        if source_reference is not None:
            (
                source_exit_x,
                expected_exit_root_z,
                expected_exit_sole_z,
                expected_exit_foot_x,
            ) = source_reference
        exit_x = min(scene.run_m, source_exit_x) - 0.10
        local_x = float(local_root[0].item())
        local_z = float(local_root[2].item())
        feet = torch.stack(
            (
                state.feature_body_position[
                    self.library.folder.layout.left_foot_body_index
                ],
                state.feature_body_position[
                    self.library.folder.layout.right_foot_body_index
                ],
            )
        )
        origin = torch.tensor(
            scene.terrain_position_world_xyz,
            dtype=torch.float32,
            device=self.device,
        )
        local_feet = _rotate_z(
            feet - origin,
            torch.tensor(-scene.travel_yaw_rad, device=self.device),
        )
        sole_height = local_feet[:, 2] - 0.035
        foot_speed = torch.linalg.vector_norm(
            torch.stack(
                (
                    state.feature_body_velocity[
                        self.library.folder.layout.left_foot_body_index
                    ],
                    state.feature_body_velocity[
                        self.library.folder.layout.right_foot_body_index
                    ],
                )
            ),
            dim=1,
        )
        settled_support = bool(
            torch.all(
                torch.abs(sole_height - expected_exit_sole_z) <= 0.04
            ).item()
            and torch.all(
                foot_speed <= self.config.contact_speed_threshold_mps
            ).item()
        )
        if scene.traversal == "up":
            return bool(
                local_x >= exit_x
                and local_z
                >= expected_exit_root_z
                - self.config.maximum_registered_exit_height_error_m
                and settled_support
            )
        return bool(
            local_x >= exit_x
            and local_z
            <= expected_exit_root_z
            + self.config.maximum_registered_exit_height_error_m
            and settled_support
            and bool(
                torch.all(
                    local_feet[:, 0] >= expected_exit_foot_x - 0.08
                ).item()
            )
        )

    def step(
        self,
        velocity_world_xy: tuple[float, float],
        heading_world_yaw: float,
        *,
        scene: PrivilegedStairScene | None,
    ) -> PrivilegedMotionStep:
        state = self.matcher._state
        if state is None:
            raise ContractError("reset must be called before step")
        current_is_stair = (
            self.library.clip_metadata[state.clip_index] is not None
        )
        # A registered traversal is an authored phrase, not a nearest-neighbour
        # search at every 50 Hz sample.  Once committed, advance its exact
        # successor until the far support surface is reached.  Besides being
        # the intended game-style semantics, this avoids scoring the entire
        # flat+terrain database hundreds of times during each stair climb.
        if current_is_stair and scene is not None:
            scene_key = (
                scene.traversal,
                scene.rise_m,
                scene.tread_m,
                scene.step_count,
                scene.travel_yaw_rad,
                scene.terrain_position_world_xyz,
            )
            if scene_key != self._scene_key:
                self._scene_key = scene_key
                self._scene_completed = False
            local_root, _relative_facing = self._terrain_local_state(scene)
            if self._registered_scene_completed_now(scene, local_root):
                self._scene_completed = True
            successor = self.matcher.database.row_for_source(
                state.clip_index, state.frame_index + 1
            )
            compatible_successor = bool(
                not self._scene_completed
                and successor is not None
                and bool(self._row_stair[successor].item())
                and (
                    scene.archive_clip_index is None
                    or int(self._row_compatible_target[successor].item())
                    == int(scene.archive_clip_index)
                )
            )
            if compatible_successor:
                selected = int(successor)
                direction = (
                    math.cos(scene.travel_yaw_rad),
                    math.sin(scene.travel_yaw_rad),
                )
                prepared = self.matcher.prepare_step(
                    (0.45 * direction[0], 0.45 * direction[1]),
                    scene.travel_yaw_rad,
                    forced_row=selected,
                    forced_alignment=self._stair_alignment(
                        selected,
                        scene,
                        lateral_offset_m=self._terrain_lateral_offset_m,
                    ),
                )
                result = self.matcher.commit(prepared)
                selected_clip = int(self._row_clip[selected].item())
                if selected_clip == self._last_stair_clip:
                    self._stair_clip_age += 1
                else:
                    self._last_stair_clip = selected_clip
                    self._stair_clip_age = 1
                self._last_flat_clip = None
                self._flat_clip_age = 0
                return PrivilegedMotionStep(
                    result=result,
                    selected_row=selected,
                    mode="stair",
                    searched=False,
                    terrain_local_root_xyz=tuple(
                        float(value) for value in local_root.tolist()
                    ),
                )
        effective_velocity_world_xy = velocity_world_xy
        effective_heading_world_yaw = heading_world_yaw
        if current_is_stair and scene is not None:
            # A committed stair phrase is authored motion, not a continuously
            # deformable joystick trajectory.  Keep the internal command ball
            # advancing down the stair lane while operator commands are held
            # for the flat handoff at the far side.
            direction = (
                math.cos(scene.travel_yaw_rad),
                math.sin(scene.travel_yaw_rad),
            )
            effective_velocity_world_xy = (
                0.45 * direction[0],
                0.45 * direction[1],
            )
            effective_heading_world_yaw = scene.travel_yaw_rad
        query, shaped, successor = self._preview_query(
            effective_velocity_world_xy, effective_heading_world_yaw
        )
        current_row = self.matcher.database.row_for_source(
            state.clip_index, state.frame_index
        )
        features = self.matcher.database._search_features
        cost = torch.sum(torch.square(features - query), dim=1)
        allowed = torch.ones_like(self._row_stair)
        local_root: torch.Tensor | None = None
        forced_stair_entry: int | None = None

        if scene is None:
            self._scene_key = None
            self._scene_completed = False
            allowed &= ~self._row_stair
            mode = "flat"
        else:
            scene_key = (
                scene.traversal,
                scene.rise_m,
                scene.tread_m,
                scene.step_count,
                scene.travel_yaw_rad,
                scene.terrain_position_world_xyz,
            )
            if scene_key != self._scene_key:
                self._scene_key = scene_key
                self._scene_completed = False
            local_root, relative_facing = self._terrain_local_state(scene)
            scene_rows = self._row_compatible_target == (
                -1
                if scene.archive_clip_index is None
                else scene.archive_clip_index
            )
            completed_now = self._registered_scene_completed_now(
                scene, local_root
            )
            if completed_now:
                self._scene_completed = True
            # Re-arm if the caller deliberately returns to the entry side.
            if (
                self._scene_completed
                and float(local_root[0].item()) < -0.45
            ):
                self._scene_completed = False
            entry_x = -self.config.active_entry_margin_m
            if scene.archive_clip_index is not None:
                if bool(torch.any(scene_rows).item()):
                    entry_x = float(
                        torch.min(self._row_local_root[scene_rows, 0]).item()
                    ) - 0.10
            active = (
                not self._scene_completed
                and
                float(local_root[0].item())
                >= entry_x
                and float(local_root[0].item())
                <= scene.run_m + self.config.active_exit_margin_m
                and abs(float(local_root[1].item()))
                <= scene.width_m * 0.5 + 0.65
            )
            qualified_entry = self._qualified_registered_entry_row(scene)
            if (
                active
                and not current_is_stair
                and qualified_entry is not None
            ):
                if self.registered_stair_entry_ready(scene):
                    forced_stair_entry = qualified_entry
                else:
                    active = False
            if active:
                mode = "stair"
                code = 1 if scene.traversal == "up" else -1
                allowed &= self._row_stair
                allowed &= self._row_traversal == code
                if scene.archive_clip_index is not None:
                    allowed &= self._row_compatible_target == (
                        scene.archive_clip_index
                    )
                allowed &= torch.abs(self._row_rise - scene.rise_m) <= (
                    self.config.maximum_rise_error_m
                )
                allowed &= torch.abs(self._row_tread - scene.tread_m) <= (
                    self.config.maximum_tread_error_m
                )
                allowed &= torch.abs(self._row_steps - scene.step_count) <= (
                    self.config.maximum_step_count_error
                )
                target_stage = local_root[0] / scene.tread_m
                source_stage = self._row_local_root[:, 0] / self._row_tread
                cost = cost + self.config.stage_weight * torch.square(
                    source_stage - target_stage
                )
                cost = cost + self.config.lateral_weight * torch.square(
                    (self._row_local_root[:, 1] - local_root[1]) / 0.30
                )
                target_height_steps = local_root[2] / scene.rise_m
                source_height_steps = self._row_local_root[:, 2] / self._row_rise.clamp_min(0.05)
                cost = cost + self.config.height_weight * torch.square(
                    source_height_steps - target_height_steps
                )
                facing_error = _wrapped_angle(
                    self._row_relative_facing - relative_facing
                )
                cost = cost + self.config.relative_facing_weight * torch.square(
                    facing_error / 0.35
                )
                cost = cost + self.config.rise_weight * torch.square(
                    (self._row_rise - scene.rise_m) / 0.03
                )
                cost = cost + self.config.tread_weight * torch.square(
                    (self._row_tread - scene.tread_m) / 0.08
                )
                cost = cost + self.config.step_count_weight * torch.square(
                    self._row_steps - scene.step_count
                )
                # Root proximity is a transition gate, not a playback gate.
                # Once a terrain clip is active, its authored successor must
                # remain admissible even while the entry inertializer decays.
                # Applying this mask to the successor caused ten-frame stair
                # excursions followed by a flat fallback on oblique entries.
                terrain_compatible_before_offset = allowed.clone()
                transition_forward_offset = torch.abs(
                    self._row_local_root[:, 0] - local_root[0]
                )
                transition_vertical_offset = torch.abs(
                    self._row_local_root[:, 2] - local_root[2]
                )
                transition_lateral_shift = torch.abs(
                    self._row_local_root[:, 1] - local_root[1]
                )
                allowed &= transition_forward_offset <= (
                    self.config.maximum_transition_forward_offset_m
                )
                allowed &= transition_vertical_offset <= (
                    self.config.maximum_transition_vertical_offset_m
                )
                allowed &= transition_lateral_shift <= (
                    self.config.maximum_transition_lateral_shift_m
                )
                allowed &= torch.abs(facing_error) <= (
                    self.config.maximum_transition_facing_error_rad
                )
                if (
                    current_is_stair
                    and successor is not None
                    and bool(terrain_compatible_before_offset[successor].item())
                ):
                    allowed[successor] = True
                if not bool(torch.any(allowed).item()):
                    # The operator has not reached a kinematically compatible
                    # entry yet.  Preserve the smooth flat approach instead of
                    # forcing a large inertialized terrain transition.
                    mode = "flat"
                    allowed = ~self._row_stair
            else:
                mode = "flat"
                allowed &= ~self._row_stair

        stationary_turn = False
        moving_turn = False
        turn_error = torch.zeros((), device=self.device)
        current_turn_compatible = False
        if mode == "flat":
            character_yaw = _quat_yaw(state.root_quaternion)
            # Classify operator intent from the actual joystick request.  The
            # critically damped command ball intentionally lags it; using the
            # lagged heading here let several frames of an unrelated walk play
            # before a stationary turn was even recognised.
            turn_error = _wrapped_angle(
                torch.tensor(
                    heading_world_yaw,
                    dtype=torch.float32,
                    device=self.device,
                )
                - character_yaw
            )
            requested_speed = float(np.linalg.norm(velocity_world_xy))
            stationary_turn = (
                requested_speed <= 0.08
                and abs(float(turn_error.item())) >= math.radians(7.5)
            )
            moving_turn = (
                requested_speed > 0.08
                and abs(float(turn_error.item())) >= math.radians(7.5)
            )
            if current_row is not None and not current_is_stair:
                local_turn = self._row_source_turn_delta[current_row]
                local_travel = float(
                    self._row_source_planar_displacement[current_row].item()
                )
                current_turn_compatible = bool(
                    (local_turn * turn_error > 0.0).item()
                    and abs(float(local_turn.item())) >= math.radians(2.0)
                    and (
                        not moving_turn
                        or local_travel >= 0.06
                    )
                    and (
                        not stationary_turn
                        or (
                            local_travel <= 0.12
                            and float(
                                self._row_source_swing_excursion[
                                    current_row
                                ].item()
                            )
                            >= 0.010
                        )
                    )
                )
            if stationary_turn:
                # Reject idle portions of clips whose filename says "turn".
                # Both a same-sign source yaw change and a real swing-foot
                # excursion are required, preventing command-board yaw from
                # masquerading as a stepping rotation.
                turn_candidates = (
                    self._row_source_turn_delta * turn_error > 0.0
                )
                turn_candidates &= torch.abs(self._row_source_turn_delta) >= (
                    math.radians(4.0)
                )
                turn_candidates &= self._row_source_swing_excursion >= 0.015
                turn_candidates &= (
                    self._row_source_planar_displacement <= 0.12
                )
                constrained = allowed & turn_candidates
                if bool(torch.any(constrained).item()):
                    allowed = constrained
                    target_delta = torch.clamp(
                        turn_error,
                        min=-math.radians(45.0),
                        max=math.radians(45.0),
                    )
                    cost = cost + 2.0 * torch.square(
                        (self._row_source_turn_delta - target_delta)
                        / math.radians(35.0)
                    )
            elif moving_turn:
                # Walking turns must come from an authored translating arc or
                # turn-start, not from an idle pivot with its root rigidly
                # dragged along the command path.
                turn_candidates = (
                    self._row_source_turn_delta * turn_error > 0.0
                )
                turn_candidates &= torch.abs(
                    self._row_source_turn_delta
                ) >= math.radians(2.0)
                turn_candidates &= (
                    self._row_source_planar_displacement >= 0.10
                )
                constrained = allowed & turn_candidates
                if bool(torch.any(constrained).item()):
                    allowed = constrained
                    target_delta = torch.clamp(
                        turn_error,
                        min=-math.radians(45.0),
                        max=math.radians(45.0),
                    )
                    cost = cost + 1.5 * torch.square(
                        (self._row_source_turn_delta - target_delta)
                        / math.radians(35.0)
                    )

            # Candidate filters describe where a new phrase may start.  They
            # must not discard the successor of an already committed,
            # direction-compatible turn; doing so was the direct cause of
            # the every-frame clip thrash seen in the live viewer.
            if (
                successor is not None
                and not current_is_stair
                and not bool(self._row_stair[successor].item())
                and (
                    not (stationary_turn or moving_turn)
                    or current_turn_compatible
                )
            ):
                allowed[successor] = True

        current_phase = (
            None if current_row is None else int(self._row_contact_phase[current_row].item())
        )
        due = search_is_due(
            state.sequence,
            bool(shaped.force_search),
            self.matcher.config,
        )
        successor_allowed = (
            successor is not None and bool(allowed[successor].item())
        )
        stair_dwell = (
            current_is_stair
            and self._stair_clip_age
            < self.config.minimum_stair_clip_dwell_steps
        )
        flat_dwell_limit = (
            self.config.minimum_turn_clip_dwell_steps
            if stationary_turn or moving_turn
            else self.config.minimum_flat_clip_dwell_steps
        )
        flat_dwell = (
            not current_is_stair
            and successor_allowed
            and self._flat_clip_age < flat_dwell_limit
            and (
                not bool(shaped.force_search)
                or (
                    (stationary_turn or moving_turn)
                    and current_turn_compatible
                )
            )
            and (
                not (stationary_turn or moving_turn)
                or current_turn_compatible
            )
        )
        if forced_stair_entry is not None:
            selected = int(forced_stair_entry)
            searched = True
        elif (not due or stair_dwell or flat_dwell) and successor_allowed:
            selected = int(successor)
            searched = False
        else:
            searched = True
            joint_delta = (
                self._row_joint_position - state.joint_position.unsqueeze(0)
            ) / 0.35
            cost = cost + self.config.joint_pose_weight * torch.mean(
                torch.square(joint_delta), dim=1
            )
            if current_phase is not None and current_phase != 0:
                phase_match = self._row_contact_phase == current_phase
                allowed &= phase_match
            elif successor_allowed:
                # Never jump out of flight; finish the authored support transfer.
                selected = int(successor)
                allowed = torch.zeros_like(allowed)
                allowed[selected] = True
            if current_row is not None:
                same_clip = self._row_clip == state.clip_index
                nearby = torch.abs(self._row_frame - state.frame_index) <= (
                    self.matcher.config.exclusion_frames
                )
                allowed &= ~(same_clip & nearby)
            if successor_allowed:
                allowed[successor] = True
            if not bool(torch.any(allowed).item()):
                if successor is None:
                    raise ContractError("terrain matcher has no admissible candidate")
                allowed[successor] = True
            cost = cost + self.config.transition_penalty
            if successor_allowed:
                cost[successor] -= self.config.transition_penalty
            cost = cost.masked_fill(~allowed, torch.inf)
            selected = int(torch.argmin(cost).item())

        selected_stair = bool(self._row_stair[selected].item())
        # Report the mode of the row we will actually render.  During a safe
        # entry fallback the scene can be active while the admissible
        # successor is still flat; labelling that frame as stair caused the
        # course controller to latch a portal before a terrain clip began.
        mode = "stair" if selected_stair else "flat"
        if selected_stair:
            if scene is None:
                raise ContractError("stair row selected without a stair scene")
            selected_clip = int(self._row_clip[selected].item())
            if not current_is_stair:
                assert local_root is not None
                self._terrain_lateral_offset_m = float(
                    local_root[1].item()
                    - self._row_local_root[selected, 1].item()
                )
            alignment = self._stair_alignment(
                selected,
                scene,
                lateral_offset_m=self._terrain_lateral_offset_m,
            )
        else:
            selected_clip = int(self._row_clip[selected].item())
            transitioning = (
                successor is None
                or selected != successor
                or self.library.clip_metadata[state.clip_index] is not None
                or selected_clip != state.clip_index
            )
            alignment = self._flat_transition_alignment(selected) if transitioning else None

        if selected_stair and scene is not None:
            direction = (
                math.cos(scene.travel_yaw_rad),
                math.sin(scene.travel_yaw_rad),
            )
            effective_velocity_world_xy = (
                0.45 * direction[0],
                0.45 * direction[1],
            )
            effective_heading_world_yaw = scene.travel_yaw_rad
        prepared = self.matcher.prepare_step(
            effective_velocity_world_xy,
            effective_heading_world_yaw,
            forced_row=selected,
            forced_alignment=alignment,
        )
        result = self.matcher.commit(prepared)
        selected_clip = int(self._row_clip[selected].item())
        if selected_stair:
            if selected_clip == self._last_stair_clip:
                self._stair_clip_age += 1
            else:
                self._last_stair_clip = selected_clip
                self._stair_clip_age = 1
        else:
            self._last_stair_clip = None
            self._stair_clip_age = 0
            self._terrain_lateral_offset_m = 0.0
        if not selected_stair:
            if (
                selected_clip == self._last_flat_clip
                and not bool(result.diagnostics.transitioned)
            ):
                self._flat_clip_age += 1
            else:
                self._last_flat_clip = selected_clip
                self._flat_clip_age = 1
        else:
            self._last_flat_clip = None
            self._flat_clip_age = 0
        local_tuple = (
            None
            if local_root is None
            else tuple(float(value) for value in local_root.tolist())
        )
        return PrivilegedMotionStep(
            result=result,
            selected_row=selected,
            mode=mode,
            searched=searched,
            terrain_local_root_xyz=local_tuple,  # type: ignore[arg-type]
        )
