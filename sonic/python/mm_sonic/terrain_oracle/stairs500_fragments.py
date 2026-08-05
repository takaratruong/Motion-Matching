"""Turn clean stairs500 archive clips into composable support fragments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path

import numpy as np

from .canonical import (
    CANONICAL_COORDINATE_CONVENTION,
    CLEAN_POSE_ORIGIN,
    CanonicalClip,
    CommandTrack,
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    SourceIdentity,
    TerrainBinding,
)
from .contact import (
    CONTACT_PLACEHOLDER_TAG,
    CanonicalMeshQuery,
    ContactConfig,
    SoleGeometry,
    reconstruct_contacts,
)
from .fragments import (
    FragmentDescriptor,
    FragmentRef,
    SegmentationConfig,
    SupportPhase,
    extract_fragment_descriptor,
    segment_clip,
)
from .math3d import (
    RigidTransform,
    angular_velocity_world_wxyz,
    unroll_quaternions_wxyz,
)
from .source_grail import _load_usd_mesh
from .terrain_mesh import TerrainMeshIndex
from ..joints import ContractError
from ..stair_mesh_profile import measure_archive_stair_profile


class FragmentRole(str, Enum):
    ENTRY = "entry"
    MIDDLE = "middle"
    EXIT = "exit"


@dataclass(frozen=True)
class Stairs500ClipMetadata:
    archive_clip_index: int
    clip_name: str
    family: str
    traversal: str
    rise_m: float
    tread_m: float
    step_count: int
    travel_yaw_rad: float
    terrain_position_world: tuple[float, float, float]
    terrain_rotation_world_wxyz: tuple[float, float, float, float]
    source_robot_path: str
    terrain_usd_path: str
    physical_tread_edges_m: tuple[float, ...] = ()
    physical_tread_heights_m: tuple[float, ...] = ()


@dataclass(frozen=True)
class StairFragmentRecord:
    fragment_id: str
    archive_clip_index: int
    source_clip_name: str
    source_start_frame: int
    source_stop_frame: int
    context_start_frame: int
    context_stop_frame: int
    entry_phase: SupportPhase
    exit_phase: SupportPhase
    role: FragmentRole
    segmentation_basis: str
    start_tread_index: int | None
    end_tread_index: int | None
    riser_transition_count: int
    entry_contact_confirmation: float
    exit_contact_confirmation: float
    entry_kinematic_support_evidence: bool
    exit_kinematic_support_evidence: bool
    source_direction: str
    source_rise_m: float
    source_tread_m: float
    source_step_count: int
    duration_s: float
    root_displacement_local_xyz: tuple[float, float, float]
    terrain_displacement_local_xyz: tuple[float, float, float]
    vertical_delta_m: float
    approximate_run_m: float
    approximate_lateral_m: float
    approximate_yaw_rad: float
    start_speed_local_xyz: tuple[float, float, float]
    end_speed_local_xyz: tuple[float, float, float]
    start_yaw_rate_rad_s: float
    end_yaw_rate_rad_s: float
    source_robot_path: str
    terrain_usd_path: str
    descriptor: FragmentDescriptor

    def to_dict(self) -> dict[str, object]:
        return {
            "fragment_id": self.fragment_id,
            "archive_clip_index": self.archive_clip_index,
            "source_clip_name": self.source_clip_name,
            "source_range": [self.source_start_frame, self.source_stop_frame],
            "context_range": [self.context_start_frame, self.context_stop_frame],
            "entry_phase": self.entry_phase.value,
            "exit_phase": self.exit_phase.value,
            "role": self.role.value,
            "segmentation_basis": self.segmentation_basis,
            "start_tread_index": self.start_tread_index,
            "end_tread_index": self.end_tread_index,
            "riser_transition_count": self.riser_transition_count,
            "entry_contact_confirmation": self.entry_contact_confirmation,
            "exit_contact_confirmation": self.exit_contact_confirmation,
            "entry_kinematic_support_evidence": (
                self.entry_kinematic_support_evidence
            ),
            "exit_kinematic_support_evidence": (
                self.exit_kinematic_support_evidence
            ),
            "source_direction": self.source_direction,
            "source_rise_m": self.source_rise_m,
            "source_tread_m": self.source_tread_m,
            "source_step_count": self.source_step_count,
            "duration_s": self.duration_s,
            "root_displacement_local_xyz": list(
                self.root_displacement_local_xyz
            ),
            "terrain_displacement_local_xyz": list(
                self.terrain_displacement_local_xyz
            ),
            "vertical_delta_m": self.vertical_delta_m,
            "approximate_run_m": self.approximate_run_m,
            "approximate_lateral_m": self.approximate_lateral_m,
            "approximate_yaw_rad": self.approximate_yaw_rad,
            "start_speed_local_xyz": list(self.start_speed_local_xyz),
            "end_speed_local_xyz": list(self.end_speed_local_xyz),
            "start_yaw_rate_rad_s": self.start_yaw_rate_rad_s,
            "end_yaw_rate_rad_s": self.end_yaw_rate_rad_s,
            "source_robot_path": self.source_robot_path,
            "terrain_usd_path": self.terrain_usd_path,
            "descriptor": _descriptor_dict(self.descriptor),
        }


def _endpoint_dict(endpoint: object) -> dict[str, object]:
    return {
        "phase": endpoint.phase.value,  # type: ignore[attr-defined]
        "root_position_local_xyz": list(endpoint.root_position_local_xyz),  # type: ignore[attr-defined]
        "root_quaternion_local_wxyz": list(endpoint.root_quaternion_local_wxyz),  # type: ignore[attr-defined]
        "root_height_m": endpoint.root_height_m,  # type: ignore[attr-defined]
        "root_linear_velocity_local_xyz": list(endpoint.root_linear_velocity_local_xyz),  # type: ignore[attr-defined]
        "root_angular_velocity_local_xyz": list(endpoint.root_angular_velocity_local_xyz),  # type: ignore[attr-defined]
        "joint_position": list(endpoint.joint_position),  # type: ignore[attr-defined]
        "joint_velocity": list(endpoint.joint_velocity),  # type: ignore[attr-defined]
        "sole_position_local_xyz": [
            list(value) for value in endpoint.sole_position_local_xyz  # type: ignore[attr-defined]
        ],
    }


def _descriptor_dict(descriptor: FragmentDescriptor) -> dict[str, object]:
    return {
        "fragment_id": descriptor.fragment_id,
        "root_displacement_local_xyz": list(
            descriptor.root_displacement_local_xyz
        ),
        "root_yaw_delta_rad": descriptor.root_yaw_delta_rad,
        "duration_s": descriptor.duration_s,
        "entry": _endpoint_dict(descriptor.entry),
        "exit": _endpoint_dict(descriptor.exit),
        "contact_phases": [
            {
                "phase": span.phase.value,
                "start_offset": span.start_offset,
                "stop_offset": span.stop_offset,
            }
            for span in descriptor.contact_phases
        ],
        "terrain_tags": list(descriptor.terrain_tags),
        "action_tags": list(descriptor.action_tags),
    }


@dataclass(frozen=True)
class _SemanticRef:
    ref: FragmentRef
    basis: str
    role: FragmentRole | None
    start_tread_index: int | None
    end_tread_index: int | None
    entry_confirmation: float
    exit_confirmation: float
    entry_kinematic_evidence: bool
    exit_kinematic_evidence: bool


@dataclass(frozen=True)
class _TreadEvent:
    frame: int
    run_start: int
    run_stop: int
    foot: int
    tread_index: int
    phase: SupportPhase
    confirmation: float


def classify_fragment_role(
    *,
    midpoint_height_m: float,
    low_height_m: float,
    high_height_m: float,
    traversal: str,
) -> FragmentRole:
    span = max(float(high_height_m) - float(low_height_m), 1.0e-6)
    if traversal == "down":
        progress = (float(high_height_m) - float(midpoint_height_m)) / span
    else:
        progress = (float(midpoint_height_m) - float(low_height_m)) / span
    if progress <= 0.25:
        return FragmentRole.ENTRY
    if progress >= 0.75:
        return FragmentRole.EXIT
    return FragmentRole.MIDDLE


def _terrain_delta(
    clip: CanonicalClip,
    ref: FragmentRef,
    metadata: Stairs500ClipMetadata,
) -> tuple[float, float, float]:
    transform = RigidTransform(
        np.asarray(metadata.terrain_position_world, dtype=np.float32),
        np.asarray(metadata.terrain_rotation_world_wxyz, dtype=np.float32),
    ).inverse()
    frames = (ref.source_start_frame, ref.source_stop_frame - 1)
    local = transform.apply_points(clip.root_position_world[list(frames)])
    delta = np.asarray(local[1], dtype=np.float64) - np.asarray(
        local[0], dtype=np.float64
    )
    return tuple(float(value) for value in delta)  # type: ignore[return-value]


def _phase_at(clip: CanonicalClip, frame: int) -> SupportPhase:
    left, right = np.asarray(clip.contact[frame]) >= 0.5
    if left:
        return SupportPhase.DOUBLE if right else SupportPhase.LEFT
    return SupportPhase.RIGHT if right else SupportPhase.FLIGHT


def _make_ref(
    clip: CanonicalClip,
    start: int,
    stop: int,
    entry_phase: SupportPhase,
    exit_phase: SupportPhase,
    config: SegmentationConfig,
    label: str,
) -> FragmentRef:
    return FragmentRef(
        fragment_id=(
            f"{clip.clip_id}[{start}:{stop})/"
            f"{entry_phase.value}>{exit_phase.value}/{label}"
        ),
        clip_id=clip.clip_id,
        source_start_frame=start,
        source_stop_frame=stop,
        context_start_frame=max(0, start - config.context_before),
        context_stop_frame=min(
            clip.frame_count, stop + config.context_after
        ),
        entry_phase=entry_phase,
        exit_phase=exit_phase,
    )


def _stable_tread_events(
    clip: CanonicalClip,
    metadata: Stairs500ClipMetadata,
    config: SegmentationConfig,
) -> tuple[CanonicalClip, np.ndarray, tuple[_TreadEvent, ...]]:
    transform = RigidTransform(
        np.asarray(metadata.terrain_position_world, dtype=np.float32),
        np.asarray(metadata.terrain_rotation_world_wxyz, dtype=np.float32),
    ).inverse()
    sole = transform.apply_points(
        np.asarray(clip.sole_position_world).reshape((-1, 3))
    ).reshape((clip.frame_count, 2, 3))
    height = np.asarray(sole[:, :, 2], dtype=np.float64)
    base_height = float(np.quantile(height, 0.02))
    physical_heights = np.asarray(
        metadata.physical_tread_heights_m, dtype=np.float64
    )
    if len(physical_heights):
        # GRAIL stair metadata describes the generator family, not necessarily
        # the realized mesh.  Some meshes have strongly non-uniform rises, so
        # dividing sole height by the nominal rise silently merges distinct
        # treads.  Match support poses to the exact ray-cast tread heights.
        levels = np.unique(
            np.concatenate(
                (np.asarray((base_height,), dtype=np.float64), physical_heights)
            )
        )
        levels.sort()
        level_spacing = np.diff(levels)
        minimum_spacing = float(np.min(level_spacing))
        height_tolerance = max(0.02, min(0.045, 0.30 * minimum_spacing))
        height_error = np.abs(height[..., None] - levels[None, None, :])
        tread_index = np.argmin(height_error, axis=-1).astype(np.int64)
        near_physical_tread = (
            np.take_along_axis(
                height_error, tread_index[..., None], axis=-1
            )[..., 0]
            <= height_tolerance
        )
    else:
        tread_index = np.rint(
            (height - base_height) / max(metadata.rise_m, 1.0e-6)
        ).astype(np.int64)
        tread_index = np.clip(tread_index, 0, metadata.step_count)
        near_physical_tread = np.ones_like(tread_index, dtype=np.bool_)
    vertical_speed = np.abs(np.gradient(height, axis=0) * clip.fps)
    ankle_speed = np.linalg.norm(
        np.asarray(clip.body_linear_velocity_world[:, (18, 19)]),
        axis=-1,
    )
    stable = (
        (vertical_speed <= 0.25)
        & (ankle_speed < 0.10)
        & near_physical_tread
    )
    minimum_frames = max(3, config.stable_frames)
    candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
    kinematic_support = np.zeros(
        (clip.frame_count, 2), dtype=np.bool_
    )
    for foot in range(2):
        start = 0
        key = (int(tread_index[0, foot]), bool(stable[0, foot]))
        for frame in range(1, clip.frame_count + 1):
            next_key = (
                None
                if frame == clip.frame_count
                else (
                    int(tread_index[frame, foot]),
                    bool(stable[frame, foot]),
                )
            )
            if next_key == key:
                continue
            if key[1] and frame - start >= minimum_frames:
                candidates.setdefault((foot, key[0]), []).append(
                    (start, frame)
                )
                kinematic_support[start:frame, foot] = True
            if frame < clip.frame_count:
                start = frame
                key = next_key  # type: ignore[assignment]
    by_index: dict[int, list[tuple[int, int, int]]] = {}
    for (foot, index), runs in candidates.items():
        by_index.setdefault(index, []).extend(
            (run_start, run_stop, foot) for run_start, run_stop in runs
        )
    ordered_indices = sorted(by_index, reverse=metadata.traversal == "down")
    best_by_index: dict[int, tuple[int, int, int]] = {}
    for position, index in enumerate(ordered_indices):
        runs = by_index[index]
        if len(physical_heights) and position == 0:
            # Preserve the authored approach rather than selecting a later,
            # slightly longer stance on the same broad ground/landing level.
            best_by_index[index] = min(
                runs, key=lambda value: (value[0], -(value[1] - value[0]))
            )
        elif len(physical_heights) and position == len(ordered_indices) - 1:
            # Preserve the authored settle on the final landing.
            best_by_index[index] = max(
                runs, key=lambda value: (value[1], value[1] - value[0])
            )
        elif len(physical_heights):
            # The first stable placement on a tread is its actual touchdown.
            best_by_index[index] = min(
                runs, key=lambda value: (value[0], -(value[1] - value[0]))
            )
        else:
            best_by_index[index] = max(
                runs, key=lambda value: (value[1] - value[0], value[0])
            )
    exact_contact = np.asarray(clip.contact) >= config.contact_threshold
    merged_support = exact_contact | kinematic_support
    support_clip = replace(
        clip,
        contact=np.asarray(merged_support, dtype=np.float32),
        contact_confidence=np.maximum(
            np.asarray(clip.contact_confidence, dtype=np.float32),
            np.asarray(kinematic_support, dtype=np.float32),
        ),
    )
    events: list[_TreadEvent] = []
    for index in ordered_indices:
        run_start, run_stop, foot = best_by_index[index]
        frame = (run_start + run_stop - 1) // 2
        if events and frame <= events[-1].frame:
            continue
        confirmation = float(np.mean(clip.contact[run_start:run_stop, foot]))
        events.append(
            _TreadEvent(
                frame=frame,
                run_start=run_start,
                run_stop=run_stop,
                foot=foot,
                tread_index=index,
                phase=_phase_at(support_clip, frame),
                confirmation=confirmation,
            )
        )
    return support_clip, kinematic_support, tuple(events)


def _stable_tread_refs(
    clip: CanonicalClip,
    metadata: Stairs500ClipMetadata,
    config: SegmentationConfig,
) -> tuple[CanonicalClip, np.ndarray, tuple[_SemanticRef, ...]]:
    support_clip, kinematic_support, events = _stable_tread_events(
        clip, metadata, config
    )
    if len(events) < 2:
        return clip, kinematic_support, ()
    semantic: list[_SemanticRef] = []
    segmentation_basis = (
        "stable_tread_exact_mesh_height"
        if metadata.physical_tread_heights_m
        else "stable_tread"
    )
    first, first_touchdown = events[:2]
    if abs(first_touchdown.tread_index - first.tread_index) == 1:
        entry_start = first.run_start
        entry_stop = first_touchdown.frame + 1
        entry_evidence = bool(np.any(kinematic_support[entry_start]))
        exit_evidence = bool(
            np.any(kinematic_support[first_touchdown.frame])
        )
        if entry_evidence and exit_evidence:
            semantic.append(
                _SemanticRef(
                    _make_ref(
                        support_clip,
                        entry_start,
                        entry_stop,
                        _phase_at(support_clip, entry_start),
                        _phase_at(support_clip, first_touchdown.frame),
                        config,
                        "stable-tread-entry-step",
                    ),
                    segmentation_basis,
                    FragmentRole.ENTRY,
                    first.tread_index,
                    first_touchdown.tread_index,
                    float(np.max(clip.contact[entry_start])),
                    first_touchdown.confirmation,
                    entry_evidence,
                    exit_evidence,
                )
            )
    for entry, exit in zip(events[1:-2], events[2:-1]):
        if abs(exit.tread_index - entry.tread_index) != 1:
            continue
        entry_evidence = bool(np.any(kinematic_support[entry.frame]))
        exit_evidence = bool(np.any(kinematic_support[exit.frame]))
        if not entry_evidence or not exit_evidence:
            continue
        semantic.append(
            _SemanticRef(
                _make_ref(
                    support_clip,
                    entry.frame,
                    exit.frame + 1,
                    _phase_at(support_clip, entry.frame),
                    _phase_at(support_clip, exit.frame),
                    config,
                    "stable-tread-step",
                ),
                segmentation_basis,
                FragmentRole.MIDDLE,
                entry.tread_index,
                exit.tread_index,
                entry.confirmation,
                exit.confirmation,
                entry_evidence,
                exit_evidence,
            )
        )
    last_takeoff, last_touchdown = events[-2:]
    if abs(last_touchdown.tread_index - last_takeoff.tread_index) == 1:
        exit_start = last_takeoff.frame
        exit_stop = last_touchdown.run_stop
        entry_evidence = bool(np.any(kinematic_support[exit_start]))
        exit_evidence = bool(
            np.any(kinematic_support[exit_stop - 1])
        )
        if entry_evidence and exit_evidence and exit_stop - exit_start >= 2:
            semantic.append(
                _SemanticRef(
                    _make_ref(
                        support_clip,
                        exit_start,
                        exit_stop,
                        _phase_at(support_clip, exit_start),
                        _phase_at(support_clip, exit_stop - 1),
                        config,
                        "stable-tread-exit-step",
                    ),
                    segmentation_basis,
                    FragmentRole.EXIT,
                    last_takeoff.tread_index,
                    last_touchdown.tread_index,
                    last_takeoff.confirmation,
                    float(np.max(clip.contact[exit_stop - 1])),
                    entry_evidence,
                    exit_evidence,
                )
            )
    return support_clip, kinematic_support, tuple(semantic)


def extract_stair_fragments(
    clip: CanonicalClip,
    metadata: Stairs500ClipMetadata,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[StairFragmentRecord, ...]:
    support_clip, kinematic_support, tread_semantic = (
        _stable_tread_refs(clip, metadata, config)
    )
    refs = segment_clip(support_clip, config)
    standard = tuple(
        _SemanticRef(
            ref,
            "stable_contact",
            None,
            None,
            None,
            float(np.max(clip.contact[ref.source_start_frame])),
            float(np.max(clip.contact[ref.source_stop_frame - 1])),
            bool(np.any(kinematic_support[ref.source_start_frame])),
            bool(np.any(kinematic_support[ref.source_stop_frame - 1])),
        )
        for ref in refs
        if bool(np.any(kinematic_support[ref.source_start_frame]))
        and bool(np.any(kinematic_support[ref.source_stop_frame - 1]))
    )
    semantic_refs = tread_semantic if tread_semantic else standard
    if not semantic_refs:
        return ()
    root_height = np.asarray(
        support_clip.root_position_world[:, 2], dtype=np.float64
    )
    low_height, high_height = np.quantile(root_height, (0.02, 0.98))
    records: list[StairFragmentRecord] = []
    exact_level_heights = np.sort(
        np.concatenate(
            (
                np.asarray(
                    (-float(metadata.terrain_position_world[2]),),
                    dtype=np.float64,
                ),
                np.asarray(
                    metadata.physical_tread_heights_m, dtype=np.float64
                ),
            )
        )
    )
    for semantic in semantic_refs:
        ref = semantic.ref
        descriptor = extract_fragment_descriptor(
            support_clip, ref, contact_threshold=config.contact_threshold
        )
        terrain_delta = _terrain_delta(support_clip, ref, metadata)
        midpoint_height = 0.5 * (
            float(support_clip.root_position_world[ref.source_start_frame, 2])
            + float(
                support_clip.root_position_world[
                    ref.source_stop_frame - 1, 2
                ]
            )
        )
        if semantic.role is not None:
            role = semantic.role
        elif metadata.traversal in {"up", "down"}:
            role = classify_fragment_role(
                midpoint_height_m=midpoint_height,
                low_height_m=float(low_height),
                high_height_m=float(high_height),
                traversal=metadata.traversal,
            )
        else:
            midpoint_frame = 0.5 * (
                ref.source_start_frame + ref.source_stop_frame - 1
            )
            fraction = midpoint_frame / max(clip.frame_count - 1, 1)
            role = (
                FragmentRole.ENTRY
                if fraction <= 0.25
                else FragmentRole.EXIT
                if fraction >= 0.75
                else FragmentRole.MIDDLE
            )
        source_rise_m = float(metadata.rise_m)
        if (
            semantic.start_tread_index is not None
            and semantic.end_tread_index is not None
            and len(metadata.physical_tread_heights_m)
            and max(
                semantic.start_tread_index, semantic.end_tread_index
            )
            < len(exact_level_heights)
        ):
            source_rise_m = abs(
                float(
                    exact_level_heights[semantic.end_tread_index]
                    - exact_level_heights[semantic.start_tread_index]
                )
            )
        records.append(
            StairFragmentRecord(
                fragment_id=descriptor.fragment_id,
                archive_clip_index=metadata.archive_clip_index,
                source_clip_name=metadata.clip_name,
                source_start_frame=ref.source_start_frame,
                source_stop_frame=ref.source_stop_frame,
                context_start_frame=ref.context_start_frame,
                context_stop_frame=ref.context_stop_frame,
                entry_phase=ref.entry_phase,
                exit_phase=ref.exit_phase,
                role=role,
                segmentation_basis=semantic.basis,
                start_tread_index=semantic.start_tread_index,
                end_tread_index=semantic.end_tread_index,
                riser_transition_count=(
                    0
                    if semantic.start_tread_index is None
                    or semantic.end_tread_index is None
                    else abs(
                        semantic.end_tread_index
                        - semantic.start_tread_index
                    )
                ),
                entry_contact_confirmation=semantic.entry_confirmation,
                exit_contact_confirmation=semantic.exit_confirmation,
                entry_kinematic_support_evidence=(
                    semantic.entry_kinematic_evidence
                ),
                exit_kinematic_support_evidence=(
                    semantic.exit_kinematic_evidence
                ),
                source_direction=metadata.traversal,
                source_rise_m=source_rise_m,
                source_tread_m=metadata.tread_m,
                source_step_count=metadata.step_count,
                duration_s=descriptor.duration_s,
                root_displacement_local_xyz=(
                    descriptor.root_displacement_local_xyz
                ),
                terrain_displacement_local_xyz=terrain_delta,
                vertical_delta_m=terrain_delta[2],
                approximate_run_m=terrain_delta[0],
                approximate_lateral_m=terrain_delta[1],
                approximate_yaw_rad=descriptor.root_yaw_delta_rad,
                start_speed_local_xyz=(
                    descriptor.entry.root_linear_velocity_local_xyz
                ),
                end_speed_local_xyz=(
                    descriptor.exit.root_linear_velocity_local_xyz
                ),
                start_yaw_rate_rad_s=(
                    descriptor.entry.root_angular_velocity_local_xyz[2]
                ),
                end_yaw_rate_rad_s=(
                    descriptor.exit.root_angular_velocity_local_xyz[2]
                ),
                source_robot_path=metadata.source_robot_path,
                terrain_usd_path=metadata.terrain_usd_path,
                descriptor=descriptor,
            )
        )
    return tuple(records)


def _string(array: object, index: int) -> str:
    return str(np.asarray(array)[index])


def _bundle_pose(shard: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads((shard / "clips.json").read_text())
    row = next(value for value in rows if value.get("stem") == stem)
    terrain = row["terrain"]
    return (
        np.asarray(terrain["position_env"], dtype=np.float32),
        np.asarray(terrain["rotation_env_wxyz"], dtype=np.float32),
    )


def _command_track(
    root_quaternion_wxyz: np.ndarray,
    root_velocity_world: np.ndarray,
    root_angular_velocity_world: np.ndarray,
) -> CommandTrack:
    w, x, y, z = np.moveaxis(root_quaternion_wxyz, -1, 0)
    yaw = np.arctan2(
        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    velocity = root_velocity_world[:, :2]
    local_velocity = np.stack(
        (
            cosine * velocity[:, 0] + sine * velocity[:, 1],
            -sine * velocity[:, 0] + cosine * velocity[:, 1],
        ),
        axis=1,
    )
    frames = len(yaw)
    return CommandTrack(
        observed_travel_stick_xy=np.zeros((frames, 2), dtype=np.float32),
        observed_facing_stick_xy=np.zeros((frames, 2), dtype=np.float32),
        observed_mask=np.zeros(frames, dtype=np.bool_),
        inferred_velocity_local_xy=local_velocity,
        inferred_facing_local_xy=np.tile(
            np.asarray((1.0, 0.0), dtype=np.float32), (frames, 1)
        ),
        inferred_yaw_rate_rad_s=root_angular_velocity_world[:, 2],
    )


def load_reconstructed_archive_clip(
    archive_path: str | Path,
    archive_clip_index: int,
    model_path: str | Path,
) -> tuple[CanonicalClip, Stairs500ClipMetadata]:
    """Load one clean archive clip and reconstruct contact on its paired USD."""

    import mujoco
    import zarr

    archive = zarr.open_group(str(Path(archive_path)), mode="r")
    index = int(archive_clip_index)
    start = int(archive["clip_start_idx"][index])
    stop = int(archive["clip_end_idx"][index])
    stem = _string(archive["clip_names"], index)
    if (
        "terrain_position_env" in archive
        and "terrain_rotation_env_wxyz" in archive
    ):
        # The standardized clean archives already carry the resolved world
        # transform.  Reading it here also supports C490, whose clips.json
        # uses a different source layout from the original stairs500 shards.
        terrain_position = np.asarray(
            archive["terrain_position_env"][index], dtype=np.float32
        )
        terrain_rotation = np.asarray(
            archive["terrain_rotation_env_wxyz"][index], dtype=np.float32
        )
    else:
        shard = Path(_string(archive["source_shard_path"], index))
        terrain_position, terrain_rotation = _bundle_pose(shard, stem)
    terrain_usd = Path(_string(archive["terrain_usd_path"], index))

    body_position = np.asarray(archive["body_pos_w"][start:stop], dtype=np.float32)
    body_xyzw = np.asarray(archive["body_quat_w"][start:stop], dtype=np.float32)
    body_wxyz = unroll_quaternions_wxyz(body_xyzw[..., (3, 0, 1, 2)])
    body_velocity = np.asarray(
        archive["body_lin_vel_w"][start:stop], dtype=np.float32
    )
    root_position = body_position[:, 0]
    root_quaternion = body_wxyz[:, 0]
    root_velocity = body_velocity[:, 0]
    root_angular_velocity = angular_velocity_world_wxyz(root_quaternion, 50.0)
    body_angular_velocity = angular_velocity_world_wxyz(body_wxyz, 50.0)
    joint_position = np.asarray(archive["joint_pos"][start:stop], dtype=np.float32)
    joint_velocity = np.asarray(archive["joint_vel"][start:stop], dtype=np.float32)

    mesh = _load_usd_mesh(terrain_usd, source_asset_sha256="0" * 64)
    binding = TerrainBinding(
        asset_path=str(terrain_usd),
        asset_size_bytes=terrain_usd.stat().st_size,
        asset_sha256="0" * 64,
        asset_license_id="UNRECORDED",
        mesh_sha256="1" * 64,
        world_from_terrain=RigidTransform(terrain_position, terrain_rotation),
        validity_mask_path=None,
    )
    sole_indices = (18, 19)
    frames = stop - start
    clip = CanonicalClip(
        clip_id=f"stairs500/{stem}",
        fps=50.0,
        source=SourceIdentity(
            source_format="stairs500-clean-archive",
            source_path=_string(archive["source_robot_path"], index),
            source_size_bytes=0,
            source_sha256="0" * 64,
            source_license_id="UNRECORDED",
            coordinate_convention=CANONICAL_COORDINATE_CONVENTION,
            quaternion_convention="wxyz",
            pose_origin=CLEAN_POSE_ORIGIN,
        ),
        joint_names=tuple(str(value) for value in archive["joint_names"][:]),
        body_names=tuple(str(value) for value in archive["body_names"][:]),
        root_position_world=root_position,
        root_quaternion_world_wxyz=root_quaternion,
        joint_position=joint_position,
        root_linear_velocity_world=root_velocity,
        root_angular_velocity_world=root_angular_velocity,
        joint_velocity=joint_velocity,
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_wxyz,
        body_linear_velocity_world=body_velocity,
        body_angular_velocity_world=body_angular_velocity,
        sole_position_world=body_position[:, sole_indices],
        sole_quaternion_world_wxyz=body_wxyz[:, sole_indices],
        heel_position_world=body_position[:, sole_indices],
        toe_position_world=body_position[:, sole_indices],
        contact=np.zeros((frames, 2), dtype=np.float32),
        contact_confidence=np.zeros((frames, 2), dtype=np.float32),
        commands=_command_track(
            root_quaternion, root_velocity, root_angular_velocity
        ),
        terrain=binding,
        action_tags=(
            "stairs500",
            "terrain",
            "stairs",
            _string(archive["clip_traversal"], index),
            CONTACT_PLACEHOLDER_TAG,
        ),
    )
    clip.validate()
    model = mujoco.MjModel.from_xml_path(str(Path(model_path)))
    contact_config = ContactConfig(SoleGeometry.from_model(model))
    reconstructed = reconstruct_contacts(
        clip,
        CanonicalMeshQuery(mesh, binding.world_from_terrain),
        contact_config,
    ).apply(clip)
    try:
        physical_profile = measure_archive_stair_profile(archive, index)
        physical_edges = tuple(physical_profile.tread_edges_m)
        physical_heights = tuple(physical_profile.tread_heights_m)
    except ContractError:
        # A curb course may alternate repeatedly between ground and raised
        # blocks, so it is not representable by the monotonic-stair profile.
        # Sample exact support heights along the recorded (possibly
        # round-trip) root trajectory instead.  Fragment contact
        # reconstruction still uses the full mesh.
        terrain_index = TerrainMeshIndex(mesh, binding.world_from_terrain)
        ray_height = float(np.max(terrain_index.vertices_world[:, 2]) + 1.0)
        support_points = []
        for root_point in root_position:
            hit = terrain_index.raycast(
                (float(root_point[0]), float(root_point[1]), ray_height),
                (0.0, 0.0, -1.0),
            )
            if hit is not None:
                support_points.append(hit.position_world)
        if not support_points:
            raise ContractError("root trajectory does not cross visible terrain")
        inverse = binding.world_from_terrain.inverse()
        local_heights = inverse.apply_points(
            np.asarray(support_points, dtype=np.float32)
        )[:, 2]
        physical_heights = tuple(
            float(value)
            for value in np.unique(np.round(local_heights, decimals=3))
        )
        physical_edges = tuple(
            float(value) for value in range(len(physical_heights) + 1)
        )
    metadata = Stairs500ClipMetadata(
        archive_clip_index=index,
        clip_name=stem,
        family=_string(archive["clip_family"], index),
        traversal=_string(archive["clip_traversal"], index),
        rise_m=float(archive["stair_rise_m"][index]),
        tread_m=float(archive["stair_tread_m"][index]),
        step_count=int(archive["stair_n_steps"][index]),
        travel_yaw_rad=float(archive["travel_yaw_rad"][index]),
        terrain_position_world=tuple(float(value) for value in terrain_position),
        terrain_rotation_world_wxyz=tuple(
            float(value) for value in terrain_rotation
        ),
        source_robot_path=_string(archive["source_robot_path"], index),
        terrain_usd_path=str(terrain_usd),
        physical_tread_edges_m=physical_edges,
        physical_tread_heights_m=physical_heights,
    )
    return reconstructed, metadata


def process_archive_clip(
    archive_path: str | Path,
    archive_clip_index: int,
    model_path: str | Path,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[StairFragmentRecord, ...]:
    clip, metadata = load_reconstructed_archive_clip(
        archive_path, archive_clip_index, model_path
    )
    return extract_stair_fragments(clip, metadata, config)


__all__ = (
    "FragmentRole",
    "StairFragmentRecord",
    "Stairs500ClipMetadata",
    "classify_fragment_role",
    "extract_stair_fragments",
    "load_reconstructed_archive_clip",
    "process_archive_clip",
)
