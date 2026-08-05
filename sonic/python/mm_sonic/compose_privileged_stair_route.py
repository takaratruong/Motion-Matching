"""Compose audited stair kernels with ordinary flat motion matching.

This is the deliberately privileged bootstrap: the complete target mesh and
the ordered held-out risers are known.  Each stair kernel remains fixed to its
audited target geometry.  Short flat-motion bridges absorb phase and root-pose
differences between adjacent kernels; long landings use the same mechanism for
actual travel.  Pose offsets are inertialized at every source switch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .hybrid_terrain_interactive import (
    FlatKinematicSource,
    FlatWorldTransform,
    KinematicPose,
)
from .render_stitched_motion import render_stitched_motion
from .terrain_oracle.math3d import (
    quaternion_inverse_wxyz,
    quaternion_multiply_wxyz,
    slerp_wxyz,
)
from .terrain_oracle.reference_stitch import _G1FootfallAdapter
from .terrain_oracle.stair_geometry_warp import _archive_terrain_index
from .terrain_oracle.stair_motion_collision_audit import (
    audit_stair_motion_collisions,
)
from .terrain_oracle.stitch import FrameProvenance, StitchedMotion
from .torch_motion_matcher import MatcherConfig, TorchMotionMatcher


DEFAULT_FLAT_MOTIONS = Path(
    "/move/data/terrain-aware/motion-matching/"
    "takara_bones_walk_support_v2_startstop"
)


@dataclass(frozen=True)
class MotionSegment:
    label: str
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray
    provenance: tuple[FrameProvenance, ...]

    def __post_init__(self) -> None:
        roots = np.asarray(self.root_position_world, dtype=np.float64)
        quaternions = np.asarray(
            self.root_quaternion_world_wxyz, dtype=np.float64
        )
        joints = np.asarray(self.joint_position, dtype=np.float64)
        if (
            roots.ndim != 2
            or roots.shape[1:] != (3,)
            or quaternions.shape != (len(roots), 4)
            or joints.ndim != 2
            or joints.shape[0] != len(roots)
            or len(self.provenance) != len(roots)
            or len(roots) < 2
            or not np.isfinite(roots).all()
            or not np.isfinite(quaternions).all()
            or not np.isfinite(joints).all()
        ):
            raise ValueError("motion segment arrays are inconsistent")


@dataclass(frozen=True)
class KernelCandidate:
    """One independently audited realization of a target riser."""

    rank: int
    relative: str
    fragment_ids: tuple[str, ...]
    segment: MotionSegment
    maximum_foot_penetration_m: float
    maximum_forbidden_body_penetration_m: float


def _yaw_wxyz(quaternion: object) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _pose_wxyz(pose: KinematicPose) -> np.ndarray:
    return np.asarray(
        pose.root_orientation_world_xyzw[[3, 0, 1, 2]],
        dtype=np.float64,
    )


def _kinematic_pose_from_segment(
    segment: MotionSegment,
    frame: int,
    adapter: _G1FootfallAdapter,
    body_ids: np.ndarray,
) -> KinematicPose:
    """Materialize an audited root/joint sample for support-phase matching."""

    index = int(frame)
    root = np.asarray(segment.root_position_world[index], dtype=np.float64)
    quaternion = np.asarray(
        segment.root_quaternion_world_wxyz[index], dtype=np.float64
    )
    joints = np.asarray(segment.joint_position[index], dtype=np.float64)
    adapter._set_pose(adapter._reference_data, root, quaternion, joints)
    data = adapter._reference_data
    ids = np.asarray(body_ids, dtype=np.int64)
    if ids.shape != (30,) or np.any(ids <= 0):
        raise ValueError("archive body ids must map all 30 G1 bodies")
    body_wxyz = np.asarray(data.xquat[ids], dtype=np.float32)
    return KinematicPose(
        root_position_world=np.asarray(root, dtype=np.float32),
        root_orientation_world_xyzw=np.asarray(
            quaternion[[1, 2, 3, 0]], dtype=np.float32
        ),
        joint_position=np.asarray(joints, dtype=np.float32),
        joint_velocity=np.zeros(29, dtype=np.float32),
        body_position_world=np.asarray(data.xpos[ids], dtype=np.float32),
        body_orientation_world_xyzw=np.asarray(
            body_wxyz[:, [1, 2, 3, 0]], dtype=np.float32
        ),
        body_linear_velocity_world=np.zeros((30, 3), dtype=np.float32),
        body_angular_velocity_world=np.zeros((30, 3), dtype=np.float32),
    )


def _support_contact(
    pose: KinematicPose,
    adapter: _G1FootfallAdapter,
    target_mesh: object,
    *,
    tolerance_m: float = 0.025,
) -> np.ndarray:
    supports = adapter.sole_support_points_for_pose(
        root_position=pose.root_position_world,
        root_quaternion_wxyz=_pose_wxyz(pose),
        joints=pose.joint_position,
    )
    ray_origin = float(np.max(target_mesh.vertices_world[:, 2])) + 1.0
    clearances = []
    for foot_points in supports:
        values = []
        for point in foot_points:
            hit = target_mesh.raycast(
                np.asarray((point[0], point[1], ray_origin)),
                np.asarray((0.0, 0.0, -1.0)),
            )
            surface = 0.0 if hit is None else float(hit.position_world[2])
            values.append(abs(float(point[2]) - surface))
        clearances.append(float(np.median(values)))
    contact = np.asarray(
        [value <= float(tolerance_m) for value in clearances], dtype=bool
    )
    if not np.any(contact):
        contact[int(np.argmin(clearances))] = True
    return contact


def _critical_fraction(
    frame_count: int,
    *,
    fps: float,
    halflife_s: float,
) -> np.ndarray:
    if frame_count < 1 or fps <= 0.0 or halflife_s <= 0.0:
        raise ValueError("critical decay arguments must be positive")
    time = np.arange(frame_count, dtype=np.float64) / float(fps)
    rate = 2.0 * math.log(2.0) / (float(halflife_s) + 1.0e-5)
    return np.exp(-rate * time) * (1.0 + rate * time)


def _critical_offset(
    position_offset: object,
    velocity_offset: object,
    frame_count: int,
    *,
    fps: float,
    halflife_s: float,
) -> np.ndarray:
    """Decay a pose offset while honoring its initial offset velocity.

    This uses the critically damped inertialization form

        exp(-r t) * (x0 + (v0 + r x0) t)

    rather than a pose-only cross-fade.  The linear coefficient is solved in
    discrete time so the first rendered sample is exactly ``x0 + v0 / fps``.
    That is the continuity condition that matters to the 50 Hz trajectory.
    """

    if frame_count < 1 or fps <= 0.0 or halflife_s <= 0.0:
        raise ValueError("critical decay arguments must be positive")
    position = np.asarray(position_offset, dtype=np.float64)
    velocity = np.asarray(velocity_offset, dtype=np.float64)
    if position.shape != velocity.shape:
        raise ValueError("critical position and velocity offsets must match")
    time = np.arange(frame_count, dtype=np.float64) / float(fps)
    rate = 2.0 * math.log(2.0) / (float(halflife_s) + 1.0e-5)
    shape = (frame_count,) + (1,) * position.ndim
    time = time.reshape(shape)
    decay = np.exp(-rate * time)
    step_s = 1.0 / float(fps)
    step_decay = math.exp(-rate * step_s)
    linear_coefficient = (
        (position + velocity * step_s) / step_decay - position
    ) / step_s
    return decay * (
        position[None, ...]
        + linear_coefficient[None, ...] * time
    )


def inertialize_segment_entry(
    prior_root_position: object,
    prior_root_quaternion_wxyz: object,
    prior_joint_position: object,
    segment: MotionSegment,
    *,
    prior_root_velocity_world: object | None = None,
    prior_joint_velocity: object | None = None,
    fps: float = 50.0,
    halflife_s: float = 0.10,
) -> MotionSegment:
    """Match the incoming pose and velocity, then decay onto the segment."""

    root = np.asarray(segment.root_position_world, dtype=np.float64).copy()
    quaternion = np.asarray(
        segment.root_quaternion_world_wxyz, dtype=np.float64
    ).copy()
    joint = np.asarray(segment.joint_position, dtype=np.float64).copy()
    root_offset = (
        np.asarray(prior_root_position, dtype=np.float64) - root[0]
    )
    joint_offset = (
        np.asarray(prior_joint_position, dtype=np.float64) - joint[0]
    )
    target_root_velocity = (root[1] - root[0]) * float(fps)
    target_joint_velocity = (joint[1] - joint[0]) * float(fps)
    incoming_root_velocity = (
        target_root_velocity
        if prior_root_velocity_world is None
        else np.asarray(prior_root_velocity_world, dtype=np.float64)
    )
    incoming_joint_velocity = (
        target_joint_velocity
        if prior_joint_velocity is None
        else np.asarray(prior_joint_velocity, dtype=np.float64)
    )
    root += _critical_offset(
        root_offset,
        incoming_root_velocity - target_root_velocity,
        len(root),
        fps=fps,
        halflife_s=halflife_s,
    )
    joint += _critical_offset(
        joint_offset,
        incoming_joint_velocity - target_joint_velocity,
        len(root),
        fps=fps,
        halflife_s=halflife_s,
    )
    fraction = _critical_fraction(
        len(root), fps=fps, halflife_s=halflife_s
    )
    orientation_offset = quaternion_multiply_wxyz(
        np.asarray(prior_root_quaternion_wxyz, dtype=np.float64),
        quaternion_inverse_wxyz(quaternion[0]),
    )
    scaled_offset = slerp_wxyz(
        np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        orientation_offset,
        fraction,
    )
    quaternion = quaternion_multiply_wxyz(scaled_offset, quaternion)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    return MotionSegment(
        label=segment.label,
        root_position_world=np.asarray(root, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternion, dtype=np.float32),
        joint_position=np.asarray(joint, dtype=np.float32),
        provenance=segment.provenance,
    )


def concatenate_segments(
    segments: Sequence[MotionSegment],
    *,
    fps: float = 50.0,
    halflife_s: float = 0.10,
) -> StitchedMotion:
    """Inertialize ordered segments and retain one sample at every seam."""

    values = tuple(segments)
    if not values:
        raise ValueError("at least one motion segment is required")
    roots = [np.asarray(values[0].root_position_world, dtype=np.float64)]
    quaternions = [
        np.asarray(values[0].root_quaternion_world_wxyz, dtype=np.float64)
    ]
    joints = [np.asarray(values[0].joint_position, dtype=np.float64)]
    provenance = list(values[0].provenance)
    seams: list[int] = []
    for segment in values[1:]:
        root_tail = np.concatenate(roots, axis=0)[-2:]
        joint_tail = np.concatenate(joints, axis=0)[-2:]
        adjusted = inertialize_segment_entry(
            roots[-1][-1],
            quaternions[-1][-1],
            joints[-1][-1],
            segment,
            prior_root_velocity_world=(
                root_tail[-1] - root_tail[-2]
            ) * float(fps),
            prior_joint_velocity=(
                joint_tail[-1] - joint_tail[-2]
            ) * float(fps),
            fps=fps,
            halflife_s=halflife_s,
        )
        seams.append(sum(len(value) for value in roots))
        # Frame zero equals the previous endpoint after inertialization.
        roots.append(np.asarray(adjusted.root_position_world[1:], dtype=np.float64))
        quaternions.append(
            np.asarray(
                adjusted.root_quaternion_world_wxyz[1:], dtype=np.float64
            )
        )
        joints.append(np.asarray(adjusted.joint_position[1:], dtype=np.float64))
        provenance.extend(adjusted.provenance[1:])
    return StitchedMotion(
        fps=float(fps),
        root_position_world=np.asarray(np.concatenate(roots), dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(
            np.concatenate(quaternions), dtype=np.float32
        ),
        joint_position=np.asarray(np.concatenate(joints), dtype=np.float32),
        provenance=tuple(provenance),
        seam_indices=tuple(seams),
    )


def load_kernel_segment(path: str | Path, *, label: str) -> MotionSegment:
    source = Path(path)
    with np.load(source, allow_pickle=False) as arrays:
        roots = np.asarray(arrays["root_position_world"], dtype=np.float32)
        quaternions = np.asarray(
            arrays["root_quaternion_world_wxyz"], dtype=np.float32
        )
        joints = np.asarray(arrays["joint_position"], dtype=np.float32)
        clip_indices = np.asarray(
            arrays["source_archive_clip_index"], dtype=np.int64
        )
        source_frames = np.asarray(arrays["source_frame"], dtype=np.int64)
        clip_ids = np.asarray(arrays["source_clip_id"])
    provenance = tuple(
        FrameProvenance(int(clip), int(frame), str(clip_id))
        for clip, frame, clip_id in zip(
            clip_indices, source_frames, clip_ids, strict=True
        )
    )
    return MotionSegment(label, roots, quaternions, joints, provenance)


def load_candidate_pool(
    directory: str | Path,
    *,
    label: str,
) -> tuple[KernelCandidate, ...]:
    """Load an audited phase pool shared by one-riser and block nodes."""

    source = Path(directory)
    pool_summaries = []
    for summary_path in source.glob("candidate_pool_v*/summary.json"):
        summary = json.loads(summary_path.read_text())
        pool_summaries.append(
            (
                int(summary.get("accepted_candidate_count", 0)),
                summary_path.parent.name,
                summary_path.parent,
                summary,
            )
        )
    output: list[KernelCandidate] = []
    if pool_summaries:
        _, pool_name, pool, summary = max(
            pool_summaries, key=lambda value: (value[0], value[1])
        )
        for row in summary.get("candidates", ()):
            relative = str(row["relative"])
            output.append(
                KernelCandidate(
                    rank=int(row["rank"]),
                    relative=f"{pool_name}/{relative}",
                    fragment_ids=tuple(
                        str(value) for value in row.get("fragment_ids", ())
                    ),
                    segment=load_kernel_segment(
                        pool / relative / "motion.npz",
                        label=f"{label}_rank{int(row['rank']):03d}",
                    ),
                    maximum_foot_penetration_m=float(
                        row["maximum_foot_penetration_m"]
                    ),
                    maximum_forbidden_body_penetration_m=float(
                        row["maximum_forbidden_body_penetration_m"]
                    ),
                )
            )
    return tuple(output)


def load_kernel_candidates(
    directory: str | Path,
    *,
    label: str,
) -> tuple[KernelCandidate, ...]:
    """Load the audited phase pool, falling back to the original kernel."""

    source = Path(directory)
    pooled = load_candidate_pool(source, label=label)
    if pooled:
        return pooled

    result = json.loads((source / "kernel_result.json").read_text())
    if result.get("status") != "accepted":
        raise ValueError(f"kernel was not accepted: {source}")
    audit = dict(result.get("full_body_audit", {}))
    selected_plan = dict(result.get("selected_plan", {}))
    return (
        KernelCandidate(
            rank=int(result.get("selected_rank", 0)),
            relative=".",
            fragment_ids=tuple(
                str(value) for value in selected_plan.get("fragment_ids", ())
            ),
            segment=load_kernel_segment(
                source / "motion.npz", label=f"{label}_original"
            ),
            maximum_foot_penetration_m=float(
                audit.get("maximum_foot_penetration_m", 0.0)
            ),
            maximum_forbidden_body_penetration_m=float(
                audit.get("maximum_forbidden_body_penetration_m", 0.0)
            ),
        ),
    )


def load_sweep_candidate(
    directory: str | Path,
    *,
    label: str,
) -> tuple[KernelCandidate, dict[str, object]]:
    """Load one exact-audited multi-riser sweep result as a graph node."""

    source = Path(directory)
    sweep = json.loads((source / "sweep_result.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    audit = json.loads(
        (source / "full_body_collision_audit.json").read_text()
    )
    if sweep.get("status") != "accepted" or not bool(audit.get("accepted")):
        raise ValueError(f"multi-riser block was not accepted: {source}")
    selected_plan = dict(summary.get("selected_plan", {}))
    return (
        KernelCandidate(
            rank=int(summary.get("selected_rank", 0)),
            relative=".",
            fragment_ids=tuple(
                str(value) for value in selected_plan.get("fragment_ids", ())
            ),
            segment=load_kernel_segment(
                source / "motion.npz", label=label
            ),
            maximum_foot_penetration_m=float(
                audit["maximum_foot_penetration_m"]
            ),
            maximum_forbidden_body_penetration_m=float(
                audit["maximum_forbidden_body_penetration_m"]
            ),
        ),
        summary,
    )


def load_sweep_candidates(
    directory: str | Path,
    *,
    label: str,
) -> tuple[tuple[KernelCandidate, ...], dict[str, object]]:
    """Load all audited phases for a multi-riser block when available."""

    candidate, summary = load_sweep_candidate(directory, label=label)
    pooled = load_candidate_pool(directory, label=label)
    return (pooled if pooled else (candidate,)), summary


def endpoint_seam_metrics(
    source: MotionSegment,
    target: MotionSegment,
    *,
    fps: float = 50.0,
) -> dict[str, float]:
    """Measure pose and velocity disagreement at a candidate source switch."""

    source_root = np.asarray(source.root_position_world, dtype=np.float64)
    target_root = np.asarray(target.root_position_world, dtype=np.float64)
    source_joint = np.asarray(source.joint_position, dtype=np.float64)
    target_joint = np.asarray(target.joint_position, dtype=np.float64)
    source_quaternion = np.asarray(
        source.root_quaternion_world_wxyz[-1], dtype=np.float64
    )
    target_quaternion = np.asarray(
        target.root_quaternion_world_wxyz[0], dtype=np.float64
    )
    orientation_dot = abs(float(np.dot(source_quaternion, target_quaternion)))
    joint_delta = source_joint[-1] - target_joint[0]
    source_root_velocity = (source_root[-1] - source_root[-2]) * float(fps)
    target_root_velocity = (target_root[1] - target_root[0]) * float(fps)
    source_joint_velocity = (source_joint[-1] - source_joint[-2]) * float(fps)
    target_joint_velocity = (target_joint[1] - target_joint[0]) * float(fps)
    return {
        "root_position_gap_m": float(
            np.linalg.norm(source_root[-1] - target_root[0])
        ),
        "root_orientation_gap_rad": float(
            2.0 * math.acos(np.clip(orientation_dot, 0.0, 1.0))
        ),
        "joint_position_rmse_rad": float(
            np.sqrt(np.mean(np.square(joint_delta)))
        ),
        "maximum_joint_position_gap_rad": float(np.max(np.abs(joint_delta))),
        "root_velocity_gap_m_s": float(
            np.linalg.norm(source_root_velocity - target_root_velocity)
        ),
        "joint_velocity_rmse_rad_s": float(
            np.sqrt(
                np.mean(
                    np.square(source_joint_velocity - target_joint_velocity)
                )
            )
        ),
    }


def endpoint_seam_cost(
    source: MotionSegment,
    target: MotionSegment,
    *,
    fps: float = 50.0,
) -> float:
    """Dimensionless motion-graph edge cost for phase-compatible stitching."""

    value = endpoint_seam_metrics(source, target, fps=fps)
    return float(
        value["root_position_gap_m"] / 0.05
        + value["root_orientation_gap_rad"] / 0.35
        + value["joint_position_rmse_rad"] / 0.12
        + value["maximum_joint_position_gap_rad"] / 0.35
        + value["root_velocity_gap_m_s"] / 0.40
        + value["joint_velocity_rmse_rad_s"] / 0.80
    )


def rank_candidate_chains(
    candidate_layers: Sequence[Sequence[KernelCandidate]],
    *,
    beam_width: int = 128,
    fps: float = 50.0,
) -> tuple[tuple[float, tuple[KernelCandidate, ...]], ...]:
    """Beam-search globally compatible phases over an ordered riser sequence."""

    layers = tuple(tuple(layer) for layer in candidate_layers)
    if not layers or any(not layer for layer in layers):
        raise ValueError("every riser must have at least one candidate")
    width = int(beam_width)
    if width < 1:
        raise ValueError("beam width must be positive")
    beam = [(0.0, (candidate,)) for candidate in layers[0]]
    for layer in layers[1:]:
        expanded = []
        for cost, chain in beam:
            for candidate in layer:
                expanded.append(
                    (
                        cost
                        + endpoint_seam_cost(
                            chain[-1].segment, candidate.segment, fps=fps
                        ),
                        chain + (candidate,),
                    )
                )
        expanded.sort(
            key=lambda value: (
                value[0],
                tuple(candidate.rank for candidate in value[1]),
            )
        )
        beam = expanded[:width]
    return tuple(beam)


def generate_flat_bridge(
    matcher: TorchMotionMatcher,
    source: MotionSegment,
    target: MotionSegment,
    *,
    adapter: _G1FootfallAdapter,
    body_ids: np.ndarray,
    target_mesh: object,
    travel_yaw_world: float,
    fps: float = 50.0,
    speed_m_s: float = 0.45,
    minimum_frames: int = 26,
    settle_frames: int = 14,
    maximum_endpoint_correction_m: float = 0.25,
    pose_blend_frames: int = 14,
    apply_endpoint_warp: bool = True,
) -> MotionSegment:
    """Use flat MM to connect two stair poses on their shared plateau."""

    start = np.asarray(source.root_position_world[-1], dtype=np.float64)
    end = np.asarray(target.root_position_world[0], dtype=np.float64)
    planar_distance = float(np.linalg.norm(end[:2] - start[:2]))
    travel_frames = int(math.ceil(planar_distance / speed_m_s * fps))
    frame_count = max(int(minimum_frames), travel_frames + int(settle_frames))
    source_pose = _kinematic_pose_from_segment(
        source, -1, adapter, body_ids
    )
    target_pose = _kinematic_pose_from_segment(
        target, 0, adapter, body_ids
    )
    flat = FlatKinematicSource(matcher)
    support_contact = _support_contact(source_pose, adapter, target_mesh)
    pose = flat.reset_supported(
        root_position_world=np.asarray(start, dtype=np.float32),
        root_yaw_world=_yaw_wxyz(source.root_quaternion_world_wxyz[-1]),
        support_pose_world=source_pose,
        support_contact=support_contact,
        maximum_vertical_correction_m=0.12,
        maximum_planar_correction_m=0.12,
    )
    # reset_supported aligns ankle bodies, but a different ankle pitch can
    # still put the actual sole proxy a centimetre inside a landing.  Align
    # the contacted sole samples themselves before advancing the flat matcher.
    source_soles = adapter.sole_support_points_for_pose(
        root_position=source_pose.root_position_world,
        root_quaternion_wxyz=_pose_wxyz(source_pose),
        joints=source_pose.joint_position,
    )
    flat_soles = adapter.sole_support_points_for_pose(
        root_position=pose.root_position_world,
        root_quaternion_wxyz=_pose_wxyz(pose),
        joints=pose.joint_position,
    )
    sole_delta = np.median(
        np.asarray(source_soles, dtype=np.float64)[support_contact]
        - np.asarray(flat_soles, dtype=np.float64)[support_contact],
        axis=(0, 1),
    )
    transform = flat.transform
    if transform is None:
        raise AssertionError("flat support reset did not create a transform")
    flat.transform = FlatWorldTransform(
        transform.yaw_offset,
        np.asarray(transform.translation_world + sole_delta, dtype=np.float32),
    )
    pose = KinematicPose(
        root_position_world=np.asarray(
            pose.root_position_world + sole_delta, dtype=np.float32
        ),
        root_orientation_world_xyzw=pose.root_orientation_world_xyzw,
        joint_position=pose.joint_position,
        joint_velocity=pose.joint_velocity,
        body_position_world=np.asarray(
            pose.body_position_world + sole_delta, dtype=np.float32
        ),
        body_orientation_world_xyzw=pose.body_orientation_world_xyzw,
        body_linear_velocity_world=pose.body_linear_velocity_world,
        body_angular_velocity_world=pose.body_angular_velocity_world,
    )
    poses = [pose]
    for frame in range(1, frame_count):
        remaining = end[:2] - pose.root_position_world[:2]
        remaining_norm = float(np.linalg.norm(remaining))
        if frame < frame_count - settle_frames and remaining_norm > 0.025:
            commanded_speed = min(
                float(speed_m_s), max(0.10, 1.5 * remaining_norm)
            )
            velocity = remaining / remaining_norm * commanded_speed
        else:
            velocity = np.zeros(2, dtype=np.float64)
        pose = flat.step(
            velocity_world_xy=np.asarray(velocity, dtype=np.float32),
            heading_world_yaw=float(travel_yaw_world),
        )
        poses.append(pose)

    roots = np.asarray(
        [value.root_position_world for value in poses], dtype=np.float64
    )
    quaternions = np.asarray([_pose_wxyz(value) for value in poses])
    joints = np.asarray([value.joint_position for value in poses])
    if apply_endpoint_warp:
        start_correction = start - roots[0]
        endpoint_correction = end - roots[-1]
        correction_norm = max(
            float(np.linalg.norm(start_correction)),
            float(np.linalg.norm(endpoint_correction)),
        )
        if correction_norm > float(maximum_endpoint_correction_m):
            raise ValueError(
                "flat bridge endpoint correction exceeds bound: "
                f"{correction_norm:.3f} m"
            )
        coordinate = np.linspace(0.0, 1.0, len(roots), dtype=np.float64)
        smooth = coordinate * coordinate * (3.0 - 2.0 * coordinate)
        roots += (
            (1.0 - smooth)[:, None] * start_correction
            + smooth[:, None] * endpoint_correction
        )
        blend = min(int(pose_blend_frames), max(2, len(roots) // 2))
        coordinate = np.linspace(0.0, 1.0, blend, dtype=np.float64)
        weight = coordinate * coordinate * (3.0 - 2.0 * coordinate)
        joints[:blend] = (
            (1.0 - weight)[:, None] * source.joint_position[-1]
            + weight[:, None] * joints[:blend]
        )
        joints[-blend:] = (
            (1.0 - weight)[:, None] * joints[-blend:]
            + weight[:, None] * target.joint_position[0]
        )
        quaternions[:blend] = slerp_wxyz(
            np.broadcast_to(source.root_quaternion_world_wxyz[-1], (blend, 4)),
            quaternions[:blend],
            weight,
        )
        quaternions[-blend:] = slerp_wxyz(
            quaternions[-blend:],
            np.broadcast_to(target.root_quaternion_world_wxyz[0], (blend, 4)),
            weight,
        )
        roots[0] = source.root_position_world[-1]
        roots[-1] = target.root_position_world[0]
        joints[0] = source.joint_position[-1]
        joints[-1] = target.joint_position[0]
        quaternions[0] = source.root_quaternion_world_wxyz[-1]
        quaternions[-1] = target.root_quaternion_world_wxyz[0]
    provenance = tuple(
        FrameProvenance(-1, frame, "takara_bones_flat_mm")
        for frame in range(len(poses))
    )
    return MotionSegment(
        label="flat_bridge",
        root_position_world=np.asarray(roots, dtype=np.float32),
        root_quaternion_world_wxyz=np.asarray(quaternions, dtype=np.float32),
        joint_position=np.asarray(joints, dtype=np.float32),
        provenance=provenance,
    )


def _save_motion(path: Path, motion: StitchedMotion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fps=np.asarray(motion.fps, dtype=np.float32),
        root_position_world=np.asarray(
            motion.root_position_world, dtype=np.float32
        ),
        root_quaternion_world_wxyz=np.asarray(
            motion.root_quaternion_world_wxyz, dtype=np.float32
        ),
        joint_position=np.asarray(motion.joint_position, dtype=np.float32),
        seam_indices=np.asarray(motion.seam_indices, dtype=np.int64),
        source_archive_clip_index=np.asarray(
            [value.archive_clip_index for value in motion.provenance],
            dtype=np.int64,
        ),
        source_frame=np.asarray(
            [value.source_frame for value in motion.provenance], dtype=np.int64
        ),
        source_clip_id=np.asarray(
            [value.clip_id for value in motion.provenance], dtype=np.str_
        ),
    )


def _maximum_steps(motion: StitchedMotion) -> dict[str, float]:
    roots = np.asarray(motion.root_position_world, dtype=np.float64)
    joints = np.asarray(motion.joint_position, dtype=np.float64)
    quaternions = np.asarray(
        motion.root_quaternion_world_wxyz, dtype=np.float64
    )
    dots = np.abs(np.sum(quaternions[1:] * quaternions[:-1], axis=1))
    joint_acceleration = (
        np.diff(joints, n=2, axis=0) * motion.fps * motion.fps
    )
    root_acceleration = (
        np.diff(roots, n=2, axis=0) * motion.fps * motion.fps
    )
    seam_acceleration_mask = np.zeros(len(joint_acceleration), dtype=bool)
    for seam in motion.seam_indices:
        # The velocity is matched exactly on the first sample.  Measure the
        # following decay transient, where pose-only blending used to create
        # the visible fast-foot hitch.
        seam_acceleration_mask[
            max(0, int(seam) - 2) : min(
                len(seam_acceleration_mask), int(seam) + 12
            )
        ] = True
    seam_joint_acceleration = (
        float(np.max(np.abs(joint_acceleration[seam_acceleration_mask])))
        if np.any(seam_acceleration_mask)
        else 0.0
    )
    seam_root_acceleration = (
        float(
            np.max(
                np.linalg.norm(
                    root_acceleration[seam_acceleration_mask], axis=1
                )
            )
        )
        if np.any(seam_acceleration_mask)
        else 0.0
    )
    return {
        "maximum_root_translation_step_m": float(
            np.max(np.linalg.norm(np.diff(roots, axis=0), axis=1))
        ),
        "maximum_root_rotation_step_rad": float(
            np.max(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
        ),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(joints, axis=0)))),
        "maximum_root_acceleration_m_s2": float(
            np.max(np.linalg.norm(root_acceleration, axis=1))
        ),
        "maximum_seam_root_acceleration_m_s2": seam_root_acceleration,
        "maximum_seam_joint_acceleration_rad_s2": seam_joint_acceleration,
    }


def compose_target(
    *,
    kernel_manifest_path: str | Path,
    kernel_manifest_overlays: Sequence[str | Path] = (),
    block_manifest_paths: Sequence[str | Path] = (),
    target_array_index: int,
    output: str | Path,
    flat_motions: str | Path = DEFAULT_FLAT_MOTIONS,
    flat_device: str = "cpu",
    inertialization_halflife_s: float | None = None,
    candidate_beam_width: int = 128,
    maximum_candidate_chains: int = 128,
    minimum_flat_bridge_gap_m: float = 0.08,
    render: bool = True,
) -> dict[str, object]:
    manifest = json.loads(Path(kernel_manifest_path).read_text())
    primary_entries = sorted(
        (
            entry
            for entry in manifest["kernels"]
            if int(entry["target_array_index"]) == int(target_array_index)
        ),
        key=lambda entry: int(entry["transition_ordinal"]),
    )
    if not primary_entries:
        raise ValueError("target has no riser kernels")
    manifests = [manifest]
    manifests.extend(
        json.loads(Path(path).expanduser().resolve().read_text())
        for path in kernel_manifest_overlays
    )
    for overlay in manifests[1:]:
        for key in ("archive_path", "model_path"):
            if str(overlay[key]) != str(manifest[key]):
                raise ValueError(f"kernel overlay has a different {key}")
    block_manifests = [
        json.loads(Path(path).expanduser().resolve().read_text())
        for path in block_manifest_paths
    ]
    for block_manifest in block_manifests:
        for key in ("archive_path", "model_path"):
            if str(block_manifest[key]) != str(manifest[key]):
                raise ValueError(f"block manifest has a different {key}")

    # A block replaces the exact consecutive primary risers bounded by its
    # start/end coordinates.  This retains the compact one-riser graph in the
    # common case while allowing a coherent multi-step primitive where a
    # narrow high-tread sequence cannot be reconstructed one step at a time.
    block_starts: dict[int, tuple[dict[str, object], dict[str, object], int]] = {}
    blocked_ordinals: set[int] = set()
    target_clip_index = int(primary_entries[0]["target_clip_index"])
    target_traversal = str(primary_entries[0]["traversal"])
    for block_manifest in block_manifests:
        for block_target in block_manifest.get("targets", ()):
            if (
                int(block_target["target_clip_index"]) != target_clip_index
                or str(block_target.get("traversal")) != target_traversal
            ):
                continue
            block_start = np.asarray(block_target["start_xy"], dtype=np.float64)
            block_end = np.asarray(block_target["end_xy"], dtype=np.float64)
            start_candidates = [
                index
                for index, entry in enumerate(primary_entries)
                if np.linalg.norm(
                    np.asarray(entry["start_xy"], dtype=np.float64)
                    - block_start
                )
                <= 1.0e-5
            ]
            end_candidates = [
                index
                for index, entry in enumerate(primary_entries)
                if np.linalg.norm(
                    np.asarray(entry["end_xy"], dtype=np.float64) - block_end
                )
                <= 1.0e-5
            ]
            spans = [
                (start, end)
                for start in start_candidates
                for end in end_candidates
                if end >= start
            ]
            if len(spans) != 1:
                raise ValueError(
                    "block does not uniquely bound consecutive primary risers: "
                    f"{block_target.get('output_relative')} -> {spans}"
                )
            start, end = spans[0]
            start_ordinal = int(primary_entries[start]["transition_ordinal"])
            end_ordinal = int(primary_entries[end]["transition_ordinal"])
            overlap = blocked_ordinals.intersection(range(start_ordinal, end_ordinal + 1))
            if overlap:
                raise ValueError(f"overlapping multi-riser blocks: {sorted(overlap)}")
            blocked_ordinals.update(range(start_ordinal, end_ordinal + 1))
            block_starts[start_ordinal] = (
                block_manifest,
                block_target,
                end_ordinal,
            )
    overlay_entries: dict[
        tuple[int, int], list[tuple[dict[str, object], dict[str, object]]]
    ] = {}
    for source_manifest in manifests:
        for entry in source_manifest["kernels"]:
            if int(entry["target_array_index"]) != int(target_array_index):
                continue
            key = (
                int(entry["target_array_index"]),
                int(entry["transition_ordinal"]),
            )
            overlay_entries.setdefault(key, []).append((source_manifest, entry))

    entries: list[dict[str, object]] = []
    resolved_sources: list[tuple[dict[str, object], Path]] = []
    candidate_layers: list[tuple[KernelCandidate, ...]] = []
    kernel_results: list[dict[str, object]] = []
    for primary_entry in primary_entries:
        ordinal = int(primary_entry["transition_ordinal"])
        block = block_starts.get(ordinal)
        if block is not None:
            block_manifest, block_target, end_ordinal = block
            directory = Path(str(block_manifest["output_root"])) / str(
                block_target["output_relative"]
            )
            candidates, block_summary = load_sweep_candidates(
                directory,
                label=f"riser_block_{ordinal:02d}_{end_ordinal:02d}",
            )
            entry = dict(primary_entry)
            entry["end_xy"] = list(block_target["end_xy"])
            entry["transition_end_ordinal"] = int(end_ordinal)
            entry["output_relative"] = str(block_target["output_relative"])
            entries.append(entry)
            resolved_sources.append((block_manifest, directory))
            candidate_layers.append(candidates)
            kernel_results.append({**block_target, **block_summary})
            continue
        if ordinal in blocked_ordinals:
            continue
        key = (
            int(primary_entry["target_array_index"]),
            int(primary_entry["transition_ordinal"]),
        )
        accepted_source = None
        checked = []
        # Later overlays are explicit repairs and therefore take precedence
        # over an already-accepted but geometrically less useful primary.
        for source_manifest, entry in reversed(overlay_entries.get(key, ())):
            directory = Path(str(source_manifest["output_root"])) / str(
                entry["output_relative"]
            )
            result_path = directory / "kernel_result.json"
            if not result_path.exists():
                checked.append(f"missing:{result_path}")
                continue
            result = json.loads(result_path.read_text())
            checked.append(f"{result.get('status')}:{result_path}")
            if result.get("status") == "accepted":
                accepted_source = (source_manifest, entry, directory, result)
                break
        if accepted_source is None:
            raise ValueError(
                "kernel "
                f"{primary_entry['transition_ordinal']} has no accepted source "
                f"in primary/overlay manifests: {checked}"
            )
        source_manifest, entry, directory, result = accepted_source
        entries.append(entry)
        resolved_sources.append((source_manifest, directory))
        candidate_layers.append(
            load_kernel_candidates(
                directory,
                label=f"riser_{int(entry['transition_ordinal']):02d}",
            )
        )
        kernel_results.append(result)

    import zarr

    archive = zarr.open_group(str(manifest["archive_path"]), mode="r")
    joint_names = tuple(str(value) for value in archive["joint_names"][:])
    body_names = tuple(str(value) for value in archive["body_names"][:])
    adapter = _G1FootfallAdapter(
        Path(str(manifest["model_path"])),
        joint_names,
        maximum_joint_correction_rad=0.5,
    )
    body_ids = np.asarray(
        [
            adapter._mujoco.mj_name2id(
                adapter.model,
                adapter._mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            for name in body_names
        ],
        dtype=np.int64,
    )
    target_mesh = _archive_terrain_index(
        archive, int(entries[0]["target_clip_index"])
    )
    first_start = np.asarray(entries[0]["start_xy"], dtype=np.float64)
    last_end = np.asarray(entries[-1]["end_xy"], dtype=np.float64)
    route_direction = last_end - first_start
    route_yaw = math.atan2(float(route_direction[1]), float(route_direction[0]))
    gap_lengths = tuple(
        float(
            np.linalg.norm(
                np.asarray(source["end_xy"], dtype=np.float64)
                - np.asarray(target["start_xy"], dtype=np.float64)
            )
        )
        for source, target in zip(entries, entries[1:])
    )
    matcher = None
    if any(value > float(minimum_flat_bridge_gap_m) for value in gap_lengths):
        matcher = TorchMotionMatcher.from_folder(
            flat_motions,
            device=flat_device,
            config=MatcherConfig(
                trajectory_model="takara_ball",
                max_source_joint_step_rad=0.35,
            ),
        )

    ranked_chains = rank_candidate_chains(
        candidate_layers,
        beam_width=int(candidate_beam_width),
        fps=50.0,
    )
    # Import lazily because the coherent compositor reuses this module's
    # motion I/O and mechanics helpers.
    from .compose_coherent_block_plan import _repair_exact_mesh_clearance

    halflives = (
        (float(inertialization_halflife_s),)
        if inertialization_halflife_s is not None
        else (0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25)
    )
    trial_rows: list[dict[str, object]] = []
    selected = None
    best_diagnostic = None
    bridge_cache: dict[
        tuple[int, int, str, int, str], MotionSegment
    ] = {}
    chain_limit = min(int(maximum_candidate_chains), len(ranked_chains))
    for chain_index, (chain_cost, chain) in enumerate(
        ranked_chains[:chain_limit]
    ):
        accepted_for_chain = []
        kernels = [candidate.segment for candidate in chain]
        segments: list[MotionSegment] = [kernels[0]]
        for seam_index, (source, target) in enumerate(
            zip(kernels, kernels[1:])
        ):
            if gap_lengths[seam_index] > float(minimum_flat_bridge_gap_m):
                if matcher is None:
                    raise AssertionError("flat matcher was not initialized")
                source_candidate = chain[seam_index]
                target_candidate = chain[seam_index + 1]
                bridge_key = (
                    seam_index,
                    int(source_candidate.rank),
                    source_candidate.relative,
                    int(target_candidate.rank),
                    target_candidate.relative,
                )
                if bridge_key not in bridge_cache:
                    bridge_cache[bridge_key] = generate_flat_bridge(
                        matcher,
                        source,
                        target,
                        adapter=adapter,
                        body_ids=body_ids,
                        target_mesh=target_mesh,
                        travel_yaw_world=route_yaw,
                        apply_endpoint_warp=False,
                    )
                segments.append(bridge_cache[bridge_key])
            segments.append(target)
        for halflife in halflives:
            motion = concatenate_segments(
                segments,
                fps=50.0,
                halflife_s=float(halflife),
            )
            mechanics = _maximum_steps(motion)
            row: dict[str, object] = {
                "chain_index": int(chain_index),
                "chain_cost": float(chain_cost),
                "candidate_ranks": [candidate.rank for candidate in chain],
                "inertialization_halflife_s": float(halflife),
                "mechanics": mechanics,
            }
            mechanically_accepted = (
                mechanics["maximum_root_translation_step_m"] <= 0.06
                and mechanics["maximum_root_rotation_step_rad"] <= 0.35
                and mechanics["maximum_joint_step_rad"] <= 0.25
                and mechanics["maximum_root_acceleration_m_s2"] <= 40.0
            )
            if not mechanically_accepted:
                row["status"] = "mechanics_rejected"
                trial_rows.append(row)
                continue
            raw_audit = audit_stair_motion_collisions(
                motion,
                archive_path=manifest["archive_path"],
                target_clip_index=int(entries[0]["target_clip_index"]),
                model_path=manifest["model_path"],
                maximum_foot_penetration_m=0.005,
                maximum_forbidden_body_penetration_m=1.0e-6,
            )
            motion, audit, clearance_repair = _repair_exact_mesh_clearance(
                motion,
                raw_audit,
                archive_path=Path(manifest["archive_path"]),
                target_clip_index=int(entries[0]["target_clip_index"]),
                model_path=Path(manifest["model_path"]),
            )
            mechanics = _maximum_steps(motion)
            row["mechanics"] = mechanics
            row["pre_repair_full_body_audit"] = {
                "accepted": bool(raw_audit.accepted),
                "maximum_foot_penetration_m": float(
                    raw_audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    raw_audit.maximum_forbidden_body_penetration_m
                ),
            }
            row["clearance_repair_maximum_m"] = float(clearance_repair)
            row["full_body_audit"] = {
                "accepted": bool(audit.accepted),
                "maximum_foot_penetration_m": float(
                    audit.maximum_foot_penetration_m
                ),
                "maximum_forbidden_body_penetration_m": float(
                    audit.maximum_forbidden_body_penetration_m
                ),
            }
            if not (
                mechanics["maximum_root_translation_step_m"] <= 0.06
                and mechanics["maximum_root_rotation_step_rad"] <= 0.35
                and mechanics["maximum_joint_step_rad"] <= 0.25
                and mechanics["maximum_root_acceleration_m_s2"] <= 40.0
            ):
                row["status"] = "repaired_mechanics_rejected"
                trial_rows.append(row)
                continue
            row["status"] = "accepted" if audit.accepted else "audit_rejected"
            row["quality_score"] = float(
                mechanics["maximum_root_acceleration_m_s2"] / 25.0
                + mechanics["maximum_seam_root_acceleration_m_s2"] / 20.0
                + mechanics["maximum_seam_joint_acceleration_rad_s2"] / 180.0
                + mechanics["maximum_root_rotation_step_rad"] / 0.35
                + mechanics["maximum_joint_step_rad"] / 0.25
                + float(audit.maximum_foot_penetration_m) / 0.005
            )
            trial_rows.append(row)
            diagnostic_key = (
                float(audit.maximum_forbidden_body_penetration_m),
                float(audit.maximum_foot_penetration_m),
                float(chain_cost),
            )
            if best_diagnostic is None or diagnostic_key < best_diagnostic[0]:
                best_diagnostic = (
                    diagnostic_key,
                    motion,
                    audit,
                    chain,
                    segments,
                    float(chain_cost),
                    float(halflife),
                    float(clearance_repair),
                )
            if audit.accepted:
                accepted_for_chain.append(
                    (
                        float(row["quality_score"]),
                        motion,
                        audit,
                        chain,
                        segments,
                        float(chain_cost),
                        float(halflife),
                        float(clearance_repair),
                    )
                )
        if accepted_for_chain:
            (
                _,
                motion,
                audit,
                chain,
                segments,
                chain_cost,
                selected_halflife,
                selected_clearance_repair,
            ) = min(accepted_for_chain, key=lambda value: value[0])
            selected = (
                motion,
                audit,
                chain,
                segments,
                chain_cost,
                selected_halflife,
                selected_clearance_repair,
            )
            break
    if selected is None:
        if best_diagnostic is None:
            raise RuntimeError("no candidate chain passed the mechanical gate")
        (
            _,
            motion,
            audit,
            chain,
            segments,
            chain_cost,
            selected_halflife,
            selected_clearance_repair,
        ) = (
            best_diagnostic
        )
    else:
        (
            motion,
            audit,
            chain,
            segments,
            chain_cost,
            selected_halflife,
            selected_clearance_repair,
        ) = selected
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    _save_motion(output_path / "motion.npz", motion)
    (output_path / "full_body_collision_audit.json").write_text(
        json.dumps(audit.to_dict(), indent=2) + "\n"
    )
    summary = {
        "schema": "privileged-flat-stair-kernel-composition/v2",
        "target_array_index": int(target_array_index),
        "target_clip_index": int(entries[0]["target_clip_index"]),
        "traversal": str(entries[0]["traversal"]),
        "status": "accepted" if audit.accepted else "audit_rejected",
        "kernel_count": len(chain),
        "flat_bridge_count": sum(
            value > float(minimum_flat_bridge_gap_m) for value in gap_lengths
        ),
        "frame_count": len(motion.root_position_world),
        "duration_s": (len(motion.root_position_world) - 1) / motion.fps,
        "inertialization_halflife_s": float(selected_halflife),
        "clearance_repair_maximum_m": float(selected_clearance_repair),
        "candidate_chain_cost": float(chain_cost),
        "selected_candidates": [
            {
                "transition_ordinal": int(entry["transition_ordinal"]),
                "transition_end_ordinal": int(
                    entry.get("transition_end_ordinal", entry["transition_ordinal"])
                ),
                "rank": int(candidate.rank),
                "relative": candidate.relative,
                "fragment_ids": list(candidate.fragment_ids),
                "maximum_individual_foot_penetration_m": float(
                    candidate.maximum_foot_penetration_m
                ),
            }
            for entry, candidate in zip(entries, chain, strict=True)
        ],
        "candidate_layer_sizes": [len(layer) for layer in candidate_layers],
        "candidate_chain_trials": trial_rows,
        "kernel_gap_lengths_m": list(gap_lengths),
        "minimum_flat_bridge_gap_m": float(minimum_flat_bridge_gap_m),
        "segments": [value.label for value in segments],
        "seam_indices": list(motion.seam_indices),
        "mechanics": _maximum_steps(motion),
        "full_body_audit": {
            "accepted": bool(audit.accepted),
            "maximum_foot_penetration_m": float(
                audit.maximum_foot_penetration_m
            ),
            "maximum_forbidden_body_penetration_m": float(
                audit.maximum_forbidden_body_penetration_m
            ),
        },
        "kernel_results": kernel_results,
        "resolved_kernel_sources": [
            {
                "transition_ordinal": int(entry["transition_ordinal"]),
                "manifest_output_root": str(source_manifest["output_root"]),
                "directory": str(directory),
            }
            for entry, (source_manifest, directory) in zip(
                entries, resolved_sources, strict=True
            )
        ],
        "artifacts": {"motion": "motion.npz"},
    }
    if render:
        media = render_stitched_motion(
            motion,
            archive_path=manifest["archive_path"],
            target_clip_index=int(entries[0]["target_clip_index"]),
            model_path=manifest["model_path"],
            output_path=output_path / "g1_kinematic_50fps.mp4",
        )
        summary["artifacts"]["video"] = Path(media["video"]).name
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-manifest", type=Path, required=True)
    parser.add_argument(
        "--kernel-manifest-overlay", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--block-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--target-array-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flat-motions", type=Path, default=DEFAULT_FLAT_MOTIONS)
    parser.add_argument("--flat-device", default="cpu")
    parser.add_argument("--inertialization-halflife-s", type=float, default=None)
    parser.add_argument("--candidate-beam-width", type=int, default=128)
    parser.add_argument("--maximum-candidate-chains", type=int, default=128)
    parser.add_argument("--minimum-flat-bridge-gap-m", type=float, default=0.08)
    parser.add_argument("--no-render", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = compose_target(
        kernel_manifest_path=arguments.kernel_manifest,
        kernel_manifest_overlays=arguments.kernel_manifest_overlay,
        block_manifest_paths=arguments.block_manifest,
        target_array_index=arguments.target_array_index,
        output=arguments.output,
        flat_motions=arguments.flat_motions,
        flat_device=arguments.flat_device,
        inertialization_halflife_s=arguments.inertialization_halflife_s,
        candidate_beam_width=arguments.candidate_beam_width,
        maximum_candidate_chains=arguments.maximum_candidate_chains,
        minimum_flat_bridge_gap_m=arguments.minimum_flat_bridge_gap_m,
        render=not arguments.no_render,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "KernelCandidate",
    "MotionSegment",
    "compose_target",
    "concatenate_segments",
    "endpoint_seam_cost",
    "endpoint_seam_metrics",
    "generate_flat_bridge",
    "inertialize_segment_entry",
    "load_kernel_candidates",
    "load_kernel_segment",
    "main",
    "rank_candidate_chains",
)
