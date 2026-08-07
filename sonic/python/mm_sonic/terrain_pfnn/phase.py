"""Exact four-channel sole contact and continuous gait-phase reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from mm_sonic.grail_terrain_source import G1MujocoFK
from mm_sonic.joints import ContractError
from mm_sonic.terrain_oracle.canonical import ISAACLAB_BODY_NAMES
from mm_sonic.terrain_oracle.contact import (
    CanonicalMeshQuery,
    ContactConfig,
    SoleGeometry,
    _rotate_wxyz,
)
from mm_sonic.terrain_oracle.math3d import RigidTransform
from mm_sonic.terrain_oracle.source_grail import _load_usd_mesh
from mm_sonic.terrain_pfnn.sources import (
    GrailSlopeRecord,
    PFNNSourceClip,
    load_grail_source,
)


_REASON_NAMES = (
    "unstable_rise",
    "simultaneous_rise",
    "same_side",
    "half_cycle_too_short",
    "half_cycle_too_long",
    "invalid_geometry",
    "backward_phase",
)
_SLOPE_RE = re.compile(r"(?:^|__)(slope_\d{3})(?:__|$)")


@dataclass(frozen=True)
class ContactPhaseTrack:
    contact: np.ndarray
    confidence: np.ndarray
    phase: np.ndarray
    phase_advance: np.ndarray
    valid: np.ndarray


def _strike_events(
    channels: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray, dict[str, int]]:
    foot = np.column_stack(
        (channels[:, 0] | channels[:, 1], channels[:, 2] | channels[:, 3])
    )
    rising = foot & ~np.vstack((np.zeros((1, 2), dtype=bool), foot[:-1]))
    barrier = np.zeros(len(foot), dtype=bool)
    strikes: list[tuple[int, int]] = []
    reasons = {name: 0 for name in _REASON_NAMES}
    for frame in range(len(foot)):
        sides = np.flatnonzero(rising[frame])
        if not len(sides):
            continue
        stable = {
            int(side): bool(
                frame >= 3 and np.all(~foot[frame - 3 : frame, side])
            )
            for side in sides
        }
        reasons["unstable_rise"] += sum(not value for value in stable.values())
        if len(sides) == 2:
            reasons["simultaneous_rise"] += 1
            barrier[frame] = True
        elif stable[int(sides[0])]:
            strikes.append((frame, int(sides[0])))
        else:
            barrier[frame] = True
    return strikes, barrier, reasons


def _phase_reconstruction(
    channels: np.ndarray,
    confidence_array: np.ndarray,
    fps: float,
) -> tuple[ContactPhaseTrack, dict[str, int]]:
    foot = np.column_stack(
        (channels[:, 0] | channels[:, 1], channels[:, 2] | channels[:, 3])
    )
    strikes, barrier, reasons = _strike_events(channels)
    unwrapped = np.zeros(len(foot), dtype=np.float64)
    valid = np.zeros(len(foot), dtype=bool)
    rejected = np.zeros(len(foot), dtype=bool)
    previous_stop_phase: float | None = None
    previous_stop_frame: int | None = None
    valid_cycles: list[tuple[int, int, int, float]] = []
    tainted_strikes: set[int] = set()
    for (start, side), (stop, next_side) in zip(strikes[:-1], strikes[1:]):
        duration = (stop - start) / float(fps)
        barrier_between = bool(np.any(barrier[start + 1 : stop]))
        if start in tainted_strikes:
            rejected[start:stop] = True
            previous_stop_phase = None
            previous_stop_frame = None
            continue
        if barrier_between:
            rejected[start:stop] = True
            previous_stop_phase = None
            previous_stop_frame = None
            continue
        if next_side == side:
            reasons["same_side"] += 1
            rejected[start:stop] = True
            previous_stop_phase = None
            previous_stop_frame = None
            continue
        if duration < 0.20:
            reasons["half_cycle_too_short"] += 1
            rejected[start : stop + 1] = True
            tainted_strikes.update((start, stop))
            previous_stop_phase = None
            previous_stop_frame = None
            continue
        if duration > 1.00:
            reasons["half_cycle_too_long"] += 1
            rejected[start:stop] = True
            previous_stop_phase = None
            previous_stop_frame = None
            continue
        base_phase = 0.0 if side == 0 else math.pi
        if previous_stop_frame == start and previous_stop_phase is not None:
            start_phase = previous_stop_phase
        else:
            start_phase = base_phase
        stop_phase = start_phase + math.pi
        unwrapped[start : stop + 1] = np.linspace(
            start_phase, stop_phase, stop - start + 1
        )
        valid[start : stop + 1] = True
        valid_cycles.append((start, stop, side, start_phase))
        previous_stop_phase = stop_phase
        previous_stop_frame = stop

    valid[rejected] = False
    if np.any(valid):
        last = int(np.flatnonzero(valid)[-1])
        bilateral = foot[:, 0] & foot[:, 1]
        first_valid_cycle = next(
            (cycle for cycle in valid_cycles if valid[cycle[0]]), None
        )
        if first_valid_cycle is not None and bilateral[0]:
            start, stop, side, start_phase = first_valid_cycle
            prefix_end = 0
            while prefix_end + 1 < start and bilateral[prefix_end + 1]:
                prefix_end += 1
            bridge = slice(prefix_end + 1, start)
            bridge_length = start - prefix_end - 1
            if (
                bridge_length >= 3
                and not np.any(foot[bridge, side])
                and np.all(foot[bridge, 1 - side])
            ):
                omega = math.pi / (stop - start)
                boundary_phase = start_phase - omega * (start - prefix_end)
                unwrapped[: prefix_end + 1] = boundary_phase
                bridge_frames = np.arange(prefix_end + 1, start)
                unwrapped[bridge] = start_phase - omega * (start - bridge_frames)
                valid[:start] = True
        trailing = last
        while trailing + 1 < len(foot) and bilateral[trailing + 1]:
            trailing += 1
        if trailing > last:
            unwrapped[last + 1 : trailing + 1] = unwrapped[last]
            valid[last + 1 : trailing + 1] = True

    valid[rejected] = False
    delta = np.diff(unwrapped)
    backward = np.flatnonzero(valid[:-1] & valid[1:] & (delta < 0.0))
    reasons["backward_phase"] = int(len(backward))
    if len(backward):
        valid[backward] = False
        valid[backward + 1] = False

    wrapped = np.remainder(unwrapped, 2.0 * math.pi)
    advance = np.zeros(len(foot), dtype=np.float64)
    advance[:-1] = np.where(valid[:-1] & valid[1:], delta, 0.0)
    return (
        ContactPhaseTrack(
            channels,
            confidence_array,
            wrapped.astype(np.float32),
            advance.astype(np.float32),
            valid,
        ),
        reasons,
    )


def phase_from_contacts(
    contact: object,
    *,
    confidence: object | None = None,
    fps: float,
) -> ContactPhaseTrack:
    channels = np.asarray(contact, dtype=bool)
    if (
        channels.ndim != 2
        or channels.shape[1] != 4
        or not np.isfinite(fps)
        or fps <= 0.0
    ):
        raise ValueError("phase reconstruction expects contact[T,4] and positive fps")
    if float(fps) != 30.0:
        raise ValueError("phase reconstruction requires exactly 30 Hz")
    confidence_array = (
        channels.astype(np.float32)
        if confidence is None
        else np.asarray(confidence, dtype=np.float32)
    )
    if confidence_array.shape != channels.shape or not np.isfinite(
        confidence_array
    ).all():
        raise ValueError("confidence must be finite with shape [T,4]")
    track, _ = _phase_reconstruction(channels, confidence_array, float(fps))
    return track


def reconstruct_heel_toe_contacts(
    source: PFNNSourceClip,
    query: CanonicalMeshQuery,
    geometry: SoleGeometry,
) -> ContactPhaseTrack:
    """Reconstruct left heel/toe then right heel/toe against exact terrain."""

    if not isinstance(source, PFNNSourceClip):
        raise TypeError("source must be a PFNNSourceClip")
    if not isinstance(query, CanonicalMeshQuery):
        raise TypeError("query must be a CanonicalMeshQuery")
    if (
        source.terrain_sha256 is not None
        and source.terrain_sha256 != query.source_asset_sha256
    ):
        raise ValueError("invalid geometry: paired terrain digest mismatch")
    try:
        config = ContactConfig(geometry=geometry)
    except ContractError as error:
        raise ValueError(f"invalid geometry: {error}") from error
    try:
        body_indices = np.array(
            [ISAACLAB_BODY_NAMES.index(name) for name in geometry.body_names],
            dtype=np.int64,
        )
    except ValueError as error:
        raise ValueError("source is missing a named ankle-roll body") from error

    body_position = np.asarray(source.body_position_world[:, body_indices], dtype=np.float64)
    body_quaternion = np.asarray(
        source.body_quaternion_world_wxyz[:, body_indices], dtype=np.float64
    )
    body_linear_velocity = np.asarray(
        source.body_linear_velocity_world[:, body_indices], dtype=np.float64
    )
    body_angular_velocity = np.asarray(
        source.body_angular_velocity_world[:, body_indices], dtype=np.float64
    )
    corners = np.asarray(geometry.corner_positions_body, dtype=np.float64)
    grouped_offsets = np.stack(
        (corners[0, :2], corners[0, 2:], corners[1, :2], corners[1, 2:]),
        axis=0,
    )
    channel_foot = np.array((0, 0, 1, 1), dtype=np.int64)
    channel_position = body_position[:, channel_foot]
    channel_quaternion = body_quaternion[:, channel_foot]
    channel_linear_velocity = body_linear_velocity[:, channel_foot]
    channel_angular_velocity = body_angular_velocity[:, channel_foot]
    rotated_probes = _rotate_wxyz(
        channel_quaternion[:, :, None, :],
        np.broadcast_to(
            grouped_offsets[None, ...],
            (source.frame_count, 4, grouped_offsets.shape[1], 3),
        ),
    )
    probe_world = channel_position[:, :, None, :] + rotated_probes
    try:
        surface = query.query(probe_world.reshape((-1, 3)))
    except (AttributeError, ContractError, TypeError, ValueError) as error:
        raise ValueError(f"invalid geometry: surface query failed: {error}") from error
    shape = (source.frame_count, 4, grouped_offsets.shape[1])
    try:
        closest_distance = np.asarray(surface.distance_m).reshape(shape)
        closest_normal = np.asarray(surface.surface_normal_world).reshape((*shape, 3))
        ray_distance = np.asarray(surface.downward_ray_distance_m).reshape(shape)
        ray_normal = np.asarray(surface.downward_ray_normal_world).reshape((*shape, 3))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid geometry: malformed surface query result: {error}"
        ) from error
    if (
        not np.isfinite(closest_distance).all()
        or not np.isfinite(closest_normal).all()
        or np.isnan(ray_distance).any()
        or not np.isfinite(ray_normal).all()
    ):
        raise ValueError("invalid geometry: nonfinite surface query result")
    probe_velocity = channel_linear_velocity[:, :, None, :] + np.cross(
        channel_angular_velocity[:, :, None, :], rotated_probes
    )
    angular_speed = np.linalg.norm(channel_angular_velocity, axis=-1)

    distance = np.array(closest_distance, copy=True)
    normal = np.array(closest_normal, copy=True)
    maximum_tangential_speed = np.empty((source.frame_count, 4), dtype=np.float64)
    contact = np.zeros((source.frame_count, 4), dtype=bool)
    for channel in range(4):
        active = False
        for frame in range(source.frame_count):
            if active:
                distance_threshold = config.leave_distance_m
                tangential_threshold = config.leave_tangential_speed_m_s
                angular_threshold = config.leave_angular_speed_rad_s
            else:
                distance_threshold = config.enter_distance_m
                tangential_threshold = config.enter_tangential_speed_m_s
                angular_threshold = config.enter_angular_speed_rad_s
            ray_support = (
                np.isfinite(ray_distance[frame, channel])
                & (ray_distance[frame, channel] <= distance_threshold)
                & (ray_normal[frame, channel, :, 2] >= config.minimum_surface_normal_z)
            )
            closest_support = (
                (closest_distance[frame, channel] <= distance_threshold)
                & (
                    closest_normal[frame, channel, :, 2]
                    >= config.minimum_surface_normal_z
                )
            )
            distance[frame, channel] = np.where(
                ray_support,
                ray_distance[frame, channel],
                closest_distance[frame, channel],
            )
            normal[frame, channel] = np.where(
                ray_support[:, None],
                ray_normal[frame, channel],
                closest_normal[frame, channel],
            )
            normal_velocity = np.sum(
                probe_velocity[frame, channel] * normal[frame, channel], axis=-1
            )
            tangential_velocity = probe_velocity[frame, channel] - (
                normal_velocity[:, None] * normal[frame, channel]
            )
            maximum_tangential_speed[frame, channel] = float(
                np.max(np.linalg.norm(tangential_velocity, axis=-1))
            )
            active = bool(
                np.all(ray_support | closest_support)
                and maximum_tangential_speed[frame, channel] <= tangential_threshold
                and angular_speed[frame, channel] <= angular_threshold
            )
            contact[frame, channel] = active

    distance_score = np.clip(1.0 - distance / config.leave_distance_m, 0.0, 1.0)
    normal_score = np.clip(
        (normal[..., 2] - config.minimum_surface_normal_z)
        / (1.0 - config.minimum_surface_normal_z),
        0.0,
        1.0,
    )
    support_score = np.mean(distance_score * normal_score, axis=2)
    tangential_score = np.clip(
        1.0
        - maximum_tangential_speed / config.leave_tangential_speed_m_s,
        0.0,
        1.0,
    )
    angular_score = np.clip(
        1.0 - angular_speed / config.leave_angular_speed_rad_s, 0.0, 1.0
    )
    confidence = np.clip(
        support_score * tangential_score * angular_score, 0.0, 1.0
    ).astype(np.float32)
    if not np.isfinite(confidence).all():
        raise ValueError("contact reconstruction produced nonfinite confidence")
    return phase_from_contacts(
        contact, confidence=confidence, fps=float(source.fps)
    )


def _audit(arguments: argparse.Namespace) -> dict[str, object]:
    import mujoco

    robot = arguments.robot.expanduser().resolve()
    terrain = arguments.terrain.expanduser().resolve()
    model_path = arguments.model_path.expanduser().resolve()
    if robot.stem != terrain.stem:
        raise ValueError("robot and terrain must have the same source stem")
    match = _SLOPE_RE.search(robot.stem)
    if match is None:
        raise ValueError("source stem must contain slope_NNN")
    record = GrailSlopeRecord(
        stem=robot.stem,
        terrain_id=match.group(1),
        robot_path=robot,
        terrain_path=terrain,
    )
    source = load_grail_source(record, G1MujocoFK(model_path))
    terrain_sha = hashlib.sha256(terrain.read_bytes()).hexdigest()
    mesh = _load_usd_mesh(terrain, source_asset_sha256=terrain_sha)
    query = CanonicalMeshQuery(
        mesh,
        RigidTransform(
            source.terrain_position_world,
            source.terrain_quaternion_world_from_usd_wxyz,
        ),
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    geometry = SoleGeometry.from_model(model)
    track = reconstruct_heel_toe_contacts(source, query, geometry)
    _, reasons = _phase_reconstruction(
        track.contact, track.confidence, float(source.fps)
    )
    valid_count = int(np.count_nonzero(track.valid))
    return {
        "clip_id": source.clip_id,
        "frame_count": source.frame_count,
        "channel_order": ["left_heel", "left_toe", "right_heel", "right_toe"],
        "contact_frames": np.count_nonzero(track.contact, axis=0).astype(int).tolist(),
        "mean_confidence": np.mean(track.confidence, axis=0).tolist(),
        "valid_phase_frames": valid_count,
        "valid_phase_fraction": valid_count / source.frame_count,
        "finite_phase": bool(
            np.isfinite(track.phase).all() and np.isfinite(track.phase_advance).all()
        ),
        "maximum_phase_advance": float(np.max(track.phase_advance)),
        "rejected_reason_counts": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(_audit(arguments), sort_keys=True))
    return 0


__all__ = (
    "ContactPhaseTrack",
    "phase_from_contacts",
    "reconstruct_heel_toe_contacts",
)


if __name__ == "__main__":
    raise SystemExit(main())
