"""Build a contact-annotated terrain database from clean Justin stair clips."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .joints import ContractError
from .terrain_motion import TerrainFrameDatabase, wrap_angle


JUSTIN_STAIR_RUN_M = 0.3302
JUSTIN_STAIR_RISE_M = 0.1778
JUSTIN_STAIR_WIDTH_M = 0.5704
JUSTIN_STAIR_TREADS = 3
# Physical midpoint of the first tread's leading edge in Justin's canonical
# MuJoCo scene.  The staircase ascends toward -X, so the leading edge is the
# +X face of ``multi_boxes_link_1``.  The former (0.05, -4.36) value was a
# nearby scene reference, not the stair boundary; using it displaced every
# re-instanced pose by 260.65 mm (most of one tread) and put feet/knees through
# the procedural staircase.
JUSTIN_STAIR_ORIGIN_WORLD_XY = np.asarray(
    (-0.21065, -4.3529), dtype=np.float32
)
JUSTIN_STAIR_ASCENT_YAW_WORLD = np.pi
FUTURE_OFFSETS = (6, 12, 18, 24)


@dataclass(frozen=True)
class SourceAudit:
    source: str
    readable: bool
    clean_kinematics: bool
    clip_count: int
    frame_count: int
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "readable": self.readable,
            "clean_kinematics": self.clean_kinematics,
            "clip_count": self.clip_count, "frame_count": self.frame_count,
            "notes": list(self.notes),
        }


def _yaw_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(quaternion), -1, 0)
    return np.arctan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def _world_to_root_xy(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((c * vector[..., 0] + s * vector[..., 1],
                     -s * vector[..., 0] + c * vector[..., 1]), axis=-1)


def _world_to_stair_xyz(position: np.ndarray) -> np.ndarray:
    delta = np.asarray(position) - np.asarray(
        (JUSTIN_STAIR_ORIGIN_WORLD_XY[0], JUSTIN_STAIR_ORIGIN_WORLD_XY[1], 0.0))
    c, s = np.cos(JUSTIN_STAIR_ASCENT_YAW_WORLD), np.sin(JUSTIN_STAIR_ASCENT_YAW_WORLD)
    return np.stack((c * delta[..., 0] + s * delta[..., 1],
                     -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), axis=-1)


def _contact_annotation(feet_world: np.ndarray, feet_velocity_world: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    feet_stair = _world_to_stair_xyz(feet_world)
    level = np.rint((feet_stair[..., 2] - 0.035) / JUSTIN_STAIR_RISE_M).astype(np.int16)
    level = np.clip(level, 0, JUSTIN_STAIR_TREADS)
    expected_z = level.astype(np.float32) * JUSTIN_STAIR_RISE_M + 0.035
    height_error = np.abs(feet_stair[..., 2] - expected_z)
    speed = np.linalg.norm(feet_velocity_world, axis=-1)
    contact = (height_error <= 0.045) & (speed <= 0.22)
    tread = np.where(contact, level, -1).astype(np.int16)
    return contact, tread


def _first_stable_run(
    mask: np.ndarray, *, minimum_frames: int = 5
) -> int | None:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ContractError("stable-run mask must be one-dimensional")
    minimum = int(minimum_frames)
    if minimum <= 0:
        raise ContractError("minimum stable-run length must be positive")
    padded = np.concatenate(
        (
            np.asarray((False,)),
            values,
            np.asarray((False,)),
        )
    )
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(changes[::2], changes[1::2], strict=True):
        if int(end - start) >= minimum:
            return int(start)
    return None


def _annotate_transition_windows(
    clip_name: str,
    contact: np.ndarray,
    tread_id: np.ndarray,
    feet_xyz_stair: np.ndarray,
    *,
    usable_frame_count: int,
    entry_lookback_frames: int = 15,
    entry_follow_frames: int = 5,
    exit_lead_frames: int = 20,
    minimum_landing_support_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Mark flat↔stair splice windows from traversal semantics.

    A tread-height label alone cannot distinguish the top tread from a top
    landing, and "last stair contact" is undefined for clips that finish
    standing on the third tread.  Direction and stable endpoint support make
    those events unambiguous:

    - ascent entry is the first supported nonzero tread;
    - ascent landing is stable double support at the top level;
    - descent entry is the first supported level below the top;
    - descent landing is stable double support at ground level.

    Continuous mid-stair loops are useful search candidates but are never
    legal flat-ground handoff clips.
    """

    name = str(clip_name)
    contacts = np.asarray(contact, dtype=bool)
    tread = np.asarray(tread_id, dtype=np.int16)
    feet = np.asarray(feet_xyz_stair, dtype=np.float32)
    if (
        contacts.ndim != 2
        or contacts.shape[1:] != (2,)
        or tread.shape != contacts.shape
        or feet.shape != (len(contacts), 2, 3)
    ):
        raise ContractError("transition annotations require [T,2] feet data")
    count = len(contacts)
    usable = int(usable_frame_count)
    if not 0 <= usable <= count:
        raise ContractError("usable_frame_count is outside the clip")
    entry = np.zeros(count, dtype=bool)
    exit_ = np.zeros(count, dtype=bool)
    if "continuous" in name:
        return entry, exit_
    is_up = name.startswith("up_") or name.startswith("grail_up_")
    is_down = name.startswith("down_") or name.startswith("grail_down_")
    if not (is_up or is_down):
        raise ContractError(f"unknown Justin traversal direction: {name}")

    length = JUSTIN_STAIR_RUN_M * JUSTIN_STAIR_TREADS
    in_stair_footprint = (
        (feet[..., 0] >= 0.0)
        & (feet[..., 0] <= length)
        & (np.abs(feet[..., 1]) <= JUSTIN_STAIR_WIDTH_M * 0.5 + 0.05)
    )
    if is_up:
        entry_support = contacts & in_stair_footprint & (tread >= 1)
        landing_support = np.all(
            contacts & (tread == JUSTIN_STAIR_TREADS), axis=1
        )
    else:
        entry_support = (
            contacts
            & in_stair_footprint
            & (tread >= 0)
            & (tread < JUSTIN_STAIR_TREADS)
        )
        landing_support = np.all(contacts & (tread == 0), axis=1)

    entry_frames = np.flatnonzero(np.any(entry_support, axis=1))
    if entry_frames.size:
        event = int(entry_frames[0])
        first = max(0, event - int(entry_lookback_frames))
        last = min(usable, event + int(entry_follow_frames) + 1)
        if first < last:
            entry[first:last] = True

    landing = _first_stable_run(
        landing_support,
        minimum_frames=minimum_landing_support_frames,
    )
    if landing is not None and landing <= usable:
        first = max(0, landing - int(exit_lead_frames))
        last = min(usable, landing)
        if first < last:
            exit_[first:last] = True
    return entry, exit_


def _select_feature_scale(
    feature: object,
    searchable: object,
    *,
    reference_feature_scale: object | None = None,
) -> tuple[np.ndarray, str]:
    """Choose a stable matcher metric for an expanded motion catalog."""

    values = np.asarray(feature, dtype=np.float32)
    mask = np.asarray(searchable, dtype=bool)
    if (
        values.ndim != 2
        or len(values) == 0
        or mask.shape != (len(values),)
        or not np.any(mask)
        or not np.isfinite(values).all()
    ):
        raise ContractError("feature-scale inputs are invalid")
    if reference_feature_scale is None:
        scale = np.std(values[mask], axis=0).astype(np.float32)
        return (
            np.ascontiguousarray(np.maximum(scale, 1.0e-3)),
            "computed_from_searchable_rows",
        )
    reference = np.asarray(reference_feature_scale, dtype=np.float32)
    if (
        reference.shape != (values.shape[1],)
        or not np.isfinite(reference).all()
        or np.any(reference <= 0.0)
    ):
        raise ContractError(
            "reference feature scale must be finite, positive, and match "
            "the feature dimension"
        )
    return np.ascontiguousarray(reference), "reference_catalog"


def build_justin_database(
    path: str | Path,
    *,
    reference_feature_scale: object | None = None,
    entry_lookback_frames: int = 15,
) -> tuple[TerrainFrameDatabase, dict[str, Any]]:
    """Load the 18-clip clean zarr and derive searchable contact frames."""
    try:
        import zarr
    except ImportError as error:
        raise ContractError("zarr is required to read the Justin archive") from error
    root = zarr.open(str(Path(path)), mode="r")
    entry_lookback = int(entry_lookback_frames)
    if entry_lookback < 0:
        raise ContractError("entry_lookback_frames must be nonnegative")
    required = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "clip_names",
                "clip_start_idx", "clip_end_idx", "joint_pos", "joint_vel")
    if any(name not in root for name in required):
        raise ContractError("Justin archive is missing required arrays")
    if root.attrs.get("quaternion_convention") != "xyzw":
        raise ContractError("Justin archive must declare xyzw quaternions")
    body_pos = np.asarray(root["body_pos_w"], dtype=np.float32)
    body_quat = np.asarray(root["body_quat_w"], dtype=np.float32)
    body_vel = np.asarray(root["body_lin_vel_w"], dtype=np.float32)
    joint_pos = np.asarray(root["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(root["joint_vel"], dtype=np.float32)
    starts = np.asarray(root["clip_start_idx"], dtype=np.int64)
    ends = np.asarray(root["clip_end_idx"], dtype=np.int64)
    names = [str(value) for value in np.asarray(root["clip_names"])]
    n = len(body_pos)
    if body_pos.shape != (n, 30, 3) or body_quat.shape != (n, 30, 4):
        raise ContractError("Justin archive has an unexpected G1 body layout")
    yaw = _yaw_xyzw(body_quat[:, 0])
    origin_delta = JUSTIN_STAIR_ORIGIN_WORLD_XY[None, :] - body_pos[:, 0, :2]
    stair_origin_root = _world_to_root_xy(origin_delta, yaw)
    stair_yaw_root = wrap_angle(JUSTIN_STAIR_ASCENT_YAW_WORLD - yaw).astype(np.float32)
    feet_world = body_pos[:, (18, 19)]
    feet_stair = _world_to_stair_xyz(feet_world).astype(np.float32)
    contact, tread = _contact_annotation(feet_world, body_vel[:, (18, 19)])
    root_stair = _world_to_stair_xyz(body_pos[:, 0])

    offsets = np.asarray(FUTURE_OFFSETS, dtype=np.int64)
    future_root = np.empty((n, len(offsets), 2), dtype=np.float32)
    future_facing = np.empty_like(future_root)
    future_feet = np.empty((n, len(offsets), 2, 3), dtype=np.float32)
    future_contact = np.empty((n, len(offsets), 2), dtype=bool)
    future_tread = np.empty((n, len(offsets), 2), dtype=np.int16)
    source_clip = np.empty(n, dtype=np.int32)
    source_frame = np.empty(n, dtype=np.int32)
    entry = np.zeros(n, dtype=bool)
    exit = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    for clip, (start, end) in enumerate(zip(starts, ends, strict=True)):
        source_clip[start:end] = clip
        source_frame[start:end] = np.arange(end - start, dtype=np.int32)
        future = np.minimum(np.arange(start, end)[:, None] + offsets[None, :], end - 1)
        delta = body_pos[future, 0, :2] - body_pos[start:end, 0, :2, None].transpose(0, 2, 1)
        future_root[start:end] = _world_to_root_xy(delta, yaw[start:end, None])
        delta_yaw = yaw[future] - yaw[start:end, None]
        future_facing[start:end, :, 0] = np.cos(delta_yaw)
        future_facing[start:end, :, 1] = np.sin(delta_yaw)
        future_feet[start:end] = feet_stair[future]
        future_contact[start:end] = contact[future]
        future_tread[start:end] = tread[future]
        usable_end = max(start, end - int(offsets[-1]))
        valid[start:usable_end] = True
        clip_entry, clip_exit = _annotate_transition_windows(
            names[clip],
            contact[start:end],
            tread[start:end],
            feet_stair[start:end],
            usable_frame_count=usable_end - start,
            entry_lookback_frames=entry_lookback,
        )
        entry[start:end] = clip_entry
        exit[start:end] = clip_exit

    # Pose feature is intentionally joint/body-local; environment alignment is
    # kept out of normalization and scored separately by TerrainMotionMatcher.
    root_velocity_local = _world_to_root_xy(body_vel[:, 0, :2], yaw)
    feature = np.concatenate((joint_pos, joint_vel, root_velocity_local), axis=1)
    finite = np.isfinite(feature).all(axis=1)
    searchable = valid & finite
    scale, normalization_provenance = _select_feature_scale(
        feature,
        searchable,
        reference_feature_scale=reference_feature_scale,
    )
    database = TerrainFrameDatabase(
        feature=feature[searchable], feature_scale=scale,
        source_clip=source_clip[searchable], source_frame=source_frame[searchable],
        stair_origin_root_xy=stair_origin_root[searchable],
        stair_ascent_yaw_root=stair_yaw_root[searchable],
        root_xy_stair=root_stair[searchable, :2],
        root_height_above_stair_base_m=root_stair[searchable, 2],
        contact=contact[searchable], tread_id=tread[searchable],
        feet_xyz_stair=feet_stair[searchable],
        future_root_xy=future_root[searchable],
        future_facing_xy=future_facing[searchable],
        future_feet_xyz_stair=future_feet[searchable],
        future_contact=future_contact[searchable],
        future_tread_id=future_tread[searchable],
        inferred_travel_velocity_local_xy=(
            future_root[searchable, -1] / (FUTURE_OFFSETS[-1] / 50.0)),
        inferred_facing_local_xy=future_facing[searchable, -1],
        entry=entry[searchable], exit=exit[searchable],
    )
    report = {
        "schema": "sonic-terrain-catalog/v1", "source": str(Path(path).resolve()),
        "quaternion_convention": "xyzw", "fps": int(np.asarray(root["fps"])[0]),
        "clips": names, "clip_count": len(names), "frame_count": n,
        "searchable_frame_count": database.row_count,
        "command_provenance": (
            "inverse-fitted from clean +24-frame root displacement and facing; "
            "not recorded human joystick input"),
        "future_knot_offsets_frames": list(FUTURE_OFFSETS),
        "feature_scale_provenance": normalization_provenance,
        "entry_lookback_frames": entry_lookback,
        "contact_frame_count": int(np.count_nonzero(np.any(contact, axis=1))),
        "stair_contact_frame_count": int(np.count_nonzero(np.any(contact & (tread > 0), axis=1))),
        "geometry": {"run_m": JUSTIN_STAIR_RUN_M, "rise_m": JUSTIN_STAIR_RISE_M,
                     "width_m": JUSTIN_STAIR_WIDTH_M, "tread_count": JUSTIN_STAIR_TREADS},
    }
    return database, report


def write_database(path: str | Path, database: TerrainFrameDatabase,
                   report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(database, name) for name in database.__dataclass_fields__}
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


def load_database(path: str | Path) -> TerrainFrameDatabase:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = tuple(TerrainFrameDatabase.__dataclass_fields__)
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ContractError(f"terrain database is missing arrays: {missing}")
        values = {name: np.asarray(archive[name]) for name in required}
    return TerrainFrameDatabase(**values)
