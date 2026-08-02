"""Deterministic mechanical audit for canonical G1 source motion."""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, NamedTuple

import numpy as np

from mm_sonic.joints import ContractError

from .canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
    CanonicalClip,
)
from .contact import CanonicalMeshQuery
from .math3d import angular_velocity_world_wxyz, finite_difference


_SCHEMA = "terrain-oracle-audit/v1"
_SHA256 = frozenset("0123456789abcdef")
_FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
_REASON_CODES = frozenset(
    (
        "joint_limit",
        "joint_velocity",
        "quaternion_discontinuity",
        "root_plausibility",
        "derivative_discontinuity",
        "contact_inconsistency",
        "incomplete_sole_support",
        "stance_skate",
        "foot_penetration",
        "body_penetration",
        "terrain_registration",
        "model_registration",
        "source_registration",
    )
)
_METRIC_NAMES = (
    "max_joint_limit_violation_rad",
    "max_joint_speed_rad_s",
    "max_quaternion_step_rad",
    "max_root_speed_m_s",
    "min_root_height_m",
    "max_root_height_m",
    "max_root_acceleration_m_s2",
    "max_root_angular_speed_rad_s",
    "max_derivative_error",
    "max_contact_disagreement",
    "max_incomplete_sole_fraction",
    "max_stance_skate_m",
    "max_foot_penetration_m",
    "max_body_penetration_m",
)


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def _sha(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


class AuditInterval(NamedTuple):
    """Immutable half-open frame interval."""

    start_frame: int
    end_frame: int


def _interval(value: object, frame_count: int, label: str) -> AuditInterval:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or type(value[0]) is not int
        or type(value[1]) is not int
        or not 0 <= value[0] < value[1] <= frame_count
    ):
        raise ContractError(f"{label} must be a valid half-open frame interval")
    return AuditInterval(value[0], value[1])


@dataclass(frozen=True)
class AuditThresholds:
    """Every numerical audit decision, with no module-private magic threshold."""

    joint_limit_tolerance_rad: float = 1.0e-5
    joint_velocity_limit_rad_s: float = 40.0
    quaternion_step_limit_rad: float = 1.0
    root_speed_limit_m_s: float = 5.0
    root_height_min_m: float = 0.25
    root_height_max_m: float = 2.0
    root_acceleration_limit_m_s2: float = 50.0
    root_angular_speed_limit_rad_s: float = 12.0
    derivative_root_linear_tolerance_m_s: float = 0.02
    derivative_root_angular_tolerance_rad_s: float = 0.05
    derivative_joint_tolerance_rad_s: float = 0.10
    derivative_body_linear_tolerance_m_s: float = 0.02
    derivative_body_angular_tolerance_rad_s: float = 0.05
    derivative_acceleration_limit_rad_s2: float = 500.0
    support_distance_m: float = 0.012
    minimum_surface_normal_z: float = 0.50
    contact_label_threshold: float = 0.50
    stance_skate_limit_m: float = 0.03
    foot_penetration_tolerance_m: float = 0.005
    body_penetration_tolerance_m: float = 0.005
    collision_epsilon_m: float = 1.0e-9
    cylinder_radial_segments: int = 32
    guard_frames: int = 2
    minimum_interval_frames: int = 8

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in (
                "cylinder_radial_segments",
                "guard_frames",
                "minimum_interval_frames",
            ):
                if type(value) is not int or value < (
                    3 if field.name == "cylinder_radial_segments" else 0
                ):
                    raise ContractError(f"{field.name} is invalid")
            elif not isinstance(value, (int, float)) or not math.isfinite(
                float(value)
            ):
                raise ContractError(f"{field.name} must be finite")
        if (
            self.joint_limit_tolerance_rad < 0.0
            or self.joint_velocity_limit_rad_s <= 0.0
            or self.quaternion_step_limit_rad <= 0.0
            or self.root_speed_limit_m_s <= 0.0
            or self.root_height_min_m < 0.0
            or self.root_height_max_m <= self.root_height_min_m
            or self.root_acceleration_limit_m_s2 <= 0.0
            or self.root_angular_speed_limit_rad_s <= 0.0
            or min(
                self.derivative_root_linear_tolerance_m_s,
                self.derivative_root_angular_tolerance_rad_s,
                self.derivative_joint_tolerance_rad_s,
                self.derivative_body_linear_tolerance_m_s,
                self.derivative_body_angular_tolerance_rad_s,
                self.derivative_acceleration_limit_rad_s2,
                self.support_distance_m,
                self.stance_skate_limit_m,
                self.collision_epsilon_m,
            )
            <= 0.0
            or not 0.0 <= self.minimum_surface_normal_z < 1.0
            or not 0.0 < self.contact_label_threshold < 1.0
            or self.foot_penetration_tolerance_m < 0.0
            or self.body_penetration_tolerance_m < 0.0
            or self.minimum_interval_frames < 1
        ):
            raise ContractError("audit thresholds violate their ranges")

    def to_dict(self) -> dict[str, float | int]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }


@dataclass(frozen=True)
class AuditReason:
    code: str
    severity: str
    frame_interval: AuditInterval | tuple[int, int]
    observed_maximum: float
    threshold: float

    def __post_init__(self) -> None:
        if self.code not in _REASON_CODES:
            raise ContractError("unknown audit reason code")
        if self.severity not in ("error", "fatal"):
            raise ContractError("audit reason severity must be error or fatal")
        if (
            not isinstance(self.frame_interval, (tuple, list))
            or len(self.frame_interval) != 2
            or type(self.frame_interval[1]) is not int
        ):
            raise ContractError("reason frame_interval must be half-open")
        interval = _interval(
            self.frame_interval,
            max(2, self.frame_interval[1]),
            "reason frame_interval",
        )
        if interval.end_frame <= interval.start_frame:
            raise ContractError("reason frame_interval must be half-open")
        object.__setattr__(self, "frame_interval", interval)
        _finite(self.observed_maximum, "reason observed maximum")
        _finite(self.threshold, "reason threshold")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "frame_interval": list(self.frame_interval),
            "observed_maximum": float(self.observed_maximum),
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class ClipAudit:
    schema: str
    clip_id: str
    frame_count: int
    source_sha256: str
    model_sha256: str
    terrain_sha256: str
    status: str
    accepted_intervals: tuple[AuditInterval | tuple[int, int], ...]
    thresholds: Mapping[str, float | int]
    metrics: Mapping[str, float]
    reasons: tuple[AuditReason, ...]

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ContractError("unknown audit schema")
        if type(self.clip_id) is not str or not self.clip_id:
            raise ContractError("audit clip_id must be nonempty")
        if type(self.frame_count) is not int or self.frame_count < 2:
            raise ContractError("audit frame_count must be at least two")
        for name in ("source_sha256", "model_sha256", "terrain_sha256"):
            _sha(getattr(self, name), name)
        if self.status not in (
            "accepted",
            "accepted_with_intervals_removed",
            "rejected",
        ):
            raise ContractError("invalid audit status")
        intervals = tuple(
            _interval(value, self.frame_count, "accepted interval")
            for value in self.accepted_intervals
        )
        if any(
            intervals[index - 1][1] >= intervals[index][0]
            for index in range(1, len(intervals))
        ):
            raise ContractError("accepted intervals must be sorted and disjoint")
        threshold_names = tuple(field.name for field in fields(AuditThresholds))
        if set(self.thresholds) != set(threshold_names):
            raise ContractError("audit threshold map is incomplete")
        threshold_values = {
            name: getattr(AuditThresholds(**dict(self.thresholds)), name)
            for name in threshold_names
        }
        if set(self.metrics) != set(_METRIC_NAMES):
            raise ContractError("audit metric map is incomplete")
        metric_values = {
            name: _finite(self.metrics[name], f"metric {name}")
            for name in _METRIC_NAMES
        }
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, AuditReason) for reason in reasons):
            raise ContractError("audit reasons must be AuditReason values")
        object.__setattr__(self, "accepted_intervals", intervals)
        object.__setattr__(
            self, "thresholds", MappingProxyType(threshold_values)
        )
        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "clip_id": self.clip_id,
            "frame_count": self.frame_count,
            "source_sha256": self.source_sha256,
            "model_sha256": self.model_sha256,
            "terrain_sha256": self.terrain_sha256,
            "status": self.status,
            "accepted_intervals": [
                list(interval) for interval in self.accepted_intervals
            ],
            "thresholds": dict(self.thresholds),
            "metrics": dict(self.metrics),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result


def _update_hash_text(digest: "hashlib._Hash", value: object) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_hash_array(
    digest: "hashlib._Hash", value: object, dtype: np.dtype
) -> None:
    array = np.ascontiguousarray(value, dtype=dtype)
    _update_hash_text(digest, array.shape)
    digest.update(array.tobytes(order="C"))


def _names(model: object, kind: str, count: int) -> tuple[str, ...]:
    accessor = getattr(model, kind)
    return tuple(str(accessor(index).name) for index in range(count))


def structural_model_sha256(model: object) -> str:
    """Hash exact structural mechanics used by this audit, not source XML bytes."""

    digest = hashlib.sha256(b"terrain-oracle-g1-structural-model/v1\0")
    for kind, count_name in (
        ("body", "nbody"),
        ("joint", "njnt"),
        ("geom", "ngeom"),
    ):
        count = int(getattr(model, count_name))
        _update_hash_text(digest, kind)
        for name in _names(model, kind, count):
            _update_hash_text(digest, name)
    arrays = (
        ("body_parentid", np.int64),
        ("jnt_type", np.int64),
        ("jnt_bodyid", np.int64),
        ("jnt_qposadr", np.int64),
        ("jnt_limited", np.int64),
        ("jnt_range", np.float64),
        ("geom_bodyid", np.int64),
        ("geom_type", np.int64),
        ("geom_dataid", np.int64),
        ("geom_contype", np.int64),
        ("geom_conaffinity", np.int64),
        ("geom_pos", np.float64),
        ("geom_quat", np.float64),
        ("geom_size", np.float64),
    )
    for name, dtype in arrays:
        _update_hash_text(digest, name)
        _update_hash_array(digest, getattr(model, name), np.dtype(dtype))
    for name, dtype in (
        ("mesh_vertadr", np.int64),
        ("mesh_vertnum", np.int64),
        ("mesh_faceadr", np.int64),
        ("mesh_facenum", np.int64),
        ("mesh_vert", np.float64),
        ("mesh_face", np.int64),
        ("mesh_scale", np.float64),
        ("mesh_pos", np.float64),
        ("mesh_quat", np.float64),
    ):
        if hasattr(model, name):
            _update_hash_text(digest, name)
            _update_hash_array(digest, getattr(model, name), np.dtype(dtype))
    return digest.hexdigest()


def _terrain_sha256(query: CanonicalMeshQuery) -> str:
    digest = hashlib.sha256(b"terrain-oracle-exact-query-surface/v1\0")
    _update_hash_text(digest, query.mesh_sha256)
    _update_hash_text(digest, query.source_asset_sha256)
    _update_hash_array(digest, query._triangles, np.dtype(np.float64))
    _update_hash_array(digest, query._normals, np.dtype(np.float64))
    _update_hash_array(digest, query._face_indices, np.dtype(np.int64))
    _update_hash_array(
        digest,
        query.world_from_terrain.translation_world,
        np.dtype(np.float64),
    )
    _update_hash_array(
        digest,
        query.world_from_terrain.quaternion_world_from_local_wxyz,
        np.dtype(np.float64),
    )
    return digest.hexdigest()


def _expected_joint_body(name: str) -> str:
    if name == "waist_pitch_joint":
        return "torso_link"
    return name.removesuffix("_joint") + "_link"


@dataclass(frozen=True)
class _ModelRegistration:
    joint_ids: tuple[int, ...]
    body_ids: tuple[int, ...]
    foot_geom_ids: tuple[tuple[int, ...], tuple[int, ...]]
    body_geom_ids: tuple[int, ...]


def _descended(model: object, body_id: int, ancestor_id: int) -> bool:
    cursor = int(body_id)
    while cursor > 0:
        if cursor == ancestor_id:
            return True
        cursor = int(model.body_parentid[cursor])
    return False


def _validate_model(model: object) -> _ModelRegistration:
    body_names = _names(model, "body", int(model.nbody))
    joint_names = _names(model, "joint", int(model.njnt))
    if (
        len(body_names) != 31
        or body_names.count("world") != 1
        or set(body_names) != {"world", *ISAACLAB_BODY_NAMES}
    ):
        raise ContractError("model robot body names do not equal canonical G1")
    if (
        len(joint_names) != 30
        or set(joint_names)
        != {"floating_base_joint", *ISAACLAB_JOINT_NAMES}
    ):
        raise ContractError("model joint names do not equal canonical G1")
    body_lookup = {name: index for index, name in enumerate(body_names)}
    joint_lookup = {name: index for index, name in enumerate(joint_names)}
    root = joint_lookup["floating_base_joint"]
    if (
        int(model.jnt_type[root]) != 0
        or body_names[int(model.jnt_bodyid[root])] != "pelvis"
    ):
        raise ContractError("model free root is not floating_base_joint on pelvis")
    for name in ISAACLAB_JOINT_NAMES:
        index = joint_lookup[name]
        ranges = np.asarray(model.jnt_range[index], dtype=np.float64)
        if (
            int(model.jnt_type[index]) != 3
            or int(model.jnt_limited[index]) == 0
            or ranges.shape != (2,)
            or not np.isfinite(ranges).all()
            or ranges[0] >= ranges[1]
            or body_names[int(model.jnt_bodyid[index])]
            != _expected_joint_body(name)
        ):
            raise ContractError(f"invalid exact G1 joint association for {name}")
    geom_count = int(model.ngeom)
    foot_bodies = tuple(body_lookup[name] for name in _FOOT_NAMES)
    foot_geoms: list[list[int]] = [[], []]
    body_geoms: list[int] = []
    for geom in range(geom_count):
        if (
            int(model.geom_contype[geom]) == 0
            and int(model.geom_conaffinity[geom]) == 0
        ):
            continue
        body = int(model.geom_bodyid[geom])
        if body <= 0 or body >= len(body_names):
            continue
        geom_type = int(model.geom_type[geom])
        if geom_type not in (2, 3, 5, 6, 7):
            raise ContractError(
                f"unsupported collision-enabled G1 geom type {geom_type}"
            )
        if (
            not np.isfinite(np.asarray(model.geom_pos[geom])).all()
            or not np.isfinite(np.asarray(model.geom_quat[geom])).all()
            or not np.isfinite(np.asarray(model.geom_size[geom])).all()
        ):
            raise ContractError("model collision geom transform is nonfinite")
        matched = False
        for foot, ancestor in enumerate(foot_bodies):
            if _descended(model, body, ancestor):
                foot_geoms[foot].append(geom)
                matched = True
                break
        if not matched:
            body_geoms.append(geom)
    if any(not values for values in foot_geoms):
        raise ContractError("model named foot collision geometry resolved empty")
    return _ModelRegistration(
        joint_ids=tuple(joint_lookup[name] for name in ISAACLAB_JOINT_NAMES),
        body_ids=tuple(body_lookup[name] for name in ISAACLAB_BODY_NAMES),
        foot_geom_ids=(tuple(foot_geoms[0]), tuple(foot_geoms[1])),
        body_geom_ids=tuple(body_geoms),
    )


def _same_transform(left: object, right: object) -> bool:
    return np.array_equal(
        left.translation_world, right.translation_world
    ) and np.array_equal(
        left.quaternion_world_from_local_wxyz,
        right.quaternion_world_from_local_wxyz,
    )


def _validate_terrain(clip: CanonicalClip, query: object) -> CanonicalMeshQuery:
    if clip.terrain is None:
        raise ContractError("clip has no explicit terrain binding")
    if not isinstance(query, CanonicalMeshQuery):
        raise ContractError("audit terrain must be a CanonicalMeshQuery")
    if (
        clip.terrain.mesh_sha256 != query.mesh_sha256
        or clip.terrain.asset_sha256 != query.source_asset_sha256
        or not _same_transform(
            clip.terrain.world_from_terrain, query.world_from_terrain
        )
    ):
        raise ContractError("terrain query and clip binding provenance mismatch")
    return query


def _spans(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    values = np.asarray(mask, dtype=np.bool_)
    edges = np.diff(
        np.concatenate(
            (np.array((False,)), values, np.array((False,)))
        ).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _guard(mask: np.ndarray, frames: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=np.bool_)
    for start, end in _spans(mask):
        result[max(0, start - frames) : min(len(mask), end + frames)] = True
    return result


def _rotate_wxyz(quaternion: object, vector: object) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    cross = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * cross + np.cross(q[..., 1:], cross)


def _quaternion_multiply(left: object, right: object) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _transform_points(
    body_position: np.ndarray,
    body_quaternion: np.ndarray,
    geom_position: np.ndarray,
    geom_quaternion: np.ndarray,
    local_points: np.ndarray,
) -> np.ndarray:
    geom_points = _rotate_wxyz(geom_quaternion, local_points) + geom_position
    return _rotate_wxyz(body_quaternion, geom_points) + body_position


def _sole_probes(
    clip: CanonicalClip,
    model: object,
    registration: _ModelRegistration,
) -> np.ndarray:
    probes = np.empty((clip.frame_count, 2, 6, 3), dtype=np.float64)
    sphere_type = 2
    for foot, geom_ids in enumerate(registration.foot_geom_ids):
        spheres = [
            geom
            for geom in geom_ids
            if int(model.geom_type[geom]) == sphere_type
        ]
        if len(spheres) != 4:
            raise ContractError(
                f"{_FOOT_NAMES[foot]} must have exactly four support spheres"
            )
        corners = np.empty((clip.frame_count, 4, 3), dtype=np.float64)
        reference_x = []
        for corner, geom in enumerate(spheres):
            body_name = str(model.body(int(model.geom_bodyid[geom])).name)
            body_index = clip.body_names.index(body_name)
            center = _transform_points(
                np.asarray(clip.body_position_world[:, body_index]),
                np.asarray(clip.body_quaternion_world_wxyz[:, body_index]),
                np.asarray(model.geom_pos[geom]),
                np.asarray(model.geom_quat[geom]),
                np.zeros(3),
            )
            corners[:, corner] = center
            corners[:, corner, 2] -= float(model.geom_size[geom, 0])
            reference_x.append(float(model.geom_pos[geom, 0]))
        order = np.argsort(np.asarray(reference_x), kind="stable")
        ordered = corners[:, order]
        probes[:, foot, :4] = ordered
        probes[:, foot, 4] = np.mean(ordered[:, :2], axis=1)
        probes[:, foot, 5] = np.mean(ordered[:, 2:], axis=1)
    return probes


def _support_audit(
    clip: CanonicalClip,
    model: object,
    registration: _ModelRegistration,
    query: CanonicalMeshQuery,
    thresholds: AuditThresholds,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    probes = _sole_probes(clip, model, registration)
    surface = query.query(probes.reshape((-1, 3)))
    shape = (clip.frame_count, 2, 6)
    closest = np.asarray(surface.closest_point_world).reshape((*shape, 3))
    normal = np.asarray(surface.surface_normal_world).reshape((*shape, 3))
    distance = np.asarray(surface.distance_m).reshape(shape)
    ray_distance = np.asarray(surface.downward_ray_distance_m).reshape(shape)
    ray_normal = np.asarray(surface.downward_ray_normal_world).reshape(
        (*shape, 3)
    )
    ray_support = (
        np.isfinite(ray_distance)
        & (ray_distance <= thresholds.support_distance_m)
        & (ray_normal[..., 2] >= thresholds.minimum_surface_normal_z)
    )
    closest_support = (
        (distance <= thresholds.support_distance_m)
        & (normal[..., 2] >= thresholds.minimum_surface_normal_z)
    )
    supported = ray_support | closest_support
    complete = np.all(supported, axis=2)
    any_support = np.any(supported, axis=2)
    labels = np.asarray(clip.contact) > thresholds.contact_label_threshold
    disagreement = labels != complete
    incomplete = labels & any_support & ~complete
    signed = np.sum((probes - closest) * normal, axis=-1)
    penetration = np.maximum(0.0, -signed)
    foot_penetration = np.max(penetration, axis=(1, 2))

    sole_center = np.mean(probes[:, :, :4], axis=2)
    skate = np.zeros(clip.frame_count, dtype=np.float64)
    for foot in range(2):
        anchor: np.ndarray | None = None
        for frame in range(clip.frame_count):
            if complete[frame, foot]:
                if anchor is None:
                    anchor = sole_center[frame, foot].copy()
                skate[frame] = max(
                    skate[frame],
                    float(np.linalg.norm(sole_center[frame, foot] - anchor)),
                )
            else:
                anchor = None
    masks = {
        "contact_inconsistency": np.any(disagreement, axis=1),
        "incomplete_sole_support": np.any(incomplete, axis=1),
        "stance_skate": skate > thresholds.stance_skate_limit_m,
        "foot_penetration": (
            foot_penetration > thresholds.foot_penetration_tolerance_m
        ),
    }
    values = {
        "contact_inconsistency": np.max(
            disagreement.astype(np.float64), axis=1
        ),
        "incomplete_sole_support": 1.0 - np.mean(
            supported.astype(np.float64), axis=(1, 2)
        ),
        "stance_skate": skate,
        "foot_penetration": foot_penetration,
    }
    return masks, values, probes


def _closest_point_triangle(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Exact closest point on one nondegenerate triangle."""

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    return a + ab * v + ac * w


def _segment_triangle_hit(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    epsilon: float,
) -> tuple[bool, np.ndarray]:
    """Möller-Trumbore segment/triangle intersection."""

    direction = end - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    h = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, h))
    if abs(determinant) <= epsilon:
        return False, np.zeros(3, dtype=np.float64)
    inverse = 1.0 / determinant
    relative = start - triangle[0]
    u = inverse * float(np.dot(relative, h))
    if u < -epsilon or u > 1.0 + epsilon:
        return False, np.zeros(3, dtype=np.float64)
    q = np.cross(relative, edge1)
    v = inverse * float(np.dot(direction, q))
    if v < -epsilon or u + v > 1.0 + epsilon:
        return False, np.zeros(3, dtype=np.float64)
    fraction = inverse * float(np.dot(edge2, q))
    if fraction < -epsilon or fraction > 1.0 + epsilon:
        return False, np.zeros(3, dtype=np.float64)
    return True, start + np.clip(fraction, 0.0, 1.0) * direction


def _closest_segment_segment(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact closest points between two finite segments."""

    first = first_end - first_start
    second = second_end - second_start
    relative = first_start - second_start
    a = float(np.dot(first, first))
    e = float(np.dot(second, second))
    f = float(np.dot(second, relative))
    if a <= epsilon and e <= epsilon:
        return first_start, second_start
    if a <= epsilon:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(first, relative))
        if e <= epsilon:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = float(np.dot(first, second))
            denominator = a * e - b * b
            s = (
                np.clip((b * f - c * e) / denominator, 0.0, 1.0)
                if abs(denominator) > epsilon
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    return first_start + s * first, second_start + t * second


def _closest_segment_triangle(
    start: np.ndarray,
    end: np.ndarray,
    triangle: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    hit, point = _segment_triangle_hit(start, end, triangle, epsilon)
    if hit:
        return point, point
    candidates: list[tuple[np.ndarray, np.ndarray]] = [
        (start, _closest_point_triangle(start, triangle)),
        (end, _closest_point_triangle(end, triangle)),
    ]
    for edge in ((0, 1), (1, 2), (2, 0)):
        candidates.append(
            _closest_segment_segment(
                start,
                end,
                triangle[edge[0]],
                triangle[edge[1]],
                epsilon,
            )
        )
    distances = [
        float(np.dot(left - right, left - right))
        for left, right in candidates
    ]
    return candidates[int(np.argmin(distances))]


def _triangles_intersect(
    left: np.ndarray, right: np.ndarray, epsilon: float
) -> bool:
    """Exact triangle intersection, including coplanar edge/vertex touches."""

    for triangle, target in ((left, right), (right, left)):
        for edge in ((0, 1), (1, 2), (2, 0)):
            hit, _point = _segment_triangle_hit(
                triangle[edge[0]], triangle[edge[1]], target, epsilon
            )
            if hit:
                return True
    left_normal = np.cross(left[1] - left[0], left[2] - left[0])
    right_normal = np.cross(right[1] - right[0], right[2] - right[0])
    left_norm = float(np.linalg.norm(left_normal))
    right_norm = float(np.linalg.norm(right_normal))
    if left_norm <= epsilon or right_norm <= epsilon:
        return False
    left_normal /= left_norm
    right_normal /= right_norm
    if (
        float(np.linalg.norm(np.cross(left_normal, right_normal))) > epsilon
        or abs(float(np.dot(right[0] - left[0], left_normal))) > epsilon
    ):
        return False
    drop = int(np.argmax(np.abs(left_normal)))
    left_2d = np.delete(left, drop, axis=1)
    right_2d = np.delete(right, drop, axis=1)

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )

    def point_in(point: np.ndarray, triangle: np.ndarray) -> bool:
        signs = np.asarray(
            [
                orient(triangle[0], triangle[1], point),
                orient(triangle[1], triangle[2], point),
                orient(triangle[2], triangle[0], point),
            ]
        )
        return bool(
            np.all(signs >= -epsilon) or np.all(signs <= epsilon)
        )

    def segments_intersect(
        a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
    ) -> bool:
        if np.any(
            np.maximum(np.minimum(a, b), np.minimum(c, d))
            > np.minimum(np.maximum(a, b), np.maximum(c, d)) + epsilon
        ):
            return False
        first = orient(a, b, c)
        second = orient(a, b, d)
        third = orient(c, d, a)
        fourth = orient(c, d, b)
        return bool(
            min(first, second) <= epsilon
            and max(first, second) >= -epsilon
            and min(third, fourth) <= epsilon
            and max(third, fourth) >= -epsilon
        )

    if point_in(left_2d[0], right_2d) or point_in(right_2d[0], left_2d):
        return True
    for first in ((0, 1), (1, 2), (2, 0)):
        for second in ((0, 1), (1, 2), (2, 0)):
            if segments_intersect(
                left_2d[first[0]],
                left_2d[first[1]],
                right_2d[second[0]],
                right_2d[second[1]],
            ):
                return True
    return False


def _box_triangles(size: np.ndarray) -> np.ndarray:
    vertices = np.asarray(
        [
            (x * size[0], y * size[1], z * size[2])
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
            (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        ),
        dtype=np.int64,
    )
    return vertices[faces]


def _cylinder_triangles(
    radius: float, half_length: float, segments: int
) -> np.ndarray:
    angles = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
    ring = np.stack(
        (radius * np.cos(angles), radius * np.sin(angles)), axis=1
    )
    bottom = np.column_stack(
        (ring, np.full(segments, -half_length, dtype=np.float64))
    )
    top = np.column_stack(
        (ring, np.full(segments, half_length, dtype=np.float64))
    )
    vertices = np.concatenate(
        (bottom, top, np.array(((0, 0, -half_length), (0, 0, half_length)))),
        axis=0,
    )
    triangles = []
    for index in range(segments):
        following = (index + 1) % segments
        triangles.extend(
            (
                (index, following, segments + following),
                (index, segments + following, segments + index),
                (2 * segments, following, index),
                (2 * segments + 1, segments + index, segments + following),
            )
        )
    return vertices[np.asarray(triangles, dtype=np.int64)]


def _mesh_triangles(model: object, geom: int) -> np.ndarray:
    mesh = int(model.geom_dataid[geom])
    if mesh < 0 or mesh >= int(model.nmesh):
        raise ContractError("collision mesh geom has an invalid mesh id")
    vertex_start = int(model.mesh_vertadr[mesh])
    vertex_count = int(model.mesh_vertnum[mesh])
    face_start = int(model.mesh_faceadr[mesh])
    face_count = int(model.mesh_facenum[mesh])
    vertices = np.asarray(
        model.mesh_vert[vertex_start : vertex_start + vertex_count],
        dtype=np.float64,
    )
    faces = np.asarray(
        model.mesh_face[face_start : face_start + face_count],
        dtype=np.int64,
    )
    if (
        vertices.shape != (vertex_count, 3)
        or faces.shape != (face_count, 3)
        or not np.isfinite(vertices).all()
        or np.any(faces < 0)
        or np.any(faces >= vertex_count)
    ):
        raise ContractError("collision mesh arrays are invalid")
    return vertices[faces]


def _triangle_candidate_mask(
    minimum: np.ndarray,
    maximum: np.ndarray,
    terrain_minimum: np.ndarray,
    terrain_maximum: np.ndarray,
    terrain_normals: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    overlap_xy = np.all(
        (maximum[None, :2] + epsilon >= terrain_minimum[:, :2])
        & (minimum[None, :2] - epsilon <= terrain_maximum[:, :2]),
        axis=1,
    )
    upward = terrain_normals[:, 2] >= 0.5
    vertical_overlap = (
        (maximum[2] + epsilon >= terrain_minimum[:, 2])
        & (minimum[2] - epsilon <= terrain_maximum[:, 2])
    )
    below_upward_surface = minimum[2] <= terrain_maximum[:, 2] + epsilon
    return overlap_xy & np.where(upward, below_upward_surface, vertical_overlap)


def _triangle_surface_penetration(
    robot_triangles: np.ndarray,
    terrain_triangles: np.ndarray,
    terrain_normals: np.ndarray,
    terrain_minimum: np.ndarray,
    terrain_maximum: np.ndarray,
    epsilon: float,
) -> float:
    maximum_penetration = 0.0
    robot_minimum = np.min(robot_triangles, axis=1)
    robot_maximum = np.max(robot_triangles, axis=1)
    for index, robot in enumerate(robot_triangles):
        candidates = np.flatnonzero(
            _triangle_candidate_mask(
                robot_minimum[index],
                robot_maximum[index],
                terrain_minimum,
                terrain_maximum,
                terrain_normals,
                epsilon,
            )
        )
        for terrain_index in candidates:
            terrain = terrain_triangles[terrain_index]
            normal = terrain_normals[terrain_index]
            closest = np.asarray(
                [_closest_point_triangle(vertex, terrain) for vertex in robot]
            )
            signed = np.sum((robot - closest) * normal, axis=1)
            intersects = _triangles_intersect(robot, terrain, epsilon)
            if intersects or float(np.min(signed)) < 0.0:
                maximum_penetration = max(
                    maximum_penetration,
                    max(
                        epsilon if intersects else 0.0,
                        float(max(0.0, -np.min(signed))),
                    ),
                )
    return maximum_penetration


def _sphere_surface_penetration(
    center: np.ndarray,
    radius: float,
    terrain_triangles: np.ndarray,
    terrain_normals: np.ndarray,
    terrain_minimum: np.ndarray,
    terrain_maximum: np.ndarray,
    epsilon: float,
) -> float:
    minimum = center - radius
    maximum = center + radius
    candidates = np.flatnonzero(
        _triangle_candidate_mask(
            minimum,
            maximum,
            terrain_minimum,
            terrain_maximum,
            terrain_normals,
            epsilon,
        )
    )
    penetration = 0.0
    for index in candidates:
        closest = _closest_point_triangle(center, terrain_triangles[index])
        delta = center - closest
        distance = float(np.linalg.norm(delta))
        signed = float(np.dot(delta, terrain_normals[index]))
        candidate = (
            radius + distance if signed < 0.0 else radius - distance
        )
        penetration = max(penetration, candidate)
    return max(0.0, penetration)


def _capsule_surface_penetration(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    terrain_triangles: np.ndarray,
    terrain_normals: np.ndarray,
    terrain_minimum: np.ndarray,
    terrain_maximum: np.ndarray,
    epsilon: float,
) -> float:
    minimum = np.minimum(start, end) - radius
    maximum = np.maximum(start, end) + radius
    candidates = np.flatnonzero(
        _triangle_candidate_mask(
            minimum,
            maximum,
            terrain_minimum,
            terrain_maximum,
            terrain_normals,
            epsilon,
        )
    )
    penetration = 0.0
    for index in candidates:
        on_segment, on_triangle = _closest_segment_triangle(
            start, end, terrain_triangles[index], epsilon
        )
        delta = on_segment - on_triangle
        distance = float(np.linalg.norm(delta))
        signed = float(np.dot(delta, terrain_normals[index]))
        candidate = (
            radius + distance if signed < 0.0 else radius - distance
        )
        penetration = max(penetration, candidate)
    return max(0.0, penetration)


def _body_penetration(
    clip: CanonicalClip,
    model: object,
    registration: _ModelRegistration,
    query: CanonicalMeshQuery,
    thresholds: AuditThresholds,
) -> np.ndarray:
    """Exact per-geom surface collision with deterministic AABB broadphase."""

    result = np.zeros(clip.frame_count, dtype=np.float64)
    terrain_triangles = np.asarray(query._triangles, dtype=np.float64)
    terrain_normals = np.asarray(query._normals, dtype=np.float64)
    terrain_minimum = np.min(terrain_triangles, axis=1)
    terrain_maximum = np.max(terrain_triangles, axis=1)
    for geom in registration.body_geom_ids:
        geom_type = int(model.geom_type[geom])
        body_name = str(model.body(int(model.geom_bodyid[geom])).name)
        body_index = clip.body_names.index(body_name)
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        local_triangles: np.ndarray | None = None
        if geom_type == 7:
            local_triangles = _mesh_triangles(model, geom)
        elif geom_type == 6:
            local_triangles = _box_triangles(size)
        elif geom_type == 5:
            local_triangles = _cylinder_triangles(
                float(size[0]),
                float(size[1]),
                thresholds.cylinder_radial_segments,
            )
        if local_triangles is not None:
            local_minimum = np.min(local_triangles, axis=(0, 1))
            local_maximum = np.max(local_triangles, axis=(0, 1))
            local_bounds = np.asarray(
                [
                    (x, y, z)
                    for x in (local_minimum[0], local_maximum[0])
                    for y in (local_minimum[1], local_maximum[1])
                    for z in (local_minimum[2], local_maximum[2])
                ],
                dtype=np.float64,
            )
        for frame in range(clip.frame_count):
            body_position = np.asarray(
                clip.body_position_world[frame, body_index]
            )
            body_quaternion = np.asarray(
                clip.body_quaternion_world_wxyz[frame, body_index]
            )
            geom_position = np.asarray(model.geom_pos[geom])
            geom_quaternion = np.asarray(model.geom_quat[geom])
            if local_triangles is not None:
                world_bounds = _transform_points(
                    body_position,
                    body_quaternion,
                    geom_position,
                    geom_quaternion,
                    local_bounds,
                )
                if not np.any(
                    _triangle_candidate_mask(
                        np.min(world_bounds, axis=0),
                        np.max(world_bounds, axis=0),
                        terrain_minimum,
                        terrain_maximum,
                        terrain_normals,
                        thresholds.collision_epsilon_m,
                    )
                ):
                    continue
                world = _transform_points(
                    body_position,
                    body_quaternion,
                    geom_position,
                    geom_quaternion,
                    local_triangles.reshape((-1, 3)),
                ).reshape(local_triangles.shape)
                penetration = _triangle_surface_penetration(
                    world,
                    terrain_triangles,
                    terrain_normals,
                    terrain_minimum,
                    terrain_maximum,
                    thresholds.collision_epsilon_m,
                )
            elif geom_type == 2:
                center = _transform_points(
                    body_position,
                    body_quaternion,
                    geom_position,
                    geom_quaternion,
                    np.zeros(3),
                )
                penetration = _sphere_surface_penetration(
                    center,
                    float(size[0]),
                    terrain_triangles,
                    terrain_normals,
                    terrain_minimum,
                    terrain_maximum,
                    thresholds.collision_epsilon_m,
                )
            elif geom_type == 3:
                endpoints = _transform_points(
                    body_position,
                    body_quaternion,
                    geom_position,
                    geom_quaternion,
                    np.asarray(
                        ((0.0, 0.0, -size[1]), (0.0, 0.0, size[1]))
                    ),
                )
                penetration = _capsule_surface_penetration(
                    endpoints[0],
                    endpoints[1],
                    float(size[0]),
                    terrain_triangles,
                    terrain_normals,
                    terrain_minimum,
                    terrain_maximum,
                    thresholds.collision_epsilon_m,
                )
            else:
                raise ContractError(
                    f"unsupported collision geom type {geom_type}"
                )
            result[frame] = max(result[frame], penetration)
    return result


def _quaternion_steps(clip: CanonicalClip) -> np.ndarray:
    traces = (
        np.asarray(clip.root_quaternion_world_wxyz).reshape(
            (clip.frame_count, -1, 4)
        ),
        np.asarray(clip.body_quaternion_world_wxyz).reshape(
            (clip.frame_count, -1, 4)
        ),
        np.asarray(clip.sole_quaternion_world_wxyz).reshape(
            (clip.frame_count, -1, 4)
        ),
    )
    result = np.zeros(clip.frame_count, dtype=np.float64)
    for trace in traces:
        dot = np.abs(np.sum(trace[1:] * trace[:-1], axis=-1))
        angle = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
        result[1:] = np.maximum(result[1:], np.max(angle, axis=1))
    return result


def _segment_derivatives(
    value: np.ndarray, fps: float, valid: np.ndarray, *, angular: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    output_shape = value.shape[:-1] + (3,) if angular else value.shape
    output = np.zeros(output_shape, dtype=np.float64)
    defined = np.zeros(len(value), dtype=np.bool_)
    for start, end in _spans(valid):
        if end - start < 2:
            continue
        function = angular_velocity_world_wxyz if angular else finite_difference
        output[start:end] = function(value[start:end], fps)
        defined[start:end] = True
    return output, defined


def _reasons_from_mask(
    code: str,
    mask: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> list[AuditReason]:
    return [
        AuditReason(
            code=code,
            severity="error",
            frame_interval=(start, end),
            observed_maximum=float(np.max(values[start:end])),
            threshold=float(threshold),
        )
        for start, end in _spans(mask)
    ]


def _empty_metrics() -> dict[str, float]:
    return {name: 0.0 for name in _METRIC_NAMES}


def _fatal_report(
    clip: CanonicalClip,
    thresholds: AuditThresholds,
    code: str,
    *,
    model_sha256: str,
    terrain_sha256: str,
) -> ClipAudit:
    frame_count = max(2, int(getattr(clip, "frame_count", 2)))
    source = getattr(getattr(clip, "source", None), "source_sha256", "0" * 64)
    if not (
        isinstance(source, str)
        and len(source) == 64
        and all(character in _SHA256 for character in source)
    ):
        source = "0" * 64
    return ClipAudit(
        schema=_SCHEMA,
        clip_id=str(getattr(clip, "clip_id", "invalid-source")) or "invalid-source",
        frame_count=frame_count,
        source_sha256=source,
        model_sha256=model_sha256,
        terrain_sha256=terrain_sha256,
        status="rejected",
        accepted_intervals=(),
        thresholds=thresholds.to_dict(),
        metrics=_empty_metrics(),
        reasons=(
            AuditReason(code, "fatal", (0, frame_count), 1.0, 0.0),
        ),
    )


def audit_clip(
    clip: CanonicalClip,
    model: object,
    terrain: CanonicalMeshQuery | None,
    *,
    thresholds: AuditThresholds = AuditThresholds(),
) -> ClipAudit:
    """Audit one disk-loaded canonical clip against exact mechanics and terrain."""

    if not isinstance(thresholds, AuditThresholds):
        raise ContractError("thresholds must be AuditThresholds")
    if not isinstance(clip, CanonicalClip):
        raise ContractError("audit_clip requires a CanonicalClip")
    try:
        model_hash = structural_model_sha256(model)
    except Exception:
        model_hash = "0" * 64
    try:
        terrain_hash = (
            _terrain_sha256(terrain)
            if isinstance(terrain, CanonicalMeshQuery)
            else "0" * 64
        )
    except Exception:
        terrain_hash = "0" * 64
    try:
        clip.validate()
    except ContractError:
        return _fatal_report(
            clip,
            thresholds,
            "source_registration",
            model_sha256=model_hash,
            terrain_sha256=terrain_hash,
        )
    try:
        registration = _validate_model(model)
    except Exception:
        return _fatal_report(
            clip,
            thresholds,
            "model_registration",
            model_sha256=model_hash,
            terrain_sha256=terrain_hash,
        )
    try:
        query = _validate_terrain(clip, terrain)
    except Exception:
        return _fatal_report(
            clip,
            thresholds,
            "terrain_registration",
            model_sha256=model_hash,
            terrain_sha256=terrain_hash,
        )

    frames_count = clip.frame_count
    fps = float(clip.fps)
    masks: dict[str, np.ndarray] = {}
    values: dict[str, np.ndarray] = {}
    threshold_by_code = {
        "joint_limit": thresholds.joint_limit_tolerance_rad,
        "joint_velocity": thresholds.joint_velocity_limit_rad_s,
        "quaternion_discontinuity": thresholds.quaternion_step_limit_rad,
        "root_plausibility": 1.0,
        "derivative_discontinuity": 1.0,
        "contact_inconsistency": 0.0,
        "incomplete_sole_support": 0.0,
        "stance_skate": thresholds.stance_skate_limit_m,
        "foot_penetration": thresholds.foot_penetration_tolerance_m,
        "body_penetration": thresholds.body_penetration_tolerance_m,
    }

    ranges = np.asarray(model.jnt_range)[
        np.asarray(registration.joint_ids, dtype=np.int64)
    ]
    joints = np.asarray(clip.joint_position, dtype=np.float64)
    below = np.maximum(0.0, ranges[None, :, 0] - joints)
    above = np.maximum(0.0, joints - ranges[None, :, 1])
    joint_violation = np.max(np.maximum(below, above), axis=1)
    masks["joint_limit"] = (
        joint_violation > thresholds.joint_limit_tolerance_rad
    )
    values["joint_limit"] = joint_violation
    derived_joint = np.asarray(finite_difference(joints, fps))
    joint_speed = np.max(np.abs(derived_joint), axis=1)
    masks["joint_velocity"] = (
        joint_speed > thresholds.joint_velocity_limit_rad_s
    )
    values["joint_velocity"] = joint_speed

    quaternion_step = _quaternion_steps(clip)
    masks["quaternion_discontinuity"] = (
        quaternion_step > thresholds.quaternion_step_limit_rad
    )
    values["quaternion_discontinuity"] = quaternion_step

    root_velocity = np.asarray(
        finite_difference(clip.root_position_world, fps), dtype=np.float64
    )
    root_speed = np.linalg.norm(root_velocity, axis=1)
    root_acceleration = np.linalg.norm(
        finite_difference(root_velocity, fps), axis=1
    )
    root_angular = np.asarray(
        angular_velocity_world_wxyz(
            clip.root_quaternion_world_wxyz, fps
        )
    )
    root_angular_speed = np.linalg.norm(root_angular, axis=1)
    root_surface = query.query(np.asarray(clip.root_position_world))
    root_signed_height = np.sum(
        (
            np.asarray(clip.root_position_world)
            - np.asarray(root_surface.closest_point_world)
        )
        * np.asarray(root_surface.surface_normal_world),
        axis=1,
    )
    root_excess = np.maximum.reduce(
        (
            root_speed / thresholds.root_speed_limit_m_s,
            root_acceleration / thresholds.root_acceleration_limit_m_s2,
            root_angular_speed
            / thresholds.root_angular_speed_limit_rad_s,
            np.maximum(
                0.0,
                thresholds.root_height_min_m - root_signed_height,
            )
            / thresholds.root_height_min_m,
            np.maximum(
                0.0,
                root_signed_height - thresholds.root_height_max_m,
            )
            / thresholds.root_height_max_m,
        )
    )
    masks["root_plausibility"] = root_excess > 1.0
    values["root_plausibility"] = root_excess

    support_masks, support_values, _probes = _support_audit(
        clip, model, registration, query, thresholds
    )
    masks.update(support_masks)
    values.update(support_values)
    body_penetration = _body_penetration(
        clip, model, registration, query, thresholds
    )
    masks["body_penetration"] = (
        body_penetration > thresholds.body_penetration_tolerance_m
    )
    values["body_penetration"] = body_penetration

    primary_invalid = np.zeros(frames_count, dtype=np.bool_)
    for mask in masks.values():
        primary_invalid |= mask
    candidate = ~_guard(primary_invalid, thresholds.guard_frames)
    derivative_error = np.zeros(frames_count, dtype=np.float64)
    derivative_defined = np.zeros(frames_count, dtype=np.bool_)
    derivative_specs = (
        (
            np.asarray(clip.root_position_world),
            np.asarray(clip.root_linear_velocity_world),
            thresholds.derivative_root_linear_tolerance_m_s,
            False,
        ),
        (
            np.asarray(clip.root_quaternion_world_wxyz),
            np.asarray(clip.root_angular_velocity_world),
            thresholds.derivative_root_angular_tolerance_rad_s,
            True,
        ),
        (
            np.asarray(clip.joint_position),
            np.asarray(clip.joint_velocity),
            thresholds.derivative_joint_tolerance_rad_s,
            False,
        ),
        (
            np.asarray(clip.body_position_world),
            np.asarray(clip.body_linear_velocity_world),
            thresholds.derivative_body_linear_tolerance_m_s,
            False,
        ),
        (
            np.asarray(clip.body_quaternion_world_wxyz),
            np.asarray(clip.body_angular_velocity_world),
            thresholds.derivative_body_angular_tolerance_rad_s,
            True,
        ),
    )
    for position, stored, tolerance, angular in derivative_specs:
        derived, defined = _segment_derivatives(
            position, fps, candidate, angular=angular
        )
        error = np.max(np.abs(derived - stored), axis=tuple(range(1, derived.ndim)))
        derivative_error = np.maximum(
            derivative_error, np.where(defined, error / tolerance, 0.0)
        )
        derivative_defined |= defined
    segmented_joint_velocity, segmented_joint_defined = _segment_derivatives(
        joints, fps, candidate
    )
    segmented_joint_acceleration, segmented_acceleration_defined = (
        _segment_derivatives(segmented_joint_velocity, fps, candidate)
    )
    joint_acceleration = np.max(
        np.abs(segmented_joint_acceleration), axis=1
    )
    derivative_error = np.maximum(
        derivative_error,
        np.where(
            segmented_joint_defined & segmented_acceleration_defined,
            joint_acceleration
            / thresholds.derivative_acceleration_limit_rad_s2,
            0.0,
        ),
    )
    masks["derivative_discontinuity"] = (
        derivative_defined & (derivative_error > 1.0)
    )
    values["derivative_discontinuity"] = derivative_error

    reasons: list[AuditReason] = []
    for code in (
        "joint_limit",
        "joint_velocity",
        "quaternion_discontinuity",
        "root_plausibility",
        "derivative_discontinuity",
        "contact_inconsistency",
        "incomplete_sole_support",
        "stance_skate",
        "foot_penetration",
        "body_penetration",
    ):
        reasons.extend(
            _reasons_from_mask(
                code, masks[code], values[code], threshold_by_code[code]
            )
        )
    reasons.sort(
        key=lambda reason: (
            reason.frame_interval[0],
            reason.frame_interval[1],
            reason.code,
        )
    )
    invalid = np.zeros(frames_count, dtype=np.bool_)
    for mask in masks.values():
        invalid |= mask
    guarded = _guard(invalid, thresholds.guard_frames)
    accepted = tuple(
        (start, end)
        for start, end in _spans(~guarded)
        if end - start >= thresholds.minimum_interval_frames
    )
    if not accepted:
        status = "rejected"
    elif accepted == ((0, frames_count),):
        status = "accepted"
    else:
        status = "accepted_with_intervals_removed"

    metrics = {
        "max_joint_limit_violation_rad": float(np.max(joint_violation)),
        "max_joint_speed_rad_s": float(np.max(joint_speed)),
        "max_quaternion_step_rad": float(np.max(quaternion_step)),
        "max_root_speed_m_s": float(np.max(root_speed)),
        "min_root_height_m": float(np.min(root_signed_height)),
        "max_root_height_m": float(np.max(root_signed_height)),
        "max_root_acceleration_m_s2": float(np.max(root_acceleration)),
        "max_root_angular_speed_rad_s": float(np.max(root_angular_speed)),
        "max_derivative_error": float(np.max(derivative_error)),
        "max_contact_disagreement": float(
            np.max(values["contact_inconsistency"])
        ),
        "max_incomplete_sole_fraction": float(
            np.max(values["incomplete_sole_support"])
        ),
        "max_stance_skate_m": float(np.max(values["stance_skate"])),
        "max_foot_penetration_m": float(
            np.max(values["foot_penetration"])
        ),
        "max_body_penetration_m": float(np.max(body_penetration)),
    }
    return ClipAudit(
        schema=_SCHEMA,
        clip_id=clip.clip_id,
        frame_count=frames_count,
        source_sha256=clip.source.source_sha256,
        model_sha256=model_hash,
        terrain_sha256=terrain_hash,
        status=status,
        accepted_intervals=accepted,
        thresholds=thresholds.to_dict(),
        metrics=metrics,
        reasons=tuple(reasons),
    )


__all__ = (
    "AuditInterval",
    "AuditReason",
    "AuditThresholds",
    "ClipAudit",
    "audit_clip",
    "structural_model_sha256",
)
