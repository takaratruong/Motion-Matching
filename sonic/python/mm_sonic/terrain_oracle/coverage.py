"""Leakage-safe grouping and density-invariant terrain-motion coverage."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from mm_sonic.joints import ContractError

from .audit import AuditInterval, ClipAudit
from .canonical import CanonicalClip, CanonicalTerrainMesh
from .contact import (
    CONTACT_PLACEHOLDER_TAG,
    CONTACT_RECONSTRUCTION_TAG,
    CanonicalMeshQuery,
)
from .storage import ClipRecord, clip_digest


SCHEMA = "terrain-oracle-coverage/v1"
ACTION_CLASSES = (
    "walk", "run", "start", "stop", "reverse", "ascent", "descent", "turn",
    "sidestep", "other",
)
AXES = (
    "movement_facing_offset", "planar_speed", "yaw_rate", "transition",
    "support_phase", "support_leg", "step_forward", "step_lateral",
    "step_vertical", "surface_normal", "contact_yaw", "swing_clearance",
    "duration", "action_class",
)
SPLIT_PROPORTIONS = (0.8, 0.1, 0.1)
DIRECTION_SPEED_THRESHOLD_M_S = 0.1
_NONE_NUMERIC_AXES = frozenset(
    (
        "movement_facing_offset",
        "step_forward",
        "step_lateral",
        "step_vertical",
        "contact_yaw",
        "swing_clearance",
    )
)
_NUMERIC_EDGES = {
    "movement_facing_offset": (-math.pi, -2.35619449, -1.57079633, -0.78539816, 0.0, 0.78539816, 1.57079633, 2.35619449, math.pi),
    "planar_speed": (0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5),
    "yaw_rate": (-3.0, -1.5, -0.5, -0.1, 0.1, 0.5, 1.5, 3.0),
    "step_forward": (-0.6, -0.2, -0.05, 0.05, 0.2, 0.6),
    "step_lateral": (-0.4, -0.15, -0.03, 0.03, 0.15, 0.4),
    "step_vertical": (-0.3, -0.08, -0.015, 0.015, 0.08, 0.3),
    "contact_yaw": (-math.pi, -1.57079633, -0.39269908, 0.39269908, 1.57079633, math.pi),
    "swing_clearance": (0.0, 0.015, 0.04, 0.08, 0.16, 0.32),
    "duration": (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
}
_CATEGORIES = {
    "transition": ("start", "steady", "stop", "reverse"),
    "support_phase": ("flight", "single", "double"),
    "support_leg": ("none", "left", "right"),
    "surface_normal": tuple(
        f"{tilt}:{azimuth}"
        for tilt in ("flat", "moderate", "steep", "overhang")
        for azimuth in ("level", "forward", "left", "backward", "right")
    ) + ("none",),
    "action_class": ACTION_CLASSES,
}
_SHA_CHARS = frozenset("0123456789abcdef")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
ADJACENCY_RULE = (
    "Two cells are adjacent iff exactly one numeric axis differs by one "
    "published ordered category rank; none is nonordered; all categorical "
    "axes and other numeric axes are equal."
)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _SHA_CHARS for c in value):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{label} must be a nonempty string")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("coverage data must be finite JSON") from error


def _hash_text(digest: "hashlib._Hash", value: object) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_array(digest: "hashlib._Hash", value: object, dtype: np.dtype) -> None:
    array = np.ascontiguousarray(value, dtype=dtype)
    _hash_text(digest, array.shape)
    digest.update(array.tobytes(order="C"))


def _query_digest(query: CanonicalMeshQuery) -> str:
    digest = hashlib.sha256(b"terrain-oracle-exact-query-surface/v1\0")
    _hash_text(digest, query.mesh_sha256)
    _hash_text(digest, query.source_asset_sha256)
    _hash_array(digest, query._triangles, np.dtype(np.float64))
    _hash_array(digest, query._normals, np.dtype(np.float64))
    _hash_array(digest, query._face_indices, np.dtype(np.int64))
    _hash_array(digest, query.world_from_terrain.translation_world, np.dtype(np.float64))
    _hash_array(digest, query.world_from_terrain.quaternion_world_from_local_wxyz, np.dtype(np.float64))
    return digest.hexdigest()


def _same_transform(left: object, right: object) -> bool:
    return np.array_equal(left.translation_world, right.translation_world) and np.array_equal(
        left.quaternion_world_from_local_wxyz, right.quaternion_world_from_local_wxyz
    )


@dataclass(frozen=True)
class AcceptedClipRecord:
    """One fully reviewed authority bundle; no dictionary inference is allowed."""

    clip: CanonicalClip
    clip_record: ClipRecord
    audit: ClipAudit
    accepted_intervals: tuple[AuditInterval | tuple[int, int], ...]
    terrain_mesh: CanonicalTerrainMesh
    terrain_query: CanonicalMeshQuery
    model_sha256: str
    action_class: str
    procedural_family_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.clip, CanonicalClip):
            raise ContractError("accepted record clip must be CanonicalClip")
        self.clip.validate()
        if not isinstance(self.clip_record, ClipRecord):
            raise ContractError("accepted record requires ClipRecord")
        if not isinstance(self.audit, ClipAudit):
            raise ContractError("accepted record requires reviewed ClipAudit")
        if not isinstance(self.terrain_mesh, CanonicalTerrainMesh) or not isinstance(self.terrain_query, CanonicalMeshQuery):
            raise ContractError("accepted record requires exact mesh and query")
        _sha(self.model_sha256, "model_sha256")
        if self.action_class not in ACTION_CLASSES:
            raise ContractError("action_class is not a published coverage category")
        if self.procedural_family_id is not None:
            _text(self.procedural_family_id, "procedural_family_id")
        intervals = tuple(AuditInterval(*value) for value in self.accepted_intervals)
        if intervals != self.audit.accepted_intervals:
            raise ContractError("accepted intervals must exactly equal reviewed audit intervals")
        object.__setattr__(self, "accepted_intervals", intervals)
        clip = self.clip
        record = ClipRecord.from_dict(self.clip_record.to_dict())
        object.__setattr__(self, "clip_record", record)
        audit = self.audit
        if (
            record.clip_id != clip.clip_id
            or record.frame_count != clip.frame_count
            or audit.clip_id != clip.clip_id
            or audit.frame_count != clip.frame_count
            or audit.source_sha256 != clip.source.source_sha256
            or audit.model_sha256 != self.model_sha256
        ):
            raise ContractError("clip, artifact, source, model, or frame identities mismatch")
        actual_artifact = clip_digest(clip)
        if record.sha256 != actual_artifact or audit.clip_sha256 != actual_artifact:
            raise ContractError("audit and ClipRecord digest must bind the exact CanonicalClip")
        if audit.status == "rejected" or not intervals:
            raise ContractError("rejected or unreviewed clips cannot enter coverage")
        if (
            CONTACT_RECONSTRUCTION_TAG not in clip.action_tags
            or CONTACT_PLACEHOLDER_TAG in clip.action_tags
        ):
            raise ContractError("stale contact provenance is not accepted")
        if clip.terrain is None:
            raise ContractError("coverage requires explicit terrain")
        query = self.terrain_query
        verified = CanonicalMeshQuery(self.terrain_mesh, query.world_from_terrain)
        if (
            query.mesh_sha256 != verified.mesh_sha256
            or query.source_asset_sha256 != verified.source_asset_sha256
            or not np.array_equal(query._triangles, verified._triangles)
            or not np.array_equal(query._normals, verified._normals)
            or not np.array_equal(query._face_indices, verified._face_indices)
        ):
            raise ContractError("terrain query does not bind the accepted mesh")
        binding = clip.terrain
        if (
            binding.mesh_sha256 != query.mesh_sha256
            or binding.asset_sha256 != query.source_asset_sha256
            or not _same_transform(binding.world_from_terrain, query.world_from_terrain)
            or audit.terrain_sha256 != _query_digest(query)
        ):
            raise ContractError("terrain audit/query/binding identities mismatch")


def _semantic_digest(record: AcceptedClipRecord) -> str:
    clip = record.clip
    digest = hashlib.sha256(b"terrain-oracle-semantic-motion/v1\0")
    _hash_text(digest, clip.fps)
    _hash_text(digest, clip.joint_names)
    _hash_text(digest, clip.body_names)
    for name in (
        "root_position_world", "root_quaternion_world_wxyz", "joint_position",
        "root_linear_velocity_world", "root_angular_velocity_world", "joint_velocity",
        "body_position_world", "body_quaternion_world_wxyz",
        "body_linear_velocity_world", "body_angular_velocity_world",
        "sole_position_world", "sole_quaternion_world_wxyz", "heel_position_world",
        "toe_position_world", "contact", "contact_confidence",
    ):
        value = np.asarray(getattr(clip, name))
        _hash_text(digest, name)
        _hash_array(digest, value, value.dtype)
    for name in (
        "observed_travel_stick_xy", "observed_facing_stick_xy", "observed_mask",
        "inferred_velocity_local_xy", "inferred_facing_local_xy",
        "inferred_yaw_rate_rad_s",
    ):
        value = np.asarray(getattr(clip.commands, name))
        _hash_text(digest, name)
        _hash_array(digest, value, value.dtype)
    binding = clip.terrain
    _hash_text(digest, binding.asset_sha256)
    _hash_text(digest, binding.mesh_sha256)
    _hash_array(digest, binding.world_from_terrain.translation_world, np.dtype(np.float32))
    _hash_array(digest, binding.world_from_terrain.quaternion_world_from_local_wxyz, np.dtype(np.float32))
    return digest.hexdigest()


def _record_signature(record: AcceptedClipRecord) -> tuple[object, ...]:
    return (
        _semantic_digest(record), record.clip_record.sha256,
        json.dumps(record.audit.to_dict(), sort_keys=True),
        record.action_class, record.procedural_family_id,
    )


@dataclass(frozen=True)
class Deduplication:
    unique_records: tuple[AcceptedClipRecord, ...]
    semantic_digest_by_clip: Mapping[str, str]
    duplicate_clips_by_digest: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        records = tuple(self.unique_records)
        if any(not isinstance(record, AcceptedClipRecord) for record in records):
            raise ContractError("unique_records must contain AcceptedClipRecord values")
        record_digests = tuple(_semantic_digest(record) for record in records)
        if (
            record_digests != tuple(sorted(record_digests))
            or len(set(record_digests)) != len(record_digests)
        ):
            raise ContractError("unique_records must be sorted unique semantic records")
        semantic = dict(self.semantic_digest_by_clip)
        duplicates = {
            _sha(digest, "semantic digest"): tuple(clip_ids)
            for digest, clip_ids in dict(self.duplicate_clips_by_digest).items()
        }
        for clip_id, digest in semantic.items():
            _text(clip_id, "clip ID")
            _sha(digest, "semantic digest")
        if any(
            not values
            or tuple(sorted(values)) != values
            or len(values) != len(set(values))
            or any(type(value) is not str or not value for value in values)
            for values in duplicates.values()
        ):
            raise ContractError("duplicate clip groups must be sorted unique nonempty tuples")
        if (
            set(duplicates) != set(record_digests)
            or set(semantic) != {
                clip_id for values in duplicates.values() for clip_id in values
            }
            or any(
                semantic.get(clip_id) != digest
                for digest, values in duplicates.items()
                for clip_id in values
            )
            or any(
                record.clip.clip_id not in duplicates[digest]
                for record, digest in zip(records, record_digests)
            )
        ):
            raise ContractError("deduplication maps do not match unique records")
        object.__setattr__(self, "unique_records", records)
        object.__setattr__(self, "semantic_digest_by_clip", MappingProxyType(semantic))
        object.__setattr__(self, "duplicate_clips_by_digest", MappingProxyType(duplicates))

    def to_dict(self) -> dict[str, object]:
        return {
            "unique_clip_ids": [record.clip.clip_id for record in self.unique_records],
            "semantic_digest_by_clip": dict(
                sorted(self.semantic_digest_by_clip.items())
            ),
            "duplicate_clips_by_digest": {
                digest: list(self.duplicate_clips_by_digest[digest])
                for digest in sorted(self.duplicate_clips_by_digest)
            },
        }


def deduplicate(records: Sequence[AcceptedClipRecord]) -> Deduplication:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ContractError("records must be a sequence of AcceptedClipRecord")
    by_id: dict[str, AcceptedClipRecord] = {}
    by_digest: dict[str, list[AcceptedClipRecord]] = {}
    for record in records:
        if not isinstance(record, AcceptedClipRecord):
            raise ContractError("records must contain AcceptedClipRecord values")
        prior = by_id.get(record.clip.clip_id)
        if prior is not None and _record_signature(prior) != _record_signature(record):
            raise ContractError(f"conflicting content for clip ID {record.clip.clip_id}")
        by_id[record.clip.clip_id] = record
    for record in by_id.values():
        by_digest.setdefault(_semantic_digest(record), []).append(record)
    unique: list[AcceptedClipRecord] = []
    duplicate_map: dict[str, tuple[str, ...]] = {}
    semantic_map: dict[str, str] = {}
    for digest in sorted(by_digest):
        group = sorted(by_digest[digest], key=lambda item: item.clip.clip_id)
        labels = {item.action_class for item in group}
        if len(labels) != 1:
            raise ContractError("semantic duplicates have conflicting action classes")
        evidence = {
            (
                item.accepted_intervals,
                item.model_sha256,
                item.audit.terrain_sha256,
            )
            for item in group
        }
        if len(evidence) != 1:
            raise ContractError("semantic duplicates have conflicting reviewed evidence")
        unique.append(group[0])
        duplicate_map[digest] = tuple(item.clip.clip_id for item in group)
        semantic_map.update((item.clip.clip_id, digest) for item in group)
    return Deduplication(tuple(unique), semantic_map, duplicate_map)


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _groups(records: Sequence[AcceptedClipRecord]) -> tuple[dict[str, str], dict[str, str]]:
    deduped = deduplicate(records)
    ids = sorted(deduped.semantic_digest_by_clip)
    uf = _UnionFind(ids)
    buckets: dict[tuple[str, str], list[str]] = {}
    by_id = {record.clip.clip_id: record for record in records}
    for clip_id in ids:
        record = by_id[clip_id]
        semantic = deduped.semantic_digest_by_clip[clip_id]
        keys = [
            ("semantic", semantic),
            ("source", record.clip.source.source_sha256),
            ("terrain", record.clip.terrain.asset_sha256),
        ]
        if record.procedural_family_id is not None:
            keys.append(("procedural", record.procedural_family_id))
        for key in keys:
            buckets.setdefault(key, []).append(clip_id)
    for members in buckets.values():
        for member in members[1:]:
            uf.union(members[0], member)
    for clip_id in ids:
        record = by_id[clip_id]
        lineage = record.clip.mirror_of
        if lineage is None:
            if clip_id.endswith("__mirror"):
                raise ContractError("reserved mirror suffix requires explicit mirror_of lineage")
            continue
        if (
            clip_id != f"{lineage}__mirror"
            or lineage.endswith("__mirror")
            or lineage not in by_id
            or by_id[lineage].clip.mirror_of is not None
            or by_id[lineage].clip.clip_id.endswith("__mirror")
        ):
            raise ContractError("mirror_of must name a present valid original record")
        uf.union(clip_id, lineage)
    components: dict[str, list[str]] = {}
    for clip_id in ids:
        components.setdefault(uf.find(clip_id), []).append(clip_id)
    group_for_clip: dict[str, str] = {}
    semantic_group: dict[str, str] = {}
    for members in components.values():
        semantics = sorted({deduped.semantic_digest_by_clip[item] for item in members})
        group_id = _split_group_id(semantics)
        for member in members:
            group_for_clip[member] = group_id
            semantic_group[deduped.semantic_digest_by_clip[member]] = group_id
    return group_for_clip, semantic_group


def _split_group_id(semantic_digests: Sequence[str]) -> str:
    values = tuple(sorted(set(semantic_digests)))
    if not values or any(_sha(value, "semantic digest") != value for value in values):
        raise ContractError("split group requires semantic digests")
    return hashlib.sha256(
        ("terrain-oracle-split-group/v1\0" + "\0".join(values)).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    seed: str
    rule: str
    proportions: tuple[float, float, float]
    assignments: Mapping[str, str]
    group_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        _text(self.seed, "seed")
        _text(self.rule, "split rule")
        proportions = tuple(self.proportions)
        if proportions != SPLIT_PROPORTIONS:
            raise ContractError("split proportions must equal the published v1 boundaries")
        assignments = dict(self.assignments)
        groups = dict(self.group_ids)
        if set(assignments) != set(groups):
            raise ContractError("split assignments and group IDs must cover identical clips")
        for clip_id in assignments:
            _text(clip_id, "clip ID")
            if assignments[clip_id] not in ("train", "validation", "test"):
                raise ContractError("unknown split assignment")
            _sha(groups[clip_id], "split group ID")
        object.__setattr__(self, "proportions", proportions)
        object.__setattr__(self, "assignments", MappingProxyType(assignments))
        object.__setattr__(self, "group_ids", MappingProxyType(groups))

    def __getitem__(self, clip_id: str) -> str:
        return self.assignments[clip_id]

    def values(self):
        return self.assignments.values()

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed, "rule": self.rule, "proportions": list(self.proportions),
            "assignments": dict(sorted(self.assignments.items())),
            "group_ids": dict(sorted(self.group_ids.items())),
        }


def assign_grouped_splits(records: Sequence[AcceptedClipRecord], seed: str) -> SplitManifest:
    _text(seed, "seed")
    group_for_clip, _ = _groups(records)
    group_split: dict[str, str] = {}
    for group in sorted(set(group_for_clip.values())):
        digest = hashlib.sha256(b"terrain-oracle-split/v1\0" + seed.encode("utf-8") + b"\0" + group.encode("ascii")).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
        group_split[group] = "train" if fraction < 0.8 else ("validation" if fraction < 0.9 else "test")
    return SplitManifest(
        seed, "sha256('terrain-oracle-split/v1\\0',utf8(seed),'\\0',group-id); first-u64-be / 2^64; [0,.8)=train,[.8,.9)=validation,[.9,1)=test",
        SPLIT_PROPORTIONS,
        {clip_id: group_split[group] for clip_id, group in sorted(group_for_clip.items())},
        dict(sorted(group_for_clip.items())),
    )


def _number_label(value: float, edges: tuple[float, ...]) -> tuple[str, int]:
    if value < edges[0]:
        return f"[-inf,{edges[0]:g})", 0
    for index in range(len(edges) - 1):
        if value < edges[index + 1]:
            return f"[{edges[index]:g},{edges[index + 1]:g})", index + 1
    return f"[{edges[-1]:g},+inf)", len(edges)


def _yaw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    return np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))


def _surface_category(normal: np.ndarray) -> str:
    normal = np.asarray(normal, dtype=np.float64)
    tilt = math.acos(float(np.clip(normal[2] / max(np.linalg.norm(normal), 1e-12), -1, 1)))
    tilt_name = "flat" if tilt < math.radians(5) else ("moderate" if tilt < math.radians(20) else ("steep" if tilt < math.radians(60) else "overhang"))
    horizontal = math.hypot(float(normal[0]), float(normal[1]))
    if horizontal < 1e-8:
        azimuth = "level"
    else:
        angle = math.atan2(float(normal[1]), float(normal[0]))
        azimuth = ("forward", "left", "backward", "right")[int(math.floor(((angle + math.pi / 4) % (2 * math.pi)) / (math.pi / 2)))]
    return f"{tilt_name}:{azimuth}"


def _support_surface(
    query: CanonicalMeshQuery, probes: np.ndarray
) -> tuple[str, float | None]:
    surface = query.query(np.asarray(probes, dtype=np.float64))
    ray_distance = np.asarray(surface.downward_ray_distance_m, dtype=np.float64)
    ray_normal = np.asarray(surface.downward_ray_normal_world, dtype=np.float64)
    ray_valid = (
        (np.asarray(surface.downward_ray_face_index) >= 0)
        & np.isfinite(ray_distance)
        & (ray_normal[:, 2] >= 0.5)
    )
    if np.any(ray_valid):
        candidates = np.flatnonzero(ray_valid)
        selected = int(candidates[np.argmin(ray_distance[candidates])])
        return _surface_category(ray_normal[selected]), float(ray_distance[selected])
    closest_distance = np.asarray(surface.distance_m, dtype=np.float64)
    closest_normal = np.asarray(surface.surface_normal_world, dtype=np.float64)
    closest_valid = closest_normal[:, 2] >= 0.5
    if np.any(closest_valid):
        candidates = np.flatnonzero(closest_valid)
        selected = int(candidates[np.argmin(closest_distance[candidates])])
        return _surface_category(closest_normal[selected]), float(
            closest_distance[selected]
        )
    return "none", None


def _definitions() -> dict[str, object]:
    result: dict[str, object] = {}
    for axis in AXES:
        if axis in _NUMERIC_EDGES:
            edges = _NUMERIC_EDGES[axis]
            labels = [_number_label(edges[0] - 1, edges)[0]]
            labels.extend(_number_label(edges[i], edges)[0] for i in range(len(edges)))
            if axis in _NONE_NUMERIC_AXES:
                labels.append("none")
            result[axis] = {"kind": "numeric", "edges": list(edges), "categories": labels, "boundary_rule": "half-open; exact edge enters bin beginning at that edge; final category is overflow"}
            if axis == "movement_facing_offset":
                result[axis]["direction_speed_threshold_m_s"] = DIRECTION_SPEED_THRESHOLD_M_S
        else:
            result[axis] = {"kind": "categorical", "categories": list(_CATEGORIES[axis])}
    return result


def _freeze_definitions(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(AXES):
        raise ContractError("bin definition axes do not match v1")
    frozen = {}
    try:
        for axis in AXES:
            raw = value[axis]
            if not isinstance(raw, Mapping):
                raise ContractError("each bin definition must be a mapping")
            item = {"kind": raw["kind"], "categories": tuple(raw["categories"])}
            if raw["kind"] == "numeric":
                item["edges"] = tuple(float(edge) for edge in raw["edges"])
                item["boundary_rule"] = raw["boundary_rule"]
                if axis == "movement_facing_offset":
                    item["direction_speed_threshold_m_s"] = float(
                        raw["direction_speed_threshold_m_s"]
                    )
            frozen[axis] = MappingProxyType(item)
    except ContractError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("invalid bin definitions") from error
    return MappingProxyType(frozen)


def _plain_definitions(value: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for axis in AXES:
        raw = value[axis]
        item = {"kind": raw["kind"], "categories": list(raw["categories"])}
        if raw["kind"] == "numeric":
            item["edges"] = list(raw["edges"])
            item["boundary_rule"] = raw["boundary_rule"]
            if axis == "movement_facing_offset":
                item["direction_speed_threshold_m_s"] = raw[
                    "direction_speed_threshold_m_s"
                ]
        result[axis] = item
    return result


@dataclass(frozen=True)
class CoverageContribution:
    contribution_id: str
    semantic_digest: str
    dedup_group_id: str
    accepted_interval: tuple[int, int]

    def __post_init__(self) -> None:
        for name in (
            "contribution_id",
            "semantic_digest",
            "dedup_group_id",
        ):
            _sha(getattr(self, name), name)
        interval = tuple(self.accepted_interval)
        if (
            len(interval) != 2
            or type(interval[0]) is not int
            or type(interval[1]) is not int
            or not 0 <= interval[0] < interval[1]
        ):
            raise ContractError("contribution accepted_interval must be half-open")
        object.__setattr__(self, "accepted_interval", interval)
        if self.contribution_id != _contribution_id(
            self.semantic_digest, self.dedup_group_id, interval
        ):
            raise ContractError("contribution_id does not match its semantic evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "contribution_id": self.contribution_id,
            "semantic_digest": self.semantic_digest,
            "dedup_group_id": self.dedup_group_id,
            "accepted_interval": list(self.accepted_interval),
        }

    @classmethod
    def from_dict(cls, value: object) -> "CoverageContribution":
        expected = {
            "contribution_id", "semantic_digest", "dedup_group_id",
            "accepted_interval",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ContractError("coverage contribution fields do not match v1")
        try:
            return cls(
                value["contribution_id"],
                value["semantic_digest"],
                value["dedup_group_id"],
                tuple(value["accepted_interval"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("invalid coverage contribution") from error


def _contribution_id(
    semantic_digest: str, dedup_group_id: str, interval: tuple[int, int]
) -> str:
    return hashlib.sha256(
        _json_bytes(
            {
                "schema": "terrain-oracle-coverage-contribution/v1",
                "semantic_digest": semantic_digest,
                "dedup_group_id": dedup_group_id,
                "accepted_interval": list(interval),
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class CoverageCell:
    cell_id: str
    coordinates: tuple[str, ...]
    dedup_group_ids: tuple[str, ...]
    contribution_count: int
    intervals: tuple[tuple[int, int], ...]
    contributions: tuple[CoverageContribution, ...]

    def __post_init__(self) -> None:
        _sha(self.cell_id, "cell_id")
        coordinates = tuple(self.coordinates)
        if (
            len(coordinates) != len(AXES)
            or any(type(value) is not str or not value for value in coordinates)
            or self.cell_id != _cell_id(coordinates)
        ):
            raise ContractError("coverage cell coordinates/id mismatch")
        contributions = tuple(self.contributions)
        if (
            any(not isinstance(item, CoverageContribution) for item in contributions)
            or not contributions
            or tuple(item.contribution_id for item in contributions)
            != tuple(sorted(item.contribution_id for item in contributions))
            or len({item.contribution_id for item in contributions}) != len(contributions)
        ):
            raise ContractError("cell contributions must be sorted unique evidence")
        groups = tuple(sorted({item.dedup_group_id for item in contributions}))
        intervals = tuple(sorted({item.accepted_interval for item in contributions}))
        if (
            tuple(self.dedup_group_ids) != groups
            or tuple(tuple(value) for value in self.intervals) != intervals
            or self.contribution_count != len(contributions)
        ):
            raise ContractError("cell aggregates do not match contribution evidence")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "dedup_group_ids", groups)
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "contributions", contributions)

    @property
    def interval(self) -> tuple[int, int]:
        return self.intervals[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "coordinates": {axis: value for axis, value in zip(AXES, self.coordinates)},
            "dedup_group_ids": list(self.dedup_group_ids),
            "contribution_count": self.contribution_count,
            "intervals": [list(value) for value in self.intervals],
            "contributions": [value.to_dict() for value in self.contributions],
        }

    @classmethod
    def from_dict(cls, value: object) -> "CoverageCell":
        expected = {
            "cell_id", "coordinates", "dedup_group_ids",
            "contribution_count", "intervals", "contributions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ContractError("coverage cell fields do not match v1")
        coordinates = value["coordinates"]
        if not isinstance(coordinates, dict) or set(coordinates) != set(AXES):
            raise ContractError("coverage cell coordinate axes/order do not match v1")
        try:
            return cls(
                value["cell_id"],
                tuple(coordinates[axis] for axis in AXES),
                tuple(value["dedup_group_ids"]),
                value["contribution_count"],
                tuple(tuple(interval) for interval in value["intervals"]),
                tuple(
                    CoverageContribution.from_dict(item)
                    for item in value["contributions"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("invalid coverage cell") from error


def _cell_id(coordinates: tuple[str, ...]) -> str:
    return hashlib.sha256(_json_bytes({axis: value for axis, value in zip(AXES, coordinates)})).hexdigest()


def _interval_coordinates(record: AcceptedClipRecord, interval: AuditInterval):
    clip = record.clip
    start, end = interval
    count = end - start
    position = np.asarray(clip.root_position_world[start:end], np.float64)
    if count == 1:
        velocity = np.zeros((1, 3))
    else:
        velocity = np.gradient(position, 1.0 / clip.fps, axis=0)
    speed = np.linalg.norm(velocity[:, :2], axis=1)
    facing = _yaw(np.asarray(clip.root_quaternion_world_wxyz[start:end]))
    yaw_rate = np.gradient(facing, 1.0 / clip.fps) if count > 1 else np.zeros(1)
    movement = np.arctan2(velocity[:, 1], velocity[:, 0])
    offset = (movement - facing + math.pi) % (2 * math.pi) - math.pi
    transition = np.full(count, "steady", dtype=object)
    for i in range(1, count):
        before, after = speed[i - 1], speed[i]
        if before < 0.1 <= after:
            transition[i] = "start"
        elif before >= 0.1 > after:
            transition[i] = "stop"
        elif before >= 0.1 and after >= 0.1 and np.dot(velocity[i - 1, :2], velocity[i, :2]) < 0:
            transition[i] = "reverse"
    contact = np.asarray(clip.contact[start:end]) >= 0.5
    support_phase = np.where(np.sum(contact, axis=1) == 0, "flight", np.where(np.sum(contact, axis=1) == 2, "double", "single"))
    sole = np.asarray(clip.sole_position_world[start:end], np.float64)
    heel = np.asarray(clip.heel_position_world[start:end], np.float64)
    toe = np.asarray(clip.toe_position_world[start:end], np.float64)
    probes = np.stack((sole, heel, toe), axis=2)
    sole_yaw = np.stack(
        (
            _yaw(np.asarray(clip.sole_quaternion_world_wxyz[start:end, 0])),
            _yaw(np.asarray(clip.sole_quaternion_world_wxyz[start:end, 1])),
        ),
        axis=1,
    )
    step = np.full((count, 2, 3), np.nan)
    last_touchdown: list[np.ndarray | None] = [None, None]
    for i in range(count):
        for leg in range(2):
            touchdown = contact[i, leg] and (i == 0 or not contact[i - 1, leg])
            if touchdown:
                if last_touchdown[leg] is not None:
                    delta = sole[i, leg] - last_touchdown[leg]
                    c, s = math.cos(-facing[i]), math.sin(-facing[i])
                    step[i, leg] = (
                        c * delta[0] - s * delta[1],
                        s * delta[0] + c * delta[1],
                        delta[2],
                    )
                last_touchdown[leg] = sole[i, leg].copy()
    duration = count / clip.fps
    for i in range(count):
        contacting = tuple(int(leg) for leg in np.flatnonzero(contact[i]))
        contribution_legs: tuple[int | None, ...] = contacting or (None,)
        swing_clearance = "none"
        if len(contacting) == 1:
            swing_leg = 1 - contacting[0]
            _surface, clearance = _support_surface(
                record.terrain_query, probes[i, swing_leg]
            )
            if clearance is not None:
                swing_clearance = _number_label(
                    clearance, _NUMERIC_EDGES["swing_clearance"]
                )[0]
        for leg in contribution_legs:
            values: list[str] = [
                (
                    "none"
                    if speed[i] < DIRECTION_SPEED_THRESHOLD_M_S
                    else _number_label(
                        float(offset[i]), _NUMERIC_EDGES["movement_facing_offset"]
                    )[0]
                ),
                _number_label(float(speed[i]), _NUMERIC_EDGES["planar_speed"])[0],
                _number_label(float(yaw_rate[i]), _NUMERIC_EDGES["yaw_rate"])[0],
                str(transition[i]),
                str(support_phase[i]),
                "none" if leg is None else ("left" if leg == 0 else "right"),
            ]
            for component, axis in enumerate(
                ("step_forward", "step_lateral", "step_vertical")
            ):
                values.append(
                    "none"
                    if leg is None or np.isnan(step[i, leg, component])
                    else _number_label(
                        float(step[i, leg, component]), _NUMERIC_EDGES[axis]
                    )[0]
                )
            if leg is None:
                values.extend(("none", "none"))
            else:
                surface_normal, _clearance = _support_surface(
                    record.terrain_query, probes[i, leg]
                )
                values.append(surface_normal)
                contact_angle = (
                    float(sole_yaw[i, leg]) + math.pi
                ) % (2 * math.pi) - math.pi
                values.append(
                    _number_label(
                        contact_angle, _NUMERIC_EDGES["contact_yaw"]
                    )[0]
                )
            values.extend(
                (
                    swing_clearance,
                    _number_label(duration, _NUMERIC_EDGES["duration"])[0],
                    record.action_class,
                )
            )
            yield tuple(values)


def _adjacent(left: tuple[str, ...], right: tuple[str, ...], definitions: Mapping[str, object]) -> bool:
    differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(differences) != 1:
        return False
    index = differences[0]
    axis = AXES[index]
    if definitions[axis]["kind"] != "numeric":
        return False
    if left[index] == "none" or right[index] == "none":
        return False
    categories = definitions[axis]["categories"]
    if left[index] not in categories or right[index] not in categories:
        return False
    return abs(categories.index(left[index]) - categories.index(right[index])) == 1


def _derived_marginals(
    cells: tuple[CoverageCell, ...]
) -> dict[str, dict[str, int]]:
    evidence: dict[tuple[str, str], set[str]] = {}
    for cell in cells:
        contribution_ids = {item.contribution_id for item in cell.contributions}
        for axis, category in zip(AXES, cell.coordinates):
            evidence.setdefault((axis, category), set()).update(contribution_ids)
    result: dict[str, dict[str, int]] = {axis: {} for axis in AXES}
    for (axis, category), identifiers in sorted(evidence.items()):
        result[axis][category] = len(identifiers)
    return result


def _validate_contribution_groups(cells: tuple[CoverageCell, ...]) -> None:
    by_identifier: dict[str, CoverageContribution] = {}
    semantics_by_group: dict[str, set[str]] = {}
    for cell in cells:
        for contribution in cell.contributions:
            previous = by_identifier.setdefault(
                contribution.contribution_id, contribution
            )
            if previous != contribution:
                raise ContractError("one contribution ID has conflicting evidence")
            semantics_by_group.setdefault(
                contribution.dedup_group_id, set()
            ).add(contribution.semantic_digest)
    for group_id, semantic_digests in semantics_by_group.items():
        if group_id != _split_group_id(tuple(semantic_digests)):
            raise ContractError("dedup group ID does not match semantic evidence")


def _derived_components(
    cells: tuple[CoverageCell, ...], definitions: Mapping[str, object]
) -> tuple[tuple[str, ...], ...]:
    by_id = {cell.cell_id: cell for cell in cells}
    remaining = set(by_id)
    components: list[tuple[str, ...]] = []
    while remaining:
        queue = [min(remaining)]
        component: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in component:
                continue
            component.add(current)
            for candidate in sorted(remaining - component):
                if _adjacent(
                    by_id[current].coordinates,
                    by_id[candidate].coordinates,
                    definitions,
                ):
                    queue.append(candidate)
        remaining -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


@dataclass(frozen=True)
class CoverageManifest:
    schema: str
    adjacency_rule: str
    bin_definitions: Mapping[str, object]
    marginal_counts: Mapping[str, Mapping[str, int]]
    occupied_cells: tuple[CoverageCell, ...]
    connected_components: tuple[tuple[str, ...], ...]
    content_sha256: str

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ContractError("unknown coverage schema")
        if self.adjacency_rule != ADJACENCY_RULE:
            raise ContractError("adjacency rule does not equal the published v1 rule")
        _sha(self.content_sha256, "content_sha256")
        definitions = _freeze_definitions(self.bin_definitions)
        if _plain_definitions(definitions) != _definitions():
            raise ContractError("bin definitions do not equal the published v1 atlas")
        cells = tuple(self.occupied_cells)
        if (
            any(not isinstance(cell, CoverageCell) for cell in cells)
            or tuple(cell.cell_id for cell in cells)
            != tuple(sorted(cell.cell_id for cell in cells))
            or len({cell.cell_id for cell in cells}) != len(cells)
        ):
            raise ContractError("occupied cells must be canonically sorted and unique")
        for cell in cells:
            for axis, category in zip(AXES, cell.coordinates):
                if category not in definitions[axis]["categories"]:
                    raise ContractError("cell coordinate is outside published categories")
        _validate_contribution_groups(cells)
        marginal = {
            axis: dict(self.marginal_counts[axis])
            for axis in AXES
        } if isinstance(self.marginal_counts, Mapping) and set(self.marginal_counts) == set(AXES) else None
        if marginal is None or marginal != _derived_marginals(cells):
            raise ContractError("marginal counts do not match contribution evidence")
        components = tuple(tuple(component) for component in self.connected_components)
        if components != _derived_components(cells, definitions):
            raise ContractError("connected components do not match deterministic adjacency")
        object.__setattr__(self, "bin_definitions", definitions)
        object.__setattr__(self, "marginal_counts", MappingProxyType({axis: MappingProxyType(dict(sorted(marginal[axis].items()))) for axis in AXES}))
        object.__setattr__(self, "occupied_cells", cells)
        object.__setattr__(self, "connected_components", components)
        expected = hashlib.sha256(_json_bytes(self._dict_without_hash())).hexdigest()
        if self.content_sha256 != expected:
            raise ContractError("coverage content hash mismatch")

    @property
    def connected_cells(self):
        return self.connected_components

    def _dict_without_hash(self) -> dict[str, object]:
        return {
            "schema": self.schema, "adjacency_rule": self.adjacency_rule,
            "bin_definitions": _plain_definitions(self.bin_definitions),
            "marginal_counts": {axis: dict(sorted(self.marginal_counts[axis].items())) for axis in AXES},
            "occupied_cells": [cell.to_dict() for cell in self.occupied_cells],
            "connected_components": [list(component) for component in self.connected_components],
        }

    def to_dict(self) -> dict[str, object]:
        result = self._dict_without_hash()
        result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, value: object) -> "CoverageManifest":
        if not isinstance(value, dict) or set(value) != {
            "schema", "adjacency_rule", "bin_definitions", "marginal_counts",
            "occupied_cells", "connected_components", "content_sha256",
        }:
            raise ContractError("coverage document fields do not match schema")
        if (
            set(value["bin_definitions"]) != set(AXES)
            or set(value["marginal_counts"]) != set(AXES)
        ):
            raise ContractError("coverage axes/order do not match v1")
        try:
            cells = tuple(CoverageCell.from_dict(raw) for raw in value["occupied_cells"])
            components = tuple(tuple(component) for component in value["connected_components"])
            return cls(value["schema"], value["adjacency_rule"], value["bin_definitions"], value["marginal_counts"], cells, components, value["content_sha256"])
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError("invalid coverage manifest") from error


def build_coverage(records: Sequence[AcceptedClipRecord]) -> CoverageManifest:
    deduped = deduplicate(records)
    _, semantic_groups = _groups(records)
    definitions = _definitions()
    contributions: dict[
        tuple[str, ...], dict[str, CoverageContribution]
    ] = {}
    for record in deduped.unique_records:
        semantic = _semantic_digest(record)
        group = semantic_groups[semantic]
        for interval in record.accepted_intervals:
            interval_tuple = tuple(interval)
            contribution = CoverageContribution(
                _contribution_id(semantic, group, interval_tuple),
                semantic,
                group,
                interval_tuple,
            )
            for coordinates in _interval_coordinates(record, interval):
                existing = contributions.setdefault(coordinates, {}).get(
                    contribution.contribution_id
                )
                if existing is not None and existing != contribution:
                    raise ContractError("density-equivalent contribution evidence conflicts")
                contributions[coordinates][contribution.contribution_id] = contribution
    cells = tuple(sorted((
        (
            lambda evidence: CoverageCell(
                _cell_id(coordinates),
                coordinates,
                tuple(sorted({item.dedup_group_id for item in evidence})),
                len(evidence),
                tuple(sorted({item.accepted_interval for item in evidence})),
                evidence,
            )
        )(tuple(contributions[coordinates][key] for key in sorted(contributions[coordinates])))
        for coordinates in contributions
    ), key=lambda cell: cell.cell_id))
    marginal = _derived_marginals(cells)
    components = _derived_components(cells, definitions)
    without_hash = {
        "schema": SCHEMA,
        "adjacency_rule": ADJACENCY_RULE,
        "bin_definitions": definitions,
        "marginal_counts": marginal,
        "occupied_cells": [cell.to_dict() for cell in cells],
        "connected_components": [list(component) for component in components],
    }
    content_hash = hashlib.sha256(_json_bytes(without_hash)).hexdigest()
    return CoverageManifest(
        SCHEMA, without_hash["adjacency_rule"], definitions, marginal, cells,
        components, content_hash,
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    result = renameat2(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(destination), _RENAME_NOREPLACE)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(number, os.strerror(number), str(destination))
    raise OSError(number, os.strerror(number), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_coverage(path: Path, manifest: CoverageManifest) -> None:
    if not isinstance(path, Path) or path.name in ("", ".", ".."):
        raise ContractError("freeze_coverage requires a narrow pathlib.Path target")
    if not isinstance(manifest, CoverageManifest):
        raise ContractError("freeze_coverage requires CoverageManifest")
    CoverageManifest.from_dict(manifest.to_dict())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(manifest.to_dict()))
            stream.flush()
            os.fsync(stream.fileno())
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = (
    "ACTION_CLASSES", "AcceptedClipRecord", "CoverageCell",
    "CoverageContribution", "CoverageManifest",
    "Deduplication", "SplitManifest", "assign_grouped_splits", "build_coverage",
    "deduplicate", "freeze_coverage",
)
