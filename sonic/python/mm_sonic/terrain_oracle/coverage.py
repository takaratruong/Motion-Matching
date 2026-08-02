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
from .storage import ClipRecord, _clip_arrays, _deterministic_npz_bytes


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
    "support_leg": ("none", "left", "right", "both"),
    "surface_normal": tuple(
        f"{tilt}:{azimuth}"
        for tilt in ("flat", "moderate", "steep", "overhang")
        for azimuth in ("level", "forward", "left", "backward", "right")
    ),
    "action_class": ACTION_CLASSES,
}
_SHA_CHARS = frozenset("0123456789abcdef")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


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
        record = self.clip_record
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
        actual_artifact = hashlib.sha256(_deterministic_npz_bytes(_clip_arrays(clip))).hexdigest()
        if record.sha256 != actual_artifact:
            raise ContractError("ClipRecord digest does not bind the validated CanonicalClip")
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
        object.__setattr__(self, "semantic_digest_by_clip", MappingProxyType(dict(self.semantic_digest_by_clip)))
        object.__setattr__(self, "duplicate_clips_by_digest", MappingProxyType(dict(self.duplicate_clips_by_digest)))


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
        lineage = record.clip.mirror_of or (
            clip_id[:-10] if clip_id.endswith("__mirror") else clip_id
        )
        keys.append(("mirror", lineage))
        if record.procedural_family_id is not None:
            keys.append(("procedural", record.procedural_family_id))
        for key in keys:
            buckets.setdefault(key, []).append(clip_id)
    for members in buckets.values():
        for member in members[1:]:
            uf.union(members[0], member)
    components: dict[str, list[str]] = {}
    for clip_id in ids:
        components.setdefault(uf.find(clip_id), []).append(clip_id)
    group_for_clip: dict[str, str] = {}
    semantic_group: dict[str, str] = {}
    for members in components.values():
        semantics = sorted({deduped.semantic_digest_by_clip[item] for item in members})
        group_id = hashlib.sha256(("terrain-oracle-split-group/v1\0" + "\0".join(semantics)).encode()).hexdigest()
        for member in members:
            group_for_clip[member] = group_id
            semantic_group[deduped.semantic_digest_by_clip[member]] = group_id
    return group_for_clip, semantic_group


@dataclass(frozen=True)
class SplitManifest:
    seed: str
    rule: str
    proportions: tuple[float, float, float]
    assignments: Mapping[str, str]
    group_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", MappingProxyType(dict(self.assignments)))
        object.__setattr__(self, "group_ids", MappingProxyType(dict(self.group_ids)))

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


def _definitions() -> dict[str, object]:
    result: dict[str, object] = {}
    for axis in AXES:
        if axis in _NUMERIC_EDGES:
            edges = _NUMERIC_EDGES[axis]
            labels = [_number_label(edges[0] - 1, edges)[0]]
            labels.extend(_number_label(edges[i], edges)[0] for i in range(len(edges)))
            result[axis] = {"kind": "numeric", "edges": list(edges), "categories": labels, "boundary_rule": "half-open; exact edge enters bin beginning at that edge; final category is overflow"}
        else:
            result[axis] = {"kind": "categorical", "categories": list(_CATEGORIES[axis])}
    return result


def _freeze_definitions(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = {}
    for axis in AXES:
        raw = value[axis]
        item = {"kind": raw["kind"], "categories": tuple(raw["categories"])}
        if raw["kind"] == "numeric":
            item["edges"] = tuple(float(edge) for edge in raw["edges"])
            item["boundary_rule"] = raw["boundary_rule"]
        frozen[axis] = MappingProxyType(item)
    return MappingProxyType(frozen)


def _plain_definitions(value: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for axis in AXES:
        raw = value[axis]
        item = {"kind": raw["kind"], "categories": list(raw["categories"])}
        if raw["kind"] == "numeric":
            item["edges"] = list(raw["edges"])
            item["boundary_rule"] = raw["boundary_rule"]
        result[axis] = item
    return result


@dataclass(frozen=True)
class CoverageCell:
    cell_id: str
    coordinates: tuple[str, ...]
    source_ids: tuple[str, ...]
    dedup_group_ids: tuple[str, ...]
    contribution_count: int
    intervals: tuple[tuple[int, int], ...]

    @property
    def interval(self) -> tuple[int, int]:
        return self.intervals[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "coordinates": {axis: value for axis, value in zip(AXES, self.coordinates)},
            "source_ids": list(self.source_ids), "dedup_group_ids": list(self.dedup_group_ids),
            "contribution_count": self.contribution_count,
            "intervals": [list(value) for value in self.intervals],
        }


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
    support_leg = np.array([
        "both" if left and right else ("left" if left else ("right" if right else "none"))
        for left, right in contact
    ], dtype=object)
    sole = np.asarray(clip.sole_position_world[start:end], np.float64)
    surface = record.terrain_query.query(sole.reshape((-1, 3)))
    clearance = np.asarray(surface.distance_m).reshape((count, 2))
    root_surface = record.terrain_query.query(position)
    normals = np.asarray(root_surface.surface_normal_world)
    sole_yaw = _yaw(np.asarray(clip.sole_quaternion_world_wxyz[start:end, 0]))
    step = np.full((count, 3), np.nan)
    last_touchdown: list[np.ndarray | None] = [None, None]
    for i in range(count):
        for leg in range(2):
            touchdown = contact[i, leg] and (i == 0 or not contact[i - 1, leg])
            if touchdown:
                if last_touchdown[leg] is not None:
                    delta = sole[i, leg] - last_touchdown[leg]
                    c, s = math.cos(-facing[i]), math.sin(-facing[i])
                    step[i] = (c * delta[0] - s * delta[1], s * delta[0] + c * delta[1], delta[2])
                last_touchdown[leg] = sole[i, leg].copy()
    duration = count / clip.fps
    for i in range(count):
        values: list[str] = []
        for axis, scalar in (
            ("movement_facing_offset", offset[i]), ("planar_speed", speed[i]),
            ("yaw_rate", yaw_rate[i]),
        ):
            values.append(_number_label(float(scalar), _NUMERIC_EDGES[axis])[0])
        values.extend((str(transition[i]), str(support_phase[i]), str(support_leg[i])))
        for index, axis in enumerate(("step_forward", "step_lateral", "step_vertical")):
            values.append("none" if np.isnan(step[i, index]) else _number_label(float(step[i, index]), _NUMERIC_EDGES[axis])[0])
        values.append(_surface_category(normals[i]))
        values.append(_number_label(float(sole_yaw[i]), _NUMERIC_EDGES["contact_yaw"])[0] if np.any(contact[i]) else "none")
        swing = clearance[i, ~contact[i]]
        values.append(_number_label(float(np.max(swing)) if len(swing) else 0.0, _NUMERIC_EDGES["swing_clearance"])[0])
        values.append(_number_label(duration, _NUMERIC_EDGES["duration"])[0])
        values.append(record.action_class)
        yield tuple(values)


def _adjacent(left: tuple[str, ...], right: tuple[str, ...], definitions: Mapping[str, object]) -> bool:
    differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(differences) != 1:
        return False
    index = differences[0]
    axis = AXES[index]
    if definitions[axis]["kind"] != "numeric":
        return False
    categories = definitions[axis]["categories"]
    if left[index] not in categories or right[index] not in categories:
        return False
    return abs(categories.index(left[index]) - categories.index(right[index])) == 1


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
        _sha(self.content_sha256, "content_sha256")
        object.__setattr__(self, "bin_definitions", _freeze_definitions(self.bin_definitions))
        object.__setattr__(self, "marginal_counts", MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.marginal_counts.items()}))
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
        if tuple(value["bin_definitions"]) != AXES or tuple(value["marginal_counts"]) != AXES:
            raise ContractError("coverage axes/order do not match v1")
        cells = []
        ids = set()
        for raw in value["occupied_cells"]:
            if not isinstance(raw, dict) or set(raw) != {"cell_id", "coordinates", "source_ids", "dedup_group_ids", "contribution_count", "intervals"}:
                raise ContractError("invalid occupied cell fields")
            coordinates = tuple(raw["coordinates"].get(axis) for axis in AXES)
            if set(raw["coordinates"]) != set(AXES) or _cell_id(coordinates) != raw["cell_id"]:
                raise ContractError("occupied cell coordinates/id mismatch")
            if raw["cell_id"] in ids:
                raise ContractError("duplicate occupied cell")
            ids.add(raw["cell_id"])
            cells.append(CoverageCell(raw["cell_id"], coordinates, tuple(raw["source_ids"]), tuple(raw["dedup_group_ids"]), raw["contribution_count"], tuple(tuple(x) for x in raw["intervals"])))
        if [cell.cell_id for cell in cells] != sorted(ids):
            raise ContractError("occupied cells must be canonically sorted")
        components = tuple(tuple(component) for component in value["connected_components"])
        flattened = [cell for component in components for cell in component]
        if sorted(flattened) != sorted(ids) or len(flattened) != len(set(flattened)):
            raise ContractError("connected components must partition occupied cells")
        return cls(value["schema"], value["adjacency_rule"], value["bin_definitions"], value["marginal_counts"], tuple(cells), components, value["content_sha256"])


def build_coverage(records: Sequence[AcceptedClipRecord]) -> CoverageManifest:
    deduped = deduplicate(records)
    _, semantic_groups = _groups(records)
    definitions = _definitions()
    contributions: dict[tuple[str, ...], set[tuple[str, int]]] = {}
    intervals_by_cell: dict[tuple[str, ...], set[tuple[int, int]]] = {}
    sources_by_cell: dict[tuple[str, ...], set[str]] = {}
    groups_by_cell: dict[tuple[str, ...], set[str]] = {}
    for record in deduped.unique_records:
        semantic = _semantic_digest(record)
        for interval_index, interval in enumerate(record.accepted_intervals):
            for coordinates in _interval_coordinates(record, interval):
                contributions.setdefault(coordinates, set()).add((semantic, interval_index))
                intervals_by_cell.setdefault(coordinates, set()).add(tuple(interval))
                sources_by_cell.setdefault(coordinates, set()).add(semantic)
                groups_by_cell.setdefault(coordinates, set()).add(semantic_groups[semantic])
    cells = tuple(sorted((
        CoverageCell(
            _cell_id(coordinates), coordinates,
            tuple(sorted(sources_by_cell[coordinates])),
            tuple(sorted(groups_by_cell[coordinates])),
            len(contributions[coordinates]),
            tuple(sorted(intervals_by_cell[coordinates])),
        )
        for coordinates in contributions
    ), key=lambda cell: cell.cell_id))
    marginal: dict[str, dict[str, int]] = {axis: {} for axis in AXES}
    marginal_seen: dict[tuple[str, str], set[tuple[str, int]]] = {}
    for coordinates, contribution_set in contributions.items():
        for axis, category in zip(AXES, coordinates):
            marginal_seen.setdefault((axis, category), set()).update(contribution_set)
    for (axis, category), values in sorted(marginal_seen.items()):
        marginal[axis][category] = len(values)
    remaining = {cell.cell_id: cell for cell in cells}
    components: list[tuple[str, ...]] = []
    while remaining:
        seed = min(remaining)
        queue = [seed]
        component = set()
        while queue:
            current = queue.pop(0)
            if current in component:
                continue
            component.add(current)
            left = remaining[current].coordinates
            for candidate in sorted(remaining):
                if candidate not in component and _adjacent(left, remaining[candidate].coordinates, definitions):
                    queue.append(candidate)
        for item in component:
            remaining.pop(item)
        components.append(tuple(sorted(component)))
    components.sort()
    without_hash = {
        "schema": SCHEMA,
        "adjacency_rule": "Two cells are adjacent iff exactly one numeric axis differs by one published category rank; all categorical axes and other numeric axes are equal.",
        "bin_definitions": definitions,
        "marginal_counts": marginal,
        "occupied_cells": [cell.to_dict() for cell in cells],
        "connected_components": [list(component) for component in components],
    }
    content_hash = hashlib.sha256(_json_bytes(without_hash)).hexdigest()
    return CoverageManifest(
        SCHEMA, without_hash["adjacency_rule"], definitions, marginal, cells,
        tuple(components), content_hash,
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
    "ACTION_CLASSES", "AcceptedClipRecord", "CoverageCell", "CoverageManifest",
    "Deduplication", "SplitManifest", "assign_grouped_splits", "build_coverage",
    "deduplicate", "freeze_coverage",
)
