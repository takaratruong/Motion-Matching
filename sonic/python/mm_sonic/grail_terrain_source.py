"""Build a small, clean 50 Hz terrain-motion archive from GRAIL sweep shards.

The released GRAIL robot records are Z-up, 25 Hz, and use XYZW root
quaternions with 29 joints in MuJoCo order.  This module keeps those
conventions explicit at the import boundary and emits the flat XYZW archive
consumed by the terrain catalog and kinematic viewer.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_GRAIL_FAMILIES = (
    "bfix_stair_p1",
    "bfix_stair_p2",
    "gnd_stair_p2",
    "num_stair_p1",
    "num_stair_p2",
)

EXPECTED_GRAIL_FAMILY_COUNTS = (
    ("bfix_stair_p1", 2082),
    ("bfix_stair_p2", 1660),
    ("gnd_stair_p2", 971),
    ("num_stair_p1", 4012),
    ("num_stair_p2", 3463),
)

ISAACLAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

ISAACLAB_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

MUJOCO_TO_ISAACLAB_JOINT = np.asarray(
    (
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
        16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
    ),
    dtype=np.int64,
)


@dataclass(frozen=True)
class ResampledGrailMotion:
    root_position: np.ndarray
    root_quaternion_xyzw: np.ndarray
    dof_mujoco: np.ndarray
    fps: float


@dataclass(frozen=True)
class RawGrailMotion:
    """One validated raw GRAIL robot record in its released conventions."""

    root_position: np.ndarray
    root_quaternion_xyzw: np.ndarray
    dof_mujoco: np.ndarray
    fps: float
    source_key: str


@dataclass(frozen=True)
class GrailClipRecord:
    family: str
    shard_path: Path
    stem: str
    robot_path: Path
    usd_path: Path
    n_frames: int
    terrain_position_env: np.ndarray
    terrain_rotation_env_wxyz: np.ndarray
    pose_source: str


@dataclass(frozen=True)
class GrailGeometry:
    """Geometry and motion facts used to stratify the staged import."""

    stem: str
    fps: float
    n_frames: int
    n_steps: int
    rise_m: float
    tread_m: float
    travel_yaw_rad: float
    ascending: bool
    root_delta_z_m: float


@dataclass(frozen=True)
class GrailSelectionCandidate:
    """One strictly joined source clip with its selection measurements."""

    record: GrailClipRecord
    geometry: GrailGeometry
    traversal: str
    approach_heading_delta_deg: float

    @property
    def approach_angle_bin(self) -> str:
        return classify_approach_heading(self.approach_heading_delta_deg)


@dataclass(frozen=True)
class StagedSelectionConfig:
    """Exact source-clip marginals for a reproducible staged import."""

    seed: str
    family_targets: tuple[tuple[str, str, int], ...]
    angle_targets: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class GrailForwardKinematics:
    body_position_world: np.ndarray
    body_quaternion_world_xyzw: np.ndarray
    body_names: tuple[str, ...]


@dataclass(frozen=True)
class GrailArchiveClip:
    """One converted clip, before concatenation into the flat zarr archive."""

    candidate: GrailSelectionCandidate
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_xyzw: np.ndarray
    body_linear_velocity_world: np.ndarray
    fps: float

    def __post_init__(self) -> None:
        joint_position = _finite_array(
            self.joint_position,
            shape_tail=(29,),
            label="archive joint_position",
        )
        frames = len(joint_position)
        joint_velocity = _finite_array(
            self.joint_velocity,
            shape_tail=(29,),
            label="archive joint_velocity",
        )
        body_position = _finite_array(
            self.body_position_world,
            shape_tail=(30, 3),
            label="archive body_position_world",
        )
        body_quaternion = _finite_array(
            self.body_quaternion_world_xyzw,
            shape_tail=(30, 4),
            label="archive body_quaternion_world_xyzw",
        )
        body_velocity = _finite_array(
            self.body_linear_velocity_world,
            shape_tail=(30, 3),
            label="archive body_linear_velocity_world",
        )
        if (
            frames < 3
            or len(joint_velocity) != frames
            or len(body_position) != frames
            or len(body_quaternion) != frames
            or len(body_velocity) != frames
        ):
            raise ValueError("archive clip arrays must have the same T >= 3")
        candidate = self.candidate
        if not isinstance(candidate, GrailSelectionCandidate):
            raise TypeError(
                "archive candidate must be a GrailSelectionCandidate"
            )
        record = candidate.record
        geometry = candidate.geometry
        if (
            record.stem != geometry.stem
            or not record.stem
            or not record.family
            or not record.pose_source
        ):
            raise ValueError("archive record/geometry metadata is inconsistent")
        if (
            type(record.n_frames) is not int
            or record.n_frames < 2
            or geometry.n_frames != record.n_frames
            or geometry.fps != 25.0
            or type(geometry.n_steps) is not int
            or geometry.n_steps <= 0
            or type(geometry.ascending) is not bool
        ):
            raise ValueError("archive source frame/geometry metadata is invalid")
        expected_frames = 2 * record.n_frames - 1
        if frames != expected_frames:
            raise ValueError(
                "archive output must contain 2 * source frames - 1 rows; "
                f"got {frames}, expected {expected_frames}"
            )
        geometry_scalars = (
            geometry.rise_m,
            geometry.tread_m,
            geometry.travel_yaw_rad,
            geometry.root_delta_z_m,
            candidate.approach_heading_delta_deg,
        )
        if (
            not all(np.isfinite(float(value)) for value in geometry_scalars)
            or geometry.rise_m <= 0.0
            or geometry.tread_m <= 0.0
            or candidate.traversal not in {"up", "down", "roundtrip"}
        ):
            raise ValueError("archive selection metadata is invalid")
        terrain_position = np.asarray(record.terrain_position_env)
        terrain_rotation = np.asarray(record.terrain_rotation_env_wxyz)
        if (
            terrain_position.shape != (3,)
            or terrain_rotation.shape != (4,)
            or terrain_position.dtype.kind not in "iuf"
            or terrain_rotation.dtype.kind not in "iuf"
            or not np.isfinite(terrain_position).all()
            or not np.isfinite(terrain_rotation).all()
            or not np.isclose(
                np.linalg.norm(terrain_rotation),
                1.0,
                atol=1.0e-4,
                rtol=0.0,
            )
        ):
            raise ValueError("archive terrain pose metadata is invalid")
        norms = np.linalg.norm(body_quaternion, axis=-1)
        if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=0.0):
            raise ValueError("archive body XYZW quaternions must be normalized")
        if float(self.fps) != 50.0:
            raise ValueError("archive clips must be exactly 50 Hz")
        object.__setattr__(
            self,
            "joint_position",
            np.ascontiguousarray(joint_position, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "joint_velocity",
            np.ascontiguousarray(joint_velocity, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "body_position_world",
            np.ascontiguousarray(body_position, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "body_quaternion_world_xyzw",
            np.ascontiguousarray(body_quaternion, dtype=np.float32),
        )
        object.__setattr__(
            self,
            "body_linear_velocity_world",
            np.ascontiguousarray(body_velocity, dtype=np.float32),
        )
        object.__setattr__(self, "fps", 50.0)


def stage_one_selection_config(
    *, seed: str = "grail-stage-one-v1"
) -> StagedSelectionConfig:
    """Return the reviewed 120-source stage-one sampling contract."""

    if not isinstance(seed, str) or not seed:
        raise ValueError("selection seed must be a nonempty string")
    return StagedSelectionConfig(
        seed=seed,
        family_targets=(
            ("up", "bfix_stair_p2", 12),
            ("up", "gnd_stair_p2", 12),
            ("up", "num_stair_p1", 12),
            ("up", "num_stair_p2", 12),
            ("down", "bfix_stair_p1", 16),
            ("down", "bfix_stair_p2", 16),
            ("down", "num_stair_p1", 16),
            ("roundtrip", "bfix_stair_p1", 12),
            ("roundtrip", "bfix_stair_p2", 12),
        ),
        angle_targets=(
            ("up", "straight", 20),
            ("up", "left_10_25", 10),
            ("up", "right_10_25", 10),
            ("up", "left_25_50", 4),
            ("up", "right_25_50", 4),
            ("down", "straight", 20),
            ("down", "left_10_25", 10),
            ("down", "right_10_25", 10),
            ("down", "left_25_50", 4),
            ("down", "right_25_50", 4),
            ("roundtrip", "straight", 10),
            ("roundtrip", "left_10_25", 5),
            ("roundtrip", "right_10_25", 5),
            ("roundtrip", "left_25_50", 2),
            ("roundtrip", "right_25_50", 2),
        ),
    )


def classify_approach_heading(delta_degrees: float) -> str:
    """Bin signed path-to-stair heading error; positive angles turn left."""

    delta = float(delta_degrees)
    if not np.isfinite(delta):
        raise ValueError("approach heading delta must be finite")
    magnitude = abs(delta)
    if magnitude <= 10.0:
        return "straight"
    side = "left" if delta > 0.0 else "right"
    if magnitude <= 25.0:
        return f"{side}_10_25"
    if magnitude <= 50.0:
        return f"{side}_25_50"
    return "hard_back"


def classify_traversal(
    root_height: object,
    *,
    minimum_vertical_range_m: float = 0.25,
) -> str:
    """Classify a clip from robust root-height endpoints and vertical range."""

    height = np.asarray(root_height, dtype=np.float64)
    threshold = float(minimum_vertical_range_m)
    if (
        height.ndim != 1
        or len(height) < 20
        or not np.isfinite(height).all()
        or not np.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise ValueError(
            "traversal classification needs at least 20 finite root heights "
            "and a positive range threshold"
        )
    low, high = np.quantile(height, (0.02, 0.98))
    vertical_range = float(high - low)
    if vertical_range < threshold:
        return "partial"

    endpoint_count = max(5, min(10, len(height) // 10))
    start = (float(np.median(height[:endpoint_count])) - low) / vertical_range
    end = (float(np.median(height[-endpoint_count:])) - low) / vertical_range
    near_low = 0.2
    near_high = 0.8
    if start <= near_low and end >= near_high:
        return "up"
    if start >= near_high and end <= near_low:
        return "down"
    if start <= near_low and end <= near_low:
        return "roundtrip"
    return "partial"


def _geometry_bucket(geometry: GrailGeometry) -> tuple[str, str, str]:
    if geometry.rise_m < 0.16:
        rise = "low"
    elif geometry.rise_m <= 0.195:
        rise = "nominal"
    else:
        rise = "high"
    if geometry.tread_m < 0.30:
        tread = "short"
    elif geometry.tread_m <= 0.45:
        tread = "nominal"
    else:
        tread = "deep"
    if geometry.n_steps <= 4:
        steps = "3_4"
    elif geometry.n_steps <= 6:
        steps = "5_6"
    else:
        steps = "7_plus"
    return rise, tread, steps


def _seeded_candidate_order(
    candidates: list[GrailSelectionCandidate],
    *,
    seed: str,
) -> list[GrailSelectionCandidate]:
    """Round-robin geometry buckets, with stable hashing inside each bucket."""

    buckets: dict[
        tuple[str, str, str], list[GrailSelectionCandidate]
    ] = {}
    for candidate in candidates:
        buckets.setdefault(_geometry_bucket(candidate.geometry), []).append(
            candidate
        )
    for bucket_candidates in buckets.values():
        bucket_candidates.sort(
            key=lambda candidate: hashlib.sha256(
                f"{seed}\0{candidate.record.stem}".encode("utf-8")
            ).digest()
        )
    ordered: list[GrailSelectionCandidate] = []
    bucket_names = sorted(buckets)
    offset = 0
    while True:
        added = False
        for bucket_name in bucket_names:
            bucket_candidates = buckets[bucket_name]
            if offset < len(bucket_candidates):
                ordered.append(bucket_candidates[offset])
                added = True
        if not added:
            return ordered
        offset += 1


def _quota_flow(
    *,
    family_targets: tuple[tuple[str, int], ...],
    angle_targets: tuple[tuple[str, int], ...],
    pair_capacity: dict[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Solve the bounded family-by-angle contingency table as max flow."""

    source = ("source", "")
    sink = ("sink", "")
    residual: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def add_edge(
        left: tuple[str, str], right: tuple[str, str], capacity: int
    ) -> None:
        residual[(left, right)] = capacity
        residual[(right, left)] = 0
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    for family, target in family_targets:
        add_edge(source, ("family", family), target)
    for family, _target in family_targets:
        for angle, _angle_target in angle_targets:
            add_edge(
                ("family", family),
                ("angle", angle),
                pair_capacity.get((family, angle), 0),
            )
    for angle, target in angle_targets:
        add_edge(("angle", angle), sink, target)

    total_flow = 0
    while True:
        parent: dict[tuple[str, str], tuple[str, str] | None] = {
            source: None
        }
        queue = deque((source,))
        while queue and sink not in parent:
            left = queue.popleft()
            for right in adjacency.get(left, ()):
                if right not in parent and residual[(left, right)] > 0:
                    parent[right] = left
                    queue.append(right)
        if sink not in parent:
            break
        increment = sum(target for _name, target in family_targets)
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            increment = min(increment, residual[(previous, node)])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[(previous, node)] -= increment
            residual[(node, previous)] += increment
            node = previous
        total_flow += increment

    expected = sum(target for _name, target in family_targets)
    if total_flow != expected:
        availability = {
            f"{family}/{angle}": pair_capacity.get((family, angle), 0)
            for family, _target in family_targets
            for angle, _angle_target in angle_targets
        }
        raise ValueError(
            "candidate pool cannot satisfy staged family/angle quotas; "
            f"flow={total_flow}/{expected}, availability={availability}"
        )
    return {
        (family, angle): pair_capacity.get((family, angle), 0)
        - residual[
            (("family", family), ("angle", angle))
        ]
        for family, _target in family_targets
        for angle, _angle_target in angle_targets
    }


def select_staged_subset(
    candidates: Iterable[GrailSelectionCandidate],
    *,
    config: StagedSelectionConfig,
) -> tuple[GrailSelectionCandidate, ...]:
    """Select exact family and heading marginals with deterministic diversity."""

    if not isinstance(config, StagedSelectionConfig):
        raise TypeError("config must be a StagedSelectionConfig")
    materialized = tuple(candidates)
    stems = [candidate.record.stem for candidate in materialized]
    if len(stems) != len(set(stems)):
        raise ValueError("selection candidates must have unique stems")
    for candidate in materialized:
        if candidate.record.stem != candidate.geometry.stem:
            raise ValueError(
                f"record/geometry stem mismatch: {candidate.record.stem}"
            )

    traversal_order = tuple(
        dict.fromkeys(mode for mode, _family, _count in config.family_targets)
    )
    selected: list[GrailSelectionCandidate] = []
    for traversal in traversal_order:
        family_targets = tuple(
            (family, count)
            for mode, family, count in config.family_targets
            if mode == traversal
        )
        angle_targets = tuple(
            (angle, count)
            for mode, angle, count in config.angle_targets
            if mode == traversal
        )
        if sum(count for _family, count in family_targets) != sum(
            count for _angle, count in angle_targets
        ):
            raise ValueError(
                f"{traversal}: family and angle quota totals differ"
            )
        pairs: dict[
            tuple[str, str], list[GrailSelectionCandidate]
        ] = {}
        allowed_families = {family for family, _count in family_targets}
        allowed_angles = {angle for angle, _count in angle_targets}
        for candidate in materialized:
            angle = candidate.approach_angle_bin
            if (
                candidate.traversal == traversal
                and candidate.record.family in allowed_families
                and angle in allowed_angles
            ):
                pairs.setdefault(
                    (candidate.record.family, angle), []
                ).append(candidate)
        flow = _quota_flow(
            family_targets=family_targets,
            angle_targets=angle_targets,
            pair_capacity={
                pair: len(pair_candidates)
                for pair, pair_candidates in pairs.items()
            },
        )
        for family, _family_count in family_targets:
            for angle, _angle_count in angle_targets:
                count = flow[(family, angle)]
                if count:
                    ordered = _seeded_candidate_order(
                        pairs[(family, angle)],
                        seed=(
                            f"{config.seed}\0{traversal}\0{family}\0{angle}"
                        ),
                    )
                    selected.extend(ordered[:count])
    return tuple(selected)


def parse_grail_motion_blob(
    blob: object,
    *,
    expected_frames: int | None = None,
) -> RawGrailMotion:
    """Validate one joblib payload without assuming its unrelated internal key."""

    if not isinstance(blob, Mapping) or len(blob) != 1:
        raise ValueError("GRAIL robot pickle must contain exactly one motion")
    source_key, payload = next(iter(blob.items()))
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("GRAIL motion internal key must be a nonempty string")
    if not isinstance(payload, Mapping):
        raise ValueError("GRAIL motion payload must be a mapping")
    missing = sorted(
        {"root_trans_offset", "root_rot", "dof", "fps"} - set(payload)
    )
    if missing:
        raise ValueError(
            f"GRAIL motion payload is missing fields: {', '.join(missing)}"
        )
    root = _finite_array(
        payload["root_trans_offset"],
        shape_tail=(3,),
        label="GRAIL root_trans_offset",
    )
    quaternion = _finite_array(
        payload["root_rot"],
        shape_tail=(4,),
        label="GRAIL root_rot XYZW",
    )
    joints = _finite_array(
        payload["dof"],
        shape_tail=(29,),
        label="GRAIL dof",
    )
    frames = len(root)
    if frames < 2 or len(quaternion) != frames or len(joints) != frames:
        raise ValueError("GRAIL motion arrays must have the same T >= 2")
    if expected_frames is not None and (
        type(expected_frames) is not int or frames != expected_frames
    ):
        raise ValueError(
            f"GRAIL motion frame count is {frames}, expected {expected_frames}"
        )
    fps_array = np.asarray(payload["fps"])
    if (
        fps_array.size != 1
        or fps_array.dtype.kind not in "iuf"
        or not np.isfinite(fps_array).all()
        or float(fps_array.reshape(-1)[0]) != 25.0
    ):
        raise ValueError("GRAIL motion fps must be exactly 25")
    quaternion_norm = np.linalg.norm(quaternion, axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("GRAIL root_rot must contain normalized XYZW quaternions")
    return RawGrailMotion(
        root_position=np.ascontiguousarray(root, dtype=np.float32),
        root_quaternion_xyzw=np.ascontiguousarray(
            quaternion, dtype=np.float32
        ),
        dof_mujoco=np.ascontiguousarray(joints, dtype=np.float32),
        fps=25.0,
        source_key=source_key,
    )


def load_grail_motion(
    path: str | Path,
    *,
    expected_frames: int | None = None,
) -> RawGrailMotion:
    """Load and validate one released joblib robot pickle."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("joblib is required to read GRAIL robot pickles") from error
    try:
        blob = joblib.load(source)
    except Exception as error:
        raise ValueError(f"could not load GRAIL robot pickle: {source}") from error
    try:
        return parse_grail_motion_blob(blob, expected_frames=expected_frames)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error


def load_grail_geometry(
    path: str | Path,
) -> dict[str, GrailGeometry]:
    """Read the authoritative corpus geometry table keyed by filename stem."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    required = {
        "seq_name",
        "fps",
        "n_frames",
        "n_steps",
        "rise",
        "tread",
        "travel_yaw",
        "ascending",
        "dz",
        "ok",
    }
    result: dict[str, GrailGeometry] = {}
    try:
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(
                    f"geometry CSV is missing columns: {', '.join(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                if row.get("ok") != "True":
                    continue
                stem = row.get("seq_name", "")
                if not stem:
                    raise ValueError(
                        f"geometry CSV row {row_number} has an empty seq_name"
                    )
                if stem in result:
                    raise ValueError(f"duplicate geometry stem: {stem}")
                try:
                    fps = float(row["fps"])
                    n_frames = int(row["n_frames"])
                    n_steps = int(row["n_steps"])
                    rise = float(row["rise"])
                    tread = float(row["tread"])
                    travel_yaw = float(row["travel_yaw"])
                    root_delta = float(row["dz"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"geometry CSV row {row_number} has invalid numerics"
                    ) from error
                ascending_text = row.get("ascending")
                if ascending_text not in {"True", "False"}:
                    raise ValueError(
                        f"geometry CSV row {row_number} has invalid ascending"
                    )
                numeric = (fps, rise, tread, travel_yaw, root_delta)
                if (
                    not all(np.isfinite(value) for value in numeric)
                    or fps != 25.0
                    or n_frames < 2
                    or n_steps <= 0
                    or rise <= 0.0
                    or tread <= 0.0
                ):
                    raise ValueError(
                        f"geometry CSV row {row_number} violates the stair contract"
                    )
                result[stem] = GrailGeometry(
                    stem=stem,
                    fps=fps,
                    n_frames=n_frames,
                    n_steps=n_steps,
                    rise_m=rise,
                    tread_m=tread,
                    travel_yaw_rad=travel_yaw,
                    ascending=ascending_text == "True",
                    root_delta_z_m=root_delta,
                )
    except OSError as error:
        raise ValueError(f"could not read geometry CSV: {source}") from error
    if not result:
        raise ValueError("geometry CSV contains no valid stair records")
    return result


def _yaw_xyzw(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = np.moveaxis(value, -1, 0)
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def approach_heading_delta_degrees(
    root_quaternion_xyzw: object,
    *,
    travel_yaw_rad: float,
    initial_frames: int = 25,
) -> float:
    """Measure initial robot facing relative to the corpus travel direction."""

    quaternion = _finite_array(
        root_quaternion_xyzw,
        shape_tail=(4,),
        label="root_quaternion_xyzw",
    )
    if len(quaternion) < 2:
        raise ValueError("approach heading needs at least two frames")
    count = min(len(quaternion), int(initial_frames))
    if count <= 0:
        raise ValueError("initial_frames must be positive")
    travel_yaw = float(travel_yaw_rad)
    if not np.isfinite(travel_yaw):
        raise ValueError("travel_yaw_rad must be finite")
    yaw = _yaw_xyzw(quaternion[:count])
    mean_yaw = float(
        np.arctan2(np.mean(np.sin(yaw)), np.mean(np.cos(yaw)))
    )
    delta = np.arctan2(
        np.sin(mean_yaw - travel_yaw),
        np.cos(mean_yaw - travel_yaw),
    )
    return float(np.rad2deg(delta))


def candidate_from_motion(
    record: GrailClipRecord,
    geometry: GrailGeometry,
    motion: RawGrailMotion,
) -> GrailSelectionCandidate:
    """Measure traversal and initial heading for one joined source record."""

    if record.stem != geometry.stem:
        raise ValueError(f"record/geometry stem mismatch: {record.stem}")
    if (
        record.n_frames != geometry.n_frames
        or record.n_frames != len(motion.root_position)
        or geometry.fps != motion.fps
    ):
        raise ValueError(f"{record.stem}: joined source frame/fps mismatch")
    if len(motion.root_position) >= 20:
        traversal = classify_traversal(motion.root_position[:, 2])
    elif "_updown__" in record.stem:
        traversal = "roundtrip"
    elif abs(geometry.root_delta_z_m) < 0.5:
        traversal = "partial"
    else:
        traversal = "up" if geometry.root_delta_z_m > 0.0 else "down"
    return GrailSelectionCandidate(
        record=record,
        geometry=geometry,
        traversal=traversal,
        approach_heading_delta_deg=approach_heading_delta_degrees(
            motion.root_quaternion_xyzw,
            travel_yaw_rad=geometry.travel_yaw_rad,
        ),
    )


def build_archive_clip(
    candidate: GrailSelectionCandidate,
    motion: RawGrailMotion,
    *,
    fk: G1MujocoFK,
) -> GrailArchiveClip:
    """Resample, run FK, reorder joints, then derive one clip's velocities."""

    if len(motion.root_position) != candidate.record.n_frames:
        raise ValueError(
            f"{candidate.record.stem}: selected motion frame count changed"
        )
    resampled = resample_grail_motion(
        motion.root_position,
        motion.root_quaternion_xyzw,
        motion.dof_mujoco,
        source_fps=motion.fps,
        target_fps=50.0,
    )
    kinematics = fk.forward(
        resampled.root_position,
        resampled.root_quaternion_xyzw,
        resampled.dof_mujoco,
    )
    if tuple(kinematics.body_names) != ISAACLAB_BODY_NAMES:
        raise ValueError("FK result does not use the IsaacLab body order")
    joint_position = mujoco_to_isaaclab_joints(resampled.dof_mujoco)
    return GrailArchiveClip(
        candidate=candidate,
        joint_position=joint_position,
        joint_velocity=derive_clip_velocity(
            joint_position, fps=resampled.fps
        ),
        body_position_world=kinematics.body_position_world,
        body_quaternion_world_xyzw=(
            kinematics.body_quaternion_world_xyzw
        ),
        body_linear_velocity_world=derive_clip_velocity(
            kinematics.body_position_world, fps=resampled.fps
        ),
        fps=resampled.fps,
    )


def discover_grail_clips(
    shard_root: str | Path,
    *,
    families: Iterable[str] = DEFAULT_GRAIL_FAMILIES,
    expected_family_counts: Mapping[str, int] | None = None,
) -> tuple[GrailClipRecord, ...]:
    """Strictly join shard metadata, individual robot pickle, and terrain USD."""

    root = Path(shard_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    requested = tuple(str(family) for family in families)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("families must be a nonempty unique sequence")
    records: list[GrailClipRecord] = []
    seen_stems: set[str] = set()
    for family in requested:
        shard_paths = sorted(
            path
            for path in root.glob(f"{family}_*")
            if path.is_dir() and (path / "clips.json").is_file()
        )
        if not shard_paths:
            raise ValueError(f"no committed clips.json shards found for {family}")
        for shard in shard_paths:
            try:
                raw_records = json.loads((shard / "clips.json").read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid shard metadata: {shard / 'clips.json'}"
                ) from error
            if not isinstance(raw_records, list):
                raise ValueError(f"{shard / 'clips.json'} must contain a list")
            for raw in raw_records:
                if not isinstance(raw, dict):
                    raise ValueError("clips.json records must be objects")
                stem = raw.get("stem")
                if not isinstance(stem, str) or not stem:
                    raise ValueError("clips.json stem must be a nonempty string")
                if stem in seen_stems:
                    raise ValueError(f"duplicate GRAIL stem across shards: {stem}")
                robot_path = shard / "robot" / f"{stem}.pkl"
                usd_path = shard / "object_usd" / f"{stem}.usd"
                if not robot_path.is_file():
                    raise ValueError(f"{stem}: missing paired robot pickle")
                if not usd_path.is_file():
                    raise ValueError(f"{stem}: missing paired USD")
                n_frames = raw.get("n_frames")
                if type(n_frames) is not int or n_frames < 2:
                    raise ValueError(f"{stem}: n_frames must be an integer >= 2")
                terrain = raw.get("terrain")
                if not isinstance(terrain, dict):
                    raise ValueError(f"{stem}: terrain pose must be an object")
                position = np.asarray(
                    terrain.get("position_env"), dtype=np.float64
                )
                rotation = np.asarray(
                    terrain.get("rotation_env_wxyz"), dtype=np.float64
                )
                if (
                    position.shape != (3,)
                    or rotation.shape != (4,)
                    or not np.isfinite(position).all()
                    or not np.isfinite(rotation).all()
                ):
                    raise ValueError(
                        f"{stem}: terrain pose must contain finite position_env[3] "
                        "and rotation_env_wxyz[4]"
                    )
                norm = float(np.linalg.norm(rotation))
                if abs(norm - 1.0) > 1.0e-4:
                    raise ValueError(
                        f"{stem}: terrain WXYZ quaternion is not normalized"
                    )
                seen_stems.add(stem)
                records.append(
                    GrailClipRecord(
                        family=family,
                        shard_path=shard,
                        stem=stem,
                        robot_path=robot_path,
                        usd_path=usd_path,
                        n_frames=n_frames,
                        terrain_position_env=np.ascontiguousarray(
                            position, dtype=np.float32
                        ),
                        terrain_rotation_env_wxyz=np.ascontiguousarray(
                            rotation, dtype=np.float32
                        ),
                        pose_source=str(raw.get("pose_source") or "shard_legacy"),
                    )
                )
    if expected_family_counts is not None:
        expected = dict(expected_family_counts)
        if set(expected) != set(requested) or any(
            type(count) is not int or count <= 0
            for count in expected.values()
        ):
            raise ValueError(
                "expected_family_counts must contain one positive integer "
                "for every requested family"
            )
        actual = Counter(record.family for record in records)
        if actual != Counter(expected):
            raise ValueError(
                "GRAIL inventory count mismatch; "
                f"expected={expected}, actual={dict(actual)}"
            )
    return tuple(records)


class G1MujocoFK:
    """Name-validated G1 forward kinematics in terrain archive body order."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "mujoco is required for GRAIL forward kinematics"
            ) from error

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self._mujoco = mujoco
        self.model_path = path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        if self.model.nq != 36 or self.model.nbody != 31:
            raise ValueError(
                "GRAIL G1 model must have nq=36 and 30 robot bodies"
            )
        mujoco_body_names = tuple(
            str(self.model.body(index).name)
            for index in range(1, self.model.nbody)
        )
        self.body_permutation = body_permutation_to_isaaclab(
            mujoco_body_names
        )
        self._body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )
                for name in ISAACLAB_BODY_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self._body_ids <= 0):
            raise ValueError("G1 model is missing an expected robot body")

        actual_joint_names = tuple(
            str(self.model.joint(index).name)
            for index in range(self.model.njnt)
            if self.model.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE
        )
        isaaclab_to_mujoco = np.argsort(MUJOCO_TO_ISAACLAB_JOINT)
        expected_joint_names = tuple(
            ISAACLAB_JOINT_NAMES[index] for index in isaaclab_to_mujoco
        )
        if actual_joint_names != expected_joint_names:
            raise ValueError(
                "G1 model actuated-joint order does not match raw GRAIL dof order"
            )

    def forward(
        self,
        root_position: object,
        root_quaternion_xyzw: object,
        dof_mujoco: object,
    ) -> GrailForwardKinematics:
        root = _finite_array(
            root_position, shape_tail=(3,), label="root_position"
        )
        quaternion = _finite_array(
            root_quaternion_xyzw,
            shape_tail=(4,),
            label="root_quaternion_xyzw",
        )
        joints = _finite_array(
            dof_mujoco, shape_tail=(29,), label="dof_mujoco"
        )
        if len(root) == 0 or len(quaternion) != len(root) or len(joints) != len(root):
            raise ValueError("FK arrays must have the same nonzero frame count")
        norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norms < 1.0e-12):
            raise ValueError("root quaternions must be nonzero")
        quaternion = quaternion / norms
        root_wxyz = xyzw_to_wxyz(quaternion)

        data = self._mujoco.MjData(self.model)
        positions = np.empty((len(root), 30, 3), dtype=np.float64)
        quaternions_wxyz = np.empty((len(root), 30, 4), dtype=np.float64)
        for frame in range(len(root)):
            data.qpos[:3] = root[frame]
            data.qpos[3:7] = root_wxyz[frame]
            data.qpos[7:] = joints[frame]
            self._mujoco.mj_forward(self.model, data)
            positions[frame] = data.xpos[self._body_ids]
            quaternions_wxyz[frame] = data.xquat[self._body_ids]
        quaternions_xyzw = wxyz_to_xyzw(quaternions_wxyz)
        quaternion_norms = np.linalg.norm(
            quaternions_xyzw, axis=-1, keepdims=True
        )
        if np.any(quaternion_norms < 1.0e-12):
            raise ValueError("MuJoCo FK returned a zero body quaternion")
        quaternions_xyzw = quaternions_xyzw / quaternion_norms
        return GrailForwardKinematics(
            body_position_world=np.ascontiguousarray(
                positions, dtype=np.float32
            ),
            body_quaternion_world_xyzw=np.ascontiguousarray(
                quaternions_xyzw, dtype=np.float32
            ),
            body_names=ISAACLAB_BODY_NAMES,
        )


def xyzw_to_wxyz(value: object) -> np.ndarray:
    """Reorder explicit XYZW quaternion arrays for the MuJoCo qpos boundary."""

    array = np.asarray(value)
    if (
        array.ndim < 1
        or array.shape[-1] != 4
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError("XYZW quaternions must end in four finite values")
    return np.ascontiguousarray(array[..., (3, 0, 1, 2)])


def wxyz_to_xyzw(value: object) -> np.ndarray:
    """Reorder MuJoCo WXYZ quaternion arrays into the terrain archive contract."""

    array = np.asarray(value)
    if (
        array.ndim < 1
        or array.shape[-1] != 4
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError("WXYZ quaternions must end in four finite values")
    return np.ascontiguousarray(array[..., (1, 2, 3, 0)])


def mujoco_to_isaaclab_joints(value: object) -> np.ndarray:
    """Map ``[...,29]`` MuJoCo joint arrays to IsaacLab breadth-first order."""

    array = np.asarray(value)
    if (
        array.ndim < 1
        or array.shape[-1] != 29
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError("MuJoCo joints must end in 29 finite values")
    return np.ascontiguousarray(array[..., MUJOCO_TO_ISAACLAB_JOINT])


def body_permutation_to_isaaclab(
    mujoco_body_names: object,
) -> np.ndarray:
    """Return indices selecting MuJoCo body rows into IsaacLab body order."""

    names = tuple(str(name) for name in mujoco_body_names)  # type: ignore[arg-type]
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError("MuJoCo body names must contain 30 unique robot bodies")
    missing = sorted(set(ISAACLAB_BODY_NAMES) - set(names))
    extra = sorted(set(names) - set(ISAACLAB_BODY_NAMES))
    if missing or extra:
        raise ValueError(
            f"MuJoCo/IsaacLab body-name mismatch; missing={missing}, extra={extra}"
        )
    return np.asarray([names.index(name) for name in ISAACLAB_BODY_NAMES], np.int64)


def mujoco_to_isaaclab_bodies(
    value: object,
    *,
    mujoco_body_names: object,
) -> np.ndarray:
    """Map ``[T,30,...]`` MuJoCo body arrays to IsaacLab body order."""

    array = np.asarray(value)
    if (
        array.ndim < 2
        or array.shape[1] != 30
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError("MuJoCo bodies must be finite arrays with shape [T,30,...]")
    permutation = body_permutation_to_isaaclab(mujoco_body_names)
    return np.ascontiguousarray(np.take(array, permutation, axis=1))


def derive_clip_velocity(value: object, *, fps: float) -> np.ndarray:
    """Differentiate one clip without leaking across concatenated clip seams."""

    array = np.asarray(value, dtype=np.float64)
    rate = float(fps)
    if (
        array.ndim < 1
        or len(array) < 3
        or not np.isfinite(array).all()
        or not np.isfinite(rate)
        or rate <= 0.0
    ):
        raise ValueError(
            "clip derivative needs at least three finite rows and positive fps"
        )
    velocity = np.gradient(array, axis=0, edge_order=2) * rate
    return np.ascontiguousarray(velocity, dtype=np.float32)


def derive_clip_angular_velocity(
    quaternion_xyzw: object,
    *,
    fps: float,
) -> np.ndarray:
    """Differentiate one XYZW quaternion clip in the world frame."""

    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    rate = float(fps)
    if (
        quaternion.ndim < 2
        or quaternion.shape[-1] != 4
        or len(quaternion) < 3
        or not np.isfinite(quaternion).all()
        or not np.isfinite(rate)
        or rate <= 0.0
    ):
        raise ValueError(
            "clip angular derivative needs at least three finite XYZW rows "
            "and positive fps"
        )
    norms = np.linalg.norm(quaternion, axis=-1)
    if np.any(norms < 1.0e-8) or np.max(np.abs(norms - 1.0)) > 1.0e-3:
        raise ValueError("clip quaternions must be normalized")
    flattened = quaternion.reshape(len(quaternion), -1, 4)
    current = Rotation.from_quat(flattened[1:].reshape(-1, 4))
    previous = Rotation.from_quat(flattened[:-1].reshape(-1, 4))
    frame_delta = (current * previous.inv()).as_rotvec().reshape(
        len(quaternion) - 1,
        *quaternion.shape[1:-1],
        3,
    ) * rate
    angular = np.empty((*quaternion.shape[:-1], 3), dtype=np.float64)
    angular[0] = frame_delta[0]
    angular[-1] = frame_delta[-1]
    angular[1:-1] = 0.5 * (frame_delta[:-1] + frame_delta[1:])
    return np.ascontiguousarray(angular, dtype=np.float32)


def _finite_array(
    value: object,
    *,
    shape_tail: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != len(shape_tail) + 1
        or array.shape[1:] != shape_tail
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{label} must have shape [T,{','.join(map(str, shape_tail))}] "
            "and contain finite values"
        )
    return array


def _slerp_midpoint_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    norm0 = np.linalg.norm(q0, axis=1, keepdims=True)
    norm1 = np.linalg.norm(q1, axis=1, keepdims=True)
    if np.any(norm0 < 1.0e-12) or np.any(norm1 < 1.0e-12):
        raise ValueError("root quaternions must be nonzero")
    q0 = q0 / norm0
    q1 = q1 / norm1
    q1 = np.where(np.sum(q0 * q1, axis=1, keepdims=True) < 0.0, -q1, q1)
    midpoint = q0 + q1
    midpoint_norm = np.linalg.norm(midpoint, axis=1, keepdims=True)
    if np.any(midpoint_norm < 1.0e-12):
        raise ValueError("root quaternion interpolation is antipodal")
    return midpoint / midpoint_norm


def _slerp_xyzw(
    left: np.ndarray, right: np.ndarray, fraction: np.ndarray
) -> np.ndarray:
    """Shortest-arc interpolation of normalized XYZW quaternion rows."""

    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    norm0 = np.linalg.norm(q0, axis=1, keepdims=True)
    norm1 = np.linalg.norm(q1, axis=1, keepdims=True)
    if np.any(norm0 < 1.0e-12) or np.any(norm1 < 1.0e-12):
        raise ValueError("root quaternions must be nonzero")
    q0 = q0 / norm0
    q1 = q1 / norm1
    q1 = np.where(np.sum(q0 * q1, axis=1, keepdims=True) < 0.0, -q1, q1)
    dot = np.clip(np.sum(q0 * q1, axis=1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    fraction_array = np.asarray(fraction, dtype=np.float64)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        left_weight = np.sin((1.0 - fraction_array) * theta) / sine
        right_weight = np.sin(fraction_array * theta) / sine
    linear = (1.0 - fraction_array) * q0 + fraction_array * q1
    output = np.where(sine < 1.0e-7, linear, left_weight * q0 + right_weight * q1)
    return output / np.linalg.norm(output, axis=1, keepdims=True)


def resample_grail_motion(
    root_position: object,
    root_quaternion_xyzw: object,
    dof_mujoco: object,
    *,
    source_fps: float,
    target_fps: float,
) -> ResampledGrailMotion:
    """Interpolate one GRAIL clip from exactly 25 Hz to 30 or 50 Hz.

    A clip with ``T`` source samples spans ``T - 1`` source intervals, so the
    result contains ``2 * (T - 1) + 1`` rows.  In particular, a released
    250-frame clip becomes 499 rows; no duplicate 9.98-second endpoint is
    invented.
    """

    if float(source_fps) != 25.0 or float(target_fps) not in (30.0, 50.0):
        raise ValueError("GRAIL terrain import supports exactly 25 Hz to 30 or 50 Hz")
    root = _finite_array(
        root_position, shape_tail=(3,), label="root_position"
    )
    quaternion = _finite_array(
        root_quaternion_xyzw,
        shape_tail=(4,),
        label="root_quaternion_xyzw",
    )
    joints = _finite_array(
        dof_mujoco, shape_tail=(29,), label="dof_mujoco"
    )
    if len(root) < 2 or len(quaternion) != len(root) or len(joints) != len(root):
        raise ValueError("GRAIL motion arrays must have the same T >= 2")

    if float(target_fps) == 30.0:
        count = round(len(root) * 30 / 25)
        target_coordinates = np.arange(count, dtype=np.float64) * (25.0 / 30.0)
        left = np.minimum(np.floor(target_coordinates).astype(np.int64), len(root) - 1)
        right = np.minimum(left + 1, len(root) - 1)
        fraction = np.clip(target_coordinates - left, 0.0, 1.0)
        output_root = (1.0 - fraction[:, None]) * root[left] + fraction[:, None] * root[right]
        output_joints = (1.0 - fraction[:, None]) * joints[left] + fraction[:, None] * joints[right]
        output_quaternion = _slerp_xyzw(quaternion[left], quaternion[right], fraction)
        return ResampledGrailMotion(
            root_position=np.ascontiguousarray(output_root, dtype=np.float32),
            root_quaternion_xyzw=np.ascontiguousarray(output_quaternion, dtype=np.float32),
            dof_mujoco=np.ascontiguousarray(output_joints, dtype=np.float32),
            fps=30.0,
        )

    count = 2 * (len(root) - 1) + 1
    output_root = np.empty((count, 3), dtype=np.float64)
    output_quaternion = np.empty((count, 4), dtype=np.float64)
    output_joints = np.empty((count, 29), dtype=np.float64)
    output_root[0::2] = root
    output_root[1::2] = 0.5 * (root[:-1] + root[1:])
    output_joints[0::2] = joints
    output_joints[1::2] = 0.5 * (joints[:-1] + joints[1:])

    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norms < 1.0e-12):
        raise ValueError("root quaternions must be nonzero")
    normalized = quaternion / norms
    aligned = normalized.copy()
    for index in range(1, len(aligned)):
        if float(np.dot(aligned[index - 1], aligned[index])) < 0.0:
            aligned[index] *= -1.0
    output_quaternion[0::2] = aligned
    output_quaternion[1::2] = _slerp_midpoint_xyzw(
        aligned[:-1], aligned[1:]
    )
    return ResampledGrailMotion(
        root_position=np.ascontiguousarray(output_root, dtype=np.float32),
        root_quaternion_xyzw=np.ascontiguousarray(
            output_quaternion, dtype=np.float32
        ),
        dof_mujoco=np.ascontiguousarray(output_joints, dtype=np.float32),
        fps=50.0,
    )
