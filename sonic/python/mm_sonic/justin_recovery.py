"""Offline matching contract for the Justin S13 tracker-recovery experiment.

This module intentionally stops at a recovery *proposal*.  It consumes a
passive full-state MuJoCo trace, finds reference frames in the exact Justin
motion family used by the SONIC tracker, and emits enough provenance for a
separate Isaac process to test the handoff.  No learner action or recovered
trajectory is added to a training dataset here.

The learner and reference scenes contain the same 13-by-7-inch three-step
asset but use different world frames.  The fixed planar transform below maps
the MuJoCo evaluation scene (front edge at ``[3, 0]``, ascent along +X) into
the original Justin collection scene (front edge at ``[-0.21065, -4.3529]``,
ascent along -X).  Applying this rigid transform preserves the learner's
actual forward/lateral error; matching must never snap its root to a clean
reference pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CONTROL_HZ = 50
HISTORY_FRAMES = 20
DEFAULT_REWINDS_S = (0.25, 0.5, 0.75, 1.0)
DEFAULT_TOP_K = 16
MIN_REMAINING_FRAMES = 50
FAILURE_PELVIS_HEIGHT_M = 0.55
FAILURE_SUSTAIN_FRAMES = 10

# The trace drops MuJoCo's world body via ``data.xpos[1:31]``.  These are
# therefore robot-array indices, one below the absolute MuJoCo body ids used
# by the evaluator's live metrics (left/right ankle 7/13).
LEARNER_LEFT_FOOT_BODY = 6
LEARNER_RIGHT_FOOT_BODY = 12
LEARNER_ROOT_BODY = 0
REFERENCE_ROOT_BODY = 0
REFERENCE_LEFT_FOOT_BODY = 18
REFERENCE_RIGHT_FOOT_BODY = 19

S13_TREAD_M = 0.3302
S13_LEARNER_FRONT_XY = np.asarray((3.0, 0.0), dtype=np.float64)
S13_REFERENCE_FRONT_XY = np.asarray((-0.21065, -4.3529), dtype=np.float64)
S13_SCENE_YAW_DELTA_RAD = math.pi

REQUIRED_TRACE_FIELDS = {
    "physics_step": (),
    "policy_call_index": (),
    "qpos_mujoco": (36,),
    "qvel_mujoco": (35,),
    "joint_pos_isaac": (29,),
    "joint_vel_isaac": (29,),
    "body_pos_mujoco": (30, 3),
    "body_quat_wxyz_mujoco": (30, 4),
    "body_lin_vel_world_mujoco": (30, 3),
    "body_ang_vel_world_mujoco": (30, 3),
    "left_foot_contact": (),
    "right_foot_contact": (),
    "forbidden_nonfoot_contact": (),
    "minimum_contact_distance_m": (),
}

REFERENCE_FIELDS = {
    "fps": None,
    "joint_pos": (29,),
    "joint_vel": (29,),
    "body_pos_w": (30, 3),
    "body_quat_w": (30, 4),
    "body_lin_vel_w": (30, 3),
    "body_ang_vel_w": (30, 3),
}


class RecoveryContractError(ValueError):
    """The trace, bank, or proposed handoff violates the frozen contract."""


@dataclass(frozen=True)
class RecoveryTrace:
    path: Path
    arrays: Mapping[str, np.ndarray]
    sha256: str

    @property
    def frame_count(self) -> int:
        return int(self.arrays["physics_step"].shape[0])


@dataclass(frozen=True)
class ReferenceClip:
    path: Path
    name: str
    arrays: Mapping[str, np.ndarray]
    sha256: str

    @property
    def frame_count(self) -> int:
        return int(self.arrays["joint_pos"].shape[0])


@dataclass(frozen=True)
class RecoveryQuery:
    rewind_s: float
    trace_index: int
    physics_step: int
    history_start: int


@dataclass(frozen=True)
class RecoveryCandidate:
    clip_name: str
    clip_path: str
    clip_sha256: str
    frame: int
    total_cost: float
    pose_cost: float
    velocity_cost: float
    foot_cost: float
    stage_cost: float
    support: str
    stage: int
    frames_remaining: int

    def to_json(self) -> dict[str, object]:
        return {
            "clip_name": self.clip_name,
            "clip_path": self.clip_path,
            "clip_sha256": self.clip_sha256,
            "frame": self.frame,
            "total_cost": self.total_cost,
            "pose_cost": self.pose_cost,
            "velocity_cost": self.velocity_cost,
            "foot_cost": self.foot_cost,
            "stage_cost": self.stage_cost,
            "support": self.support,
            "stage": self.stage,
            "frames_remaining": self.frames_remaining,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_numeric_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = {name: np.asarray(archive[name]) for name in archive.files}
    except Exception as error:
        raise RecoveryContractError(f"cannot load {path}: {type(error).__name__}") from error
    return result


def load_recovery_trace(path: str | Path) -> RecoveryTrace:
    source = Path(path).resolve()
    arrays = _load_numeric_npz(source)
    lengths: set[int] = set()
    for name, tail in REQUIRED_TRACE_FIELDS.items():
        if name not in arrays:
            raise RecoveryContractError(f"trace is missing {name}")
        value = arrays[name]
        if value.ndim < 1 or tuple(value.shape[1:]) != tail:
            raise RecoveryContractError(
                f"trace {name} has shape {value.shape}, expected [T]{tail}"
            )
        lengths.add(int(value.shape[0]))
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise RecoveryContractError(f"trace {name} contains non-finite values")
    if len(lengths) != 1:
        raise RecoveryContractError(f"trace fields have different lengths: {sorted(lengths)}")
    steps = arrays["physics_step"].astype(np.int64)
    calls = arrays["policy_call_index"].astype(np.int64)
    if len(steps) < HISTORY_FRAMES:
        raise RecoveryContractError(
            f"trace has {len(steps)} rows, needs at least {HISTORY_FRAMES}"
        )
    if not np.all(np.diff(steps) == 4):
        raise RecoveryContractError("trace is not on the exact 50 Hz / four-physics-step cadence")
    if not np.array_equal(calls, np.arange(len(calls))):
        raise RecoveryContractError("trace policy-call indices are not contiguous")
    return RecoveryTrace(source, arrays, _sha256(source))


def load_reference_clip(path: str | Path) -> ReferenceClip:
    source = Path(path).resolve()
    arrays = _load_numeric_npz(source)
    lengths: set[int] = set()
    for name, tail in REFERENCE_FIELDS.items():
        if name not in arrays:
            raise RecoveryContractError(f"{source.name} is missing {name}")
        value = arrays[name]
        if name == "fps":
            if value.size != 1 or float(value.reshape(-1)[0]) != CONTROL_HZ:
                raise RecoveryContractError(f"{source.name} fps must equal {CONTROL_HZ}")
            continue
        if value.ndim < 1 or tuple(value.shape[1:]) != tail:
            raise RecoveryContractError(
                f"{source.name} {name} has shape {value.shape}, expected [T]{tail}"
            )
        lengths.add(int(value.shape[0]))
        if not np.isfinite(value).all():
            raise RecoveryContractError(f"{source.name} {name} contains non-finite values")
    if len(lengths) != 1:
        raise RecoveryContractError(f"{source.name} fields have different lengths")
    count = next(iter(lengths))
    if count < HISTORY_FRAMES + MIN_REMAINING_FRAMES:
        raise RecoveryContractError(f"{source.name} is too short for recovery matching")
    return ReferenceClip(source, source.stem, arrays, _sha256(source))


def load_reference_bank(paths: Iterable[str | Path]) -> tuple[ReferenceClip, ...]:
    clips = tuple(load_reference_clip(path) for path in paths)
    if not 4 <= len(clips) <= 8:
        raise RecoveryContractError(
            f"Justin recovery bank must contain 4..8 clips, got {len(clips)}"
        )
    names = [clip.name for clip in clips]
    if len(set(names)) != len(names):
        raise RecoveryContractError("Justin recovery bank clip names are not unique")
    return clips


def select_rewind_queries(
    trace: RecoveryTrace,
    *,
    first_fall_physics_step: int,
    rewinds_s: Sequence[float] = DEFAULT_REWINDS_S,
) -> tuple[RecoveryQuery, ...]:
    """Select safe control rows before the observed MuJoCo fall step."""
    steps = trace.arrays["physics_step"].astype(np.int64)
    failure_anchor = recovery_failure_anchor_physics_step(
        trace, first_fall_physics_step=first_fall_physics_step
    )
    result: list[RecoveryQuery] = []
    for rewind in rewinds_s:
        rewind_value = float(rewind)
        if not math.isfinite(rewind_value) or rewind_value <= 0.0:
            raise RecoveryContractError(f"invalid rewind {rewind!r}")
        target = int(failure_anchor - round(rewind_value / 0.005))
        index = int(np.searchsorted(steps, target, side="right") - 1)
        if index < HISTORY_FRAMES - 1:
            continue
        penetration = float(trace.arrays["minimum_contact_distance_m"][index])
        pelvis_height = float(trace.arrays["qpos_mujoco"][index, 2])
        if penetration < -0.03 or pelvis_height < FAILURE_PELVIS_HEIGHT_M:
            continue
        result.append(
            RecoveryQuery(
                rewind_s=rewind_value,
                trace_index=index,
                physics_step=int(steps[index]),
                history_start=index - HISTORY_FRAMES + 1,
            )
        )
    if not result:
        raise RecoveryContractError("no safe pre-failure rewind has a complete history")
    return tuple(result)


def recovery_failure_anchor_physics_step(
    trace: RecoveryTrace, *, first_fall_physics_step: int
) -> int:
    """Return the onset of the sustained collapse, not its delayed confirmation."""
    steps = trace.arrays["physics_step"].astype(np.int64)
    fall_index = int(np.searchsorted(steps, int(first_fall_physics_step), side="right") - 1)
    if fall_index < 0:
        raise RecoveryContractError("reported fall precedes the recovery trace")
    low = np.asarray(
        trace.arrays["qpos_mujoco"][: fall_index + 1, 2] < FAILURE_PELVIS_HEIGHT_M,
        dtype=np.int64,
    )
    if len(low) >= FAILURE_SUSTAIN_FRAMES:
        sustained = np.convolve(
            low, np.ones(FAILURE_SUSTAIN_FRAMES, dtype=np.int64), mode="valid"
        )
        starts = np.flatnonzero(sustained == FAILURE_SUSTAIN_FRAMES)
        if len(starts):
            return int(steps[int(starts[0])])
    return int(first_fall_physics_step)


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_xy(value: np.ndarray, yaw: np.ndarray | float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    angle = np.asarray(yaw, dtype=np.float64)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.stack(
        (c * source[..., 0] - s * source[..., 1],
         s * source[..., 0] + c * source[..., 1]),
        axis=-1,
    )


def learner_xy_to_reference(xy: np.ndarray) -> np.ndarray:
    source = np.asarray(xy, dtype=np.float64)
    return S13_REFERENCE_FRONT_XY + _rotate_xy(
        source - S13_LEARNER_FRONT_XY, S13_SCENE_YAW_DELTA_RAD
    )


def learner_state_in_reference_frame(
    trace: RecoveryTrace, query: RecoveryQuery
) -> dict[str, np.ndarray]:
    """Return exact learner state, changed only by the scene's planar rigid transform."""
    i = query.trace_index
    qpos = np.asarray(trace.arrays["qpos_mujoco"][i], dtype=np.float64).copy()
    qpos[:2] = learner_xy_to_reference(qpos[:2])
    half = 0.5 * S13_SCENE_YAW_DELTA_RAD
    yaw_quat = np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)))
    root_quat = qpos[3:7].copy()
    # Hamilton product: fixed world-yaw rotation * learner root orientation.
    aw, ax, ay, az = yaw_quat
    bw, bx, by, bz = root_quat
    qpos[3:7] = (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    root_body = LEARNER_ROOT_BODY
    root_linear = np.asarray(
        trace.arrays["body_lin_vel_world_mujoco"][i, root_body], dtype=np.float64
    ).copy()
    root_angular = np.asarray(
        trace.arrays["body_ang_vel_world_mujoco"][i, root_body], dtype=np.float64
    ).copy()
    root_linear[:2] = _rotate_xy(root_linear[:2], S13_SCENE_YAW_DELTA_RAD)
    root_angular[:2] = _rotate_xy(root_angular[:2], S13_SCENE_YAW_DELTA_RAD)
    return {
        "root_pose_wxyz": np.concatenate((qpos[:3], qpos[3:7])).astype(np.float32),
        "root_velocity_world": np.concatenate((root_linear, root_angular)).astype(np.float32),
        "joint_pos_isaac": np.asarray(trace.arrays["joint_pos_isaac"][i], np.float32).copy(),
        "joint_vel_isaac": np.asarray(trace.arrays["joint_vel_isaac"][i], np.float32).copy(),
    }


def _heading_local(vector: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    result[..., :2] = _rotate_xy(result[..., :2], -yaw)
    return result


def _support_label(left: bool, right: bool) -> str:
    if left and right:
        return "double"
    if left:
        return "left"
    if right:
        return "right"
    return "flight"


def _reference_support(clip: ReferenceClip) -> np.ndarray:
    body_vel = np.asarray(clip.arrays["body_lin_vel_w"], dtype=np.float64)
    left_speed = np.linalg.norm(body_vel[:, REFERENCE_LEFT_FOOT_BODY], axis=-1)
    right_speed = np.linalg.norm(body_vel[:, REFERENCE_RIGHT_FOOT_BODY], axis=-1)
    left = left_speed < 0.16
    right = right_speed < 0.16
    labels = np.full(clip.frame_count, "flight", dtype="U6")
    labels[left] = "left"
    labels[right] = "right"
    labels[left & right] = "double"
    return labels


def _terrain_stage(progress_m: np.ndarray | float) -> np.ndarray:
    progress = np.asarray(progress_m, dtype=np.float64)
    return np.where(
        progress < 0.0,
        0,
        np.clip(np.floor(progress / S13_TREAD_M).astype(np.int64) + 1, 1, 3),
    ).astype(np.int64)


def _history_parts(
    *,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_vel: np.ndarray,
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    left_vel: np.ndarray,
    right_vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = _yaw_from_wxyz(root_quat)
    feet_rel = np.concatenate((left_pos - root_pos, right_pos - root_pos), axis=-1)
    feet_rel[..., :3] = _heading_local(feet_rel[..., :3], yaw)
    feet_rel[..., 3:] = _heading_local(feet_rel[..., 3:], yaw)
    foot_vel = np.concatenate(
        (_heading_local(left_vel, yaw), _heading_local(right_vel, yaw)), axis=-1
    )
    root_vel_local = _heading_local(root_vel, yaw)
    pose = np.concatenate((joint_pos / 0.75, feet_rel / 0.30), axis=-1)
    velocity = np.concatenate((joint_vel / 4.0, root_vel_local / 1.5), axis=-1)
    return pose, velocity, foot_vel / 1.5


def _query_history(trace: RecoveryTrace, query: RecoveryQuery):
    sl = slice(query.history_start, query.trace_index + 1)
    a = trace.arrays
    # These features are heading-local and translation invariant.  Keeping all
    # learner points in their original frame avoids mixing transformed roots
    # with untransformed MuJoCo body positions.
    root_pos = np.asarray(a["qpos_mujoco"][sl, :3], dtype=np.float64)
    root_quat = np.asarray(a["qpos_mujoco"][sl, 3:7], dtype=np.float64)
    # A constant pi yaw cancels when all vectors are expressed in root heading,
    # so it is unnecessary for the invariant history feature itself.
    return _history_parts(
        joint_pos=a["joint_pos_isaac"][sl],
        joint_vel=a["joint_vel_isaac"][sl],
        root_pos=root_pos,
        root_quat=root_quat,
        root_vel=a["body_lin_vel_world_mujoco"][sl, LEARNER_ROOT_BODY],
        left_pos=a["body_pos_mujoco"][sl, LEARNER_LEFT_FOOT_BODY],
        right_pos=a["body_pos_mujoco"][sl, LEARNER_RIGHT_FOOT_BODY],
        left_vel=a["body_lin_vel_world_mujoco"][sl, LEARNER_LEFT_FOOT_BODY],
        right_vel=a["body_lin_vel_world_mujoco"][sl, LEARNER_RIGHT_FOOT_BODY],
    )


def _reference_history(clip: ReferenceClip, frame: int):
    sl = slice(frame - HISTORY_FRAMES + 1, frame + 1)
    a = clip.arrays
    return _history_parts(
        joint_pos=a["joint_pos"][sl],
        joint_vel=a["joint_vel"][sl],
        root_pos=a["body_pos_w"][sl, REFERENCE_ROOT_BODY],
        root_quat=a["body_quat_w"][sl, REFERENCE_ROOT_BODY],
        root_vel=a["body_lin_vel_w"][sl, REFERENCE_ROOT_BODY],
        left_pos=a["body_pos_w"][sl, REFERENCE_LEFT_FOOT_BODY],
        right_pos=a["body_pos_w"][sl, REFERENCE_RIGHT_FOOT_BODY],
        left_vel=a["body_lin_vel_w"][sl, REFERENCE_LEFT_FOOT_BODY],
        right_vel=a["body_lin_vel_w"][sl, REFERENCE_RIGHT_FOOT_BODY],
    )


def rank_recovery_candidates(
    trace: RecoveryTrace,
    query: RecoveryQuery,
    clips: Sequence[ReferenceClip],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[RecoveryCandidate, ...]:
    """Hard-filter Justin ascent rows, then rank 0.4 s pose/velocity histories."""
    if top_k <= 0:
        raise RecoveryContractError("top_k must be positive")
    query_pose, query_velocity, query_feet = _query_history(trace, query)
    weights = np.linspace(0.25, 1.0, HISTORY_FRAMES, dtype=np.float64)[:, None]
    i = query.trace_index
    a = trace.arrays
    left_speed = float(
        np.linalg.norm(a["body_lin_vel_world_mujoco"][i, LEARNER_LEFT_FOOT_BODY])
    )
    right_speed = float(
        np.linalg.norm(a["body_lin_vel_world_mujoco"][i, LEARNER_RIGHT_FOOT_BODY])
    )
    query_support = _support_label(
        bool(a["left_foot_contact"][i]) or left_speed < 0.16,
        bool(a["right_foot_contact"][i]) or right_speed < 0.16,
    )
    query_progress = float(a["qpos_mujoco"][i, 0] - S13_LEARNER_FRONT_XY[0])
    query_stage = int(_terrain_stage(query_progress))

    candidates: list[RecoveryCandidate] = []
    for clip in clips:
        root_position = np.asarray(
            clip.arrays["body_pos_w"][:, REFERENCE_ROOT_BODY], dtype=np.float64
        )
        root_z = root_position[:, 2]
        progress = S13_REFERENCE_FRONT_XY[0] - root_position[:, 0]
        stages = _terrain_stage(progress)
        support = _reference_support(clip)
        start = HISTORY_FRAMES - 1
        stop = clip.frame_count - MIN_REMAINING_FRAMES
        for frame in range(start, stop):
            # Hard filters: ascent family, same support/terrain stage, and enough
            # future motion to run the 0.5 s screen plus a continuation.
            if int(stages[frame]) != query_stage:
                continue
            if abs(float(progress[frame]) - query_progress) > 0.20:
                continue
            if support[frame] != query_support:
                continue
            future = min(frame + MIN_REMAINING_FRAMES, clip.frame_count - 1)
            if root_z[future] < root_z[frame] - 0.06:
                continue
            ref_pose, ref_velocity, ref_feet = _reference_history(clip, frame)
            pose_cost = float(np.sum(weights * (query_pose - ref_pose) ** 2) / np.sum(weights))
            velocity_cost = float(
                np.sum(weights * (query_velocity - ref_velocity) ** 2) / np.sum(weights)
            )
            foot_cost = float(
                np.sum(weights * (query_feet - ref_feet) ** 2) / np.sum(weights)
            )
            stage_cost = float(
                ((float(progress[frame]) - query_progress) / S13_TREAD_M) ** 2
            )
            total = pose_cost + 0.45 * velocity_cost + 0.35 * foot_cost + 0.20 * stage_cost
            candidates.append(
                RecoveryCandidate(
                    clip_name=clip.name,
                    clip_path=str(clip.path),
                    clip_sha256=clip.sha256,
                    frame=frame,
                    total_cost=total,
                    pose_cost=pose_cost,
                    velocity_cost=velocity_cost,
                    foot_cost=foot_cost,
                    stage_cost=stage_cost,
                    support=str(support[frame]),
                    stage=int(stages[frame]),
                    frames_remaining=clip.frame_count - frame - 1,
                )
            )
    candidates.sort(key=lambda row: (row.total_cost, row.clip_name, row.frame))
    if not candidates:
        raise RecoveryContractError(
            f"no Justin reference candidate survived filters for rewind {query.rewind_s}s"
        )
    return tuple(candidates[:top_k])


def build_recovery_proposal(
    trace: RecoveryTrace,
    clips: Sequence[ReferenceClip],
    *,
    first_fall_physics_step: int,
    failure_source: str = "reported_first_fall",
    rewinds_s: Sequence[float] = DEFAULT_REWINDS_S,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, object]:
    if failure_source not in {"reported_first_fall", "terminal_degradation"}:
        raise RecoveryContractError(f"unsupported failure source {failure_source!r}")
    queries = select_rewind_queries(
        trace, first_fall_physics_step=first_fall_physics_step, rewinds_s=rewinds_s
    )
    query_rows: list[dict[str, object]] = []
    for query in queries:
        state = learner_state_in_reference_frame(trace, query)
        row: dict[str, object] = {
            "rewind_s": query.rewind_s,
            "trace_index": query.trace_index,
            "physics_step": query.physics_step,
            "history_start": query.history_start,
            "learner_state": {name: value.tolist() for name, value in state.items()},
        }
        try:
            candidates = rank_recovery_candidates(trace, query, clips, top_k=top_k)
        except RecoveryContractError as error:
            # A rejected rewind is evidence about match feasibility, not a reason
            # to discard valid candidates at the other rewind horizons.
            row["candidates"] = []
            row["matching_error"] = str(error)
        else:
            row["candidates"] = [candidate.to_json() for candidate in candidates]
        query_rows.append(row)
    if not any(row["candidates"] for row in query_rows):
        raise RecoveryContractError("no Justin reference candidate survived any rewind")
    return {
        "schema": "justin-s13-tracker-recovery-proposal/v1",
        "trace_path": str(trace.path),
        "trace_sha256": trace.sha256,
        "first_fall_physics_step": (
            int(first_fall_physics_step)
            if failure_source == "reported_first_fall"
            else None
        ),
        "failure_detection": {
            "physics_step": int(first_fall_physics_step),
            "source": failure_source,
        },
        "failure_onset_physics_step": recovery_failure_anchor_physics_step(
            trace, first_fall_physics_step=first_fall_physics_step
        ),
        "control_hz": CONTROL_HZ,
        "history_frames": HISTORY_FRAMES,
        "matching_only": True,
        "fine_tuning_performed": False,
        "scene_transform": {
            "learner_front_xy": S13_LEARNER_FRONT_XY.tolist(),
            "reference_front_xy": S13_REFERENCE_FRONT_XY.tolist(),
            "yaw_delta_rad": S13_SCENE_YAW_DELTA_RAD,
        },
        "reference_bank": [
            {
                "name": clip.name,
                "path": str(clip.path),
                "sha256": clip.sha256,
                "frames": clip.frame_count,
            }
            for clip in clips
        ],
        "queries": query_rows,
    }


def write_recovery_proposal(proposal: Mapping[str, object], path: str | Path) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n")
    return destination
