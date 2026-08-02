"""Deterministic difficult-state corpus for terrain motion quality oracles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Callable, Mapping

import numpy as np


_ARRAY_SHAPES = {
    "qpos": (36,),
    "joint_position": (29,),
    "joint_velocity": (29,),
    "root_position_world": (3,),
    "foot_position_world": (2, 3),
    "source_support_mask": (2,),
    "command_velocity_world_xy": (2,),
    "terrain_patch_world_xyh": (441, 3),
}
_REQUIRED_ROUTE_ARRAYS = frozenset(
    (
        "qpos",
        "joint_position",
        "joint_velocity",
        "root_position_world",
        "root_yaw_world",
        "foot_position_world",
        "foot_surface_height_m",
        "command_velocity_world_xy",
        "command_heading_world_yaw",
        "command_segment_index",
        "selected_clip_path",
        "selected_source_frame",
    )
)
_REASON_ORDER = {
    "reset": 0,
    "command-boundary": 1,
    "source-transition": 2,
    "split-height-stance": 3,
    "elevated-direction-change": 4,
}


def _readonly(value, shape: tuple[int, ...], name: str, *, boolean=False):
    dtype = np.bool_ if boolean else np.float64
    result = np.array(value, dtype=dtype, copy=True)
    if result.shape != shape or (
        not boolean and not np.isfinite(result).all()
    ):
        qualifier = "boolean" if boolean else "finite"
        raise ValueError(f"{name} must have {qualifier} shape {shape}")
    result.setflags(write=False)
    return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _state_digest(
    *,
    route_name: str,
    route_frame: int,
    reason: str,
    arrays: Mapping[str, np.ndarray],
    root_yaw_world: float,
    command_heading_world_yaw: float,
    selected_clip_path: str,
    selected_source_frame: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-terrain-quality-state/v1\0")
    digest.update(
        _canonical_json(
            {
                "route_name": route_name,
                "route_frame": route_frame,
                "reason": reason,
                "root_yaw_world": root_yaw_world,
                "command_heading_world_yaw": command_heading_world_yaw,
                "selected_clip_path": selected_clip_path,
                "selected_source_frame": selected_source_frame,
            }
        )
    )
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        digest.update(b"\0" + name.encode("ascii") + b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(_canonical_json(value.shape))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenQualityState:
    state_id: str
    route_name: str
    route_frame: int
    reason: str
    qpos: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    root_position_world: np.ndarray
    root_yaw_world: float
    foot_position_world: np.ndarray
    source_support_mask: np.ndarray
    command_velocity_world_xy: np.ndarray
    command_heading_world_yaw: float
    terrain_patch_world_xyh: np.ndarray
    selected_clip_path: str
    selected_source_frame: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.route_name, str)
            or not self.route_name
            or type(self.route_frame) is not int
            or self.route_frame < 0
            or self.reason not in _REASON_ORDER
            or not isinstance(self.selected_clip_path, str)
            or not self.selected_clip_path
            or type(self.selected_source_frame) is not int
            or self.selected_source_frame < 0
        ):
            raise ValueError("frozen quality state identity is invalid")
        for name, shape in _ARRAY_SHAPES.items():
            object.__setattr__(
                self,
                name,
                _readonly(
                    getattr(self, name),
                    shape,
                    name,
                    boolean=name == "source_support_mask",
                ),
            )
        for name in ("root_yaw_world", "command_heading_world_yaw"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        arrays = {name: getattr(self, name) for name in _ARRAY_SHAPES}
        expected = _state_digest(
            route_name=self.route_name,
            route_frame=self.route_frame,
            reason=self.reason,
            arrays=arrays,
            root_yaw_world=self.root_yaw_world,
            command_heading_world_yaw=self.command_heading_world_yaw,
            selected_clip_path=self.selected_clip_path,
            selected_source_frame=self.selected_source_frame,
        )
        if self.state_id != expected:
            raise ValueError("frozen quality state hash is invalid")


def _route_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    frame_count: int,
    tail: tuple[int, ...],
    *,
    numeric: bool = True,
) -> np.ndarray:
    value = np.asarray(arrays[name])
    if value.shape != (frame_count,) + tail or (
        numeric and value.dtype.kind in "fc" and not np.isfinite(value).all()
    ):
        raise ValueError(f"route array {name} has invalid shape or values")
    return value


def _capture_reasons(
    arrays: Mapping[str, np.ndarray], support: np.ndarray
) -> tuple[tuple[int, str], ...]:
    frame_count = support.shape[0]
    reasons: set[tuple[int, str]] = {(0, "reset")}
    segments = np.asarray(arrays["command_segment_index"])
    paths = np.asarray(arrays["selected_clip_path"])
    source = np.asarray(arrays["selected_source_frame"])
    surface = np.asarray(arrays["foot_surface_height_m"], dtype=np.float64)
    velocity = np.asarray(arrays["command_velocity_world_xy"], dtype=np.float64)
    heading = np.asarray(arrays["command_heading_world_yaw"], dtype=np.float64)
    for frame in range(1, frame_count):
        command_changed = segments[frame] != segments[frame - 1]
        if command_changed:
            reasons.add((frame, "command-boundary"))
        if paths[frame] != paths[frame - 1] or source[frame] != source[frame - 1] + 1:
            reasons.add((frame, "source-transition"))
        if command_changed and max(surface[frame - 1]) >= 0.08:
            reversal = float(np.dot(velocity[frame - 1], velocity[frame])) < 0.0
            yaw_delta = math.atan2(
                math.sin(float(heading[frame] - heading[frame - 1])),
                math.cos(float(heading[frame] - heading[frame - 1])),
            )
            if reversal or abs(yaw_delta) >= math.radians(30.0):
                reasons.add((frame, "elevated-direction-change"))
    split = np.flatnonzero(
        support.all(axis=1)
        & (np.abs(surface[:, 0] - surface[:, 1]) >= 0.08)
    )
    if split.size:
        reasons.add((int(split[0]), "split-height-stance"))
    return tuple(
        sorted(reasons, key=lambda item: (item[0], _REASON_ORDER[item[1]]))
    )


def _terrain_patch(
    root_xy: np.ndarray,
    terrain_patch_sampler: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    axis = np.linspace(-0.5, 0.5, 21, dtype=np.float64)
    offset_x, offset_y = np.meshgrid(axis, axis, indexing="xy")
    xy = np.stack((offset_x.reshape(-1), offset_y.reshape(-1)), axis=1)
    xy = xy + np.asarray(root_xy, dtype=np.float64)
    try:
        height = np.asarray(terrain_patch_sampler(xy), dtype=np.float64)
    except Exception as error:
        raise ValueError("terrain patch sampling failed") from error
    if height.shape != (441,) or not np.isfinite(height).all():
        raise ValueError("terrain patch sampler must return finite shape (441,)")
    return np.concatenate((xy, height[:, None]), axis=1)


def _make_state(
    route_name: str,
    frame: int,
    reason: str,
    arrays: Mapping[str, np.ndarray],
    support: np.ndarray,
    terrain_patch_sampler: Callable[[np.ndarray], np.ndarray],
) -> FrozenQualityState:
    values = {
        "qpos": arrays["qpos"][frame],
        "joint_position": arrays["joint_position"][frame],
        "joint_velocity": arrays["joint_velocity"][frame],
        "root_position_world": arrays["root_position_world"][frame],
        "foot_position_world": arrays["foot_position_world"][frame],
        "source_support_mask": support[frame],
        "command_velocity_world_xy": arrays["command_velocity_world_xy"][frame],
        "terrain_patch_world_xyh": _terrain_patch(
            arrays["root_position_world"][frame, :2], terrain_patch_sampler
        ),
    }
    scalar = {
        "route_name": route_name,
        "route_frame": frame,
        "reason": reason,
        "root_yaw_world": float(arrays["root_yaw_world"][frame]),
        "command_heading_world_yaw": float(
            arrays["command_heading_world_yaw"][frame]
        ),
        "selected_clip_path": str(arrays["selected_clip_path"][frame]),
        "selected_source_frame": int(arrays["selected_source_frame"][frame]),
    }
    state_id = _state_digest(arrays=values, **scalar)
    return FrozenQualityState(state_id=state_id, **scalar, **values)


def capture_quality_states(
    *,
    route_name: str,
    arrays: Mapping[str, np.ndarray],
    source_support_mask: np.ndarray,
    terrain_patch_sampler: Callable[[np.ndarray], np.ndarray],
) -> tuple[FrozenQualityState, ...]:
    """Capture reset, command, source, and split-height diagnostic states."""

    if not isinstance(route_name, str) or not route_name:
        raise ValueError("quality state route name is invalid")
    if not isinstance(arrays, Mapping):
        raise ValueError("quality state arrays must be a mapping")
    missing = sorted(_REQUIRED_ROUTE_ARRAYS - set(arrays))
    if missing:
        raise ValueError("quality state arrays are missing: " + ", ".join(missing))
    qpos = np.asarray(arrays["qpos"])
    if qpos.ndim != 2 or qpos.shape[1:] != (36,) or qpos.shape[0] < 1:
        raise ValueError("route array qpos has invalid shape or values")
    frame_count = int(qpos.shape[0])
    tails = {
        "joint_position": (29,),
        "joint_velocity": (29,),
        "root_position_world": (3,),
        "root_yaw_world": (),
        "foot_position_world": (2, 3),
        "foot_surface_height_m": (2,),
        "command_velocity_world_xy": (2,),
        "command_heading_world_yaw": (),
        "command_segment_index": (),
        "selected_clip_path": (),
        "selected_source_frame": (),
    }
    _route_array(arrays, "qpos", frame_count, (36,))
    for name, tail in tails.items():
        _route_array(
            arrays,
            name,
            frame_count,
            tail,
            numeric=name != "selected_clip_path",
        )
    support = np.asarray(source_support_mask)
    if support.dtype != np.bool_ or support.shape != (frame_count, 2):
        raise ValueError("source_support_mask must have boolean shape (T, 2)")
    if not callable(terrain_patch_sampler):
        raise ValueError("terrain_patch_sampler must be callable")
    return tuple(
        _make_state(
            route_name,
            frame,
            reason,
            arrays,
            support,
            terrain_patch_sampler,
        )
        for frame, reason in _capture_reasons(arrays, support)
    )


def _corpus_hash(states: tuple[FrozenQualityState, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"g1-terrain-quality-state-corpus/v1\0")
    for state in states:
        digest.update(state.state_id.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_quality_state_corpus(
    states: tuple[FrozenQualityState, ...], output: str | Path
) -> str:
    """Atomically save a state corpus and return its deterministic identity."""

    if not isinstance(states, tuple) or any(
        not isinstance(state, FrozenQualityState) for state in states
    ):
        raise ValueError("quality state corpus must be a tuple of states")
    if len({state.state_id for state in states}) != len(states):
        raise ValueError("quality state corpus contains duplicate identities")
    target = Path(output).resolve()
    if target.exists():
        raise ValueError(f"quality state output already exists: {target}")
    staging = target.with_name(target.name + ".tmp")
    if staging.exists():
        raise ValueError(f"quality state staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        stacked = {
            name: np.stack([getattr(state, name) for state in states])
            if states
            else np.empty((0,) + shape)
            for name, shape in _ARRAY_SHAPES.items()
        }
        np.savez_compressed(staging / "states.npz", **stacked)
        identity = _corpus_hash(states)
        manifest = {
            "schema": "g1-terrain-quality-state-corpus/v1",
            "deterministic_sha256": identity,
            "states": [
                {
                    "state_id": state.state_id,
                    "route_name": state.route_name,
                    "route_frame": state.route_frame,
                    "reason": state.reason,
                    "root_yaw_world": state.root_yaw_world,
                    "command_heading_world_yaw": state.command_heading_world_yaw,
                    "selected_clip_path": state.selected_clip_path,
                    "selected_source_frame": state.selected_source_frame,
                }
                for state in states
            ],
        }
        (staging / "manifest.json").write_bytes(_canonical_json(manifest))
        os.replace(staging, target)
        return identity
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_quality_state_corpus(
    root: str | Path,
) -> tuple[FrozenQualityState, ...]:
    """Load and authenticate a saved difficult-state corpus."""

    target = Path(root).resolve()
    try:
        manifest = json.loads((target / "manifest.json").read_text())
        archive = np.load(target / "states.npz", allow_pickle=False)
    except Exception as error:
        raise ValueError("cannot load quality state corpus") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "g1-terrain-quality-state-corpus/v1"
        or not isinstance(manifest.get("states"), list)
    ):
        raise ValueError("quality state manifest is invalid")
    metadata = manifest["states"]
    if any(
        name not in archive or archive[name].shape[0] != len(metadata)
        for name in _ARRAY_SHAPES
    ):
        raise ValueError("quality state archive is invalid")
    states = tuple(
        FrozenQualityState(
            **entry,
            **{name: archive[name][index] for name in _ARRAY_SHAPES},
        )
        for index, entry in enumerate(metadata)
    )
    if _corpus_hash(states) != manifest.get("deterministic_sha256"):
        raise ValueError("quality state corpus authentication failed")
    return states
