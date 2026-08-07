"""30 Hz source adapters for GRAIL slope and G1-retargeted LAFAN clips."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re

import numpy as np

from mm_sonic.grail_terrain_source import (
    G1MujocoFK,
    load_grail_motion,
    mujoco_to_isaaclab_joints,
    resample_grail_motion,
)
from mm_sonic.terrain_oracle.canonical import (
    ISAACLAB_BODY_NAMES,
    ISAACLAB_JOINT_NAMES,
)
from mm_sonic.terrain_oracle.math3d import (
    finite_difference,
    quaternion_inverse_wxyz,
    quaternion_multiply_wxyz,
    unroll_quaternions_wxyz,
)
from mm_sonic.terrain_oracle.source_lafan import _load_rows


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLOPE_RE = re.compile(r"(?:^|__)slope_\d{3}(?:__|$)")
_FPS = 30.0
_GRAIL_SOURCE_LICENSE_ID = "UNRECORDED"
_LAFAN_SOURCE_LICENSE_ID = "CC-BY-NC-ND-4.0"
_TERRAIN_QUATERNION_WORLD_FROM_USD_WXYZ = np.array(
    (np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)), dtype=np.float32
)


def _readonly_float32(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be a real numeric array")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(array, dtype=np.float64).astype(
            np.float32, order="C", casting="unsafe", copy=True
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain finite binary32 values")
    result = np.ascontiguousarray(result).copy(order="C")
    result.flags.writeable = False
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _angular_velocity_world_wxyz(
    quaternion_world_wxyz: np.ndarray, fps: float
) -> np.ndarray:
    """World-frame angular velocity from shortest consecutive rotations."""

    trace = unroll_quaternions_wxyz(quaternion_world_wxyz)
    relative = quaternion_multiply_wxyz(
        trace[1:], quaternion_inverse_wxyz(trace[:-1])
    )
    relative = np.asarray(relative, dtype=np.float64)
    relative *= np.where(relative[..., :1] < 0.0, -1.0, 1.0)
    vector = relative[..., 1:]
    magnitude = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(
        magnitude, np.clip(relative[..., 0], -1.0, 1.0)
    )
    scale = np.divide(
        angle * fps,
        magnitude,
        out=np.zeros_like(magnitude),
        where=magnitude > 1.0e-12,
    )
    segment = vector * scale[..., None]
    output = np.empty((len(trace), *trace.shape[1:-1], 3), dtype=np.float64)
    output[0] = segment[0]
    output[-1] = segment[-1]
    if len(output) > 2:
        output[1:-1] = 0.5 * (segment[:-1] + segment[1:])
    return _readonly_float32(output, "angular velocity")


@dataclass(frozen=True)
class GrailSlopeRecord:
    """A strict one-to-one GRAIL robot/USD slope pairing."""

    stem: str
    terrain_id: str
    robot_path: Path
    terrain_path: Path
    expected_frames: int = 250
    terrain_position_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    terrain_quaternion_world_from_usd_wxyz: np.ndarray = field(
        default_factory=lambda: _TERRAIN_QUATERNION_WORLD_FROM_USD_WXYZ.copy()
    )

    def __post_init__(self) -> None:
        if type(self.stem) is not str or not self.stem:
            raise ValueError("GRAIL stem must be a nonempty string")
        if type(self.terrain_id) is not str or not self.terrain_id:
            raise ValueError("GRAIL terrain_id must be a nonempty string")
        if type(self.expected_frames) is not int or self.expected_frames < 2:
            raise ValueError("GRAIL expected_frames must be an integer >= 2")
        object.__setattr__(self, "robot_path", Path(self.robot_path))
        object.__setattr__(self, "terrain_path", Path(self.terrain_path))
        position = _readonly_float32(self.terrain_position_world, "terrain position")
        quaternion = _readonly_float32(
            self.terrain_quaternion_world_from_usd_wxyz, "terrain quaternion"
        )
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("GRAIL terrain pose must have shapes (3,) and (4,)")
        if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-5):
            raise ValueError("GRAIL terrain quaternion must be normalized WXYZ")
        object.__setattr__(self, "terrain_position_world", position)
        object.__setattr__(self, "terrain_quaternion_world_from_usd_wxyz", quaternion)


@dataclass(frozen=True)
class PFNNSourceClip:
    clip_id: str
    terrain_id: str
    fps: float
    root_position_world: np.ndarray
    root_quaternion_world_wxyz: np.ndarray
    joint_position: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_world_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray
    root_linear_velocity_world: np.ndarray
    root_angular_velocity_world: np.ndarray
    joint_velocity: np.ndarray
    terrain_path: Path | None
    terrain_position_world: np.ndarray
    terrain_quaternion_world_from_usd_wxyz: np.ndarray
    motion_sha256: str
    terrain_sha256: str | None
    source_license_id: str

    def __post_init__(self) -> None:
        if type(self.clip_id) is not str or not self.clip_id:
            raise ValueError("clip_id must be a nonempty string")
        if type(self.terrain_id) is not str or not self.terrain_id:
            raise ValueError("terrain_id must be a nonempty string")
        if float(self.fps) != _FPS:
            raise ValueError("PFNN source clips must be exactly 30 Hz")
        if type(self.source_license_id) is not str or not self.source_license_id:
            raise ValueError("source_license_id must be a nonempty string")
        if _SHA256_RE.fullmatch(self.motion_sha256) is None:
            raise ValueError("motion_sha256 must be a lowercase SHA-256 digest")
        if self.terrain_sha256 is not None and _SHA256_RE.fullmatch(self.terrain_sha256) is None:
            raise ValueError("terrain_sha256 must be a lowercase SHA-256 digest or None")
        if (self.terrain_path is None) != (self.terrain_sha256 is None):
            raise ValueError("terrain_path and terrain_sha256 must both be present or absent")
        if self.terrain_path is not None:
            object.__setattr__(self, "terrain_path", Path(self.terrain_path))

        arrays = (
            "root_position_world",
            "root_quaternion_world_wxyz",
            "joint_position",
            "body_position_world",
            "body_quaternion_world_wxyz",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
            "root_linear_velocity_world",
            "root_angular_velocity_world",
            "joint_velocity",
            "terrain_position_world",
            "terrain_quaternion_world_from_usd_wxyz",
        )
        for name in arrays:
            object.__setattr__(self, name, _readonly_float32(getattr(self, name), name))
        frames = len(self.root_position_world)
        expected_shapes = {
            "root_position_world": (frames, 3),
            "root_quaternion_world_wxyz": (frames, 4),
            "joint_position": (frames, 29),
            "body_position_world": (frames, 30, 3),
            "body_quaternion_world_wxyz": (frames, 30, 4),
            "body_linear_velocity_world": (frames, 30, 3),
            "body_angular_velocity_world": (frames, 30, 3),
            "root_linear_velocity_world": (frames, 3),
            "root_angular_velocity_world": (frames, 3),
            "joint_velocity": (frames, 29),
            "terrain_position_world": (3,),
            "terrain_quaternion_world_from_usd_wxyz": (4,),
        }
        if frames < 2:
            raise ValueError("PFNN source clips need at least two frames")
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name in (
            "root_quaternion_world_wxyz",
            "body_quaternion_world_wxyz",
            "terrain_quaternion_world_from_usd_wxyz",
        ):
            norms = np.linalg.norm(getattr(self, name), axis=-1)
            if not np.allclose(norms, 1.0, atol=1.0e-5, rtol=0.0):
                raise ValueError(f"{name} must contain normalized WXYZ quaternions")

    @property
    def frame_count(self) -> int:
        return len(self.root_position_world)


def discover_grail_slope_records(root: Path) -> tuple[GrailSlopeRecord, ...]:
    """Discover sorted GRAIL slope robot/USD pairs with the reviewed -90° yaw."""

    grail_root = Path(root).expanduser().resolve()
    robot_dir = grail_root / "data" / "slope" / "robot"
    terrain_dir = grail_root / "data" / "slope" / "object_usd"
    if not robot_dir.is_dir() or not terrain_dir.is_dir():
        raise FileNotFoundError("GRAIL slope robot and object_usd directories are required")
    records: list[GrailSlopeRecord] = []
    seen: set[str] = set()
    for robot_path in sorted(robot_dir.glob("*.pkl")):
        stem = robot_path.stem
        if stem in seen:
            raise ValueError(f"duplicate GRAIL slope stem: {stem}")
        match = _SLOPE_RE.search(stem)
        if match is None:
            raise ValueError(f"{stem}: GRAIL slope stem must contain slope_NNN")
        terrain_path = terrain_dir / f"{stem}.usd"
        if not terrain_path.is_file():
            raise ValueError(f"{stem}: missing paired USD")
        seen.add(stem)
        records.append(
            GrailSlopeRecord(
                stem=stem,
                terrain_id=match.group(0).strip("_"),
                robot_path=robot_path,
                terrain_path=terrain_path,
            )
        )
    return tuple(records)


def _validated_forward(fk: object, root: np.ndarray, quaternion: np.ndarray, joints: np.ndarray) -> object:
    forward_method = getattr(fk, "forward", None)
    if not callable(forward_method):
        raise ValueError("fk must provide a callable forward method")
    result = forward_method(root, quaternion, joints)
    if tuple(getattr(result, "body_names", ())) != ISAACLAB_BODY_NAMES:
        raise ValueError("FK result does not use the canonical IsaacLab body order")
    return result


def _make_clip(
    *,
    clip_id: str,
    terrain_id: str,
    root_position: np.ndarray,
    root_wxyz: np.ndarray,
    joints_mujoco: np.ndarray,
    forward: object,
    terrain_path: Path | None,
    terrain_position: np.ndarray,
    terrain_quaternion: np.ndarray,
    motion_sha256: str,
    terrain_sha256: str | None,
    source_license_id: str,
) -> PFNNSourceClip:
    body_wxyz = unroll_quaternions_wxyz(
        np.asarray(forward.body_quaternion_world_xyzw)[..., (3, 0, 1, 2)]
    )
    joint_position = mujoco_to_isaaclab_joints(joints_mujoco)
    body_position = np.asarray(forward.body_position_world)
    return PFNNSourceClip(
        clip_id=clip_id,
        terrain_id=terrain_id,
        fps=_FPS,
        root_position_world=root_position,
        root_quaternion_world_wxyz=root_wxyz,
        joint_position=joint_position,
        body_position_world=body_position,
        body_quaternion_world_wxyz=body_wxyz,
        body_linear_velocity_world=finite_difference(body_position, _FPS),
        body_angular_velocity_world=_angular_velocity_world_wxyz(body_wxyz, _FPS),
        root_linear_velocity_world=finite_difference(root_position, _FPS),
        root_angular_velocity_world=_angular_velocity_world_wxyz(root_wxyz, _FPS),
        joint_velocity=finite_difference(joint_position, _FPS),
        terrain_path=terrain_path,
        terrain_position_world=terrain_position,
        terrain_quaternion_world_from_usd_wxyz=terrain_quaternion,
        motion_sha256=motion_sha256,
        terrain_sha256=terrain_sha256,
        source_license_id=source_license_id,
    )


def load_grail_source(record: GrailSlopeRecord, fk: object) -> PFNNSourceClip:
    """Load one paired 25 Hz GRAIL slope clip into the 30 Hz PFNN contract."""

    if not isinstance(record, GrailSlopeRecord):
        raise TypeError("record must be a GrailSlopeRecord")
    raw = load_grail_motion(record.robot_path, expected_frames=record.expected_frames)
    resampled = resample_grail_motion(
        raw.root_position,
        raw.root_quaternion_xyzw,
        raw.dof_mujoco,
        source_fps=25.0,
        target_fps=_FPS,
    )
    forward = _validated_forward(
        fk,
        resampled.root_position,
        resampled.root_quaternion_xyzw,
        resampled.dof_mujoco,
    )
    return _make_clip(
        clip_id=record.stem,
        terrain_id=record.terrain_id,
        root_position=resampled.root_position,
        root_wxyz=unroll_quaternions_wxyz(
            resampled.root_quaternion_xyzw[:, (3, 0, 1, 2)]
        ),
        joints_mujoco=resampled.dof_mujoco,
        forward=forward,
        terrain_path=record.terrain_path,
        terrain_position=record.terrain_position_world,
        terrain_quaternion=record.terrain_quaternion_world_from_usd_wxyz,
        motion_sha256=_sha256(record.robot_path),
        terrain_sha256=_sha256(record.terrain_path),
        source_license_id=_GRAIL_SOURCE_LICENSE_ID,
    )


def load_lafan_source(path: Path, fk: object) -> PFNNSourceClip:
    """Load one 30 Hz G1-retargeted LAFAN CSV without temporal resampling."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    rows = _load_rows(source_path)
    root_position = np.ascontiguousarray(rows[:, :3], dtype=np.float32)
    root_xyzw = np.ascontiguousarray(rows[:, 3:7], dtype=np.float32)
    joints_mujoco = np.ascontiguousarray(rows[:, 7:], dtype=np.float32)
    forward = _validated_forward(fk, root_position, root_xyzw, joints_mujoco)
    return _make_clip(
        clip_id=source_path.stem,
        terrain_id="flat",
        root_position=root_position,
        root_wxyz=unroll_quaternions_wxyz(rows[:, (6, 3, 4, 5)]),
        joints_mujoco=joints_mujoco,
        forward=forward,
        terrain_path=None,
        terrain_position=np.zeros(3, dtype=np.float32),
        terrain_quaternion=np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
        motion_sha256=_sha256(source_path),
        terrain_sha256=None,
        source_license_id=_LAFAN_SOURCE_LICENSE_ID,
    )


def _finite_clip(clip: PFNNSourceClip) -> bool:
    return all(
        np.isfinite(getattr(clip, name)).all()
        for name in (
            "root_position_world",
            "root_quaternion_world_wxyz",
            "joint_position",
            "body_position_world",
            "body_quaternion_world_wxyz",
            "body_linear_velocity_world",
            "body_angular_velocity_world",
            "root_linear_velocity_world",
            "root_angular_velocity_world",
            "joint_velocity",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grail-root", type=Path, required=True)
    parser.add_argument("--lafan-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--inspect-one", action="store_true")
    args = parser.parse_args(argv)
    if not args.inspect_one:
        parser.error("only --inspect-one is supported")
    records = discover_grail_slope_records(args.grail_root)
    if not records:
        raise ValueError("no GRAIL slope records found")
    lafan_paths = sorted(args.lafan_root.rglob("*.csv"))
    if not lafan_paths:
        raise ValueError("no LAFAN CSV files found")
    fk = G1MujocoFK(args.model_path)
    grail = load_grail_source(records[0], fk)
    lafan = load_lafan_source(lafan_paths[0], fk)
    print(
        f"GRAIL clip={grail.clip_id} fps={grail.fps:g} finite={_finite_clip(grail)} "
        f"terrain={grail.terrain_path}"
    )
    print(f"LAFAN clip={lafan.clip_id} fps={lafan.fps:g} finite={_finite_clip(lafan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
