"""Stream released PFNN post-IK source poses through pinned GMR into G1 qpos.

The PFNN process owns trajectory, terrain response, phase, and source pose.  This
module performs only the existing GMR morphology adapter and emits display qpos.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import struct
import sys
from typing import BinaryIO, TextIO

import numpy as np
from scipy.spatial.transform import Rotation


PFNN_POSITION_SCALE = 5.6444
# The released network/demo has already applied PFNN_POSITION_SCALE during
# database generation.  Its runtime export contract is therefore centimeters.
_SOURCE_TO_GMR_POSITION_SCALE = 1.0 / 100.0
_SOURCE_TO_GMR_ROTATION = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
SOURCE_JOINT_NAMES = (
    "Hips",
    "LHipJoint",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RHipJoint",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LowerBack",
    "Spine",
    "Spine1",
    "Neck",
    "Neck1",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftFingerBase",
    "LeftHandIndex1",
    "LThumb",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightFingerBase",
    "RightHandIndex1",
    "RThumb",
)
_GMR_JOINT_NAMES = tuple(
    "Spine2" if name == "Spine1" else name for name in SOURCE_JOINT_NAMES
)
_HEADER = struct.Struct("<8sIIIIfI")
_RECORD_PREFIX = struct.Struct("<QIIff3f2f")
_MAGIC = b"PFNNXFM"


@dataclass(frozen=True)
class ExportMetadata:
    joint_count: int
    record_bytes: int
    coordinate: int
    fps: float


@dataclass(frozen=True)
class ExportRecord:
    frame: int
    world: int
    flags: int
    phase: float
    sampled_root_height: float
    trajectory_root_xyz: np.ndarray
    trajectory_forward_xz: np.ndarray
    positions: np.ndarray
    quaternions_wxyz: np.ndarray


def _read_exact(stream: BinaryIO, count: int, *, allow_eof: bool = False) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            if allow_eof and not chunks:
                return None
            raise EOFError(f"truncated PFNN export: wanted {count}, got {count - remaining}")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def read_header(stream: BinaryIO) -> ExportMetadata:
    payload = _read_exact(stream, _HEADER.size)
    assert payload is not None
    magic, version, joints, record_bytes, coordinate, fps, reserved = _HEADER.unpack(
        payload
    )
    if magic.rstrip(b"\0") != _MAGIC:
        raise ValueError(f"invalid PFNN export magic {magic!r}")
    if version != 1 or joints != 31 or record_bytes != 912 or coordinate != 1:
        raise ValueError(
            "unsupported PFNN export contract: "
            f"version={version} joints={joints} bytes={record_bytes} coord={coordinate}"
        )
    if fps != 60.0 or reserved != 0:
        raise ValueError(f"invalid PFNN export timing/reserved fields: {fps}, {reserved}")
    return ExportMetadata(joints, record_bytes, coordinate, float(fps))


def read_record(
    stream: BinaryIO, metadata: ExportMetadata
) -> ExportRecord | None:
    payload = _read_exact(stream, metadata.record_bytes, allow_eof=True)
    if payload is None:
        return None
    prefix = _RECORD_PREFIX.unpack_from(payload)
    pose = np.frombuffer(
        payload,
        dtype="<f4",
        count=metadata.joint_count * 7,
        offset=_RECORD_PREFIX.size,
    ).reshape(metadata.joint_count, 7)
    if not np.isfinite(pose).all():
        raise ValueError("PFNN export pose contains non-finite values")
    return ExportRecord(
        frame=int(prefix[0]),
        world=int(prefix[1]),
        flags=int(prefix[2]),
        phase=float(prefix[3]),
        sampled_root_height=float(prefix[4]),
        trajectory_root_xyz=np.asarray(prefix[5:8], dtype=np.float64),
        trajectory_forward_xz=np.asarray(prefix[8:10], dtype=np.float64),
        positions=np.asarray(pose[:, :3], dtype=np.float64),
        quaternions_wxyz=np.asarray(pose[:, 3:7], dtype=np.float64),
    )


def source_pose_to_gmr_frame(
    positions: object, quaternions_wxyz: object
) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    source_position = np.asarray(positions, dtype=np.float64)
    source_quaternion = np.asarray(quaternions_wxyz, dtype=np.float64)
    if source_position.shape != (31, 3) or source_quaternion.shape != (31, 4):
        raise ValueError("source PFNN pose must have position [31,3] and quaternion [31,4]")
    if not np.isfinite(source_position).all() or not np.isfinite(source_quaternion).all():
        raise ValueError("source PFNN pose must be finite")
    norms = np.linalg.norm(source_quaternion, axis=1)
    if np.any(norms < 1.0e-8):
        raise ValueError("source PFNN pose contains a zero quaternion")
    source_quaternion = source_quaternion / norms[:, None]

    gmr_position = (
        source_position @ _SOURCE_TO_GMR_ROTATION.T
    ) * _SOURCE_TO_GMR_POSITION_SCALE
    source_rotation = Rotation.from_quat(
        source_quaternion[:, (1, 2, 3, 0)]
    ).as_matrix()
    gmr_rotation = _SOURCE_TO_GMR_ROTATION[None, :, :] @ source_rotation
    gmr_quaternion = Rotation.from_matrix(gmr_rotation).as_quat(scalar_first=True)

    frame = {
        name: [gmr_position[index].copy(), gmr_quaternion[index].copy()]
        for index, name in enumerate(_GMR_JOINT_NAMES)
    }
    frame["LeftFootMod"] = [
        frame["LeftFoot"][0].copy(),
        frame["LeftToeBase"][1].copy(),
    ]
    frame["RightFootMod"] = [
        frame["RightFoot"][0].copy(),
        frame["RightToeBase"][1].copy(),
    ]
    return frame, gmr_position[0].copy()


class G1RetargetBridge:
    def __init__(self, gmr_root: Path) -> None:
        root = Path(gmr_root).resolve(strict=True)
        sys.path.insert(0, str(root))
        with redirect_stdout(sys.stderr):
            from general_motion_retargeting import GeneralMotionRetargeting

            self._retargeter = GeneralMotionRetargeting(
                src_human="bvh_nokov",
                tgt_robot="unitree_g1",
                actual_human_height=1.75,
                verbose=False,
            )
    def retarget(self, record: ExportRecord) -> dict[str, object]:
        frame, source_root = source_pose_to_gmr_frame(
            record.positions, record.quaternions_wxyz
        )
        qpos = np.asarray(self._retargeter.retarget(frame), dtype=np.float64)
        return {
            "frame": record.frame,
            "world": record.world,
            "flags": record.flags,
            "phase": record.phase,
            "fps": 60.0,
            "qpos_wxyz": qpos.tolist(),
            "source_root_pos_zup_m": qpos[:3].tolist(),
            "source_hips_zup_m": source_root.tolist(),
            "trajectory_root_zup_m": (
                record.trajectory_root_xyz @ _SOURCE_TO_GMR_ROTATION.T
                * _SOURCE_TO_GMR_POSITION_SCALE
            ).tolist(),
        }


def run_stream(
    source: BinaryIO,
    destination: TextIO,
    bridge: G1RetargetBridge,
    *,
    max_frames: int | None = None,
) -> int:
    metadata = read_header(source)
    count = 0
    while max_frames is None or count < max_frames:
        record = read_record(source, metadata)
        if record is None:
            break
        destination.write(json.dumps(bridge.retarget(record), separators=(",", ":")))
        destination.write("\n")
        destination.flush()
        count += 1
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="PFNN binary export file/FIFO; omit for stdin")
    parser.add_argument(
        "--gmr-root",
        type=Path,
        default=Path("/home/ubuntu/.cache/native-g1-pfnn/GMR"),
    )
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.max_frames is not None and arguments.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    bridge = G1RetargetBridge(arguments.gmr_root)
    if arguments.input is None:
        source: BinaryIO = sys.stdin.buffer
        return run_stream(source, sys.stdout, bridge, max_frames=arguments.max_frames)
    with arguments.input.open("rb", buffering=0) as source:
        return run_stream(source, sys.stdout, bridge, max_frames=arguments.max_frames)


if __name__ == "__main__":
    main()
