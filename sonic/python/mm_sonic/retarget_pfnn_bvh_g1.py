"""Retarget one released PFNN BVH clip to the native 29-DoF Unitree G1.

The expensive retargeting dependencies are imported only by the CLI.  Source
preparation and artifact validation intentionally depend on NumPy alone so the
contracts remain cheap to test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Mapping

import numpy as np


GMR_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
RETARGET_PROJECT_COMMIT = "fb3433a6310ab4198102d3905e74b73944fc1f6b"
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_ALIASES = (("Spine1", "Spine2"),)
_FRAMES_RE = re.compile(r"(?m)^\s*Frames:\s*(\d+)\s*$")
_FRAME_TIME_RE = re.compile(
    r"(?m)^\s*Frame\s+Time:\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)\s*$"
)
_SPINE1_RE = re.compile(r"(?m)^(\s*JOINT\s+)Spine1(\s*)$")
_SPINE2_RE = re.compile(r"(?m)^\s*JOINT\s+Spine2\s*$")
_LEFT_TOE_RE = re.compile(r"(?m)^\s*JOINT\s+LeftToeBase\s*$")
_RIGHT_TOE_RE = re.compile(r"(?m)^\s*JOINT\s+RightToeBase\s*$")


@dataclass(frozen=True)
class SourceReceipt:
    source_sha256: str
    prepared_sha256: str
    frame_count: int
    fps: float
    aliases: tuple[tuple[str, str], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bvh_metadata(contents: str) -> tuple[int, float, int]:
    frames = _FRAMES_RE.findall(contents)
    frame_times = _FRAME_TIME_RE.findall(contents)
    if len(frames) != 1 or len(frame_times) != 1:
        raise ValueError("BVH must contain exactly one Frames and Frame Time declaration")
    frame_count = int(frames[0])
    frame_time = float(frame_times[0])
    if frame_count < 1 or not np.isfinite(frame_time) or frame_time <= 0.0:
        raise ValueError("BVH frame metadata is invalid")
    fps = round(1.0 / frame_time)
    if fps != 120 or abs(frame_time - 1.0 / 120.0) > 1.0e-5:
        raise ValueError("approval BVH must be native 120 Hz")
    match = _FRAME_TIME_RE.search(contents)
    assert match is not None
    rows = [line for line in contents[match.end() :].splitlines() if line.strip()]
    if len(rows) != frame_count:
        raise ValueError(
            f"BVH declares {frame_count} frames but contains {len(rows)} motion rows"
        )
    for index, row in enumerate(rows):
        try:
            values = np.asarray([float(value) for value in row.split()])
        except ValueError as error:
            raise ValueError(f"BVH motion row {index} is not numeric") from error
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"BVH motion row {index} must be finite and nonempty")
    return frame_count, float(fps), len(rows)


def prepare_pfnn_bvh(source: Path, destination: Path) -> SourceReceipt:
    """Prepare a PFNN BVH for GMR without changing its timeline or motion rows."""

    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    contents = source.read_text()
    frame_count, fps, _ = _bvh_metadata(contents)
    if len(_SPINE1_RE.findall(contents)) != 1:
        raise ValueError("PFNN BVH must declare exactly one JOINT Spine1")
    if _SPINE2_RE.search(contents) is not None:
        raise ValueError("PFNN BVH must not already declare JOINT Spine2")
    if _LEFT_TOE_RE.search(contents) is None or _RIGHT_TOE_RE.search(contents) is None:
        raise ValueError("PFNN BVH must declare LeftToeBase and RightToeBase")
    prepared, substitutions = _SPINE1_RE.subn(r"\1Spine2\2", contents)
    if substitutions != 1:
        raise AssertionError("exact Spine1 alias substitution failed")
    _atomic_bytes(destination, prepared.encode())
    return SourceReceipt(
        source_sha256=_sha256(source),
        prepared_sha256=_sha256(destination),
        frame_count=frame_count,
        fps=fps,
        aliases=_ALIASES,
    )


def _field(motion: object, name: str) -> object:
    if isinstance(motion, Mapping):
        if name not in motion:
            raise ValueError(f"motion is missing {name}")
        return motion[name]
    if not hasattr(motion, name):
        raise ValueError(f"motion is missing {name}")
    return getattr(motion, name)


def validate_g1_motion(
    motion: object,
    *,
    expected_frames: int,
    expected_fps: float,
    joint_limits: np.ndarray,
) -> None:
    """Fail closed on malformed neutral G1 motion artifacts."""

    if type(expected_frames) is not int or expected_frames < 1:
        raise ValueError("expected_frames must be a positive integer")
    fps = float(_field(motion, "fps"))
    if not np.isfinite(fps) or fps != float(expected_fps):
        raise ValueError(f"motion fps must be exactly {float(expected_fps)}")
    root_pos = np.asarray(_field(motion, "root_pos"))
    root_quat = np.asarray(_field(motion, "root_quat"))
    dof = np.asarray(_field(motion, "dof"))
    expected = {
        "root_pos": (expected_frames, 3),
        "root_quat": (expected_frames, 4),
        "dof": (expected_frames, len(G1_JOINT_NAMES)),
    }
    for label, array in (
        ("root_pos", root_pos),
        ("root_quat", root_quat),
        ("dof", dof),
    ):
        if array.shape != expected[label]:
            raise ValueError(f"{label} must have shape {expected[label]}, got {array.shape}")
        if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError(f"{label} must contain finite real values")
    quaternion_norm = np.linalg.norm(root_quat.astype(np.float64), axis=1)
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-5, rtol=0.0):
        raise ValueError("root_quat must contain normalized XYZW quaternions")
    limits = np.asarray(joint_limits, dtype=np.float64)
    if limits.shape != (len(G1_JOINT_NAMES), 2) or not np.isfinite(limits).all():
        raise ValueError("joint_limits must have finite shape (29, 2)")
    if np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("joint limit lower bounds must not exceed upper bounds")
    positions = dof.astype(np.float64)
    if np.any(positions < limits[:, 0] - 1.0e-6) or np.any(
        positions > limits[:, 1] + 1.0e-6
    ):
        raise ValueError("motion contains a native G1 joint-limit violation")


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_external_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import external module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def g1_joint_limits(model: object) -> np.ndarray:
    """Return limited hinge ranges in qpos order, validating the 29-joint ABI."""

    import mujoco

    joints: list[tuple[int, int, str, np.ndarray]] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joints.append(
            (
                int(model.jnt_qposadr[joint_id]),
                joint_id,
                str(name),
                np.asarray(model.jnt_range[joint_id], dtype=np.float64).copy(),
            )
        )
    joints.sort(key=lambda item: item[0])
    names = tuple(item[2] for item in joints)
    if names != G1_JOINT_NAMES:
        raise ValueError(f"GMR G1 joint order does not match the native ABI: {names}")
    if any(not bool(model.jnt_limited[item[1]]) for item in joints):
        raise ValueError("all native G1 hinge joints must have finite ranges")
    return np.stack([item[3] for item in joints], axis=0)


def _save_motion(
    output: Path,
    motion: object,
    *,
    receipt: SourceReceipt,
    gmr_commit: str,
    retarget_project_commit: str,
    grounding_offset: float,
    joint_limits: np.ndarray,
) -> Path:
    root_pos = np.asarray(_field(motion, "root_pos"), dtype=np.float64)
    root_quat = np.asarray(_field(motion, "root_quat"), dtype=np.float64)
    dof = np.asarray(_field(motion, "dof"), dtype=np.float64)
    fps = float(_field(motion, "fps"))
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                root_pos=root_pos,
                root_quat=root_quat,
                dof=dof,
                fps=np.asarray(fps, dtype=np.float64),
                engine=np.asarray("gmr"),
                joint_names=np.asarray(G1_JOINT_NAMES),
                joint_limits=np.asarray(joint_limits, dtype=np.float64),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    output_sha256 = _sha256(output)
    receipt_path = output.with_suffix(".receipt.json")
    document = {
        "schema": "native-g1-pfnn-sample-retarget/v1",
        "status": "accepted",
        "source_sha256": receipt.source_sha256,
        "prepared_sha256": receipt.prepared_sha256,
        "output_sha256": output_sha256,
        "gmr_commit": gmr_commit,
        "retarget_project_commit": retarget_project_commit,
        "fps": fps,
        "frame_count": receipt.frame_count,
        "aliases": [list(alias) for alias in receipt.aliases],
        "joint_names": list(G1_JOINT_NAMES),
        "root_quaternion_order": "xyzw",
        "grounding_offset_m": float(grounding_offset),
    }
    _atomic_bytes(
        receipt_path,
        (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(),
    )
    return receipt_path


def retarget_sample(
    *,
    source: Path,
    gmr_root: Path,
    retarget_project_root: Path,
    output: Path,
) -> dict[str, object]:
    """Run pinned GMR sequentially on every native source frame."""

    gmr_root = Path(gmr_root).resolve(strict=True)
    retarget_project_root = Path(retarget_project_root).resolve(strict=True)
    gmr_commit = _git_commit(gmr_root)
    project_commit = _git_commit(retarget_project_root)
    if gmr_commit != GMR_COMMIT:
        raise ValueError(f"GMR must be pinned to {GMR_COMMIT}, got {gmr_commit}")
    if project_commit != RETARGET_PROJECT_COMMIT:
        raise ValueError(
            f"retargeting_project must be pinned to {RETARGET_PROJECT_COMMIT}, got {project_commit}"
        )
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=gmr_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ValueError("GMR checkout must be clean")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=retarget_project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ValueError("retargeting_project checkout must be clean")

    sys.path.insert(0, str(gmr_root))
    try:
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        with tempfile.TemporaryDirectory(prefix="pfnn-bvh-g1-") as directory:
            prepared = Path(directory) / "LocomotionFlat01_000-gmr.bvh"
            source_receipt = prepare_pfnn_bvh(source, prepared)
            frames, human_height = load_bvh_file(str(prepared), format="nokov")
            if len(frames) != source_receipt.frame_count:
                raise ValueError(
                    f"GMR loaded {len(frames)} frames, expected {source_receipt.frame_count}"
                )
            retargeter = GeneralMotionRetargeting(
                src_human="bvh_nokov",
                tgt_robot="unitree_g1",
                actual_human_height=human_height,
                verbose=False,
            )
            qpos = np.empty((len(frames), 36), dtype=np.float64)
            for index, frame in enumerate(frames):
                qpos[index] = retargeter.retarget(frame)
                if index == 0 or (index + 1) % 240 == 0 or index + 1 == len(frames):
                    print(
                        f"retargeted {index + 1}/{len(frames)} frames",
                        flush=True,
                    )
            limits = g1_joint_limits(retargeter.model)

            project_source = str(retarget_project_root / "src")
            sys.path.insert(0, project_source)
            try:
                common_motion = _load_external_module(
                    "common_motion",
                    retarget_project_root / "src" / "common_motion.py",
                )
                project_driver = _load_external_module(
                    "native_g1_retarget_project",
                    retarget_project_root / "retarget.py",
                )
            finally:
                if sys.path and sys.path[0] == project_source:
                    sys.path.pop(0)
            project_driver.GMR = str(gmr_root)
            motion = common_motion.CommonMotion(
                root_pos=qpos[:, :3],
                root_quat=qpos[:, 3:7][:, [1, 2, 3, 0]],
                dof=qpos[:, 7:],
                fps=source_receipt.fps,
                engine="gmr",
            )
            root_z_before = motion.root_pos[:, 2].copy()
            motion = project_driver.postprocess(motion)
            grounding_offset = float(
                np.median(root_z_before - np.asarray(motion.root_pos)[:, 2])
            )
            validate_g1_motion(
                motion,
                expected_frames=source_receipt.frame_count,
                expected_fps=source_receipt.fps,
                joint_limits=limits,
            )
            receipt_path = _save_motion(
                output,
                motion,
                receipt=source_receipt,
                gmr_commit=gmr_commit,
                retarget_project_commit=project_commit,
                grounding_offset=grounding_offset,
                joint_limits=limits,
            )
    finally:
        if sys.path and sys.path[0] == str(gmr_root):
            sys.path.pop(0)
    return {
        "status": "accepted",
        "output": str(Path(output).resolve()),
        "receipt": str(receipt_path.resolve()),
        "frame_count": source_receipt.frame_count,
        "fps": source_receipt.fps,
        "output_sha256": _sha256(Path(output)),
        "grounding_offset_m": grounding_offset,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--retarget-project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = retarget_sample(
        source=args.source,
        gmr_root=args.gmr_root,
        retarget_project_root=args.retarget_project_root,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
