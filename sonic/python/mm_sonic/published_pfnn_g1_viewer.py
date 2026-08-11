#!/usr/bin/env python3
"""Temporary live Holden-PFNN globals -> pinned-GMR -> MuJoCo G1 viewer.

Accepted inputs:
  * PFNNXFM v1 binary stream (the released C++ publisher contract)
  * JSONL source frames with ``positions[31,3]`` and ``rotations[31,4]``
  * JSONL pre-retarget frames with ``qpos_wxyz[36]`` and
    ``source_root_pos_zup_m[3]``
  * NPZ recordings with ``positions[T,31,3]`` and ``rotations[T,31,4]``

This is deliberately a visual bridge only: no controller, foot lock, filtering,
root correction, or post-retarget adjustment is performed.  The only root
override is the published source Hips translation, preserving source authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import queue
import struct
import sys
import threading
import time
from typing import BinaryIO, Iterator

import numpy as np


GMR_ROOT_DEFAULT = Path("/home/ubuntu/.cache/native-g1-pfnn/GMR")
GMR_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
DEFAULT_SCENE = GMR_ROOT_DEFAULT / "assets/unitree_g1/g1_mocap_29dof.xml"
# The live exporter emits the post-pre_render global skeleton in centimetres.
# The historical PFNN BVH writer's separate 5.6444 scale must not be applied a
# second time here.
SOURCE_POSITION_SCALE_M = 1.0 / 100.0
PFNNXFM_HEADER = struct.Struct("<8sIIIIfI")
PFNNXFM_PREFIX = struct.Struct("<QIIff3f2f")
PFNNXFM_MAGIC = b"PFNNXFM\0"
PFNNXFM_JOINTS = 31
PFNNXFM_RECORD_BYTES = 912

BONES = (
    "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot",
    "LeftToeBase", "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot",
    "RightToeBase", "LowerBack", "Spine", "Spine2", "Neck", "Neck1",
    "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftFingerBase", "LeftHandIndex1", "LThumb", "RightShoulder",
    "RightArm", "RightForeArm", "RightHand", "RightFingerBase",
    "RightHandIndex1", "RThumb",
)


@dataclass
class Frame:
    positions: np.ndarray | None = None
    rotations_wxyz: np.ndarray | None = None
    qpos_wxyz: np.ndarray | None = None
    source_root_pos_zup_m: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    input_space: str = "source"


def _exact_read(stream: BinaryIO, size: int, *, allow_eof: bool = False) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_eof and remaining == size:
                return None
            raise EOFError(f"truncated stream: wanted {size} bytes, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_pose_arrays(positions: object, rotations: object) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions, dtype=np.float64)
    quat = np.asarray(rotations, dtype=np.float64)
    if pos.shape != (PFNNXFM_JOINTS, 3):
        raise ValueError(f"positions must have shape (31,3), got {pos.shape}")
    if quat.shape != (PFNNXFM_JOINTS, 4):
        raise ValueError(f"rotations must have shape (31,4), got {quat.shape}")
    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("source frame contains non-finite values")
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1.0e-8):
        raise ValueError("source frame contains a zero quaternion")
    return pos, quat / norms[:, None]


def _frame_from_json(document: object, default_space: str) -> Frame | None:
    if not isinstance(document, dict):
        raise ValueError("each JSONL row must be an object")
    if document.get("type") == "meta":
        return None
    metadata = {
        key: document[key]
        for key in ("frame", "world", "phase", "flags", "fps")
        if key in document
    }
    if "qpos_wxyz" in document:
        qpos = np.asarray(document["qpos_wxyz"], dtype=np.float64)
        root = np.asarray(document.get("source_root_pos_zup_m", qpos[:3]), dtype=np.float64)
        if qpos.shape != (36,) or root.shape != (3,):
            raise ValueError("qpos_wxyz/source_root_pos_zup_m must have shapes (36,)/(3,)")
        if not np.isfinite(qpos).all() or not np.isfinite(root).all():
            raise ValueError("qpos JSONL row contains non-finite values")
        return Frame(qpos_wxyz=qpos, source_root_pos_zup_m=root, metadata=metadata,
                     input_space="gmr")
    positions = document.get("positions", document.get("global_positions"))
    rotations = document.get(
        "rotations", document.get("quaternions", document.get("global_quaternions"))
    )
    if positions is None or rotations is None:
        raise ValueError("JSONL row needs qpos_wxyz or positions+rotations")
    pos, quat = _validate_pose_arrays(positions, rotations)
    return Frame(positions=pos, rotations_wxyz=quat, metadata=metadata,
                 input_space=str(document.get("input_space", default_space)))


def _jsonl_frames(stream: BinaryIO, first: bytes = b"", *, input_space: str) -> Iterator[Frame]:
    text_stream = io.TextIOWrapper(io.BufferedReader(_PrefixReader(first, stream)), encoding="utf-8")
    for line_number, line in enumerate(text_stream, 1):
        if not line.strip():
            continue
        # Pinned GMR prints startup banners even with verbose=False.  Permit an
        # upstream retarget bridge to share stdout without treating those
        # human-readable banner lines as frame records.
        if not line.lstrip().startswith("{"):
            print(f"skipping non-JSON input line {line_number}: {line.strip()[:120]}",
                  file=sys.stderr, flush=True)
            continue
        try:
            frame = _frame_from_json(json.loads(line), input_space)
        except Exception as error:
            raise ValueError(f"invalid JSONL frame {line_number}: {error}") from error
        if frame is not None:
            yield frame


class _PrefixReader(io.RawIOBase):
    def __init__(self, prefix: bytes, source: BinaryIO) -> None:
        self._prefix = io.BytesIO(prefix)
        self._source = source

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        chunk = self._prefix.read(len(buffer))
        if not chunk:
            chunk = self._source.read(len(buffer))
        count = len(chunk)
        buffer[:count] = chunk
        return count


def _pfnnxfm_frames(stream: BinaryIO, header_bytes: bytes | None = None) -> Iterator[Frame]:
    raw_header = header_bytes if header_bytes is not None else _exact_read(stream, PFNNXFM_HEADER.size)
    assert raw_header is not None
    magic, version, joints, record_bytes, coord, fps, _reserved = PFNNXFM_HEADER.unpack(raw_header)
    if (magic, version, joints, record_bytes, coord) != (
        PFNNXFM_MAGIC, 1, PFNNXFM_JOINTS, PFNNXFM_RECORD_BYTES, 1
    ):
        raise ValueError(
            "unsupported PFNNXFM header: "
            f"magic={magic!r} version={version} joints={joints} "
            f"record_bytes={record_bytes} coord={coord}"
        )
    while True:
        raw = _exact_read(stream, record_bytes, allow_eof=True)
        if raw is None:
            return
        prefix = PFNNXFM_PREFIX.unpack_from(raw, 0)
        poses = np.frombuffer(raw, dtype="<f4", count=joints * 7,
                              offset=PFNNXFM_PREFIX.size).reshape(joints, 7)
        pos, quat = _validate_pose_arrays(poses[:, :3], poses[:, 3:])
        yield Frame(
            positions=pos,
            rotations_wxyz=quat,
            input_space="source",
            metadata={
                "frame": int(prefix[0]), "world": int(prefix[1]),
                "flags": int(prefix[2]), "phase": float(prefix[3]),
                "root_height_cm": float(prefix[4]),
                "trajectory_root": list(prefix[5:8]),
                "trajectory_forward_xz": list(prefix[8:10]),
                "fps": float(fps),
            },
        )


def _raw_f32_frames(stream: BinaryIO, *, input_space: str) -> Iterator[Frame]:
    record_bytes = PFNNXFM_JOINTS * 7 * 4
    index = 0
    while True:
        raw = _exact_read(stream, record_bytes, allow_eof=True)
        if raw is None:
            return
        poses = np.frombuffer(raw, dtype="<f4").reshape(PFNNXFM_JOINTS, 7)
        pos, quat = _validate_pose_arrays(poses[:, :3], poses[:, 3:])
        yield Frame(positions=pos, rotations_wxyz=quat,
                    input_space=input_space, metadata={"frame": index})
        index += 1


def _npz_frames(path: Path, *, input_space: str) -> Iterator[Frame]:
    with np.load(path, allow_pickle=False) as archive:
        if "qpos_wxyz" in archive.files:
            qpos = np.asarray(archive["qpos_wxyz"], dtype=np.float64)
            roots = np.asarray(archive.get("source_root_pos_zup_m", qpos[:, :3]), dtype=np.float64)
            if qpos.ndim != 2 or qpos.shape[1] != 36 or roots.shape != (len(qpos), 3):
                raise ValueError("NPZ qpos/root arrays must have shapes [T,36]/[T,3]")
            for index in range(len(qpos)):
                yield Frame(qpos_wxyz=qpos[index], source_root_pos_zup_m=roots[index],
                            input_space="gmr", metadata={"frame": index})
            return
        positions = np.asarray(archive["positions"], dtype=np.float64)
        key = "rotations" if "rotations" in archive.files else "quaternions"
        rotations = np.asarray(archive[key], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (31, 3):
            raise ValueError("NPZ positions must have shape [T,31,3]")
        if rotations.shape != (len(positions), 31, 4):
            raise ValueError("NPZ rotations must have shape [T,31,4]")
        for index in range(len(positions)):
            pos, quat = _validate_pose_arrays(positions[index], rotations[index])
            yield Frame(positions=pos, rotations_wxyz=quat, input_space=input_space,
                        metadata={"frame": index})


def _open_frames(path: str, format_name: str, *, input_space: str) -> Iterator[Frame]:
    if path != "-" and (format_name == "npz" or (format_name == "auto" and path.endswith(".npz"))):
        yield from _npz_frames(Path(path).resolve(strict=True), input_space=input_space)
        return
    stream = sys.stdin.buffer if path == "-" else Path(path).resolve(strict=True).open("rb")
    close = path != "-"
    try:
        if format_name == "pfnnxfm":
            yield from _pfnnxfm_frames(stream)
        elif format_name == "f32":
            yield from _raw_f32_frames(stream, input_space=input_space)
        elif format_name == "jsonl":
            yield from _jsonl_frames(stream, input_space=input_space)
        elif format_name == "auto":
            prefix = _exact_read(stream, PFNNXFM_HEADER.size, allow_eof=True)
            if prefix is None:
                return
            if prefix[:8] == PFNNXFM_MAGIC:
                yield from _pfnnxfm_frames(stream, prefix)
            else:
                yield from _jsonl_frames(stream, prefix, input_space=input_space)
        else:
            raise ValueError(f"unsupported format {format_name!r}")
    finally:
        if close:
            stream.close()


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), axis=-1)


def to_gmr_frame(
    frame: Frame, *, source_position_scale_m: float = SOURCE_POSITION_SCALE_M
) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    assert frame.positions is not None and frame.rotations_wxyz is not None
    if frame.input_space == "source":
        pos = frame.positions[:, (0, 2, 1)].copy()
        pos[:, 1] *= -1.0
        pos *= source_position_scale_m
        qconv = np.asarray((np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0))
        quat = _quat_multiply_wxyz(np.broadcast_to(qconv, frame.rotations_wxyz.shape),
                                  frame.rotations_wxyz)
    elif frame.input_space == "gmr":
        pos = frame.positions.copy()
        quat = frame.rotations_wxyz.copy()
    else:
        raise ValueError(f"input_space must be source or gmr, got {frame.input_space!r}")
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    human = {name: [pos[index].copy(), quat[index].copy()]
             for index, name in enumerate(BONES)}
    human["LeftFootMod"] = [human["LeftFoot"][0].copy(), human["LeftToeBase"][1].copy()]
    human["RightFootMod"] = [human["RightFoot"][0].copy(), human["RightToeBase"][1].copy()]
    return human, pos[0].copy()


def _gmr_joint_qpos_addresses(model: object) -> dict[str, int]:
    import mujoco
    result: dict[str, int] = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            result[name] = int(model.jnt_qposadr[joint_id])
    return result


def _load_model(scene_xml: Path, terrain_npz: Path | None, hide_floor: bool) -> object:
    import mujoco
    scene_xml = scene_xml.resolve(strict=True)
    if terrain_npz is None:
        model = mujoco.MjModel.from_xml_path(str(scene_xml))
    else:
        with np.load(terrain_npz.resolve(strict=True), allow_pickle=False) as archive:
            vertices = np.asarray(archive["vertices"], dtype=np.float64)
            faces = np.asarray(archive["faces"], dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("terrain NPZ needs vertices[N,3] and faces[M,3]")
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        spec.add_mesh(name="live_source_terrain", uservert=vertices.ravel(),
                      userface=faces.ravel(),
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
                      smoothnormal=1)
        spec.worldbody.add_geom(name="live_source_terrain", type=mujoco.mjtGeom.mjGEOM_MESH,
                                meshname="live_source_terrain", contype=0, conaffinity=0,
                                rgba=(0.20, 0.42, 0.20, 1.0))
        model = spec.compile()
    if hide_floor:
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id >= 0:
            model.geom_rgba[floor_id, 3] = 0.0
    return model


class SceneViewer:
    def __init__(self, model: object, gmr_model: object, *, title_fps: float) -> None:
        import mujoco
        import mujoco.viewer
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.closed = False

        def on_key(key: int) -> None:
            if key == 256:  # GLFW Escape
                self.closed = True

        self.viewer = mujoco.viewer.launch_passive(
            model=model, data=self.data, show_left_ui=False, show_right_ui=False,
            key_callback=on_key,
        )
        source = _gmr_joint_qpos_addresses(gmr_model)
        target = _gmr_joint_qpos_addresses(model)
        self.joint_map = tuple((source[name], target[name]) for name in source if name in target)
        if len(self.joint_map) != len(source):
            missing = sorted(set(source) - set(target))
            raise ValueError(f"scene is missing GMR G1 joints: {missing}")
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError(f"scene must contain exactly one free robot joint, got {len(free)}")
        self.root_qpos_adr = int(model.jnt_qposadr[int(free[0])])
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id < 0:
            self.pelvis_id = 1
        self.fps = title_fps

    def running(self) -> bool:
        return not self.closed and self.viewer.is_running()

    def render(self, qpos: np.ndarray, source_root: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        root = self.root_qpos_adr
        self.data.qpos[root:root + 3] = source_root
        self.data.qpos[root + 3:root + 7] = qpos[3:7]
        for source_adr, target_adr in self.joint_map:
            self.data.qpos[target_adr] = qpos[source_adr]
        self.mujoco.mj_forward(self.model, self.data)
        self.viewer.cam.lookat = self.data.xpos[self.pelvis_id]
        self.viewer.cam.distance = 2.6
        self.viewer.cam.elevation = -14
        self.viewer.sync()

    def idle(self) -> None:
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer.is_running():
            self.viewer.close()


def _git_head(path: Path) -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


def _load_retargeter(gmr_root: Path):
    root = gmr_root.resolve(strict=True)
    if _git_head(root) != GMR_COMMIT:
        raise ValueError(f"GMR checkout is not pinned to {GMR_COMMIT}")
    sys.path.insert(0, str(root))
    from general_motion_retargeting import GeneralMotionRetargeting
    return GeneralMotionRetargeting(src_human="bvh_nokov", tgt_robot="unitree_g1",
                                    actual_human_height=1.75, verbose=False)


def _put_frame(frame: Frame, output: queue.Queue[object]) -> None:
    if frame.qpos_wxyz is None:
        output.put(frame)
        return

    while True:
        try:
            output.put_nowait(frame)
            return
        except queue.Full:
            try:
                stale = output.get_nowait()
            except queue.Empty:
                continue
            if isinstance(stale, Frame) and stale.qpos_wxyz is not None:
                continue
            output.put(stale)
            output.put(frame)
            return


def _producer(iterator: Iterator[Frame], output: queue.Queue[object]) -> None:
    try:
        for frame in iterator:
            _put_frame(frame, output)
    except BaseException as error:
        output.put(error)
    finally:
        output.put(None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="-", help="input path, or - for stdin")
    parser.add_argument("--format", choices=("auto", "pfnnxfm", "jsonl", "npz", "f32"),
                        default="auto")
    parser.add_argument("--input-space", choices=("source", "gmr"), default="source",
                        help="source=y-up PFNN units; gmr=z-up metres (JSONL/raw/NPZ)")
    parser.add_argument("--source-position-scale-m", type=float,
                        default=SOURCE_POSITION_SCALE_M,
                        help="metres per live source position unit (default: exporter cm -> m)")
    parser.add_argument("--gmr-root", type=Path, default=GMR_ROOT_DEFAULT)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--terrain-npz", type=Path,
                        help="optional exact terrain mesh with vertices/faces arrays")
    parser.add_argument("--hide-floor", action="store_true")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 means consume through EOF")
    parser.add_argument("--trace-every", type=int, default=60)
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()
    if (not np.isfinite(args.fps) or args.fps <= 0 or args.max_frames < 0
            or args.trace_every < 1 or not np.isfinite(args.source_position_scale_m)
            or args.source_position_scale_m <= 0):
        parser.error("fps/trace-every must be positive and max-frames nonnegative")

    frames = _open_frames(args.input, args.format, input_space=args.input_space)
    retargeter = None
    model = None
    viewer = None
    if not args.no_viewer:
        model = _load_model(args.scene_xml, args.terrain_npz, args.hide_floor)

    incoming: queue.Queue[object] = queue.Queue(maxsize=1)
    thread = threading.Thread(target=_producer, args=(frames, incoming), daemon=True)
    thread.start()
    count = 0
    eof = False
    last_status = time.monotonic()
    next_deadline = time.monotonic()
    print(json.dumps({"event": "ready", "input": args.input, "format": args.format,
                      "scene_xml": str(args.scene_xml.resolve()), "fps": args.fps}), flush=True)
    try:
        while not eof and (args.max_frames == 0 or count < args.max_frames):
            try:
                item = incoming.get(timeout=0.01)
            except queue.Empty:
                if viewer is not None:
                    viewer.idle()
                    if not viewer.running():
                        break
                continue
            if item is None:
                eof = True
                continue
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, Frame)
            frame = item
            if frame.qpos_wxyz is not None:
                qpos = frame.qpos_wxyz.copy()
                source_root = frame.source_root_pos_zup_m.copy()
            else:
                human, source_root = to_gmr_frame(
                    frame, source_position_scale_m=args.source_position_scale_m
                )
                if retargeter is None:
                    retargeter = _load_retargeter(args.gmr_root)
                qpos = np.asarray(retargeter.retarget(human), dtype=np.float64)
            if qpos.shape != (36,) or not np.isfinite(qpos).all():
                raise ValueError(f"retargeter returned invalid qpos shape/data: {qpos.shape}")
            if viewer is None and not args.no_viewer:
                assert model is not None
                if retargeter is None:
                    retargeter = _load_retargeter(args.gmr_root)
                viewer = SceneViewer(model, retargeter.model, title_fps=args.fps)
                print("viewer controls: Escape closes", flush=True)
            if viewer is not None:
                viewer.render(qpos, source_root)
                if not viewer.running():
                    break
            count += 1
            if count == 1 or count % args.trace_every == 0:
                meta = " ".join(f"{key}={value}" for key, value in frame.metadata.items()
                                    if key in ("frame", "world", "phase"))
                print(f"frame_count={count} root=({source_root[0]:+.3f},"
                      f"{source_root[1]:+.3f},{source_root[2]:+.3f}) {meta}", flush=True)
            if args.input != "-" and viewer is not None:
                next_deadline += 1.0 / args.fps
                delay = next_deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_deadline = time.monotonic()
            now = time.monotonic()
            if now - last_status >= 5.0:
                print(f"alive frames={count} queue={incoming.qsize()}", flush=True)
                last_status = now
    finally:
        if viewer is not None:
            viewer.close()
    print(json.dumps({"event": "eof", "frames": count}), flush=True)


if __name__ == "__main__":
    main()
