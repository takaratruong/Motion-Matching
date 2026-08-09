import os
import hashlib
import json
import joblib
import numpy as np

from .schema import SourceClip


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < 1e-8):
        raise ValueError("zero-length root quaternion")
    q = q / n
    if not np.isfinite(q).all():
        raise ValueError("non-finite root quaternion")
    return q.astype(np.float32)


def load_takara(path: str, remap_path: str) -> SourceClip:
    data = np.load(path)
    remap = np.load(remap_path)
    joints = np.asarray(data["joint_pos"], np.float32)
    root = np.asarray(data["body_pos_w"][:, 0], np.float32)
    quat = _normalized_wxyz(data["body_quat_w"][:, 0])
    qpos = np.zeros((len(joints), 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = quat
    qpos[:, 7:] = joints[:, remap]
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    clip = SourceClip(
        "takara_walk_50hz", fps, qpos, np.arange(len(qpos)), "flat",
    )
    clip.validate()
    return clip


def load_grail(path: str) -> SourceClip:
    records = joblib.load(path)
    if len(records) != 1:
        raise ValueError(f"{path}: expected one robot record, got {len(records)}")
    record = next(iter(records.values()))
    dof = np.asarray(record["dof"], np.float32)
    root = np.asarray(record["root_trans_offset"], np.float32)
    xyzw = np.asarray(record["root_rot"], np.float32)
    quat = _normalized_wxyz(xyzw[:, [3, 0, 1, 2]])
    qpos = np.zeros((len(dof), 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = quat
    qpos[:, 7:] = dof
    fps = float(record["fps"])
    name = os.path.splitext(os.path.basename(path))[0]
    clip = SourceClip(name, fps, qpos, np.arange(len(qpos)), name)
    clip.validate()
    return clip


def load_retarget_npz(path: str, receipt_path: str) -> SourceClip:
    with open(receipt_path, encoding="utf-8") as stream:
        receipt = json.load(stream)
    required = {
        "schema": "native-g1-pfnn-sample-retarget/v1",
        "status": "accepted",
        "root_quaternion_order": "xyzw",
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"retarget receipt {key} is not {expected!r}")
    output_sha256 = _sha256(path)
    if receipt.get("output_sha256") != output_sha256:
        raise ValueError("retarget NPZ SHA-256 does not match receipt")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {
            "root_pos", "root_quat", "dof", "fps", "engine",
            "joint_names", "joint_limits",
        }:
            raise ValueError("retarget NPZ key set changed")
        root = np.asarray(data["root_pos"], np.float64)
        xyzw = np.asarray(data["root_quat"], np.float64)
        dof = np.asarray(data["dof"], np.float64)
        fps = float(np.asarray(data["fps"]).item())
        joint_names = [str(value) for value in data["joint_names"].tolist()]
        engine = str(np.asarray(data["engine"]).item())
        limits = np.asarray(data["joint_limits"], np.float64)
    frames = len(root)
    if root.shape != (frames, 3) or xyzw.shape != (frames, 4) \
            or dof.shape != (frames, 29) or limits.shape != (29, 2):
        raise ValueError("retarget NPZ array dimensions changed")
    if receipt.get("frame_count") != frames or receipt.get("fps") != fps:
        raise ValueError("retarget receipt rate/frame count changed")
    if receipt.get("joint_names") != joint_names:
        raise ValueError("retarget receipt joint names changed")
    if engine != "gmr":
        raise ValueError("retarget engine identity changed")
    qpos = np.empty((frames, 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = _normalized_wxyz(xyzw[:, [3, 0, 1, 2]])
    qpos[:, 7:] = dof
    clip = SourceClip(
        os.path.splitext(os.path.basename(path))[0], fps, qpos,
        np.arange(frames, dtype=np.int32), "flat",
        provenance={
            "path": os.path.abspath(path),
            "sha256": output_sha256,
            "receipt_path": os.path.abspath(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
            "receipt": receipt,
        },
    )
    clip.validate()
    return clip


load_released_pfnn_retarget = load_retarget_npz
