from dataclasses import dataclass
import os
import joblib
import numpy as np

from .schema import SourceClip


GRAIL_OBJECT_POSITION_STATIC_ATOL_M = 1e-5
GRAIL_OBJECT_QUATERNION_STATIC_ATOL = 1e-5
GRAIL_OBJECT_QUATERNION_NORM_ATOL = 1e-5


@dataclass(frozen=True)
class GrailObjectPose:
    root_pos: np.ndarray
    root_quat: np.ndarray
    fps: float
    scale: np.ndarray


def _grail_real_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) \
            or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"GRAIL object {name} must be a real numeric array")
    array = np.array(array, np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"GRAIL object {name} must contain only finite values")
    return array


def load_grail_object_pose(path: str) -> GrailObjectPose:
    records = joblib.load(path)
    if type(records) is not dict or len(records) != 1:
        count = len(records) if isinstance(records, dict) else "non-dict"
        raise ValueError(f"{path}: expected one object record, got {count}")
    record = next(iter(records.values()))
    if type(record) is not dict:
        raise TypeError(f"{path}: object record must be a dictionary")
    for name in ("root_pos", "root_quat", "fps", "scale"):
        if name not in record:
            raise ValueError(f"{path}: object record is missing {name}")

    root_pos = _grail_real_array(record["root_pos"], "root_pos")
    root_quat_xyzw = _grail_real_array(
        record["root_quat"], "root_quat")
    scale = _grail_real_array(record["scale"], "scale")
    if root_pos.ndim != 3 or root_pos.shape[1:] != (1, 3) \
            or not len(root_pos):
        raise ValueError(
            f"{path}: root_pos must have non-empty shape (T, 1, 3), "
            f"got {root_pos.shape}")
    if root_quat_xyzw.shape != (len(root_pos), 1, 4):
        raise ValueError(
            f"{path}: root_quat must have shape ({len(root_pos)}, 1, 4), "
            f"got {root_quat_xyzw.shape}")
    if scale.shape != (3, 1):
        raise ValueError(f"{path}: scale must have shape (3, 1), got {scale.shape}")
    if np.any(scale <= 0.0):
        raise ValueError(f"{path}: scale must be finite and positive")

    fps_value = np.asarray(record["fps"])
    if fps_value.shape != () or not np.issubdtype(fps_value.dtype, np.number) \
            or np.issubdtype(fps_value.dtype, np.complexfloating):
        raise TypeError(f"{path}: fps must be a real numeric scalar")
    fps = float(fps_value)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: fps must be finite and positive")

    norms = np.linalg.norm(root_quat_xyzw[:, 0], axis=1)
    if not np.allclose(
            norms, 1.0, rtol=0.0,
            atol=GRAIL_OBJECT_QUATERNION_NORM_ATOL):
        raise ValueError(f"{path}: root_quat must contain normalized quaternions")
    if np.max(np.abs(root_pos[:, 0] - root_pos[0, 0])) \
            > GRAIL_OBJECT_POSITION_STATIC_ATOL_M:
        raise ValueError(f"{path}: terrain object root_pos must be static")
    quaternions = root_quat_xyzw[:, 0] / norms[:, None]
    signs = np.where(
        np.sum(quaternions * quaternions[0], axis=1, keepdims=True) < 0.0,
        -1.0,
        1.0,
    )
    aligned = quaternions * signs
    if np.max(np.abs(aligned - aligned[0])) \
            > GRAIL_OBJECT_QUATERNION_STATIC_ATOL:
        raise ValueError(f"{path}: terrain object root_quat must be static")

    position = np.array(root_pos[0, 0], np.float64, copy=True)
    quaternion_wxyz = np.array(
        aligned[0, [3, 0, 1, 2]], np.float64, copy=True)
    object_scale = np.array(scale[:, 0], np.float64, copy=True)
    for array in (position, quaternion_wxyz, object_scale):
        array.setflags(write=False)
    return GrailObjectPose(position, quaternion_wxyz, fps, object_scale)


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
