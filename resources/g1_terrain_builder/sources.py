import os
import hashlib
import io
import json
import joblib
import numpy as np
import re

from .schema import SourceClip


RETARGET_RECEIPT_FIELDS = frozenset({
    "schema", "status", "source_sha256", "prepared_sha256",
    "output_sha256", "gmr_commit", "retarget_project_commit", "fps",
    "source_frame_count", "start_frame", "frame_count", "warmup_frames",
    "aliases", "joint_names", "root_quaternion_order",
    "grounding_offset_m", "grounding", "pfnn_position_scale",
})
RETARGET_GMR_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
RETARGET_PROJECT_COMMIT = "fb3433a6310ab4198102d3905e74b73944fc1f6b"
RETARGET_ALIASES = [["Spine1", "Spine2"]]
PFNN_POSITION_SCALE = 5.6444
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
AUTHORED_SLOPE_NAME = "terrain_slopes__slope_000__000"
AUTHORED_SLOPE_ROBOT_SHA256 = (
    "b77480d5f8f3339a3064276d6f9d443ac3a3456f20eb9195d48add176e561ee1")
AUTHORED_SLOPE_ROBOT_SIZE_BYTES = 198603


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


def _grail_source_clip(records, path: str) -> SourceClip:
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


def load_grail(path: str) -> SourceClip:
    return _grail_source_clip(joblib.load(path), path)


def load_authenticated_grail_slope(path: str) -> SourceClip:
    try:
        with open(path, "rb") as stream:
            payload = stream.read(AUTHORED_SLOPE_ROBOT_SIZE_BYTES + 1)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"authored slope robot authentication failed: {error}") from error
    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != AUTHORED_SLOPE_ROBOT_SIZE_BYTES \
            or digest != AUTHORED_SLOPE_ROBOT_SHA256:
        raise ValueError("authored slope robot content SHA-256/size changed")
    if os.path.splitext(os.path.basename(path))[0] != AUTHORED_SLOPE_NAME:
        raise ValueError("authored slope robot basename changed")
    clip = _grail_source_clip(joblib.load(io.BytesIO(payload)), path)
    if clip.name != AUTHORED_SLOPE_NAME or clip.terrain_id != AUTHORED_SLOPE_NAME \
            or clip.fps != 25.0 or clip.qpos.shape != (250, 36) \
            or not np.array_equal(
                clip.source_frames, np.arange(250, dtype=np.int32)):
        raise ValueError("authored slope robot identity/rate/frame count changed")
    clip.provenance = {
        "path": os.path.abspath(path),
        "sha256": digest,
        "size_bytes": len(payload),
    }
    return clip


def load_retarget_npz(path: str, receipt_path: str) -> SourceClip:
    with open(receipt_path, encoding="utf-8") as stream:
        receipt = json.load(stream)
    if type(receipt) is not dict or set(receipt) != RETARGET_RECEIPT_FIELDS:
        raise ValueError("retarget receipt fields changed")
    required = {
        "schema": "native-g1-pfnn-sample-retarget/v1",
        "status": "accepted",
        "root_quaternion_order": "xyzw",
        "gmr_commit": RETARGET_GMR_COMMIT,
        "retarget_project_commit": RETARGET_PROJECT_COMMIT,
        "aliases": RETARGET_ALIASES,
        "grounding": "flat",
        "pfnn_position_scale": PFNN_POSITION_SCALE,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"retarget receipt {key} is not {expected!r}")
    for key in ("source_sha256", "prepared_sha256", "output_sha256"):
        value = receipt[key]
        if type(value) is not str or not _HEX_SHA256.fullmatch(value):
            raise ValueError(f"retarget receipt {key} is not a SHA-256")
    for key in (
        "source_frame_count", "start_frame", "frame_count", "warmup_frames",
    ):
        if type(receipt[key]) is not int or receipt[key] < 0:
            raise ValueError(f"retarget receipt {key} is invalid")
    source_frames = receipt["source_frame_count"]
    start_frame = receipt["start_frame"]
    frames = receipt["frame_count"]
    warmup_frames = receipt["warmup_frames"]
    if frames < 1 or source_frames < start_frame + frames \
            or warmup_frames > start_frame \
            or source_frames > np.iinfo(np.int32).max:
        raise ValueError("retarget receipt source interval is invalid")
    if type(receipt["fps"]) is not float or receipt["fps"] != 120.0:
        raise ValueError("retarget receipt fps is not 120.0")
    grounding_offset = receipt["grounding_offset_m"]
    if type(grounding_offset) is not float \
            or not np.isfinite(grounding_offset):
        raise ValueError("retarget receipt grounding_offset_m is invalid")
    if type(receipt["joint_names"]) is not list \
            or len(receipt["joint_names"]) != 29 \
            or any(type(name) is not str or not name
                   for name in receipt["joint_names"]):
        raise ValueError("retarget receipt joint names are invalid")
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
    artifact_frames = len(root)
    if root.shape != (artifact_frames, 3) \
            or xyzw.shape != (artifact_frames, 4) \
            or dof.shape != (artifact_frames, 29) \
            or limits.shape != (29, 2):
        raise ValueError("retarget NPZ array dimensions changed")
    if frames != artifact_frames or receipt["fps"] != fps:
        raise ValueError("retarget receipt rate/frame count changed")
    if receipt.get("joint_names") != joint_names:
        raise ValueError("retarget receipt joint names changed")
    if engine != "gmr":
        raise ValueError("retarget engine identity changed")
    if not all(np.isfinite(value).all()
               for value in (root, xyzw, dof, limits)):
        raise ValueError("retarget NPZ contains non-finite values")
    if np.any(limits[:, 0] > limits[:, 1]) \
            or np.any(dof < limits[:, 0] - 1e-6) \
            or np.any(dof > limits[:, 1] + 1e-6):
        raise ValueError("retarget NPZ contains a joint-limit violation")
    qpos = np.empty((artifact_frames, 36), np.float32)
    qpos[:, :3] = root
    qpos[:, 3:7] = _normalized_wxyz(xyzw[:, [3, 0, 1, 2]])
    qpos[:, 7:] = dof
    clip = SourceClip(
        os.path.splitext(os.path.basename(path))[0], fps, qpos,
        np.arange(
            start_frame, start_frame + artifact_frames, dtype=np.int32),
        "flat",
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
