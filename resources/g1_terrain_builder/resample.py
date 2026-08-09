import numpy as np


def output_frame_count(frames: int, source_fps: float, target_fps: float) -> int:
    if frames < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("frames and sample rates must be positive")
    duration = (frames - 1) / source_fps
    return int(np.floor(duration * target_fps + 1e-9)) + 1


def _times(frames: int, fps: float) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / fps


def resample_map(
    frames: int, source_fps: float, target_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = output_frame_count(frames, source_fps, target_fps)
    source_u = np.arange(count, dtype=np.float64) * source_fps / target_fps
    left = np.floor(source_u).astype(np.int64)
    left = np.clip(left, 0, frames - 1)
    right = np.minimum(left + 1, frames - 1)
    alpha = (source_u - left).astype(np.float32)
    exact = (alpha == 0.0) | (left == right)
    right[exact] = left[exact]
    alpha[exact] = 0.0
    return left.astype(np.int32), right.astype(np.int32), alpha


def resample_vectors(
    values: np.ndarray, source_fps: float, target_fps: float,
) -> np.ndarray:
    values = np.asarray(values, np.float64)
    output_frame_count(len(values), source_fps, target_fps)
    if not np.all(np.isfinite(values)):
        raise ValueError("vector samples must be finite")
    left, right, alpha = resample_map(
        len(values), source_fps, target_fps)
    shape = (len(alpha),) + (1,) * (values.ndim - 1)
    blend = alpha.astype(np.float64).reshape(shape)
    return values[left] * (1.0 - blend) + values[right] * blend


def resample_quaternions_wxyz(
    values: np.ndarray, source_fps: float, target_fps: float,
) -> np.ndarray:
    q = np.asarray(values, np.float64).copy()
    output_frame_count(len(q), source_fps, target_fps)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(q)) or np.any(norms < 1e-12):
        raise ValueError("quaternion samples must be finite and nonzero")
    q /= norms
    flat = q.reshape(len(q), -1, 4)
    for t in range(1, len(flat)):
        signs = np.sum(flat[t - 1] * flat[t], axis=-1) < 0
        flat[t, signs] *= -1
    left, right, alpha = resample_map(len(q), source_fps, target_fps)
    count = len(left)
    out = np.empty((count, flat.shape[1], 4), np.float64)
    for j in range(flat.shape[1]):
        for k, (lo, hi, u) in enumerate(zip(left, right, alpha)):
            a, b = flat[lo, j], flat[hi, j]
            dot = np.clip(np.dot(a, b), -1.0, 1.0)
            if dot > 0.9995:
                x = a + u * (b - a)
            else:
                theta = np.arccos(dot)
                x = (
                    np.sin((1 - u) * theta) * a
                    + np.sin(u * theta) * b
                ) / np.sin(theta)
            out[k, j] = x / np.linalg.norm(x)
    return out.reshape((count,) + q.shape[1:])
