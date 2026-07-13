import numpy as np


def output_frame_count(frames: int, source_fps: float, target_fps: float) -> int:
    if frames < 1 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("frames and sample rates must be positive")
    duration = (frames - 1) / source_fps
    return int(np.floor(duration * target_fps + 1e-9)) + 1


def _times(frames: int, fps: float) -> np.ndarray:
    return np.arange(frames, dtype=np.float64) / fps


def resample_vectors(
    values: np.ndarray, source_fps: float, target_fps: float,
) -> np.ndarray:
    values = np.asarray(values, np.float64)
    output_frame_count(len(values), source_fps, target_fps)
    if not np.all(np.isfinite(values)):
        raise ValueError("vector samples must be finite")
    src_t = _times(len(values), source_fps)
    count = output_frame_count(len(values), source_fps, target_fps)
    dst_t = _times(count, target_fps)
    flat = values.reshape(len(values), -1)
    out = np.stack([
        np.interp(dst_t, src_t, flat[:, i])
        for i in range(flat.shape[1])
    ], axis=1)
    return out.reshape((count,) + values.shape[1:])


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
    src_t = _times(len(q), source_fps)
    count = output_frame_count(len(q), source_fps, target_fps)
    dst_t = _times(count, target_fps)
    out = np.empty((count, flat.shape[1], 4), np.float64)
    for j in range(flat.shape[1]):
        for k, t in enumerate(dst_t):
            hi = min(
                np.searchsorted(src_t, t, side="right"), len(src_t) - 1)
            lo = max(0, hi - 1)
            u = (
                0.0 if hi == lo
                else (t - src_t[lo]) / (src_t[hi] - src_t[lo])
            )
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
