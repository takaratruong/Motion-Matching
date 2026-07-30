"""Synthetic native Takara motion fixtures for the torch motion loader tests.

The fixture writes one finite native Z-up / wxyz G1 clip that satisfies the
frozen ``g1-29dof-isaaclab-v1`` export contract: 29 joints, 30 bodies, pelvis
body index ``0``, feet body indices ``18``/``19``, and unit wxyz quaternions
with analytically consistent linear and angular body velocities.
"""

from pathlib import Path

import numpy as np

TAKARA_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

_JOINT_COUNT = 29
_BODY_COUNT = 30


def write_takara_clip(
    path: Path,
    *,
    frames: int = 60,
    fps: int = 50,
    yaw_rate: float = 0.0,
    root_velocity_xy: tuple[float, float] = (0.4, 0.0),
) -> Path:
    """Write one finite native Z-up/wxyz G1 NPZ and return motion.npz."""
    arrays = build_takara_arrays(
        frames=frames,
        fps=fps,
        yaw_rate=yaw_rate,
        root_velocity_xy=root_velocity_xy,
    )
    return write_takara_arrays(path, arrays)


def build_takara_arrays(
    *,
    frames: int = 60,
    fps: int = 50,
    yaw_rate: float = 0.0,
    root_velocity_xy: tuple[float, float] = (0.4, 0.0),
) -> dict[str, np.ndarray]:
    """Return the native float32 arrays for one synthetic Takara clip."""
    dt = 1.0 / float(fps)
    t = np.arange(frames, dtype=np.float64) * dt

    # Distinct joint values per frame with a constant analytic velocity.
    joint_base = 0.01 * np.arange(_JOINT_COUNT, dtype=np.float64)
    joint_rate = 0.001 * (1.0 + np.arange(_JOINT_COUNT, dtype=np.float64))
    joint_pos = joint_base[None, :] + joint_rate[None, :] * t[:, None]
    joint_vel = np.broadcast_to(joint_rate[None, :], (frames, _JOINT_COUNT))

    body_pos = np.zeros((frames, _BODY_COUNT, 3), dtype=np.float64)
    body_quat = np.zeros((frames, _BODY_COUNT, 4), dtype=np.float64)
    body_lin = np.zeros((frames, _BODY_COUNT, 3), dtype=np.float64)
    body_ang = np.zeros((frames, _BODY_COUNT, 3), dtype=np.float64)

    vx, vy = root_velocity_xy
    for b in range(_BODY_COUNT):
        if b == 0:
            vel = np.array([vx, vy, 0.0])
            rate = float(yaw_rate)
            offset = np.array([0.0, 0.0, 0.8])
        else:
            vel = np.array([0.01 * b, 0.005 * b, 0.0])
            rate = 0.02 * b
            offset = np.array([0.1 * b, -0.05 * b, 0.5 + 0.01 * b])
        body_pos[:, b, :] = offset[None, :] + vel[None, :] * t[:, None]
        body_lin[:, b, :] = vel[None, :]
        angle = rate * t
        body_quat[:, b, 0] = np.cos(angle / 2.0)
        body_quat[:, b, 3] = np.sin(angle / 2.0)
        body_ang[:, b, 2] = rate

    return {
        "fps": np.array([fps], dtype=np.float32),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": np.ascontiguousarray(joint_vel, dtype=np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
    }


def write_takara_arrays(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write ``arrays`` as ``path/motion.npz`` (allow_pickle-free) and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    motion = path / "motion.npz"
    np.savez(motion, **arrays)
    return motion
