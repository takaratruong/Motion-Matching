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


def build_varying_takara_arrays(
    *,
    frames: int = 70,
    fps: int = 50,
    yaw_amplitude: float = 0.6,
    yaw_omega: float = 4.0,
) -> dict[str, np.ndarray]:
    """Return a Takara clip whose root yaw turns non-linearly.

    A constant yaw rate makes the future-facing feature group constant across
    searchable frames (relative facing depends only on the fixed horizon), which
    is a degenerate zero-variance case. A sinusoidal root yaw exercises genuine
    variation in every feature group, matching how real walk data turns.
    """
    arrays = build_takara_arrays(frames=frames, fps=fps, yaw_rate=0.0)
    dt = 1.0 / float(fps)
    t = np.arange(frames, dtype=np.float64) * dt
    yaw = yaw_amplitude * np.sin(yaw_omega * t)
    quat = arrays["body_quat_w"]
    quat[:, 0, 0] = np.cos(yaw / 2.0).astype(np.float32)
    quat[:, 0, 1] = 0.0
    quat[:, 0, 2] = 0.0
    quat[:, 0, 3] = np.sin(yaw / 2.0).astype(np.float32)
    arrays["body_quat_w"] = quat
    return arrays


# ---------------------------------------------------------------------------
# Independent NumPy oracle for the 27-value search feature vector.
#
# This oracle deliberately re-derives every feature and normalization value
# from raw native Z-up / wxyz arrays without importing any production feature
# helper, so equivalence tests compare the Torch extractor against a distinct
# implementation rather than against itself.
# ---------------------------------------------------------------------------

# The frozen source offsets, root/foot body indices, and group weights are
# duplicated here on purpose: the oracle must not import production constants.
ORACLE_HORIZON_FRAMES = (15, 30, 45)
ORACLE_ROOT_BODY = 0
ORACLE_LEFT_FOOT_BODY = 18
ORACLE_RIGHT_FOOT_BODY = 19
ORACLE_GROUPS = (
    ("left_foot_position", slice(0, 3), 0.75),
    ("right_foot_position", slice(3, 6), 0.75),
    ("left_foot_velocity", slice(6, 9), 1.0),
    ("right_foot_velocity", slice(9, 12), 1.0),
    ("pelvis_velocity", slice(12, 15), 1.0),
    ("trajectory_position", slice(15, 21), 1.0),
    ("trajectory_facing", slice(21, 27), 1.5),
)


def oracle_yaw_from_wxyz(quat: np.ndarray) -> float:
    """Z-up yaw of a wxyz quaternion, matching atan2 heading extraction."""
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def oracle_inverse_yaw_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    """Rotate a planar vector by ``-yaw`` (into the current heading frame)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return (c * x + s * y, -s * x + c * y)


def oracle_feature_row(
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_vel: np.ndarray,
    left_foot_pos: np.ndarray,
    right_foot_pos: np.ndarray,
    left_foot_vel: np.ndarray,
    right_foot_vel: np.ndarray,
    trajectory_pos_xy: np.ndarray,
    trajectory_facing_xy: np.ndarray,
) -> np.ndarray:
    """Return the 27-value feature vector for one native Z-up sample."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_vel = np.asarray(root_vel, dtype=np.float64)
    yaw = oracle_yaw_from_wxyz(root_quat)

    out: list[float] = []
    # Groups 0-1: foot positions relative to the root in the heading frame.
    for foot in (left_foot_pos, right_foot_pos):
        rel = np.asarray(foot, dtype=np.float64) - root_pos
        rx, ry = oracle_inverse_yaw_xy(rel[0], rel[1], yaw)
        out += [rx, ry, float(rel[2])]
    # Groups 2-4: foot and pelvis world velocities rotated (not translated).
    for vel in (left_foot_vel, right_foot_vel, root_vel):
        vel = np.asarray(vel, dtype=np.float64)
        rx, ry = oracle_inverse_yaw_xy(vel[0], vel[1], yaw)
        out += [rx, ry, float(vel[2])]
    # Group 5: future root XY positions relative to the current root.
    traj_pos = np.asarray(trajectory_pos_xy, dtype=np.float64)
    for h in range(3):
        rel_x = traj_pos[h, 0] - root_pos[0]
        rel_y = traj_pos[h, 1] - root_pos[1]
        rx, ry = oracle_inverse_yaw_xy(rel_x, rel_y, yaw)
        out += [rx, ry]
    # Group 6: future facing XY directions rotated into the heading frame.
    traj_face = np.asarray(trajectory_facing_xy, dtype=np.float64)
    for h in range(3):
        rx, ry = oracle_inverse_yaw_xy(traj_face[h, 0], traj_face[h, 1], yaw)
        out += [rx, ry]

    return np.asarray(out, dtype=np.float64)


def oracle_clip_row(arrays: dict[str, np.ndarray], frame: int) -> np.ndarray:
    """Compute the oracle feature row for one searchable clip frame."""
    body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    body_lin = np.asarray(arrays["body_lin_vel_w"], dtype=np.float64)

    traj_pos_xy = np.stack(
        [body_pos[frame + off, ORACLE_ROOT_BODY, :2] for off in ORACLE_HORIZON_FRAMES]
    )
    traj_face_xy = np.stack(
        [
            np.array(
                [
                    np.cos(oracle_yaw_from_wxyz(body_quat[frame + off, ORACLE_ROOT_BODY])),
                    np.sin(oracle_yaw_from_wxyz(body_quat[frame + off, ORACLE_ROOT_BODY])),
                ]
            )
            for off in ORACLE_HORIZON_FRAMES
        ]
    )
    return oracle_feature_row(
        root_pos=body_pos[frame, ORACLE_ROOT_BODY],
        root_quat=body_quat[frame, ORACLE_ROOT_BODY],
        root_vel=body_lin[frame, ORACLE_ROOT_BODY],
        left_foot_pos=body_pos[frame, ORACLE_LEFT_FOOT_BODY],
        right_foot_pos=body_pos[frame, ORACLE_RIGHT_FOOT_BODY],
        left_foot_vel=body_lin[frame, ORACLE_LEFT_FOOT_BODY],
        right_foot_vel=body_lin[frame, ORACLE_RIGHT_FOOT_BODY],
        trajectory_pos_xy=traj_pos_xy,
        trajectory_facing_xy=traj_face_xy,
    )


def oracle_clip_rows(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Every searchable oracle feature row for a single clip."""
    frames = int(np.asarray(arrays["joint_pos"]).shape[0])
    valid_stop = frames - ORACLE_HORIZON_FRAMES[-1]
    return np.stack([oracle_clip_row(arrays, f) for f in range(valid_stop)])


def oracle_normalization(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Component means and per-group scales for stacked feature rows."""
    rows = np.asarray(rows, dtype=np.float64)
    mean = rows.mean(axis=0)
    component_std = rows.std(axis=0)  # population std (ddof=0)
    scale = np.zeros(rows.shape[1], dtype=np.float64)
    for _name, group_slice, weight in ORACLE_GROUPS:
        group_std = component_std[group_slice].mean()
        scale[group_slice] = group_std / weight
    return mean, scale
