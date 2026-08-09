from __future__ import annotations

import numpy as np

from resources import quat as holden_quat

from .schema import ArtifactSet, FeatureSet


FEATURE_NAMES = (
    "left_foot_position_x", "left_foot_position_y", "left_foot_position_z",
    "right_foot_position_x", "right_foot_position_y", "right_foot_position_z",
    "left_foot_velocity_x", "left_foot_velocity_y", "left_foot_velocity_z",
    "right_foot_velocity_x", "right_foot_velocity_y", "right_foot_velocity_z",
    "hip_velocity_x", "hip_velocity_y", "hip_velocity_z",
    "root_position_20_x", "root_position_20_z",
    "root_position_40_x", "root_position_40_z",
    "root_position_60_x", "root_position_60_z",
    "root_facing_20_x", "root_facing_20_z",
    "root_facing_40_x", "root_facing_40_z",
    "root_facing_60_x", "root_facing_60_z",
    "terrain_height_025", "terrain_height_050",
    "terrain_height_075", "terrain_height_100",
)
FEATURE_WEIGHTS = (0.75, 1.0, 1.0, 1.0, 1.5, 1.0)
_GROUPS = (
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 15), (15, 21), (21, 27), (27, 31),
)
_NORMALIZATION_WEIGHTS = (0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0)
_DISABLED_SCALE = np.finfo(np.float32).max


def _global_kinematics(artifacts: ArtifactSet):
    frames, bones = artifacts.positions.shape[:2]
    gp = np.empty((frames, bones, 3), np.float64)
    gq = np.empty((frames, bones, 4), np.float64)
    gv = np.empty((frames, bones, 3), np.float64)
    ga = np.empty((frames, bones, 3), np.float64)
    lp = artifacts.positions.astype(np.float64)
    lq = artifacts.rotations.astype(np.float64)
    lv = artifacts.velocities.astype(np.float64)
    la = artifacts.angular_velocities.astype(np.float64)
    for bone, parent in enumerate(artifacts.parents):
        if parent < 0:
            gp[:, bone], gq[:, bone] = lp[:, bone], lq[:, bone]
            gv[:, bone], ga[:, bone] = lv[:, bone], la[:, bone]
            continue
        rotated_position = holden_quat.mul_vec(gq[:, parent], lp[:, bone])
        gp[:, bone] = gp[:, parent] + rotated_position
        gq[:, bone] = holden_quat.mul(gq[:, parent], lq[:, bone])
        gv[:, bone] = (
            gv[:, parent]
            + holden_quat.mul_vec(gq[:, parent], lv[:, bone])
            + np.cross(ga[:, parent], rotated_position)
        )
        ga[:, bone] = (
            ga[:, parent]
            + holden_quat.mul_vec(gq[:, parent], la[:, bone])
        )
    return gp, gq, gv


def _future_indices(artifacts: ArtifactSet, horizon: int) -> np.ndarray:
    indices = np.empty(len(artifacts.positions), np.int64)
    for start, stop in zip(artifacts.range_starts, artifacts.range_stops):
        rows = np.arange(int(start), int(stop), dtype=np.int64)
        indices[rows] = np.minimum(rows + horizon, int(stop) - 1)
    return indices


def _normalize(raw: np.ndarray) -> FeatureSet:
    values = np.empty_like(raw, np.float32)
    offset = np.empty(31, np.float32)
    scale = np.empty(31, np.float32)
    for (start, stop), weight in zip(_GROUPS, _NORMALIZATION_WEIGHTS):
        group = raw[:, start:stop].astype(np.float64)
        group_offset = group.mean(axis=0)
        group_std = float(np.mean(np.sqrt(np.mean(
            np.square(group - group_offset), axis=0))))
        offset[start:stop] = group_offset.astype(np.float32)
        if not np.isfinite(group_std) or group_std <= 0.0 or weight == 0.0:
            scale[start:stop] = _DISABLED_SCALE
            values[:, start:stop] = 0.0
        else:
            group_scale = np.float32(group_std / weight)
            scale[start:stop] = group_scale
            values[:, start:stop] = (
                (group - offset[start:stop]) / group_scale).astype(np.float32)
    result = FeatureSet(values, offset, scale)
    result.validate()
    return result


def build_matching_features(
    artifacts: ArtifactSet,
    fps: float,
    horizons: tuple[int, int, int],
) -> FeatureSet:
    if not isinstance(artifacts, ArtifactSet):
        raise TypeError("artifacts must be an ArtifactSet")
    artifacts.validate()
    if not np.isfinite(fps) or float(fps) <= 0.0:
        raise ValueError("feature fps must be positive and finite")
    if tuple(horizons) != horizons or len(horizons) != 3 \
            or any(type(value) is not int or value <= 0 for value in horizons) \
            or tuple(sorted(horizons)) != horizons:
        raise ValueError("feature horizons must be three increasing integers")
    if artifacts.positions.shape[1] != 31:
        raise ValueError("G1 matching features require exactly 31 bones")

    gp, gq, gv = _global_kinematics(artifacts)
    root_rotation_inv = holden_quat.inv(gq[:, 0])
    raw = np.empty((len(gp), 31), np.float32)
    for output, bone in ((0, 6), (3, 12)):
        raw[:, output:output + 3] = holden_quat.mul_vec(
            root_rotation_inv, gp[:, bone] - gp[:, 0])
    for output, bone in ((6, 6), (9, 12), (12, 1)):
        raw[:, output:output + 3] = holden_quat.mul_vec(
            root_rotation_inv, gv[:, bone])
    for slot, horizon in enumerate(horizons):
        future = _future_indices(artifacts, horizon)
        position = holden_quat.mul_vec(
            root_rotation_inv, gp[future, 0] - gp[:, 0])
        facing = holden_quat.mul_vec(
            root_rotation_inv,
            holden_quat.mul_vec(gq[future, 0], np.array([0.0, 0.0, 1.0])),
        )
        raw[:, 15 + slot * 2:17 + slot * 2] = position[:, [0, 2]]
        raw[:, 21 + slot * 2:23 + slot * 2] = facing[:, [0, 2]]
    raw[:, 27:31] = artifacts.terrain_features
    if not np.isfinite(raw).all():
        raise ValueError("raw matching features must be finite")
    return _normalize(raw)
