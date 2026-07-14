import mujoco
import numpy as np
from scipy import signal

from resources import quat as holden_quat

from .resample import resample_quaternions_wxyz, resample_vectors
from .schema import HoldenClip, SkeletonSpec, SourceClip

Q_ZUP_TO_YUP = np.array([2**-0.5, -2**-0.5, 0, 0], np.float64)


def vectors_zup_to_yup(values: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(
        Q_ZUP_TO_YUP, np.asarray(values).shape[:-1] + (4,)
    )
    return holden_quat.mul_vec(q, np.asarray(values))


def quaternions_zup_to_yup(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    q = np.broadcast_to(Q_ZUP_TO_YUP, values.shape)
    qi = np.broadcast_to(holden_quat.inv(Q_ZUP_TO_YUP), values.shape)
    return holden_quat.normalize(
        holden_quat.mul(holden_quat.mul(q, values), qi)
    )


class G1Kinematics:
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.body_ids = np.arange(1, self.model.nbody)
        id_to_index = {int(body): i for i, body in enumerate(self.body_ids)}
        rename = {
            "pelvis": "Hips",
            "left_hip_pitch_link": "LeftHipPitch",
            "left_hip_roll_link": "LeftHipRoll",
            "left_hip_yaw_link": "LeftHipYaw",
            "left_knee_link": "LeftKnee",
            "left_ankle_pitch_link": "LeftAnkle",
            "left_ankle_roll_link": "LeftToe",
            "right_hip_pitch_link": "RightHipPitch",
            "right_hip_roll_link": "RightHipRoll",
            "right_hip_yaw_link": "RightHipYaw",
            "right_knee_link": "RightKnee",
            "right_ankle_pitch_link": "RightAnkle",
            "right_ankle_roll_link": "RightToe",
            "waist_yaw_link": "Spine",
            "waist_roll_link": "Spine1",
            "torso_link": "Spine2",
            "left_shoulder_pitch_link": "LeftShoulderPitch",
            "left_shoulder_roll_link": "LeftShoulderRoll",
            "left_shoulder_yaw_link": "LeftShoulderYaw",
            "left_elbow_link": "LeftElbow",
            "left_wrist_roll_link": "LeftWristRoll",
            "left_wrist_pitch_link": "LeftWristPitch",
            "left_wrist_yaw_link": "LeftWrist",
            "right_shoulder_pitch_link": "RightShoulderPitch",
            "right_shoulder_roll_link": "RightShoulderRoll",
            "right_shoulder_yaw_link": "RightShoulderYaw",
            "right_elbow_link": "RightElbow",
            "right_wrist_roll_link": "RightWristRoll",
            "right_wrist_pitch_link": "RightWristPitch",
            "right_wrist_yaw_link": "RightWrist",
        }
        raw_names = tuple(
            self.model.body(int(i)).name for i in self.body_ids)
        missing = [name for name in raw_names if name not in rename]
        if missing:
            raise ValueError(f"unmapped G1 bodies: {missing}")
        self.names = tuple(rename[name] for name in raw_names)
        self.parents = np.array([
            -1 if int(self.model.body(int(i)).parentid[0]) == 0
            else id_to_index[int(self.model.body(int(i)).parentid[0])]
            for i in self.body_ids
        ], np.int32)

    def keyframe_or_zero_qpos(self) -> np.ndarray:
        if self.model.nkey:
            return self.model.key_qpos[0].copy()
        q = np.zeros(self.model.nq)
        q[3] = 1.0
        return q

    def world_from_qpos(
        self, qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos)
        gp = np.empty((len(qpos), len(self.body_ids), 3))
        gq = np.empty((len(qpos), len(self.body_ids), 4))
        for t, q in enumerate(qpos):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            gp[t] = self.data.xpos[self.body_ids]
            gq[t] = self.data.xquat[self.body_ids]
        return gp, gq

    def forward_local(
        self, lp: np.ndarray, lq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return forward_local_hierarchy(lp, lq, self.parents)


def forward_local_hierarchy(
    lp: np.ndarray, lq: np.ndarray, parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gp = np.empty_like(lp)
    gq = np.empty_like(lq)
    for i, parent in enumerate(parents):
        if parent < 0:
            gp[:, i], gq[:, i] = lp[:, i], lq[:, i]
        else:
            gq[:, i] = holden_quat.mul(gq[:, parent], lq[:, i])
            gp[:, i] = gp[:, parent] + holden_quat.mul_vec(
                gq[:, parent], lp[:, i])
    return gp, gq


def change_basis_zup_to_yup(
    gp: np.ndarray, gq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.broadcast_to(Q_ZUP_TO_YUP, gq.shape)
    qi = np.broadcast_to(holden_quat.inv(Q_ZUP_TO_YUP), gq.shape)
    return (
        holden_quat.mul_vec(q, gp),
        holden_quat.mul(holden_quat.mul(q, gq), qi),
    )


def world_to_local(
    gp: np.ndarray, gq: np.ndarray, parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lp = np.empty_like(gp)
    lq = np.empty_like(gq)
    for i, parent in enumerate(parents):
        if parent < 0:
            lp[:, i], lq[:, i] = gp[:, i], gq[:, i]
        else:
            inv = holden_quat.inv(gq[:, parent])
            lp[:, i] = holden_quat.mul_vec(
                inv, gp[:, i] - gp[:, parent])
            lq[:, i] = holden_quat.mul(inv, gq[:, i])
    return lp, lq


def heading_quaternions(forward: np.ndarray) -> np.ndarray:
    forward = np.asarray(forward, np.float64)
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    q = np.stack([
        np.cos(0.5 * yaw),
        np.zeros_like(yaw),
        np.sin(0.5 * yaw),
        np.zeros_like(yaw),
    ], axis=-1)
    return holden_quat.unroll(q)


def _prepend_simulation(
    gp: np.ndarray, gq: np.ndarray, names: tuple[str, ...],
    parents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SkeletonSpec]:
    hips = names.index("Hips")
    torso = names.index("Spine2")
    sim_position = gp[:, torso].copy()
    sim_position[:, 1] = 0.0
    pos_window = min(
        13, len(sim_position) if len(sim_position) % 2
        else len(sim_position) - 1)
    if pos_window >= 5:
        sim_position = signal.savgol_filter(
            sim_position, pos_window, min(3, pos_window - 2),
            axis=0, mode="interp")
    forward = holden_quat.mul_vec(
        gq[:, hips], np.array([1.0, 0.0, 0.0], np.float64))
    forward[:, 1] = 0.0
    lengths = np.linalg.norm(forward, axis=1, keepdims=True)
    if np.any(lengths < 1e-6):
        raise ValueError("G1 pelvis forward projects to zero")
    forward /= lengths
    dir_window = min(
        25, len(forward) if len(forward) % 2 else len(forward) - 1)
    if dir_window >= 5:
        forward = signal.savgol_filter(
            forward, dir_window, min(3, dir_window - 2),
            axis=0, mode="interp")
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    sim_rotation = heading_quaternions(forward)
    lp, lq = world_to_local(gp, gq, parents)
    lp[:, hips] = holden_quat.mul_vec(
        holden_quat.inv(sim_rotation), gp[:, hips] - sim_position)
    lq[:, hips] = holden_quat.mul(
        holden_quat.inv(sim_rotation), gq[:, hips])
    positions = np.concatenate([sim_position[:, None], lp], axis=1)
    rotations = np.concatenate([sim_rotation[:, None], lq], axis=1)
    rotations = holden_quat.unroll(holden_quat.normalize(rotations))
    skeleton = SkeletonSpec(
        ("Simulation",) + names,
        np.concatenate([np.array([-1], np.int32), parents + 1]),
    )
    return positions, rotations, skeleton


def convert_source_clip(
    source: SourceClip, kinematics: G1Kinematics, target_fps: float = 25.0,
) -> tuple[HoldenClip, SkeletonSpec, dict]:
    source.validate()
    gp_z, gq_z = kinematics.world_from_qpos(source.qpos)
    gp_y, gq_y = change_basis_zup_to_yup(gp_z, gq_z)
    gp = resample_vectors(gp_y, source.fps, target_fps)
    gq = resample_quaternions_wxyz(gq_y, source.fps, target_fps)
    positions, rotations, skeleton = _prepend_simulation(
        gp, gq, kinematics.names, kinematics.parents)
    output_t = np.arange(len(positions), dtype=np.float64) / target_fps
    output_frames = np.rint(output_t * source.fps).astype(np.int64)
    output_frames = np.clip(output_frames, 0, len(source.qpos) - 1)
    positions = positions.astype(np.float32)
    rotations = rotations.astype(np.float32)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
        raise ValueError(f"{source.name}: non-finite exported transform")
    quaternion_error = float(np.max(np.abs(
        np.linalg.norm(rotations, axis=-1) - 1.0)))
    if quaternion_error > 1e-4:
        raise ValueError(
            f"{source.name}: exported quaternion norm error "
            f"{quaternion_error}")
    exported_gp, exported_gq = forward_local_hierarchy(
        positions.astype(np.float64), rotations.astype(np.float64),
        skeleton.parents)
    fk_error = float(np.max(np.linalg.norm(
        exported_gp[:, 1:] - gp, axis=-1)))
    dots = np.clip(
        np.abs(np.sum(exported_gq[:, 1:] * gq, axis=-1)), 0.0, 1.0
    )
    fk_rotation_error_degrees = float(
        np.degrees(np.max(2.0 * np.arccos(dots)))
    )
    if fk_error > 0.001:
        raise ValueError(f"{source.name}: exported FK error {fk_error} m")
    if fk_rotation_error_degrees > 0.1:
        raise ValueError(
            f"{source.name}: exported rotational FK error "
            f"{fk_rotation_error_degrees} degrees"
        )
    clip = HoldenClip(
        source.name,
        positions,
        np.zeros_like(positions, np.float32),
        rotations,
        np.zeros_like(positions, np.float32),
        np.zeros((len(positions), 2), np.uint8),
        np.zeros((len(positions), 4), np.float32),
        source.source_frames[output_frames],
        source.terrain_id,
    )
    source_duration = (len(source.qpos) - 1) / source.fps
    output_duration = (len(positions) - 1) / target_fps
    duration_error = abs(output_duration - source_duration)
    if duration_error > 1.0 / target_fps + 1e-12:
        raise ValueError(f"{source.name}: duration error {duration_error} s")
    return clip, skeleton, {
        "fk_max_error_m": fk_error,
        "fk_rotation_max_error_degrees": fk_rotation_error_degrees,
        "duration_error_s": duration_error,
        "quaternion_norm_max_error": quaternion_error,
    }
