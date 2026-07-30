import re
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from mm_sonic.torch_motion_data import MotionFolder
from resources.g1_torch_stair_builder.conversion import (
    ISAAC_BODY_HOLDEN_NAMES,
    convert_source_to_native_50hz,
)
from resources.g1_torch_stair_builder.corpus import (
    STAIR_BASES,
    load_pinned_sources,
)


def _robot_record(frames: int, fps: float) -> dict:
    time = np.arange(frames, dtype=np.float64) / fps
    dof = np.stack(
        [0.01 * joint + (0.1 + 0.001 * joint) * time for joint in range(29)],
        axis=1,
    ).astype(np.float32)
    root = np.stack(
        (0.2 * time, -0.1 * time, 0.8 + 0.02 * time), axis=1
    ).astype(np.float32)
    angle = 0.3 * time
    xyzw = np.zeros((frames, 4), np.float32)
    xyzw[:, 2] = np.sin(angle / 2.0)
    xyzw[:, 3] = np.cos(angle / 2.0)
    xyzw[frames // 2 :] *= -1.0
    return {
        "dof": dof,
        "root_trans_offset": root,
        "root_rot": xyzw,
        "pose_aa": np.zeros((frames, 30, 3), np.float32),
        "smpl_joints": np.zeros((frames, 24, 3), np.float32),
        "fps": float(fps),
    }


def _object_record(frames: int, fps: float) -> dict:
    root = np.broadcast_to(
        np.array([0.2, -0.3, 0.15], np.float32), (frames, 1, 3)
    ).copy()
    # GRAIL object records use wxyz (unlike robot root_rot, which uses xyzw).
    wxyz = np.broadcast_to(
        np.array([2**-0.5, 0.0, 0.0, 2**-0.5], np.float32),
        (frames, 1, 4),
    ).copy()
    return {
        "root_pos": root,
        "root_quat": wxyz,
        "fps": float(fps),
        "scale": np.ones((3, 1), np.float32),
        "contact_points_left_hand": {},
        "contact_points_right_hand": {},
    }


def write_synthetic_pinned_corpus(
    root: Path,
    *,
    fps: float = 25.0,
    wrong_outer_key_for: str | None = None,
) -> None:
    for directory in ("robot", "objects", "object_usd"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for index, base in enumerate(STAIR_BASES):
        identity = f"scene-{index}"
        robot_identity = (
            identity + "-wrong" if base == wrong_outer_key_for else identity
        )
        joblib.dump(
            {robot_identity: _robot_record(250, fps)},
            root / "robot" / f"{base}.pkl",
        )
        joblib.dump(
            {identity: _object_record(250, fps)},
            root / "objects" / f"{base}.pkl",
        )
        (root / "object_usd" / f"{base}.usd").write_text(
            "#usda 1.0\n", encoding="utf-8"
        )


class _FakeKinematics:
    def __init__(self) -> None:
        self.names = (
            "Hips",
            "LeftHipPitch",
            "LeftHipRoll",
            "LeftHipYaw",
            "LeftKnee",
            "LeftAnkle",
            "LeftToe",
            "RightHipPitch",
            "RightHipRoll",
            "RightHipYaw",
            "RightKnee",
            "RightAnkle",
            "RightToe",
            "Spine",
            "Spine1",
            "Spine2",
            "LeftShoulderPitch",
            "LeftShoulderRoll",
            "LeftShoulderYaw",
            "LeftElbow",
            "LeftWristRoll",
            "LeftWristPitch",
            "LeftWrist",
            "RightShoulderPitch",
            "RightShoulderRoll",
            "RightShoulderYaw",
            "RightElbow",
            "RightWristRoll",
            "RightWristPitch",
            "RightWrist",
        )

    def world_from_qpos(
        self, qpos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        frames = len(qpos)
        positions = np.zeros((frames, len(self.names), 3), np.float64)
        quaternions = np.zeros((frames, len(self.names), 4), np.float64)
        quaternions[..., 0] = 1.0
        for body in range(len(self.names)):
            positions[:, body, 0] = qpos[:, 0] + body
            positions[:, body, 1] = qpos[:, 1]
            positions[:, body, 2] = qpos[:, 2]
        quaternions[:, 0] = qpos[:, 3:7]
        return positions, quaternions


class PinnedCorpusTests(unittest.TestCase):
    def test_inventory_is_exactly_four_named_04d99a9e43_records(self):
        self.assertEqual(len(STAIR_BASES), 4)
        self.assertTrue(all("04d99a9e43" in name for name in STAIR_BASES))
        self.assertEqual(
            [name.rsplit("__", 1)[-1] for name in STAIR_BASES[:3]],
            ["0000", "0001", "0002"],
        )
        self.assertTrue(STAIR_BASES[3].endswith("_updown__0000"))

    def test_loader_owns_exact_finite_source_arrays_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root)
            sources = load_pinned_sources(root)

        self.assertEqual([source.base for source in sources], list(STAIR_BASES))
        for source in sources:
            self.assertEqual(source.source_fps, 25.0)
            self.assertEqual(source.robot_qpos_mujoco.shape, (250, 36))
            self.assertEqual(source.object_position_world.shape, (3,))
            self.assertEqual(source.object_quaternion_world_xyzw.shape, (4,))
            np.testing.assert_allclose(
                source.object_quaternion_world_xyzw,
                [0.0, 0.0, 2**-0.5, 2**-0.5],
                atol=1e-7,
            )
            self.assertEqual(source.object_scale.shape, (3,))
            self.assertEqual(
                set(source.source_sha256), {"robot", "object", "usd"}
            )
            self.assertTrue(
                all(len(value) == 64 for value in source.source_sha256.values())
            )
            self.assertTrue(np.isfinite(source.robot_qpos_mujoco).all())
            self.assertFalse(source.robot_qpos_mujoco.flags.writeable)
            norms = np.linalg.norm(source.robot_qpos_mujoco[:, 3:7], axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_loader_rejects_missing_mismatched_identity_and_non_25hz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root)
            (root / "robot" / f"{STAIR_BASES[0]}.pkl").unlink()
            with self.assertRaisesRegex(
                ValueError, rf"{re.escape(STAIR_BASES[0])}.*missing"
            ):
                load_pinned_sources(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(
                root, wrong_outer_key_for=STAIR_BASES[1]
            )
            with self.assertRaisesRegex(ValueError, "outer record identity"):
                load_pinned_sources(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root, fps=50.0)
            with self.assertRaisesRegex(ValueError, "fps.*25"):
                load_pinned_sources(root)


class NativeConversionTests(unittest.TestCase):
    def test_250_samples_at_25hz_become_499_native_50hz_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root)
            source = load_pinned_sources(root)[0]

        converted = convert_source_to_native_50hz(
            source, _FakeKinematics()
        )

        self.assertEqual(converted.fps, 50)
        self.assertEqual(converted.joint_position.shape, (499, 29))
        self.assertEqual(converted.joint_velocity.shape, (499, 29))
        self.assertEqual(converted.body_position_world.shape, (499, 30, 3))
        self.assertEqual(
            converted.body_quaternion_world_wxyz.shape, (499, 30, 4)
        )
        self.assertEqual(converted.body_linear_velocity_world.shape, (499, 30, 3))
        self.assertEqual(
            converted.body_angular_velocity_world.shape, (499, 30, 3)
        )
        source_mujoco_joints = source.robot_qpos_mujoco[:, 7:]
        isaac_to_source = np.array(
            [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
             16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        )
        np.testing.assert_allclose(
            converted.joint_position[[0, -1]],
            source_mujoco_joints[[0, -1]][:, isaac_to_source],
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(converted.body_quaternion_world_wxyz, axis=-1),
            1.0,
            rtol=0.0,
            atol=1e-5,
        )
        fake_source_index = _FakeKinematics().names.index("LeftToe")
        self.assertEqual(ISAAC_BODY_HOLDEN_NAMES[18], "LeftToe")
        np.testing.assert_allclose(
            converted.body_position_world[:, 18, 0],
            converted.body_position_world[:, 0, 0] + fake_source_index,
            atol=1e-5,
        )
        for array in converted.as_npz_fields().values():
            self.assertTrue(np.isfinite(array).all())

    def test_output_loads_through_strict_native_motion_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_synthetic_pinned_corpus(root / "source")
            source = load_pinned_sources(root / "source")[0]
            converted = convert_source_to_native_50hz(
                source, _FakeKinematics()
            )
            motion_dir = root / "motions" / "stair"
            motion_dir.mkdir(parents=True)
            np.savez(motion_dir / "motion.npz", **converted.as_npz_fields())

            folder = MotionFolder.load(root / "motions")

        self.assertEqual(len(folder.clips), 1)
        self.assertEqual(folder.clips[0].frame_count, 499)
        self.assertEqual(folder.clips[0].fps, 50)


if __name__ == "__main__":
    unittest.main()
