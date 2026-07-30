from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from mm_sonic.joints import TARGET_JOINT_ORDER
from resources.g1_torch_terrain_builder.motion import (
    load_native_motion_50hz,
)


def _write_motion(
    path: Path,
    *,
    fps: float,
    frames: int = 50,
    joint_names: tuple[str, ...] | None = TARGET_JOINT_ORDER,
) -> None:
    phase = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    joint = phase[:, None] * np.arange(1, 30, dtype=np.float32)[None, :]
    body = np.zeros((frames, 30, 3), np.float32)
    body[..., 0] = phase[:, None]
    body[..., 2] = 0.75
    quaternion = np.zeros((frames, 30, 4), np.float32)
    quaternion[..., 0] = 1.0
    fields = {
        "fps": np.array([fps], np.float32),
        "joint_pos": joint,
        "joint_vel": np.zeros_like(joint),
        "body_pos_w": body,
        "body_quat_w": quaternion,
        "body_lin_vel_w": np.zeros_like(body),
        "body_ang_vel_w": np.zeros_like(body),
    }
    if joint_names is not None:
        fields["joint_names"] = np.asarray(joint_names)
    np.savez(path, **fields)


class ExpandedTerrainMotionTests(unittest.TestCase):
    def test_25_33_and_50_hz_preserve_endpoints_in_native_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for fps in (25.0, 33.0, 50.0):
                with self.subTest(fps=fps):
                    path = root / f"motion-{int(fps)}.npz"
                    _write_motion(path, fps=fps)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        motion = load_native_motion_50hz(
                            path,
                            source_fps=fps,
                            layout_provenance="g1-29dof-isaaclab-v1",
                        )
                    self.assertEqual(caught, [])
                    self.assertEqual(motion.fps, 50)
                    self.assertEqual(motion.joint_position.shape[1:], (29,))
                    self.assertEqual(
                        motion.body_position_world.shape[1:], (30, 3)
                    )
                    np.testing.assert_array_equal(
                        motion.joint_position[[0, -1]],
                        np.load(path)["joint_pos"][[0, -1]],
                    )
                    np.testing.assert_array_equal(
                        motion.body_position_world[[0, -1]],
                        np.load(path)["body_pos_w"][[0, -1]],
                    )
                    np.testing.assert_allclose(
                        np.linalg.norm(
                            motion.body_quaternion_world_wxyz, axis=-1
                        ),
                        1.0,
                        rtol=0.0,
                        atol=1e-5,
                    )

    def test_layout_must_be_proven_and_joint_names_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_names = root / "no-names.npz"
            _write_motion(no_names, fps=50.0, joint_names=None)
            with self.assertRaisesRegex(ValueError, "layout provenance"):
                load_native_motion_50hz(no_names)

            wrong_names = root / "wrong-names.npz"
            _write_motion(
                wrong_names,
                fps=50.0,
                joint_names=tuple(reversed(TARGET_JOINT_ORDER)),
            )
            with self.assertRaisesRegex(ValueError, "joint_names"):
                load_native_motion_50hz(
                    wrong_names,
                    layout_provenance="g1-29dof-isaaclab-v1",
                )

    def test_invalid_rate_shape_quaternion_and_short_clip_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "motion.npz"
            _write_motion(path, fps=50.0)
            with self.assertRaisesRegex(ValueError, "source_fps"):
                load_native_motion_50hz(
                    path,
                    source_fps=25.0,
                    layout_provenance="g1-29dof-isaaclab-v1",
                )

            with np.load(path, allow_pickle=False) as archive:
                fields = {name: archive[name] for name in archive.files}
            fields["body_quat_w"] = fields["body_quat_w"].copy()
            fields["body_quat_w"][4, 3] = 0.0
            fields["body_quat_w"][4, 0] = 0.0
            np.savez(path, **fields)
            with self.assertRaisesRegex(ValueError, "quaternion"):
                load_native_motion_50hz(
                    path,
                    layout_provenance="g1-29dof-isaaclab-v1",
                )

            short = root / "short.npz"
            _write_motion(short, fps=50.0, frames=45)
            with self.assertRaisesRegex(ValueError, "46"):
                load_native_motion_50hz(
                    short,
                    layout_provenance="g1-29dof-isaaclab-v1",
                )


if __name__ == "__main__":
    unittest.main()
