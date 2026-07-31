import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from resources.g1_torch_stair_builder import corpus


_BASE = "terrain_curbs__curb_000__000"
_FRAMES = 250


def _write_source(
    root: Path,
    *,
    translation_excursion_m: float,
    rotation_excursion_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    for directory in ("robot", "objects", "object_usd"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    robot_quaternion = np.zeros((_FRAMES, 4), np.float32)
    robot_quaternion[:, 3] = 1.0
    robot = {
        "dof": np.zeros((_FRAMES, 29), np.float32),
        "root_trans_offset": np.zeros((_FRAMES, 3), np.float32),
        "root_rot": robot_quaternion,
        "fps": 25.0,
    }

    fraction = np.linspace(0.0, 1.0, _FRAMES, dtype=np.float64)
    positions = np.zeros((_FRAMES, 1, 3), np.float32)
    positions[:, 0, 0] = fraction * translation_excursion_m
    angle = np.deg2rad(rotation_excursion_deg) * fraction
    quaternions_wxyz = np.zeros((_FRAMES, 1, 4), np.float32)
    quaternions_wxyz[:, 0, 0] = np.cos(0.5 * angle)
    quaternions_wxyz[:, 0, 3] = np.sin(0.5 * angle)
    objects = {
        "root_pos": positions,
        "root_quat": quaternions_wxyz,
        "fps": 25.0,
        "scale": np.ones((3, 1), np.float32),
    }
    identity = "synthetic-scene"
    joblib.dump({identity: robot}, root / "robot" / f"{_BASE}.pkl")
    joblib.dump({identity: objects}, root / "objects" / f"{_BASE}.pkl")
    (root / "object_usd" / f"{_BASE}.usd").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    return positions, quaternions_wxyz


class GeneralGrailSourceTests(unittest.TestCase):
    def test_exposes_public_single_source_loader(self):
        self.assertTrue(hasattr(corpus, "load_source"))

    def test_freezes_frame_zero_and_reports_bounded_object_pose_excursion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positions, _ = _write_source(
                root,
                translation_excursion_m=0.01,
                rotation_excursion_deg=1.0,
            )
            try:
                source = corpus.load_source(root, _BASE)
            except ValueError as error:
                self.fail(f"bounded object-pose jitter was rejected: {error}")

        np.testing.assert_array_equal(
            source.object_position_world, positions[0, 0]
        )
        np.testing.assert_array_equal(
            source.object_quaternion_world_xyzw,
            np.array([0.0, 0.0, 0.0, 1.0], np.float32),
        )
        self.assertAlmostEqual(
            source.object_translation_excursion_m, 0.01, places=6
        )
        self.assertAlmostEqual(
            source.object_rotation_excursion_rad,
            np.deg2rad(1.0),
            places=6,
        )

    def test_rejects_object_translation_excursion_above_two_centimeters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(
                root,
                translation_excursion_m=0.021,
                rotation_excursion_deg=0.0,
            )
            with self.assertRaisesRegex(
                ValueError, "translation excursion"
            ):
                corpus.load_source(root, _BASE)

    def test_rejects_object_rotation_excursion_above_two_degrees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(
                root,
                translation_excursion_m=0.0,
                rotation_excursion_deg=2.1,
            )
            with self.assertRaisesRegex(ValueError, "rotation excursion"):
                corpus.load_source(root, _BASE)


if __name__ == "__main__":
    unittest.main()
