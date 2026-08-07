import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

from mm_sonic.terrain_oracle.canonical import ISAACLAB_BODY_NAMES
from mm_sonic.terrain_pfnn.sources import (
    discover_grail_slope_records,
    load_grail_source,
    load_lafan_source,
)


class _FakeForwardResult:
    def __init__(self, frames: int) -> None:
        timeline = np.arange(frames, dtype=np.float32)
        self.body_names = ISAACLAB_BODY_NAMES
        self.body_position_world = np.zeros((frames, 30, 3), dtype=np.float32)
        self.body_position_world[:, :, 0] = timeline[:, None]
        self.body_quaternion_world_xyzw = np.zeros(
            (frames, 30, 4), dtype=np.float32
        )
        self.body_quaternion_world_xyzw[..., 3] = 1.0


class _FakeFK:
    def forward(
        self,
        root_position: np.ndarray,
        root_quaternion_xyzw: np.ndarray,
        dof_mujoco: np.ndarray,
    ) -> _FakeForwardResult:
        del root_quaternion_xyzw, dof_mujoco
        return _FakeForwardResult(len(root_position))


class TerrainPFNNSourcesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.grail_root = self.root / "GRAIL"
        robot_dir = self.grail_root / "data" / "slope" / "robot"
        usd_dir = self.grail_root / "data" / "slope" / "object_usd"
        robot_dir.mkdir(parents=True)
        usd_dir.mkdir(parents=True)
        stem = "terrain_slopes__slope_113__000"
        timeline = np.arange(4, dtype=np.float32)
        root_position = np.zeros((4, 3), dtype=np.float32)
        root_position[:, 0] = timeline
        root_quaternion_xyzw = np.zeros((4, 4), dtype=np.float32)
        root_quaternion_xyzw[:, 3] = 1.0
        joblib.dump(
            {
                "fixture": {
                    "root_trans_offset": root_position,
                    "root_rot": root_quaternion_xyzw,
                    "dof": timeline[:, None]
                    + np.arange(29, dtype=np.float32)[None, :],
                    "fps": 25.0,
                }
            },
            robot_dir / f"{stem}.pkl",
        )
        (usd_dir / f"{stem}.usd").write_text("#usda 1.0\n")
        self.grail_record = replace(
            discover_grail_slope_records(self.grail_root)[0], expected_frames=4
        )
        self.lafan_csv = self.root / "walk_fixture.csv"
        rows = np.zeros((4, 36), dtype=np.float64)
        rows[:, 0] = timeline
        rows[:, 6] = 1.0
        rows[:, 7:] = timeline[:, None] + np.arange(29)[None, :]
        np.savetxt(self.lafan_csv, rows, delimiter=",", fmt="%.9f")
        self.fake_fk = _FakeFK()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_grail_conversion_is_30hz_wxyz_and_canonical_joints(self) -> None:
        source = load_grail_source(self.grail_record, self.fake_fk)
        self.assertEqual(source.fps, 30.0)
        self.assertEqual(source.root_quaternion_world_wxyz.shape[1:], (4,))
        self.assertEqual(source.joint_position.shape[1:], (29,))
        self.assertEqual(source.body_position_world.shape[1:], (30, 3))
        np.testing.assert_allclose(
            np.linalg.norm(source.root_quaternion_world_wxyz, axis=1),
            1.0,
            atol=1.0e-6,
        )

    def test_lafan_30hz_rows_are_not_temporally_resampled(self) -> None:
        source = load_lafan_source(self.lafan_csv, self.fake_fk)
        self.assertEqual(source.frame_count, 4)
        np.testing.assert_allclose(
            source.root_position_world[:, 0], np.arange(4)
        )

    def test_discovery_pairs_every_robot_with_same_stem_usd(self) -> None:
        records = discover_grail_slope_records(self.grail_root)
        self.assertEqual(
            [record.stem for record in records],
            ["terrain_slopes__slope_113__000"],
        )
