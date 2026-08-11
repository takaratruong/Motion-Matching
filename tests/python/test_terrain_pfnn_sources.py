import tempfile
import unittest
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from mm_sonic.terrain_oracle.canonical import ISAACLAB_BODY_NAMES
from mm_sonic.retarget_pfnn_bvh_g1 import (
    G1_JOINT_NAMES,
    GMR_COMMIT,
    PFNN_POSITION_SCALE,
    RETARGET_PROJECT_COMMIT,
)
from mm_sonic.terrain_pfnn.sources import (
    discover_grail_slope_records,
    load_grail_source,
    load_lafan_source,
    load_pfnn_retarget_source,
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
        self.pfnn_motion = self.root / "pfnn-retarget.npz"
        pfnn_frames = 12
        pfnn_root = np.zeros((pfnn_frames, 3), dtype=np.float64)
        pfnn_root[:, 0] = np.arange(pfnn_frames, dtype=np.float64)
        pfnn_quaternion = np.zeros((pfnn_frames, 4), dtype=np.float64)
        pfnn_quaternion[:, 3] = 1.0
        pfnn_dof = (
            np.square(np.arange(pfnn_frames, dtype=np.float64))[:, None]
            + np.arange(29, dtype=np.float64)[None, :]
        ) / 1000.0
        with self.pfnn_motion.open("wb") as stream:
            np.savez_compressed(
                stream,
                root_pos=pfnn_root,
                root_quat=pfnn_quaternion,
                dof=pfnn_dof,
                fps=np.asarray(120.0, dtype=np.float64),
                engine=np.asarray("gmr"),
                joint_names=np.asarray(G1_JOINT_NAMES),
                joint_limits=np.repeat(
                    np.asarray([[-2.0, 2.0]], dtype=np.float64), 29, axis=0
                ),
            )
        output_sha = hashlib.sha256(self.pfnn_motion.read_bytes()).hexdigest()
        self.pfnn_source_sha = "a" * 64
        self.pfnn_motion.with_suffix(".receipt.json").write_text(
            json.dumps(
                {
                    "schema": "native-g1-pfnn-sample-retarget/v1",
                    "status": "accepted",
                    "source_sha256": self.pfnn_source_sha,
                    "prepared_sha256": "c" * 64,
                    "output_sha256": output_sha,
                    "gmr_commit": GMR_COMMIT,
                    "retarget_project_commit": RETARGET_PROJECT_COMMIT,
                    "fps": 120.0,
                    "source_frame_count": 2000,
                    "start_frame": 240,
                    "frame_count": pfnn_frames,
                    "warmup_frames": 120,
                    "aliases": [["Spine1", "Spine2"]],
                    "joint_names": list(G1_JOINT_NAMES),
                    "root_quaternion_order": "xyzw",
                    "grounding_offset_m": 0.0,
                    "grounding": "source",
                    "pfnn_position_scale": PFNN_POSITION_SCALE,
                }
            ),
            encoding="utf-8",
        )

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

    def test_pfnn_retarget_samples_exact_120hz_frames_at_30hz(self) -> None:
        source = load_pfnn_retarget_source(
            self.pfnn_motion,
            self.fake_fk,
            clip_id="pfnn__flat_turn__00240_00252",
            terrain_id="flat",
            expected_source_sha256=self.pfnn_source_sha,
            expected_start_frame=240,
        )
        self.assertEqual(source.fps, 30.0)
        self.assertEqual(source.frame_count, 3)
        np.testing.assert_array_equal(source.root_position_world[:, 0], [0.0, 4.0, 8.0])
        self.assertEqual(source.joint_position.shape, (3, 29))
        self.assertEqual(source.body_position_world.shape, (3, 30, 3))
        self.assertEqual(source.joint_velocity_source, "unique_predecessor")
        backward = (
            source.joint_position[1] - source.joint_position[0]
        ) * np.float32(30.0)
        central = (
            source.joint_position[2] - source.joint_position[0]
        ) * np.float32(15.0)
        np.testing.assert_array_equal(source.joint_velocity[1], backward)
        self.assertFalse(np.array_equal(source.joint_velocity[1], central))

    def test_pfnn_retarget_rejects_receipt_or_artifact_tampering(self) -> None:
        receipt_path = self.pfnn_motion.with_suffix(".receipt.json")
        original = receipt_path.read_text(encoding="utf-8")
        for field, value, message in (
            ("source_sha256", "b" * 64, "selected source digest"),
            ("start_frame", 241, "selected interval"),
            ("grounding", "flat", "source grounding"),
        ):
            payload = json.loads(original)
            payload[field] = value
            receipt_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                load_pfnn_retarget_source(
                    self.pfnn_motion,
                    self.fake_fk,
                    clip_id="pfnn__fixture",
                    terrain_id="flat",
                    expected_source_sha256=self.pfnn_source_sha,
                    expected_start_frame=240,
                )
        receipt_path.write_text(original, encoding="utf-8")
        with self.pfnn_motion.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "output digest"):
            load_pfnn_retarget_source(
                self.pfnn_motion,
                self.fake_fk,
                clip_id="pfnn__fixture",
                terrain_id="flat",
                expected_source_sha256=self.pfnn_source_sha,
                expected_start_frame=240,
            )
