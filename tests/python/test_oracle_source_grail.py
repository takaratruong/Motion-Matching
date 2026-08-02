from collections import Counter
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from mm_sonic.terrain_oracle.source_grail import (
    C490_EXPECTED_COUNTS,
    C490_FAMILIES,
    discover_clean_c490_records,
    iter_grail_clips,
)
from mm_sonic.terrain_oracle.storage import write_mesh


SHARD_ROOT = Path("/move/data/terrain-aware/grail-sweep/shards")
MODEL = Path(
    "/move/u/justingu/Projects/TWIST2/assets/g1/"
    "g1_29dof_rev_1_0.xml"
)


def _write_usd(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "model"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "model"
{
    def Mesh "mesh"
    {
        point3f[] points = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
    }
}
"""
    )
    return path


def _write_grail_fixture(root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    family = "c490_stair_p1"
    shard = root / f"{family}_0000"
    (shard / "robot").mkdir(parents=True)
    (shard / "object_usd").mkdir()
    stem = "terrain_stairs__fixture__0000"
    frames = 4
    timeline = np.arange(frames, dtype=np.float32)
    root_position = np.zeros((frames, 3), dtype=np.float32)
    root_position[:, 0] = timeline * 0.1
    root_quaternion_xyzw = np.zeros((frames, 4), dtype=np.float32)
    root_quaternion_xyzw[:, 3] = 1.0
    joints = timeline[:, None] + np.arange(29, dtype=np.float32)[None, :]
    joblib.dump(
        {
            "fixture": {
                "root_trans_offset": root_position,
                "root_rot": root_quaternion_xyzw,
                "dof": joints,
                "fps": 25.0,
            }
        },
        shard / "robot" / f"{stem}.pkl",
    )
    _write_usd(shard / "object_usd" / f"{stem}.usd")
    position = np.array((1.25, -2.5, 0.375), dtype=np.float64)
    rotation = np.array(
        (np.cos(0.3), 0.0, 0.0, np.sin(0.3)), dtype=np.float64
    )
    records = [
        {
            "stem": stem,
            "category": "stairs",
            "n_frames": frames,
            "terrain": {
                "position_env": position.tolist(),
                "rotation_env_wxyz": rotation.tolist(),
            },
            "pose_source": "fixture-exact",
        }
    ]
    for index in range(1, C490_EXPECTED_COUNTS[family]):
        extra_stem = f"terrain_stairs__fixture__{index:04d}"
        (shard / "robot" / f"{extra_stem}.pkl").write_bytes(b"unused")
        _write_usd(shard / "object_usd" / f"{extra_stem}.usd")
        records.append(
            {
                "stem": extra_stem,
                "category": "stairs",
                "n_frames": 250,
                "terrain": {
                    "position_env": [0.0, 0.0, 0.0],
                    "rotation_env_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "pose_source": "fixture-unused",
            }
        )
    (shard / "clips.json").write_text(json.dumps(records))
    return root, position, rotation


def _write_legacy_family(
    root: Path,
    *,
    family: str = "c490_slope",
    category: str = "slope",
    terrain_value: str = "slope",
) -> Path:
    shard = root / f"{family}_0000"
    (shard / "robot").mkdir(parents=True)
    (shard / "object_usd").mkdir()
    count = C490_EXPECTED_COUNTS.get(family, 1)
    rows = []
    for index in range(count):
        stem = f"terrain_{category}s__{category}_{index:03d}__000"
        (shard / "robot" / f"{stem}.pkl").write_bytes(b"fixture")
        _write_usd(shard / "object_usd" / f"{stem}.usd")
        rows.append(
            {
                "stem": stem,
                "category": category,
                "n_frames": 250,
                "terrain": terrain_value,
            }
        )
    (shard / "clips.json").write_text(json.dumps(rows))
    return root


class GrailDiscoveryTests(unittest.TestCase):
    @unittest.skipUnless(SHARD_ROOT.is_dir(), "clean GRAIL shards unavailable")
    def test_c490_inventory_is_exact_and_never_reads_rollout_windows(self):
        records = discover_clean_c490_records(SHARD_ROOT)
        counts = Counter(record.family for record in records)

        self.assertEqual(counts, C490_EXPECTED_COUNTS)
        self.assertEqual(counts["c490_stair_p1"] + counts["c490_stair_p2"], 340)
        self.assertEqual(counts["c490_slope"], 77)
        self.assertEqual(counts["c490_curb"], 72)
        self.assertEqual(sum(counts.values()), 489)
        self.assertEqual(C490_FAMILIES, tuple(C490_EXPECTED_COUNTS))
        self.assertTrue(
            all(
                "/sonic-rollouts/" not in str(record.robot_path)
                for record in records
            )
        )

    def test_legacy_slope_uses_a_tagged_minus_90_degree_pose_not_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = _write_legacy_family(Path(temporary))

            records = discover_clean_c490_records(
                root, families=("c490_slope",)
            )

            self.assertEqual(len(records), 77)
            self.assertEqual({record.pose_source for record in records}, {
                "legacy_default_yaw"
            })
            np.testing.assert_array_equal(
                records[0].terrain_position_env,
                np.zeros(3, dtype=np.float32),
            )
            np.testing.assert_allclose(
                records[0].terrain_rotation_env_wxyz,
                (np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)),
                rtol=0.0,
                atol=1.0e-7,
            )
            self.assertFalse(
                np.array_equal(
                    records[0].terrain_rotation_env_wxyz,
                    (1.0, 0.0, 0.0, 0.0),
                )
            )

    def test_legacy_string_pose_is_limited_to_matching_slope_and_curb_families(self):
        cases = (
            ("c490_slope", "curb", "slope"),
            ("c490_slope", "slope", "curb"),
            ("c490_stair_p1", "stairs", "stairs"),
        )
        for family, category, terrain_value in cases:
            with self.subTest(
                family=family,
                category=category,
                terrain=terrain_value,
            ), tempfile.TemporaryDirectory() as temporary:
                root = _write_legacy_family(
                    Path(temporary),
                    family=family,
                    category=category,
                    terrain_value=terrain_value,
                )
                with self.assertRaisesRegex(
                    ValueError, "legacy|terrain|category"
                ):
                    discover_clean_c490_records(root, families=(family,))


@unittest.skipUnless(MODEL.is_file(), "real G1 MuJoCo model is unavailable")
class GrailSourceAdapterTests(unittest.TestCase):
    def test_emits_clip_and_paired_content_addressed_mesh_with_exact_pose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, expected_position, expected_rotation = _write_grail_fixture(
                Path(temporary)
            )

            imported = next(
                iter_grail_clips(
                    root, families=("c490_stair_p1",), model_path=MODEL
                )
            )

            clip = imported.clip
            mesh = imported.mesh
            self.assertEqual(clip.frame_count, 7)
            self.assertEqual(mesh.vertices_local.shape, (4, 3))
            np.testing.assert_array_equal(
                mesh.faces, np.array(((0, 1, 2), (0, 2, 3)))
            )
            np.testing.assert_array_equal(mesh.valid_faces, (True, True))
            asset = next(root.glob("*/object_usd/*.usd"))
            asset_bytes = asset.read_bytes()
            self.assertEqual(
                clip.terrain.asset_sha256,
                hashlib.sha256(asset_bytes).hexdigest(),
            )
            self.assertEqual(clip.terrain.asset_size_bytes, len(asset_bytes))
            self.assertEqual(clip.terrain.asset_license_id, "UNRECORDED")
            np.testing.assert_allclose(
                clip.terrain.world_from_terrain.translation_world,
                expected_position,
                rtol=0.0,
                atol=1.0e-7,
            )
            self.assertEqual(imported.source_record.pose_source, "fixture-exact")
            np.testing.assert_allclose(
                clip.terrain.world_from_terrain.quaternion_world_from_local_wxyz,
                expected_rotation,
                rtol=0.0,
                atol=1.0e-7,
            )
            with tempfile.TemporaryDirectory() as mesh_directory:
                record = write_mesh(Path(mesh_directory), mesh)
            self.assertEqual(clip.terrain.mesh_sha256, record.sha256)
            self.assertEqual(mesh.source_asset_sha256, clip.terrain.asset_sha256)
            clip.validate()

    def test_source_hash_order_contacts_and_kinematics_are_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _position, _rotation = _write_grail_fixture(Path(temporary))
            imported = next(
                iter_grail_clips(
                    root, families=("c490_stair_p1",), model_path=MODEL
                )
            )
            clip = imported.clip
            robot = imported.source_record.robot_path
            robot_bytes = robot.read_bytes()

            self.assertEqual(clip.source.source_path, str(robot.resolve()))
            self.assertEqual(clip.source.source_size_bytes, len(robot_bytes))
            self.assertEqual(
                clip.source.source_sha256,
                hashlib.sha256(robot_bytes).hexdigest(),
            )
            self.assertEqual(clip.source.source_license_id, "UNRECORDED")
            np.testing.assert_allclose(
                clip.root_position_world[:, 0],
                np.arange(7, dtype=np.float32) * 0.05,
                rtol=0.0,
                atol=1.0e-7,
            )
            self.assertIn("contacts-unreconstructed", clip.action_tags)
            self.assertFalse(np.any(clip.contact))
            self.assertFalse(np.any(clip.contact_confidence))
            self.assertFalse(np.any(clip.commands.observed_mask))
            self.assertFalse(clip.body_position_world.flags.writeable)
            self.assertFalse(imported.mesh.faces.flags.writeable)


if __name__ == "__main__":
    unittest.main()
