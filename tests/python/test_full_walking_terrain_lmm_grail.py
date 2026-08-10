from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from mm_sonic.full_walking_terrain_lmm_grail import (
    _publish_directory_exclusive,
    _canonical_lane_artifacts,
    _reconstruct_inherited_terrain_channels,
    _terrain_grid_from_four_features,
    _resample_quaternions_vectorized,
    authenticate_broad_bank,
    load_inherited_grail_work,
    missing_slope_source_ids,
    resample_holden_range,
    resample_terrain_grid,
)

from resources.g1_terrain_builder.resample import resample_quaternions_wxyz
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    ArtifactSet,
    HoldenClip,
    SkeletonSpec,
)


def _clip(frames: int = 250, *, offset: float = 0.0) -> HoldenClip:
    positions = np.zeros((frames, 31, 3), np.float32)
    positions[:, 0, 0] = np.linspace(offset, offset + 1.0, frames)
    positions[:, 7, 1] = 0.03
    positions[:, 13, 1] = 0.03
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    rotations[1::2] *= -1.0
    terrain = np.stack(
        [np.linspace(index, index + 1.0, frames) for index in range(4)], axis=1
    ).astype(np.float32)
    support = np.stack(
        [np.linspace(index, index + 0.5, frames) for index in range(3)], axis=1
    ).astype(np.float32)
    return HoldenClip(
        name="fixture",
        positions=positions,
        velocities=np.full_like(positions, 99.0),
        rotations=rotations,
        angular_velocities=np.full_like(positions, 99.0),
        contacts=np.ones((frames, 2), np.uint8),
        terrain_features=terrain,
        terrain_support=support,
        source_frames=np.arange(frames, dtype=np.int32),
        terrain_id="fixture",
        source_left_indices=np.arange(frames, dtype=np.int32),
        source_right_indices=np.arange(frames, dtype=np.int32),
        source_alpha=np.zeros(frames, np.float32),
    )


SKELETON = SkeletonSpec(
    names=G1_SKELETON_NAMES,
    parents=np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
)


class FullWalkingGrailResampleTests(unittest.TestCase):
    def test_four_feature_compatibility_grid_preserves_controller_columns(self):
        features = np.arange(20, dtype=np.float32).reshape(5, 4)

        grid = _terrain_grid_from_four_features(features)

        np.testing.assert_array_equal(grid[:, (1, 3, 5, 7)], features)
        np.testing.assert_array_equal(grid[:, 12:24], grid[:, :12])
        np.testing.assert_array_equal(grid[:, 24:36], grid[:, :12])

    def test_inherited_terrain_channels_reconstruct_from_retained_authority(self):
        grid = np.arange(3 * 36, dtype=np.float32).reshape(3, 36)
        support = np.asarray(
            [[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        left = np.asarray([0, 1], dtype=np.int32)
        right = np.asarray([1, 2], dtype=np.int32)
        alpha = np.asarray([0.25, 0.75], dtype=np.float32)

        features, reconstructed_support = _reconstruct_inherited_terrain_channels(
            grid[:2], support, left, right, alpha
        )

        np.testing.assert_array_equal(features, grid[:2, (1, 3, 5, 7)])
        np.testing.assert_allclose(
            reconstructed_support,
            np.asarray([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], np.float32),
            rtol=0,
            atol=0,
        )

    def test_lane_artifacts_canonicalize_fortran_contact_layout(self):
        rows = 5
        artifacts = ArtifactSet.empty(rows, 31)
        artifacts.contacts = np.asfortranarray(np.zeros((rows, 2), dtype=np.uint8))
        self.assertFalse(artifacts.contacts.flags.c_contiguous)

        canonical = _canonical_lane_artifacts(artifacts)

        self.assertTrue(canonical.contacts.flags.c_contiguous)
        self.assertEqual(canonical.contacts.dtype.str, "|u1")

    def test_vectorized_slerp_matches_reviewed_reference(self):
        rng = np.random.default_rng(20260810)
        quaternions = rng.normal(size=(17, 5, 4))
        quaternions[1::3] *= -1.0

        expected = resample_quaternions_wxyz(quaternions, 25.0, 60.0)
        actual = _resample_quaternions_vectorized(
            quaternions, source_fps=25.0, target_fps=60.0
        )

        np.testing.assert_allclose(
            actual.astype(np.float32), expected.astype(np.float32), rtol=0, atol=1e-7
        )

    def test_parent_range_25_to_60_has_exact_count_endpoints_and_map(self):
        source = _clip()

        result, provenance = resample_holden_range(
            source, source_fps=25.0, target_fps=60.0, skeleton=SKELETON
        )

        self.assertEqual(len(result.positions), 598)
        np.testing.assert_array_equal(result.positions[0], source.positions[0])
        self.assertAlmostEqual(
            float(result.positions[-1, 0, 0]), (248 + 0.75) / 249, places=6
        )
        self.assertEqual(int(provenance["left"][-1]), 248)
        self.assertEqual(int(provenance["right"][-1]), 249)
        self.assertAlmostEqual(float(provenance["alpha"][-1]), 0.75, places=6)
        self.assertLess(
            float(np.max(np.abs(np.linalg.norm(result.rotations, axis=-1) - 1.0))),
            1e-6,
        )
        self.assertGreaterEqual(float(np.min(result.rotations[..., 0])), 0.0)

    def test_takara_count_is_exact(self):
        source = _clip(34_863)

        result, _ = resample_holden_range(
            source, source_fps=50.0, target_fps=60.0, skeleton=SKELETON
        )

        self.assertEqual(len(result.positions), 41_835)

    def test_dynamics_are_rederived_inside_each_range(self):
        first, _ = resample_holden_range(
            _clip(offset=0.0), source_fps=25.0, target_fps=60.0, skeleton=SKELETON
        )
        second, _ = resample_holden_range(
            _clip(offset=1000.0), source_fps=25.0, target_fps=60.0, skeleton=SKELETON
        )

        self.assertLess(float(np.max(np.abs(first.velocities[:, 0, 0]))), 1.0)
        self.assertLess(float(np.max(np.abs(second.velocities[:, 0, 0]))), 1.0)
        np.testing.assert_allclose(
            first.velocities[:, 0], second.velocities[:, 0], atol=5e-3
        )
        self.assertTrue(np.isin(first.contacts, (0, 1)).all())
        self.assertTrue(np.isin(second.contacts, (0, 1)).all())

    def test_terrain_grid_uses_same_range_local_clock(self):
        grid = np.arange(250 * 36, dtype=np.float32).reshape(250, 36)

        result = resample_terrain_grid(grid, source_fps=25.0, target_fps=60.0)

        self.assertEqual(result.shape, (598, 36))
        np.testing.assert_array_equal(result[0], grid[0])
        np.testing.assert_allclose(result[-1], grid[248] * 0.25 + grid[249] * 0.75)


class FullWalkingGrailAuthorityTests(unittest.TestCase):
    def test_inherited_work_directory_publication_is_atomic_and_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "work"

            def write_members(staging: Path) -> None:
                (staging / "member.bin").write_bytes(b"complete")

            def interrupted_write(staging: Path) -> None:
                (staging / "member.bin").write_bytes(b"partial")
                raise RuntimeError("interrupted")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                _publish_directory_exclusive(output, interrupted_write)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".work.staging-*")), [])

            _publish_directory_exclusive(output, write_members)
            self.assertEqual((output / "member.bin").read_bytes(), b"complete")
            with self.assertRaises(FileExistsError):
                _publish_directory_exclusive(output, write_members)

    BANK = Path(
        "/home/ubuntu/projects/motion-matching/resources/g1_terrain_banks_candidate"
    )
    INVENTORY = Path(
        "/home/ubuntu/worktrees/motion-matching-hill-conditioning/sonic/runs/"
        "g1-full-walking-terrain-lmm/inventory-v1.json"
    )
    SPLIT_LEDGER = INVENTORY.with_name("split-ledger-v2.json")
    HARDENED_INVENTORY = INVENTORY.with_name("inventory-v2.json")
    INHERITED_WORK = Path(
        "/home/ubuntu/worktrees/motion-matching-full-walking-task2/sonic/runs/"
        "g1-full-walking-terrain-lmm/work/inherited-grail-60hz-v2-local"
    )

    @unittest.skipUnless(
        BANK.exists() and INVENTORY.exists(), "real authorities absent"
    )
    def test_real_bank_and_missing_slope_inventory_are_exact(self):
        authority = authenticate_broad_bank(self.BANK)

        self.assertEqual(len(authority.sources), 15_815)
        self.assertEqual(
            sum(source["terrain_family"] == "curb" for source in authority.sources),
            1_769,
        )
        self.assertEqual(
            sum(source["terrain_family"] == "slope" for source in authority.sources),
            1_857,
        )
        self.assertEqual(
            sum(source["terrain_family"] == "stair" for source in authority.sources),
            12_188,
        )
        missing = missing_slope_source_ids(self.INVENTORY, authority)
        self.assertEqual(len(missing), 23)
        self.assertTrue(
            all(source_id.startswith("grail:slope:") for source_id in missing)
        )

    @unittest.skipUnless(
        INHERITED_WORK.exists()
        and HARDENED_INVENTORY.exists()
        and SPLIT_LEDGER.exists(),
        "real inherited work or hardened authorities absent",
    )
    def test_real_inherited_work_rebinds_without_rederivation(self):
        lane = load_inherited_grail_work(
            self.INHERITED_WORK,
            inventory=self.HARDENED_INVENTORY,
            split_ledger=self.SPLIT_LEDGER,
        )

        self.assertEqual(lane.artifacts.positions.shape, (9_456_772, 31, 3))
        self.assertEqual(lane.terrain_grid.shape, (9_456_772, 36))
        self.assertEqual(len(lane.ranges), 15_814)
        self.assertEqual(len(lane.source_ids), 15_814)
        self.assertEqual(int(lane.source_left_indices.min()), 0)
        self.assertEqual(int(lane.source_right_indices.max()), 249)
        self.assertTrue(
            all(record.authority["source_frame_count"] == 250 for record in lane.ranges)
        )
        self.assertEqual(
            lane.split_ledger_manifest_sha256,
            "5185bd42c153518c80b67c3dd68f54e9122e2e81970893209e9235ff5cc2feb5",
        )


if __name__ == "__main__":
    unittest.main()
