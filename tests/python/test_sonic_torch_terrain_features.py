import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mm_sonic.torch_motion_features import (
    CommandTrajectory,
    GeneratedFeatureState,
)
from mm_sonic.torch_terrain_features import (
    DENSE_FORWARD_M,
    DENSE_LATERAL_M,
    TerrainDataset,
    TerrainFeatureExtension,
    TerrainSceneAlignment,
)
from resources.g1_torch_stair_builder.publish import publish_stair_slice
from resources.g1_torch_stair_builder.surface import ZUpHeightGrid
from tests.python.test_torch_stair_conversion import (
    _FakeKinematics,
    write_synthetic_pinned_corpus,
)
from tests.python.torch_motion_test_utils import write_takara_clip


def _curved_grid(_source) -> ZUpHeightGrid:
    cell = 0.05
    axis = np.arange(-5.0, 5.0 + cell / 2.0, cell, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    height = 0.1 * x + 0.2 * y + 0.05 * x * x
    return ZUpHeightGrid(
        origin_xy=np.array([-5.0, -5.0], np.float32),
        cell_size_m=cell,
        height_z=height.astype(np.float32),
    )


def _state(root_xy=(0.0, 0.0), yaw=0.0) -> GeneratedFeatureState:
    root = torch.tensor([root_xy[0], root_xy[1], 0.8], dtype=torch.float32)
    quaternion = torch.tensor(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=torch.float32,
    )
    zero = torch.zeros(3, dtype=torch.float32)
    left = torch.tensor([root_xy[0], root_xy[1] + 0.1, 0.0])
    right = torch.tensor([root_xy[0], root_xy[1] - 0.1, 0.0])
    return GeneratedFeatureState(
        root_position_world=root,
        root_orientation_world_wxyz=quaternion,
        root_linear_velocity_world=zero,
        left_foot_position_world=left,
        right_foot_position_world=right,
        left_foot_velocity_world=zero,
        right_foot_velocity_world=zero,
    )


def _straight_trajectory(stopped=False) -> CommandTrajectory:
    if stopped:
        position = torch.zeros((3, 2), dtype=torch.float32)
    else:
        position = torch.tensor(
            [[0.3, 0.0], [0.6, 0.0], [0.9, 0.0]], dtype=torch.float32
        )
    facing = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32
    )
    return CommandTrajectory(position, facing)


class TerrainFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        grail = cls.root / "grail"
        write_synthetic_pinned_corpus(grail)
        flat_motion = write_takara_clip(cls.root / "flat", frames=80)
        g1_xml = cls.root / "g1.xml"
        g1_xml.write_text("<mujoco/>", encoding="utf-8")
        cls.dataset_root = cls.root / "dataset"
        publish_stair_slice(
            output=cls.dataset_root,
            grail_root=grail,
            g1_xml=g1_xml,
            flat_motion=flat_motion,
            kinematics=_FakeKinematics(),
            grid_builder=_curved_grid,
        )
        cls.dataset = TerrainDataset.load(cls.dataset_root, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_dataset_validates_manifest_hashes_and_exact_clip_mapping(self):
        self.assertEqual(len(self.dataset.folder.clips), 5)
        self.assertEqual(len(self.dataset.clip_grids), 5)
        self.assertIsNone(self.dataset.clip_grids[0])
        self.assertTrue(all(grid is not None for grid in self.dataset.clip_grids[1:]))
        self.assertEqual(len(self.dataset.manifest_sha256), 64)

        manifest_path = self.dataset_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["clips"][1]["motion_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            with self.assertRaisesRegex(Exception, "hash"):
                TerrainDataset.load(self.dataset_root, device="cpu")
        finally:
            # Restore the publisher's canonical bytes for the remaining tests.
            original = self.dataset.manifest
            manifest_path.write_text(
                json.dumps(
                    original,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )

    def test_dense_patch_has_exact_forward_major_coordinates_and_values(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        row = extension.query_row(_state(), _straight_trajectory())
        forward, lateral = np.meshgrid(
            np.asarray(DENSE_FORWARD_M),
            np.asarray(DENSE_LATERAL_M),
            indexing="ij",
        )
        expected = (
            0.1 * forward + 0.2 * lateral + 0.05 * forward * forward
        ).reshape(-1)

        self.assertEqual(extension.dimension, 91)
        self.assertEqual(tuple(row.shape), (91,))
        np.testing.assert_allclose(row.numpy(), expected, rtol=0.0, atol=2e-5)

    def test_legacy_uses_arc_distances_and_stopped_heading_extension(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="legacy",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        moving = extension.query_row(_state(), _straight_trajectory())
        stopped = extension.query_row(
            _state(), _straight_trajectory(stopped=True)
        )
        distance = np.array([0.25, 0.5, 0.75, 1.0])
        expected = 0.1 * distance + 0.05 * distance * distance

        self.assertEqual(extension.dimension, 4)
        np.testing.assert_allclose(moving.numpy(), expected, atol=2e-5)
        np.testing.assert_allclose(stopped.numpy(), expected, atol=2e-5)

    def test_database_rows_match_clip_lengths_and_flat_rows_are_zero(self):
        dense = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        legacy = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="legacy",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        dense_rows = dense.database_rows(self.dataset.folder, torch.device("cpu"))
        legacy_rows = legacy.database_rows(
            self.dataset.folder, torch.device("cpu")
        )

        self.assertEqual(len(dense_rows), 5)
        for clip, dense_clip, legacy_clip in zip(
            self.dataset.folder.clips, dense_rows, legacy_rows
        ):
            self.assertEqual(
                tuple(dense_clip.shape), (clip.valid_frame_stop, 91)
            )
            self.assertEqual(
                tuple(legacy_clip.shape), (clip.valid_frame_stop, 4)
            )
        self.assertTrue(torch.equal(dense_rows[0], torch.zeros_like(dense_rows[0])))
        self.assertTrue(
            torch.equal(legacy_rows[0], torch.zeros_like(legacy_rows[0]))
        )
        self.assertGreater(float(dense_rows[1].std()), 0.0)

    def test_scene_alignment_maps_matcher_world_into_recorded_world(self):
        alignment = TerrainSceneAlignment(
            translation_scene_xy=torch.tensor([2.0, -1.0]),
            yaw_scene_from_matcher=torch.tensor(math.pi / 2.0),
        )
        mapped = alignment.matcher_to_scene_xy(
            torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        )
        np.testing.assert_allclose(
            mapped.numpy(), [[2.0, -1.0], [2.0, 0.0]], atol=1e-6
        )

    def test_outside_query_and_invalid_condition_fail_closed(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        with self.assertRaisesRegex(Exception, "outside"):
            extension.query_row(
                _state(root_xy=(100.0, 100.0)), _straight_trajectory()
            )
        with self.assertRaisesRegex(Exception, "condition"):
            TerrainFeatureExtension.for_condition(
                self.dataset,
                condition="unknown",
                query_scene="stair/0000/motion.npz",
                weight=4.0,
            )


@unittest.skipUnless(
    os.environ.get("MM_REAL_STAIR_ALIGNMENT_ORACLE") == "1",
    "real alignment oracle is opt-in",
)
class RealStairAlignmentOracleTests(unittest.TestCase):
    def test_native_surface_matches_recorded_foot_contacts(self):
        from resources.g1_torch_stair_builder.alignment_oracle import (
            compare_real_contact_alignment,
        )

        error = compare_real_contact_alignment()
        self.assertLessEqual(error, 1e-4)


if __name__ == "__main__":
    unittest.main()
