import hashlib
import json
import math
import os
import shutil
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
    TerrainFootClearanceValidator,
    TerrainSceneAlignment,
    TerrainTransitionTerminalEvaluator,
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


def _right_angle_trajectory() -> CommandTrajectory:
    return CommandTrajectory(
        torch.tensor(
            [[0.3, 0.0], [0.3, 0.3], [0.3, 0.6]], dtype=torch.float32
        ),
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        ),
    )


def _lateral_trajectory() -> CommandTrajectory:
    return CommandTrajectory(
        torch.tensor(
            [[0.0, 0.3], [0.0, 0.6], [0.0, 0.9]], dtype=torch.float32
        ),
        torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        ),
    )


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

    def test_dense_feature_combines_body_patch_and_command_path_profile(self):
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
        body_expected = (
            0.1 * forward + 0.2 * lateral + 0.05 * forward * forward
        ).reshape(-1)
        distance = np.array([0.25, 0.5, 0.75, 1.0])
        path_expected = 0.1 * distance + 0.05 * distance * distance
        expected = np.concatenate((body_expected, path_expected))

        self.assertEqual(extension.dimension, 95)
        self.assertEqual(tuple(row.shape), (95,))
        np.testing.assert_allclose(row.numpy(), expected, rtol=0.0, atol=2e-5)

    def test_dense_body_patch_stays_body_relative_while_profile_follows_command(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        forward, lateral = np.meshgrid(
            np.asarray(DENSE_FORWARD_M),
            np.asarray(DENSE_LATERAL_M),
            indexing="ij",
        )
        body_expected = (
            0.1 * forward + 0.2 * lateral + 0.05 * forward * forward
        ).reshape(-1)
        distance = np.array([0.25, 0.5, 0.75, 1.0])
        lateral_profile = 0.2 * distance

        lateral_row = extension.query_row(_state(), _lateral_trajectory())
        straight_row = extension.query_row(_state(), _straight_trajectory())

        np.testing.assert_allclose(
            lateral_row[:91].numpy(), body_expected, rtol=0.0, atol=2e-5
        )
        np.testing.assert_allclose(
            lateral_row[91:].numpy(), lateral_profile, rtol=0.0, atol=2e-5
        )
        self.assertTrue(torch.equal(lateral_row[:91], straight_row[:91]))
        self.assertFalse(torch.allclose(lateral_row[91:], straight_row[91:]))

    def test_dense_command_profile_bends_along_command_polyline(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        row = extension.query_row(_state(), _right_angle_trajectory())
        center_xy = np.array(
            [[0.25, 0.0], [0.30, 0.20], [0.30, 0.45], [0.30, 0.70]]
        )
        expected = (
            0.1 * center_xy[:, 0]
            + 0.2 * center_xy[:, 1]
            + 0.05 * center_xy[:, 0] * center_xy[:, 0]
        )
        np.testing.assert_allclose(
            row[91:].numpy(), expected, rtol=0.0, atol=2e-5
        )

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
                tuple(dense_clip.shape), (clip.valid_frame_stop, 95)
            )
            self.assertEqual(
                tuple(legacy_clip.shape), (clip.valid_frame_stop, 4)
            )
        self.assertTrue(
            torch.equal(dense_rows[0], torch.zeros_like(dense_rows[0]))
        )
        self.assertTrue(
            torch.equal(legacy_rows[0], torch.zeros_like(legacy_rows[0]))
        )
        self.assertGreater(float(dense_rows[1].std()), 0.0)

    def test_expanded_dataset_applies_motion_to_terrain_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            expanded_root = Path(tmp) / "expanded"
            shutil.copytree(self.dataset_root, expanded_root)
            manifest_path = expanded_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema"] = "g1-torch-terrain-corpus/v1"
            for descriptor in manifest["clips"][1:]:
                descriptor["kind"] = "terrain"
                descriptor["terrain"]["motion_to_terrain_xy_yaw"] = [
                    0.0,
                    0.0,
                    0.0,
                ]
            transform = [1.0, -2.0, math.pi / 2.0]
            manifest["clips"][1]["terrain"][
                "motion_to_terrain_xy_yaw"
            ] = transform
            manifest["accepted_clips"] = manifest["clips"]
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = TerrainDataset.load(expanded_root, device="cpu")
            extension = TerrainFeatureExtension.for_condition(
                dataset,
                condition="dense",
                query_scene="stair/0000/motion.npz",
                weight=4.0,
            )
            rows = extension.database_rows(
                dataset.folder, torch.device("cpu")
            )

            clip = dataset.folder.clips[1]
            root_index = dataset.folder.layout.root_body_index
            root_xy = clip.body_position_world[0, root_index, :2]
            quaternion = clip.body_quaternion_world_wxyz[0, root_index]
            w, x, y, z = quaternion
            root_yaw = math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            scene_root = np.array(
                [-root_xy[1] + transform[0], root_xy[0] + transform[1]]
            )

            def curved_height(xy):
                return (
                    0.1 * xy[..., 0]
                    + 0.2 * xy[..., 1]
                    + 0.05 * xy[..., 0] * xy[..., 0]
                )

            forward, lateral = np.meshgrid(
                np.asarray(DENSE_FORWARD_M),
                np.asarray(DENSE_LATERAL_M),
                indexing="ij",
            )
            local = np.stack((forward, lateral), axis=-1).reshape(-1, 2)
            scene_yaw = root_yaw + transform[2]
            cosine = math.cos(scene_yaw)
            sine = math.sin(scene_yaw)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            body_points = scene_root + local @ rotation.T
            legacy = TerrainFeatureExtension.for_condition(
                dataset,
                condition="legacy",
                query_scene="stair/0000/motion.npz",
                weight=4.0,
            )
            legacy_rows = legacy.database_rows(
                dataset.folder, torch.device("cpu")
            )
            database_expected = np.concatenate(
                (
                    curved_height(body_points) - curved_height(scene_root),
                    legacy_rows[1][0].numpy(),
                )
            )
            np.testing.assert_allclose(
                rows[1][0].numpy(), database_expected, rtol=0.0, atol=4e-5
            )

            state = _state()
            trajectory = _straight_trajectory()
            query_scene_root = extension.alignment.matcher_to_scene_xy(
                state.root_position_world[:2]
            ).numpy()
            query_body_points = query_scene_root + local @ rotation.T
            query_expected = np.concatenate(
                (
                    curved_height(query_body_points)
                    - curved_height(query_scene_root),
                    legacy.query_row(state, trajectory).numpy(),
                )
            )
            np.testing.assert_allclose(
                extension.query_row(
                    state, trajectory
                ).numpy(),
                query_expected,
                rtol=0.0,
                atol=4e-5,
            )

    def test_expanded_dataset_requires_finite_motion_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            expanded_root = Path(tmp) / "expanded"
            shutil.copytree(self.dataset_root, expanded_root)
            manifest_path = expanded_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema"] = "g1-torch-terrain-corpus/v1"
            for descriptor in manifest["clips"][1:]:
                descriptor["kind"] = "terrain"
            manifest["accepted_clips"] = manifest["clips"]
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                Exception, "motion-to-terrain registration"
            ):
                TerrainDataset.load(expanded_root, device="cpu")

    def test_foot_clearance_validator_uses_exact_emitted_preview(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        validator = TerrainFootClearanceValidator(
            extension=extension,
            preview_steps=10,
            minimum_clearance_m=-0.03,
        )
        body = torch.zeros((46, 3, 3), dtype=torch.float32)
        scene_xy = extension.alignment.matcher_to_scene_xy(
            body[:, :, :2]
        )
        surface = extension.query_grid.sample_xy(scene_xy)
        body[:, :, 2] = surface
        body[:, 1:, 2] += 0.035
        self.assertTrue(validator(body))

        unsafe = body.clone()
        unsafe[9, 1, 2] = surface[9, 1] - 0.031
        self.assertFalse(validator(unsafe))

        just_outside_preview = body.clone()
        just_outside_preview[10, 1, 2] = surface[10, 1] - 0.031
        self.assertTrue(validator(just_outside_preview))

        for steps in (0, 47, True):
            with self.subTest(preview_steps=steps):
                with self.assertRaises(Exception):
                    TerrainFootClearanceValidator(
                        extension=extension,
                        preview_steps=steps,
                        minimum_clearance_m=-0.03,
                    )
        for clearance in (float("nan"), "unsafe"):
            with self.subTest(clearance=clearance):
                with self.assertRaises(Exception):
                    TerrainFootClearanceValidator(
                        extension=extension,
                        preview_steps=10,
                        minimum_clearance_m=clearance,
                    )

        invalid_windows = (
            torch.zeros((45, 3, 3), dtype=torch.float32),
            body.double(),
            body.clone(),
        )
        invalid_windows[2][0, 0, 0] = float("nan")
        for invalid in invalid_windows:
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                with self.assertRaises(Exception):
                    validator(invalid)
        outside = body.clone()
        outside[:, :, :2] = 100.0
        with self.assertRaisesRegex(Exception, "outside"):
            validator(outside)

    def test_transition_terminal_requires_safe_commanded_surface_drop(self):
        extension = TerrainFeatureExtension.for_condition(
            self.dataset,
            condition="dense",
            query_scene="stair/0000/motion.npz",
            weight=4.0,
        )
        evaluator = TerrainTransitionTerminalEvaluator(
            extension=extension,
            horizon_steps=46,
            minimum_command_progress_m=0.05,
            minimum_surface_drop_m=0.05,
            minimum_clearance_m=-0.03,
        )
        body = torch.zeros((46, 3, 3), dtype=torch.float32)
        body[:, :, 1] = torch.linspace(0.0, -1.0, 46)[:, None]
        scene_xy = extension.alignment.matcher_to_scene_xy(
            body[:, :, :2]
        )
        surface = extension.query_grid.sample_xy(scene_xy)
        body[:, :, 2] = surface
        body[:, 0, 2] += 0.8
        body[:, 1:, 2] += 0.035
        command = torch.tensor((0.0, -0.5), dtype=torch.float32)
        self.assertTrue(evaluator(body, 2, 30, command))

        no_foot_surface_drop = body.clone()
        no_foot_surface_drop[:, 1:, :2] = 0.0
        flat_foot_scene = extension.alignment.matcher_to_scene_xy(
            no_foot_surface_drop[:, 1:, :2]
        )
        flat_foot_surface = extension.query_grid.sample_xy(flat_foot_scene)
        no_foot_surface_drop[:, 1:, 2] = flat_foot_surface + 0.035
        self.assertFalse(
            evaluator(no_foot_surface_drop, 2, 30, command)
        )

        one_foot_surface_drop = body.clone()
        one_foot_surface_drop[:, 1, :2] = 0.0
        one_foot_scene = extension.alignment.matcher_to_scene_xy(
            one_foot_surface_drop[:, 1, :2]
        )
        one_foot_surface = extension.query_grid.sample_xy(one_foot_scene)
        one_foot_surface_drop[:, 1, 2] = one_foot_surface + 0.035
        self.assertFalse(
            evaluator(one_foot_surface_drop, 2, 30, command)
        )

        hovering = body.clone()
        hovering[:, 1:, 2] += 1.0
        self.assertFalse(evaluator(hovering, 2, 30, command))

        staggered_transient_support = body.clone()
        staggered_transient_support[-10:, 1:, 2] += 1.0
        staggered_transient_support[-10, 1, 2] = body[-10, 1, 2]
        staggered_transient_support[-9, 2, 2] = body[-9, 2, 2]
        self.assertFalse(
            evaluator(staggered_transient_support, 2, 30, command)
        )

        early_double_support_then_hover = body.clone()
        early_double_support_then_hover[-10:, 1:, 2] += 1.0
        early_double_support_then_hover[-10, 1:, :2] = body[0, 1:, :2]
        early_double_support_then_hover[-10, 1:, 2] = body[0, 1:, 2]
        self.assertFalse(
            evaluator(early_double_support_then_hover, 2, 30, command)
        )

        lower_double_support_then_hover = body.clone()
        lower_double_support_then_hover[-10:, 1:, 2] += 1.0
        lower_double_support_then_hover[-5, 1:, 2] = body[-5, 1:, 2]
        self.assertFalse(
            evaluator(lower_double_support_then_hover, 2, 30, command)
        )

        unsafe_terminal = body.clone()
        unsafe_terminal[45, 1, 2] = surface[45, 1] - 0.031
        self.assertFalse(evaluator(unsafe_terminal, 2, 30, command))
        self.assertFalse(
            evaluator(
                body,
                2,
                30,
                torch.tensor((0.0, 0.0), dtype=torch.float32),
            )
        )
        with self.assertRaises(Exception):
            TerrainTransitionTerminalEvaluator(
                extension=extension,
                horizon_steps=45,
                minimum_command_progress_m=0.05,
                minimum_surface_drop_m=0.05,
                minimum_clearance_m=-0.03,
            )

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
