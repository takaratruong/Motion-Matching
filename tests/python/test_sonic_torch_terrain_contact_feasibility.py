import unittest
from pathlib import Path

import numpy as np
import torch

from mm_sonic.joints import ContractError
from mm_sonic.torch_contact_segments import ANKLE_ORIGIN_SOLE_M


class TerrainContactFeasibilityContractTest(unittest.TestCase):
    def test_contact_feasibility_defaults_are_frozen(self):
        from mm_sonic.torch_terrain_contact_feasibility import (
            TerrainContactFeasibilityConfig,
        )

        value = TerrainContactFeasibilityConfig()

        self.assertEqual(value.sample_stride, 1)
        self.assertEqual(value.edge_margin_m, 0.04)
        self.assertEqual(value.maximum_edge_height_range_m, 0.025)
        self.assertEqual(value.minimum_swing_clearance_m, -0.005)
        self.assertEqual(value.minimum_sole_clearance_m, -0.025)

    def test_result_requires_reason_exactly_when_rejected(self):
        from mm_sonic.torch_terrain_contact_feasibility import (
            TerrainContactFeasibilityResult,
        )

        with self.assertRaisesRegex(ContractError, "result"):
            TerrainContactFeasibilityResult(
                accepted=False,
                reason=None,
                maximum_stance_error_m=0.0,
                landing_error_m=0.0,
                minimum_swing_clearance_m=0.0,
                minimum_sole_clearance_m=0.0,
                maximum_footprint_height_range_m=0.0,
                maximum_height_deformation_m=0.0,
            )


class TerrainContactFeasibilityLayerTest(unittest.TestCase):
    def setUp(self):
        from mm_sonic.torch_terrain_contact_feasibility import (
            TerrainContactFeasibilityConfig,
        )

        self.config = TerrainContactFeasibilityConfig()
        self.feet = torch.zeros((4, 2, 3), dtype=torch.float32)
        self.feet[:, 0, 1] = -0.10
        self.feet[:, 1, 1] = 0.10
        self.feet[:, :, 2] = float(ANKLE_ORIGIN_SOLE_M)
        self.support = torch.tensor(
            [[True, False], [True, False], [True, True], [True, True]]
        )
        self.source_surface = torch.zeros((4, 2), dtype=torch.float32)

    def validate(
        self,
        sample_surface,
        *,
        align_initial_support=False,
        landing_footprint_position_world=None,
    ):
        from mm_sonic.torch_terrain_contact_feasibility import (
            validate_placed_contact_trace,
        )

        return validate_placed_contact_trace(
            foot_position_world=self.feet,
            support_mask=self.support,
            source_surface_height_m=self.source_surface,
            sample_surface=sample_surface,
            config=self.config,
            align_initial_support=align_initial_support,
            landing_footprint_position_world=(
                landing_footprint_position_world
            ),
        )

    @staticmethod
    def flat(points):
        return torch.zeros(points.shape[:-1], dtype=points.dtype, device=points.device)

    def test_terrain_domain_is_first_rejection(self):
        def outside(_points):
            raise ContractError("outside")

        self.assertEqual(self.validate(outside).reason, "terrain-domain")

    def test_stance_height_rejects_before_landing_layers(self):
        self.feet[1, 0, 2] += 0.051

        result = self.validate(self.flat)

        self.assertEqual(result.reason, "stance-height")
        self.assertGreater(result.maximum_stance_error_m, 0.05)

    def test_landing_height_has_its_own_tighter_bound(self):
        self.feet[2:, 1, 2] += 0.031

        result = self.validate(self.flat)

        self.assertEqual(result.reason, "landing-height")
        self.assertGreater(result.landing_error_m, 0.03)

    def test_landing_cross_footprint_rejects_an_edge(self):
        self.feet[:, 1, 0] = 0.01

        def step(points):
            return torch.where(points[..., 0] >= 0.0, 0.03, 0.0)

        self.feet[:, 1, 2] = 0.03 + float(ANKLE_ORIGIN_SOLE_M)
        self.source_surface[:, 1] = 0.03

        result = self.validate(step)

        self.assertEqual(result.reason, "landing-edge-margin")
        self.assertGreater(result.maximum_footprint_height_range_m, 0.025)

    def test_oriented_sole_points_detect_a_toe_crossing_the_riser(self):
        self.feet[:, 1, 0] = -0.08
        sole = self.feet[:, :, None, :].expand(-1, -1, 5, -1).clone()
        sole[:, 1, :, 0] += torch.tensor((0.0, 0.12, -0.05, 0.12, -0.05))
        sole[:, 1, :, 1] += torch.tensor((0.0, 0.0, 0.0, 0.04, -0.04))

        def step(points):
            return torch.where(points[..., 0] >= 0.0, 0.03, 0.0)

        result = self.validate(
            step, landing_footprint_position_world=sole
        )

        self.assertEqual(result.reason, "landing-edge-margin")
        self.assertGreater(result.maximum_footprint_height_range_m, 0.025)

    def test_oriented_sole_penetration_is_checked_outside_landings(self):
        sole = self.feet[:, :, None, :].expand(-1, -1, 5, -1).clone()
        sole[..., 2] = 0.0
        sole[1, 1, 1, 2] = -0.026

        result = self.validate(
            self.flat, landing_footprint_position_world=sole
        )

        self.assertEqual(result.reason, "sole-penetration")
        self.assertLess(result.minimum_sole_clearance_m, -0.025)

    def test_unsupported_swing_penetration_is_checked_every_frame(self):
        self.feet[1, 1, 2] -= 0.006

        result = self.validate(self.flat)

        self.assertEqual(result.reason, "swing-penetration")
        self.assertLess(result.minimum_swing_clearance_m, -0.005)

    def test_source_to_scene_landing_deformation_is_hard_gated(self):
        self.source_surface[2:, 1] = 0.061

        result = self.validate(self.flat)

        self.assertEqual(result.reason, "height-deformation")
        self.assertGreater(result.maximum_height_deformation_m, 0.06)

    def test_valid_trace_reports_all_measured_extrema(self):
        result = self.validate(self.flat)

        self.assertTrue(result.accepted)
        self.assertIsNone(result.reason)
        self.assertEqual(result.maximum_stance_error_m, 0.0)
        self.assertEqual(result.landing_error_m, 0.0)
        self.assertEqual(result.minimum_swing_clearance_m, 0.0)
        self.assertEqual(result.minimum_sole_clearance_m, 0.0)

    def test_initial_support_anchor_removes_only_global_vertical_offset(self):
        self.feet[..., 2] -= 0.70

        unaligned = self.validate(self.flat)
        aligned = self.validate(self.flat, align_initial_support=True)

        self.assertEqual(unaligned.reason, "stance-height")
        self.assertTrue(aligned.accepted)
        self.assertAlmostEqual(aligned.maximum_stance_error_m, 0.0, places=6)


class TerrainSkillContactTraceIntegrationTest(unittest.TestCase):
    def build_fixture(self, *, buried_swing: bool):
        from mm_sonic.torch_motion_data import (
            G1_TAKARA_LAYOUT,
            MotionClip,
            MotionFolder,
        )
        from mm_sonic.torch_motion_features import (
            FeatureNormalization,
            TorchMotionDatabase,
        )
        from mm_sonic.torch_terrain_features import (
            TerrainDataset,
            TerrainFeatureExtension,
            TerrainSceneAlignment,
            _TorchHeightGrid,
        )
        from mm_sonic.torch_terrain_skills import SkillInterval, TerrainSkill

        frames = 4
        body_position = np.zeros((frames, 30, 3), dtype=np.float32)
        body_position[:, 0, 2] = 0.80
        body_position[:, 18, 1] = -0.10
        body_position[:, 19, 1] = 0.10
        body_position[:, (18, 19), 2] = float(ANKLE_ORIGIN_SOLE_M)
        if buried_swing:
            body_position[1, 19, 2] -= 0.006
        body_quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
        body_quaternion[..., 0] = 1.0
        clip = MotionClip(
            relative_path="terrain/motion.npz",
            fps=50,
            joint_position=np.zeros((frames, 29), dtype=np.float32),
            joint_velocity=np.zeros((frames, 29), dtype=np.float32),
            body_position_world=body_position,
            body_quaternion_world_wxyz=body_quaternion,
            body_linear_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
            body_angular_velocity_world=np.zeros((frames, 30, 3), dtype=np.float32),
        )
        folder = MotionFolder(Path("."), G1_TAKARA_LAYOUT, (clip,), "identity")
        grid = _TorchHeightGrid(
            origin_xy=torch.tensor([-1.0, -1.0]),
            cell_size_m=0.1,
            height_z=torch.zeros((21, 21), dtype=torch.float32),
        )
        identity = TerrainSceneAlignment(
            translation_scene_xy=torch.zeros(2),
            yaw_scene_from_matcher=torch.zeros(()),
        )
        dataset = TerrainDataset(
            root=Path("."),
            folder=folder,
            clip_grids=(grid,),
            clip_alignments=(identity,),
            manifest_sha256="manifest",
            _manifest={},
            device=torch.device("cpu"),
        )
        extension = TerrainFeatureExtension(
            dataset=dataset,
            condition="dense",
            query_clip_index=0,
            query_grid=grid,
            alignment=identity,
            weight=1.0,
        )
        zeros = torch.zeros(27)
        database = TorchMotionDatabase(
            folder=folder,
            device=torch.device("cpu"),
            normalization=FeatureNormalization(zeros, torch.ones_like(zeros)),
            reset_row=0,
            _search_features=torch.zeros((1, 27)),
            _search_clip_index=torch.zeros(1, dtype=torch.long),
            _search_frame_index=torch.zeros(1, dtype=torch.long),
            _source_row_map={(0, 0): 0},
        )
        support = torch.tensor(
            [[True, False], [True, False], [True, True], [True, True]]
        )
        skill = TerrainSkill(
            skill_index=0,
            clip_index=0,
            interval=SkillInterval(0, 0, frames),
            entry_rows=(0,),
            support_mask=support,
            foot_surface_height_m=torch.zeros((frames, 2)),
        )
        return skill, dataset, database, extension

    def test_wrapper_places_and_checks_every_source_frame(self):
        from mm_sonic.torch_terrain_contact_feasibility import (
            TerrainContactFeasibilityConfig,
            validate_terrain_skill_contact_trace,
        )

        skill, dataset, database, extension = self.build_fixture(
            buried_swing=True
        )
        result = validate_terrain_skill_contact_trace(
            skill=skill,
            canonical_entry_row=0,
            endpoint_frame_exclusive=4,
            dataset=dataset,
            database=database,
            query_terrain=extension,
            current_root_position_world=torch.tensor([0.0, 0.0, 0.8]),
            current_root_orientation_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            config=TerrainContactFeasibilityConfig(),
        )

        self.assertEqual(result.reason, "swing-penetration")


if __name__ == "__main__":
    unittest.main()
