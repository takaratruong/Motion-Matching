import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from resources import build_g1_terrain_database as builder
from resources.g1_terrain_builder import terrain as terrain_module
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics,
    change_basis_zup_to_yup,
    forward_local_hierarchy,
)
from resources.g1_terrain_builder.resample import (
    resample_map,
    resample_quaternions_wxyz,
    resample_vectors,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
)
from resources.g1_terrain_builder.sources import (
    load_authenticated_grail_slope,
)
from mm_sonic.grail_terrain_source import resample_grail_motion


SLOPE_NAME = "terrain_slopes__slope_000__000"
SLOPE_ROOT = "/home/ubuntu/datasets/GRAIL/data/slope"
G1_XML = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"
FROZEN_FEATURE_MINIMUM = np.array([
    -0.05486539751291275,
    -0.09694159030914307,
    -0.12729622423648834,
    -0.12729622423648834,
], np.float32)
FROZEN_FEATURE_MAXIMUM = np.array([
    0.05370394513010979,
    0.09506101161241531,
    0.12704572081565857,
    0.12723475694656372,
], np.float32)
FROZEN_FEATURE_STD = np.array([
    0.02696162149121752,
    0.04589913504391204,
    0.05615584307619531,
    0.06524118885644932,
], np.float64)


def authored_slope_args():
    return SimpleNamespace(
        output_fps=60.0,
        slope_robot=f"{SLOPE_ROOT}/robot/{SLOPE_NAME}.pkl",
        slope_usd=f"{SLOPE_ROOT}/object_usd/{SLOPE_NAME}.usd",
        slope_recon=f"{SLOPE_ROOT}/recon/{SLOPE_NAME}.pkl",
        slope_metadata=f"{SLOPE_ROOT}/meta/{SLOPE_NAME}.pkl",
        g1_xml=G1_XML,
    )


class GrailTerrainSourceResamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        timeline = np.arange(4, dtype=np.float32)
        self.root_position = np.stack(
            (timeline, np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)),
            axis=1,
        )
        self.root_quaternion_xyzw = np.zeros((4, 4), dtype=np.float32)
        self.root_quaternion_xyzw[:, 3] = 1.0
        self.root_quaternion_xyzw[1, 3] = -1.0
        self.dof_mujoco = timeline[:, None] + np.arange(29, dtype=np.float32)[None, :]

    def test_existing_25_to_50_path_keeps_legacy_samples_and_count(self) -> None:
        result = resample_grail_motion(
            self.root_position,
            self.root_quaternion_xyzw,
            self.dof_mujoco,
            source_fps=25.0,
            target_fps=50.0,
        )
        self.assertEqual(result.fps, 50.0)
        self.assertEqual(len(result.root_position), 7)
        np.testing.assert_allclose(result.root_position[:, 0], np.arange(7) * 0.5)
        np.testing.assert_allclose(result.root_position[-1], self.root_position[-1])

    def test_25_to_30_uses_target_timestamps_and_holds_terminal_pose(self) -> None:
        result = resample_grail_motion(
            self.root_position,
            self.root_quaternion_xyzw,
            self.dof_mujoco,
            source_fps=25.0,
            target_fps=30.0,
        )
        self.assertEqual(result.fps, 30.0)
        self.assertEqual(len(result.root_position), round(4 * 30 / 25))
        np.testing.assert_allclose(
            result.root_position[:, 0],
            (0.0, 5.0 / 6.0, 5.0 / 3.0, 2.5, 3.0),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(result.root_position[-1], self.root_position[-1])
        np.testing.assert_allclose(
            np.linalg.norm(result.root_quaternion_xyzw, axis=1), 1.0, atol=1.0e-6
        )


class AuthoredSlopePairTest(unittest.TestCase):
    _candidate = None

    def candidate(self):
        assemble = getattr(builder, "_assemble_authored_slope_source", None)
        self.assertTrue(
            callable(assemble), "authored slope source seam is missing")
        if assemble is None:
            return None
        if type(self)._candidate is None:
            type(self)._candidate = assemble(authored_slope_args())
        return type(self)._candidate

    def test_candidate_handoff_has_exact_four_field_api(self):
        candidate = self.candidate()
        if candidate is None:
            return
        self.assertEqual(
            tuple(field.name for field in fields(candidate)),
            ("source", "terrain", "scene", "provenance"),
        )
        self.assertEqual(len(candidate.source.positions), 598)
        candidate.source.validate()

    def test_exact_250_at_25_to_598_at_60_provenance_and_span(self):
        candidate = self.candidate()
        if candidate is None:
            return
        provisional = candidate.source
        self.assertEqual(len(provisional.positions), 598)
        left, right, alpha = resample_map(250, 25.0, 60.0)
        np.testing.assert_array_equal(provisional.source_left_indices, left)
        np.testing.assert_array_equal(provisional.source_right_indices, right)
        np.testing.assert_array_equal(provisional.source_alpha, alpha)
        self.assertEqual(candidate.provenance["source_frames"], 250)
        self.assertEqual(candidate.provenance["source_fps"], 25.0)
        self.assertEqual(candidate.provenance["source_span_s"], 9.96)
        self.assertEqual(candidate.provenance["output_span_s"], 9.95)
        self.assertEqual(
            candidate.provenance["output_shortfall_s"],
            0.010000000000001563,
        )
        self.assertEqual(
            candidate.provenance["rejected_output_ranges"], [[0, 3]])
        np.testing.assert_array_equal(
            provisional.terrain_features[:3], np.zeros((3, 4), np.float32))

    def test_simulation_is_planar_while_fk_preserves_native_global_hips(self):
        candidate = self.candidate()
        if candidate is None:
            return
        provisional = candidate.source
        np.testing.assert_array_equal(
            provisional.positions[:, 0, 1], np.zeros(598, np.float32))
        np.testing.assert_allclose(
            provisional.rotations[:, 0][:, (1, 3)], 0.0,
            rtol=0.0, atol=1e-7,
        )
        exported_gp, exported_gq = forward_local_hierarchy(
            provisional.positions.astype(np.float64),
            provisional.rotations.astype(np.float64),
            np.asarray(G1_SKELETON_PARENTS, np.int32),
        )
        kinematics = G1Kinematics(G1_XML)
        raw_source = load_authenticated_grail_slope(
            authored_slope_args().slope_robot)
        source_gp, source_gq = kinematics.world_from_qpos(raw_source.qpos)
        source_gp, source_gq = change_basis_zup_to_yup(source_gp, source_gq)
        expected_gp = resample_vectors(source_gp, 25.0, 60.0)
        expected_gq = resample_quaternions_wxyz(source_gq, 25.0, 60.0)
        hips = G1_SKELETON_NAMES.index("Hips")
        np.testing.assert_allclose(
            exported_gp[:, hips], expected_gp[:, hips - 1],
            rtol=0.0, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.abs(np.sum(exported_gq[:, hips] * expected_gq[:, hips - 1], axis=1)),
            1.0, rtol=0.0, atol=2e-6,
        )

    def test_exact_mesh_features_match_frozen_slope_columns(self):
        candidate = self.candidate()
        if candidate is None:
            return
        features = candidate.source.terrain_features[3:]
        np.testing.assert_array_equal(
            np.min(features, axis=0), FROZEN_FEATURE_MINIMUM)
        np.testing.assert_array_equal(
            np.max(features, axis=0), FROZEN_FEATURE_MAXIMUM)
        np.testing.assert_allclose(
            np.std(features.astype(np.float64), axis=0),
            FROZEN_FEATURE_STD,
            rtol=0.0, atol=1e-12,
        )

    def test_feature_gate_rejects_a_std_drift_below_old_rounded_tolerance(self):
        validator = getattr(
            builder, "_require_authored_slope_feature_statistics", None)
        self.assertTrue(callable(validator), "exact feature-stat gate is missing")
        if validator is None:
            return
        candidate = self.candidate()
        if candidate is None:
            return
        features = candidate.source.terrain_features[3:].copy()
        original_minimum = np.min(features, axis=0)
        original_maximum = np.max(features, axis=0)
        features[100, 0] += np.float32(1e-5)
        np.testing.assert_array_equal(np.min(features, axis=0), original_minimum)
        np.testing.assert_array_equal(np.max(features, axis=0), original_maximum)
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            validator(features)

    def test_reconstruction_basis_calibration_and_inverse_round_trip(self):
        candidate = self.candidate()
        if candidate is None:
            return
        terrain = candidate.terrain
        self.assertEqual(terrain.support_calibration_m, 0.012000000104308128)
        self.assertEqual(terrain.exterior_source_height_m, 0.0)
        self.assertEqual(terrain.height(100.0, 100.0), -0.012000000104308128)
        reconstructed = terrain.source_to_runtime(terrain.source_vertices)
        recovered = terrain.runtime_to_source(reconstructed)
        self.assertLess(
            float(np.max(np.linalg.norm(
                recovered - terrain.source_vertices, axis=1))),
            1e-9,
        )
        self.assertLess(
            candidate.provenance["inverse_round_trip_max_error_m"], 1e-9)

    def test_synthetic_tilted_plane_locks_basis_axes_and_height_sign(self):
        factory = getattr(
            terrain_module.GrailTerrain, "from_reconstructed_mesh", None)
        self.assertTrue(callable(factory), "exact reconstructed-mesh adapter is missing")
        if factory is None:
            return
        # Native Z-up plane z = 2*x + 3*y becomes Holden
        # y = 2*x - 3*z. The support calibration shifts terrain Y only.
        vertices = np.array([
            [-1.0, -1.0, -5.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 5.0],
            [-1.0, 1.0, 1.0],
        ], np.float64)
        terrain = factory(
            vertices,
            np.array([4], np.int32),
            np.array([0, 1, 2, 3], np.int32),
            np.eye(3), np.zeros(3), support_calibration_m=0.125,
        )
        self.assertAlmostEqual(terrain.height(0.25, 0.50), -1.125, places=12)
        self.assertAlmostEqual(terrain.height(-0.25, -0.50), 0.875, places=12)
        self.assertEqual(terrain.height(2.0, 2.0), -0.125)


if __name__ == "__main__":
    unittest.main()
