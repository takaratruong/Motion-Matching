import unittest
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
from mm_sonic.grail_terrain_source import resample_grail_motion


SLOPE_NAME = "terrain_slopes__slope_000__000"
SLOPE_ROOT = "/home/ubuntu/datasets/GRAIL/data/slope"
G1_XML = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"


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

    def test_exact_250_at_25_to_598_at_60_provenance_and_span(self):
        candidate = self.candidate()
        if candidate is None:
            return
        provisional = candidate.provisional_clip
        self.assertEqual(len(candidate.source.qpos), 250)
        self.assertEqual(candidate.source.fps, 25.0)
        self.assertEqual(len(provisional.positions), 598)
        left, right, alpha = resample_map(250, 25.0, 60.0)
        np.testing.assert_array_equal(provisional.source_left_indices, left)
        np.testing.assert_array_equal(provisional.source_right_indices, right)
        np.testing.assert_array_equal(provisional.source_alpha, alpha)
        self.assertEqual(candidate.receipt["source_span_s"], 9.96)
        self.assertEqual(candidate.receipt["output_span_s"], 9.95)
        self.assertEqual(
            candidate.receipt["output_shortfall_s"],
            0.010000000000001563,
        )
        self.assertEqual(candidate.receipt["rejected_output_ranges"], [[0, 3]])
        self.assertEqual(len(candidate.admitted_clip.positions), 595)
        np.testing.assert_array_equal(
            candidate.admitted_clip.source_left_indices,
            provisional.source_left_indices[3:],
        )

    def test_simulation_is_planar_while_fk_preserves_native_global_hips(self):
        candidate = self.candidate()
        if candidate is None:
            return
        provisional = candidate.provisional_clip
        np.testing.assert_array_equal(
            provisional.positions[:, 0, 1], np.zeros(598, np.float32))
        np.testing.assert_allclose(
            provisional.rotations[:, 0][:, (1, 3)], 0.0,
            rtol=0.0, atol=1e-7,
        )
        exported_gp, exported_gq = forward_local_hierarchy(
            provisional.positions.astype(np.float64),
            provisional.rotations.astype(np.float64),
            candidate.skeleton.parents,
        )
        kinematics = G1Kinematics(G1_XML)
        source_gp, source_gq = kinematics.world_from_qpos(candidate.source.qpos)
        source_gp, source_gq = change_basis_zup_to_yup(source_gp, source_gq)
        expected_gp = resample_vectors(source_gp, 25.0, 60.0)
        expected_gq = resample_quaternions_wxyz(source_gq, 25.0, 60.0)
        hips = candidate.skeleton.names.index("Hips")
        np.testing.assert_allclose(
            exported_gp[:, hips], expected_gp[:, hips - 1],
            rtol=0.0, atol=1e-5,
        )
        np.testing.assert_allclose(
            np.abs(np.sum(exported_gq[:, hips] * expected_gq[:, hips - 1], axis=1)),
            1.0, rtol=0.0, atol=2e-6,
        )
        # The pelvis already follows the authored ramp. Terrain is never added
        # again to Simulation or Hips.
        np.testing.assert_allclose(
            candidate.admitted_clip.positions[:, 1, 1],
            provisional.positions[3:, 1, 1],
            rtol=0.0, atol=0.0,
        )

    def test_exact_mesh_features_match_frozen_slope_columns(self):
        candidate = self.candidate()
        if candidate is None:
            return
        features = candidate.admitted_clip.terrain_features.astype(np.float64)
        np.testing.assert_allclose(
            np.min(features, axis=0),
            [-0.054865, -0.096942, -0.127296, -0.127296],
            rtol=0.0, atol=5e-7,
        )
        np.testing.assert_allclose(
            np.max(features, axis=0),
            [0.053704, 0.095061, 0.127046, 0.127235],
            rtol=0.0, atol=5e-7,
        )
        np.testing.assert_allclose(
            np.std(features, axis=0),
            [0.026962, 0.045899, 0.056156, 0.065241],
            rtol=0.0, atol=5e-7,
        )

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
        self.assertLess(candidate.receipt["inverse_round_trip_max_error_m"], 1e-9)

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
