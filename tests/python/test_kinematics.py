import unittest
import numpy as np
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics, change_basis_zup_to_yup, convert_source_clip,
    forward_local_hierarchy, heading_quaternions, world_to_local,
)
from resources.g1_terrain_builder.schema import SourceClip

G1_XML = "/home/ubuntu/projects/mjx-diffphysics/env/g1/assets/g1_29dof.xml"


class KinematicsTests(unittest.TestCase):
    def test_change_of_basis_maps_z_to_y(self):
        p = np.array([[[0.0, 0.0, 1.0]]])
        q = np.array([[[1.0, 0.0, 0.0, 0.0]]])
        py, qy = change_basis_zup_to_yup(p, q)
        np.testing.assert_allclose(py[0, 0], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(qy, axis=-1), 1, atol=1e-7)

    def test_change_of_basis_conjugates_nontrivial_rotation(self):
        c = 2**-0.5
        p = np.zeros((1, 1, 3))
        qz = np.array([[[c, 0, 0, c]]], np.float64)
        _, qy = change_basis_zup_to_yup(p, qz)
        expected = np.array([c, 0, c, 0])
        self.assertLess(
            min(
                np.linalg.norm(qy[0, 0] - expected),
                np.linalg.norm(qy[0, 0] + expected),
            ),
            1e-7,
        )

    def test_heading_handles_antiparallel_forward(self):
        forward = np.array([
            [0, 0, 1], [0, 0, -1], [1, 0, 0],
        ], np.float64)
        q = heading_quaternions(forward)
        self.assertTrue(np.all(np.isfinite(q)))
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-7)
        c = 2**-0.5
        np.testing.assert_allclose(
            q, [[1, 0, 0, 0], [0, 0, 1, 0], [c, 0, c, 0]],
            atol=1e-7,
        )

    def test_world_local_round_trip_is_submillimeter(self):
        kin = G1Kinematics(G1_XML)
        qpos = kin.keyframe_or_zero_qpos()
        gp, gq = kin.world_from_qpos(qpos[None])
        lp, lq = world_to_local(gp, gq, kin.parents)
        rp, rq = kin.forward_local(lp, lq)
        self.assertLess(np.max(np.linalg.norm(rp - gp, axis=-1)), 1e-6)

    def test_convert_source_clip_prepends_simulation_bone(self):
        kin = G1Kinematics(G1_XML)
        qpos = np.tile(kin.keyframe_or_zero_qpos(), (5, 1))
        qpos[:, 0] = np.arange(5) * 0.01
        qpos[:, 7] = np.linspace(0.0, 0.1, 5)
        source = SourceClip(
            "synthetic", 25.0, qpos, np.arange(5), "flat")
        clip, skeleton, report = convert_source_clip(source, kin)
        self.assertEqual(clip.positions.shape, (5, 31, 3))
        self.assertEqual(skeleton.names[0], "Simulation")
        self.assertEqual(skeleton.parents[0], -1)
        self.assertLessEqual(report["fk_max_error_m"], 0.001)
        self.assertLessEqual(report["duration_error_s"], 1.0/25.0)
        np.testing.assert_array_equal(clip.source_frames, np.arange(5))
        np.testing.assert_allclose(
            np.linalg.norm(clip.rotations, axis=-1), 1.0, atol=1e-4)
        self.assertTrue(np.all(
            np.sum(clip.rotations[1:] * clip.rotations[:-1], axis=-1)
            >= -1e-7))
        expected_gp, expected_gq = kin.world_from_qpos(qpos)
        expected_gp, _ = change_basis_zup_to_yup(expected_gp, expected_gq)
        exported_gp, _ = forward_local_hierarchy(
            clip.positions.astype(np.float64),
            clip.rotations.astype(np.float64), skeleton.parents)
        self.assertLess(np.max(np.linalg.norm(
            exported_gp[:, 1:] - expected_gp, axis=-1)), 0.001)


if __name__ == "__main__":
    unittest.main()
