import unittest
import numpy as np
from resources.g1_terrain_builder.kinematics import (
    G1Kinematics, change_basis_zup_to_yup, convert_source_clip, world_to_local,
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
        source = SourceClip(
            "synthetic", 25.0, qpos, np.arange(5), "flat")
        clip, skeleton, report = convert_source_clip(source, kin)
        self.assertEqual(clip.positions.shape, (11, 31, 3))
        self.assertEqual(skeleton.names[0], "Simulation")
        self.assertEqual(skeleton.parents[0], -1)
        self.assertLessEqual(report["fk_max_error_m"], 0.001)
        self.assertLessEqual(report["duration_error_s"], 1.0/60.0)


if __name__ == "__main__":
    unittest.main()
