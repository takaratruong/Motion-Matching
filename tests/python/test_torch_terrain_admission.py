from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest

import numpy as np

from resources.g1_torch_stair_builder.conversion import NativeMotionArrays
from resources.g1_torch_stair_builder.surface import ZUpHeightGrid
from resources.g1_torch_terrain_builder.admission import admit_candidate
from resources.g1_torch_terrain_builder.registry import (
    ResolvedSource,
    SourceSpec,
)
from resources.g1_torch_terrain_builder.terrain import TerrainEvidence


def _source(root: Path) -> ResolvedSource:
    path = root / "motion.npz"
    path.write_bytes(b"motion")
    spec = SourceSpec(
        logical_name="fixture",
        family="stair-local",
        source_adapter="native-npz",
        motion_relative_path="motion.npz",
        motion_sha256="a" * 64,
        terrain_adapter="fixed-staircase",
        geometry_relative_paths=(),
    )
    return ResolvedSource(
        spec=spec,
        motion_path=path,
        geometry_paths=(),
        source_sha256=MappingProxyType({"motion": spec.motion_sha256}),
    )


def _motion(*, foot_clearance: float = 0.035) -> NativeMotionArrays:
    frames = 60
    joint = np.zeros((frames, 29), np.float32)
    body = np.zeros((frames, 30, 3), np.float32)
    body[:, 0, 2] = 0.8
    body[:, 18, 0] = -0.1
    body[:, 19, 0] = 0.1
    body[:, (18, 19), 2] = foot_clearance
    quaternion = np.zeros((frames, 30, 4), np.float32)
    quaternion[..., 0] = 1.0
    return NativeMotionArrays(
        fps=50,
        joint_position=joint,
        joint_velocity=np.zeros_like(joint),
        body_position_world=body,
        body_quaternion_world_wxyz=quaternion,
        body_linear_velocity_world=np.zeros_like(body),
        body_angular_velocity_world=np.zeros_like(body),
    )


def _terrain() -> TerrainEvidence:
    return TerrainEvidence(
        adapter="fixed-staircase",
        grid=ZUpHeightGrid(
            origin_xy=np.array([-1.0, -1.0], np.float32),
            cell_size_m=0.1,
            height_z=np.zeros((21, 21), np.float32),
        ),
        geometry_sha256=MappingProxyType({}),
        motion_to_terrain_xy_yaw=(0.0, 0.0, 0.0),
    )


class TerrainCandidateAdmissionTests(unittest.TestCase):
    def test_aligned_motion_passes_fk_and_contact_oracles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source(root)
            motion = _motion()

            def fk(qpos):
                self.assertEqual(qpos.shape, (60, 36))
                return (
                    motion.body_position_world.copy(),
                    motion.body_quaternion_world_wxyz.copy(),
                )

            report = admit_candidate(
                source,
                motion,
                _terrain(),
                fk=fk,
                accepted_motion_hashes={},
            )

        self.assertTrue(report.accepted)
        self.assertIsNone(report.reason)
        self.assertEqual(report.contact_sample_count, 120)
        self.assertLessEqual(report.contact_height_error_m["p95"], 1e-7)
        self.assertLessEqual(report.fk_position_error_m["maximum"], 1e-7)

    def test_missing_terrain_duplicate_and_fk_mismatch_are_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source(root)
            motion = _motion()
            exact_fk = lambda _qpos: (
                motion.body_position_world.copy(),
                motion.body_quaternion_world_wxyz.copy(),
            )
            missing = admit_candidate(
                source,
                motion,
                None,
                fk=exact_fk,
                accepted_motion_hashes={},
            )
            duplicate = admit_candidate(
                source,
                motion,
                _terrain(),
                fk=exact_fk,
                accepted_motion_hashes={"a" * 64: "earlier"},
            )

            def shifted_fk(_qpos):
                return (
                    motion.body_position_world + 0.1,
                    motion.body_quaternion_world_wxyz.copy(),
                )

            mismatch = admit_candidate(
                source,
                motion,
                _terrain(),
                fk=shifted_fk,
                accepted_motion_hashes={},
            )

        self.assertEqual(missing.reason, "missing_authoritative_terrain")
        self.assertEqual(duplicate.reason, "exact_duplicate")
        self.assertEqual(duplicate.duplicate_of, "earlier")
        self.assertEqual(mismatch.reason, "fk_mismatch")

    def test_insufficient_and_misaligned_contact_evidence_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source(root)
            high = _motion(foot_clearance=1.0)
            high_fk = lambda _qpos: (
                high.body_position_world.copy(),
                high.body_quaternion_world_wxyz.copy(),
            )
            insufficient = admit_candidate(
                source,
                high,
                _terrain(),
                fk=high_fk,
                accepted_motion_hashes={},
            )

            wrong = _motion(foot_clearance=0.10)
            wrong_fk = lambda _qpos: (
                wrong.body_position_world.copy(),
                wrong.body_quaternion_world_wxyz.copy(),
            )
            misaligned = admit_candidate(
                source,
                wrong,
                _terrain(),
                fk=wrong_fk,
                accepted_motion_hashes={},
            )

        self.assertEqual(insufficient.reason, "insufficient_contact_samples")
        self.assertEqual(misaligned.reason, "contact_alignment_failed")
        self.assertEqual(misaligned.contact_sample_count, 120)

    def test_fk_gate_is_strict_on_locomotion_chain_not_model_variant_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source(root)
            motion = _motion()

            def shifted_arms(_qpos):
                position = motion.body_position_world.copy()
                position[
                    :,
                    (6, 9, 12, 13, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
                ] += 0.1
                return position, motion.body_quaternion_world_wxyz.copy()

            arms = admit_candidate(
                source,
                motion,
                _terrain(),
                fk=shifted_arms,
                accepted_motion_hashes={},
            )

            def shifted_foot(_qpos):
                position = motion.body_position_world.copy()
                position[:, 18] += 0.1
                return position, motion.body_quaternion_world_wxyz.copy()

            foot = admit_candidate(
                source,
                motion,
                _terrain(),
                fk=shifted_foot,
                accepted_motion_hashes={},
            )

        self.assertTrue(arms.accepted)
        self.assertGreater(
            arms.fk_full_body_position_error_m["maximum"], 0.09
        )
        self.assertLess(
            arms.fk_position_error_m["maximum"], 1e-7
        )
        self.assertEqual(foot.reason, "fk_mismatch")

    def test_flat_only_contacts_do_not_authenticate_elevated_terrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _source(root)
            motion = _motion()
            height = np.zeros((21, 21), np.float32)
            height[:, 11:] = 0.2
            terrain = TerrainEvidence(
                adapter="fixed-staircase",
                grid=ZUpHeightGrid(
                    origin_xy=np.array([-1.0, -1.0], np.float32),
                    cell_size_m=0.1,
                    height_z=height,
                ),
                geometry_sha256=MappingProxyType({}),
                motion_to_terrain_xy_yaw=(0.0, 0.0, 0.0),
            )
            exact_fk = lambda _qpos: (
                motion.body_position_world.copy(),
                motion.body_quaternion_world_wxyz.copy(),
            )
            report = admit_candidate(
                source,
                motion,
                terrain,
                fk=exact_fk,
                accepted_motion_hashes={},
            )

        self.assertFalse(report.accepted)
        self.assertEqual(
            report.reason, "insufficient_elevated_contact_samples"
        )
        self.assertEqual(report.elevated_contact_sample_count, 0)


if __name__ == "__main__":
    unittest.main()
