import tempfile
import unittest
from pathlib import Path

import numpy as np

from resources.g1_interaction_builder.proposal_artifact import (
    MAGIC,
    ProposalArtifact,
    certify_proposals,
    deserialize_proposal_artifact,
    project_object_local_funnels,
    read_proposal_artifact,
    serialize_proposal_artifact,
    write_proposal_artifact,
)


def _smooth_artifact():
    condition = np.arange(24, dtype=np.float32) * 0.25
    condition[18:22] = (0.42, -0.17, 0.6, 0.8)
    proposals = np.zeros((32, 16, 4), dtype=np.float32)
    proposals[..., 2] = 0.6
    proposals[..., 3] = 0.8
    proposals[..., 0] = np.linspace(0.42, 0.62, 16)
    proposals[..., 1] = -0.17
    return ProposalArtifact(
        3,
        7,
        2_026_071_901,
        bytes(range(32)),
        condition,
        proposals,
        np.arange(32, dtype=np.uint64),
        np.ones(32, bool),
    )


class ProposalArtifactTests(unittest.TestCase):
    def test_projection_preserves_endpoints_and_regularizes_interior(self):
        proposals = np.zeros((32, 16, 4), dtype=np.float32)
        proposals[..., 3] = 1.0
        proposals[..., 0] = np.linspace(0.0, 0.4, 16)
        for index in range(1, 15):
            proposals[:, index, 1] = 0.15 if index % 2 else -0.15
            yaw = 0.5 if index % 2 else -0.5
            proposals[:, index, 2] = np.sin(yaw)
            proposals[:, index, 3] = np.cos(yaw)
        original_start = proposals[:, 0].copy()
        original_end = proposals[:, -1].copy()
        self.assertFalse(bool(certify_proposals(proposals).any()))

        projected = project_object_local_funnels(proposals)

        np.testing.assert_array_equal(projected[:, 0], original_start)
        np.testing.assert_array_equal(projected[:, -1], original_end)
        self.assertTrue(bool(certify_proposals(projected).all()))

    def test_certification_rejects_a_single_discontinuous_proposal(self):
        proposals = np.zeros((32, 16, 4), dtype=np.float32)
        proposals[..., 2] = 0.6
        proposals[..., 3] = 0.8
        proposals[..., 0] = np.linspace(0.0, 0.2, 16)
        proposals[0, 8, 0] = 1.0
        accepted = certify_proposals(proposals)
        self.assertFalse(bool(accepted[0]))
        self.assertTrue(bool(accepted[1]))

    def test_round_trip_preserves_strict_artifact(self):
        condition = np.zeros(24, np.float32)
        condition[18:22] = (0.42, -0.17, 0.6, 0.8)
        proposals = np.zeros((32, 16, 4), dtype=np.float32)
        proposals[..., 2] = 0.6
        proposals[..., 3] = 0.8
        proposals[..., 0] = np.linspace(0.42, 0.62, 16)
        proposals[..., 1] = -0.17
        artifact = ProposalArtifact(
            3, 7, 2_026_071_901, bytes(range(32)),
            condition, proposals,
            np.arange(32, dtype=np.uint64), np.ones(32, bool))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.npz"
            write_proposal_artifact(path, artifact)
            loaded = read_proposal_artifact(path)
        np.testing.assert_array_equal(loaded.proposals, proposals)
        self.assertEqual(int(loaded.accepted.sum()), 32)
        self.assertEqual(loaded.request_id, 7)
        self.assertEqual(loaded.batch_seed, 2_026_071_901)
        self.assertEqual(loaded.checkpoint_sha256, bytes(range(32)))

    def test_serialized_blob_is_canonical_little_endian_length(self):
        blob = serialize_proposal_artifact(_smooth_artifact())
        # header (28) + IDs (16) + digest (32) + condition (96) +
        # proposals (8192) + seeds (256) + flags (32)
        self.assertEqual(len(blob), 8652)
        self.assertEqual(blob[:8], MAGIC)

    def test_truncated_blob_is_rejected(self):
        blob = serialize_proposal_artifact(_smooth_artifact())
        with self.assertRaises(ValueError):
            deserialize_proposal_artifact(blob[:-1])

    def test_trailing_bytes_are_rejected(self):
        blob = serialize_proposal_artifact(_smooth_artifact())
        with self.assertRaises(ValueError):
            deserialize_proposal_artifact(blob + b"\x00")

    def test_bad_magic_is_rejected(self):
        blob = bytearray(serialize_proposal_artifact(_smooth_artifact()))
        blob[0] = ord("X")
        with self.assertRaises(ValueError):
            deserialize_proposal_artifact(bytes(blob))

    def test_write_rejects_duplicate_seeds(self):
        artifact = _smooth_artifact()
        seeds = np.arange(32, dtype=np.uint64)
        seeds[5] = seeds[4]
        artifact = ProposalArtifact(
            3, artifact.request_id, artifact.batch_seed,
            artifact.checkpoint_sha256, artifact.condition,
            artifact.proposals, seeds, artifact.accepted)
        with self.assertRaises(ValueError):
            serialize_proposal_artifact(artifact)

    def test_write_rejects_accepted_discontinuous_proposal(self):
        artifact = _smooth_artifact()
        proposals = np.array(artifact.proposals)
        proposals[3, 8, 0] = 1.0  # violates 0.08 m translation step
        artifact = ProposalArtifact(
            3, artifact.request_id, artifact.batch_seed,
            artifact.checkpoint_sha256, artifact.condition,
            proposals, artifact.seeds, artifact.accepted)
        with self.assertRaises(ValueError):
            serialize_proposal_artifact(artifact)

    def test_write_rejects_execution_start_not_equal_to_condition_entry(self):
        artifact = _smooth_artifact()
        proposals = np.array(artifact.proposals)
        proposals[4, 0, 0] = np.nextafter(proposals[4, 0, 0], np.float32(1.0))
        malformed = ProposalArtifact(
            3, artifact.request_id, artifact.batch_seed,
            artifact.checkpoint_sha256, artifact.condition,
            proposals, artifact.seeds, artifact.accepted)
        with self.assertRaisesRegex(ValueError, "object-local entry"):
            serialize_proposal_artifact(malformed)


if __name__ == "__main__":
    unittest.main()
