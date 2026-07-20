import tempfile
import unittest
from pathlib import Path

import numpy as np

from resources.g1_interaction_builder.proposal_artifact import (
    MAGIC,
    ProposalArtifact,
    certify_proposals,
    deserialize_proposal_artifact,
    read_proposal_artifact,
    serialize_proposal_artifact,
    write_proposal_artifact,
)


def _smooth_artifact():
    proposals = np.zeros((32, 16, 4), dtype=np.float32)
    proposals[..., 3] = 1.0
    proposals[..., 0] = np.linspace(0.0, 0.2, 16)
    return ProposalArtifact(
        2,
        7,
        2_026_071_901,
        bytes(range(32)),
        np.arange(24, dtype=np.float32) * 0.25,
        proposals,
        np.arange(32, dtype=np.uint64),
        np.ones(32, bool),
    )


class ProposalArtifactTests(unittest.TestCase):
    def test_certification_rejects_a_single_discontinuous_proposal(self):
        proposals = np.zeros((32, 16, 4), dtype=np.float32)
        proposals[..., 3] = 1.0
        proposals[..., 0] = np.linspace(0.0, 0.2, 16)
        proposals[0, 8, 0] = 1.0
        accepted = certify_proposals(proposals)
        self.assertFalse(bool(accepted[0]))
        self.assertTrue(bool(accepted[1]))

    def test_round_trip_preserves_strict_artifact(self):
        proposals = np.zeros((32, 16, 4), dtype=np.float32)
        proposals[..., 3] = 1.0
        proposals[..., 0] = np.linspace(0.0, 0.2, 16)
        artifact = ProposalArtifact(
            2, 7, 2_026_071_901, bytes(range(32)),
            np.zeros(24, np.float32), proposals,
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
            2, artifact.request_id, artifact.batch_seed,
            artifact.checkpoint_sha256, artifact.condition,
            artifact.proposals, seeds, artifact.accepted)
        with self.assertRaises(ValueError):
            serialize_proposal_artifact(artifact)

    def test_write_rejects_accepted_discontinuous_proposal(self):
        artifact = _smooth_artifact()
        proposals = np.array(artifact.proposals)
        proposals[3, 8, 0] = 1.0  # violates 0.08 m translation step
        artifact = ProposalArtifact(
            2, artifact.request_id, artifact.batch_seed,
            artifact.checkpoint_sha256, artifact.condition,
            proposals, artifact.seeds, artifact.accepted)
        with self.assertRaises(ValueError):
            serialize_proposal_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
