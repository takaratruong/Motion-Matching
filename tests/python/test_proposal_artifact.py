import tempfile
import unittest
from pathlib import Path

import numpy as np

from resources.g1_interaction_builder.proposal_artifact import (
    ProposalArtifact,
    certify_proposals,
    read_proposal_artifact,
    write_proposal_artifact,
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
        artifact = ProposalArtifact(1, np.zeros(18, np.float32), proposals, np.arange(32, dtype=np.uint64), np.ones(32, bool))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposals.npz"
            write_proposal_artifact(path, artifact)
            loaded = read_proposal_artifact(path)
        np.testing.assert_array_equal(loaded.proposals, proposals)
        self.assertEqual(int(loaded.accepted.sum()), 32)


if __name__ == "__main__":
    unittest.main()
