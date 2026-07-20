import struct
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ProposalRequestTests(unittest.TestCase):
    def _request(self):
        from resources.g1_interaction_builder.proposal_request import ProposalRequest

        return ProposalRequest(
            request_id=7,
            batch_seed=2_026_071_901,
            checkpoint_sha256=bytes(range(32)),
            condition=np.arange(24, dtype=np.float32) / 8.0,
        )

    def test_canonical_little_endian_round_trip(self):
        from resources.g1_interaction_builder.proposal_request import (
            deserialize_proposal_request,
            serialize_proposal_request,
        )

        request = self._request()
        blob = serialize_proposal_request(request)
        self.assertEqual(len(blob), 156)
        self.assertEqual(blob[:8], b"G1FREQ02")
        self.assertEqual(struct.unpack_from("<IQQ", blob, 8),
                         (2, request.request_id, request.batch_seed))
        self.assertEqual(blob[28:60], bytes(range(32)))
        loaded = deserialize_proposal_request(blob)
        self.assertEqual(loaded.request_id, request.request_id)
        self.assertEqual(loaded.batch_seed, request.batch_seed)
        self.assertEqual(loaded.checkpoint_sha256, request.checkpoint_sha256)
        np.testing.assert_array_equal(loaded.condition, request.condition)

    def test_rejects_malformed_length_and_nonfinite_condition(self):
        from resources.g1_interaction_builder.proposal_request import (
            ProposalRequest,
            deserialize_proposal_request,
            serialize_proposal_request,
        )

        blob = serialize_proposal_request(self._request())
        for malformed in (blob[:-1], blob + b"\0"):
            with self.assertRaisesRegex(ValueError, "length"):
                deserialize_proposal_request(malformed)
        bad = self._request()
        condition = bad.condition.copy()
        condition[3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            serialize_proposal_request(ProposalRequest(
                bad.request_id, bad.batch_seed, bad.checkpoint_sha256, condition))

    def test_atomic_file_round_trip(self):
        from resources.g1_interaction_builder.proposal_request import (
            read_proposal_request,
            write_proposal_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.req"
            write_proposal_request(path, self._request())
            self.assertFalse(path.with_suffix(".req.tmp").exists())
            loaded = read_proposal_request(path)
        self.assertEqual(loaded.request_id, 7)

    def test_worker_returns_matching_execution_order_artifact(self):
        from resources.g1_interaction_builder.funnel_dataset import load_dataset
        from resources.g1_interaction_builder.proposal_artifact import read_proposal_artifact
        from resources.g1_interaction_builder.proposal_request import (
            ProposalRequest,
            write_proposal_request,
        )
        from tools.run_g1_funnel_proposal_worker import run_proposal_worker

        checkpoint = Path("build/g1-funnels/checkpoint.pt")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).digest()
        condition = load_dataset("build/smart-pickup/full-pack").conditions[0]
        request = ProposalRequest(19, 2_026_071_901, checkpoint_sha256, condition)
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "attempt.req"
            response_path = Path(directory) / "attempt.funnel"
            write_proposal_request(request_path, request)
            run_proposal_worker(request_path, checkpoint, response_path, device="cpu")
            artifact = read_proposal_artifact(response_path)
        self.assertEqual(artifact.request_id, request.request_id)
        self.assertEqual(artifact.batch_seed, request.batch_seed)
        self.assertEqual(artifact.checkpoint_sha256, request.checkpoint_sha256)
        np.testing.assert_array_equal(artifact.condition, condition)
        self.assertEqual(artifact.schema_version, 3)
        np.testing.assert_array_equal(
            artifact.proposals[:, 0],
            np.broadcast_to(condition[18:22], artifact.proposals[:, 0].shape),
        )
        self.assertGreaterEqual(int(artifact.accepted.sum()), 1)


if __name__ == "__main__":
    unittest.main()
