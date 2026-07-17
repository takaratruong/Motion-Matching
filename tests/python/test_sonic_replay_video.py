from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.replay_video import (
    ReplayState,
    load_state_stream,
    validate_probe,
)


class ReplayStateContractTests(unittest.TestCase):
    def test_loads_exact_nested_state_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.jsonl"
            rows = [
                {"step": 4, "sim_time_s": 0.02,
                 "state": {"qpos": [0.0] * 50, "qvel": [0.0] * 49}},
                {"step": 8, "sim_time_s": 0.04,
                 "state": {"qpos": [1.0] * 50, "qvel": [2.0] * 49}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), "utf-8")
            states = load_state_stream(path, expected_count=2, nq=50, nv=49)
        self.assertEqual([state.step for state in states], [4, 8])
        self.assertEqual([state.sim_time_s for state in states], [0.02, 0.04])
        np.testing.assert_array_equal(states[1].qpos, np.ones(50))

    def test_rejects_noncontiguous_step_or_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.jsonl"
            path.write_text(json.dumps({
                "step": 8,
                "sim_time_s": 0.02,
                "state": {"qpos": [0.0] * 50, "qvel": [0.0] * 49},
            }) + "\n", "utf-8")
            with self.assertRaisesRegex(ContractError, "step sequence"):
                load_state_stream(path, expected_count=1, nq=50, nv=49)

    def test_rejects_nonfinite_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.jsonl"
            path.write_text(json.dumps({
                "step": 4,
                "sim_time_s": 0.02,
                "state": {"qpos": [float("nan")] + [0.0] * 49,
                          "qvel": [0.0] * 49},
            }) + "\n", "utf-8")
            with self.assertRaisesRegex(ContractError, "finite"):
                load_state_stream(path, expected_count=1, nq=50, nv=49)

    def test_probe_requires_exact_media_contract(self) -> None:
        valid = {"streams": [{"codec_name": "h264", "width": 1280,
                 "height": 720, "pix_fmt": "yuv420p", "avg_frame_rate": "50/1",
                 "nb_frames": "600"}], "format": {"duration": "12.000000"}}
        validate_probe(valid)
        invalid = json.loads(json.dumps(valid))
        invalid["streams"][0]["nb_frames"] = "599"
        with self.assertRaisesRegex(ContractError, "600 frames"):
            validate_probe(invalid)


if __name__ == "__main__":
    unittest.main()
