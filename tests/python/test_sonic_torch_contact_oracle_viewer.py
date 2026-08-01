import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.torch_contact_oracle_rollout import (
    ORACLE_ARRAY_SHAPES,
    oracle_arrays_sha256,
)
from mm_sonic.torch_contact_oracle_viewer import (
    apply_oracle_frame,
    load_oracle_route,
)


class _Mujoco:
    forwarded = False

    @classmethod
    def mj_forward(cls, _model, _data):
        cls.forwarded = True


class ContactOracleViewerTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        route = root / "cross-tread-left-to-right"
        route.mkdir(parents=True)
        arrays = {}
        for name, tail in ORACLE_ARRAY_SHAPES.items():
            if name == "selected_clip_path":
                arrays[name] = np.asarray(["clip-a", "clip-b"])
            elif name in {
                "command_segment_index",
                "selected_source_frame",
                "selected_action_index",
                "planned_horizon",
                "plan_time_ns",
            }:
                arrays[name] = np.zeros((2, *tail), dtype=np.int64)
            else:
                arrays[name] = np.zeros((2, *tail), dtype=np.float64)
        arrays["qpos"][:, 3] = 1.0
        arrays["selected_source_frame"][:] = (7, 8)
        arrays["selected_action_index"][:] = 3
        arrays["planned_horizon"][:] = 4
        arrays["landing_error_m"][:] = 0.01
        np.savez_compressed(route / "rollout.npz", **arrays)
        (route / "diagnostics.json").write_text(
            json.dumps(
                {
                    "schema": "g1-contact-space-oracle-route/v1",
                    "route": route.name,
                    "arrays_sha256": oracle_arrays_sha256(arrays),
                }
            )
        )
        return route

    def test_applies_saved_qpos_and_exposes_oracle_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = load_oracle_route(self._fixture(Path(temporary)))
            model = SimpleNamespace(nq=36)
            data = SimpleNamespace(qpos=np.zeros(36), qvel=np.ones(35))
            _Mujoco.forwarded = False

            overlay = apply_oracle_frame(_Mujoco, model, data, saved, 1)

            np.testing.assert_array_equal(data.qpos, saved.arrays["qpos"][1])
            self.assertTrue(_Mujoco.forwarded)
            self.assertIn("action=3", overlay)
            self.assertIn("horizon=4", overlay)
            self.assertIn("landing_error=0.0100", overlay)

    def test_rejects_hash_mismatch_and_out_of_range_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            route = self._fixture(Path(temporary))
            diagnostics = json.loads((route / "diagnostics.json").read_text())
            diagnostics["arrays_sha256"] = "0" * 64
            (route / "diagnostics.json").write_text(json.dumps(diagnostics))
            with self.assertRaisesRegex(Exception, "hash"):
                load_oracle_route(route)

            route = self._fixture(Path(temporary) / "second")
            saved = load_oracle_route(route)
            with self.assertRaisesRegex(Exception, "frame"):
                apply_oracle_frame(
                    _Mujoco,
                    SimpleNamespace(nq=36),
                    SimpleNamespace(qpos=np.zeros(36), qvel=np.zeros(35)),
                    saved,
                    2,
                )


if __name__ == "__main__":
    unittest.main()
