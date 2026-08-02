import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.torch_terrain_contact_ablation_viewer import (
    apply_contact_preview_frame,
    load_contact_preview_catalog,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class _Mujoco:
    @staticmethod
    def mj_forward(model, data):
        del model
        data.forwarded = True


class _Data:
    def __init__(self):
        self.qpos = np.zeros(36)
        self.qvel = np.ones(35)
        self.forwarded = False


class ContactAblationViewerTests(unittest.TestCase):
    def test_loader_chooses_first_accepted_action_and_exact_arrays(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            state_id = "a" * 64
            state = {
                "state_id": state_id,
                "route_name": "turn-90-middle-left",
                "route_frame": 120,
                "reason": "elevated-direction-change",
                "accepted_action_indices": [7],
                "best_action_index": 3,
                "actions": [
                    {"action_index": 3, "success": True, "accepted": False},
                    {"action_index": 7, "success": True, "accepted": True},
                ],
            }
            identity = {
                "schema": "g1-terrain-contact-composition-ablation/v1",
                "states": [state],
            }
            summary = {
                **identity,
                "deterministic_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
            }
            (root / "summary.json").write_bytes(_canonical(summary) + b"\n")
            state_root = root / "states" / state_id
            state_root.mkdir(parents=True)
            qpos = np.zeros((4, 36), np.float64)
            qpos[:, 3] = 1.0
            projected = qpos.copy()
            projected[:, 0] = np.arange(4)
            np.savez_compressed(
                state_root / "contact-previews.npz",
                action_7_contact_qpos=qpos,
                action_7_projected_qpos=projected,
                action_7_contact_feet=np.zeros((4, 2, 3)),
                action_7_projected_feet=np.zeros((4, 2, 3)),
                action_7_support=np.ones((4, 2), dtype=bool),
            )

            catalog = load_contact_preview_catalog(root)

        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog[0].action_index, 7)
        np.testing.assert_array_equal(catalog[0].projected_qpos, projected)
        self.assertFalse(catalog[0].projected_qpos.flags.writeable)

    def test_apply_uses_requested_stage_without_physics(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            state_id = "b" * 64
            state = {
                "state_id": state_id,
                "route_name": "route",
                "route_frame": 1,
                "reason": "source-transition",
                "accepted_action_indices": [2],
                "best_action_index": 2,
                "actions": [
                    {"action_index": 2, "success": True, "accepted": True}
                ],
            }
            identity = {
                "schema": "g1-terrain-contact-composition-ablation/v1",
                "states": [state],
            }
            summary = {
                **identity,
                "deterministic_sha256": hashlib.sha256(_canonical(identity)).hexdigest(),
            }
            (root / "summary.json").write_bytes(_canonical(summary) + b"\n")
            state_root = root / "states" / state_id
            state_root.mkdir(parents=True)
            contact = np.zeros((2, 36)); contact[:, 3] = 1.0
            projected = contact.copy(); projected[:, 2] = 0.25
            np.savez_compressed(
                state_root / "contact-previews.npz",
                action_2_contact_qpos=contact,
                action_2_projected_qpos=projected,
                action_2_contact_feet=np.zeros((2, 2, 3)),
                action_2_projected_feet=np.zeros((2, 2, 3)),
                action_2_support=np.ones((2, 2), dtype=bool),
            )
            preview = load_contact_preview_catalog(root)[0]

        data = _Data()
        overlay = apply_contact_preview_frame(
            _Mujoco, object(), data, preview, 1, projected=True
        )
        np.testing.assert_array_equal(data.qpos, projected[1])
        np.testing.assert_array_equal(data.qvel, 0.0)
        self.assertTrue(data.forwarded)
        self.assertIn("PROJECTED", overlay)


if __name__ == "__main__":
    unittest.main()
