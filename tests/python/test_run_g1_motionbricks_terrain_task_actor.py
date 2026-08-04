import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

from mm_sonic.joints import ContractError


_PATH = (
    Path(__file__).resolve().parents[2]
    / "resources"
    / "run_g1_motionbricks_terrain_task_actor.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "run_g1_motionbricks_terrain_task_actor", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _qpos(root_x: float) -> np.ndarray:
    output = np.zeros((4, 36), dtype=np.float32)
    output[:, 0] = root_x + np.arange(4) * 0.01
    output[:, 2] = 0.8
    output[:, 3] = 1.0
    return output


class _FakeConverter:
    def convert_mujoco_qpos_to_motion_transforms(
        self, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames = qpos.shape[:2]
        positions = torch.zeros((batch, frames, 3, 3))
        positions[:, :, 0] = qpos[:, :, :3]
        positions[:, :, 1] = qpos[:, :, :3] + torch.tensor(
            (0.1, 0.2, 0.3)
        )
        positions[:, :, 2] = qpos[:, :, :3] + torch.tensor(
            (-0.1, 0.4, -0.3)
        )
        rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(
            batch, frames, 3, 1, 1
        )
        return positions, rotations


class _FakeAgent:
    _device = "cpu"
    _converter = _FakeConverter()


class MotionBricksTerrainTaskActorRunnerTests(unittest.TestCase):
    def test_constraints_preserve_four_context_and_target_frames(self):
        context = _qpos(0.0)
        target = _qpos(1.0)

        constraints = _MODULE.motionbricks_constraints(
            _FakeAgent(), context, target
        )

        self.assertEqual(
            tuple(constraints["context_global_joint_positions"].shape),
            (1, 4, 3, 3),
        )
        self.assertEqual(
            tuple(constraints["target_global_joint_positions"].shape),
            (1, 4, 3, 3),
        )
        positions, _ = _FakeAgent._converter.convert_mujoco_qpos_to_motion_transforms(
            torch.tensor(target)[None]
        )
        expected_root = positions[:, :, 0] * torch.tensor(
            (1.0, 0.0, 1.0)
        )
        torch.testing.assert_close(
            constraints["target_global_root_positions"], expected_root
        )
        torch.testing.assert_close(
            constraints["target_global_joint_positions"]
            + expected_root[:, :, None],
            positions,
        )
        torch.testing.assert_close(
            constraints["target_root_headings"], torch.zeros((1, 4))
        )

    def test_constraints_reject_malformed_qpos(self):
        with self.assertRaisesRegex(ContractError, "qpos"):
            _MODULE.motionbricks_constraints(
                _FakeAgent(), np.zeros((3, 36)), _qpos(1.0)
            )

    def test_normalizes_only_small_generated_quaternion_roundoff(self):
        qpos = _qpos(0.0).astype(np.float64)
        qpos[2, 3:7] *= 1.00011

        normalized = _MODULE.normalize_generated_qpos(qpos)

        np.testing.assert_allclose(
            np.linalg.norm(normalized[:, 3:7], axis=1), 1.0
        )
        qpos[2, 3:7] = np.array((0.9, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(ContractError, "quaternion"):
            _MODULE.normalize_generated_qpos(qpos)

    def test_allowed_token_masks_select_each_checkpoint_duration(self):
        masks = _MODULE.allowed_token_masks(6, 8)

        self.assertEqual(tuple(masks.keys()), (6, 7, 8))
        torch.testing.assert_close(
            masks[7], torch.tensor(((0, 1, 0),), dtype=torch.int64)
        )

    def test_select_candidate_rejects_penetration_before_smoothness(self):
        bad = {
            "candidate_id": "bad",
            "minimum_sole_clearance_m": -0.026,
            "maximum_stance_contact_error_m": 0.0,
            "maximum_stance_horizontal_step_m": 0.0,
            "maximum_heading_error_rad": 0.0,
            "endpoint_root_error_m": 0.0,
            "unsupported_frame_count": 0,
            "mean_joint_acceleration": 0.0,
            "maximum_joint_step_rad": 0.0,
        }
        good = dict(bad)
        good.update(
            candidate_id="good",
            minimum_sole_clearance_m=-0.010,
            endpoint_root_error_m=0.05,
        )

        selected = _MODULE.select_candidate((bad, good))

        self.assertEqual(selected["candidate_id"], "good")

    def test_select_candidate_fails_closed_when_every_candidate_is_invalid(self):
        heading = {
            "candidate_id": "heading",
            "minimum_sole_clearance_m": 0.0,
            "maximum_stance_contact_error_m": 0.0,
            "maximum_stance_horizontal_step_m": 0.0,
            "maximum_heading_error_rad": 0.31,
            "endpoint_root_error_m": 0.0,
            "unsupported_frame_count": 0,
            "mean_joint_acceleration": 0.0,
            "maximum_joint_step_rad": 0.0,
        }
        unsupported = dict(heading)
        unsupported.update(
            candidate_id="unsupported",
            maximum_heading_error_rad=0.0,
            unsupported_frame_count=1,
        )

        with self.assertRaisesRegex(ContractError, "no valid"):
            _MODULE.select_candidate((heading, unsupported))


if __name__ == "__main__":
    unittest.main()
