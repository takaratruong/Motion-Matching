from __future__ import annotations

import unittest

import numpy as np
import torch

from mm_sonic.terrain_pfnn.dataset import normalize_pfnn_input, pfnn_input_sha256
from mm_sonic.terrain_pfnn.layout import (
    CLASSIC_G1_INPUT_LAYOUT_V3,
    OUTPUT_LAYOUT,
)
from mm_sonic.terrain_pfnn.training import choose_runtime_seed


class _JointStateRows:
    split = "train"
    x_mean = np.zeros(CLASSIC_G1_INPUT_LAYOUT_V3.size, np.float32)
    x_std = np.ones(CLASSIC_G1_INPUT_LAYOUT_V3.size, np.float32)
    y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
    y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

    def __init__(self) -> None:
        self.rows = [self._row(center) for center in range(10, 31)]

    @staticmethod
    def _trajectory(center: int) -> tuple[np.ndarray, np.ndarray]:
        position = np.zeros((12, 2), np.float32)
        position[:, 0] = np.linspace(-0.2, 0.8, 12) + center * 0.001
        direction = np.zeros((12, 2), np.float32)
        direction[:, 0] = 1.0
        return position, direction

    @classmethod
    def _row(cls, center: int) -> dict[str, object]:
        position, direction = cls._trajectory(center)
        target_position, target_direction = cls._trajectory(center + 1)
        body = np.full((30, 3), center * 0.001, np.float32)
        target_body = np.full((30, 3), (center + 1) * 0.001, np.float32)
        joints = np.full(29, center * 0.001, np.float32)
        target_joints = np.full(29, (center + 1) * 0.001, np.float32)
        raw = np.zeros(CLASSIC_G1_INPUT_LAYOUT_V3.size, np.float32)
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["trajectory_position"]] = position.ravel()
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["trajectory_direction"]] = direction.ravel()
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["semantic_intent"]] = np.tile(
            (1.0, 0.0), (12, 1)
        ).ravel()
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["previous_body_position"]] = body.ravel()
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["previous_body_velocity"]] = body.ravel()
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["joint_position"]] = joints
        raw[CLASSIC_G1_INPUT_LAYOUT_V3["joint_velocity"]] = np.full(
            29, 0.03, np.float32
        )
        target = np.zeros(OUTPUT_LAYOUT.size, np.float32)
        target[OUTPUT_LAYOUT["trajectory_position"]] = target_position.ravel()
        target[OUTPUT_LAYOUT["trajectory_direction"]] = target_direction.ravel()
        target[OUTPUT_LAYOUT["body_position"]] = target_body.ravel()
        target[OUTPUT_LAYOUT["body_velocity"]] = target_body.ravel()
        target[OUTPUT_LAYOUT["root_height"]] = 0.8
        target[OUTPUT_LAYOUT["joint_position"]] = target_joints
        target[OUTPUT_LAYOUT["root_planar_velocity"]] = (0.1, 0.0)
        target[OUTPUT_LAYOUT["phase_advance"]] = 0.1
        target[OUTPUT_LAYOUT["contact_logit"]] = (1.0, 0.0, 1.0, 0.0)
        return {
            "x": normalize_pfnn_input(raw, cls.x_mean, cls.x_std),
            "y": target,
            "phase": np.float32(center * 0.1),
            "clip_id": "released_walk",
            "center_frame": center,
            "split_identity": "released_walk",
            "split": "train",
            "sequence_lane": "motion",
            "terrain_class": "flat",
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


class TerrainPFNNTrainingV3SeedTests(unittest.TestCase):
    def test_seed_binds_full_v3_input_q_and_qdot(self) -> None:
        rows = _JointStateRows()

        seed = choose_runtime_seed(
            rows,
            torch.tensor([[-2.0, 2.0]] * 29),
            input_layout=CLASSIC_G1_INPUT_LAYOUT_V3,
        )

        first = np.asarray(rows[1]["x"], dtype=np.float32)
        torch.testing.assert_close(seed["normalized_input"], torch.from_numpy(first))
        torch.testing.assert_close(
            seed["joint_position"],
            torch.from_numpy(
                first[CLASSIC_G1_INPUT_LAYOUT_V3["joint_position"]]
            ),
        )
        torch.testing.assert_close(
            seed["joint_velocity"],
            torch.from_numpy(
                first[CLASSIC_G1_INPUT_LAYOUT_V3["joint_velocity"]]
            ),
        )
        self.assertEqual(
            seed["normalized_input_sha256"], pfnn_input_sha256(first)
        )


if __name__ == "__main__":
    unittest.main()
