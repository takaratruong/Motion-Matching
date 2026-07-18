from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.hands import (
    LEFT_HAND_JOINT_ORDER,
    NEUTRAL_HAND_TARGETS,
    RIGHT_HAND_JOINT_ORDER,
    hand_targets_record,
    measure_hand_tracking,
    parse_hand_targets_record,
    resolve_hand_targets,
)
from mm_sonic.joints import ContractError


class Dex3HandContractTests(unittest.TestCase):
    def test_neutral_profile_has_exact_mirrored_motor_order_and_targets(self) -> None:
        self.assertEqual(
            LEFT_HAND_JOINT_ORDER,
            ("left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
             "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
             "left_hand_middle_1_joint", "left_hand_index_0_joint",
             "left_hand_index_1_joint"),
        )
        self.assertEqual(
            RIGHT_HAND_JOINT_ORDER,
            ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
             "right_hand_thumb_2_joint", "right_hand_index_0_joint",
             "right_hand_index_1_joint", "right_hand_middle_0_joint",
             "right_hand_middle_1_joint"),
        )
        np.testing.assert_array_equal(
            NEUTRAL_HAND_TARGETS.left_f32,
            np.asarray((0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875), dtype="<f4"),
        )
        np.testing.assert_array_equal(
            NEUTRAL_HAND_TARGETS.right_f32,
            np.asarray((0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875), dtype="<f4"),
        )
        record = hand_targets_record(NEUTRAL_HAND_TARGETS)
        self.assertEqual(record["profile"], "dex3-relaxed-fist-v1")
        self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            parse_hand_targets_record(record),
            NEUTRAL_HAND_TARGETS,
        )

    def test_one_side_override_is_nonpersistent_data(self) -> None:
        left = (0.1, 0.2, 0.8, -0.7, -0.8, -0.6, -0.7)
        overridden = resolve_hand_targets(left_hand_joints=left)
        self.assertEqual(overridden.left, left)
        self.assertEqual(overridden.right, NEUTRAL_HAND_TARGETS.right)
        self.assertEqual(resolve_hand_targets(), NEUTRAL_HAND_TARGETS)

    def test_invalid_override_fails_closed(self) -> None:
        for value in ((0.0,) * 6, (0.0,) * 6 + (float("nan"),),
                      (0.0,) * 6 + (True,), (0.0,) * 6 + (9.0,)):
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    resolve_hand_targets(left_hand_joints=value)

    def test_tracking_uses_final_second_and_all_fourteen_joints(self) -> None:
        target = np.asarray(NEUTRAL_HAND_TARGETS.left + NEUTRAL_HAND_TARGETS.right)
        states = []
        for index in range(101):
            qpos = np.zeros(21)
            qpos[7:21] = target + (0.5 if index < 50 else 0.05)
            states.append(SimpleNamespace(sim_time_s=0.02 * index, qpos=qpos))
        report = measure_hand_tracking(
            states,
            qpos_addresses=tuple(range(7, 21)),
            joint_ranges=tuple((-2.0, 2.0) for _ in range(14)),
            targets=NEUTRAL_HAND_TARGETS,
            final_seconds=1.0,
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.median_absolute_error_rad), 14)
        self.assertLessEqual(max(report.median_absolute_error_rad), 0.20)


if __name__ == "__main__":
    unittest.main()
