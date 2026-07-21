import unittest

import numpy as np

from mm_sonic.gear_action import (
    policy_action_lowcmd_target_bounds,
    policy_action_to_lowcmd_target,
)


class PolicyActionToLowCmdTargetTests(unittest.TestCase):
    def test_zero_action_recovers_exact_float32_default_pose(self):
        expected = np.asarray(
            [
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0,
                0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )

        actual = policy_action_to_lowcmd_target((0.0,) * 29)

        self.assertEqual(actual, tuple(float(value) for value in expected))

    def test_reconstructs_pinned_cpp_permutation_scale_and_float32_cast(self):
        action = np.linspace(-0.875, 0.875, 29, dtype=np.float32)
        permutation = np.asarray(
            [
                0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
            ]
        )
        natural = 10.0 * 2.0 * 3.1415926535
        stiffness = {
            "5020": 0.003609725 * natural * natural,
            "7520_14": 0.010177520 * natural * natural,
            "7520_22": 0.025101925 * natural * natural,
            "4010": 0.00425 * natural * natural,
        }
        effort = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
        kinds = (
            "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
            "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
            "7520_14", "5020", "5020", "5020", "5020", "5020", "5020",
            "5020", "4010", "4010", "5020", "5020", "5020", "5020",
            "5020", "4010", "4010",
        )
        scales = np.asarray(
            [0.25 * effort[kind] / stiffness[kind] for kind in kinds],
            dtype=np.float64,
        )
        defaults = np.asarray(
            [
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0,
                0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            ],
            dtype=np.float64,
        )
        expected = np.asarray(
            defaults + action[permutation].astype(np.float64) * scales,
            dtype=np.float32,
        )

        actual = policy_action_to_lowcmd_target(tuple(float(x) for x in action))

        self.assertEqual(actual, tuple(float(value) for value in expected))

    def test_rejects_wrong_width_boolean_and_nonfinite_values(self):
        for action in ((0.0,) * 28, (0.0,) * 28 + (True,), (0.0,) * 28 + (float("nan"),)):
            with self.subTest(action=action[-1]), self.assertRaises(ValueError):
                policy_action_to_lowcmd_target(action)

    def test_nine_decimal_bounds_include_target_from_discarded_float32_bits(self):
        actual = [np.float32(0.0)] * 29
        actual[0] = np.float32(0.006111744325608015)
        logged = tuple(f"{float(value):.9f}" for value in actual)
        target = policy_action_to_lowcmd_target(actual)

        lower, upper = policy_action_lowcmd_target_bounds(logged)

        self.assertEqual(logged[0], "0.006111744")
        self.assertNotEqual(
            target[0],
            policy_action_to_lowcmd_target(tuple(float(value) for value in logged))[0],
        )
        self.assertLessEqual(lower[0], target[0])
        self.assertGreaterEqual(upper[0], target[0])

    def test_decimal_bounds_require_exact_fixed_nine_action_fields(self):
        for value in ("0", "0.0", "+0.000000000", "nan", " 0.000000000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy_action_lowcmd_target_bounds((value,) + ("0.000000000",) * 28)


if __name__ == "__main__":
    unittest.main()
