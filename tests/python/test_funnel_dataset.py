import unittest

import numpy as np

from resources.g1_interaction_builder import funnel_dataset


class FunnelDatasetTests(unittest.TestCase):
    def test_dataset_rejects_any_row_count_mismatch(self):
        dataset = funnel_dataset.FunnelDataset(
            conditions=np.zeros((1, 24), np.float32),
            funnels=np.zeros((1, 16, 4), np.float32),
            sequence_indices=np.zeros((0,), np.int32),
        )
        with self.assertRaisesRegex(ValueError, "row counts"):
            dataset.validate()

    def test_object_local_entry_funnel_is_world_transform_invariant(self):
        convert = getattr(funnel_dataset, "object_local_execution", None)
        self.assertTrue(callable(convert))
        positions = np.zeros((16, 3), dtype=np.float32)
        positions[:, 0] = np.linspace(-0.2, 0.4, 16, dtype=np.float32)
        positions[:, 2] = np.linspace(0.7, 0.1, 16, dtype=np.float32)
        yaws = np.linspace(-0.4, 0.2, 16, dtype=np.float32)
        object_position = np.array([0.1, 0.5, -0.3], np.float32)
        object_yaw = 0.25
        expected = convert(positions, yaws, object_position, object_yaw)

        common_yaw = -0.7
        c, s = np.cos(common_yaw), np.sin(common_yaw)
        transformed = positions.copy()
        transformed[:, 0] = c * positions[:, 0] - s * positions[:, 2] + 1.25
        transformed[:, 2] = s * positions[:, 0] + c * positions[:, 2] - 0.8
        transformed_object = object_position.copy()
        transformed_object[0] = c * object_position[0] - s * object_position[2] + 1.25
        transformed_object[2] = s * object_position[0] + c * object_position[2] - 0.8
        actual = convert(
            transformed,
            yaws + common_yaw,
            transformed_object,
            object_yaw + common_yaw,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.0)

    def test_full_pack_contains_only_certifiable_training_funnels(self):
        dataset = funnel_dataset.load_dataset("build/smart-pickup/full-pack")
        self.assertEqual(len(dataset.conditions), 1548)

    def test_entry_condition_and_outward_endpoint_contract(self):
        self.assertEqual(
            tuple(getattr(funnel_dataset, "KNOT_FRAME_OFFSETS", ())),
            (0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 49, 54, 59, 64, 69, 74),
        )
        dataset = funnel_dataset.load_dataset("build/smart-pickup/full-pack")
        self.assertEqual(dataset.conditions.shape[1], 24)
        expected_entry = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        np.testing.assert_array_equal(
            dataset.funnels[:, 15],
            np.broadcast_to(expected_entry, dataset.funnels[:, 15].shape),
        )
        self.assertTrue(np.isfinite(dataset.conditions).all())
        self.assertTrue(
            np.allclose(
                np.linalg.norm(dataset.conditions[:, 20:22], axis=1),
                1.0,
                atol=2e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
