import unittest

from resources.g1_interaction_builder.funnel_dataset import load_dataset


class FunnelDatasetTests(unittest.TestCase):
    def test_full_pack_contains_only_certifiable_training_funnels(self):
        dataset = load_dataset("build/smart-pickup/full-pack")
        self.assertEqual(len(dataset.conditions), 43)


if __name__ == "__main__":
    unittest.main()
