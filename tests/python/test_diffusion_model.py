import unittest
from pathlib import Path
import tempfile

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised by the environment gate
    torch = None


@unittest.skipIf(torch is None, "torch is required for the diffusion model")
class DiffusionModelTests(unittest.TestCase):
    def test_schedule_has_frozen_ddpm_boundaries(self):
        from resources.g1_interaction_builder.diffusion import make_schedule

        schedule = make_schedule()
        self.assertEqual(schedule.betas.shape, (1000,))
        self.assertEqual(schedule.alpha_bars.shape, (1000,))
        self.assertAlmostEqual(float(schedule.betas[0]), 0.0001, places=7)
        self.assertAlmostEqual(float(schedule.betas[-1]), 0.02, places=7)
        self.assertTrue(np.all(np.diff(schedule.alpha_bars) < 0.0))

    def test_denoiser_has_fixed_condition_and_funnel_shapes(self):
        from resources.g1_interaction_builder.diffusion import FunnelDenoiser

        model = FunnelDenoiser()
        x = torch.zeros((2, 4, 16), dtype=torch.float32)
        condition = torch.zeros((2, 18), dtype=torch.float32)
        timestep = torch.tensor([0, 999], dtype=torch.long)
        output = model(x, timestep, condition)
        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 285124)

    def test_ddim_sampling_is_deterministic_and_batch_ordered(self):
        from resources.g1_interaction_builder.diffusion import (
            FunnelDenoiser,
            make_schedule,
            sample_ddim,
        )

        torch.manual_seed(7)
        model = FunnelDenoiser()
        schedule = make_schedule()
        condition = torch.zeros((2, 18), dtype=torch.float32)
        first = sample_ddim(model, condition, schedule, seed=123)
        second = sample_ddim(model, condition, schedule, seed=123)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, 32, 16, 4))
        self.assertEqual(first.dtype, torch.float32)

    def test_tiny_training_publishes_self_contained_checkpoint(self):
        from resources.g1_interaction_builder.diffusion import train_funnel

        conditions = torch.zeros((4, 18), dtype=torch.float32)
        funnels = torch.zeros((4, 16, 4), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            train_funnel(conditions, funnels, path, steps=1, batch_size=4, device="cpu")
            checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["schema_version"], 1)
        self.assertIn("model", checkpoint)
        self.assertEqual(tuple(checkpoint["condition_mean"].shape), (18,))


if __name__ == "__main__":
    unittest.main()
