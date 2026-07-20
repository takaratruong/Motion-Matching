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
    def test_model_condition_dimension_is_entry_conditioned(self):
        from resources.g1_interaction_builder import diffusion

        self.assertEqual(diffusion.MODEL_CONDITION_DIM, 24)

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
        condition = torch.zeros((2, 24), dtype=torch.float32)
        timestep = torch.tensor([0, 999], dtype=torch.long)
        output = model(x, timestep, condition)
        self.assertEqual(tuple(output.shape), (2, 4, 16))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 285892)

    def test_ddim_sampling_is_deterministic_and_batch_ordered(self):
        from resources.g1_interaction_builder.diffusion import (
            FunnelDenoiser,
            make_schedule,
            sample_ddim,
        )

        torch.manual_seed(7)
        model = FunnelDenoiser()
        schedule = make_schedule()
        condition = torch.zeros((2, 24), dtype=torch.float32)
        first = sample_ddim(model, condition, schedule, seed=123)
        second = sample_ddim(model, condition, schedule, seed=123)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (2, 32, 16, 4))
        self.assertEqual(first.dtype, torch.float32)

    def test_full_reverse_process_produces_certifiable_sample_shape(self):
        from resources.g1_interaction_builder.diffusion import (
            FunnelDenoiser,
            make_schedule,
            sample_ddim,
        )

        model = FunnelDenoiser()
        samples = sample_ddim(
            model, torch.zeros((1, 24)), make_schedule(), seed=3, step_count=1000
        )
        self.assertEqual(tuple(samples.shape), (1, 32, 16, 4))

    def test_x0_sampling_is_one_step_and_deterministic(self):
        from resources.g1_interaction_builder.diffusion import sample_x0

        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x, timestep, condition):
                self.calls += 1
                return torch.zeros_like(x)

        model = CountingModel()
        condition = torch.zeros((1, 24), dtype=torch.float32)
        first = sample_x0(model, condition, seed=11)
        second = sample_x0(model, condition, seed=11)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 32, 16, 4))
        self.assertEqual(model.calls, 2)

    def test_tiny_training_publishes_self_contained_checkpoint(self):
        from resources.g1_interaction_builder.diffusion import (
            sample_checkpoint,
            train_funnel,
        )

        conditions = torch.zeros((4, 24), dtype=torch.float32)
        funnels = torch.zeros((4, 16, 4), dtype=torch.float32)
        funnels[..., 3] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            train_funnel(conditions, funnels, path, steps=1, batch_size=4, device="cpu")
            checkpoint = torch.load(path, weights_only=False)
            samples = sample_checkpoint(path, torch.zeros((1, 24)), seed=11)
        self.assertEqual(checkpoint["schema_version"], 2)
        self.assertEqual(checkpoint["prediction_type"], "epsilon")
        self.assertIn("model", checkpoint)
        self.assertEqual(tuple(checkpoint["condition_mean"].shape), (24,))
        self.assertEqual(
            tuple(checkpoint["knot_frame_offsets"]),
            (0, 5, 10, 15, 20, 25, 30, 35, 39, 44, 49, 54, 59, 64, 69, 74),
        )
        self.assertEqual(tuple(checkpoint["funnel_mean"].shape), (1, 16, 4))
        self.assertEqual(tuple(checkpoint["funnel_scale"].shape), (1, 16, 4))
        expected_entry = torch.tensor([0.0, 0.0, 0.0, 1.0])
        self.assertTrue(torch.equal(samples[:, :, 15], expected_entry.expand(1, 32, 4)))


if __name__ == "__main__":
    unittest.main()
