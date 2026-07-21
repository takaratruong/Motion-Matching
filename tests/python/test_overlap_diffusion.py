import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment gate
    torch = None


@unittest.skipIf(torch is None, "torch is required for overlap diffusion")
class OverlapDiffusionTests(unittest.TestCase):
    def _condition(self):
        from resources.g1_interaction_builder.overlap_diffusion import CoupledCondition

        return CoupledCondition(
            static=torch.zeros(25, dtype=torch.float32),
            temporal=torch.zeros((80, 9), dtype=torch.float32),
        )

    def test_overlap_endpoints_and_shared_latent_contract(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            sample_coupled,
            overlap_weight,
        )

        class ZeroExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x)

        self.assertEqual(overlap_weight(0), 0.0)
        self.assertEqual(overlap_weight(19), 1.0)
        output, trace = sample_coupled(
            ZeroExpert(), ZeroExpert(), self._condition(), seed=9, steps=2, return_trace=True
        )
        self.assertEqual(tuple(output.shape), (8, 80, 192))
        self.assertEqual(len(trace), 2)
        for step in trace:
            self.assertTrue(torch.equal(step.walk_input[:, 30:50], step.pickup_input[:, 0:20]))
        self.assertTrue(torch.equal(trace[1].walk_input, trace[0].global_latent[:, :50]))
        self.assertTrue(torch.equal(trace[1].pickup_input, trace[0].global_latent[:, 30:80]))

    def test_overlap_epsilon_blend_selects_endpoints_and_mixes_interior(self):
        from resources.g1_interaction_builder.overlap_diffusion import sample_coupled

        class ConstantExpert(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x, timestep, static, temporal):
                return torch.full_like(x, self.value)

        _, trace = sample_coupled(
            ConstantExpert(2.0), ConstantExpert(10.0), self._condition(),
            seed=7, candidates=1, steps=1, return_trace=True,
        )
        epsilon = trace[0].global_epsilon
        self.assertTrue(torch.equal(epsilon[:, 30], torch.full((1, 192), 2.0)))
        self.assertTrue(torch.equal(epsilon[:, 49], torch.full((1, 192), 10.0)))
        expected = 2.0 * (1.0 - 10.0 / 19.0) + 10.0 * (10.0 / 19.0)
        self.assertTrue(torch.equal(epsilon[:, 40], torch.full((1, 192), expected)))

    def test_fixed_frame_zero_is_exact_in_every_global_step_and_output(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            FixedFrames,
            sample_coupled,
        )

        class ZeroExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x)

        values = torch.zeros((80, 192), dtype=torch.float32)
        values[0] = torch.linspace(-1.0, 1.0, 192)
        mask = torch.zeros(80, dtype=torch.bool)
        mask[0] = True
        fixed = FixedFrames(values=values, mask=mask)
        output, trace = sample_coupled(
            ZeroExpert(), ZeroExpert(), self._condition(), seed=3, steps=3,
            fixed_frames=fixed, return_trace=True,
        )
        self.assertTrue(torch.equal(output[:, 0], values[0].expand(8, -1)))
        for step in trace:
            self.assertTrue(torch.equal(step.global_latent[:, 0], values[0].expand(8, -1)))

    def test_cpu_seed_generation_is_repeatable_despite_global_rng_changes(self):
        from resources.g1_interaction_builder.overlap_diffusion import sample_coupled

        class ZeroExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x)

        first = sample_coupled(ZeroExpert(), ZeroExpert(), self._condition(), seed=111, steps=2)
        torch.manual_seed(9999)
        _ = torch.randn(4096)
        second = sample_coupled(ZeroExpert(), ZeroExpert(), self._condition(), seed=111, steps=2)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.device.type, "cpu")

    def test_condition_fixed_frames_and_guidance_are_shape_and_finite_checked(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            CoupledCondition,
            FixedFrames,
            TaskGuidance,
            sample_coupled,
        )

        class ZeroExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x)

        with self.assertRaisesRegex(ValueError, "static"):
            CoupledCondition(torch.zeros(24), torch.zeros((80, 9)))
        with self.assertRaisesRegex(ValueError, "temporal"):
            CoupledCondition(torch.zeros(25), torch.zeros((79, 9)))
        bad_values = torch.zeros((80, 192))
        bad_values[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            FixedFrames(bad_values, torch.zeros(80, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "finite"):
            sample_coupled(
                ZeroExpert(), ZeroExpert(), self._condition(), seed=1, steps=1,
                task_guidance=TaskGuidance(lambda clean, condition, timestep: clean.new_tensor(float("nan"))),
            )

    def test_task_guidance_is_differentiable_and_changes_clean_estimate(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            TaskGuidance,
            sample_coupled,
        )

        class ZeroExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x)

        guidance = TaskGuidance(
            lambda clean, condition, timestep: clean[..., 0].sum(), strength=0.25
        )
        _, trace = sample_coupled(
            ZeroExpert(), ZeroExpert(), self._condition(), seed=5, candidates=1,
            steps=1, task_guidance=guidance, return_trace=True,
        )
        self.assertTrue(torch.all(trace[0].guided_clean[..., 0] < trace[0].unguided_clean[..., 0]))

    def test_denoiser_schema_is_identical_but_weights_are_independent(self):
        from resources.g1_interaction_builder.overlap_diffusion import MotionWindowDenoiser

        walk = MotionWindowDenoiser()
        pickup = MotionWindowDenoiser()
        self.assertEqual(walk.schema, pickup.schema)
        self.assertEqual(len(walk.blocks), 8)
        self.assertEqual(walk.blocks[0].attention.num_heads, 8)
        self.assertFalse(any(left is right for left in walk.parameters() for right in pickup.parameters()))
        output = walk(
            torch.zeros((1, 50, 192)), torch.tensor([9]), torch.zeros((1, 25)),
            torch.zeros((1, 50, 9)),
        )
        self.assertEqual(tuple(output.shape), (1, 50, 192))

    def test_named_losses_are_zero_for_exact_prediction(self):
        from resources.g1_interaction_builder.overlap_diffusion import named_training_losses

        target = torch.zeros((1, 50, 192), dtype=torch.float32)
        target[..., 189:192] = 1.0
        losses = named_training_losses(
            target, target, torch.zeros_like(target), torch.zeros_like(target),
            fk=self._identity_fk, grasp_position=torch.zeros((1, 3)),
            grasp_orientation=torch.zeros((1, 6)), object_position=torch.zeros((1, 3)),
            overlap_walk=target[:, 30:50], overlap_pickup=target[:, 30:50],
        )
        self.assertEqual(
            set(losses),
            {
                "epsilon", "pose_6d", "fk_hand", "fk_feet", "velocity", "acceleration",
                "foot_contact", "foot_sliding", "grasp_position", "grasp_orientation",
                "attachment", "overlap_agreement",
            },
        )
        for name, loss in losses.items():
            self.assertTrue(torch.equal(loss, torch.zeros_like(loss)), name)

    def test_each_named_loss_is_positive_for_its_isolated_perturbation(self):
        from resources.g1_interaction_builder.overlap_diffusion import named_training_losses

        target = torch.zeros((1, 50, 192), dtype=torch.float32)
        target[..., 189:192] = 1.0
        kwargs = dict(
            fk=self._identity_fk, grasp_position=torch.zeros((1, 3)),
            grasp_orientation=torch.zeros((1, 6)), object_position=torch.zeros((1, 3)),
            overlap_walk=target[:, 30:50], overlap_pickup=target[:, 30:50],
        )
        cases = {
            "epsilon": (target.clone(), torch.ones_like(target)),
            "pose_6d": (self._perturb(target, 3), torch.zeros_like(target)),
            "fk_hand": (self._perturb(target, 0), torch.zeros_like(target)),
            "fk_feet": (self._perturb(target, 3), torch.zeros_like(target)),
            "velocity": (self._ramp(target, 20), torch.zeros_like(target)),
            "acceleration": (self._curve(target, 21), torch.zeros_like(target)),
            "foot_contact": (self._perturb(target, 189, value=0.0), torch.zeros_like(target)),
            "foot_sliding": (self._ramp(target, 6), torch.zeros_like(target)),
            "grasp_position": (self._perturb(target, 0), torch.zeros_like(target)),
            "grasp_orientation": (self._perturb(target, 3), torch.zeros_like(target)),
            "attachment": (self._perturb(target, 0), torch.zeros_like(target)),
            "overlap_agreement": (target.clone(), torch.zeros_like(target)),
        }
        for name, (prediction, epsilon) in cases.items():
            with self.subTest(name=name):
                per_case = dict(kwargs)
                if name == "overlap_agreement":
                    per_case["overlap_pickup"] = torch.ones_like(target[:, 30:50])
                losses = named_training_losses(
                    prediction, target, epsilon, torch.zeros_like(target), **per_case
                )
                self.assertGreater(float(losses[name]), 0.0)

    @staticmethod
    def _identity_fk(frames):
        return {
            "hand": frames[..., 0:3],
            "left_foot": frames[..., 3:6],
            "right_foot": frames[..., 6:9],
        }

    @staticmethod
    def _perturb(target, channel, value=1.0):
        result = target.clone()
        result[:, 25, channel] = value
        return result

    @staticmethod
    def _ramp(target, channel):
        result = target.clone()
        result[..., channel] = torch.arange(50, dtype=target.dtype).view(1, -1)
        return result

    @staticmethod
    def _curve(target, channel):
        result = target.clone()
        points = torch.arange(50, dtype=target.dtype)
        result[..., channel] = points.square().view(1, -1)
        return result


if __name__ == "__main__":
    unittest.main()
