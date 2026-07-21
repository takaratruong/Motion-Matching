import unittest

try:
    import torch
except ImportError:  # pragma: no cover - environment gate
    torch = None


@unittest.skipIf(torch is None, "torch is required for overlap diffusion")
class OverlapDiffusionTests(unittest.TestCase):
    contact_frame = 25

    def _condition(self):
        from resources.g1_interaction_builder.overlap_diffusion import CoupledCondition

        return CoupledCondition(
            static=torch.zeros(25, dtype=torch.float32),
            temporal=torch.zeros((80, 9), dtype=torch.float32),
        )

    @staticmethod
    def _zero_expert():
        class ZeroExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x)

        return ZeroExpert()

    def test_overlap_endpoints_and_shared_latent_contract(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            overlap_weight,
            sample_coupled,
        )

        output, trace = sample_coupled(
            self._zero_expert(), self._zero_expert(), self._condition(),
            seed=9, steps=2, return_trace=True,
        )
        self.assertEqual(tuple(output.shape), (8, 80, 195))
        self.assertEqual(overlap_weight(0), 0.0)
        self.assertEqual(overlap_weight(19), 1.0)
        for step in trace:
            self.assertTrue(torch.equal(step.walk_input[:, 30:50], step.pickup_input[:, :20]))
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
        self.assertTrue(torch.equal(epsilon[:, 30], torch.full((1, 195), 2.0)))
        self.assertTrue(torch.equal(epsilon[:, 49], torch.full((1, 195), 10.0)))
        expected = 2.0 * (1.0 - 10.0 / 19.0) + 10.0 * (10.0 / 19.0)
        self.assertTrue(torch.equal(epsilon[:, 40], torch.full((1, 195), expected)))

    def test_ddim_timestep_grid_and_update_follow_numeric_schedule(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            _alpha_bars,
            _ddim_timesteps,
            sample_coupled,
        )

        self.assertEqual(_ddim_timesteps(1), (999,))
        self.assertEqual(_ddim_timesteps(4), (999, 666, 333, 0))

        class ConstantExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.full_like(x, 0.25)

        _, trace = sample_coupled(
            ConstantExpert(), ConstantExpert(), self._condition(), seed=4,
            candidates=1, steps=2, return_trace=True,
        )
        alpha_bars = _alpha_bars(torch.device("cpu"))
        alpha = alpha_bars[999]
        previous = alpha_bars[0]
        expected_clean = (
            trace[0].walk_input.new_empty((1, 80, 195))
        )
        latent = trace[0].walk_input.new_empty((1, 80, 195))
        latent[:, :50] = trace[0].walk_input
        latent[:, 30:80] = trace[0].pickup_input
        expected_clean = (latent - torch.sqrt(1.0 - alpha) * 0.25) / torch.sqrt(alpha)
        expected_latent = (
            torch.sqrt(previous) * expected_clean
            + torch.sqrt(1.0 - previous) * 0.25
        )
        self.assertTrue(torch.allclose(trace[0].unguided_clean, expected_clean, atol=1e-6))
        self.assertTrue(torch.allclose(trace[0].global_latent, expected_latent, atol=1e-6))

    def test_sampler_rejects_expert_outputs_with_wrong_batch_or_dtype(self):
        from resources.g1_interaction_builder.overlap_diffusion import sample_coupled

        class BatchOneExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros((1, 50, 195), device=x.device, dtype=x.dtype)

        class Float64Expert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros_like(x, dtype=torch.float64)

        class LegacySchemaExpert(torch.nn.Module):
            def forward(self, x, timestep, static, temporal):
                return torch.zeros((x.shape[0], 50, 192), device=x.device, dtype=x.dtype)

        with self.assertRaisesRegex(ValueError, "walk epsilon.*shape"):
            sample_coupled(BatchOneExpert(), self._zero_expert(), self._condition(), seed=1)
        with self.assertRaisesRegex(ValueError, "walk epsilon.*dtype"):
            sample_coupled(Float64Expert(), self._zero_expert(), self._condition(), seed=1)
        with self.assertRaisesRegex(ValueError, "walk epsilon.*195"):
            sample_coupled(LegacySchemaExpert(), self._zero_expert(), self._condition(), seed=1)

    def test_sampler_forces_eval_without_mutating_batchnorm_or_dropout_state(self):
        from resources.g1_interaction_builder.overlap_diffusion import sample_coupled

        class StatefulExpert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_norm = torch.nn.BatchNorm1d(195)
                self.dropout = torch.nn.Dropout(p=0.75)

            def forward(self, x, timestep, static, temporal):
                values = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
                return self.dropout(values)

        walk = StatefulExpert().train()
        pickup = StatefulExpert().eval()
        walk_mean = walk.batch_norm.running_mean.clone()
        walk_var = walk.batch_norm.running_var.clone()
        output = sample_coupled(walk, pickup, self._condition(), seed=3, candidates=1, steps=1)
        self.assertTrue(walk.training)
        self.assertFalse(pickup.training)
        self.assertTrue(torch.equal(walk.batch_norm.running_mean, walk_mean))
        self.assertTrue(torch.equal(walk.batch_norm.running_var, walk_var))
        self.assertTrue(torch.isfinite(output).all())

        class FailingExpert(StatefulExpert):
            def forward(self, x, timestep, static, temporal):
                super().forward(x, timestep, static, temporal)
                raise RuntimeError("expert failure")

        failing = FailingExpert().train()
        with self.assertRaisesRegex(RuntimeError, "expert failure"):
            sample_coupled(failing, pickup, self._condition(), seed=3, candidates=1, steps=1)
        self.assertTrue(failing.training)

    def test_fixed_frame_zero_is_exact_in_every_global_step_and_output(self):
        from resources.g1_interaction_builder.overlap_diffusion import FixedFrames, sample_coupled

        values = torch.zeros((80, 195), dtype=torch.float32)
        values[0] = torch.linspace(-1.0, 1.0, 195)
        mask = torch.zeros(80, dtype=torch.bool)
        mask[0] = True
        output, trace = sample_coupled(
            self._zero_expert(), self._zero_expert(), self._condition(), seed=3,
            steps=3, fixed_frames=FixedFrames(values=values, mask=mask), return_trace=True,
        )
        self.assertTrue(torch.equal(output[:, 0], values[0].expand(8, -1)))
        for step in trace:
            self.assertTrue(torch.equal(step.global_latent[:, 0], values[0].expand(8, -1)))

    def test_cpu_seed_generation_is_repeatable_despite_global_rng_changes(self):
        from resources.g1_interaction_builder.overlap_diffusion import sample_coupled

        first = sample_coupled(self._zero_expert(), self._zero_expert(), self._condition(), seed=111, steps=2)
        torch.manual_seed(9999)
        _ = torch.randn(4096)
        second = sample_coupled(self._zero_expert(), self._zero_expert(), self._condition(), seed=111, steps=2)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.device.type, "cpu")

    def test_condition_fixed_frames_and_guidance_are_shape_and_finite_checked(self):
        from resources.g1_interaction_builder.overlap_diffusion import (
            CoupledCondition,
            FixedFrames,
            TaskGuidance,
            sample_coupled,
        )

        with self.assertRaisesRegex(ValueError, "static"):
            CoupledCondition(torch.zeros(24), torch.zeros((80, 9)))
        with self.assertRaisesRegex(ValueError, "temporal"):
            CoupledCondition(torch.zeros(25), torch.zeros((79, 9)))
        bad_values = torch.zeros((80, 195))
        bad_values[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            FixedFrames(bad_values, torch.zeros(80, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "195"):
            FixedFrames(torch.zeros((80, 192)), torch.zeros(80, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "finite"):
            sample_coupled(
                self._zero_expert(), self._zero_expert(), self._condition(), seed=1, steps=1,
                task_guidance=TaskGuidance(lambda clean, condition, timestep: clean.new_tensor(float("nan"))),
            )

    def test_task_guidance_is_differentiable_and_changes_clean_estimate(self):
        from resources.g1_interaction_builder.overlap_diffusion import TaskGuidance, sample_coupled

        guidance = TaskGuidance(lambda clean, condition, timestep: clean[..., 0].sum(), strength=0.25)
        _, trace = sample_coupled(
            self._zero_expert(), self._zero_expert(), self._condition(), seed=5,
            candidates=1, steps=1, task_guidance=guidance, return_trace=True,
        )
        self.assertTrue(torch.all(trace[0].guided_clean[..., 0] < trace[0].unguided_clean[..., 0]))

    def test_denoiser_schema_is_exact_and_weights_are_independent(self):
        from resources.g1_interaction_builder.overlap_diffusion import MotionWindowDenoiser

        walk = MotionWindowDenoiser()
        pickup = MotionWindowDenoiser()
        self.assertEqual(walk.schema, pickup.schema)
        self.assertEqual((walk.schema.width, walk.schema.blocks, walk.schema.heads), (256, 8, 8))
        self.assertEqual(len(walk.blocks), 8)
        self.assertEqual(walk.blocks[0].attention.num_heads, 8)
        self.assertFalse(any(left is right for left in walk.parameters() for right in pickup.parameters()))
        output = walk(
            torch.zeros((1, 50, 195)), torch.tensor([9]), torch.zeros((1, 25)),
            torch.zeros((1, 50, 9)),
        )
        self.assertEqual(tuple(output.shape), (1, 50, 195))

    def test_named_losses_are_zero_for_exact_target(self):
        losses = self._losses(self._target())
        self.assertEqual(
            set(losses),
            {
                "epsilon", "pose_6d", "fk_hand", "fk_feet", "velocity", "acceleration",
                "foot_contact", "foot_sliding", "grasp_position", "grasp_orientation",
                "pre_contact_separation", "attachment", "overlap_agreement",
            },
        )
        for name, loss in losses.items():
            self.assertTrue(torch.equal(loss, torch.zeros_like(loss)), name)

    def test_fk_contract_requires_active_hand_orientation(self):
        def incomplete_fk(frames):
            result = self._identity_fk(frames)
            del result["hand_orientation"]
            return result

        with self.assertRaisesRegex(ValueError, "hand_orientation"):
            self._losses(self._target(), fk=incomplete_fk)

    def test_grasp_orientation_uses_fk_hand_orientation_at_first_contact(self):
        target = self._target()
        root_only = target.clone()
        root_only[:, self.contact_frame, 6:12] += 1.0
        root_losses = self._losses(root_only)
        self.assertEqual(float(root_losses["grasp_orientation"]), 0.0)

        hand_oriented = target.clone()
        hand_oriented[:, self.contact_frame, 12:18] += 1.0
        losses = self._losses(hand_oriented)
        self.assertGreater(float(losses["grasp_orientation"]), 0.0)
        self.assertEqual(float(losses["grasp_position"]), 0.0)
        self.assertEqual(float(losses["attachment"]), 0.0)

    def test_grasp_position_is_first_contact_only_and_attachment_is_sustained(self):
        target = self._target()
        first_contact = target.clone()
        first_contact[:, self.contact_frame, 0] += 0.5
        first_losses = self._losses(first_contact)
        self.assertGreater(float(first_losses["grasp_position"]), 0.0)
        self.assertEqual(float(first_losses["attachment"]), 0.0)

        sustained = target.clone()
        sustained[:, self.contact_frame + 1, 0] += 0.5
        sustained_losses = self._losses(sustained)
        self.assertEqual(float(sustained_losses["grasp_position"]), 0.0)
        self.assertGreater(float(sustained_losses["attachment"]), 0.0)

        sustained_orientation = target.clone()
        sustained_orientation[:, self.contact_frame + 1, 12:18] += 0.5
        orientation_losses = self._losses(sustained_orientation)
        self.assertEqual(float(orientation_losses["grasp_orientation"]), 0.0)
        self.assertGreater(float(orientation_losses["attachment"]), 0.0)

    def test_pre_contact_clearance_penalizes_intrusion_but_not_post_contact_attachment(self):
        target = self._target()
        pre_contact = target.clone()
        pre_contact[:, self.contact_frame - 1, 0:3] = 0.0
        losses = self._losses(pre_contact)
        self.assertGreater(float(losses["pre_contact_separation"]), 0.0)
        self.assertEqual(float(losses["grasp_position"]), 0.0)
        self.assertEqual(float(losses["attachment"]), 0.0)

        post_contact = target.clone()
        post_contact[:, self.contact_frame + 1, 0:3] = 0.0
        post_losses = self._losses(post_contact)
        self.assertEqual(float(post_losses["pre_contact_separation"]), 0.0)
        self.assertGreater(float(post_losses["attachment"]), 0.0)

    def test_attachment_targets_conditioned_grasp_pose_not_object_center(self):
        target = self._target()
        losses = self._losses(target)
        self.assertEqual(float(losses["attachment"]), 0.0)
        self.assertEqual(float(losses["pre_contact_separation"]), 0.0)

    def test_object_clearance_contract_is_validated(self):
        with self.assertRaisesRegex(ValueError, "object_clearance"):
            self._losses(self._target(), object_clearance=torch.tensor([-0.1]))
        with self.assertRaisesRegex(ValueError, "object_clearance"):
            self._losses(self._target(), object_clearance=torch.tensor([float("nan")]))

    def test_epsilon_and_overlap_terms_are_separable(self):
        target = self._target()
        epsilon_losses = self._losses(target, predicted_epsilon=torch.ones_like(target))
        self.assertGreater(float(epsilon_losses["epsilon"]), 0.0)
        for name, loss in epsilon_losses.items():
            if name != "epsilon":
                self.assertEqual(float(loss), 0.0, name)

        overlap_losses = self._losses(target, overlap_pickup=torch.ones((1, 20, 195)))
        self.assertGreater(float(overlap_losses["overlap_agreement"]), 0.0)
        for name, loss in overlap_losses.items():
            if name != "overlap_agreement":
                self.assertEqual(float(loss), 0.0, name)

    def test_pose_velocity_and_acceleration_terms_target_distinct_derivatives(self):
        target = self._target()
        pose = target.clone()
        pose[..., 30] += 1.0
        pose_losses = self._losses(pose)
        self.assertGreater(float(pose_losses["pose_6d"]), 0.0)
        self.assertEqual(float(pose_losses["velocity"]), 0.0)
        self.assertEqual(float(pose_losses["acceleration"]), 0.0)

        velocity = target.clone()
        velocity[..., 31] = torch.arange(50, dtype=target.dtype)
        velocity_losses = self._losses(velocity)
        self.assertGreater(float(velocity_losses["velocity"]), 0.0)
        self.assertEqual(float(velocity_losses["acceleration"]), 0.0)

        acceleration = target.clone()
        acceleration[..., 32] = torch.arange(50, dtype=target.dtype).square()
        acceleration_losses = self._losses(acceleration)
        self.assertGreater(float(acceleration_losses["acceleration"]), 0.0)
        # A second derivative necessarily changes the first derivative too.
        self.assertGreater(float(acceleration_losses["velocity"]), 0.0)

    def test_fk_feet_foot_contact_and_foot_sliding_terms_are_targeted(self):
        target = self._target()
        feet = target.clone()
        feet[:, 10, 18] += 1.0
        feet_losses = self._losses(feet)
        self.assertGreater(float(feet_losses["fk_feet"]), 0.0)
        self.assertEqual(float(feet_losses["grasp_position"]), 0.0)
        self.assertEqual(float(feet_losses["attachment"]), 0.0)

        contacts = target.clone()
        contacts[..., 192] = 1.0
        contact_losses = self._losses(contacts)
        self.assertGreater(float(contact_losses["foot_contact"]), 0.0)
        self.assertEqual(float(contact_losses["foot_sliding"]), 0.0)

        sliding_target = self._target()
        sliding_target[:, 1:, 192] = 1.0
        sliding = sliding_target.clone()
        sliding[:, 10, 18] += 1.0
        sliding_losses = self._losses(sliding, target=sliding_target)
        self.assertGreater(float(sliding_losses["foot_sliding"]), 0.0)
        # FK-position and finite-difference terms are necessarily coupled to sliding.
        self.assertGreater(float(sliding_losses["fk_feet"]), 0.0)
        self.assertGreater(float(sliding_losses["velocity"]), 0.0)

    @staticmethod
    def _identity_fk(frames):
        return {
            "hand": frames[..., 0:3],
            "hand_orientation": frames[..., 12:18],
            "left_foot": frames[..., 18:21],
            "right_foot": frames[..., 21:24],
        }

    def _target(self):
        target = torch.zeros((1, 50, 195), dtype=torch.float32)
        target[..., 0] = 2.0
        target[:, self.contact_frame:, 0] = 1.0
        target[..., 12] = 1.0
        target[..., 16] = 1.0
        target[:, self.contact_frame:, -1] = 1.0
        return target

    def _losses(
        self,
        prediction,
        *,
        target=None,
        predicted_epsilon=None,
        overlap_pickup=None,
        object_clearance=None,
        fk=None,
    ):
        from resources.g1_interaction_builder.overlap_diffusion import named_training_losses

        target = self._target() if target is None else target
        return named_training_losses(
            prediction,
            target,
            torch.zeros_like(target) if predicted_epsilon is None else predicted_epsilon,
            torch.zeros_like(target),
            fk=self._identity_fk if fk is None else fk,
            grasp_position=torch.tensor([[1.0, 0.0, 0.0]]),
            grasp_orientation=torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            object_position=torch.zeros((1, 3)),
            object_clearance=torch.tensor([0.5]) if object_clearance is None else object_clearance,
            overlap_walk=target[:, 30:50],
            overlap_pickup=target[:, 30:50] if overlap_pickup is None else overlap_pickup,
        )


if __name__ == "__main__":
    unittest.main()
