from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from mm_sonic.evaluate_terrain_pfnn import (
    claim_sealed_test_receipt,
    validate_checkpoint_kinematics,
)
from mm_sonic.train_terrain_pfnn import (
    materialize_subset,
    overfit_gate_accepted,
    stratified_subset,
)
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.training import (
    CHECKPOINT_SCHEMA,
    autoregressive_unroll,
    finite_runtime_seed,
    load_checkpoint,
    pfnn_losses,
    restore_training_state,
    save_checkpoint,
    selection_metadata,
    training_phase_advance_q99,
)


def _normalization() -> dict[str, np.ndarray]:
    return {
        "x_mean": np.zeros(INPUT_LAYOUT.size, np.float32),
        "x_std": np.ones(INPUT_LAYOUT.size, np.float32),
        "y_mean": np.zeros(OUTPUT_LAYOUT.size, np.float32),
        "y_std": np.ones(OUTPUT_LAYOUT.size, np.float32),
    }


class TerrainPFNNTrainingTests(unittest.TestCase):
    def test_overfit_gate_uses_the_final_post_rollout_score(self) -> None:
        self.assertTrue(overfit_gate_accepted(10.0, 4.9, 0.5))
        self.assertFalse(overfit_gate_accepted(10.0, 5.1, 0.5))

    def test_materialized_overfit_subset_reads_source_once_in_sorted_order(self) -> None:
        class Rows:
            split = "train"
            x_mean = np.zeros(INPUT_LAYOUT.size, np.float32)
            x_std = np.ones(INPUT_LAYOUT.size, np.float32)
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                self.calls: list[int] = []

            def __getitem__(self, index: int) -> dict[str, object]:
                self.calls.append(index)
                return {
                    "x": np.full(INPUT_LAYOUT.size, index, np.float32),
                    "y": np.full(OUTPUT_LAYOUT.size, index, np.float32),
                    "phase": np.float32(index),
                    "clip_id": f"clip-{index}",
                    "center_frame": index,
                    "terrain_class": "flat",
                }

        source = Rows()
        cached = materialize_subset(source, [2, 0])
        self.assertEqual(source.calls, [0, 2])
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0]["clip_id"], "clip-2")
        self.assertEqual(cached[1]["clip_id"], "clip-0")
        self.assertEqual(source.calls, [0, 2])

    def test_phase_q99_clamps_only_normalized_round_trip_zero(self) -> None:
        class TrainingRows:
            split = "train"
            y_mean = np.zeros(OUTPUT_LAYOUT.size, np.float32)
            y_std = np.ones(OUTPUT_LAYOUT.size, np.float32)

            def __init__(self) -> None:
                self.y_mean[OUTPUT_LAYOUT["phase_advance"]] = np.float32(0.11376687)
                self.y_std[OUTPUT_LAYOUT["phase_advance"]] = np.float32(0.10085002)
                self.row = np.zeros(OUTPUT_LAYOUT.size, np.float32)
                # This is the exact normalized float32 emitted for a raw zero
                # in the remediated canary normalization contract.
                self.row[OUTPUT_LAYOUT["phase_advance"]] = np.float32(-1.1280799)

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> dict[str, object]:
                return {"y": self.row}

        self.assertEqual(training_phase_advance_q99(TrainingRows()), 0.0)

    def test_stratified_subset_is_balanced_and_deterministic(self) -> None:
        classes = ["flat"] * 9 + ["ascent"] * 5 + ["descent"] * 4 + ["transition"] * 2
        first = stratified_subset(classes, 8, seed=7)
        second = stratified_subset(classes, 8, seed=7)
        self.assertEqual(first, second)
        counts = {name: sum(classes[index] == name for index in first) for name in set(classes)}
        self.assertEqual(set(counts.values()), {2})

    def test_sealed_test_receipt_is_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            receipt = claim_sealed_test_receipt(
                run, checkpoint_sha256="a" * 64, dataset_digest="b" * 64
            )
            self.assertTrue(receipt.is_file())
            with self.assertRaisesRegex(RuntimeError, "already"):
                claim_sealed_test_receipt(
                    run, checkpoint_sha256="a" * 64, dataset_digest="b" * 64
                )

    def test_losses_are_finite_and_report_every_group(self) -> None:
        torch.manual_seed(3)
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        prediction = model(
            torch.zeros(4, INPUT_LAYOUT.size), torch.linspace(0.0, 1.0, 4)
        )
        target = torch.zeros_like(prediction)
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            (0.0, 1.0, 0.0, 1.0)
        )
        losses = pfnn_losses(
            prediction, target, model=model, normalization=_normalization()
        )
        self.assertEqual(
            set(losses),
            {
                "trajectory_mse", "body_mse", "root_pose_mse", "joint_mse",
                "root_motion_mse", "phase_mse", "contact_bce",
                "trajectory_direction", "phase_nonnegative", "fk_consistency",
                "regularization", "total",
            },
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_contact_loss_uses_raw_logits_and_structural_terms_are_physical(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
        prediction = torch.zeros(2, OUTPUT_LAYOUT.size)
        target = torch.zeros_like(prediction)
        prediction[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            ((-2.0, 2.0, -1.0, 1.0), (-3.0, 3.0, -4.0, 4.0))
        )
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            ((0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0))
        )
        normal = _normalization()
        normal["y_mean"][OUTPUT_LAYOUT["trajectory_direction"]] = 10.0
        normal["y_std"][OUTPUT_LAYOUT["trajectory_direction"]] = 2.0
        losses = pfnn_losses(
            prediction, target, model=model, normalization=normal
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction[:, OUTPUT_LAYOUT["contact_logit"]],
            target[:, OUTPUT_LAYOUT["contact_logit"]],
        )
        torch.testing.assert_close(losses["contact_bce"], expected)
        # A normalized zero direction denormalizes to (10,10), so this proves
        # the unit-direction constraint is evaluated in physical space.
        self.assertGreater(float(losses["trajectory_direction"]), 100.0)

    def test_checkpoint_round_trip_and_contract_mismatch(self) -> None:
        torch.manual_seed(5)
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        seed = finite_runtime_seed()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path, model, optimizer, _normalization(),
                dataset_digest="abc", kinematic_signature_sha256="def",
                runtime_seed=seed, step=17,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["input_layout"], [list(field) for field in INPUT_LAYOUT.fields])
            self.assertEqual(payload["output_layout"], [list(field) for field in OUTPUT_LAYOUT.fields])
            loaded = load_checkpoint(
                path, expected_dataset_digest="abc",
                expected_kinematic_signature_sha256="def",
            )
            self.assertEqual(loaded.schema, CHECKPOINT_SCHEMA)
            self.assertEqual(loaded.step, 17)
            self.assertEqual(loaded.runtime_seed.keys(), seed.keys())
            class MatchingKinematics:
                kinematic_signature_sha256 = "def"
                joint_limits = torch.tensor(
                    [[-np.pi, np.pi]] * 29, dtype=torch.float64
                )

            validate_checkpoint_kinematics(loaded, MatchingKinematics())
            MatchingKinematics.joint_limits[0, 0] = -1.0
            with self.assertRaisesRegex(ValueError, "joint limits"):
                validate_checkpoint_kinematics(loaded, MatchingKinematics())
            resume_model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
            resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=1.0e-3)
            self.assertEqual(
                restore_training_state(
                    loaded, resume_model, resume_optimizer,
                    normalization=_normalization(),
                    train_identities=(), validation_identities=(),
                ),
                (17, 0),
            )
            self.assertTrue(
                all(
                    torch.equal(left, right)
                    for left, right in zip(model.parameters(), resume_model.parameters())
                )
            )
            with self.assertRaisesRegex(ValueError, "dataset digest"):
                load_checkpoint(
                    path, expected_dataset_digest="different",
                    expected_kinematic_signature_sha256="def",
                )
            with self.assertRaisesRegex(ValueError, "kinematic signature"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="different",
                )
            original_w0 = payload["model_state"]["W0"]
            payload["model_state"]["W0"] = torch.zeros_like(
                original_w0, dtype=torch.int64
            )
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "model state"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )
            payload["model_state"]["W0"] = original_w0
            original_optimizer_state = payload["optimizer_state"]
            payload["optimizer_state"] = {"unexpected": []}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "optimizer state"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )
            payload["optimizer_state"] = original_optimizer_state
            payload["selection"] = {"provisional": False}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "selection"):
                load_checkpoint(
                    path, expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )

    def test_three_step_rollout_feeds_predictions_back_without_detaching(self) -> None:
        torch.manual_seed(11)
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        inputs = torch.randn(3, 2, INPUT_LAYOUT.size)
        targets = torch.zeros(3, 2, OUTPUT_LAYOUT.size)
        targets[..., OUTPUT_LAYOUT["contact_logit"]] = 1.0
        normal = _normalization()
        result = autoregressive_unroll(
            model, inputs, torch.tensor((0.2, 0.7)), targets,
            normalization=normal,
        )
        first = result.predictions[0]
        expected_trajectory = torch.cat(
            (
                first[:, OUTPUT_LAYOUT["trajectory_position"]],
                first[:, OUTPUT_LAYOUT["trajectory_direction"]],
            ),
            dim=1,
        )
        torch.testing.assert_close(result.inputs[1][:, :48], expected_trajectory)
        torch.testing.assert_close(
            result.inputs[1][:, INPUT_LAYOUT["terrain_height"]],
            inputs[1, :, INPUT_LAYOUT["terrain_height"]],
        )
        torch.testing.assert_close(
            result.inputs[1][:, INPUT_LAYOUT["semantic_intent"]],
            inputs[1, :, INPUT_LAYOUT["semantic_intent"]],
        )
        torch.testing.assert_close(
            result.inputs[1][:, INPUT_LAYOUT["previous_body_position"]],
            0.1 * first[:, OUTPUT_LAYOUT["body_position"]],
        )
        gradient = torch.autograd.grad(
            result.predictions[2].square().mean(), first, retain_graph=True
        )[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().max()), 0.0)

    def test_normal_selection_requires_closed_loop_but_overfit_is_provisional(self) -> None:
        provisional = selection_metadata(
            one_step_score=1.25, pipeline_overfit=True
        )
        self.assertEqual(provisional["selection_mode"], "pipeline_overfit_one_step")
        self.assertIsNone(provisional["closed_loop_score"])
        self.assertTrue(provisional["provisional"])
        with self.assertRaisesRegex(RuntimeError, "closed-loop"):
            selection_metadata(one_step_score=1.25, pipeline_overfit=False)

    def test_small_model_overfits_and_reloads_bitwise_on_cpu(self) -> None:
        torch.manual_seed(17)
        model = PhaseFunctionedNetwork(hidden_size=32, dropout_probability=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
        x = torch.zeros(64, INPUT_LAYOUT.size)
        phase = torch.zeros(64)
        target = torch.zeros(64, OUTPUT_LAYOUT.size)
        direction = target[:, OUTPUT_LAYOUT["trajectory_direction"]].reshape(64, 12, 2)
        direction[..., 0] = 1.0
        target[:, OUTPUT_LAYOUT["phase_advance"]] = 0.1
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor((0.0, 1.0, 0.0, 1.0))
        normal = _normalization()
        model.train()
        with torch.no_grad():
            initial = float(
                pfnn_losses(
                    model(x, phase), target, model=model, normalization=normal
                )["total"]
            )
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            losses = pfnn_losses(
                model(x, phase), target, model=model, normalization=normal
            )
            losses["total"].backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(x, phase)
            final = float(
                pfnn_losses(
                    prediction, target, model=model, normalization=normal
                )["total"]
            )
        self.assertLess(final, 0.1 * initial)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                path, model, optimizer, normal, dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(), step=300,
            )
            loaded = load_checkpoint(
                path, expected_dataset_digest="abc",
                expected_kinematic_signature_sha256="def",
            )
            restored = loaded.build_model()
            restored.eval()
            with torch.no_grad():
                reloaded_prediction = restored(x, phase)
            self.assertTrue(torch.equal(prediction, reloaded_prediction))


if __name__ == "__main__":
    unittest.main()
