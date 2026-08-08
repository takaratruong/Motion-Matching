from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from mm_sonic.train_classic_g1_pfnn import (
    balanced_epoch_batches,
    classic_pfnn_loss,
    load_classic_checkpoint,
    save_classic_checkpoint,
)
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.training import finite_runtime_seed


class ClassicG1PFNNTests(unittest.TestCase):
    def test_loss_is_only_one_step_supervision(self) -> None:
        generator = torch.Generator().manual_seed(11)
        prediction = torch.randn(
            (3, OUTPUT_LAYOUT.size), generator=generator, dtype=torch.float64
        )
        target = torch.randn(
            (3, OUTPUT_LAYOUT.size), generator=generator, dtype=torch.float64
        )
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.tensor(
            ((0.0, 1.0, 0.0, 1.0),) * 3, dtype=torch.float64
        )
        expected = F.mse_loss(
            prediction[:, : OUTPUT_LAYOUT["contact_logit"].start],
            target[:, : OUTPUT_LAYOUT["contact_logit"].start],
        ) + F.binary_cross_entropy_with_logits(
            prediction[:, OUTPUT_LAYOUT["contact_logit"]],
            target[:, OUTPUT_LAYOUT["contact_logit"]],
        )

        actual = classic_pfnn_loss(prediction, target)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), ())

    def test_balanced_batches_are_deterministic_and_source_balanced(self) -> None:
        source_kind = np.asarray(
            ("grail",) * 5 + ("lafan",) * 9, dtype="<U6"
        )
        first = balanced_epoch_batches(source_kind, batch_size=4, seed=23, epoch=0)
        second = balanced_epoch_batches(source_kind, batch_size=4, seed=23, epoch=0)

        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(left), 4)
            selected = source_kind[left]
            self.assertEqual(int(np.count_nonzero(selected == "grail")), 2)
            self.assertEqual(int(np.count_nonzero(selected == "lafan")), 2)
        used = np.concatenate(first)
        self.assertTrue(set(range(len(source_kind))).issubset(set(used.tolist())))

    def test_safe_checkpoint_round_trip_preserves_exact_model_output(self) -> None:
        torch.manual_seed(7)
        model = PhaseFunctionedNetwork(hidden_size=32, dropout_probability=0.30)
        model.eval()
        x = torch.randn((4, INPUT_LAYOUT.size), dtype=torch.float32)
        phase = torch.tensor((0.0, 0.5, 2.0, 6.0), dtype=torch.float32)
        expected = model(x, phase).detach()
        normal = {
            "x_mean": torch.zeros(INPUT_LAYOUT.size),
            "x_std": torch.ones(INPUT_LAYOUT.size),
            "y_mean": torch.zeros(OUTPUT_LAYOUT.size),
            "y_std": torch.ones(OUTPUT_LAYOUT.size),
        }
        seed = finite_runtime_seed()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_classic_checkpoint(
                path,
                model=model,
                normalization=normal,
                runtime_seed=seed,
                dataset_digest="a" * 64,
                kinematic_signature_sha256="b" * 64,
                joint_limits=torch.tensor([[-2.0, 2.0]] * 29),
                phase_advance_q99=0.2,
                epoch=3,
                validation_loss=0.125,
                seed=7,
            )
            loaded = load_classic_checkpoint(path)

        actual = loaded.build_model().eval()(x, phase).detach()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(loaded.epoch, 3)
        self.assertEqual(loaded.dataset_digest, "a" * 64)
        self.assertEqual(loaded.validation_loss, 0.125)

    def test_small_fixed_set_overfits_with_classic_objective(self) -> None:
        torch.manual_seed(19)
        model = PhaseFunctionedNetwork(hidden_size=64, dropout_probability=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        x = torch.randn((16, INPUT_LAYOUT.size), dtype=torch.float32)
        phase = torch.linspace(0.0, 2.0 * np.pi, 16, dtype=torch.float32)
        target = torch.randn((16, OUTPUT_LAYOUT.size), dtype=torch.float32)
        target[:, OUTPUT_LAYOUT["contact_logit"]] = torch.randint(
            0, 2, (16, 4), dtype=torch.int64
        ).to(torch.float32)
        model.eval()
        initial = float(classic_pfnn_loss(model(x, phase), target).detach())

        model.train()
        for _ in range(300):
            optimizer.zero_grad(set_to_none=True)
            loss = classic_pfnn_loss(model(x, phase), target)
            loss.backward()
            optimizer.step()

        model.eval()
        final = float(classic_pfnn_loss(model(x, phase), target).detach())
        self.assertLess(final, 0.1 * initial)


if __name__ == "__main__":
    unittest.main()
