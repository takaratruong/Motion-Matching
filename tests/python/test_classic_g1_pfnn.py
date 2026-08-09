from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from mm_sonic.train_classic_g1_pfnn import (
    _grail_train_validation_masks,
    _mirror_normalized_examples,
    _parser,
    balanced_epoch_batches,
    classic_pfnn_loss,
    load_classic_checkpoint,
    save_classic_checkpoint,
    train,
    vertical_epoch_batches,
)
from mm_sonic.build_g1_pfnn_vertical_dataset import (
    build_vertical_dataset,
    save_vertical_dataset,
)
from tests.python.test_build_g1_pfnn_vertical_dataset import _source
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

    def test_native_g1_training_batches_exclude_lafan_retarget_rows(self) -> None:
        source_kind = np.asarray(("grail",) * 5 + ("lafan",) * 9, dtype="<U6")

        batches = balanced_epoch_batches(
            source_kind,
            batch_size=4,
            seed=23,
            epoch=0,
            source_filter="grail",
        )

        used = np.concatenate(batches)
        self.assertTrue(np.all(source_kind[used] == "grail"))
        self.assertTrue(set(range(5)).issubset(set(used.tolist())))
        self.assertEqual(_parser().parse_args(["--dataset", "d", "--model-path", "m", "--output", "o"]).train_source, "grail")

    def test_released_pfnn_batches_are_deterministic_and_cover_each_row_once(self) -> None:
        first = vertical_epoch_batches(11, batch_size=4, seed=23, epoch=2)
        second = vertical_epoch_batches(11, batch_size=4, seed=23, epoch=2)
        self.assertEqual(len(first), 3)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(11))
        self.assertEqual(
            _parser().parse_args(
                ["--dataset", "d", "--model-path", "m", "--output", "o", "--train-source", "released-pfnn"]
            ).train_source,
            "released-pfnn",
        )
        self.assertEqual(
            _parser().parse_args(
                ["--dataset", "d", "--model-path", "m", "--output", "o"]
            ).runtime_seed,
            "flat",
        )

    def test_native_g1_validation_holds_out_last_present_variant_per_family(self) -> None:
        clips = np.asarray(
            (
                "terrain_slopes__slope_a__000",
                "terrain_slopes__slope_a__004",
                "terrain_slopes__slope_b__001",
                "terrain_slopes__slope_b__003",
                "walk1_subject1",
            )
        )
        source = np.asarray(("grail", "grail", "grail", "grail", "lafan"))

        optimized, held_out = _grail_train_validation_masks(source, clips)

        np.testing.assert_array_equal(optimized, (True, False, True, False, False))
        np.testing.assert_array_equal(held_out, (False, True, False, True, False))

    def test_native_g1_training_mirrors_yaw_lateral_motion_and_phase(self) -> None:
        x = np.zeros((1, INPUT_LAYOUT.size), dtype=np.float32)
        y = np.zeros((1, OUTPUT_LAYOUT.size), dtype=np.float32)
        y[0, OUTPUT_LAYOUT["root_planar_velocity"]] = (0.5, -0.2)
        y[0, OUTPUT_LAYOUT["root_yaw_velocity"]] = 0.7
        phase = np.asarray((0.4,), dtype=np.float32)
        normal = {
            "x_mean": np.zeros(INPUT_LAYOUT.size, dtype=np.float32),
            "x_std": np.ones(INPUT_LAYOUT.size, dtype=np.float32),
            "y_mean": np.zeros(OUTPUT_LAYOUT.size, dtype=np.float32),
            "y_std": np.ones(OUTPUT_LAYOUT.size, dtype=np.float32),
        }

        mirrored_x, mirrored_y, mirrored_phase = _mirror_normalized_examples(
            x, y, phase, normal
        )

        self.assertEqual(mirrored_x.shape, x.shape)
        np.testing.assert_allclose(
            mirrored_y[0, OUTPUT_LAYOUT["root_planar_velocity"]], (0.5, 0.2)
        )
        np.testing.assert_allclose(
            mirrored_y[0, OUTPUT_LAYOUT["root_yaw_velocity"]], (-0.7,)
        )
        self.assertAlmostEqual(
            float(mirrored_phase[0]), (0.4 + np.pi) % (2.0 * np.pi), places=6
        )

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
                source_kind="released_pfnn",
                vertical_slice_receipt_sha256="c" * 64,
                terrain_receipt_set_sha256="d" * 64,
            )
            loaded = load_classic_checkpoint(path)

        actual = loaded.build_model().eval()(x, phase).detach()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(loaded.epoch, 3)
        self.assertEqual(loaded.dataset_digest, "a" * 64)
        self.assertEqual(loaded.validation_loss, 0.125)
        self.assertEqual(loaded.source_kind, "released_pfnn")
        self.assertEqual(loaded.vertical_slice_receipt_sha256, "c" * 64)
        self.assertEqual(loaded.terrain_receipt_set_sha256, "d" * 64)

    def test_trains_one_vertical_epoch_and_binds_dataset_receipts(self) -> None:
        dataset = build_vertical_dataset(
            (_source("train", "released_train"), _source("validation", "released_val"))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = save_vertical_dataset(root / "dataset", dataset)
            arguments = _parser().parse_args(
                [
                    "--dataset",
                    str(manifest),
                    "--model-path",
                    str(root / "unused.xml"),
                    "--output",
                    str(root / "run"),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "8",
                    "--evaluation-batch-size",
                    "16",
                    "--hidden-size",
                    "16",
                    "--train-source",
                    "released-pfnn",
                ]
            )
            fake_kinematics = mock.Mock(
                joint_limits=torch.tensor([[-2.0, 2.0]] * 29, dtype=torch.float64),
                kinematic_signature_sha256="9" * 64,
            )
            with mock.patch(
                "mm_sonic.train_classic_g1_pfnn.TorchG1ForwardKinematics.from_mjcf",
                return_value=fake_kinematics,
            ):
                checkpoint_path = train(arguments)
            checkpoint = load_classic_checkpoint(checkpoint_path)
        self.assertEqual(checkpoint.source_kind, "released_pfnn")
        self.assertEqual(checkpoint.dataset_digest, dataset.dataset_sha256)
        self.assertEqual(checkpoint.vertical_slice_receipt_sha256, dataset.selection_sha256)
        self.assertEqual(checkpoint.terrain_receipt_set_sha256, dataset.terrain_receipt_set_sha256)

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
