from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

import mm_sonic.train_terrain_pfnn as train_module
from mm_sonic.evaluate_terrain_pfnn import (
    claim_sealed_test_receipt,
    validate_checkpoint_kinematics,
)
from mm_sonic.train_terrain_pfnn import (
    materialize_subset,
    overfit_gate_accepted,
    promote_pipeline_best,
    stratified_subset,
)
from mm_sonic.terrain_pfnn.dataset import normalize_pfnn_input
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.training import (
    CHECKPOINT_SCHEMA,
    DEFAULT_LOSS_WEIGHTS,
    LOSS_WEIGHT_KEYS,
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
    def test_rollout_default_depends_on_pipeline_mode_and_explicit_value_wins(self) -> None:
        resolve = train_module.resolve_rollout_finetune_frames
        self.assertEqual(resolve(None, pipeline_overfit=True), 0)
        self.assertEqual(resolve(None, pipeline_overfit=False), 16)
        self.assertEqual(resolve(3, pipeline_overfit=True), 3)
        self.assertEqual(resolve(0, pipeline_overfit=False), 0)
        for invalid in (-1, 1, 17):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "rollout fine-tuning"):
                    resolve(invalid, pipeline_overfit=False)

    def test_overfit_gate_uses_the_final_post_rollout_score(self) -> None:
        self.assertTrue(overfit_gate_accepted(10.0, 4.9, 0.5))
        self.assertFalse(overfit_gate_accepted(10.0, 5.1, 0.5))
        self.assertTrue(promote_pipeline_best(pipeline_overfit=True, accepted=True))
        self.assertFalse(promote_pipeline_best(pipeline_overfit=True, accepted=False))
        self.assertFalse(promote_pipeline_best(pipeline_overfit=False, accepted=True))

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

    def test_loss_multipliers_are_frozen_to_exactly_one_at_every_boundary(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
        prediction = torch.zeros(2, OUTPUT_LAYOUT.size)
        target = torch.zeros_like(prediction)
        for name in LOSS_WEIGHT_KEYS:
            with self.subTest(training_weight=name):
                weights = dict(DEFAULT_LOSS_WEIGHTS)
                weights[name] = 0.5
                with self.assertRaisesRegex(ValueError, "loss weights"):
                    pfnn_losses(
                        prediction,
                        target,
                        model=model,
                        normalization=_normalization(),
                        loss_weights=weights,
                    )
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            for weights in (
                {name: 1.0 for name in LOSS_WEIGHT_KEYS[:-1]},
                {**DEFAULT_LOSS_WEIGHTS, "extra": 1.0},
            ):
                with self.subTest(save_weights=tuple(weights)):
                    with self.assertRaisesRegex(ValueError, "loss weights"):
                        save_checkpoint(
                            path,
                            model,
                            optimizer,
                            _normalization(),
                            dataset_digest="abc",
                            kinematic_signature_sha256="def",
                            runtime_seed=finite_runtime_seed(),
                            step=1,
                            loss_weights=weights,
                        )
            save_checkpoint(
                path,
                model,
                optimizer,
                _normalization(),
                dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(),
                step=1,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["loss_weights"]["regularization"] = 2.0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "loss weights"):
                load_checkpoint(
                    path,
                    expected_dataset_digest="abc",
                    expected_kinematic_signature_sha256="def",
                )

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
                runtime_seed=seed, step=17, epoch=3,
                sampler_epoch=2, sampler_global_offset=8,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["input_layout"], [list(field) for field in INPUT_LAYOUT.fields])
            self.assertEqual(payload["output_layout"], [list(field) for field in OUTPUT_LAYOUT.fields])
            self.assertEqual(
                payload["normalization_contract"]["root_tilt_encoding"],
                "angle_axis_xy",
            )
            self.assertEqual(
                payload["sampler_state"], {"epoch": 2, "global_offset": 8}
            )
            loaded = load_checkpoint(
                path, expected_dataset_digest="abc",
                expected_kinematic_signature_sha256="def",
            )
            self.assertEqual(loaded.schema, CHECKPOINT_SCHEMA)
            self.assertEqual(loaded.step, 17)
            self.assertEqual(loaded.sampler_epoch, 2)
            self.assertEqual(loaded.sampler_global_offset, 8)
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
                (17, 3),
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

    def test_checkpoint_rejects_inexact_nested_tensor_contracts(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.pt"
            tampered = root / "tampered.pt"
            save_checkpoint(
                baseline,
                model,
                optimizer,
                _normalization(),
                dataset_digest="abc",
                kinematic_signature_sha256="def",
                runtime_seed=finite_runtime_seed(),
                step=1,
            )

            def rejected(label: str, mutation: object, match: str) -> None:
                payload = torch.load(baseline, map_location="cpu", weights_only=True)
                mutation(payload)
                torch.save(payload, tampered)
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, match):
                        load_checkpoint(
                            tampered,
                            expected_dataset_digest="abc",
                            expected_kinematic_signature_sha256="def",
                        )

            rejected(
                "normalization extra key",
                lambda value: value["normalization"].__setitem__("extra", torch.zeros(1)),
                "normalization",
            )
            rejected(
                "normalization missing key",
                lambda value: value["normalization"].pop("x_mean"),
                "normalization",
            )
            rejected(
                "normalization list",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", value["normalization"]["x_mean"].tolist()
                ),
                "normalization",
            )
            rejected(
                "normalization dtype",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", value["normalization"]["x_mean"].to(torch.float64)
                ),
                "normalization",
            )
            rejected(
                "normalization numpy",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", value["normalization"]["x_mean"].numpy()
                ),
                "loaded safely|normalization",
            )
            rejected(
                "normalization stride",
                lambda value: value["normalization"].__setitem__(
                    "x_mean", torch.zeros(INPUT_LAYOUT.size, 2)[:, 0]
                ),
                "normalization",
            )
            rejected(
                "runtime seed extra key",
                lambda value: value["runtime_seed"].__setitem__("extra", None),
                "runtime seed",
            )
            rejected(
                "runtime seed missing key",
                lambda value: value["runtime_seed"].pop("world_xy"),
                "runtime seed",
            )
            rejected(
                "runtime seed list",
                lambda value: value["runtime_seed"].__setitem__(
                    "joint_position", value["runtime_seed"]["joint_position"].tolist()
                ),
                "runtime seed",
            )
            rejected(
                "runtime seed dtype",
                lambda value: value["runtime_seed"].__setitem__(
                    "joint_position",
                    value["runtime_seed"]["joint_position"].to(torch.float64),
                ),
                "runtime seed",
            )
            rejected(
                "runtime seed numpy",
                lambda value: value["runtime_seed"].__setitem__(
                    "joint_position",
                    value["runtime_seed"]["joint_position"].numpy(),
                ),
                "loaded safely|runtime seed",
            )
            rejected(
                "runtime seed stride",
                lambda value: value["runtime_seed"].__setitem__(
                    "body_position", torch.zeros(30, 6)[:, ::2]
                ),
                "runtime seed",
            )
            rejected(
                "runtime provenance extra key",
                lambda value: value["runtime_seed"]["provenance"].__setitem__(
                    "extra", 0
                ),
                "runtime seed provenance",
            )
            rejected(
                "joint limits list",
                lambda value: value.__setitem__(
                    "joint_limits", value["joint_limits"].tolist()
                ),
                "joint limits",
            )
            rejected(
                "joint limits dtype",
                lambda value: value.__setitem__(
                    "joint_limits", value["joint_limits"].to(torch.float32)
                ),
                "joint limits",
            )
            rejected(
                "joint limits numpy",
                lambda value: value.__setitem__(
                    "joint_limits", value["joint_limits"].numpy()
                ),
                "loaded safely|joint limits",
            )
            rejected(
                "joint limits stride",
                lambda value: value.__setitem__(
                    "joint_limits", torch.zeros(29, 4, dtype=torch.float64)[:, ::2]
                ),
                "joint limits",
            )
            rejected(
                "sampler missing key",
                lambda value: value["sampler_state"].pop("global_offset"),
                "sampler state",
            )
            rejected(
                "sampler extra key",
                lambda value: value["sampler_state"].__setitem__("extra", 0),
                "sampler state",
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

    def test_nonzero_normalization_rollout_matches_public_input_helper(self) -> None:
        torch.manual_seed(19)
        model = PhaseFunctionedNetwork(hidden_size=8, dropout_probability=0.0)
        normal = _normalization()
        normal["x_mean"] = np.linspace(-2.0, 3.0, INPUT_LAYOUT.size, dtype=np.float32)
        normal["x_std"] = np.linspace(0.5, 2.5, INPUT_LAYOUT.size, dtype=np.float32)
        normal["y_mean"] = np.linspace(-1.0, 2.0, OUTPUT_LAYOUT.size, dtype=np.float32)
        normal["y_std"] = np.linspace(0.75, 1.75, OUTPUT_LAYOUT.size, dtype=np.float32)
        normal["y_mean"][OUTPUT_LAYOUT["contact_logit"]] = 0.0
        normal["y_std"][OUTPUT_LAYOUT["contact_logit"]] = 1.0
        raw_inputs = np.linspace(
            -4.0, 5.0, 2 * INPUT_LAYOUT.size, dtype=np.float32
        ).reshape(2, 1, INPUT_LAYOUT.size)
        normalized_inputs = normalize_pfnn_input(
            raw_inputs, normal["x_mean"], normal["x_std"]
        )
        targets = torch.zeros(2, 1, OUTPUT_LAYOUT.size)
        result = autoregressive_unroll(
            model,
            torch.from_numpy(normalized_inputs),
            torch.tensor((0.3,)),
            targets,
            normalization=normal,
        )
        prediction_physical = (
            result.predictions[0].detach().numpy() * normal["y_std"]
            + normal["y_mean"]
        )
        expected_raw = raw_inputs[1].copy()
        for y_field, x_field in (
            ("trajectory_position", "trajectory_position"),
            ("trajectory_direction", "trajectory_direction"),
            ("body_position", "previous_body_position"),
            ("body_velocity", "previous_body_velocity"),
        ):
            expected_raw[:, INPUT_LAYOUT[x_field]] = prediction_physical[
                :, OUTPUT_LAYOUT[y_field]
            ]
        expected = normalize_pfnn_input(
            expected_raw, normal["x_mean"], normal["x_std"]
        )
        torch.testing.assert_close(result.inputs[1], torch.from_numpy(expected))

    def test_phase_recurrence_clamps_forward_but_keeps_negative_advance_gradient(self) -> None:
        model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.b2[:, OUTPUT_LAYOUT["phase_advance"]] = -0.25
        inputs = torch.zeros(2, 1, INPUT_LAYOUT.size)
        targets = torch.zeros(2, 1, OUTPUT_LAYOUT.size)
        initial_phase = torch.tensor((0.7,))
        result = autoregressive_unroll(
            model, inputs, initial_phase, targets, normalization=_normalization()
        )
        self.assertLess(
            float(
                result.predictions[0][0, OUTPUT_LAYOUT["phase_advance"]][0].detach()
            ),
            0.0,
        )
        torch.testing.assert_close(result.phases[1], initial_phase)
        later_phase_only_loss = result.phases[1].square().sum()
        gradient = torch.autograd.grad(
            later_phase_only_loss, result.predictions[0], retain_graph=True
        )[0]
        phase_gradient = gradient[:, OUTPUT_LAYOUT["phase_advance"]]
        self.assertTrue(torch.isfinite(phase_gradient).all())
        self.assertGreater(float(phase_gradient.abs().max()), 0.0)

    def test_rollout_sequences_ignore_storage_order_and_reject_bad_metadata(self) -> None:
        rows = [
            {"clip_id": "walk4_subject1", "center_frame": 11, "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 2, "terrain_class": "flat"},
            {"clip_id": "walk4_subject1", "center_frame": 10, "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 1, "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 4, "terrain_class": "flat"},
            {"clip_id": "walk1_subject1", "center_frame": 5, "terrain_class": "flat"},
        ]
        rows.extend(
            {"clip_id": "walk1_subject1", "center_frame": 3, "terrain_class": "flat"}
            for _ in range(8)
        )

        class Rows:
            split = "train"

            def __init__(self, values: list[dict[str, object]]) -> None:
                self.values = values

            def __len__(self) -> int:
                return len(self.values)

            def __getitem__(self, index: int) -> dict[str, object]:
                value = dict(self.values[index])
                value.setdefault("split", "train")
                value.setdefault("split_identity", value["clip_id"])
                return value

        def metadata_sequences(values: list[dict[str, object]]) -> list[tuple[object, ...]]:
            sequences, _ = train_module._consecutive_starts(Rows(values), 2)
            return [
                tuple(
                    (values[index]["clip_id"], values[index]["center_frame"])
                    for index in sequence
                )
                for sequence in sequences
            ]

        expected = [
            (("walk1_subject1", 1), ("walk1_subject1", 2)),
            (("walk1_subject1", 4), ("walk1_subject1", 5)),
            (("walk4_subject1", 10), ("walk4_subject1", 11)),
        ]
        self.assertEqual(metadata_sequences(rows), expected)
        shuffled = [rows[index] for index in (10, 0, 5, 3, 8, 2, 13, 1, 6, 4, 12, 7, 9, 11)]
        self.assertEqual(metadata_sequences(shuffled), expected)
        self.assertFalse(
            any(center == 3 for sequence in expected for _, center in sequence)
        )

        mismatched = [dict(rows[3]), dict(rows[1])]
        mismatched[1]["split_identity"] = "different"
        with self.assertRaisesRegex(ValueError, "split/identity"):
            train_module._consecutive_starts(Rows(mismatched), 2)
        validation = Rows([rows[3], rows[1]])
        validation.split = "validation"
        with self.assertRaisesRegex(ValueError, "training split"):
            train_module._consecutive_starts(validation, 2)

    def test_sampler_resume_reconstructs_identical_remaining_stream(self) -> None:
        classes = ["flat"] * 4 + ["ascent"] * 4 + ["descent"] * 4 + ["transition"] * 4
        global_epoch = train_module._balanced_epoch_indices(
            list(range(16)), classes, seed=23, epoch=5
        )
        uninterrupted, padding = train_module._padded_rank_epoch(
            global_epoch, batch_size=2, rank=1, world_size=2
        )
        consumed_local = 4
        resumed, resumed_offset, resumed_padding = (
            train_module._local_epoch_at_global_offset(
                global_epoch,
                global_offset=consumed_local * 2,
                batch_size=2,
                rank=1,
                world_size=2,
            )
        )
        self.assertEqual(resumed_padding, padding)
        self.assertEqual(resumed_offset, consumed_local)
        self.assertEqual(resumed[resumed_offset:], uninterrupted[consumed_local:])
        for invalid in (-1, 2, len(global_epoch) + 4):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "sampler"):
                    train_module._local_epoch_at_global_offset(
                        global_epoch,
                        global_offset=invalid,
                        batch_size=2,
                        rank=1,
                        world_size=2,
                    )

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
