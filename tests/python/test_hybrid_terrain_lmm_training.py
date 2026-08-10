from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from mm_sonic.hybrid_terrain_lmm_training import (
    HybridModelConfig,
    _MetricAccumulator,
    _physical_metrics_receipt_accepted,
    _read_bound_artifact,
    _rename_directory_noreplace,
    assemble_training_batch,
    calculate_physical_metrics,
    evaluate_hybrid_generator,
    hybrid_generator_loss,
    iter_coverage_batches,
    load_hybrid_generator,
    main,
    sample_training_rows,
    train_hybrid_generator,
)

from resources.g1_terrain_builder.schema import ArtifactSet, FeatureSet


def _synthetic_corpus(*, manifest_sha256: str = "a" * 64) -> SimpleNamespace:
    frames = 24
    bones = 31
    positions = np.zeros((frames, bones, 3), dtype=np.float32)
    positions[:, 1:, 1] = np.float32(0.04)
    positions[:, 1:, 0] = np.linspace(0.0, 0.12, bones - 1, dtype=np.float32)[None]
    velocities = np.zeros_like(positions)
    angular_velocities = np.zeros_like(positions)
    rotations = np.zeros((frames, bones, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    parents = np.arange(-1, bones - 1, dtype=np.int32)
    range_starts = np.arange(0, frames, 4, dtype=np.int32)
    range_stops = range_starts + 4
    contacts = np.zeros((frames, 2), dtype=np.uint8)
    contacts[::2, 0] = 1
    contacts[1::2, 1] = 1
    artifacts = ArtifactSet(
        positions=positions,
        velocities=velocities,
        rotations=rotations,
        angular_velocities=angular_velocities,
        parents=parents,
        range_starts=range_starts,
        range_stops=range_stops,
        contacts=contacts,
        terrain_features=np.zeros((frames, 4), dtype=np.float32),
        terrain_support=np.zeros((frames, 3), dtype=np.float32),
    )
    feature_values = (
        np.random.default_rng(11).normal(size=(frames, 31)).astype(np.float32)
    )
    features = FeatureSet(
        values=feature_values,
        offset=np.zeros(31, dtype=np.float32),
        scale=np.ones(31, dtype=np.float32),
    )
    family_ids = np.repeat(np.arange(3, dtype=np.int32), 8)
    evaluation_mask = np.zeros(frames, dtype=bool)
    evaluation_mask[4:8] = True
    evaluation_mask[12:16] = True
    evaluation_mask[20:24] = True
    train_mask = ~evaluation_mask
    return SimpleNamespace(
        artifacts=artifacts,
        features=features,
        family_ids=family_ids,
        family_names=("takara", "grail-slope", "grail-stair"),
        train_mask=train_mask,
        evaluation_mask=evaluation_mask,
        manifest_sha256=manifest_sha256,
        manifest={"schema": "synthetic-hybrid-corpus/v1"},
    )


class HybridTerrainLmmTrainingTests(unittest.TestCase):
    def test_immutable_model_publication_never_replaces_a_racing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            output = root / "model"
            staging.mkdir()
            output.mkdir()
            (staging / "new").write_text("new", encoding="utf-8")
            (output / "sentinel").write_text("owned", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                _rename_directory_noreplace(staging, output)

            self.assertEqual((output / "sentinel").read_text(encoding="utf-8"), "owned")
            self.assertTrue(staging.is_dir())

    def test_supported_model_variants_have_the_frozen_architectures(self) -> None:
        expected = {
            "latent32": (32, 512),
            "latent64": (64, 512),
            "wider": (64, 1024),
        }
        for variant, (latent, width) in expected.items():
            config = HybridModelConfig(variant=variant, device="cpu")
            self.assertEqual(config.latent_size, latent)
            self.assertEqual(config.width, width)
            self.assertEqual(config.dt, 0.04)

        sixty_hz = HybridModelConfig(dt=1.0 / 60.0, device="cpu")
        self.assertEqual(sixty_hz.dt, 1.0 / 60.0)
        with self.assertRaisesRegex(ValueError, "dt"):
            HybridModelConfig(dt=1.0 / 50.0, device="cpu")

    def test_visual_articulation_loss_prioritizes_native_pose_channels(self) -> None:
        def loss_with_error(profile: str, channel: int) -> float:
            prediction = torch.zeros((1, 458), dtype=torch.float32)
            prediction[:, 456:] = -100.0
            prediction[0, channel] = 2.0
            target = torch.zeros((1, 458), dtype=torch.float32)
            latent = torch.zeros((1, 32), dtype=torch.float32)
            return float(
                hybrid_generator_loss(
                    prediction,
                    target,
                    latent,
                    torch.zeros(456),
                    torch.ones(456),
                    loss_profile=profile,
                )
            )

        uniform_rotation = loss_with_error("uniform", 90)
        uniform_angular_velocity = loss_with_error("uniform", 360)
        visual_rotation = loss_with_error("visual-articulation", 90)
        visual_hips_height = loss_with_error("visual-articulation", 1)
        visual_angular_velocity = loss_with_error("visual-articulation", 360)

        self.assertAlmostEqual(uniform_rotation, 1.5 / 456.0, places=7)
        self.assertAlmostEqual(uniform_rotation, uniform_angular_velocity, places=7)
        self.assertGreater(visual_rotation, uniform_rotation)
        self.assertGreater(visual_rotation, 20.0 * visual_angular_velocity)
        self.assertGreater(visual_hips_height, 20.0 * visual_rotation)
        self.assertEqual(
            HybridModelConfig(loss_profile="visual-articulation").loss_weights,
            {
                "rotation_6d_mean": 1.0,
                "hips_height": 0.25,
                "auxiliary_continuous_mean": 0.05,
                "contact_bce": 0.1,
                "latent_l2": 0.0001,
            },
        )
        with self.assertRaisesRegex(ValueError, "loss profile"):
            HybridModelConfig(loss_profile="unsupported")

    def test_batches_are_family_balanced_and_never_use_evaluation_rows(self) -> None:
        corpus = _synthetic_corpus()
        rows = sample_training_rows(
            corpus, batch_size=12, generator=np.random.default_rng(7)
        )
        self.assertTrue(np.all(corpus.train_mask[rows]))
        self.assertEqual(set(corpus.family_ids[rows].tolist()), {0, 1, 2})
        counts = np.bincount(corpus.family_ids[rows], minlength=3)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_coverage_pass_is_shuffled_deterministic_and_exactly_once(self) -> None:
        corpus = _synthetic_corpus()
        first = list(
            iter_coverage_batches(
                corpus, batch_size=5, generator=np.random.default_rng(0)
            )
        )
        second = list(
            iter_coverage_batches(
                corpus, batch_size=5, generator=np.random.default_rng(0)
            )
        )
        np.testing.assert_array_equal(np.concatenate(first), np.concatenate(second))
        covered = np.concatenate(first)
        np.testing.assert_array_equal(
            np.sort(covered), np.flatnonzero(corpus.train_mask)
        )
        self.assertFalse(np.array_equal(covered, np.sort(covered)))

    def test_physical_metrics_apply_the_practical_held_out_gates(self) -> None:
        corpus = _synthetic_corpus()
        rows = np.flatnonzero(corpus.evaluation_mask)
        target = assemble_training_batch(corpus, rows).target
        metrics = calculate_physical_metrics(corpus, rows, target)
        self.assertTrue(metrics["accepted"])
        self.assertEqual(metrics["joint_geodesic_mae_rad"], 0.0)
        self.assertEqual(metrics["joint_frame_max_p95_rad"], 0.0)
        self.assertEqual(metrics["local_position_p95_m"], 0.0)
        self.assertEqual(metrics["fk_body_position_p95_m"], 0.0)
        self.assertEqual(metrics["support_foot_position_p95_m"], 0.0)
        self.assertEqual(metrics["contact_f1"], [1.0, 1.0])

        bad = target.copy()
        bad[:, :90] += np.float32(0.2)
        rejected = calculate_physical_metrics(corpus, rows, bad)
        self.assertFalse(rejected["accepted"])
        self.assertGreater(rejected["local_position_p95_m"], 0.03)

    def test_support_metric_uses_canonical_toes_not_ankles(self) -> None:
        corpus = _synthetic_corpus()
        # Make every body independent in FK so an ankle-local perturbation does
        # not propagate into its toe child and obscure which indices are gated.
        corpus.artifacts.parents[1:] = 0
        rows = np.flatnonzero(corpus.evaluation_mask)
        target = assemble_training_batch(corpus, rows).target

        ankle_only = target.copy()
        ankle_only[:, (6 - 1) * 3] += np.float32(0.1)
        ankle_metrics = calculate_physical_metrics(corpus, rows, ankle_only)
        self.assertEqual(ankle_metrics["support_foot_position_p95_m"], 0.0)

        toe_only = target.copy()
        toe_only[:, (7 - 1) * 3] += np.float32(0.1)
        toe_metrics = calculate_physical_metrics(corpus, rows, toe_only)
        self.assertGreater(toe_metrics["support_foot_position_p95_m"], 0.09)

    def test_nonfinite_decodes_are_rejected(self) -> None:
        corpus = _synthetic_corpus()
        rows = np.flatnonzero(corpus.evaluation_mask)
        target = assemble_training_batch(corpus, rows).target
        target[0, 0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            calculate_physical_metrics(corpus, rows, target)

    def test_metric_accumulator_keeps_local_position_gate_diagnostic_only(self) -> None:
        accumulator = _MetricAccumulator.empty()
        accumulator.rows = 4
        accumulator.joint_count = 4
        accumulator.joint_sum = 0.04
        accumulator.frame_joint_max = [np.full(4, 0.02, dtype=np.float32)]
        accumulator.local_position_max = [np.full(4, 0.50, dtype=np.float32)]
        accumulator.fk_position_max = [np.full(4, 0.02, dtype=np.float32)]
        accumulator.support_position_max = [np.full(4, 0.01, dtype=np.float32)]
        accumulator.true_positive = np.asarray([4, 4], dtype=np.int64)
        accumulator.false_positive = np.zeros(2, dtype=np.int64)
        accumulator.false_negative = np.zeros(2, dtype=np.int64)

        metrics = accumulator.finish()

        self.assertTrue(metrics["accepted"])
        self.assertEqual(metrics["local_position_p95_m"], 0.50)
        self.assertEqual(
            metrics["gate_limits"],
            {
                "joint_geodesic_mae_rad": 0.03,
                "joint_frame_max_p95_rad": 0.10,
                "fk_body_position_p95_m": 0.08,
                "support_foot_position_p95_m": 0.05,
                "minimum_contact_f1": 0.85,
            },
        )

    def test_local_position_metric_is_diagnostic_only_not_an_acceptance_gate(
        self,
    ) -> None:
        metrics = {
            "finite": True,
            "rows": 4,
            "joint_geodesic_mae_rad": 0.01,
            "joint_frame_max_p95_rad": 0.02,
            "local_position_p95_m": 0.50,
            "fk_body_position_p95_m": 0.02,
            "support_foot_position_p95_m": 0.01,
            "contact_f1": [0.99, 0.99],
            "gate_limits": {
                "joint_geodesic_mae_rad": 0.03,
                "joint_frame_max_p95_rad": 0.10,
                "fk_body_position_p95_m": 0.08,
                "support_foot_position_p95_m": 0.05,
                "minimum_contact_f1": 0.85,
            },
            "accepted": True,
        }
        self.assertTrue(_physical_metrics_receipt_accepted(metrics))

    def test_tiny_training_is_deterministic_reloadable_immutable_and_hash_bound(
        self,
    ) -> None:
        corpus = _synthetic_corpus()
        config = HybridModelConfig(
            variant="latent32",
            seed=23,
            device="cpu",
            batch_size=8,
            post_coverage_steps=1,
            normalization_chunk_size=7,
            evaluation_chunk_size=6,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path = root / "first"
            second_path = root / "second"
            first_receipt = train_hybrid_generator(corpus, first_path, config=config)
            second_receipt = train_hybrid_generator(corpus, second_path, config=config)
            self.assertEqual(
                first_receipt["coverage_rows"], int(np.count_nonzero(corpus.train_mask))
            )
            self.assertEqual(first_receipt["losses"], second_receipt["losses"])
            self.assertEqual(
                {path.name for path in first_path.iterdir()},
                {
                    "latent.npy",
                    "model.pt",
                    "training.json",
                    "evaluation.json",
                    "manifest.json",
                },
            )
            first = load_hybrid_generator(first_path, corpus=corpus, device="cpu")
            second = load_hybrid_generator(second_path, corpus=corpus, device="cpu")
            self.assertFalse(first.canonical_selection_verified)
            self.assertFalse(first.selection_provenance_verified)
            flipped_load_path = root / "flipped-load"
            shutil.copytree(first_path, flipped_load_path)
            flipped_load_manifest = json.loads(
                (flipped_load_path / "manifest.json").read_text(encoding="utf-8")
            )
            flipped_load_manifest["canonical_selection_accepted"] = True
            (flipped_load_path / "manifest.json").write_text(
                json.dumps(flipped_load_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical.*inconsistent"):
                load_hybrid_generator(flipped_load_path, corpus=corpus, device="cpu")
            latent_swap_path = root / "latent-swap"
            shutil.copytree(first_path, latent_swap_path)
            expected_latent = np.load(
                latent_swap_path / "latent.npy", allow_pickle=False
            ).copy()
            alternate_latent = root / "alternate-latent.npy"
            np.save(alternate_latent, np.full_like(expected_latent, 123.0))
            alternate_latent_bytes = alternate_latent.read_bytes()
            original_np_load = np.load
            swapped = False

            def swap_before_numpy_open(*args: object, **kwargs: object) -> object:
                nonlocal swapped
                if not swapped:
                    with (latent_swap_path / "latent.npy").open("r+b") as output:
                        output.write(alternate_latent_bytes)
                        output.flush()
                    swapped = True
                return original_np_load(*args, **kwargs)

            with mock.patch.object(np, "load", side_effect=swap_before_numpy_open):
                swapped_generator = load_hybrid_generator(
                    latent_swap_path, corpus=corpus, device="cpu"
                )
            np.testing.assert_array_equal(swapped_generator.latent, expected_latent)
            rows = np.array([0, 3, 8], dtype=np.int64)
            decoded_first = first.decode_rows(corpus.features.values[rows], rows)
            decoded_second = second.decode_rows(corpus.features.values[rows], rows)
            np.testing.assert_array_equal(decoded_first, decoded_second)
            np.testing.assert_array_equal(first.latent, second.latent)
            evaluation = evaluate_hybrid_generator(corpus, first, chunk_size=5)
            self.assertEqual(evaluation["corpus_manifest_sha256"], "a" * 64)
            with self.assertRaises(FileExistsError):
                train_hybrid_generator(corpus, first_path, config=config)

            changed = copy.copy(corpus)
            changed.manifest_sha256 = "b" * 64
            with self.assertRaisesRegex(ValueError, "corpus manifest"):
                load_hybrid_generator(first_path, corpus=changed, device="cpu")

            manifest = json.loads(
                (first_path / "manifest.json").read_text(encoding="utf-8")
            )
            evaluation_bytes = (first_path / "evaluation.json").read_bytes()
            with mock.patch.object(Path, "read_bytes", return_value=b"swapped"):
                self.assertEqual(
                    _read_bound_artifact(first_path, manifest, "evaluation.json"),
                    evaluation_bytes,
                )
            self.assertEqual(manifest["corpus_manifest_sha256"], "a" * 64)
            self.assertEqual(manifest["config"]["loss_profile"], "uniform")
            self.assertEqual(manifest["config"]["loss_weights"]["contact_bce"], 1.0)
            for name in ("latent.npy", "model.pt", "training.json", "evaluation.json"):
                self.assertEqual(len(manifest["artifacts"][name]["sha256"]), 64)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["inspect", "--model", str(first_path)]), 0)
            inspected = json.loads(stdout.getvalue())
            self.assertEqual(inspected["manifest"]["schema"], manifest["schema"])

            refit_config = HybridModelConfig(
                variant="latent32",
                seed=23,
                device="cpu",
                batch_size=8,
                post_coverage_steps=1,
                normalization_chunk_size=7,
                evaluation_chunk_size=6,
                fit_all_rows=True,
            )
            with self.assertRaisesRegex(ValueError, "selection model"):
                train_hybrid_generator(
                    corpus, root / "missing-selection", config=refit_config
                )
            with self.assertRaisesRegex(TypeError, "selection model"):
                train_hybrid_generator(
                    corpus,
                    root / "mapping-selection",
                    config=refit_config,
                    selection_model={"canonical_selection_accepted": True},
                )

            # Flipping only the old trust Boolean must not authorize a refit.
            flipped_path = root / "flipped-selection"
            shutil.copytree(first_path, flipped_path)
            flipped_evaluation = json.loads(
                (flipped_path / "evaluation.json").read_text(encoding="utf-8")
            )
            self.assertFalse(flipped_evaluation["canonical_selection_accepted"])
            flipped_evaluation["canonical_selection_accepted"] = True
            (flipped_path / "evaluation.json").write_text(
                json.dumps(flipped_evaluation, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            flipped_payload = (flipped_path / "evaluation.json").read_bytes()
            flipped_manifest = json.loads(
                (flipped_path / "manifest.json").read_text(encoding="utf-8")
            )
            flipped_manifest["canonical_selection_accepted"] = True
            flipped_manifest["artifacts"]["evaluation.json"].update(
                {
                    "size_bytes": len(flipped_payload),
                    "sha256": hashlib.sha256(flipped_payload).hexdigest(),
                }
            )
            flipped_training = json.loads(
                (flipped_path / "training.json").read_text(encoding="utf-8")
            )
            flipped_training.update(
                {
                    "canonical_selection_accepted": True,
                    "evaluation_accepted": False,
                    "status": "canonical-gates-green-runtime-pending",
                }
            )
            flipped_training_payload = (
                json.dumps(flipped_training, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            (flipped_path / "training.json").write_bytes(flipped_training_payload)
            flipped_manifest["artifacts"]["training.json"].update(
                {
                    "size_bytes": len(flipped_training_payload),
                    "sha256": hashlib.sha256(flipped_training_payload).hexdigest(),
                }
            )
            (flipped_path / "manifest.json").write_text(
                json.dumps(flipped_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "artifact|canonical|descriptor"):
                train_hybrid_generator(
                    corpus,
                    root / "flipped-refit",
                    config=refit_config,
                    selection_model=flipped_path,
                    selection_model_manifest_sha256=hashlib.sha256(
                        (flipped_path / "manifest.json").read_bytes()
                    ).hexdigest(),
                )

            # Build one internally consistent, hash-bound green selection fixture.
            green_path = root / "green-selection"
            shutil.copytree(first_path, green_path)
            green_evaluation = json.loads(
                (green_path / "evaluation.json").read_text(encoding="utf-8")
            )
            green_evaluation["canonical_selection_accepted"] = True
            for population in ("train", "source_held_out"):
                metrics = green_evaluation[population]
                metrics.update(
                    {
                        "accepted": True,
                        "joint_geodesic_mae_rad": 0.01,
                        "joint_frame_max_p95_rad": 0.02,
                        "local_position_p95_m": 0.01,
                        "fk_body_position_p95_m": 0.02,
                        "support_foot_position_p95_m": 0.01,
                        "contact_f1": [0.99, 0.99],
                    }
                )
            green_evaluation["selection_tuple"] = [0.01, 0.02, 1.0 - 0.99]
            evaluation_payload = (
                json.dumps(green_evaluation, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            (green_path / "evaluation.json").write_bytes(evaluation_payload)
            green_training = json.loads(
                (green_path / "training.json").read_text(encoding="utf-8")
            )
            green_training.update(
                {
                    "canonical_selection_accepted": True,
                    "evaluation_accepted": False,
                    "status": "canonical-gates-green-runtime-pending",
                }
            )
            training_payload = (
                json.dumps(green_training, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            (green_path / "training.json").write_bytes(training_payload)
            green_manifest = json.loads(
                (green_path / "manifest.json").read_text(encoding="utf-8")
            )
            green_manifest["canonical_selection_accepted"] = True
            green_manifest["artifacts"]["evaluation.json"].update(
                {
                    "size_bytes": len(evaluation_payload),
                    "sha256": hashlib.sha256(evaluation_payload).hexdigest(),
                }
            )
            green_manifest["artifacts"]["training.json"].update(
                {
                    "size_bytes": len(training_payload),
                    "sha256": hashlib.sha256(training_payload).hexdigest(),
                }
            )
            (green_path / "manifest.json").write_text(
                json.dumps(green_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            inconsistent_path = root / "inconsistent-selection"
            shutil.copytree(green_path, inconsistent_path)
            inconsistent_training = json.loads(
                (inconsistent_path / "training.json").read_text(encoding="utf-8")
            )
            inconsistent_training["coverage_rows"] = 1
            inconsistent_payload = (
                json.dumps(inconsistent_training, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            (inconsistent_path / "training.json").write_bytes(inconsistent_payload)
            inconsistent_manifest = json.loads(
                (inconsistent_path / "manifest.json").read_text(encoding="utf-8")
            )
            inconsistent_manifest["artifacts"]["training.json"].update(
                {
                    "size_bytes": len(inconsistent_payload),
                    "sha256": hashlib.sha256(inconsistent_payload).hexdigest(),
                }
            )
            (inconsistent_path / "manifest.json").write_text(
                json.dumps(inconsistent_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "training.*status|coverage/status"):
                train_hybrid_generator(
                    corpus,
                    root / "inconsistent-refit",
                    config=refit_config,
                    selection_model=inconsistent_path,
                    selection_model_manifest_sha256=hashlib.sha256(
                        (inconsistent_path / "manifest.json").read_bytes()
                    ).hexdigest(),
                )
            incomplete_path = root / "incomplete-selection"
            shutil.copytree(green_path, incomplete_path)
            incomplete_evaluation = json.loads(
                (incomplete_path / "evaluation.json").read_text(encoding="utf-8")
            )
            incomplete_evaluation["source_held_out"]["rows"] = 1
            incomplete_payload = (
                json.dumps(incomplete_evaluation, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
            (incomplete_path / "evaluation.json").write_bytes(incomplete_payload)
            incomplete_manifest = json.loads(
                (incomplete_path / "manifest.json").read_text(encoding="utf-8")
            )
            incomplete_manifest["artifacts"]["evaluation.json"].update(
                {
                    "size_bytes": len(incomplete_payload),
                    "sha256": hashlib.sha256(incomplete_payload).hexdigest(),
                }
            )
            (incomplete_path / "manifest.json").write_text(
                json.dumps(incomplete_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "population row counts"):
                train_hybrid_generator(
                    corpus,
                    root / "incomplete-refit",
                    config=refit_config,
                    selection_model=incomplete_path,
                    selection_model_manifest_sha256=hashlib.sha256(
                        (incomplete_path / "manifest.json").read_bytes()
                    ).hexdigest(),
                )
            green_manifest_sha256 = hashlib.sha256(
                (green_path / "manifest.json").read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValueError, "expected selection manifest"):
                train_hybrid_generator(
                    corpus,
                    root / "missing-selection-digest",
                    config=refit_config,
                    selection_model=green_path,
                )
            with self.assertRaisesRegex(ValueError, "selection manifest SHA"):
                train_hybrid_generator(
                    corpus,
                    root / "wrong-selection-digest",
                    config=refit_config,
                    selection_model=green_path,
                    selection_model_manifest_sha256="0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "selection model config"):
                train_hybrid_generator(
                    corpus,
                    root / "wrong-refit-config",
                    config=HybridModelConfig(
                        variant="latent32",
                        seed=23,
                        device="cpu",
                        batch_size=8,
                        post_coverage_steps=0,
                        normalization_chunk_size=7,
                        evaluation_chunk_size=6,
                        fit_all_rows=True,
                    ),
                    selection_model=green_path,
                    selection_model_manifest_sha256=green_manifest_sha256,
                )
            refit_receipt = train_hybrid_generator(
                corpus,
                root / "refit",
                config=refit_config,
                selection_model=green_path,
                selection_model_manifest_sha256=green_manifest_sha256,
            )
            self.assertEqual(refit_receipt["coverage_rows"], 24)
            self.assertEqual(len(refit_receipt["selection_evaluation_sha256"]), 64)
            self.assertEqual(len(refit_receipt["selection_model_manifest_sha256"]), 64)
            refit_manifest = json.loads(
                (root / "refit" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                refit_manifest["selection_evaluation_sha256"],
                refit_receipt["selection_evaluation_sha256"],
            )
            self.assertEqual(
                refit_manifest["selection_model_manifest_sha256"],
                refit_receipt["selection_model_manifest_sha256"],
            )
            load_hybrid_generator(root / "refit", corpus=corpus, device="cpu")
            green_manifest_payload = (green_path / "manifest.json").read_bytes()
            (green_path / "manifest.json").unlink()
            try:
                with self.assertRaisesRegex(ValueError, "selection authority"):
                    load_hybrid_generator(root / "refit", corpus=corpus, device="cpu")
            finally:
                (green_path / "manifest.json").write_bytes(green_manifest_payload)
            refit_manifest.pop("selection_model_manifest_sha256")
            (root / "refit" / "manifest.json").write_text(
                json.dumps(refit_manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "selection provenance"):
                load_hybrid_generator(root / "refit", corpus=corpus, device="cpu")


if __name__ == "__main__":
    unittest.main()
