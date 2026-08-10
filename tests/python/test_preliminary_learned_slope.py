import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mm_sonic.preliminary_learned_slope as preliminary
import numpy as np
from mm_sonic.preliminary_learned_slope import (
    MODEL_ARTIFACT_NAMES,
    PRELIMINARY_LABEL,
    LearnedKeys,
    LearnedSlopeState,
    PreliminaryTrainingConfig,
    _full_route_rollout,
    _publish_rejected_fit,
    build_parser,
    build_preliminary_model_manifest,
    build_viewer_model,
    decoded_pose_to_native_qpos,
    load_preliminary_model,
    load_preliminary_slope_bundle,
    normalized_terrain_features,
    orange_duck_training_config,
    overlay_text,
)

from resources.g1_lmm.dataset import build_training_arrays


class _ConstantNetwork:
    def __init__(self, output: np.ndarray) -> None:
        self.output = np.asarray(output, dtype=np.float32)
        self.calls = 0

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        rows = np.asarray(values, dtype=np.float32)
        self.calls += 1
        return np.broadcast_to(self.output, (len(rows), len(self.output))).copy()


class PreliminaryLearnedSlopeContractTests(unittest.TestCase):
    def test_label_and_artifact_contract_cannot_be_mistaken_for_acceptance(self):
        self.assertEqual(
            PRELIMINARY_LABEL,
            "PRELIMINARY LEARNED EXACT-ROUTE OVERFIT (NOT ACCEPTED/NO GENERALIZATION)",
        )
        self.assertEqual(
            MODEL_ARTIFACT_NAMES,
            ("latent.bin", "decompressor.bin", "stepper.bin"),
        )
        self.assertNotIn("projector.bin", MODEL_ARTIFACT_NAMES)

    def test_training_budget_is_the_one_deterministic_gpu3_fit(self):
        config = PreliminaryTrainingConfig()

        self.assertEqual(config.seed, 1234)
        self.assertEqual(config.device, "cuda:0")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.overfit_steps, 1000)
        self.assertEqual(config.decompressor_steps, 100_000)
        self.assertEqual(config.stepper_steps, 100_000)
        self.assertEqual(config.stepper_window, 20)
        self.assertEqual(config.learning_rate, 1.0e-3)
        self.assertFalse(hasattr(config, "projector_steps"))

        orange_duck = orange_duck_training_config(config)
        self.assertTrue(orange_duck.single_clip_overfit_canary)
        self.assertEqual(orange_duck.seed, 1234)
        self.assertEqual(orange_duck.decompressor_steps, 100_000)
        self.assertEqual(orange_duck.stepper_steps, 100_000)
        self.assertFalse(hasattr(orange_duck, "projector_steps"))

    def test_published_manifest_is_explicitly_preliminary_despite_green_gates(self):
        artifacts = {
            name: {"path": name, "size_bytes": index + 1, "sha256": "0" * 64}
            for index, name in enumerate(MODEL_ARTIFACT_NAMES)
        }

        self.assertIn(
            "training_receipt",
            build_preliminary_model_manifest.__code__.co_varnames,
        )
        training_receipt = {
            "path": "training.json",
            "size_bytes": 123,
            "sha256": "2" * 64,
        }
        manifest = build_preliminary_model_manifest(
            data_manifest_sha256="1" * 64,
            artifacts=artifacts,
            training_receipt=training_receipt,
            numerical_gates_passed=True,
        )

        self.assertEqual(manifest["status"], "preliminary-not-accepted")
        self.assertIs(manifest["accepted"], False)
        self.assertEqual(manifest["generalization_claim"], "none")
        self.assertEqual(manifest["evaluation_scope"], "exact-route-all-row-overfit")
        self.assertEqual(manifest["projector"], "absent")
        self.assertEqual(tuple(manifest["artifacts"]), MODEL_ARTIFACT_NAMES)
        self.assertEqual(manifest["training_receipt"], training_receipt)

    def test_cli_has_separate_train_smoke_and_view_commands(self):
        parser = build_parser()

        train = parser.parse_args(("train", "--output", "/tmp/model"))
        smoke = parser.parse_args(("smoke", "--model", "/tmp/model"))
        view = parser.parse_args(("view", "--model", "/tmp/model"))

        self.assertEqual(train.command, "train")
        self.assertEqual(train.output, Path("/tmp/model"))
        self.assertEqual(smoke.command, "smoke")
        self.assertEqual(view.command, "view")

    def test_viewer_key_levels_make_release_an_actual_pause(self):
        keys = LearnedKeys()

        keys.press("w")
        self.assertEqual(keys.snapshot(), (True, False))
        keys.release("w")
        self.assertEqual(keys.snapshot(), (False, False))
        keys.press("x")
        self.assertEqual(keys.snapshot(), (False, True))

    def test_nonfinite_failure_metrics_still_publish_an_immutable_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            output = root / "model-v1"
            staging.mkdir()
            (staging / "latent.bin").write_bytes(b"must be removed")
            receipt = {
                "metrics": {
                    "nan": float("nan"),
                    "positive_inf": float("inf"),
                    "negative_inf": -float("inf"),
                }
            }

            _publish_rejected_fit(
                staging,
                output,
                receipt,
                stopped_after="decompressor",
                error=ValueError("nonfinite gate"),
            )

            published = json.loads((output / "training.json").read_text())
            self.assertEqual(published["status"], "rejected")
            self.assertEqual(
                published["metrics"],
                {"nan": None, "negative_inf": None, "positive_inf": None},
            )
            self.assertFalse((output / "latent.bin").exists())

    def test_sole_fit_uses_one_fixed_output_and_an_atomic_persistent_claim(self):
        self.assertTrue(hasattr(preliminary, "_acquire_atomic_fit_claim"))
        self.assertEqual(
            preliminary.DEFAULT_MODEL_OUTPUT.name,
            "model-v1",
        )
        with tempfile.TemporaryDirectory() as temporary:
            claim = Path(temporary) / ".sole-fit.claim"

            descriptor = preliminary._acquire_atomic_fit_claim(
                claim,
                Path(temporary) / "model-v1",
            )

            self.assertTrue(claim.is_file())
            self.assertEqual(descriptor["path"], str(claim))
            with self.assertRaisesRegex(FileExistsError, "sole.*claimed"):
                preliminary._acquire_atomic_fit_claim(
                    claim,
                    Path(temporary) / "model-v1",
                )
            with self.assertRaisesRegex(ValueError, "output is fixed"):
                preliminary.train_preliminary_model(
                    Path(temporary) / "different-output"
                )


class ExactAuthoredSlopeBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_preliminary_slope_bundle()
        cls.arrays = build_training_arrays(cls.bundle.training)

    def test_uses_all_595_exact_task1_rows_at_60hz(self):
        training = self.bundle.training
        self.assertEqual(self.bundle.clip_id, "terrain_slopes__slope_000__000")
        self.assertEqual(self.bundle.fps, 60.0)
        self.assertEqual(training.frames, 595)
        np.testing.assert_array_equal(training.range_starts, [0])
        np.testing.assert_array_equal(training.range_stops, [595])
        self.assertTrue(np.all(training.admitted_mask))
        self.assertEqual(self.bundle.reference_qpos.shape, (595, 36))
        self.assertEqual(
            hashlib.sha256(
                self.bundle.reference_qpos.astype("<f4").tobytes()
            ).hexdigest(),
            "b2abecf5e0423708b6b9330347aab051502158a965e9fb0fa2d7873b9d8ee4be",
        )

    def test_all_four_terrain_dimensions_are_active_and_nonflat(self):
        training = self.bundle.training
        self.assertTrue(np.all(np.isfinite(training.feature_scale[27:31])))
        self.assertTrue(np.all(training.feature_scale[27:31] > 0.0))
        self.assertTrue(
            np.all(training.feature_scale[27:31] < np.finfo(np.float32).max)
        )
        raw = (
            training.features[:, 27:31] * training.feature_scale[27:31]
            + training.feature_offset[27:31]
        )
        self.assertTrue(np.all(np.any(raw[165:539] != 0.0, axis=1)))
        self.assertTrue(np.all(np.std(raw.astype(np.float64), axis=0) > 1.0e-6))

    def test_runtime_resample_matches_authored_terrain_channels(self):
        for row in (0, 165, 274, 538, 594):
            np.testing.assert_allclose(
                normalized_terrain_features(self.bundle, row),
                self.bundle.training.features[row, 27:31],
                rtol=0.0,
                atol=2.0e-7,
            )

    def test_exact_decompressor_targets_round_trip_to_native_g1_qpos(self):
        for row in (0, 165, 274, 538, 594):
            predicted = decoded_pose_to_native_qpos(
                self.arrays.decompressor_target[row],
                self.bundle.training.positions[row, 0],
                self.bundle.training.rotations[row, 0],
                self.bundle.native_model,
            )
            expected = self.bundle.reference_qpos[row]
            self.assertLess(float(np.linalg.norm(predicted[:3] - expected[:3])), 2e-7)
            quaternion_dot = float(abs(np.dot(predicted[3:7], expected[3:7])))
            self.assertLess(
                2.0 * float(np.arccos(np.clip(quaternion_dot, 0.0, 1.0))),
                1.1e-3,
            )
            joint_error = (predicted[7:] - expected[7:] + np.pi) % (2.0 * np.pi) - np.pi
            self.assertLess(float(np.max(np.abs(joint_error))), 5e-5)

    def test_viewer_model_contains_native_g1_exact_ramp_and_flat_extension(self):
        import mujoco

        model = build_viewer_model(self.bundle)

        self.assertEqual(model.nq, 36)
        for name in (
            "preliminary_learned_slope_exact_mesh",
            "preliminary_learned_slope_exterior_flat",
        ):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), 0
            )

    def test_overlay_always_names_learned_preliminary_scope(self):
        title, body = overlay_text(row=12, frame_count=595, w_down=True)

        self.assertEqual(title, PRELIMINARY_LABEL)
        self.assertIn("LEARNED recurrent step", body)
        self.assertIn("terrain overwrite", body)
        self.assertIn("NO PROJECTOR", body)

    def _write_model_fixture(self, root: Path) -> None:
        from resources.g1_lmm.models import Decompressor, Stepper
        from resources.g1_lmm.training import export_network

        dimensions = self.bundle.training.dimensions
        latent = np.zeros((595, 32), dtype="<f4")
        (root / "latent.bin").write_bytes(
            struct.pack("<II", *latent.shape) + latent.tobytes()
        )
        export_network(
            root / "decompressor.bin",
            Decompressor(dimensions).layers,
            input_mean=np.zeros(63, np.float32),
            input_std=np.ones(63, np.float32),
            output_mean=np.zeros(458, np.float32),
            output_std=np.ones(458, np.float32),
        )
        export_network(
            root / "stepper.bin",
            Stepper(dimensions).layers,
            input_mean=np.zeros(63, np.float32),
            input_std=np.ones(63, np.float32),
            output_mean=np.zeros(63, np.float32),
            output_std=np.ones(63, np.float32),
        )
        artifacts = {}
        for name in MODEL_ARTIFACT_NAMES:
            payload = (root / name).read_bytes()
            artifacts[name] = {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        training_receipt = {
            "schema": "g1-lmm-preliminary-slope-training/v1",
            "label": PRELIMINARY_LABEL,
            "status": "preliminary-not-accepted",
            "accepted": False,
            "generalization_claim": "none",
            "evaluation_scope": "exact-route-all-row-overfit",
            "data_manifest_sha256": self.bundle.training.manifest_sha256,
            "source_hashes": dict(self.bundle.hashes),
            "sole_fit_claim": {
                "path": str(preliminary.SOLE_FIT_CLAIM),
                "size_bytes": 321,
                "sha256": "3" * 64,
            },
            "numerical_gates_passed": True,
            "stopped_after": "full-route-rollout",
            "hardware": {
                "physical_index": 3,
                "logical_device": "cuda:0",
                "uuid": preliminary.GPU3_UUID,
                "name": "NVIDIA L40S",
                "deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
            },
            "config": {
                "seed": 1234,
                "stage_seeds": {
                    "overfit": 1234,
                    "decompressor": 1235,
                    "stepper": 1236,
                },
                "device": "cuda:0",
                "batch_size": 32,
                "learning_rate": 1.0e-3,
                "overfit_steps": 1_000,
                "decompressor_steps": 100_000,
                "stepper_steps": 100_000,
                "stepper_window": 20,
                "dt": 1.0 / 60.0,
                "projector": "absent",
            },
            "architecture": {
                "compressor": [908, 512, 512, 512, 32],
                "decompressor": [63, 512, 458],
                "stepper": [63, 512, 512, 63],
                "projector": None,
            },
            "overfit_gate": {"accepted": True},
            "decompressor_gate": {"accepted": True},
            "stepper_gate": {"accepted": True},
            "full_route_rollout_gate": {
                "accepted": True,
                "evaluated_pose_rows": 595,
                "joint_mae_rad": 0.01,
                "maximum_planar_root_error_m": 0.02,
                "contact_f1": [0.95, 0.96],
                "projector_evaluations": 0,
            },
            "artifacts": artifacts,
        }
        training_payload = (
            json.dumps(training_receipt, sort_keys=True, indent=2) + "\n"
        ).encode()
        (root / "training.json").write_bytes(training_payload)
        training_descriptor = {
            "path": "training.json",
            "size_bytes": len(training_payload),
            "sha256": hashlib.sha256(training_payload).hexdigest(),
        }
        manifest = {
            "schema": "g1-lmm-preliminary-slope-model/v1",
            "label": PRELIMINARY_LABEL,
            "status": "preliminary-not-accepted",
            "accepted": False,
            "generalization_claim": "none",
            "projector": "absent",
            "rows": 595,
            "output_fps": 60.0,
            "data_manifest_sha256": self.bundle.training.manifest_sha256,
            "numerical_gates_passed": True,
            "training_receipt": training_descriptor,
            "artifacts": artifacts,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    def test_model_loader_authenticates_only_latent_decompressor_and_stepper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_model_fixture(root)

            model = load_preliminary_model(root, self.bundle)

            self.assertEqual(model.latent.shape, (595, 32))
            self.assertEqual(tuple(model.manifest["artifacts"]), MODEL_ARTIFACT_NAMES)
            self.assertFalse(model.manifest["accepted"])
            self.assertEqual(model.manifest["generalization_claim"], "none")
            self.assertEqual(model.training_receipt["config"]["seed"], 1234)

    def test_model_loader_requires_and_validates_bound_training_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_model_fixture(root)
            (root / "training.json").unlink()

            with self.assertRaisesRegex(ValueError, "training receipt"):
                load_preliminary_model(root, self.bundle)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_model_fixture(root)
            receipt_path = root / "training.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["config"]["seed"] = 999
            payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
            receipt_path.write_bytes(payload)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["training_receipt"] = {
                "path": "training.json",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))

            with self.assertRaisesRegex(ValueError, "training receipt"):
                load_preliminary_model(root, self.bundle)

    def test_model_loader_rejects_any_projector_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_model_fixture(root)
            (root / "projector.bin").write_bytes(b"forbidden")

            with self.assertRaisesRegex(ValueError, "projector"):
                load_preliminary_model(root, self.bundle)

    def test_release_is_bitwise_pause_and_w_advances_one_learned_state(self):
        latent = np.zeros((595, 32), dtype=np.float32)
        decompressor = _ConstantNetwork(self.arrays.decompressor_target[0])
        derivative = np.zeros(63, dtype=np.float32)
        derivative[0] = 0.25
        stepper = _ConstantNetwork(derivative)
        state = LearnedSlopeState(
            self.bundle,
            latent=latent,
            decompressor=decompressor,
            stepper=stepper,
        )

        before = state.fingerprint()
        self.assertEqual(state.tick(w_down=False), 0)
        self.assertEqual(state.fingerprint(), before)
        self.assertEqual(stepper.calls, 0)

        self.assertEqual(state.tick(w_down=True), 1)
        self.assertEqual(stepper.calls, 1)
        self.assertEqual(state.terrain_overwrite_count, 1)
        np.testing.assert_allclose(
            state.recurrent_state[27:31],
            self.bundle.training.features[1, 27:31],
            rtol=0.0,
            atol=2.0e-7,
        )

        held = state.fingerprint()
        for _ in range(8):
            self.assertEqual(state.tick(w_down=False), 1)
        self.assertEqual(state.fingerprint(), held)
        self.assertEqual(stepper.calls, 1)

    def test_full_route_gate_rejects_smooth_finite_non_route_output(self):
        latent = np.zeros((595, 32), dtype=np.float32)
        decoded = self.arrays.decompressor_target[0].copy()
        decoded[450:456] = 0.0
        model = type(
            "ConstantPreliminaryModel",
            (),
            {
                "latent": latent,
                "decompressor": _ConstantNetwork(decoded),
                "stepper": _ConstantNetwork(np.zeros(63, dtype=np.float32)),
            },
        )()

        receipt = _full_route_rollout(self.bundle, model)

        self.assertFalse(receipt["accepted"])
        self.assertGreater(receipt["maximum_planar_root_error_m"], 0.10)
        self.assertEqual(receipt["evaluated_pose_rows"], 595)
        self.assertEqual(len(receipt["contact_f1"]), 2)
        self.assertNotIn("maximum_planar_root_error_m_report_only", receipt)
        self.assertNotIn("joint_mae_rad_report_only", receipt)

    def test_viewer_reruns_full_route_gate_and_refuses_bad_model(self):
        latent = np.zeros((595, 32), dtype=np.float32)
        decoded = self.arrays.decompressor_target[0].copy()
        decoded[450:456] = 0.0
        model = type(
            "ConstantPreliminaryModel",
            (),
            {
                "latent": latent,
                "decompressor": _ConstantNetwork(decoded),
                "stepper": _ConstantNetwork(np.zeros(63, dtype=np.float32)),
            },
        )()

        with (
            patch.dict("os.environ", {"DISPLAY": ""}),
            self.assertRaisesRegex(
                preliminary.PreliminaryTrainingError, "full-route rollout"
            ),
        ):
            preliminary.run_interactive(self.bundle, model)


if __name__ == "__main__":
    unittest.main()
