import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from resources.g1_lmm.dataset import (
    G1LmmDimensions,
    build_training_arrays,
    load_training_bundle,
    range_safe_deltas,
    range_safe_windows,
)
from resources.g1_lmm.models import Compressor, Decompressor, Projector, Stepper
from resources.g1_lmm.training import (
    TrainingConfig,
    TrainingGateError,
    deterministic_withheld_ranges,
    evaluate_exported_networks,
    export_network,
    load_exported_network,
    normalized_projector_targets,
    train_flat_bundle,
    train_tiny_fixture,
)
from resources.train_g1_lmm import build_parser


def _array2_bytes(values, dtype):
    values = np.ascontiguousarray(values, dtype=dtype)
    return struct.pack("<II", *values.shape[:2]) + values.tobytes(order="C")


def _array1_bytes(values, dtype):
    values = np.ascontiguousarray(values, dtype=dtype)
    return struct.pack("<I", len(values)) + values.tobytes(order="C")


def _write_flat_training_bundle(
    path: Path,
    *,
    local_rotation_step_rad: float = 0.0,
) -> dict:
    path.mkdir()
    frames = 3853
    bones = 31
    positions = np.zeros((frames, bones, 3), dtype=np.float32)
    positions[:, 1:, 1] = np.arange(1, bones, dtype=np.float32)[None] * 0.01
    positions[:, 0, 0] = np.arange(frames, dtype=np.float32) / 60.0
    velocities = np.zeros_like(positions)
    velocities[:, 0, 0] = 1.0
    rotations = np.zeros((frames, bones, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    if local_rotation_step_rad:
        rotations[5:, 1, 0] = np.cos(local_rotation_step_rad / 2.0)
        rotations[5:, 1, 1] = np.sin(local_rotation_step_rad / 2.0)
    angular_velocities = np.zeros_like(positions)
    parents = np.concatenate(([-1], np.arange(0, bones - 1))).astype(np.int32)
    range_lengths = np.array([747] + [259] * 10 + [258] * 2, dtype=np.int32)
    range_stops = np.cumsum(range_lengths, dtype=np.int32)
    range_starts = np.concatenate((np.array([0], np.int32), range_stops[:-1]))
    contacts = np.stack(
        ((np.arange(frames) // 5) % 2, (np.arange(frames) // 5 + 1) % 2), axis=1
    ).astype(np.uint8)
    database_payload = b"".join(
        (
            _array2_bytes(positions, "<f4"),
            _array2_bytes(velocities, "<f4"),
            _array2_bytes(rotations, "<f4"),
            _array2_bytes(angular_velocities, "<f4"),
            _array1_bytes(parents, "<i4"),
            _array1_bytes(range_starts, "<i4"),
            _array1_bytes(range_stops, "<i4"),
            _array2_bytes(contacts, "u1"),
        )
    )
    (path / "database.bin").write_bytes(database_payload)

    feature_values = np.zeros((frames, 31), dtype=np.float32)
    feature_values[:, 0] = np.linspace(-1.0, 1.0, frames, dtype=np.float32)
    feature_offset = np.zeros(31, dtype=np.float32)
    feature_scale = np.ones(31, dtype=np.float32)
    feature_payload = b"".join(
        (
            _array2_bytes(feature_values, "<f4"),
            _array1_bytes(feature_offset, "<f4"),
            _array1_bytes(feature_scale, "<f4"),
        )
    )
    (path / "features.bin").write_bytes(feature_payload)
    artifacts = {
        name: {
            "path": name,
            "size_bytes": (path / name).stat().st_size,
            "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
        }
        for name in ("database.bin", "features.bin")
    }
    manifest = {
        "schema": "g1-lmm-flat-data/v2",
        "status": "accepted",
        "output_fps": 60.0,
        "trajectory_horizons": [20, 40, 60],
        "feature_dimensions": 31,
        "database_frames": frames,
        "total_clips": 1,
        "dimensions": {"bones": 31, "features": 31, "contacts": 2},
        "ranges": [
            {
                "start": int(start),
                "stop": int(stop),
                "source_first_frame": int(start),
                "source_last_frame": int(stop - 1),
                "motion_class": "flat-walk",
                "terrain_class": "flat",
            }
            for start, stop in zip(range_starts, range_stops)
        ],
        "continuity": {
            "schema": "g1-lmm-continuity/v1",
            "threshold_rad_per_frame": 0.25,
            "minimum_range_frames": 61,
            "source_native_rejected_edge_count": 31,
            "database_local_rejected_edge_count": 32,
            "union_rejected_edge_count": 32,
            "dropped_fragment_count": 20,
            "dropped_frame_count": 233,
            "published_range_count": 13,
            "published_frame_count": 3853,
            "maximum_admitted_native_step_rad": local_rotation_step_rad,
            "maximum_admitted_local_rotation_step_rad": local_rotation_step_rad,
            "range_digest_sha256": hashlib.sha256(
                np.asarray(
                    [
                        [int(start), int(stop), int(start), int(stop - 1)]
                        for start, stop in zip(range_starts, range_stops)
                    ],
                    dtype="<i4",
                ).tobytes(order="C")
            ).hexdigest(),
        },
        "artifacts": artifacts,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest


class G1LmmTrainingTest(unittest.TestCase):
    def test_loader_requires_v2_continuity_receipt_and_recomputes_local_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            manifest = _write_flat_training_bundle(legacy)
            manifest["schema"] = "g1-lmm-flat-data/v1"
            (legacy / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "v2"):
                load_training_bundle(legacy)

            unsafe = root / "unsafe"
            _write_flat_training_bundle(unsafe, local_rotation_step_rad=0.5)
            with self.assertRaisesRegex(ValueError, "0.25"):
                load_training_bundle(unsafe)

            mismatch = root / "mismatch"
            manifest = _write_flat_training_bundle(mismatch)
            manifest["continuity"]["maximum_admitted_local_rotation_step_rad"] = 0.1
            (mismatch / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "recomputed"):
                load_training_bundle(mismatch)

    def test_all_stage_stops_after_failed_decompressor_gate_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "rejected"
            _write_flat_training_bundle(data_directory)
            config = TrainingConfig(
                seed=1234,
                device="cpu",
                batch_size=8,
                overfit_steps=24,
                decompressor_steps=1,
                stepper_steps=1,
                projector_steps=1,
                withheld_frames=16,
                withheld_halo=2,
            )
            with self.assertRaisesRegex(TrainingGateError, "decompressor"):
                train_flat_bundle(
                    data_directory,
                    output_directory,
                    config=config,
                    stage="all",
                )
            self.assertTrue((output_directory / "latent.bin").is_file())
            self.assertTrue((output_directory / "decompressor.bin").is_file())
            self.assertFalse((output_directory / "stepper.bin").exists())
            self.assertFalse((output_directory / "projector.bin").exists())
            self.assertFalse((output_directory / "manifest.json").exists())
            training = json.loads((output_directory / "training.json").read_text(encoding="utf-8"))
            self.assertFalse(training["accepted"])
            self.assertEqual(training["stopped_after"], "decompressor")

    def test_real_bundle_overfit_stage_gates_64_adjacent_rows_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "overfit"
            _write_flat_training_bundle(data_directory)
            receipt = train_flat_bundle(
                data_directory,
                output_directory,
                config=TrainingConfig(
                    seed=1234,
                    device="cpu",
                    batch_size=32,
                    overfit_steps=24,
                    decompressor_steps=1,
                    stepper_steps=1,
                    projector_steps=1,
                ),
                stage="overfit",
            )
            self.assertTrue(receipt["accepted"])
            self.assertEqual(receipt["overfit_gate"]["frames"], 64)
            self.assertEqual(receipt["overfit_gate"]["range"], [0, 64])
            self.assertLess(
                receipt["overfit_gate"]["final_loss"],
                0.1 * receipt["overfit_gate"]["initial_loss"],
            )
            self.assertTrue((output_directory / "training.json").is_file())
            self.assertTrue((output_directory / "evaluation.json").is_file())
            self.assertFalse((output_directory / "manifest.json").exists())

    def test_cli_exposes_device_stage_seed_and_configurable_budgets(self):
        arguments = build_parser().parse_args(
            [
                "data",
                "model",
                "--device",
                "cuda:1",
                "--stage",
                "all",
                "--seed",
                "77",
                "--overfit-steps",
                "101",
                "--decompressor-steps",
                "202",
                "--stepper-steps",
                "303",
                "--projector-steps",
                "404",
            ]
        )
        self.assertEqual(arguments.data_directory, Path("data"))
        self.assertEqual(arguments.output_directory, Path("model"))
        self.assertEqual(arguments.device, "cuda:1")
        self.assertEqual(arguments.stage, "all")
        self.assertEqual(
            (
                arguments.seed,
                arguments.overfit_steps,
                arguments.decompressor_steps,
                arguments.stepper_steps,
                arguments.projector_steps,
            ),
            (77, 101, 202, 303, 404),
        )

    def test_deterministic_withheld_blocks_have_full_feature_and_window_halo(self):
        starts, stops = deterministic_withheld_ranges(
            np.array([0, 1000], np.int32),
            np.array([800, 1800], np.int32),
            frames=64,
            seed=1234,
        )
        self.assertEqual(starts.shape, stops.shape)
        np.testing.assert_array_equal(stops - starts, np.array([64, 64], np.int64))
        fitted = range_safe_windows(
            np.array([0, 1000], np.int32),
            np.array([800, 1800], np.int32),
            20,
            excluded_starts=starts,
            excluded_stops=stops,
            exclusion_halo=TrainingConfig().withheld_halo,
        )
        for window in fitted:
            for start, stop in zip(starts, stops):
                self.assertFalse(
                    window[0] < stop + TrainingConfig().withheld_halo
                    and start - TrainingConfig().withheld_halo <= window[-1]
                )

    def test_accepted_flat_bundle_builds_exact_legacy_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory) / "data"
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)

            self.assertEqual(bundle.dimensions, G1LmmDimensions())
            self.assertEqual(bundle.frames, 3853)
            self.assertEqual(arrays.compressor_input.shape, (3853, 908))
            self.assertEqual(arrays.decompressor_target.shape, (3853, 458))
            np.testing.assert_array_equal(arrays.compressor_input[:, :90], bundle.positions[:, 1:].reshape(3853, -1))
            np.testing.assert_array_equal(arrays.decompressor_target[:, :90], bundle.positions[:, 1:].reshape(3853, -1))
            np.testing.assert_array_equal(arrays.decompressor_target[:, -2:], bundle.contacts.astype(np.float32))
            self.assertTrue(np.isfinite(arrays.compressor_input).all())
            self.assertTrue(np.isfinite(arrays.decompressor_target).all())

            with (data_directory / "features.bin").open("r+b") as stream:
                stream.seek(12)
                original = stream.read(1)
                stream.seek(12)
                stream.write(bytes((original[0] ^ 1,)))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_training_bundle(data_directory)

    def test_flat_dimensions_and_exact_orange_duck_architectures(self):
        dims = G1LmmDimensions(features=31, latent=32, bones=31, contacts=2)
        self.assertEqual(dims.state, 63)
        self.assertEqual(dims.compressor_input, 908)
        self.assertEqual(dims.decompressor_output, 458)

        compressor = Compressor(dims)
        decompressor = Decompressor(dims)
        stepper = Stepper(dims)
        projector = Projector(dims)

        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in compressor.layers],
            [(908, 512), (512, 512), (512, 512), (512, 32)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in decompressor.layers],
            [(63, 512), (512, 458)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in stepper.layers],
            [(63, 512), (512, 512), (512, 63)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in projector.layers],
            [(31, 512), (512, 512), (512, 512), (512, 512), (512, 63)],
        )

        with torch.no_grad():
            self.assertEqual(tuple(compressor(torch.zeros(2, 3, 908)).shape), (2, 3, 32))
            self.assertEqual(tuple(decompressor(torch.zeros(2, 3, 63)).shape), (2, 3, 458))
            self.assertEqual(tuple(stepper(torch.zeros(2, 3, 63)).shape), (2, 3, 63))
            self.assertEqual(tuple(projector(torch.zeros(2, 3, 31)).shape), (2, 3, 63))

    def test_windows_derivatives_and_withheld_halos_are_range_safe(self):
        starts = np.array([0, 100], dtype=np.int32)
        stops = np.array([80, 180], dtype=np.int32)
        windows = range_safe_windows(starts, stops, 20)
        self.assertEqual(windows.shape, (122, 20))
        self.assertFalse(any(window[0] < 80 <= window[-1] for window in windows))
        self.assertFalse(any(window[0] < 100 <= window[-1] for window in windows))

        fitted = range_safe_windows(
            starts,
            stops,
            20,
            excluded_starts=np.array([30], dtype=np.int32),
            excluded_stops=np.array([40], dtype=np.int32),
            exclusion_halo=2,
        )
        self.assertTrue(all(not np.any((window >= 28) & (window < 42)) for window in fitted))

        rows = np.arange(180, dtype=np.float32)[:, None]
        derivatives = range_safe_deltas(rows, starts, stops, dt=1.0 / 60.0)
        self.assertEqual(derivatives.shape, (158, 1))
        np.testing.assert_array_equal(derivatives, np.full((158, 1), 60.0, np.float32))

    def test_projector_targets_and_costs_stay_in_normalized_space(self):
        features_normalized = np.array(
            [[-1.0, 0.0], [0.25, 0.5], [2.0, -0.5]], dtype=np.float32
        )
        latent = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        query_normalized = np.array([[0.20, 0.45], [1.8, -0.6]], dtype=np.float32)

        result = normalized_projector_targets(query_normalized, features_normalized, latent)
        np.testing.assert_array_equal(result.indices, np.array([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(result.features, features_normalized[[1, 2]])
        np.testing.assert_array_equal(result.latent, latent[[1, 2]])
        expected = np.sqrt(np.sum((query_normalized - features_normalized[[1, 2]]) ** 2, axis=1))
        np.testing.assert_allclose(result.distance, expected, rtol=0.0, atol=0.0)

    def test_network_binary_is_little_endian_transposed_and_fail_closed(self):
        layer = torch.nn.Linear(2, 3)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32))
            layer.bias.copy_(torch.tensor([7, 8, 9], dtype=torch.float32))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.bin"
            export_network(
                path,
                [layer],
                input_mean=np.array([10, 11], dtype=np.float32),
                input_std=np.array([12, 13], dtype=np.float32),
                output_mean=np.array([14, 15, 16], dtype=np.float32),
                output_std=np.array([17, 18, 19], dtype=np.float32),
            )
            payload = path.read_bytes()

            cursor = 0
            vectors = []
            for expected_size in (2, 2, 3, 3):
                size = struct.unpack_from("<I", payload, cursor)[0]
                cursor += 4
                self.assertEqual(size, expected_size)
                vectors.append(np.frombuffer(payload, dtype="<f4", count=size, offset=cursor).copy())
                cursor += size * 4
            self.assertEqual(struct.unpack_from("<I", payload, cursor)[0], 1)
            cursor += 4
            self.assertEqual(struct.unpack_from("<II", payload, cursor), (2, 3))
            cursor += 8
            np.testing.assert_array_equal(
                np.frombuffer(payload, dtype="<f4", count=6, offset=cursor),
                np.array([1, 3, 5, 2, 4, 6], dtype=np.float32),
            )

            loaded = load_exported_network(path, expected_layers=[(2, 3)])
            np.testing.assert_array_equal(loaded.input_mean, vectors[0])
            np.testing.assert_array_equal(loaded.layers[0].weight, layer.weight.detach().numpy())
            np.testing.assert_array_equal(loaded.layers[0].bias, layer.bias.detach().numpy())

            path.write_bytes(payload + b"trailing")
            with self.assertRaisesRegex(ValueError, "trailing"):
                load_exported_network(path, expected_layers=[(2, 3)])

    def test_tiny_training_overfits_64_frames_and_is_bitwise_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = train_tiny_fixture(root / "first", seed=1234, device="cpu")
            second = train_tiny_fixture(root / "second", seed=1234, device="cpu")

            self.assertTrue(first["finite"])
            self.assertTrue(first["overfit_gate"]["accepted"])
            self.assertEqual(first["overfit_gate"]["frames"], 64)
            self.assertLess(first["overfit_gate"]["final_loss"], first["overfit_gate"]["initial_loss"])
            self.assertLess(
                first["overfit_gate"]["final_loss"],
                0.1 * first["overfit_gate"]["initial_loss"],
            )
            self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])
            self.assertEqual(evaluate_exported_networks(root / "first"), first["reload_metrics"])

            expected_files = {
                "latent.bin",
                "decompressor.bin",
                "stepper.bin",
                "projector.bin",
                "training.json",
                "evaluation.json",
                "manifest.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in (root / "first").iterdir()}))
            manifest = json.loads((root / "first" / "manifest.json").read_text(encoding="utf-8"))
            for name in ("latent.bin", "decompressor.bin", "stepper.bin", "projector.bin"):
                self.assertEqual(
                    manifest["artifacts"][name]["sha256"],
                    hashlib.sha256((root / "first" / name).read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
