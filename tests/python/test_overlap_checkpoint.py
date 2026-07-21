import base64
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - dependency gate
    torch = None

from resources.g1_interaction_builder.schema import G1_SKELETON


_CANONICAL_OFFSETS = np.frombuffer(base64.b64decode(
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACwJFBU0r08/4O9AGACpKmR+bzJ/VS9vc3MPDgx/r0AAMSkdE2gvd6XNb6P1Ay7AID9ozqamb7GEMY4AAAjJKrVj7wAwA4kAHi4JFFU0r1E/4M9ELQfLoCR+byr/VQ9eM3MPJgx/r0AgMEkSk2gvQCYNb6P1Aw7AABApNiZmb7GEMa4AIDwI4bVj7wAQJ2iAAAAAAAAAAAAAAAAP+CBux1cDz0AiAGkbQC3rNmlmzwAbImk2qOBO5F8cz4IQM29ALgMpd+aYrzGpRu9AACno+ZZ072Cd8y7WkuBPCvmpL2qSIwzXsvMPQrXI7y/c/e6tKQbPQAAQKTynn4zCGk8PQAAkKNm4gQz2aOBO418cz7EOs09AKDfpDCbYrzEpRs9AMAfJQxa0711eMw7WkuBPGDmpL0AEF+lTMzMPQrXI7y/c/c6fqUbPQAAJqVzE5myvmk8PQAAAKSIcomy"
), dtype="<f4").reshape(31, 3).copy()


@unittest.skipIf(torch is None, "torch is required for overlap checkpoint tests")
class OverlapCheckpointTests(unittest.TestCase):
    def _motion(self, rows: int, *, contact: bool) -> np.ndarray:
        frames = np.zeros((rows, 50, 195), np.float32)
        frames[..., 6:192] = 0.0
        rotation = frames[..., 6:192].reshape(rows, 50, 31, 6)
        rotation[..., 0] = 1.0
        rotation[..., 4] = 1.0
        frames[..., 0] = np.linspace(-0.2, 0.0, 50, dtype=np.float32)
        frames[..., 3] = 0.02
        if contact:
            frames[:, 25:, 194] = 1.0
        return frames

    def _write_dataset(self, path: Path) -> str:
        train_walk = self._motion(2, contact=False)
        train_pickup = self._motion(2, contact=True)
        validation_walk = self._motion(1, contact=False)
        validation_pickup = self._motion(1, contact=True)
        static = np.zeros((2, 25), np.float32)
        static[:, 1] = 1.0
        static[:, 5] = 1.0
        static[:, 9] = 1.0
        static[:, 13:16] = 0.1
        static[:, 24] = 1.0
        values = {
            "train_walk_windows": train_walk,
            "train_pickup_windows": train_pickup,
            "train_static_conditions": static,
            "train_walk_temporal": np.zeros((2, 50, 9), np.float32),
            "train_pickup_temporal": np.zeros((2, 50, 9), np.float32),
            "train_walking_windows": train_walk.copy(),
            "train_walking_static_conditions": np.zeros((2, 25), np.float32),
            "train_walking_temporal": np.zeros((2, 50, 9), np.float32),
            "validation_walk_windows": validation_walk,
            "validation_pickup_windows": validation_pickup,
            "validation_static_conditions": static[:1],
            "validation_walk_temporal": np.zeros((1, 50, 9), np.float32),
            "validation_pickup_temporal": np.zeros((1, 50, 9), np.float32),
            "validation_walking_windows": validation_walk.copy(),
            "validation_walking_static_conditions": np.zeros((1, 25), np.float32),
            "validation_walking_temporal": np.zeros((1, 50, 9), np.float32),
            "test_walk_windows": np.full((1, 50, 195), 1.0e30, np.float32),
            "normalization_mean": np.zeros(195, np.float32),
            "normalization_scale": np.ones(195, np.float32),
            "skeleton_parents": G1_SKELETON.parents,
            "skeleton_names": np.asarray(G1_SKELETON.names),
            "skeleton_signature": np.asarray(G1_SKELETON.signature()),
            "canonical_local_offsets": _CANONICAL_OFFSETS,
        }
        np.savez_compressed(path, **values)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_one_step_staged_training_writes_and_strictly_loads_schema_v1(self):
        from resources.g1_interaction_builder.overlap_diffusion import load_overlap_checkpoint
        from tools.train_g1_overlap_diffusion import train_overlap

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, checkpoint = root / "dataset.npz", root / "checkpoint.pt"
            digest = self._write_dataset(dataset)
            result = train_overlap(
                dataset, checkpoint, seed=17, device="cpu", epochs_a=1, epochs_b=1,
                epochs_c=1, batch_size=2, validation_subset=1, model_width=16,
                model_blocks=1, model_heads=1,
            )
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(checkpoint.with_name("last.pt").is_file())
            self.assertTrue(checkpoint.with_name("best.pt").is_file())
            self.assertTrue(checkpoint.with_suffix(".validation.json").is_file())
            loaded = load_overlap_checkpoint(checkpoint, device="cpu")
            payload = loaded.payload
            self.assertEqual(payload["schema_version"], "overlap-diffusion-schema-v1")
            self.assertEqual(payload["frame_dim"], 195)
            self.assertEqual(tuple(payload["windows"]), (50, 50, 20, 80))
            self.assertEqual(payload["prediction_type"], "epsilon")
            self.assertEqual(payload["fps"], 25.0)
            self.assertEqual(payload["dataset_sha256"], digest)
            self.assertEqual(payload["skeleton_signature"], G1_SKELETON.signature())
            self.assertEqual(tuple(payload["skeleton_parents"]), tuple(G1_SKELETON.parents))
            self.assertEqual(tuple(payload["skeleton_names"]), G1_SKELETON.names)
            self.assertEqual(tuple(payload["normalization"]["mean"].shape), (195,))
            self.assertEqual(tuple(payload["normalization"]["scale"].shape), (195,))
            self.assertEqual(set(payload["loss_weights"]), set(result["loss_weights"]))
            self.assertEqual([row["stage"] for row in payload["stage_history"]], ["A", "B", "C"])
            self.assertIn(payload["selected_sampler_steps"], (20, 50))
            self.assertEqual(set(payload["validation_quality"]), {"20", "50"})
            self.assertFalse(any("test" in key.lower() for key in payload))
            self.assertEqual(loaded.walk_model.schema.width, 16)
            self.assertEqual(loaded.pickup_model.schema.blocks, 1)

    def test_quality_gate_requires_a_successful_50_step_oracle(self):
        from tools.train_g1_overlap_diffusion import _select_sampler_steps

        quality = {
            "20": {"attach_proxy_at_8": 0.0, "no_stop_proxy": 1.0,
                   "median_grasp_position_m": 0.001, "median_grasp_orientation_degrees": 1.0, "rows": 1.0},
            "50": {"attach_proxy_at_8": 0.0, "no_stop_proxy": 1.0,
                   "median_grasp_position_m": 1.0, "median_grasp_orientation_degrees": 1.0, "rows": 1.0},
        }
        self.assertEqual(_select_sampler_steps(quality), 50)
        quality["50"]["attach_proxy_at_8"] = 0.5
        quality["20"]["attach_proxy_at_8"] = 0.48
        quality["20"]["median_grasp_position_m"] = 1.002
        quality["20"]["median_grasp_orientation_degrees"] = 3.0
        self.assertEqual(_select_sampler_steps(quality), 20)

    def test_no_stop_uses_global_contact_and_exact_boundaries(self):
        from tools.train_g1_overlap_diffusion import _passes_no_stop

        frames = torch.zeros((80, 195), dtype=torch.float32)
        frames[:, 6] = 1.0
        frames[:, 10] = 1.0
        frames[:, 0] = torch.arange(80, dtype=torch.float32) * 0.005
        self.assertTrue(_passes_no_stop(frames, 55))
        frames[30, 0] = frames[29, 0] + 0.005  # entry speed is 0.125 m/s
        frames[31, 0] = frames[30, 0] + 0.001
        frames[32, 0] = frames[31, 0] + 0.001
        frames[33, 0] = frames[32, 0] + 0.001
        self.assertFalse(_passes_no_stop(frames, 55))
        frames[33, 0] = frames[32, 0] + 0.0012  # exactly 0.03 m/s is not below
        self.assertTrue(_passes_no_stop(frames, 55))
        self.assertTrue(_passes_no_stop(frames, 33))  # Contact excludes frame 33

    def test_contact_metrics_use_target_hand_contact_in_global_frame(self):
        from tools.train_g1_overlap_diffusion import _contact_metrics

        generated = torch.zeros((2, 80, 195), dtype=torch.float32)
        generated[..., 6] = 1.0
        generated[..., 10] = 1.0
        generated[0, 55, 0] = 0.039
        generated[1, 55, 0] = 0.041
        generated[0, 55, 12] = 1.0  # 90 degrees, fails Attach orientation.
        target = torch.zeros((80, 195), dtype=torch.float32)
        target[55, 194] = 1.0
        static = torch.zeros(25, dtype=torch.float32)
        static[5] = 1.0
        static[9] = 1.0
        metrics = _contact_metrics(generated, target, static, _CANONICAL_OFFSETS)
        self.assertEqual(metrics["target_contact_frame"], 55)
        self.assertFalse(metrics["attach"][0].item())
        self.assertFalse(metrics["attach"][1].item())
        self.assertEqual(metrics["selected"], 0)

    def test_c_stage_is_required_and_best_pair_rolls_back(self):
        from tools.train_g1_overlap_diffusion import _atomic_best_pair, train_overlap

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, checkpoint = root / "dataset.npz", root / "checkpoint.pt"
            self._write_dataset(dataset)
            with self.assertRaisesRegex(ValueError, "epochs_c"):
                train_overlap(dataset, checkpoint, seed=19, device="cpu", epochs_a=1, epochs_b=1,
                              epochs_c=0, batch_size=2, validation_subset=1, model_width=16,
                              model_blocks=1, model_heads=1)
            best, requested = root / "best.pt", root / "requested.pt"
            best.write_bytes(b"old-best")
            requested.write_bytes(b"old-requested")
            real_replace = __import__("tools.train_g1_overlap_diffusion", fromlist=["os"]).os.replace
            calls = 0
            def fail_second(source, destination):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise OSError("injected requested publish failure")
                return real_replace(source, destination)
            with patch("tools.train_g1_overlap_diffusion.os.replace", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    _atomic_best_pair({"new": True}, best, requested)
            self.assertEqual(best.read_bytes(), b"old-best")
            self.assertEqual(requested.read_bytes(), b"old-requested")

            calls = 0
            def fail_second_backup(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected requested backup failure")
                return real_replace(source, destination)
            with patch("tools.train_g1_overlap_diffusion.os.replace", side_effect=fail_second_backup):
                with self.assertRaisesRegex(OSError, "backup"):
                    _atomic_best_pair({"newer": True}, best, requested)
            self.assertEqual(best.read_bytes(), b"old-best")
            self.assertEqual(requested.read_bytes(), b"old-requested")

    def test_quality_compares_step_counts_with_identical_candidate_noise(self):
        from resources.g1_interaction_builder.overlap_diffusion import MotionWindowDenoiser, TaskGuidance
        from tools.train_g1_overlap_diffusion import _quality_summary, _read_dataset

        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.npz"
            self._write_dataset(dataset)
            data = _read_dataset(dataset)
            model = MotionWindowDenoiser(width=16, blocks=1, heads=1)
            calls = []
            guided = []

            def fake_sample(*args, seed, candidates=8, steps, **kwargs):
                calls.append((steps, seed))
                guided.append(isinstance(kwargs.get("task_guidance"), TaskGuidance))
                result = torch.zeros((candidates, 80, 195), dtype=torch.float32)
                result[..., 6] = 1.0
                result[..., 10] = 1.0
                return result

            with patch("tools.train_g1_overlap_diffusion.sample_coupled", side_effect=fake_sample):
                _quality_summary(
                    model, model, data, torch.from_numpy(data["offsets"]),
                    seed=37, subset=2,
                )
            seeds_by_steps = {
                steps: [seed for observed_steps, seed in calls if observed_steps == steps]
                for steps in (20, 50)
            }
            self.assertEqual(seeds_by_steps[20], seeds_by_steps[50])
            self.assertTrue(all(guided))

    def test_loader_accepts_c_history_with_sparse_validation_intervals(self):
        from resources.g1_interaction_builder.overlap_diffusion import load_overlap_checkpoint
        from tools.train_g1_overlap_diffusion import train_overlap

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, checkpoint = root / "dataset.npz", root / "checkpoint.pt"
            self._write_dataset(dataset)
            train_overlap(
                dataset, checkpoint, seed=38, device="cpu", epochs_a=0, epochs_b=0,
                epochs_c=2, batch_size=2, validation_subset=1, validation_interval=2,
                model_width=16, model_blocks=1, model_heads=1,
            )
            loaded = load_overlap_checkpoint(checkpoint)
            self.assertEqual(loaded.payload["selected_stage"], {"stage": "C", "epoch": 2})
            self.assertNotIn("validation_rank", loaded.payload["stage_history"][0])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["selected_stage"] = {"stage": "C", "epoch": 1}
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_overlap_checkpoint(checkpoint)

    def test_loader_recomputes_selection_and_rejects_invalid_rates(self):
        from resources.g1_interaction_builder.overlap_diffusion import load_overlap_checkpoint
        from tools.train_g1_overlap_diffusion import train_overlap

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, checkpoint = root / "dataset.npz", root / "checkpoint.pt"
            self._write_dataset(dataset)
            train_overlap(dataset, checkpoint, seed=20, device="cpu", epochs_a=0, epochs_b=0,
                          epochs_c=1, batch_size=2, validation_subset=1, model_width=16,
                          model_blocks=1, model_heads=1)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["selected_sampler_steps"] = 20 if payload["selected_sampler_steps"] == 50 else 50
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "selection"):
                load_overlap_checkpoint(checkpoint)
            payload["selected_sampler_steps"] = 50
            payload["validation_quality"]["50"]["attach_proxy_at_8"] = 1.1
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "rate"):
                load_overlap_checkpoint(checkpoint)

    def test_best_c_checkpoint_emits_exportable_atomic_preview(self):
        from tools.train_g1_overlap_diffusion import train_overlap

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, checkpoint = root / "dataset.npz", root / "checkpoint.pt"
            self._write_dataset(dataset)
            train_overlap(dataset, checkpoint, seed=21, device="cpu", epochs_a=0, epochs_b=0,
                          epochs_c=1, batch_size=2, validation_subset=1, model_width=16,
                          model_blocks=1, model_heads=1)
            preview = checkpoint.with_suffix(".preview.npz")
            self.assertTrue(preview.is_file())
            with np.load(preview, allow_pickle=False) as archive:
                self.assertEqual(archive["selected_raw_generated"].shape, (80, 195))
                self.assertEqual(archive["local_positions"].shape, (31, 3))
                self.assertTrue(bool(archive["fixed_frame_zero_exact"]))
                np.testing.assert_array_equal(archive["fixed_frame_zero"], archive["generated_frame_zero"])
            output = root / "offline-overlap-generated.bin"
            result = subprocess.run([
                sys.executable, "tools/export_g1_overlap_pose.py", "--generated", str(preview),
                "--generated-key", "selected_raw_generated", "--local-positions", str(preview),
                "--local-positions-key", "local_positions", "--align-root-position", "0", "0", "0",
                "--align-root-yaw-degrees", "0", "--output", str(output),
            ], cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())

    def test_loader_rejects_noncanonical_skeleton_identity(self):
        from resources.g1_interaction_builder.overlap_diffusion import load_overlap_checkpoint
        from tools.train_g1_overlap_diffusion import train_overlap

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, checkpoint = root / "dataset.npz", root / "checkpoint.pt"
            self._write_dataset(dataset)
            train_overlap(
                dataset, checkpoint, seed=18, device="cpu", epochs_a=0, epochs_b=0,
                epochs_c=1, batch_size=2, validation_subset=1, model_width=16,
                model_blocks=1, model_heads=1,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["skeleton_signature"] = "not-g1"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "skeleton signature"):
                load_overlap_checkpoint(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
