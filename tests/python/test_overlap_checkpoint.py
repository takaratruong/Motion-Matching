import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - dependency gate
    torch = None

from resources.g1_interaction_builder.schema import G1_SKELETON


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
            "canonical_local_offsets": np.zeros((31, 3), np.float32),
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
