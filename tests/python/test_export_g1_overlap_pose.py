from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools import export_g1_overlap_pose as exporter


class ExportG1OverlapPoseTests(unittest.TestCase):
    def test_decodes_dynamic_hips_and_aligns_frame_zero_root_and_yaw(self):
        encoded = np.zeros((80, 195), np.float32)
        encoded[:, 0] = np.linspace(0.0, 3.16, 80, dtype=np.float32)
        encoded[:, 3] = np.linspace(0.1, 0.4, 80, dtype=np.float32)
        encoded[:, 6:192].reshape(80, 31, 6)[..., 0] = 1.0
        encoded[:, 6:192].reshape(80, 31, 6)[..., 4] = 1.0
        local_positions = np.zeros((31, 3), np.float32)
        decoded = exporter.decode_encoded_motion(
            encoded,
            local_positions,
            np.array([4.0, 1.0, -2.0], np.float32),
            np.pi / 2.0,
        )

        np.testing.assert_allclose(decoded.positions[0, 0], [4.0, 1.0, -2.0])
        self.assertGreater(decoded.positions[-1, 1, 0], 0.39)
        np.testing.assert_allclose(decoded.rotations[0, 0], [2**-0.5, 0.0, 2**-0.5, 0.0], atol=2e-6)

    def test_writes_atomic_strict_little_endian_payload(self):
        frames = 80
        positions = np.zeros((frames, 31, 3), np.float32)
        velocities = np.ones((frames, 31, 3), np.float32)
        angular_velocities = np.full((frames, 31, 3), 2.0, np.float32)
        rotations = np.zeros((frames, 31, 4), np.float32)
        rotations[..., 0] = 1.0
        contacts = np.zeros((frames, 3), np.float32)
        contacts[:, 1] = 1.0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pose.bin"
            output.write_bytes(b"previous")
            exporter.export_offline_overlap(
                output, positions, velocities, angular_velocities, rotations, contacts
            )
            payload = output.read_bytes()

        self.assertEqual(payload[:8], b"G1OVLP01")
        self.assertEqual(struct.unpack_from("<IIIII", payload, 8), (1, 25, 80, 31, 2))
        signature_size = struct.unpack_from("<I", payload, 28)[0]
        self.assertEqual(signature_size, 64)
        expected_size = 8 + 6 * 4 + 64 + sum(4 + len(name) for name in exporter.G1_BONE_NAMES)
        expected_size += frames * 31 * 3 * 4 * 3
        expected_size += frames * 31 * 4 * 4
        expected_size += frames * 2
        self.assertEqual(len(payload), expected_size)

    def test_cli_exports_generated_npz_outside_repository_working_directory(self):
        encoded = np.zeros((80, 195), np.float32)
        encoded[:, 6:192].reshape(80, 31, 6)[..., 0] = 1.0
        encoded[:, 6:192].reshape(80, 31, 6)[..., 4] = 1.0
        local_positions = np.zeros((31, 3), np.float32)
        script = Path(__file__).resolve().parents[2] / "tools" / "export_g1_overlap_pose.py"
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            generated = directory_path / "generated.npz"
            offsets = directory_path / "offsets.npz"
            output = directory_path / "pose.bin"
            np.savez(generated, generated=encoded)
            np.savez(offsets, local_positions=local_positions)
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--generated", str(generated), "--generated-key", "generated",
                    "--local-positions", str(offsets), "--local-positions-key", "local_positions",
                    "--align-root-position", "0", "0", "0",
                    "--align-root-yaw-degrees", "0",
                    "--output", str(output),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())

    def test_cli_uses_dataset_canonical_offsets_without_interaction_pack(self):
        encoded = np.zeros((50, 195), np.float32)
        encoded[:, 6:192].reshape(50, 31, 6)[..., 0] = 1.0
        encoded[:, 6:192].reshape(50, 31, 6)[..., 4] = 1.0
        script = Path(__file__).resolve().parents[2] / "tools" / "export_g1_overlap_pose.py"
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            dataset = directory_path / "dataset.npz"
            output = directory_path / "pose.bin"
            np.savez(
                dataset,
                train_walk_windows=encoded[None],
                train_pickup_windows=encoded[None],
                train_sequence_indices=np.array([0], np.int32),
                train_source_walk_ranges=np.array([[0, 50]], np.int32),
                canonical_local_positions=np.zeros((31, 3), np.float32),
            )
            result = subprocess.run(
                [
                    sys.executable, str(script), "--dataset", str(dataset),
                    "--split", "train", "--row", "0",
                    "--align-root-position", "0", "0", "0",
                    "--align-root-yaw-degrees", "0", "--output", str(output),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
