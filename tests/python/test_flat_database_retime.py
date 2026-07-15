import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from resources.retime_flat_database import (
    AUTHORITATIVE_SOURCE_SHA256,
    FlatDatabase,
    TARGET_FPS,
    read_database,
    retime_database,
    retime_file,
    write_database,
)


def _yaw_quaternion(angle_radians: np.ndarray) -> np.ndarray:
    half = 0.5 * angle_radians
    result = np.zeros(angle_radians.shape + (4,), dtype=np.float32)
    result[..., 0] = np.cos(half)
    result[..., 2] = np.sin(half)
    return result


def _fixture(range_lengths=(13, 25), source_fps=60.0) -> FlatDatabase:
    frame_count = sum(range_lengths)
    bone_count = 2
    positions = np.zeros((frame_count, bone_count, 3), dtype=np.float32)
    rotations = np.zeros((frame_count, bone_count, 4), dtype=np.float32)
    velocities = np.full_like(positions, 1234.0)
    angular_velocities = np.full_like(positions, -4321.0)
    contacts = np.zeros((frame_count, 2), dtype=np.uint8)
    starts = []
    stops = []

    cursor = 0
    for range_index, length in enumerate(range_lengths):
        starts.append(cursor)
        stops.append(cursor + length)
        time = np.arange(length, dtype=np.float32) / np.float32(source_fps)
        # Each range starts in a visibly different location. Recomputing
        # derivatives across a range boundary would therefore be obvious.
        positions[cursor:cursor + length, 0, 0] = 2.0 * range_index + 1.5 * time
        positions[cursor:cursor + length, 1, 1] = 1.0 + 0.25 * time
        rotations[cursor:cursor + length, 0] = _yaw_quaternion(0.75 * time)
        rotations[cursor:cursor + length, 1, 0] = 1.0
        # Equivalent antipodal representations must not make interpolation
        # take a long arc or create an angular-velocity spike.
        rotations[cursor + 1:cursor + length:2] *= -1.0
        contacts[cursor:cursor + length, 0] = (
            np.arange(length) >= length // 2).astype(np.uint8)
        contacts[cursor:cursor + length, 1] = (
            np.arange(length) % 3 == 0).astype(np.uint8)
        cursor += length

    return FlatDatabase(
        bone_positions=positions,
        bone_velocities=velocities,
        bone_rotations=rotations,
        bone_angular_velocities=angular_velocities,
        bone_parents=np.array([-1, 0], dtype=np.int32),
        range_starts=np.asarray(starts, dtype=np.int32),
        range_stops=np.asarray(stops, dtype=np.int32),
        contact_states=contacts,
    )


class FlatDatabaseRetimeTests(unittest.TestCase):
    def test_binary_round_trip_is_exact(self):
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "database.bin"
            write_database(path, fixture)
            loaded = read_database(path)

        for field in fixture.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(loaded, field), getattr(fixture, field), field)

    def test_ranges_retime_independently_and_preserve_endpoints(self):
        fixture = _fixture()
        output = retime_database(fixture, source_fps=60.0, target_fps=25.0)

        np.testing.assert_array_equal(output.range_starts, [0, 6])
        np.testing.assert_array_equal(output.range_stops, [6, 17])
        for source_start, source_stop, target_start, target_stop in zip(
                fixture.range_starts,
                fixture.range_stops,
                output.range_starts,
                output.range_stops):
            np.testing.assert_allclose(
                output.bone_positions[target_start],
                fixture.bone_positions[source_start],
                atol=1e-7,
                rtol=0.0)
            np.testing.assert_allclose(
                output.bone_positions[target_stop - 1],
                fixture.bone_positions[source_stop - 1],
                atol=1e-7,
                rtol=0.0)
            self.assertLessEqual(
                abs(
                    (target_stop - target_start - 1) / 25.0
                    - (source_stop - source_start - 1) / 60.0),
                0.5 / 25.0 + 1e-9)

        # A boundary jump must not leak into either range's velocity.
        np.testing.assert_allclose(
            output.bone_velocities[:, 0, 0], 1.5, atol=2e-5, rtol=0.0)
        np.testing.assert_allclose(
            output.bone_velocities[:, 1, 1], 0.25, atol=2e-5, rtol=0.0)

    def test_shortest_arc_rotations_are_normalized_with_per_second_velocity(self):
        output = retime_database(_fixture(), source_fps=60.0, target_fps=25.0)

        np.testing.assert_allclose(
            np.linalg.norm(output.bone_rotations, axis=-1),
            1.0,
            atol=2e-6,
            rtol=0.0)
        for start, stop in zip(output.range_starts, output.range_stops):
            dots = np.sum(
                output.bone_rotations[start + 1:stop]
                * output.bone_rotations[start:stop - 1],
                axis=-1)
            self.assertTrue(np.all(dots >= 0.0))
        np.testing.assert_allclose(
            output.bone_angular_velocities[:, 0, 1],
            0.75,
            atol=2e-5,
            rtol=0.0)
        np.testing.assert_allclose(
            output.bone_angular_velocities[:, 0, (0, 2)],
            0.0,
            atol=2e-5,
            rtol=0.0)

    def test_contacts_use_deterministic_nearest_source_sample(self):
        fixture = _fixture(range_lengths=(13,))
        output = retime_database(fixture, source_fps=60.0, target_fps=25.0)

        # Six output samples map to source coordinates 0, 2.4, 4.8, 7.2,
        # 9.6, 12. Nearest-half-up therefore selects 0, 2, 5, 7, 10, 12.
        np.testing.assert_array_equal(
            output.contact_states,
            fixture.contact_states[[0, 2, 5, 7, 10, 12]])

    def test_file_retime_is_reproducible_and_records_manifest(self):
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "database_60hz.bin"
            first = root / "first.bin"
            second = root / "second.bin"
            first_manifest = root / "first.json"
            second_manifest = root / "second.json"
            write_database(source, fixture)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            first_record = retime_file(
                source,
                first,
                first_manifest,
                expected_source_sha256=source_hash)
            second_record = retime_file(
                source,
                second,
                second_manifest,
                expected_source_sha256=source_hash)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_manifest.read_bytes(), second_manifest.read_bytes())
            self.assertEqual(first_record, second_record)
            self.assertEqual(first_record["source_fps"], 60.0)
            self.assertEqual(first_record["target_fps"], TARGET_FPS)
            self.assertEqual(first_record["source_sha256"], source_hash)
            self.assertEqual(first_record["bone_count"], 2)
            self.assertEqual(first_record["range_count"], 2)
            self.assertLessEqual(
                first_record["maximum_range_duration_error_seconds"],
                0.5 / TARGET_FPS + 1e-9)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                retime_file(
                    source,
                    root / "rejected.bin",
                    root / "rejected.json",
                    expected_source_sha256="0" * 64)

    def test_malformed_database_is_rejected(self):
        fixture = _fixture()
        fixture.range_starts[1] += 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            retime_database(fixture, source_fps=60.0, target_fps=25.0)


class AuthoritativeFlatDatabaseTests(unittest.TestCase):
    def test_repository_artifacts_have_25hz_provenance(self):
        repository = Path(__file__).resolve().parents[2]
        source_path = repository / "resources" / "database_60hz.bin"
        output_path = repository / "resources" / "database.bin"
        manifest_path = (
            repository / "resources" / "database_25hz_manifest.json")

        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = read_database(source_path)
        output = read_database(output_path)

        self.assertEqual(source_hash, AUTHORITATIVE_SOURCE_SHA256)
        self.assertEqual(manifest["source_sha256"], source_hash)
        self.assertEqual(manifest["output_sha256"], output_hash)
        self.assertEqual(manifest["source_fps"], 60.0)
        self.assertEqual(manifest["target_fps"], 25.0)
        self.assertEqual(source.bone_positions.shape, (53500, 23, 3))
        self.assertEqual(output.bone_positions.shape, (22296, 23, 3))
        self.assertEqual(len(source.range_starts), 6)
        self.assertEqual(len(output.range_starts), 6)
        self.assertEqual(source.contact_states.shape[1], 2)
        self.assertEqual(output.contact_states.shape[1], 2)
        self.assertEqual(manifest["source_frame_count"], 53500)
        self.assertEqual(manifest["output_frame_count"], 22296)
        self.assertEqual(manifest["bone_count"], 23)
        self.assertEqual(manifest["range_count"], 6)
        self.assertLessEqual(
            manifest["maximum_range_duration_error_seconds"],
            0.5 / 25.0 + 1e-9)

        for source_start, source_stop, target_start, target_stop in zip(
                source.range_starts,
                source.range_stops,
                output.range_starts,
                output.range_stops):
            np.testing.assert_array_equal(
                output.bone_positions[target_start],
                source.bone_positions[source_start])
            np.testing.assert_array_equal(
                output.bone_positions[target_stop - 1],
                source.bone_positions[source_stop - 1])
            first_dot = np.abs(np.sum(
                output.bone_rotations[target_start]
                * source.bone_rotations[source_start], axis=-1))
            last_dot = np.abs(np.sum(
                output.bone_rotations[target_stop - 1]
                * source.bone_rotations[source_stop - 1], axis=-1))
            np.testing.assert_allclose(first_dot, 1.0, atol=2e-6, rtol=0.0)
            np.testing.assert_allclose(last_dot, 1.0, atol=2e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
