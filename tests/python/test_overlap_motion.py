from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from resources.g1_interaction_builder import overlap_motion
from resources.g1_interaction_builder.holden_database import read_holden_database
from resources.g1_interaction_builder.schema import G1_SKELETON


BONE_COUNT = 31


def _motion_fixture(frames: int = 4):
    positions = np.zeros((frames, BONE_COUNT, 3), np.float32)
    positions[:, 0] = np.array(
        [[1.0 + frame * 0.04, 0.9, -0.5 + frame * 0.02]
         for frame in range(frames)],
        np.float32,
    )
    positions[:, 1:] = np.arange(
        (BONE_COUNT - 1) * 3, dtype=np.float32
    ).reshape(BONE_COUNT - 1, 3) * 0.01
    positions[:, 1, 0] += np.linspace(0.0, 0.08, frames, dtype=np.float32)

    angle = np.linspace(0.0, 0.3, frames, dtype=np.float32)[:, None]
    rotations = np.zeros((frames, BONE_COUNT, 4), np.float32)
    rotations[..., 0] = np.cos(angle * 0.5)
    rotations[..., 2] = np.sin(angle * 0.5)
    rotations[1::2] *= -1.0
    contacts = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        np.float32,
    )[:frames]
    return positions, rotations, contacts


def _write_database(path: Path, *, frames: int = 4, bones: int = BONE_COUNT,
                    ranges: tuple[np.ndarray, np.ndarray] | None = None,
                    contacts: int = 2, trailing: bytes = b"",
                    finite: bool = True,
                    parents: np.ndarray | None = None) -> None:
    positions = np.arange(frames * bones * 3, dtype=np.float32).reshape(
        frames, bones, 3
    ) * 0.01
    if not finite:
        positions[0, 0, 0] = np.nan
    velocities = positions + 1.0
    rotations = np.zeros((frames, bones, 4), np.float32)
    rotations[..., 0] = 1.0
    angular_velocities = positions + 2.0
    if parents is None:
        parents = (
            G1_SKELETON.parents
            if bones == BONE_COUNT
            else np.arange(-1, bones - 1, dtype=np.int32)
        )
    if ranges is None:
        starts = np.array([0, 2], np.int32)
        stops = np.array([2, frames], np.int32)
    else:
        starts, stops = ranges
    contact_values = np.zeros((frames, contacts), np.uint8)

    with path.open("wb") as stream:
        for values in (positions, velocities, rotations, angular_velocities):
            stream.write(struct.pack("<II", frames, bones))
            stream.write(values.astype("<f4", copy=False).tobytes())
        stream.write(struct.pack("<I", bones))
        stream.write(parents.astype("<i4", copy=False).tobytes())
        stream.write(struct.pack("<I", len(starts)))
        stream.write(starts.astype("<i4", copy=False).tobytes())
        stream.write(struct.pack("<I", len(stops)))
        stream.write(stops.astype("<i4", copy=False).tobytes())
        stream.write(struct.pack("<II", frames, contacts))
        stream.write(contact_values.tobytes())
        stream.write(trailing)


class OverlapMotionTests(unittest.TestCase):
    def test_frame_schema_is_frozen(self):
        self.assertEqual(overlap_motion.BONE_COUNT, 31)
        self.assertEqual(overlap_motion.FRAME_DIM, 3 + 3 + 31 * 6 + 3)
        self.assertEqual(overlap_motion.walk_slice(), slice(0, 50))
        self.assertEqual(overlap_motion.pickup_slice(), slice(30, 80))

    def test_codec_round_trips_pose_and_canonicalizes_quaternion_sign(self):
        positions, rotations, contacts = _motion_fixture()
        anchor = np.array([0.25, 0.0, -0.75], np.float32)

        encoded = overlap_motion.encode_motion(
            positions, rotations, contacts, anchor
        )
        decoded = overlap_motion.decode_motion(encoded, positions[0], anchor)

        self.assertEqual(encoded.shape, (4, 195))
        np.testing.assert_allclose(
            decoded.positions[:, 0], positions[:, 0], atol=2e-5
        )
        np.testing.assert_allclose(
            decoded.positions[:, 1], positions[:, 1], atol=2e-5,
        )
        self.assertGreaterEqual(
            np.ptp(decoded.positions[:, 1, 0]), 0.08 - 2e-5,
        )
        np.testing.assert_allclose(
            decoded.positions[:, 2:],
            np.broadcast_to(positions[:1, 2:], decoded.positions[:, 2:].shape),
            atol=2e-5,
        )
        np.testing.assert_allclose(
            np.abs((decoded.rotations * rotations).sum(-1)), 1, atol=2e-5
        )
        np.testing.assert_allclose(decoded.contacts, contacts, atol=0)
        np.testing.assert_allclose(
            decoded.velocities[1, 0], [1.0, 0.0, 0.5], atol=1e-6
        )

    def test_decoder_rejects_degenerate_rotation6d_columns(self):
        encoded = np.zeros((1, overlap_motion.FRAME_DIM), np.float32)
        local_positions, _, _ = _motion_fixture(1)
        with self.assertRaisesRegex(ValueError, "rotation6d"):
            overlap_motion.decode_motion(encoded, local_positions[0], np.zeros(3))

    def test_reader_loads_the_eight_little_endian_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.bin"
            _write_database(path)
            database = read_holden_database(path)

        self.assertEqual(database.positions.shape, (4, 31, 3))
        self.assertEqual(database.rotations.shape, (4, 31, 4))
        self.assertEqual(database.contacts.shape, (4, 2))
        np.testing.assert_array_equal(database.range_starts, [0, 2])
        np.testing.assert_array_equal(database.range_stops, [2, 4])
        self.assertEqual(database.positions.dtype, np.dtype("<f4"))

    def test_reader_rejects_incorrect_31_bone_parent_hierarchy(self):
        parents = G1_SKELETON.parents.copy()
        parents[8] = 0

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "database.bin"
            _write_database(path, parents=parents)
            with self.assertRaisesRegex(ValueError, "exact canonical G1"):
                read_holden_database(path)

    def test_reader_rejects_malformed_database_contracts(self):
        cases = (
            ("trailing", {"trailing": b"x"}, "trailing"),
            ("non_finite", {"finite": False}, "non-finite"),
            ("bone_count", {"bones": 30}, "31"),
            ("contact_count", {"contacts": 3}, "two"),
            (
                "non_contiguous_ranges",
                {"ranges": (np.array([0, 3], np.int32),
                            np.array([2, 4], np.int32))},
                "contiguous",
            ),
        )
        for name, kwargs, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "database.bin"
                _write_database(path, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    read_holden_database(path)


if __name__ == "__main__":
    unittest.main()
