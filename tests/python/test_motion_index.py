import math
import os
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from resources.g1_terrain_builder import motion_index
from resources.g1_terrain_builder.motion_index import (
    DIRECTION_BACK_LEFT,
    DIRECTION_BACK_RIGHT,
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_FORWARD_LEFT,
    DIRECTION_FORWARD_RIGHT,
    DIRECTION_IDLE,
    DIRECTION_LEFT,
    DIRECTION_OVERLAP_HALF_WIDTH_DEGREES,
    DIRECTION_RIGHT,
    ELEVATION_CHANGE_THRESHOLD_METRES,
    G1MI_MAGIC,
    G1MI_ROW_WIDTH,
    G1MI_VERSION,
    LOW_SPEED_MAX_DISPLACEMENT_METRES,
    MAX_LATERAL_HEADING_CHANGE_DEGREES,
    MOVING_MIN_DISPLACEMENT_METRES,
    SPEED_LOW,
    SPEED_MOVING,
    derive_motion_index,
    motion_index_bytes,
    motion_index_from_bytes,
    read_motion_index,
    write_motion_index,
)
from resources.g1_terrain_builder.schema import (
    SourceFrameRange,
    TERRAIN_FAMILIES,
    TerrainBank,
    TerrainBankIndex,
)


COMPASS = (
    (0.0, DIRECTION_FORWARD),
    (45.0, DIRECTION_FORWARD_RIGHT),
    (90.0, DIRECTION_RIGHT),
    (135.0, DIRECTION_BACK_RIGHT),
    (180.0, DIRECTION_BACKWARD),
    (225.0, DIRECTION_BACK_LEFT),
    (270.0, DIRECTION_LEFT),
    (315.0, DIRECTION_FORWARD_LEFT),
)


def _bank_index(frame_count, source_stops=None):
    if source_stops is None:
        source_stops = (frame_count,)
    ranges = []
    start = 0
    for source_number, stop in enumerate(source_stops):
        frames = stop - start
        ranges.append(SourceFrameRange(
            source_name=f"source-{source_number}",
            source_start=0,
            source_stop=frames,
            source_frame_count=frames,
            global_start=start,
            global_stop=stop,
        ))
        start = stop
    banks = tuple(
        TerrainBank(
            family=family,
            range_indices=tuple(range(len(ranges))) if family == "flat" else (),
        )
        for family in TERRAIN_FAMILIES
    )
    return TerrainBankIndex(frame_count, tuple(ranges), banks)


def _trajectory(
    travel_degrees=0.0,
    distance=0.25,
    elevation=0.0,
    heading_change_radians=0.0,
):
    radians = math.radians(travel_degrees)
    positions = np.zeros((26, 3), dtype=np.float64)
    positions[:, 0] = np.linspace(0.0, math.sin(radians) * distance, 26)
    positions[:, 1] = np.linspace(0.0, elevation, 26)
    positions[:, 2] = np.linspace(0.0, math.cos(radians) * distance, 26)
    headings = np.linspace(0.0, heading_change_radians, 26)
    return derive_motion_index(positions, headings, _bank_index(26))


class MotionIndexDerivationTests(unittest.TestCase):
    def test_numeric_assignments_and_predeclared_thresholds_are_locked(self):
        self.assertEqual(
            (
                DIRECTION_IDLE,
                DIRECTION_FORWARD,
                DIRECTION_FORWARD_RIGHT,
                DIRECTION_RIGHT,
                DIRECTION_BACK_RIGHT,
                DIRECTION_BACKWARD,
                DIRECTION_BACK_LEFT,
                DIRECTION_LEFT,
                DIRECTION_FORWARD_LEFT,
            ),
            (0x001, 0x002, 0x004, 0x008, 0x010, 0x020, 0x040, 0x080, 0x100),
        )
        self.assertEqual((SPEED_LOW, SPEED_MOVING), (0x01, 0x02))
        self.assertEqual(DIRECTION_OVERLAP_HALF_WIDTH_DEGREES, 5.0)
        self.assertEqual(MAX_LATERAL_HEADING_CHANGE_DEGREES, 15.0)
        self.assertEqual(MOVING_MIN_DISPLACEMENT_METRES, 0.08)
        self.assertEqual(LOW_SPEED_MAX_DISPLACEMENT_METRES, 0.12)
        self.assertEqual(ELEVATION_CHANGE_THRESHOLD_METRES, 0.03)

    def test_all_eight_compass_centers_have_exact_single_bits(self):
        for degrees, expected in COMPASS:
            with self.subTest(degrees=degrees):
                index = _trajectory(travel_degrees=degrees)
                self.assertEqual(int(index.direction_masks[0]), expected)

    def test_every_adjacent_sector_overlap_is_symmetric_and_inclusive(self):
        half_width = DIRECTION_OVERLAP_HALF_WIDTH_DEGREES
        for sector, (center, lower_bit) in enumerate(COMPASS):
            upper_bit = COMPASS[(sector + 1) % len(COMPASS)][1]
            boundary = center + 22.5
            expected_overlap = lower_bit | upper_bit
            cases = (
                (boundary - half_width - 0.25, lower_bit),
                (boundary - half_width, expected_overlap),
                (boundary, expected_overlap),
                (boundary + half_width, expected_overlap),
                (boundary + half_width + 0.25, upper_bit),
            )
            for degrees, expected in cases:
                with self.subTest(boundary=boundary % 360.0, degrees=degrees):
                    index = _trajectory(travel_degrees=degrees)
                    self.assertEqual(int(index.direction_masks[0]), expected)

    def test_mirrored_lateral_and_diagonal_motions_remain_distinct(self):
        cases = (
            (90.0, DIRECTION_RIGHT),
            (-90.0, DIRECTION_LEFT),
            (45.0, DIRECTION_FORWARD_RIGHT),
            (-45.0, DIRECTION_FORWARD_LEFT),
            (135.0, DIRECTION_BACK_RIGHT),
            (-135.0, DIRECTION_BACK_LEFT),
        )
        for degrees, expected in cases:
            with self.subTest(degrees=degrees):
                self.assertEqual(
                    int(_trajectory(travel_degrees=degrees).direction_masks[0]),
                    expected,
                )

    def test_travel_is_expressed_relative_to_source_heading(self):
        headings = np.full(26, math.pi / 2.0, dtype=np.float64)
        forward_positions = np.zeros((26, 3), dtype=np.float64)
        forward_positions[:, 0] = np.linspace(0.0, 0.25, 26)
        right_positions = np.zeros((26, 3), dtype=np.float64)
        right_positions[:, 2] = np.linspace(0.0, -0.25, 26)

        forward = derive_motion_index(
            forward_positions, headings, _bank_index(26)
        )
        right = derive_motion_index(
            right_positions, headings, _bank_index(26)
        )

        self.assertEqual(int(forward.direction_masks[0]), DIRECTION_FORWARD)
        self.assertEqual(int(right.direction_masks[0]), DIRECTION_RIGHT)

    def test_speed_overlap_covers_stationary_start_and_stop_distances(self):
        cases = (
            (0.0, SPEED_LOW, DIRECTION_IDLE),
            (0.079, SPEED_LOW, DIRECTION_IDLE),
            (0.08, SPEED_LOW | SPEED_MOVING, DIRECTION_FORWARD),
            (0.10, SPEED_LOW | SPEED_MOVING, DIRECTION_FORWARD),
            (0.12, SPEED_LOW | SPEED_MOVING, DIRECTION_FORWARD),
            (0.121, SPEED_MOVING, DIRECTION_FORWARD),
        )
        for distance, speed, direction in cases:
            with self.subTest(distance=distance):
                index = _trajectory(distance=distance)
                self.assertEqual(int(index.speed_masks[0]), speed)
                self.assertEqual(int(index.direction_masks[0]), direction)

    def test_acceleration_deceleration_and_clip_end_clamping_stay_indexed(self):
        positions = np.zeros((51, 3), dtype=np.float64)
        positions[26:, 2] = np.linspace(0.01, 0.25, 25)
        index = derive_motion_index(
            positions,
            np.zeros(51, dtype=np.float64),
            _bank_index(51),
        )

        expected = {
            0: (DIRECTION_IDLE, SPEED_LOW),
            1: (DIRECTION_IDLE, SPEED_LOW),
            10: (DIRECTION_FORWARD, SPEED_LOW | SPEED_MOVING),
            26: (DIRECTION_FORWARD, SPEED_MOVING),
            45: (DIRECTION_IDLE, SPEED_LOW),
            50: (DIRECTION_IDLE, SPEED_LOW),
        }
        for frame, (direction, speed) in expected.items():
            with self.subTest(frame=frame):
                self.assertEqual(int(index.direction_masks[frame]), direction)
                self.assertEqual(int(index.speed_masks[frame]), speed)
        self.assertTrue(np.all(index.direction_masks != 0))
        self.assertTrue(np.all(index.speed_masks != 0))

    def test_lateral_heading_limit_accepts_exactly_fifteen_and_rejects_next_value(self):
        exact = math.radians(MAX_LATERAL_HEADING_CHANGE_DEGREES)
        above = np.nextafter(exact, math.inf)

        accepted = _trajectory(
            travel_degrees=90.0,
            heading_change_radians=exact,
        )
        rejected = _trajectory(
            travel_degrees=90.0,
            heading_change_radians=above,
        )

        self.assertEqual(int(accepted.direction_masks[0]), DIRECTION_RIGHT)
        self.assertEqual(int(rejected.direction_masks[0]), DIRECTION_IDLE)

    def test_turning_forward_trajectory_is_never_relabelled_as_strafe_or_diagonal(self):
        index = _trajectory(
            travel_degrees=20.0,
            heading_change_radians=math.radians(30.0),
        )
        forbidden = (
            DIRECTION_FORWARD_RIGHT
            | DIRECTION_RIGHT
            | DIRECTION_BACK_RIGHT
            | DIRECTION_BACK_LEFT
            | DIRECTION_LEFT
            | DIRECTION_FORWARD_LEFT
        )
        self.assertEqual(int(index.direction_masks[0]), DIRECTION_FORWARD)
        self.assertEqual(int(index.direction_masks[0]) & forbidden, 0)

    def test_ascent_level_and_descent_use_exact_insignificant_threshold(self):
        threshold = ELEVATION_CHANGE_THRESHOLD_METRES
        cases = (
            (np.nextafter(threshold, math.inf), 1),
            (threshold, 0),
            (0.0, 0),
            (-threshold, 0),
            (np.nextafter(-threshold, -math.inf), -1),
        )
        for elevation, expected in cases:
            with self.subTest(elevation=elevation):
                index = _trajectory(elevation=elevation)
                self.assertEqual(int(index.elevation_modes[0]), expected)

    def test_one_second_horizons_never_cross_adjacent_source_boundaries(self):
        positions = np.zeros((52, 3), dtype=np.float64)
        positions[26:, 0] = 100.0
        positions[26:, 1] = 100.0
        headings = np.zeros(52, dtype=np.float64)
        headings[26:] = math.pi / 2.0

        index = derive_motion_index(
            positions,
            headings,
            _bank_index(52, source_stops=(26, 52)),
        )

        for frame in (0, 25, 26, 51):
            with self.subTest(frame=frame):
                self.assertEqual(int(index.direction_masks[frame]), DIRECTION_IDLE)
                self.assertEqual(int(index.speed_masks[frame]), SPEED_LOW)
                self.assertEqual(int(index.elevation_modes[frame]), 0)

    def test_derivation_is_copy_safe_read_only_and_validates_inputs(self):
        positions = np.zeros((26, 3), dtype=np.float64)
        positions[:, 2] = np.linspace(0.0, 0.25, 26)
        headings = np.zeros(26, dtype=np.float64)
        original_positions = positions.copy()
        original_headings = headings.copy()

        index = derive_motion_index(positions, headings, _bank_index(26))

        np.testing.assert_array_equal(positions, original_positions)
        np.testing.assert_array_equal(headings, original_headings)
        for values in (
            index.direction_masks,
            index.speed_masks,
            index.elevation_modes,
        ):
            self.assertTrue(values.flags.owndata)
            self.assertFalse(values.flags.writeable)

        invalid_cases = (
            (np.zeros((26, 2)), headings, _bank_index(26), "root positions"),
            (positions, np.zeros((26, 1)), _bank_index(26), "headings"),
            (
                np.full((26, 3), np.nan),
                headings,
                _bank_index(26),
                "finite",
            ),
            (positions, headings, _bank_index(25), "frame count"),
        )
        for bad_positions, bad_headings, banks, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    derive_motion_index(bad_positions, bad_headings, banks)


class MotionIndexCodecTests(unittest.TestCase):
    def _valid_arrays(self):
        return (
            np.array([DIRECTION_IDLE, DIRECTION_FORWARD_RIGHT | DIRECTION_RIGHT], np.uint16),
            np.array([SPEED_LOW, SPEED_LOW | SPEED_MOVING], np.uint8),
            np.array([-1, 1], np.int8),
        )

    def _valid_bytes(self):
        directions, speeds, elevations = self._valid_arrays()
        return motion_index_bytes(directions, speeds, elevations)

    def test_exact_header_and_four_byte_rows_include_signed_negative_elevation(self):
        directions, speeds, elevations = self._valid_arrays()
        expected = (
            struct.pack("<4sIII", b"G1MI", 1, 2, 4)
            + struct.pack("<HBb", DIRECTION_IDLE, SPEED_LOW, -1)
            + struct.pack(
                "<HBb",
                DIRECTION_FORWARD_RIGHT | DIRECTION_RIGHT,
                SPEED_LOW | SPEED_MOVING,
                1,
            )
        )

        self.assertEqual((G1MI_MAGIC, G1MI_VERSION, G1MI_ROW_WIDTH), (b"G1MI", 1, 4))
        self.assertEqual(motion_index_bytes(directions, speeds, elevations), expected)
        self.assertEqual(expected[18:20], bytes((SPEED_LOW, 0xFF)))

    def test_bytes_and_file_round_trip_are_deterministic_and_owned(self):
        arrays = self._valid_arrays()
        first = motion_index_bytes(*arrays)
        second = motion_index_bytes(*arrays)
        decoded = motion_index_from_bytes(first)
        self.assertEqual(first, second)

        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "motion-index.bin")
            write_motion_index(path, *arrays)
            with open(path, "rb") as stream:
                on_disk = stream.read()
            from_file = read_motion_index(path)

        self.assertEqual(on_disk, first)
        for actual, expected in zip(
            (
                decoded.direction_masks,
                decoded.speed_masks,
                decoded.elevation_modes,
                from_file.direction_masks,
                from_file.speed_masks,
                from_file.elevation_modes,
            ),
            (*arrays, *arrays),
        ):
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(actual.flags.owndata)
            self.assertFalse(actual.flags.writeable)

    def test_codec_rejects_zero_unknown_contradictory_and_nonadjacent_direction_masks(self):
        invalid = (
            0,
            0x200,
            DIRECTION_IDLE | DIRECTION_FORWARD,
            DIRECTION_FORWARD | DIRECTION_BACKWARD,
            DIRECTION_FORWARD | DIRECTION_RIGHT,
            DIRECTION_FORWARD | DIRECTION_FORWARD_RIGHT | DIRECTION_RIGHT,
        )
        for mask in invalid:
            with self.subTest(mask=hex(mask)):
                with self.assertRaisesRegex(ValueError, "direction mask"):
                    motion_index_bytes(
                        np.array([mask], np.uint16),
                        np.array([SPEED_LOW], np.uint8),
                        np.array([0], np.int8),
                    )

        payload = motion_index_bytes(
            np.array([DIRECTION_FORWARD_LEFT | DIRECTION_FORWARD], np.uint16),
            np.array([SPEED_MOVING], np.uint8),
            np.array([0], np.int8),
        )
        self.assertIsInstance(motion_index_from_bytes(payload).direction_masks, np.ndarray)

    def test_codec_rejects_zero_or_unknown_speed_and_invalid_elevation(self):
        cases = (
            (SPEED_LOW, 2, "elevation"),
            (SPEED_LOW, -2, "elevation"),
            (0, 0, "speed mask"),
            (0x04, 0, "speed mask"),
        )
        for speed, elevation, message in cases:
            with self.subTest(speed=speed, elevation=elevation):
                with self.assertRaisesRegex(ValueError, message):
                    motion_index_bytes(
                        np.array([DIRECTION_IDLE], np.uint16),
                        np.array([speed], np.uint8),
                        np.array([elevation], np.int8),
                    )

        with self.assertRaisesRegex(ValueError, "frame count"):
            motion_index_bytes(
                np.array([], np.uint16),
                np.array([], np.uint8),
                np.array([], np.int8),
            )

    def test_reader_rejects_all_header_payload_and_row_corruptions(self):
        valid = self._valid_bytes()
        header = struct.Struct("<4sIII")
        row = struct.Struct("<HBb")
        corruptions = (
            ("truncated-header", valid[:15], "truncated.*header"),
            ("bad-magic", b"BAD!" + valid[4:], "magic"),
            ("bad-version", header.pack(b"G1MI", 2, 2, 4) + valid[16:], "version"),
            ("bad-width", header.pack(b"G1MI", 1, 2, 5) + valid[16:], "row width"),
            ("zero-count", header.pack(b"G1MI", 1, 0, 4), "frame count"),
            ("truncated-payload", valid[:-1], "truncated.*payload"),
            ("trailing", valid + b"x", "trailing"),
            (
                "zero-direction",
                header.pack(b"G1MI", 1, 1, 4) + row.pack(0, SPEED_LOW, 0),
                "direction mask",
            ),
            (
                "unknown-direction",
                header.pack(b"G1MI", 1, 1, 4) + row.pack(0x200, SPEED_LOW, 0),
                "direction mask",
            ),
            (
                "opposite-direction",
                header.pack(b"G1MI", 1, 1, 4)
                + row.pack(DIRECTION_FORWARD | DIRECTION_BACKWARD, SPEED_LOW, 0),
                "direction mask",
            ),
            (
                "zero-speed",
                header.pack(b"G1MI", 1, 1, 4) + row.pack(DIRECTION_IDLE, 0, 0),
                "speed mask",
            ),
            (
                "unknown-speed",
                header.pack(b"G1MI", 1, 1, 4) + row.pack(DIRECTION_IDLE, 4, 0),
                "speed mask",
            ),
            (
                "bad-elevation",
                header.pack(b"G1MI", 1, 1, 4) + row.pack(DIRECTION_IDLE, SPEED_LOW, 2),
                "elevation",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for name, payload, message in corruptions:
                path = os.path.join(temporary, name + ".bin")
                with open(path, "wb") as stream:
                    stream.write(payload)
                with self.subTest(name=name, api="bytes"):
                    with self.assertRaisesRegex(ValueError, message):
                        motion_index_from_bytes(payload)
                with self.subTest(name=name, api="file"):
                    with self.assertRaisesRegex(ValueError, message):
                        read_motion_index(path)

    def test_reader_checks_size_arithmetic_before_payload_allocation(self):
        payload = struct.pack("<4sIII", b"G1MI", 1, 2, 4)
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "overflow.bin")
            with open(path, "wb") as stream:
                stream.write(payload)
            simulated_iinfo = mock.Mock(max=7)
            with (
                mock.patch.object(motion_index.np, "iinfo", return_value=simulated_iinfo),
                mock.patch.object(
                    motion_index.np,
                    "frombuffer",
                    side_effect=AssertionError("payload allocation attempted"),
                ),
                self.assertRaisesRegex(ValueError, "platform index limit"),
            ):
                read_motion_index(path)

    def test_writer_validates_everything_before_open_and_preserves_existing_file(self):
        sentinel = b"last-known-good-index"
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "motion-index.bin")
            with open(path, "wb") as stream:
                stream.write(sentinel)

            with self.assertRaisesRegex(ValueError, "direction mask"):
                write_motion_index(
                    path,
                    np.array([0], np.uint16),
                    np.array([SPEED_LOW], np.uint8),
                    np.array([0], np.int8),
                )
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), sentinel)

            simulated_iinfo = mock.Mock(max=3)
            with (
                mock.patch.object(motion_index.np, "iinfo", return_value=simulated_iinfo),
                self.assertRaisesRegex(ValueError, "platform index limit"),
            ):
                write_motion_index(
                    path,
                    np.array([DIRECTION_IDLE], np.uint16),
                    np.array([SPEED_LOW], np.uint8),
                    np.array([0], np.int8),
                )
            with open(path, "rb") as stream:
                self.assertEqual(stream.read(), sentinel)

        with (
            mock.patch("builtins.open") as open_mock,
            self.assertRaisesRegex(ValueError, "speed mask"),
        ):
            write_motion_index(
                "/must-not-open",
                np.array([DIRECTION_IDLE], np.uint16),
                np.array([0], np.uint8),
                np.array([0], np.int8),
            )
        open_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
