from __future__ import annotations

import hashlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.artifacts import RunBundle
from mm_sonic.joints import ContractError
from mm_sonic.reference import (
    ReferenceDiagnostics,
    format_f32,
    load_reference_directory,
    read_body_indexes,
    validate_reference_bundle,
    write_reference_bundle,
)
from mm_sonic.timeline import CanonicalTargetBuffer


EXPECTED_JOINT_POS_HEADER = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
    "joint_8",
    "joint_9",
    "joint_10",
    "joint_11",
    "joint_12",
    "joint_13",
    "joint_14",
    "joint_15",
    "joint_16",
    "joint_17",
    "joint_18",
    "joint_19",
    "joint_20",
    "joint_21",
    "joint_22",
    "joint_23",
    "joint_24",
    "joint_25",
    "joint_26",
    "joint_27",
    "joint_28",
)
EXPECTED_JOINT_VEL_HEADER = (
    "joint_vel_0",
    "joint_vel_1",
    "joint_vel_2",
    "joint_vel_3",
    "joint_vel_4",
    "joint_vel_5",
    "joint_vel_6",
    "joint_vel_7",
    "joint_vel_8",
    "joint_vel_9",
    "joint_vel_10",
    "joint_vel_11",
    "joint_vel_12",
    "joint_vel_13",
    "joint_vel_14",
    "joint_vel_15",
    "joint_vel_16",
    "joint_vel_17",
    "joint_vel_18",
    "joint_vel_19",
    "joint_vel_20",
    "joint_vel_21",
    "joint_vel_22",
    "joint_vel_23",
    "joint_vel_24",
    "joint_vel_25",
    "joint_vel_26",
    "joint_vel_27",
    "joint_vel_28",
)
EXPECTED_BODY_QUAT_HEADER = (
    "body_0_w",
    "body_0_x",
    "body_0_y",
    "body_0_z",
)
EXPECTED_BODY_POS_HEADER = ("body_0_x", "body_0_y", "body_0_z")
EXPECTED_ROOT_DIAGNOSTIC_HEADER = (
    "frame_index",
    "physical_pelvis_x",
    "physical_pelvis_y",
    "physical_pelvis_z",
    "virtual_root_x",
    "virtual_root_y",
    "virtual_root_z",
    "virtual_root_qw",
    "virtual_root_qx",
    "virtual_root_qy",
    "virtual_root_qz",
)


def float_bits(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)


def canonical_hash(buffer: CanonicalTargetBuffer) -> str:
    return hashlib.sha256(
        b"".join(
            (
                np.ascontiguousarray(buffer.joint_position, dtype="<f4").tobytes(),
                np.ascontiguousarray(buffer.joint_velocity, dtype="<f4").tobytes(),
                np.ascontiguousarray(buffer.body_quat_w, dtype="<f4").tobytes(),
                np.ascontiguousarray(buffer.frame_index, dtype="<i8").tobytes(),
            )
        )
    ).hexdigest()


def make_fixture(count: int = 21) -> tuple[CanonicalTargetBuffer, ReferenceDiagnostics]:
    rng = np.random.default_rng(20260716)
    joint_position = rng.uniform(-0.75, 0.75, size=(count, 29)).astype(np.float32)
    joint_velocity = rng.uniform(-4.0, 4.0, size=(count, 29)).astype(np.float32)
    special = np.array(
        [
            np.float32(0.0),
            np.float32(-0.0),
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.float32(1.2345678),
        ],
        dtype=np.float32,
    )
    joint_position.ravel()[: special.size] = special
    joint_velocity.ravel()[: special.size] = special[::-1]
    angle = np.linspace(0.0, 0.4, count, dtype=np.float64)
    body_quat = np.zeros((count, 4), dtype=np.float32)
    body_quat[:, 0] = np.cos(angle / 2.0).astype(np.float32)
    body_quat[:, 3] = np.sin(angle / 2.0).astype(np.float32)
    # Renormalize after the float32 cast to match CanonicalTargetBuffer's tolerance.
    body_quat /= np.linalg.norm(body_quat.astype(np.float64), axis=1)[:, None].astype(np.float32)
    canonical = CanonicalTargetBuffer(
        joint_position=joint_position,
        joint_velocity=joint_velocity,
        body_quat_w=body_quat,
        frame_index=np.arange(count, dtype=np.int64),
    )
    time = np.arange(count, dtype=np.float32) / np.float32(50.0)
    physical = np.stack((time, np.float32(0.8) + time, -time), axis=1)
    virtual = np.stack((np.float32(2.0) * time, np.zeros_like(time), time), axis=1)
    virtual_quat = np.zeros((count, 4), dtype=np.float32)
    virtual_quat[:, 0] = 1.0
    diagnostics = ReferenceDiagnostics(
        physical_pelvis_position=physical,
        virtual_root_position=virtual,
        virtual_root_quat_w=virtual_quat,
    )
    return canonical, diagnostics


class FloatFormatterTests(unittest.TestCase):
    def test_formatter_round_trips_registered_binary32_edge_values(self):
        values = (
            np.float32(0.0),
            np.float32(-0.0),
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            np.float32(3.4028235e38),
            np.float32(-12345.678),
        )
        for value in values:
            with self.subTest(value=value):
                decoded = np.float32(format_f32(value))
                self.assertEqual(decoded.view(np.uint32), value.view(np.uint32))
        for value in (np.float32("nan"), np.float32("inf"), np.float32("-inf")):
            with self.assertRaisesRegex(ContractError, "non-finite"):
                format_f32(value)


class OfficialReferenceTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.bundle = RunBundle.create(self.root, "stage-a", "reference-run")
        self.canonical, self.diagnostics = make_fixture()
        self.source_hash = "d" * 64

    def tearDown(self):
        self._temporary.cleanup()

    def _write(self):
        return write_reference_bundle(
            self.bundle,
            self.canonical,
            self.diagnostics,
            source_sha256=self.source_hash,
            scene_id="sonic-flat-baseline",
            route_id="direct",
        )

    def test_writes_the_exact_official_tree_headers_rows_metadata_and_newlines(self):
        result = self._write()
        reference = self.bundle.path / "reference" / "mm_sonic"
        required = {
            "joint_pos.csv",
            "joint_vel.csv",
            "body_quat.csv",
            "body_pos.csv",
            "metadata.txt",
            "info.txt",
        }
        self.assertEqual({path.name for path in reference.iterdir()}, required)
        self.assertTrue((self.bundle.path / "canonical_target.npz").is_file())
        self.assertTrue((self.bundle.path / "mm_root_diagnostic.csv").is_file())

        expected_headers = {
            "joint_pos.csv": EXPECTED_JOINT_POS_HEADER,
            "joint_vel.csv": EXPECTED_JOINT_VEL_HEADER,
            "body_quat.csv": EXPECTED_BODY_QUAT_HEADER,
            "body_pos.csv": EXPECTED_BODY_POS_HEADER,
        }
        for filename, header in expected_headers.items():
            raw = (reference / filename).read_bytes()
            self.assertTrue(raw.endswith(b"\n"), filename)
            lines = raw.decode("ascii").splitlines()
            self.assertEqual(lines[0], ",".join(header), filename)
            self.assertEqual(len(lines), self.canonical.count + 1, filename)
        diagnostic_lines = (self.bundle.path / "mm_root_diagnostic.csv").read_text("ascii").splitlines()
        self.assertEqual(
            diagnostic_lines[0], ",".join(EXPECTED_ROOT_DIAGNOSTIC_HEADER)
        )
        self.assertEqual(len(diagnostic_lines), self.canonical.count + 1)
        self.assertEqual(read_body_indexes(reference / "metadata.txt"), (0,))
        metadata = (reference / "metadata.txt").read_text("utf-8")
        self.assertIn("Body part indexes:\n[0]", metadata)
        self.assertIn(f"Total timesteps: {self.canonical.count}", metadata)
        self.assertEqual(result.canonical_target_sha256, canonical_hash(self.canonical))

    def test_every_csv_and_npz_value_is_bit_equal_to_the_full_session_buffer(self):
        result = self._write()
        loaded = load_reference_directory(result.reference_directory)
        self.assertEqual(loaded.count, self.canonical.count)
        np.testing.assert_array_equal(float_bits(loaded.joint_position), float_bits(self.canonical.joint_position))
        np.testing.assert_array_equal(float_bits(loaded.joint_velocity), float_bits(self.canonical.joint_velocity))
        np.testing.assert_array_equal(float_bits(loaded.body_quat_w), float_bits(self.canonical.body_quat_w))
        np.testing.assert_array_equal(loaded.frame_index, self.canonical.frame_index)

        body_pos = np.loadtxt(
            result.reference_directory / "body_pos.csv", delimiter=",", skiprows=1, dtype=np.float32, ndmin=2
        )
        self.assertEqual(body_pos.shape, (self.canonical.count, 3))
        np.testing.assert_array_equal(float_bits(body_pos), np.zeros_like(body_pos).view(np.uint32))
        with np.load(self.bundle.path / "canonical_target.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {"joint_position", "joint_velocity", "body_quat_w", "frame_index"})
            for name in ("joint_position", "joint_velocity", "body_quat_w"):
                np.testing.assert_array_equal(float_bits(archive[name]), float_bits(getattr(self.canonical, name)))
            np.testing.assert_array_equal(archive["frame_index"], self.canonical.frame_index)

    def test_diagnostics_never_enter_body_position_and_info_is_explicit(self):
        result = self._write()
        reference = result.reference_directory
        body_pos = np.loadtxt(reference / "body_pos.csv", delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
        self.assertFalse(np.array_equal(self.diagnostics.physical_pelvis_position, body_pos))
        self.assertTrue(np.all(body_pos == np.float32(0.0)))
        diagnostic = np.loadtxt(
            self.bundle.path / "mm_root_diagnostic.csv", delimiter=",", skiprows=1, dtype=np.float32, ndmin=2
        )
        np.testing.assert_array_equal(diagnostic[:, 0].astype(np.int64), self.canonical.frame_index)
        np.testing.assert_array_equal(float_bits(diagnostic[:, 1:4]), float_bits(self.diagnostics.physical_pelvis_position))
        np.testing.assert_array_equal(float_bits(diagnostic[:, 4:7]), float_bits(self.diagnostics.virtual_root_position))
        np.testing.assert_array_equal(float_bits(diagnostic[:, 7:11]), float_bits(self.diagnostics.virtual_root_quat_w))
        info = (reference / "info.txt").read_text("utf-8")
        for text in (
            "source_rate_hz: 25",
            "target_rate_hz: 50",
            f"frame_count: {self.canonical.count}",
            "joint_position_shape: [21, 29]",
            "body_quaternion_shape: [21, 4]",
            f"source_sha256: {self.source_hash}",
            f"canonical_target_sha256: {canonical_hash(self.canonical)}",
            "scene_id: sonic-flat-baseline",
            "route_id: direct",
            "body_position: zero and untracked",
        ):
            self.assertIn(text, info)

    def test_validation_requires_the_complete_official_reference_file_set(self):
        result = self._write()
        (result.reference_directory / "info.txt").unlink()
        with self.assertRaisesRegex(ContractError, "required.*info.txt"):
            validate_reference_bundle(
                self.bundle.path, self.canonical, self.diagnostics
            )

    def test_independent_validation_rejects_any_edited_csv(self):
        result = self._write()
        self.assertTrue(validate_reference_bundle(self.bundle.path, self.canonical, self.diagnostics))
        joint_path = result.reference_directory / "joint_pos.csv"
        lines = joint_path.read_text("ascii").splitlines()
        row = lines[1].split(",")
        original = np.float32(row[0])
        row[0] = format_f32(np.nextafter(original, np.float32("inf")))
        lines[1] = ",".join(row)
        joint_path.write_text("\n".join(lines) + "\n", encoding="ascii")
        with self.assertRaisesRegex(ContractError, "bit-equal"):
            validate_reference_bundle(self.bundle.path, self.canonical, self.diagnostics)

    def test_rejects_missing_frame_zero_wrong_diagnostics_and_nonfinite_data(self):
        without_zero = CanonicalTargetBuffer(
            joint_position=self.canonical.joint_position,
            joint_velocity=self.canonical.joint_velocity,
            body_quat_w=self.canonical.body_quat_w,
            frame_index=np.arange(1, self.canonical.count + 1, dtype=np.int64),
        )
        with self.assertRaisesRegex(ContractError, "frame zero"):
            write_reference_bundle(
                self.bundle,
                without_zero,
                self.diagnostics,
                source_sha256=self.source_hash,
                scene_id="flat",
                route_id="direct",
            )

        partial = CanonicalTargetBuffer(
            joint_position=self.canonical.joint_position[:2],
            joint_velocity=self.canonical.joint_velocity[:2],
            body_quat_w=self.canonical.body_quat_w[:2],
            frame_index=np.arange(2, dtype=np.int64),
        )
        partial_diagnostics = ReferenceDiagnostics(
            physical_pelvis_position=self.diagnostics.physical_pelvis_position[:2],
            virtual_root_position=self.diagnostics.virtual_root_position[:2],
            virtual_root_quat_w=self.diagnostics.virtual_root_quat_w[:2],
        )
        partial_bundle = RunBundle.create(self.root, "stage-a", "partial-chunk")
        with self.assertRaisesRegex(ContractError, "20-row chunks"):
            write_reference_bundle(
                partial_bundle,
                partial,
                partial_diagnostics,
                source_sha256=self.source_hash,
                scene_id="flat",
                route_id="direct",
            )

        bad_diagnostic = ReferenceDiagnostics(
            physical_pelvis_position=np.zeros((self.canonical.count - 1, 3), np.float32),
            virtual_root_position=np.zeros((self.canonical.count - 1, 3), np.float32),
            virtual_root_quat_w=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (self.canonical.count - 1, 1)),
        )
        other_bundle = RunBundle.create(self.root, "stage-a", "bad-diagnostic")
        with self.assertRaisesRegex(ContractError, "frame count"):
            write_reference_bundle(
                other_bundle,
                self.canonical,
                bad_diagnostic,
                source_sha256=self.source_hash,
                scene_id="flat",
                route_id="direct",
            )


if __name__ == "__main__":
    unittest.main()
