import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from resources.generate_g1_arm_joint_metadata import generate_header, main


G1_XML = Path(os.environ["G1_XML"])
TRACKED_HEADER = Path(__file__).resolve().parents[2] / "g1_arm_joint_metadata.h"

EXPECTED = (
    (17, (0.99026414, 0.13920102, -0.0000986868, -0.0000138722),
     (0.0, 0.0, -1.0), -3.0892, 2.6704),
    (18, (0.990268219, -0.139172031, 0.0, 0.0),
     (1.0, 0.0, 0.0), -1.5882, 2.2515),
    (19, (1.0, 0.0, 0.0, 0.0),
     (0.0, 1.0, 0.0), -2.618, 2.618),
    (20, (1.0, 0.0, 0.0, 0.0),
     (0.0, 0.0, -1.0), -1.0472, 2.0944),
    (21, (1.0, 0.0, 0.0, 0.0),
     (1.0, 0.0, 0.0), -1.97222, 1.97222),
    (22, (1.0, 0.0, 0.0, 0.0),
     (0.0, 0.0, -1.0), -1.61443, 1.61443),
    (23, (1.0, 0.0, 0.0, 0.0),
     (0.0, 1.0, 0.0), -1.61443, 1.61443),
    (24, (0.99026414, -0.13920102, 0.0000986868, -0.0000138722),
     (0.0, 0.0, -1.0), -3.0892, 2.6704),
    (25, (0.990268219, 0.139172031, 0.0, 0.0),
     (1.0, 0.0, 0.0), -2.2515, 1.5882),
    (26, (1.0, 0.0, 0.0, 0.0),
     (0.0, 1.0, 0.0), -2.618, 2.618),
    (27, (1.0, 0.0, 0.0, 0.0),
     (0.0, 0.0, -1.0), -1.0472, 2.0944),
    (28, (1.0, 0.0, 0.0, 0.0),
     (1.0, 0.0, 0.0), -1.97222, 1.97222),
    (29, (1.0, 0.0, 0.0, 0.0),
     (0.0, 0.0, -1.0), -1.61443, 1.61443),
    (30, (1.0, 0.0, 0.0, 0.0),
     (0.0, 1.0, 0.0), -1.61443, 1.61443),
)

NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
)


def descriptors(header: str):
    rows = []
    for line in header.splitlines():
        if not line.startswith("    {") or "}, {" not in line:
            continue
        values = [float(value) for value in NUMBER.findall(line)]
        if len(values) == 10:
            rows.append((
                int(values[0]), tuple(values[1:5]), tuple(values[5:8]),
                values[8], values[9],
            ))
    return tuple(rows)


class G1ArmJointMetadataTests(unittest.TestCase):
    def test_generator_is_deterministic_and_newline_terminated(self):
        first = generate_header(G1_XML)
        second = generate_header(G1_XML)
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))

    def test_generator_matches_the_tracked_header_byte_for_byte(self):
        generated = generate_header(G1_XML).encode("utf-8")
        self.assertEqual(generated, TRACKED_HEADER.read_bytes())

    def test_cli_requests_canonical_lf_newlines(self):
        output = Path("generated-arm-metadata.h")
        argv = [
            "generate_g1_arm_joint_metadata",
            "--g1-xml", str(G1_XML),
            "--output", str(output),
        ]
        with mock.patch("sys.argv", argv), mock.patch.object(
            Path, "write_text", autospec=True
        ) as write_text:
            main()

        write_text.assert_called_once()
        call = write_text.call_args
        self.assertEqual(call.args[0], output)
        self.assertEqual(call.args[1], generate_header(G1_XML))
        self.assertEqual(call.kwargs.get("encoding"), "utf-8")
        self.assertEqual(call.kwargs.get("newline"), "\n")

    def test_exact_fourteen_descriptors_and_ranges(self):
        actual = descriptors(generate_header(G1_XML))
        self.assertEqual(len(actual), 14)
        self.assertEqual(tuple(row[0] for row in actual), tuple(
            row[0] for row in EXPECTED
        ))
        for actual_row, expected_row in zip(actual, EXPECTED):
            self.assertEqual(actual_row[0], expected_row[0])
            for actual_value, expected_value in zip(
                actual_row[1], expected_row[1]
            ):
                self.assertAlmostEqual(actual_value, expected_value, places=9)
            self.assertEqual(actual_row[2:], expected_row[2:])

    def test_header_has_the_frozen_metadata_shape(self):
        header = generate_header(G1_XML)
        self.assertIn("struct HingeJoint {", header)
        self.assertEqual(header.count("std::array<HingeJoint, 7>"), 2)
        self.assertEqual(header.count("kLeftArm"), 1)
        self.assertEqual(header.count("kRightArm"), 1)

    def test_missing_named_arm_joint_is_rejected(self):
        source = G1_XML.read_text(encoding="utf-8")
        source = source.replace(
            'name="left_elbow_joint"', 'name="renamed_elbow_joint"', 1
        )
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "g1.xml"
            invalid.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "left_elbow_joint"):
                generate_header(invalid)


if __name__ == "__main__":
    unittest.main()
