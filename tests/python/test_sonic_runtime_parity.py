import contextlib
import csv
import io
import struct
import tempfile
import unittest
from pathlib import Path

from mm_sonic import runtime_parity


FIXTURE = Path("tests/fixtures/sonic/g1_runtime_flat_64.csv")
COST_COLUMNS = (
    "incumbent_cost",
    "selected_cost",
    "selected_terrain_error",
    "continuation_cost",
)


def float32_offset(text, ulps):
    bits = int.from_bytes(struct.pack(">f", float(text)), "big")
    return format(
        struct.unpack(">f", (bits + ulps).to_bytes(4, "big"))[0],
        ".9g",
    )


class RuntimeParityTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        with FIXTURE.open("r", encoding="ascii", newline="") as stream:
            self.rows = list(csv.reader(stream, strict=True))
        self.header = self.rows[0]

    def tearDown(self):
        self._temporary.cleanup()

    def _write(self, name, rows=None):
        path = self.root / name
        with path.open("w", encoding="ascii", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerows(self.rows if rows is None else rows)
        return path

    def _copy_rows(self):
        return [row.copy() for row in self.rows]

    def _column(self, name):
        return self.header.index(name)

    def test_exact_fixture_is_accepted_with_zero_cost_drift(self):
        summary = runtime_parity.compare_runtime_logs(FIXTURE, FIXTURE)

        self.assertEqual(summary.row_count, 64)
        self.assertEqual(summary.exact_column_count, 100)
        self.assertEqual(
            summary.cost_max_ulps,
            {name: 0 for name in COST_COLUMNS},
        )

    def test_each_registered_cost_accepts_exactly_eight_ulps(self):
        for column in COST_COLUMNS:
            with self.subTest(column=column):
                rows = self._copy_rows()
                index = self._column(column)
                rows[1][index] = float32_offset(rows[1][index], 8)
                actual = self._write(f"{column}.csv", rows)

                summary = runtime_parity.compare_runtime_logs(
                    FIXTURE,
                    actual,
                )

                self.assertEqual(summary.cost_max_ulps[column], 8)

    def test_registered_cost_rejects_nine_ulps(self):
        rows = self._copy_rows()
        index = self._column("selected_cost")
        rows[1][index] = float32_offset(rows[1][index], 9)
        actual = self._write("nine-ulps.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"selected_cost.*9 ULP.*limit 8",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_non_cost_field_drift_is_rejected_exactly(self):
        rows = self._copy_rows()
        index = self._column("selected_database_frame")
        rows[4][index] = str(int(rows[4][index]) + 1)
        actual = self._write("behavior-drift.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"row 4.*selected_database_frame.*exact mismatch",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_canonical_schema_is_required_even_when_both_files_match(self):
        rows = self._copy_rows()
        index = self._column("continuation_cost")
        for row in rows:
            row.pop(index)
        expected = self._write("bad-schema-expected.csv", rows)
        actual = self._write("bad-schema-actual.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"header.*canonical schema",
        ):
            runtime_parity.compare_runtime_logs(expected, actual)

    def test_reordered_header_is_rejected(self):
        rows = self._copy_rows()
        rows[0][0], rows[0][1] = rows[0][1], rows[0][0]
        actual = self._write("reordered-header.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"header.*canonical schema",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_malformed_row_is_rejected(self):
        rows = self._copy_rows()
        rows[9].pop()
        actual = self._write("malformed-row.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"row 9.*104 fields.*103",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_nonfinite_registered_cost_is_rejected(self):
        for value in ("nan", "inf", "-inf", "1e1000"):
            with self.subTest(value=value):
                rows = self._copy_rows()
                rows[2][self._column("incumbent_cost")] = value
                actual = self._write(
                    f"nonfinite-{value.replace('-', 'negative-')}.csv",
                    rows,
                )

                with self.assertRaisesRegex(
                    runtime_parity.RuntimeParityError,
                    r"incumbent_cost.*finite binary32",
                ):
                    runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_missing_data_row_is_rejected(self):
        actual = self._write("missing-row.csv", self.rows[:-1])

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"actual.*exactly 64 data rows.*63",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_extra_data_row_is_rejected(self):
        rows = self._copy_rows()
        rows.append(rows[-1].copy())
        actual = self._write("extra-row.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"actual.*exactly 64 data rows.*65",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_reordered_data_rows_are_rejected(self):
        rows = self._copy_rows()
        rows[1], rows[2] = rows[2], rows[1]
        actual = self._write("reordered-rows.csv", rows)

        with self.assertRaisesRegex(
            runtime_parity.RuntimeParityError,
            r"row 1.*frame.*exact mismatch",
        ):
            runtime_parity.compare_runtime_logs(FIXTURE, actual)

    def test_cli_reports_observed_per_cost_maxima(self):
        rows = self._copy_rows()
        for ulps, column in enumerate(COST_COLUMNS, start=1):
            index = self._column(column)
            rows[ulps][index] = float32_offset(rows[ulps][index], ulps)
        actual = self._write("reported-maxima.csv", rows)
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = runtime_parity.main((str(FIXTURE), str(actual)))

        self.assertEqual(status, 0)
        output = stdout.getvalue()
        self.assertIn("runtime parity passed: 64 rows", output)
        for ulps, column in enumerate(COST_COLUMNS, start=1):
            self.assertIn(f"{column}: max {ulps} ULP", output)


if __name__ == "__main__":
    unittest.main()
