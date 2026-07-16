import json
import os
import struct
import tempfile
import unittest

from resources import audit_g1_directional_candidates as candidate_audit


def float_bits(value):
    return struct.pack(">f", value).hex()


def source(name, start, stop):
    frames = stop - start
    return {
        "name": name,
        "terrain_id": "flat" if start == 0 else name,
        "source_fps": 25.0,
        "source_frames": frames,
        "output_frames": frames,
        "range_start": start,
        "range_stop": stop,
        "source_frame_map": list(range(frames)),
    }


def manifest():
    return {
        "schema": "g1-terrain-artifacts/v2",
        "feature_dimensions": 31,
        "database_frames": 80,
        "total_clips": 2,
        "sources": [source("alpha", 0, 40), source("beta", 40, 80)],
    }


def top_entries(first_cost=0.25):
    frames = list(range(8)) + list(range(40, 48))
    return [
        {
            "frame": frame,
            "range": 0 if frame < 40 else 1,
            "cost_bits": float_bits(first_cost + index * 0.25),
        }
        for index, frame in enumerate(frames)
    ]


def record(requested_frame=0, first_cost=0.25, **changes):
    value = {
        "schema": "g1-directional-candidate-audit/v1",
        "requested_frame": requested_frame,
        "query_bits_hex": "00000000" * 31,
        "scene_id": "stairs-standard",
        "route": "ascent-landing-descent",
        "heading": "positive-x",
        "accepted_frame": 1 if requested_frame == 0 else 41,
        "eligible_count": 40 + requested_frame // 25 * 4,
        "within_best_plus_0_25": 2 + requested_frame // 25 * 2,
        "within_best_plus_1": 5 + requested_frame // 25 * 2,
        "within_best_plus_4": 12 + requested_frame // 25 * 2,
        "top_count": 16,
        "top": top_entries(first_cost),
    }
    value.update(changes)
    return value


class DirectionalCandidateAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = self.temporary.name
        self.manifest_path = os.path.join(self.root, "manifest.json")
        self.audit_path = os.path.join(self.root, "audit.jsonl")
        self.output_path = os.path.join(self.root, "summary.json")

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, value=None):
        with open(self.manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest() if value is None else value, stream)

    def write_records(self, values):
        with open(self.audit_path, "w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, separators=(",", ":")))
                stream.write("\n")

    def summarize(self, records=None, manifest_value=None):
        self.write_manifest(manifest_value)
        self.write_records(
            [record(0), record(25, 0.5)] if records is None else records)
        return candidate_audit.summarize(
            self.manifest_path, self.audit_path)

    def assert_rejected(self, records=None, manifest_value=None):
        with self.assertRaises(ValueError):
            self.summarize(records, manifest_value)

    def test_emits_canonical_per_cell_counts_costs_and_source_coverage(self):
        summary = self.summarize()
        self.assertEqual(summary["schema"],
                         "g1-directional-candidate-summary/v1")
        self.assertEqual(summary["audit_schema"],
                         "g1-directional-candidate-audit/v1")
        self.assertEqual(summary["manifest_schema"],
                         "g1-terrain-artifacts/v2")
        self.assertEqual(summary["database_frames"], 80)
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["ranges"], [
            {"range": 0, "name": "alpha", "range_start": 0,
             "range_stop": 40},
            {"range": 1, "name": "beta", "range_start": 40,
             "range_stop": 80},
        ])
        self.assertEqual(len(summary["cells"]), 1)
        cell = summary["cells"][0]
        self.assertEqual(
            (cell["scene_id"], cell["route"], cell["heading"]),
            ("stairs-standard", "ascent-landing-descent", "positive-x"))
        self.assertEqual(cell["samples"], 2)
        self.assertEqual(cell["candidate_counts"], {
            "eligible": {"minimum": 40, "median": 42.0},
            "within_best_plus_0_25": {"minimum": 2, "median": 3.0},
            "within_best_plus_1": {"minimum": 5, "median": 6.0},
            "within_best_plus_4": {"minimum": 12, "median": 13.0},
        })
        self.assertEqual(cell["top_costs"]["bits"],
                         [float_bits(0.25), float_bits(0.5)])
        self.assertEqual(cell["top_costs"]["minimum_bits"],
                         float_bits(0.25))
        self.assertEqual(cell["top_costs"]["minimum"], 0.25)
        self.assertEqual(cell["top_costs"]["median"], 0.375)
        self.assertEqual(cell["top_costs"]["top16_bits"], [
            [entry["cost_bits"] for entry in top_entries(0.25)],
            [entry["cost_bits"] for entry in top_entries(0.5)],
        ])
        self.assertEqual(cell["top16_source_coverage"], {
            "unique_ranges": [0, 1],
            "unique_source_count": 2,
            "source_names": ["alpha", "beta"],
        })

        candidate_audit.write_summary(
            self.manifest_path, self.audit_path, self.output_path)
        with open(self.output_path, "rb") as stream:
            actual = stream.read()
        expected = (
            json.dumps(
                summary,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":")) + "\n").encode("ascii")
        self.assertEqual(actual, expected)

    def test_orders_multiple_cells_canonically(self):
        second = record(
            0, scene_id="grail-curb-low", route="curb-forward",
            heading="backward")
        first = record(
            0, scene_id="grail-curb-low", route="curb-forward",
            heading="forward")
        # Input order is already canonical by (scene, route, heading, frame).
        summary = self.summarize([second, first])
        self.assertEqual(
            [cell["heading"] for cell in summary["cells"]],
            ["backward", "forward"])

    def test_accepts_explicit_relative_nonroute_cell(self):
        value = record(
            heading="relative", route="scene-cycle",
            scene_id="scene-cycle-fixture")
        summary = self.summarize([value])
        self.assertEqual(
            (summary["cells"][0]["scene_id"],
             summary["cells"][0]["route"],
             summary["cells"][0]["heading"]),
            ("scene-cycle-fixture", "scene-cycle", "relative"))

    def test_scene_cycle_accepts_runtime_order_and_sorts_summary_cells(self):
        values = [
            record(
                0, heading="relative", route="scene-cycle",
                scene_id="ramp-10-up-down"),
            record(
                25, heading="relative", route="scene-cycle",
                scene_id="grail-curb-high"),
            record(
                50, heading="relative", route="scene-cycle",
                scene_id="ramp-10-up-down"),
        ]
        summary = self.summarize(values)
        self.assertEqual(
            [(cell["scene_id"], cell["samples"])
             for cell in summary["cells"]],
            [("grail-curb-high", 1), ("ramp-10-up-down", 2)])

        reordered = [values[1], values[0]]
        self.assert_rejected(reordered)

        mixed = [values[0], record(25)]
        self.assert_rejected(mixed)

    def test_rejects_manifest_range_and_name_mutations(self):
        value = manifest()
        value["sources"][1]["range_start"] = 41
        self.assert_rejected(manifest_value=value)

        value = manifest()
        value["sources"][0]["range_stop"] = 39
        self.assert_rejected(manifest_value=value)

        value = manifest()
        value["sources"][1]["name"] = "alpha"
        self.assert_rejected(manifest_value=value)

        value = manifest()
        value["database_frames"] = 79
        self.assert_rejected(manifest_value=value)

        value = manifest()
        value["sources"][0]["source_fps"] = -25.0
        self.assert_rejected(manifest_value=value)

        value = manifest()
        value["sources"][1]["source_frame_map"][-1] = 40
        self.assert_rejected(manifest_value=value)

    def test_rejects_wrong_runtime_range_mapping(self):
        value = record()
        value["top"][8]["range"] = 0
        self.assert_rejected([value])

        value = record()
        value["top"][0]["frame"] = 80
        self.assert_rejected([value])

        value = record(0, accepted_frame=80)
        self.assert_rejected([value])

    def test_rejects_duplicate_or_unsorted_records(self):
        self.assert_rejected([record(0), record(0)])
        self.assert_rejected([record(25), record(0)])

        first = record(0)
        second = record(0, heading="forward")
        # backward sorts before forward; reversing that cell order is invalid.
        first["heading"] = "forward"
        second["heading"] = "backward"
        self.assert_rejected([first, second])

    def test_rejects_malformed_record_shapes_and_json_duplicates(self):
        value = record()
        value["unexpected"] = 1
        self.assert_rejected([value])

        value = record()
        value.pop("query_bits_hex")
        self.assert_rejected([value])

        value = record()
        value["top_count"] = 15
        self.assert_rejected([value])

        self.write_manifest()
        raw = json.dumps(record(), separators=(",", ":"))
        raw = raw[:-1] + ',"requested_frame":0}'
        with open(self.audit_path, "w", encoding="utf-8") as stream:
            stream.write(raw + "\n")
        with self.assertRaises(ValueError):
            candidate_audit.summarize(self.manifest_path, self.audit_path)

    def test_rejects_nonfinite_or_noncanonical_bit_words(self):
        value = record()
        value["query_bits_hex"] = "7f800000" + "00000000" * 30
        self.assert_rejected([value])

        value = record()
        value["top"][0]["cost_bits"] = "7f800000"
        self.assert_rejected([value])

        value = record()
        value["top"][0]["cost_bits"] = "3E800000"
        self.assert_rejected([value])

        value = record()
        value["query_bits_hex"] = "00000000" * 30 + "0000000G"
        self.assert_rejected([value])

    def test_rejects_duplicate_or_unstable_top_entries(self):
        value = record()
        value["top"][1]["frame"] = value["top"][0]["frame"]
        value["top"][1]["range"] = value["top"][0]["range"]
        self.assert_rejected([value])

        value = record()
        value["top"][0]["cost_bits"] = float_bits(0.5)
        value["top"][1]["cost_bits"] = float_bits(0.5)
        value["top"][0]["frame"] = 1
        value["top"][1]["frame"] = 0
        self.assert_rejected([value])

        value = record()
        value["top"][1]["cost_bits"] = float_bits(0.125)
        self.assert_rejected([value])

    def test_rejects_invalid_counts_headings_and_integer_booleans(self):
        value = record(within_best_plus_1=1)
        self.assert_rejected([value])

        value = record(eligible_count=15)
        self.assert_rejected([value])

        value = record(heading="Positive-X")
        self.assert_rejected([value])

        value = record(requested_frame=True)
        self.assert_rejected([value])

    def test_cli_requires_absolute_paths_and_writes_the_same_summary(self):
        self.write_manifest()
        self.write_records([record(0), record(25, 0.5)])
        self.assertEqual(candidate_audit.main([
            "--manifest", self.manifest_path,
            "--audit", self.audit_path,
            "--output", self.output_path,
        ]), 0)
        with open(self.output_path, encoding="utf-8") as stream:
            from_cli = json.load(stream)
        self.assertEqual(
            from_cli,
            candidate_audit.summarize(self.manifest_path, self.audit_path))

        with self.assertRaises(ValueError):
            candidate_audit.main([
                "--manifest", "relative.json",
                "--audit", self.audit_path,
                "--output", self.output_path,
            ])


if __name__ == "__main__":
    unittest.main()
