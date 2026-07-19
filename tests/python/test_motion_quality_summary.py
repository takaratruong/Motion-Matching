import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

from resources import summarize_g1_motion_quality as quality
from resources.check_g1_runtime_log import RUNTIME_COLUMNS
from tests.python.test_runtime_log import runtime_row


BASELINE_ROUTES = (
    "stairs-shallow/flat-positive-z",
    "stairs-shallow/flat-positive-x",
    "stairs-standard/ascent-landing-descent",
    "ramp-10-up-down/up-landing-down",
    "stairs-standard/ascent-landing-descent@positive-x",
    "mixed-multilevel/tangent-level-boundary",
)


def quality_rows():
    rows = [runtime_row(frame) for frame in range(4)]
    changes = (
        {
            "source_terrain": "flat",
            "source_name": "takara_walk_50hz",
            "selected_cost": "1", "incumbent_cost": "1",
            "rendered_min_clearance": "-0.02",
            "sole_min_clearance": "-0.03",
            "left_sole_min_clearance": "-0.03",
            "left_sole_clearance_0": "-0.03",
            "right_sole_min_clearance": "0.02",
            "right_sole_clearance_0": "0.02",
            "left_contact": "1", "right_contact": "1",
            "left_stance_slip": "0", "right_stance_slip": "0",
            "left_stance_slip_reset": "1",
            "right_stance_slip_reset": "1",
        },
        {
            "source_terrain": "terrain_curbs__curb_000__000",
            "source_name": "terrain_curbs__curb_000__000",
            "searched": "1", "transitioned": "1",
            "query_database_frame": "100",
            "selected_database_frame": "200", "database_frame": "201",
            "query_range": "0", "range": "1", "source_range": "1",
            "selected_cost": "2", "incumbent_cost": "3",
            "rendered_min_clearance": "0.01",
            "sole_min_clearance": "0.02",
            "left_sole_min_clearance": "0.02",
            "left_sole_clearance_0": "0.02",
            "right_sole_min_clearance": "0.03",
            "right_sole_clearance_0": "0.03",
            "left_contact": "1", "right_contact": "1",
            "left_stance_slip": "0.01", "right_stance_slip": "0.02",
            "left_stance_slip_reset": "0",
            "right_stance_slip_reset": "0",
        },
        {
            "source_terrain": "curb",
            "source_name": "terrain_curbs__curb_000__001",
            "searched": "1", "transitioned": "0",
            "query_database_frame": "201",
            "selected_database_frame": "201", "database_frame": "202",
            "query_range": "1", "range": "1", "source_range": "1",
            "selected_cost": "3", "incumbent_cost": "3",
            "rendered_min_clearance": "0.04",
            "sole_min_clearance": "-0.01",
            "left_sole_min_clearance": "-0.01",
            "left_sole_clearance_0": "-0.01",
            "right_sole_min_clearance": "0.01",
            "right_sole_clearance_0": "0.01",
            "left_contact": "1", "right_contact": "0",
            "left_stance_slip": "0.03", "right_stance_slip": "0",
            "left_stance_slip_reset": "0",
            "right_stance_slip_reset": "1",
        },
        {
            "source_terrain": "stair",
            "source_name": "stair_p1__fixture",
            "searched": "1", "transitioned": "1",
            "query_database_frame": "202",
            "selected_database_frame": "300", "database_frame": "301",
            "query_range": "1", "range": "2", "source_range": "2",
            "selected_cost": "4", "incumbent_cost": "5",
            "rendered_min_clearance": "0.07",
            "sole_min_clearance": "0.04",
            "left_sole_min_clearance": "0.04",
            "left_sole_clearance_0": "0.04",
            "right_sole_min_clearance": "0.05",
            "right_sole_clearance_0": "0.05",
            "left_contact": "0", "right_contact": "1",
            "left_stance_slip": "0", "right_stance_slip": "0",
            "left_stance_slip_reset": "1",
            "right_stance_slip_reset": "1",
            "route_complete": "1",
        },
    )
    for item, update in zip(rows, changes):
        item.update(update)
        # Keep the aggregate minima exact after changing one ordered probe.
        for side in ("left", "right"):
            minimum = float(item[f"{side}_sole_min_clearance"])
            item[f"{side}_sole_clearance_0"] = str(minimum)
            for probe in range(1, 4):
                item[f"{side}_sole_clearance_{probe}"] = str(
                    max(0.001 * probe, minimum + probe * 0.001))
        item["sole_min_clearance"] = str(min(
            float(item["left_sole_min_clearance"]),
            float(item["right_sole_min_clearance"])))
    return [{name: item[name] for name in RUNTIME_COLUMNS} for item in rows]


class MotionQualitySummaryTests(unittest.TestCase):
    def test_checked_baseline_fixture_is_canonical_and_formula_consistent(self):
        fixture = pathlib.Path(__file__).resolve().parents[1] / (
            "fixtures/g1_motion_quality_baseline.json")
        text = fixture.read_text(encoding="utf-8")
        report = json.loads(text)
        self.assertEqual(quality.canonical_json(report), text)
        self.assertEqual(report["schema"], "g1-motion-quality-baseline/v1")
        self.assertFalse(report["ik_enabled"])
        self.assertEqual(report["sample_rate_hz"], 25)
        self.assertEqual(
            [report["routes"][route]["frames"] for route in BASELINE_ROUTES],
            [100, 100, 800, 800, 800, 800])
        self.assertTrue(all(
            report["routes"][route]["ik_enabled_frames"] == 0
            for route in BASELINE_ROUTES))

        thresholds = report["candidate_thresholds"]
        lateral = report["routes"][BASELINE_ROUTES[1]]
        self.assertEqual(
            thresholds["lateral_selected_cost_p95"]["baseline"],
            lateral["selected_cost"]["p95"])
        self.assertEqual(
            thresholds["lateral_transition_rate"]["baseline"],
            lateral["transition_rate"])
        expected_slip = round(sum(
            report["routes"][route]["stance_slip"]["total_m"][side]
            for route in BASELINE_ROUTES
            for side in ("left", "right")), 9)
        self.assertEqual(
            thresholds["stance_slip_total_m"]["baseline"], expected_slip)
        expected_penetration = max(
            report["routes"][route]["sole_clearance"]
                  ["worst_penetration_m"]
            for route in BASELINE_ROUTES)
        self.assertEqual(
            thresholds["sole_worst_penetration_m"]["baseline"],
            expected_penetration)

    def test_script_entrypoint_resolves_repository_imports(self):
        repository = pathlib.Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable,
             "resources/summarize_g1_motion_quality.py", "--help"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--log ROUTE=PATH", result.stdout)

    def test_summarizes_all_required_metrics_deterministically(self):
        summary = quality.summarize_route(quality_rows())
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["source_family_frames"], {
            "curb": 2, "flat": 1, "slope": 0, "stair": 1,
        })
        self.assertEqual(summary["search_count"], 3)
        self.assertEqual(summary["transition_count"], 2)
        self.assertEqual(summary["transition_rate"], 0.5)
        self.assertEqual(summary["selected_cost"], {
            "max": 4.0, "p50": 2.5, "p90": 3.7, "p95": 3.85,
        })
        self.assertEqual(summary["rendered_joint_clearance"], {
            "minimum_m": -0.02,
            "p05_m": -0.0155,
            "penetrating_frames": 1,
            "worst_penetration_m": 0.02,
        })
        self.assertEqual(summary["sole_clearance"], {
            "minimum_m": -0.03,
            "p05_m": -0.027,
            "penetrating_frames": 2,
            "penetrating_probes": 2,
            "worst_penetration_m": 0.03,
        })
        self.assertEqual(summary["stance_slip"], {
            "contact_frames": {"left": 3, "right": 3},
            "maximum_stance_m": {"left": 0.03, "right": 0.02},
            "total_m": {"left": 0.03, "right": 0.02},
        })
        self.assertTrue(summary["route_complete"])
        self.assertEqual(summary["ik_enabled_frames"], 0)

    def test_baseline_report_freezes_explicit_comparison_directions(self):
        logs = {route: quality_rows() for route in reversed(BASELINE_ROUTES)}
        report = quality.build_baseline_report(logs)
        self.assertEqual(report["schema"], "g1-motion-quality-baseline/v1")
        self.assertEqual(tuple(report["routes"]), tuple(sorted(BASELINE_ROUTES)))
        thresholds = report["candidate_thresholds"]
        self.assertEqual(thresholds["flat_wrong_family_frames"], {
            "comparison": "equal", "target": 0,
            "routes": list(BASELINE_ROUTES[:2]),
        })
        self.assertEqual(
            thresholds["activated_terrain_wrong_family_frames"]["target"], 0)
        for name in (
                "lateral_selected_cost_p95",
                "lateral_transition_rate",
                "stance_slip_total_m",
                "sole_worst_penetration_m"):
            self.assertEqual(
                thresholds[name]["comparison"],
                "strictly_less_than_baseline")
            self.assertIn("baseline", thresholds[name])

        first = quality.canonical_json(report)
        second = quality.canonical_json(quality.build_baseline_report(logs))
        self.assertEqual(first, second)
        self.assertTrue(first.endswith("\n"))
        self.assertNotIn("NaN", first)
        self.assertEqual(json.loads(first), report)

    def test_rejects_unknown_source_family_and_nonfinite_input(self):
        cases = (
            ("source_terrain", "mystery", "unknown source terrain family"),
            ("sole_min_clearance", "nan", "non-finite"),
            ("ik_enabled", "1", "IK must be disabled"),
        )
        for field, value, message in cases:
            rows = quality_rows()
            rows[1][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    quality.summarize_route(rows)

    def test_cli_reads_exact_runtime_logs_and_writes_canonical_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            arguments = []
            logs = {}
            for index, route in enumerate(BASELINE_ROUTES):
                path = root / f"route-{index}.csv"
                rows = quality_rows()
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=RUNTIME_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                arguments.extend(("--log", f"{route}={path}"))
                logs[route] = rows
            output = root / "baseline.json"
            arguments.extend(("--output", str(output)))
            self.assertEqual(quality.main(arguments), 0)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                quality.canonical_json(quality.build_baseline_report(logs)))


if __name__ == "__main__":
    unittest.main()
