import os
from pathlib import Path
import subprocess
import unittest

from resources.g1_interaction_builder import build as build_module
from resources.g1_interaction_builder.artifacts import read_artifact_set
from resources.g1_interaction_builder.schema import G1_SKELETON


NUMERIC_BOUND_LIMITS = {
    "fk_max_error_m": 0.001,
    "fk_rotation_max_error_degrees": 0.1,
    "duration_max_error_s": 1.0 / 25.0,
    "quaternion_norm_max_error": 1e-4,
}


def assert_gate1_report_contract(
    test: unittest.TestCase, report: dict
) -> None:
    for name, limit in NUMERIC_BOUND_LIMITS.items():
        test.assertLessEqual(report["numeric_bounds"][name], limit)

    reviewed_codes = set(build_module.all_schema_v1_rejection_codes())
    test.assertLessEqual(set(report["rejections_by_code"]), reviewed_codes)
    for rejection in report["rejections"]:
        test.assertTrue(
            build_module.is_schema_v1_rejection(
                rejection["stage"], rejection["code"]
            ),
            rejection,
        )


class Gate1ReportContractTests(unittest.TestCase):
    @staticmethod
    def _valid_report() -> dict:
        return {
            "numeric_bounds": dict(NUMERIC_BOUND_LIMITS),
            "rejections_by_code": {"frame_count_mismatch": 1},
            "rejections": [
                {
                    "stage": "source",
                    "code": "frame_count_mismatch",
                }
            ],
        }

    def test_accepts_exact_numeric_limits_and_reviewed_rejection(self):
        assert_gate1_report_contract(self, self._valid_report())

    def test_rejects_each_numeric_bound_above_its_limit(self):
        for name, limit in NUMERIC_BOUND_LIMITS.items():
            with self.subTest(name=name):
                report = self._valid_report()
                report["numeric_bounds"][name] = limit + 1e-6
                with self.assertRaises(AssertionError):
                    assert_gate1_report_contract(self, report)

    def test_rejects_unknown_code_and_wrong_stage(self):
        unknown = self._valid_report()
        unknown["rejections_by_code"] = {"unreviewed_code": 1}
        unknown["rejections"][0]["code"] = "unreviewed_code"
        with self.assertRaises(AssertionError):
            assert_gate1_report_contract(self, unknown)

        wrong_stage = self._valid_report()
        wrong_stage["rejections_by_code"] = {"fk_error": 1}
        wrong_stage["rejections"][0]["code"] = "fk_error"
        with self.assertRaises(AssertionError):
            assert_gate1_report_contract(self, wrong_stage)


class Gate1MakefileTests(unittest.TestCase):
    def test_gate_target_uses_safe_suite_and_configurable_paths(self):
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                "make",
                "-Bn",
                "gate1-interaction",
                "GRAIL_PICKUP_ROOT=/tmp/grail source",
                "G1_XML=/tmp/g1 robot.xml",
                "G1_INTERACTION_DIR=/tmp/g1 output",
            ],
            cwd=repository,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        normalized = " ".join(output.replace("\\\n", " ").split())
        self.assertIn("build/task12/interaction_query_probe_safe", output)
        self.assertIn('G1_INTERACTION_DIR=""', output)
        self.assertIn('--source-root "/tmp/grail source"', output)
        self.assertIn('--g1-xml "/tmp/g1 robot.xml"', output)
        self.assertIn('--output "/tmp/g1 output"', output)
        self.assertIn("--allow-rejections", output)
        self.assertIn(
            'G1_INTERACTION_DIR="/tmp/g1 output" python -m unittest '
            "tests.python.test_interaction_gate1 -v",
            normalized,
        )
        self.assertIn('./interaction_probe "/tmp/g1 output" --json', output)
        self.assertNotIn("-o interaction_query_probe ", output)


class Gate1DocumentationTests(unittest.TestCase):
    def test_readme_documents_reproducible_data_replay_scope(self):
        repository = Path(__file__).resolve().parents[2]
        readme = (repository / "README.md").read_text(encoding="utf-8")
        required = (
            "G1 tabletop interaction data replay",
            "make gate1-interaction",
            "interaction_database.bin",
            "interaction_features.bin",
            "manifest.json",
            "evaluation_split.json",
            "validation_report.json",
            "Pending full-corpus run",
            "Gate 1 replays data only; it does not yet make the character "
            "pick up an object.",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)


class ControllerDependencyMakefileTests(unittest.TestCase):
    def _dry_run(self, *targets: str) -> str:
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            ["make", "-B", "-j4", "-n", *targets],
            cwd=repository,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_linux_controller_serializes_one_pinned_bootstrap(self):
        direct = self._dry_run("controller")
        bootstrap = "./scripts/bootstrap_raylib.sh"
        compile_command = "g++ -std=c++17 -o controller"
        self.assertEqual(direct.count(bootstrap), 1)
        self.assertLess(direct.index(bootstrap), direct.index(compile_command))

        parallel = self._dry_run("bootstrap-raylib", "controller")
        self.assertEqual(parallel.count(bootstrap), 1)
        self.assertLess(
            parallel.index(bootstrap), parallel.index(compile_command)
        )

    def test_playable_gate_keeps_sequential_recursive_builds(self):
        makefile = (
            Path(__file__).resolve().parents[2] / "Makefile"
        ).read_text(encoding="utf-8")
        recipe = makefile.split("gate-playable-interaction:", 1)[1]
        self.assertEqual(recipe.count("$(MAKE) bootstrap-raylib"), 1)
        self.assertEqual(recipe.count("$(MAKE) controller"), 1)
        self.assertLess(
            recipe.index("$(MAKE) bootstrap-raylib"),
            recipe.index("$(MAKE) controller"),
        )

    def test_windows_and_web_controller_paths_do_not_gain_linux_bootstrap(self):
        bootstrap = "./scripts/bootstrap_raylib.sh"
        windows = self._dry_run("controller", "OS=Windows_NT")
        self.assertNotIn(bootstrap, windows)
        self.assertIn("-o controller.exe", windows)

        web = self._dry_run("controller", "PLATFORM=PLATFORM_WEB")
        self.assertNotIn(bootstrap, web)
        self.assertIn("emcc -std=c++17 -o controller.html", web)


@unittest.skipUnless(
    os.environ.get("G1_INTERACTION_DIR"),
    "G1_INTERACTION_DIR not set",
)
class Gate1PackTests(unittest.TestCase):
    def test_full_corpus_pack_satisfies_gate1_contract(self):
        output = Path(os.environ["G1_INTERACTION_DIR"])
        artifact, features, manifest, split, report = read_artifact_set(
            output
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["source_clips"], 2991)
        self.assertIsNone(manifest["diagnostic_limit"])
        self.assertEqual(manifest["target_fps"], 25.0)
        self.assertEqual(
            manifest["skeleton_signature"], G1_SKELETON.signature()
        )
        self.assertEqual(features.values.shape[1], 71)
        self.assertEqual(
            features.values.shape[0], artifact.positions.shape[0]
        )
        self.assertEqual(len(split["heldout_objects"]), 20)
        self.assertFalse(
            set(split["heldout_objects"])
            & set(split["database_objects"])
        )
        self.assertEqual(
            report["included_clips"] + report["rejected_clips"],
            2991,
        )
        self.assertEqual(manifest["source_clips"], report["source_clips"])
        self.assertEqual(
            manifest["included_clips"], report["included_clips"]
        )
        self.assertEqual(
            manifest["rejected_clips"], report["rejected_clips"]
        )
        self.assertGreater(report["included_clips"], 0)
        assert_gate1_report_contract(self, report)
