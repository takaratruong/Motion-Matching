from collections import Counter
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

try:
    from resources import build_g1_torch_grail_corpus as cli
except ImportError:
    cli = None


class BulkGrailCorpusCliTests(unittest.TestCase):
    def test_builds_balanced_selection_and_writes_report(self):
        self.assertIsNotNone(cli)
        candidates = tuple(
            SimpleNamespace(
                partition=partition,
                logical_name=f"{partition}-{index}",
                stem=f"{partition}-{index}",
            )
            for partition in ("curb", "stair_p1", "stair_p2")
            for index in range(3)
        )
        selected = tuple(
            candidate
            for candidate in candidates
            if not candidate.stem.endswith("-2")
        )
        resolved = tuple(SimpleNamespace() for _ in selected)
        manifest = {
            "schema": "g1-torch-terrain-corpus/v1",
            "clips": [{}] * 7,
            "accepted_clips": [{}] * 7,
            "rejected_candidates": [],
            "bulk_grail": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grail = root / "grail"
            grail.mkdir()
            (grail / "g1_mm_inventory.json").write_text(
                "{}", encoding="utf-8"
            )
            report = root / "report.json"
            output = root / "corpus"
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    cli,
                    "discover_bulk_grail_candidates",
                    return_value=candidates,
                ) as discover,
                mock.patch.object(
                    cli,
                    "select_bulk_grail_candidates",
                    return_value=selected,
                ) as select,
                mock.patch.object(
                    cli,
                    "resolve_bulk_grail_candidates",
                    return_value=resolved,
                ) as resolve,
                mock.patch.object(
                    cli,
                    "publish_expanded_corpus",
                    return_value=manifest,
                ) as publish,
                redirect_stdout(stdout),
            ):
                result = cli.main(
                    [
                        "--dataset-root",
                        str(grail),
                        "--limit-per-partition",
                        "2",
                        "--selection-seed",
                        "11",
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ]
                )
            stored = json.loads(report.read_text(encoding="utf-8"))
            summary = json.loads(stdout.getvalue())

        self.assertEqual(result, 0)
        self.assertEqual(stored, manifest)
        self.assertEqual(summary["selected"], 6)
        discover.assert_called_once_with(
            grail, ("curb", "stair_p1", "stair_p2")
        )
        select.assert_called_once_with(
            candidates, limit_per_partition=2, seed=11
        )
        resolve.assert_called_once_with(grail, selected)
        call = publish.call_args.kwargs
        self.assertEqual(call["resolved_sources"], resolved)
        metadata = call["corpus_metadata"]
        self.assertEqual(
            metadata["available_by_partition"],
            {"curb": 3, "stair_p1": 3, "stair_p2": 3},
        )
        self.assertEqual(
            metadata["selected_by_partition"],
            {"curb": 2, "stair_p1": 2, "stair_p2": 2},
        )
        self.assertEqual(metadata["selection_seed"], 11)
        self.assertEqual(metadata["limit_per_partition"], 2)

    def test_rejects_duplicate_partition_arguments(self):
        self.assertIsNotNone(cli)
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--partition",
                    "curb",
                    "--partition",
                    "curb",
                ]
            )

    def test_builds_exact_named_selection_without_balanced_sampler(self):
        self.assertIsNotNone(cli)
        candidates = tuple(
            SimpleNamespace(
                partition="stair_p1",
                logical_name=f"stair-{index}",
                stem=f"stair-{index}",
            )
            for index in range(3)
        )
        selected = (candidates[1],)
        resolved = (SimpleNamespace(),)
        manifest = {
            "schema": "g1-torch-terrain-corpus/v1",
            "clips": [{}, {}],
            "accepted_clips": [{}, {}],
            "rejected_candidates": [],
            "bulk_grail": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grail = root / "grail"
            grail.mkdir()
            (grail / "g1_mm_inventory.json").write_text(
                "{}", encoding="utf-8"
            )
            with (
                mock.patch.object(
                    cli,
                    "discover_bulk_grail_candidates",
                    return_value=candidates,
                ),
                mock.patch.object(
                    cli,
                    "select_named_bulk_grail_candidates",
                    return_value=selected,
                    create=True,
                ) as select_named,
                mock.patch.object(
                    cli, "select_bulk_grail_candidates"
                ) as select_balanced,
                mock.patch.object(
                    cli,
                    "resolve_bulk_grail_candidates",
                    return_value=resolved,
                ),
                mock.patch.object(
                    cli,
                    "publish_expanded_corpus",
                    return_value=manifest,
                ) as publish,
            ):
                result = cli.main(
                    [
                        "--dataset-root",
                        str(grail),
                        "--partition",
                        "stair_p1",
                        "--logical-name",
                        "stair-1",
                        "--output",
                        str(root / "output"),
                        "--report",
                        str(root / "report.json"),
                    ]
                )

        self.assertEqual(result, 0)
        select_named.assert_called_once_with(candidates, ("stair-1",))
        select_balanced.assert_not_called()
        metadata = publish.call_args.kwargs["corpus_metadata"]
        self.assertEqual(metadata["selection_mode"], "named")
        self.assertEqual(metadata["selected_logical_names"], ["stair-1"])

    def test_rejects_named_selection_with_balanced_limit(self):
        self.assertIsNotNone(cli)
        with self.assertRaises(SystemExit):
            cli.main(
                [
                    "--partition",
                    "stair_p1",
                    "--logical-name",
                    "stair-1",
                    "--limit-per-partition",
                    "2",
                ]
            )

    def test_partition_counter_fixture_is_balanced(self):
        values = Counter(
            namespace.partition
            for namespace in (
                SimpleNamespace(partition="curb"),
                SimpleNamespace(partition="stair_p1"),
                SimpleNamespace(partition="stair_p2"),
            )
        )
        self.assertEqual(dict(values), {"curb": 1, "stair_p1": 1, "stair_p2": 1})


if __name__ == "__main__":
    unittest.main()
