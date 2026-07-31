from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    from resources.g1_torch_terrain_builder.bulk_grail import (
        BulkGrailCandidate,
        discover_bulk_grail_candidates,
        resolve_bulk_grail_candidates,
        select_bulk_grail_candidates,
    )
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class BulkGrailCandidate:
        partition: str
        stem: str
        robot_relative_path: str
        object_relative_path: str
        usd_relative_path: str
        robot_bytes: int
        object_bytes: int | None
        usd_bytes: int
        robot_sha256: str
        object_sha256: str | None
        usd_sha256: str

    def _missing(*_args, **_kwargs):
        raise AssertionError("bulk GRAIL inventory module is missing")

    discover_bulk_grail_candidates = _missing
    resolve_bulk_grail_candidates = _missing
    select_bulk_grail_candidates = _missing

try:
    from resources.g1_torch_terrain_builder.bulk_grail import (
        select_named_bulk_grail_candidates,
    )
except ImportError:
    def select_named_bulk_grail_candidates(*_args, **_kwargs):
        raise AssertionError("named bulk GRAIL selection is missing")


_REVISION = "943946a972d5de2eb0d2ff214b236d0e43575fd7"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_inventory(
    root: Path,
    *,
    bad_revision: bool = False,
    duplicate_path: bool = False,
    omit_object_file: bool = False,
) -> None:
    entries = []
    for partition in ("curb", "stair_p1", "stair_p2"):
        stem = f"terrain_{partition}__asset__000"
        payloads = {
            "robot": f"{partition}-robot".encode(),
            "objects": f"{partition}-objects".encode(),
            "object_usd": f"{partition}-usd".encode(),
        }
        for modality, payload in payloads.items():
            directory = root / "data" / partition / modality
            directory.mkdir(parents=True, exist_ok=True)
            suffix = ".usd" if modality == "object_usd" else ".pkl"
            path = directory / f"{stem}{suffix}"
            if not (omit_object_file and partition == "curb" and modality == "objects"):
                path.write_bytes(payload)
            # Reproduce the real inventory defect: curb objects are absent.
            if partition == "curb" and modality == "objects":
                continue
            entries.append(
                {
                    "bytes": len(payload),
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(payload),
                }
            )
    if duplicate_path:
        entries.append(dict(entries[0]))
    inventory = {
        "files": entries,
        "modalities": ["object_usd", "objects", "robot"],
        "repository": {
            "id": "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL",
            "revision": "wrong" if bad_revision else _REVISION,
            "type": "dataset",
        },
        "schema": "g1-grail-terrain-inventory/v1",
    }
    (root / "g1_mm_inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )


def _candidate(partition: str, index: int) -> BulkGrailCandidate:
    stem = f"terrain_{partition}__asset_{index:03d}__000"
    return BulkGrailCandidate(
        partition=partition,
        stem=stem,
        robot_relative_path=f"data/{partition}/robot/{stem}.pkl",
        object_relative_path=f"data/{partition}/objects/{stem}.pkl",
        usd_relative_path=f"data/{partition}/object_usd/{stem}.usd",
        robot_bytes=10,
        object_bytes=None,
        usd_bytes=20,
        robot_sha256="1" * 64,
        object_sha256=None,
        usd_sha256="2" * 64,
    )


class BulkGrailDiscoveryTests(unittest.TestCase):
    def test_discovers_three_partitions_and_resolves_missing_curb_object_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root)
            candidates = discover_bulk_grail_candidates(
                root, ("curb", "stair_p1", "stair_p2")
            )
            resolved = resolve_bulk_grail_candidates(root, candidates)

        self.assertEqual(
            [candidate.partition for candidate in candidates],
            ["curb", "stair_p1", "stair_p2"],
        )
        self.assertEqual(len(resolved), 3)
        curb = resolved[0]
        self.assertEqual(curb.spec.family, "grail")
        self.assertEqual(curb.spec.source_adapter, "grail-record")
        self.assertEqual(curb.spec.terrain_adapter, "grail-usd")
        self.assertEqual(
            curb.source_sha256[
                f"geometry:{curb.spec.geometry_relative_paths[0]}"
            ],
            _sha256(b"curb-objects"),
        )
        self.assertTrue(
            all(
                source.spec.logical_name.startswith(
                    f"grail-{candidate.partition}-"
                )
                for source, candidate in zip(resolved, candidates)
            )
        )

    def test_rejects_wrong_revision_duplicate_paths_and_missing_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root, bad_revision=True)
            with self.assertRaisesRegex(ValueError, "revision"):
                discover_bulk_grail_candidates(root, ("curb",))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root, duplicate_path=True)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                discover_bulk_grail_candidates(root, ("curb",))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root, omit_object_file=True)
            with self.assertRaisesRegex(ValueError, "paired object"):
                discover_bulk_grail_candidates(root, ("curb",))

    def test_resolution_rejects_size_hash_and_symlink_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root)
            candidates = discover_bulk_grail_candidates(root, ("stair_p1",))
            robot = root / candidates[0].robot_relative_path
            robot.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size|SHA-256"):
                resolve_bulk_grail_candidates(root, candidates)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root)
            candidates = discover_bulk_grail_candidates(root, ("stair_p1",))
            usd = root / candidates[0].usd_relative_path
            outside = root / "outside"
            outside.write_bytes(usd.read_bytes())
            usd.unlink()
            usd.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "real file"):
                resolve_bulk_grail_candidates(root, candidates)

    def test_rejects_unsupported_or_duplicate_requested_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_inventory(root)
            with self.assertRaisesRegex(ValueError, "partition"):
                discover_bulk_grail_candidates(root, ("slope",))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                discover_bulk_grail_candidates(root, ("curb", "curb"))


class BulkGrailSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = tuple(
            _candidate(partition, index)
            for partition in ("curb", "stair_p1", "stair_p2")
            for index in range(17)
        )

    def test_is_balanced_deterministic_and_order_independent(self):
        first = select_bulk_grail_candidates(
            reversed(self.candidates), limit_per_partition=5, seed=7
        )
        second = select_bulk_grail_candidates(
            self.candidates, limit_per_partition=5, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(item.partition for item in first),
            {"curb": 5, "stair_p1": 5, "stair_p2": 5},
        )
        self.assertNotEqual(
            first,
            select_bulk_grail_candidates(
                self.candidates, limit_per_partition=5, seed=8
            ),
        )

    def test_none_selects_all_and_large_limit_keeps_partition(self):
        self.assertEqual(
            select_bulk_grail_candidates(
                self.candidates, limit_per_partition=None, seed=0
            ),
            self.candidates,
        )
        selected = select_bulk_grail_candidates(
            self.candidates, limit_per_partition=100, seed=0
        )
        self.assertEqual(selected, self.candidates)

    def test_rejects_nonpositive_limit_and_duplicate_candidate(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            select_bulk_grail_candidates(
                self.candidates, limit_per_partition=0, seed=0
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_bulk_grail_candidates(
                self.candidates + (self.candidates[0],),
                limit_per_partition=2,
                seed=0,
            )

    def test_named_selection_is_exact_and_requested_ordered(self):
        selected = select_named_bulk_grail_candidates(
            self.candidates,
            (
                self.candidates[2].logical_name,
                self.candidates[0].logical_name,
            ),
        )
        self.assertEqual(
            tuple(item.logical_name for item in selected),
            (
                self.candidates[2].logical_name,
                self.candidates[0].logical_name,
            ),
        )

    def test_named_selection_rejects_duplicate_and_missing_names(self):
        name = self.candidates[0].logical_name
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_named_bulk_grail_candidates(
                self.candidates, (name, name)
            )
        with self.assertRaisesRegex(ValueError, "not found"):
            select_named_bulk_grail_candidates(
                self.candidates, ("missing",)
            )


if __name__ == "__main__":
    unittest.main()
