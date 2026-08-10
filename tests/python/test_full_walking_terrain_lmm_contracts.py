from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from mm_sonic.full_walking_terrain_lmm_contracts import (
    FullWalkingInventory,
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    SplitLedger,
    build_split_ledger,
    connected_split_groups,
    inventory_manifest_bytes,
    load_lane,
    load_split_ledger,
    publish_lane_exclusive,
    split_ledger_bytes,
)

from resources.g1_terrain_builder.schema import (
    G1_SKELETON_PARENTS,
    ArtifactSet,
)


def _source(
    source_id: str,
    *,
    canonical: str | None = None,
    terrain: str | None = None,
    mirror_of: str | None = None,
    family: str = "flat",
) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        canonical_source_id=canonical or source_id,
        terrain_id=terrain or f"terrain-{source_id}",
        mirror_of=mirror_of,
        family=family,
        authority={"fixture": source_id},
    )


def _inventory(sources: tuple[SourceRecord, ...]) -> FullWalkingInventory:
    provisional = FullWalkingInventory(
        build_id="b" * 64,
        sources=sources,
        manifest_sha256="",
    )
    return FullWalkingInventory(
        build_id=provisional.build_id,
        sources=provisional.sources,
        manifest_sha256=hashlib.sha256(
            inventory_manifest_bytes(provisional)
        ).hexdigest(),
    )


def _artifacts() -> ArtifactSet:
    rows = 4
    positions = np.zeros((rows, 31, 3), dtype="<f4")
    rotations = np.zeros((rows, 31, 4), dtype="<f4")
    rotations[..., 0] = 1.0
    terrain_features = np.arange(rows * 4, dtype="<f4").reshape(rows, 4) + 0.25
    terrain_support = np.arange(rows * 3, dtype="<f4").reshape(rows, 3) - 1.5
    return ArtifactSet(
        positions=positions,
        velocities=np.zeros_like(positions),
        rotations=rotations,
        angular_velocities=np.zeros_like(positions),
        parents=np.asarray(G1_SKELETON_PARENTS, dtype="<i4"),
        range_starts=np.asarray([0, 2], dtype="<i4"),
        range_stops=np.asarray([2, 4], dtype="<i4"),
        contacts=np.zeros((rows, 2), dtype=np.uint8),
        terrain_features=terrain_features,
        terrain_support=terrain_support,
    )


def _lane_authorities() -> tuple[FullWalkingInventory, SplitLedger]:
    inventory = _inventory(tuple(_source(name) for name in ("a", "b", "c")))
    return inventory, build_split_ledger(
        inventory, build_id=inventory.build_id, seed=17
    )


def _assignment_for(ledger: SplitLedger, source_id: str):
    return next(
        assignment
        for assignment in ledger.assignments
        if source_id in assignment.source_ids
    )


def _lane(
    inventory: FullWalkingInventory | None = None,
    ledger: SplitLedger | None = None,
) -> LaneArtifact:
    if inventory is None or ledger is None:
        inventory, ledger = _lane_authorities()
    assignment_a = _assignment_for(ledger, "a")
    assignment_b = _assignment_for(ledger, "b")
    ranges = (
        RangeRecord(
            range_id="range-a",
            canonical_source_id="a",
            terrain_id="terrain-a",
            mirror_of=None,
            family="flat",
            split_group_id=assignment_a.split_group_id,
            split=assignment_a.split,
            start=0,
            stop=2,
            quality="clean",
            authority={"source_frame_count": 2},
        ),
        RangeRecord(
            range_id="range-b",
            canonical_source_id="b",
            terrain_id="terrain-b",
            mirror_of=None,
            family="flat",
            split_group_id=assignment_b.split_group_id,
            split=assignment_b.split,
            start=2,
            stop=4,
            quality="usable",
            authority={"source_frame_count": 2},
        ),
    )
    return LaneArtifact(
        root=Path("."),
        manifest_sha256="",
        inventory_manifest_sha256=inventory.manifest_sha256,
        split_ledger_manifest_sha256=ledger.manifest_sha256,
        artifacts=_artifacts(),
        terrain_grid=np.zeros((4, 36), dtype="<f4"),
        source_ids=("a", "b"),
        source_left_indices=np.asarray([0, 0, 0, 0], dtype="<i4"),
        source_right_indices=np.asarray([0, 1, 0, 1], dtype="<i4"),
        source_alpha=np.asarray([0.0, 0.5, 0.0, 0.5], dtype="<f4"),
        ranges=ranges,
    )


class FullWalkingTerrainLmmContractTests(unittest.TestCase):
    def test_package_exports_revised_authority_contracts(self):
        import mm_sonic

        self.assertIs(mm_sonic.SplitLedger, SplitLedger)
        self.assertIs(mm_sonic.load_split_ledger, load_split_ledger)

    def test_connected_groups_are_transitive_and_keep_mirrors_together(self):
        sources = (
            _source("a", canonical="canonical-a", terrain="terrain-shared"),
            _source("a-mirror", canonical="canonical-a", mirror_of="a"),
            _source("b", canonical="canonical-b", terrain="terrain-shared"),
            _source("c", canonical="canonical-b", terrain="terrain-c"),
        )

        groups = connected_split_groups(sources)

        self.assertEqual(groups, (("a", "a-mirror", "b", "c"),))

    def test_split_ledger_is_deterministic_stratified_and_seeded(self):
        sources = tuple(
            _source(f"{family}-{index}", family=family)
            for family in ("flat", "curb", "slope", "stair")
            for index in range(10)
        )
        inventory = _inventory(sources)

        first = build_split_ledger(inventory, build_id=inventory.build_id, seed=17)
        second = build_split_ledger(inventory, build_id=inventory.build_id, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(first.inventory_manifest_sha256, inventory.manifest_sha256)
        self.assertEqual(first.requested, {"train": 80, "validation": 10, "test": 10})
        by_family = {
            family: [
                assignment.split
                for assignment in first.assignments
                if assignment.source_ids[0].startswith(f"{family}-")
            ]
            for family in ("flat", "curb", "slope", "stair")
        }
        for splits in by_family.values():
            self.assertEqual(
                {name: splits.count(name) for name in set(splits)},
                {"train": 8, "validation": 1, "test": 1},
            )
        self.assertNotEqual(
            first.assignments,
            build_split_ledger(
                inventory, build_id=inventory.build_id, seed=18
            ).assignments,
        )

    def test_split_ledger_rejects_empty_required_holdout_stratum(self):
        sources = tuple(_source(f"flat-{index}") for index in range(3)) + (
            _source("curb-only", family="curb"),
        )

        with self.assertRaisesRegex(ValueError, "curb.*validation.*test"):
            inventory = _inventory(sources)
            build_split_ledger(inventory, build_id=inventory.build_id, seed=17)

    def test_split_ledger_builder_rejects_unauthenticated_inventory(self):
        inventory, _ = _lane_authorities()

        with self.assertRaisesRegex(ValueError, "inventory manifest SHA-256"):
            changed = replace(inventory, manifest_sha256="f" * 64)
            build_split_ledger(changed, build_id=changed.build_id, seed=17)

    def test_split_ledger_loader_recomputes_groups_counts_and_coverage(self):
        inventory, ledger = _lane_authorities()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split-ledger.json"
            path.write_bytes(split_ledger_bytes(ledger))

            loaded = load_split_ledger(
                path,
                inventory=inventory,
                expected_manifest_sha256=ledger.manifest_sha256,
            )

            self.assertEqual(loaded, ledger)
            changed = json.loads(path.read_text())
            changed["assignments"][0]["source_ids"] = ["missing-source"]
            path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "coverage|connected|group"):
                load_split_ledger(path, inventory=inventory)

    def test_lane_loader_recomputes_in_memory_split_authority(self):
        inventory, ledger = _lane_authorities()
        index = next(
            index
            for index, assignment in enumerate(ledger.assignments)
            if assignment.source_ids == ("c",)
        )
        assignments = list(ledger.assignments)
        assignments[index] = replace(assignments[index], source_ids=("missing",))
        provisional = replace(
            ledger, assignments=tuple(assignments), manifest_sha256=""
        )
        forged = replace(
            provisional,
            manifest_sha256=hashlib.sha256(split_ledger_bytes(provisional)).hexdigest(),
        )
        lane = replace(
            _lane(inventory, ledger),
            split_ledger_manifest_sha256=forged.manifest_sha256,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lane"
            publish_lane_exclusive(lane, output)

            with self.assertRaisesRegex(ValueError, "connected|coverage|authority"):
                load_lane(output, inventory=inventory, split_ledger=forged)

    def test_lane_publication_is_exclusive_and_byte_reproducible(self):
        inventory, ledger = _lane_authorities()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"

            publish_lane_exclusive(_lane(inventory, ledger), first)
            publish_lane_exclusive(_lane(inventory, ledger), second)

            first_members = {
                path.relative_to(first): path.read_bytes() for path in first.iterdir()
            }
            second_members = {
                path.relative_to(second): path.read_bytes() for path in second.iterdir()
            }
            self.assertEqual(first_members, second_members)
            self.assertEqual(
                set(first_members),
                {
                    Path("database.bin"),
                    Path("terrain_features.npy"),
                    Path("terrain_support.npy"),
                    Path("terrain_grid.npy"),
                    Path("source_ids.json"),
                    Path("source_left_indices.npy"),
                    Path("source_right_indices.npy"),
                    Path("source_alpha.npy"),
                    Path("ranges.json"),
                    Path("manifest.json"),
                },
            )
            with self.assertRaises(FileExistsError):
                publish_lane_exclusive(_lane(inventory, ledger), first)

    def test_lane_publication_race_never_replaces_the_winner(self):
        inventory, ledger = _lane_authorities()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lane"
            barrier = threading.Barrier(2)

            def publish():
                barrier.wait()
                try:
                    publish_lane_exclusive(_lane(inventory, ledger), output)
                    return "published"
                except FileExistsError:
                    return "exists"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = sorted(executor.map(lambda _: publish(), range(2)))

            self.assertEqual(outcomes, ["exists", "published"])
            load_lane(output, inventory=inventory, split_ledger=ledger)

    def test_lane_loader_rejects_schema_keys_and_member_tampering(self):
        inventory, ledger = _lane_authorities()
        mutations = {
            "v1 schema": lambda value: value.__setitem__(
                "schema", "g1-full-walking-terrain-lmm-lane/v1"
            ),
            "extra key": lambda value: value.__setitem__("unexpected", True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "lane"
                publish_lane_exclusive(_lane(inventory, ledger), output)
                manifest = json.loads((output / "manifest.json").read_text())
                mutate(manifest)
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                with self.assertRaisesRegex(ValueError, "schema|keys"):
                    load_lane(output, inventory=inventory, split_ledger=ledger)

        for member in (
            "database.bin",
            "terrain_features.npy",
            "terrain_support.npy",
            "source_ids.json",
            "source_alpha.npy",
        ):
            with (
                self.subTest(member=member),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "lane"
                publish_lane_exclusive(_lane(inventory, ledger), output)
                with (output / member).open("ab") as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(ValueError, f"{member}.*SHA-256"):
                    load_lane(output, inventory=inventory, split_ledger=ledger)

    def test_lane_rejects_invalid_source_provenance(self):
        inventory, ledger = _lane_authorities()
        lane = _lane(inventory, ledger)
        invalid = (
            replace(lane, source_right_indices=np.asarray([0, 2, 0, 1], dtype="<i4")),
            replace(
                lane, source_alpha=np.asarray([0.0, np.nan, 0.0, 0.5], dtype="<f4")
            ),
            replace(lane, source_ids=("a",)),
        )
        for changed in invalid:
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
                self.assertRaisesRegex(ValueError, "source|provenance|range"),
            ):
                publish_lane_exclusive(changed, Path(directory) / "lane")

    def test_loaded_lane_rejects_range_source_identity_disagreement(self):
        inventory, ledger = _lane_authorities()
        lane = replace(_lane(inventory, ledger), source_ids=("a", "a"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lane"
            publish_lane_exclusive(lane, output)

            with self.assertRaisesRegex(ValueError, "source/split"):
                load_lane(output, inventory=inventory, split_ledger=ledger)

    def test_loaded_lane_binds_manifest_digest_and_exact_ranges(self):
        inventory, ledger = _lane_authorities()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lane"
            lane = _lane(inventory, ledger)
            publish_lane_exclusive(lane, output)
            digest = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()

            loaded = load_lane(
                output,
                inventory=inventory,
                split_ledger=ledger,
                expected_manifest_sha256=digest,
                expected_inventory_manifest_sha256=inventory.manifest_sha256,
                expected_split_ledger_manifest_sha256=ledger.manifest_sha256,
            )

            self.assertEqual(loaded.manifest_sha256, digest)
            self.assertEqual(
                loaded.inventory_manifest_sha256, inventory.manifest_sha256
            )
            self.assertEqual(
                loaded.split_ledger_manifest_sha256, ledger.manifest_sha256
            )
            self.assertEqual(loaded.ranges, lane.ranges)
            self.assertEqual(loaded.source_ids, lane.source_ids)
            np.testing.assert_array_equal(
                loaded.source_left_indices, lane.source_left_indices
            )
            np.testing.assert_array_equal(
                loaded.source_right_indices, lane.source_right_indices
            )
            np.testing.assert_array_equal(loaded.source_alpha, lane.source_alpha)
            np.testing.assert_array_equal(loaded.terrain_grid, lane.terrain_grid)
            np.testing.assert_array_equal(
                loaded.artifacts.terrain_features, lane.artifacts.terrain_features
            )
            np.testing.assert_array_equal(
                loaded.artifacts.terrain_support, lane.artifacts.terrain_support
            )
            self.assertGreater(np.count_nonzero(loaded.artifacts.terrain_features), 0)
            self.assertGreater(np.count_nonzero(loaded.artifacts.terrain_support), 0)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                load_lane(
                    output,
                    inventory=inventory,
                    split_ledger=ledger,
                    expected_manifest_sha256="f" * 64,
                )
            with self.assertRaisesRegex(ValueError, "inventory manifest SHA-256"):
                load_lane(
                    output,
                    inventory=inventory,
                    split_ledger=ledger,
                    expected_inventory_manifest_sha256="f" * 64,
                )


if __name__ == "__main__":
    unittest.main()
