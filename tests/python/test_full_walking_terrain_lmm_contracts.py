from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mm_sonic.full_walking_terrain_lmm_contracts import (
    FullWalkingInventory,
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    build_split_ledger,
    connected_split_groups,
    load_lane,
    publish_lane_exclusive,
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
    return FullWalkingInventory(
        build_id="fixture-build",
        sources=sources,
        manifest_sha256="a" * 64,
    )


def _artifacts() -> ArtifactSet:
    rows = 4
    positions = np.zeros((rows, 31, 3), dtype="<f4")
    rotations = np.zeros((rows, 31, 4), dtype="<f4")
    rotations[..., 0] = 1.0
    return ArtifactSet(
        positions=positions,
        velocities=np.zeros_like(positions),
        rotations=rotations,
        angular_velocities=np.zeros_like(positions),
        parents=np.asarray(G1_SKELETON_PARENTS, dtype="<i4"),
        range_starts=np.asarray([0, 2], dtype="<i4"),
        range_stops=np.asarray([2, 4], dtype="<i4"),
        contacts=np.zeros((rows, 2), dtype=np.uint8),
        terrain_features=np.zeros((rows, 4), dtype="<f4"),
        terrain_support=np.zeros((rows, 3), dtype="<f4"),
    )


def _lane() -> LaneArtifact:
    ranges = (
        RangeRecord(
            range_id="range-a",
            canonical_source_id="canonical-a",
            terrain_id="terrain-a",
            mirror_of=None,
            family="flat",
            split_group_id="group-a",
            split="train",
            start=0,
            stop=2,
            quality="clean",
            authority={"inventory_manifest_sha256": "a" * 64},
        ),
        RangeRecord(
            range_id="range-b",
            canonical_source_id="canonical-b",
            terrain_id="terrain-b",
            mirror_of=None,
            family="slope",
            split_group_id="group-b",
            split="validation",
            start=2,
            stop=4,
            quality="usable",
            authority={"inventory_manifest_sha256": "a" * 64},
        ),
    )
    return LaneArtifact(
        root=Path("."),
        manifest_sha256="a" * 64,
        artifacts=_artifacts(),
        terrain_grid=np.zeros((4, 36), dtype="<f4"),
        ranges=ranges,
    )


class FullWalkingTerrainLmmContractTests(unittest.TestCase):
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

        first = build_split_ledger(inventory, build_id="build-a", seed=17)
        second = build_split_ledger(inventory, build_id="build-a", seed=17)

        self.assertEqual(first, second)
        by_family = {
            family: [
                assignment.split
                for assignment in first
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
            first,
            build_split_ledger(inventory, build_id="build-a", seed=18),
        )

    def test_split_ledger_rejects_empty_required_holdout_stratum(self):
        sources = tuple(_source(f"flat-{index}") for index in range(3)) + (
            _source("curb-only", family="curb"),
        )

        with self.assertRaisesRegex(ValueError, "curb.*validation.*test"):
            build_split_ledger(_inventory(sources), build_id="build-a", seed=17)

    def test_lane_publication_is_exclusive_and_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"

            publish_lane_exclusive(_lane(), first)
            publish_lane_exclusive(_lane(), second)

            first_members = {
                path.relative_to(first): path.read_bytes()
                for path in first.iterdir()
            }
            second_members = {
                path.relative_to(second): path.read_bytes()
                for path in second.iterdir()
            }
            self.assertEqual(first_members, second_members)
            self.assertEqual(
                set(first_members),
                {
                    Path("database.bin"),
                    Path("terrain_grid.npy"),
                    Path("ranges.json"),
                    Path("manifest.json"),
                },
            )
            with self.assertRaises(FileExistsError):
                publish_lane_exclusive(_lane(), first)

    def test_lane_loader_rejects_schema_keys_and_member_tampering(self):
        mutations = {
            "schema": lambda value: value.__setitem__("schema", "wrong/v1"),
            "extra key": lambda value: value.__setitem__("unexpected", True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "lane"
                publish_lane_exclusive(_lane(), output)
                manifest = json.loads((output / "manifest.json").read_text())
                mutate(manifest)
                (output / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                with self.assertRaisesRegex(ValueError, "schema|keys"):
                    load_lane(output)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lane"
            publish_lane_exclusive(_lane(), output)
            with (output / "database.bin").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "database.bin.*SHA-256"):
                load_lane(output)

    def test_loaded_lane_binds_manifest_digest_and_exact_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lane"
            publish_lane_exclusive(_lane(), output)
            digest = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()

            loaded = load_lane(output, expected_manifest_sha256=digest)

            self.assertEqual(loaded.manifest_sha256, digest)
            self.assertEqual(loaded.ranges, _lane().ranges)
            np.testing.assert_array_equal(loaded.terrain_grid, _lane().terrain_grid)
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                load_lane(output, expected_manifest_sha256="f" * 64)


if __name__ == "__main__":
    unittest.main()
