from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from mm_sonic.full_walking_terrain_lmm_contracts import (
    FullWalkingInventory,
    LaneArtifact,
    RangeRecord,
    SourceRecord,
    build_split_ledger,
    inventory_manifest_bytes,
    publish_lane_exclusive,
    split_ledger_bytes,
)
from mm_sonic.full_walking_terrain_lmm_corpus import (
    FAMILIES,
    FullWalkingCorpus,
    _BuildWorkspace,
    assemble_full_corpus,
    load_full_corpus,
    reproduce_full_corpus,
    verify_full_corpus,
)

from resources.g1_terrain_builder.artifacts import canonical_json_bytes, sha256_file
from resources.g1_terrain_builder.database import (
    combine_clips,
    refresh_lmm_clip_dynamics,
)
from resources.g1_terrain_builder.scenes import (
    COORDINATE_SIGNATURE,
    SceneDefinition,
    SceneRoute,
    build_scene,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    HoldenClip,
    SkeletonSpec,
)
from resources.g1_terrain_builder.terrain import (
    FlatTerrain,
    surface_semantics_signature,
)

FPS = 60.0
ROWS_PER_RANGE = 8
SCENE_IDS = (
    "flat-standard",
    "grail-curb-default",
    "ramp-10-up-down",
    "stairs-standard",
)


def _digest_array(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype=np.dtype(dtype)).tobytes(order="C")
    ).hexdigest()


def _inventory() -> FullWalkingInventory:
    sources: list[SourceRecord] = []
    for family in FAMILIES:
        base = f"{family}:base"
        sources.extend(
            (
                SourceRecord(
                    base,
                    f"{family}:canonical-base",
                    f"{family}:terrain-base",
                    None,
                    family,
                    {"kind": "fixture", "identity": base},
                ),
                SourceRecord(
                    f"{family}:mirror",
                    f"{family}:canonical-base",
                    f"{family}:terrain-mirror",
                    base,
                    family,
                    {"kind": "fixture", "identity": f"{family}:mirror"},
                ),
                SourceRecord(
                    f"{family}:one",
                    f"{family}:canonical-one",
                    f"{family}:terrain-one",
                    None,
                    family,
                    {"kind": "fixture", "identity": f"{family}:one"},
                ),
                SourceRecord(
                    f"{family}:two",
                    f"{family}:canonical-two",
                    f"{family}:terrain-two",
                    None,
                    family,
                    {"kind": "fixture", "identity": f"{family}:two"},
                ),
            )
        )
    sources_tuple = tuple(sorted(sources, key=lambda value: value.source_id))
    build_id = hashlib.sha256(b"full-walking-corpus-fixture/v1").hexdigest()
    provisional = FullWalkingInventory(build_id, sources_tuple, "")
    digest = hashlib.sha256(inventory_manifest_bytes(provisional)).hexdigest()
    return replace(provisional, manifest_sha256=digest)


def _clip(source_index: int, family_index: int, *, suffix: str = "") -> HoldenClip:
    rows = ROWS_PER_RANGE
    time = np.arange(rows, dtype=np.float32) / np.float32(FPS)
    positions = np.zeros((rows, 31, 3), dtype=np.float32)
    positions[:, 0, 0] = np.float32(0.15 + source_index * 0.01) * time
    positions[:, 0, 2] = np.float32(0.25 + family_index * 0.02) * time
    positions[:, 1, 1] = np.float32(0.72)
    positions[:, 6, :] = np.asarray([0.08, -0.72, 0.05], np.float32)
    positions[:, 12, :] = np.asarray([-0.08, -0.72, 0.05], np.float32)
    positions[:, 7, 2] = np.float32(0.08)
    positions[:, 13, 2] = np.float32(0.08)
    rotations = np.zeros((rows, 31, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    terrain_features = np.column_stack(
        [time + np.float32(family_index * 0.1 + column * 0.03) for column in range(4)]
    ).astype(np.float32)
    mapping = np.arange(rows, dtype=np.int32)
    source = HoldenClip(
        name=f"fixture-{source_index}{suffix}",
        positions=positions,
        velocities=np.zeros_like(positions),
        rotations=rotations,
        angular_velocities=np.zeros_like(positions),
        contacts=np.zeros((rows, 2), dtype=np.uint8),
        terrain_features=terrain_features,
        terrain_support=np.zeros((rows, 3), dtype=np.float32),
        source_frames=mapping.copy(),
        terrain_id=f"fixture-{family_index}",
        source_left_indices=mapping.copy(),
        source_right_indices=mapping.copy(),
        source_alpha=np.zeros(rows, dtype=np.float32),
    )
    return refresh_lmm_clip_dynamics(
        source,
        SkeletonSpec(
            G1_SKELETON_NAMES,
            np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
        ),
        FPS,
    )


def _terrain_grid(features: np.ndarray) -> np.ndarray:
    grid = np.zeros((len(features), 36), dtype=np.float32)
    for lane_offset in (0, 12, 24):
        grid[:, np.asarray((1, 3, 5, 7)) + lane_offset] = features
    return grid


def _publish_lanes(
    root: Path, inventory: FullWalkingInventory, ledger
) -> tuple[Path, ...]:
    assignment_by_source = {
        source_id: assignment
        for assignment in ledger.assignments
        for source_id in assignment.source_ids
    }
    source_by_id = {source.source_id: source for source in inventory.sources}
    outputs: list[Path] = []
    for family_index, family in enumerate(FAMILIES):
        source_names = sorted(
            source.source_id for source in inventory.sources if source.family == family
        )
        clips: list[HoldenClip] = []
        ranges: list[RangeRecord] = []
        range_source_ids: list[str] = []
        grids: list[np.ndarray] = []
        left: list[np.ndarray] = []
        right: list[np.ndarray] = []
        alpha: list[np.ndarray] = []
        cursor = 0
        for source_index, source_id in enumerate(reversed(source_names)):
            source = source_by_id[source_id]
            assignment = assignment_by_source[source_id]
            clip = _clip(source_index, family_index)
            stop = cursor + len(clip.positions)
            source_map = {
                "schema": "range-local-resample-map/v1",
                "source_fps": FPS,
                "rows": len(clip.positions),
                "left_sha256": _digest_array(clip.source_left_indices, "<i4"),
                "right_sha256": _digest_array(clip.source_right_indices, "<i4"),
                "alpha_sha256": _digest_array(clip.source_alpha, "<f4"),
            }
            ranges.append(
                RangeRecord(
                    range_id=f"{family}:range:{source_id}",
                    canonical_source_id=source.canonical_source_id,
                    terrain_id=source.terrain_id,
                    mirror_of=source.mirror_of,
                    family=family,
                    split_group_id=assignment.split_group_id,
                    split=assignment.split,
                    start=cursor,
                    stop=stop,
                    quality="clean" if source_index % 2 == 0 else "usable",
                    authority={
                        "kind": "fixture",
                        "source_frame_count": ROWS_PER_RANGE,
                        "source_map": source_map,
                    },
                )
            )
            clips.append(clip)
            range_source_ids.append(source_id)
            grids.append(_terrain_grid(clip.terrain_features))
            left.append(clip.source_left_indices)
            right.append(clip.source_right_indices)
            alpha.append(clip.source_alpha)
            cursor = stop

        # A quarantined interval proves clean+usable views exclude it without
        # emptying any source/split stratum.
        quarantine_source = source_by_id[source_names[0]]
        quarantine_assignment = assignment_by_source[quarantine_source.source_id]
        quarantine = _clip(len(source_names), family_index, suffix="-quarantine")
        stop = cursor + len(quarantine.positions)
        ranges.append(
            RangeRecord(
                range_id=f"{family}:range:quarantine",
                canonical_source_id=quarantine_source.canonical_source_id,
                terrain_id=quarantine_source.terrain_id,
                mirror_of=quarantine_source.mirror_of,
                family=family,
                split_group_id=quarantine_assignment.split_group_id,
                split=quarantine_assignment.split,
                start=cursor,
                stop=stop,
                quality="quarantined",
                authority={
                    "kind": "fixture",
                    "source_frame_count": ROWS_PER_RANGE,
                    "source_map": {
                        "schema": "range-local-resample-map/v1",
                        "source_fps": FPS,
                        "rows": len(quarantine.positions),
                        "left_sha256": _digest_array(
                            quarantine.source_left_indices, "<i4"
                        ),
                        "right_sha256": _digest_array(
                            quarantine.source_right_indices, "<i4"
                        ),
                        "alpha_sha256": _digest_array(quarantine.source_alpha, "<f4"),
                    },
                },
            )
        )
        clips.append(quarantine)
        range_source_ids.append(quarantine_source.source_id)
        grids.append(_terrain_grid(quarantine.terrain_features))
        left.append(quarantine.source_left_indices)
        right.append(quarantine.source_right_indices)
        alpha.append(quarantine.source_alpha)

        artifacts = combine_clips(
            clips,
            SkeletonSpec(
                G1_SKELETON_NAMES,
                np.asarray(G1_SKELETON_PARENTS, dtype=np.int32),
            ),
        )
        artifacts.contacts = np.ascontiguousarray(artifacts.contacts, dtype=np.uint8)
        lane = LaneArtifact(
            root=Path("."),
            manifest_sha256="",
            inventory_manifest_sha256=inventory.manifest_sha256,
            split_ledger_manifest_sha256=ledger.manifest_sha256,
            artifacts=artifacts,
            terrain_grid=np.concatenate(grids).astype(np.float32),
            source_ids=tuple(range_source_ids),
            source_left_indices=np.concatenate(left).astype(np.int32),
            source_right_indices=np.concatenate(right).astype(np.int32),
            source_alpha=np.concatenate(alpha).astype(np.float32),
            ranges=tuple(ranges),
        )
        output = root / f"lane-{family}"
        publish_lane_exclusive(lane, output)
        outputs.append(output)
    return tuple(outputs)


def _flat_definition(scene_id: str) -> SceneDefinition:
    return SceneDefinition(
        scene_id=scene_id,
        label=scene_id.replace("-", " ").title(),
        provenance={
            "kind": "procedural",
            "source_ids": [],
            "parameters": {"fixture": True},
        },
        surface=FlatTerrain(),
        heightfield_bounds_xz=(-1.0, 1.0, -1.0, 3.0),
        playable_bounds_xz=(-0.5, 0.5, 0.0, 2.0),
        lookahead_bounds_xz=(-1.0, 1.0, -1.0, 3.0),
        spawn_position=(0.0, 0.0, 0.0),
        spawn_yaw_radians=0.0,
        regions={
            "certified": ({"id": "route", "bounds_xz": [-0.5, 0.5, 0, 2]},),
            "stress": (),
            "blocked": (),
        },
        routes=(
            SceneRoute(
                route_id="forward",
                waypoints_xz=((0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.0, 2.0)),
                expected_outcome="traverse",
                walkability_class=1,
                landing_hold_seconds=0.0,
            ),
        ),
        walkability=lambda x, z: int(-0.75 <= x <= 0.75 and -0.25 <= z <= 2.25),
    )


def _scene_authority(root: Path) -> Path:
    scenes_root = root / "scenes"
    scenes_root.mkdir(parents=True)
    records = []
    for scene_id in SCENE_IDS:
        scene = build_scene(_flat_definition(scene_id))
        scene_root = scenes_root / scene_id
        scene_root.mkdir()
        for name, payload in (
            ("scene.json", scene.scene_json),
            ("terrain.bin", scene.terrain_bin),
            ("terrain.obj", scene.terrain_obj),
            ("walkability.bin", scene.walkability_bin),
        ):
            (scene_root / name).write_bytes(payload)
        records.append(
            {
                "id": scene_id,
                "path": f"scenes/{scene_id}/scene.json",
                "sha256": hashlib.sha256(scene.scene_json).hexdigest(),
            }
        )
    index = {
        "schema": "g1-terrain-scene-index/v1",
        "default_scene_id": "flat-standard",
        "scene_ids": list(SCENE_IDS),
        "scenes": records,
        "coordinate_signature": COORDINATE_SIGNATURE,
        "surface_signature": surface_semantics_signature(),
    }
    (scenes_root / "index.json").write_bytes(canonical_json_bytes(index))
    return root


def _write_authorities(root: Path, inventory: FullWalkingInventory, ledger) -> None:
    (root / "inventory.json").write_bytes(inventory_manifest_bytes(inventory))
    (root / "split.json").write_bytes(split_ledger_bytes(ledger))


class FullWalkingTerrainLmmCorpusTests(unittest.TestCase):
    def _fixture(self, root: Path):
        inventory = _inventory()
        ledger = build_split_ledger(
            inventory, build_id=inventory.build_id, seed=20260810
        )
        _write_authorities(root, inventory, ledger)
        lanes = _publish_lanes(root, inventory, ledger)
        scenes = _scene_authority(root / "scene-authority")
        return inventory, ledger, lanes, scenes

    def test_build_is_deterministic_mmap_and_range_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            first = assemble_full_corpus(
                lanes,
                root / "corpus-first",
                inventory=inventory,
                split_ledger=ledger,
                scene_authority=scenes,
            )
            second = assemble_full_corpus(
                tuple(reversed(lanes)),
                root / "corpus-second",
                inventory=inventory,
                split_ledger=ledger,
                scene_authority=scenes,
            )

            first_loaded = load_full_corpus(first)
            second_loaded = load_full_corpus(second)
            self.assertIsInstance(first_loaded, FullWalkingCorpus)
            self.assertIsInstance(first_loaded.artifacts.positions, np.memmap)
            self.assertFalse(first_loaded.artifacts.positions.flags.writeable)
            self.assertEqual(
                first_loaded.manifest_sha256, second_loaded.manifest_sha256
            )
            self.assertEqual(first_loaded.fps, 60.0)
            self.assertEqual(first_loaded.horizons, (20, 40, 60))
            self.assertEqual(first_loaded.terrain_grid.shape[1], 36)
            self.assertEqual(first_loaded.features.values.shape[1], 31)

            # Stable family/canonical/mirror/range ordering is independent of
            # worker completion or CLI lane order.
            range_records = [
                value["record"]
                for value in json.loads((first / "ranges.json").read_text())
            ]
            keys = [
                (
                    FAMILIES.index(value["family"]),
                    value["canonical_source_id"],
                    value["mirror_of"] or "",
                    value["range_id"],
                )
                for value in range_records
            ]
            self.assertEqual(keys, sorted(keys))

            eligible = first_loaded.eligible_mask
            self.assertGreater(np.count_nonzero(~eligible), 0)
            self.assertTrue(np.all(~eligible[first_loaded.quality_ids == 2]))
            np.testing.assert_array_equal(
                first_loaded.train_mask
                | first_loaded.validation_mask
                | first_loaded.test_mask,
                eligible,
            )
            self.assertFalse(
                np.any(first_loaded.train_mask & first_loaded.validation_mask)
            )
            self.assertFalse(np.any(first_loaded.train_mask & first_loaded.test_mask))
            for family_id in range(len(FAMILIES)):
                family = first_loaded.family_ids == family_id
                self.assertTrue(np.any(family & first_loaded.train_mask))
                self.assertTrue(np.any(family & first_loaded.validation_mask))
                self.assertTrue(np.any(family & first_loaded.test_mask))

            # Normalization statistics are training-only. Deliberately using
            # all eligible rows gives a measurably different offset.
            raw = np.asarray(first_loaded.raw_features, dtype=np.float64)
            train_mean = raw[first_loaded.train_mask].mean(axis=0)
            all_mean = raw[first_loaded.eligible_mask].mean(axis=0)
            np.testing.assert_allclose(
                first_loaded.features.offset, train_mean, rtol=0.0, atol=2e-6
            )
            self.assertGreater(float(np.max(np.abs(train_mean - all_mean))), 1e-5)

            for start, stop in zip(
                first_loaded.artifacts.range_starts,
                first_loaded.artifacts.range_stops,
            ):
                start, stop = int(start), int(stop)
                np.testing.assert_array_equal(
                    first_loaded.successor[start : stop - 1],
                    np.arange(start + 1, stop, dtype=np.int64),
                )
                self.assertEqual(int(first_loaded.successor[stop - 1]), stop - 1)
                self.assertFalse(bool(first_loaded.successor_valid[stop - 1]))
                np.testing.assert_array_equal(
                    first_loaded.root_delta_xy[stop - 1], np.zeros(2, np.float32)
                )
                self.assertEqual(float(first_loaded.root_delta_yaw[stop - 1]), 0.0)

            self.assertEqual(
                int(first_loaded.source_row_offsets[-1]),
                len(first_loaded.eligible_rows),
            )
            np.testing.assert_array_equal(
                np.sort(first_loaded.source_rows),
                np.sort(first_loaded.eligible_rows),
            )
            self.assertEqual(
                tuple(
                    json.loads((first / "scenes/index.json").read_text())["scene_ids"]
                ),
                SCENE_IDS,
            )

    def test_assembly_rejects_coherently_rehashed_source_map_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            lane = lanes[0]
            source_map_path = lane / "source_left_indices.npy"
            values = np.load(source_map_path, allow_pickle=False)
            values[1] = values[0]
            with source_map_path.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            manifest = json.loads((lane / "manifest.json").read_text())
            descriptor = manifest["members"][source_map_path.name]
            descriptor["size_bytes"] = source_map_path.stat().st_size
            descriptor["sha256"] = sha256_file(source_map_path)
            (lane / "manifest.json").write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "source map|provenance|SHA-256"):
                assemble_full_corpus(
                    lanes,
                    root / "corpus",
                    inventory=inventory,
                    split_ledger=ledger,
                    scene_authority=scenes,
                )

    def test_assembly_requires_source_local_map_bounds_for_every_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            lane = lanes[0]
            ranges_path = lane / "ranges.json"
            ranges = json.loads(ranges_path.read_text())
            ranges[0]["authority"].pop("source_frame_count")
            ranges_path.write_bytes(canonical_json_bytes(ranges))
            manifest = json.loads((lane / "manifest.json").read_text())
            manifest["members"][ranges_path.name] = {
                "path": ranges_path.name,
                "size_bytes": ranges_path.stat().st_size,
                "sha256": sha256_file(ranges_path),
            }
            (lane / "manifest.json").write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "source frame count|source map"):
                assemble_full_corpus(
                    lanes,
                    root / "corpus",
                    inventory=inventory,
                    split_ledger=ledger,
                    scene_authority=scenes,
                )

    def test_assembly_rejects_inconsistent_four_and_thirty_six_dimensional_terrain(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            lane = lanes[0]
            grid_path = lane / "terrain_grid.npy"
            grid = np.load(grid_path, allow_pickle=False)
            grid[0, 1] += np.float32(0.5)
            with grid_path.open("wb") as stream:
                np.save(stream, grid, allow_pickle=False)
            manifest = json.loads((lane / "manifest.json").read_text())
            manifest["members"][grid_path.name] = {
                "path": grid_path.name,
                "size_bytes": grid_path.stat().st_size,
                "sha256": sha256_file(grid_path),
            }
            (lane / "manifest.json").write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "terrain.*columns|terrain.*grid"):
                assemble_full_corpus(
                    lanes,
                    root / "corpus",
                    inventory=inventory,
                    split_ledger=ledger,
                    scene_authority=scenes,
                )

    def test_verify_recomputes_features_dynamics_contacts_and_split_metadata(self):
        mutations = (
            ("raw_features.npy", (0, 0), np.float32(5.0), "raw feature"),
            ("velocities.npy", (1, 0, 0), np.float32(5.0), "velocity"),
            ("contacts.npy", (0, 0), np.uint8(1), "contact"),
            ("source_alpha.npy", (0,), np.float32(0.5), "source map|provenance"),
            ("split_ids.npy", (0,), np.uint8(2), "split"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            source = assemble_full_corpus(
                lanes,
                root / "corpus",
                inventory=inventory,
                split_ledger=ledger,
                scene_authority=scenes,
            )
            verify_full_corpus(source, root / "accepted.json")
            for name, index, value, message in mutations:
                with self.subTest(name=name):
                    target = root / f"mutated-{name.removesuffix('.npy')}"
                    shutil.copytree(source, target)
                    member = target / name
                    values = np.load(member, allow_pickle=False)
                    values[index] = value
                    with member.open("wb") as stream:
                        np.save(stream, values, allow_pickle=False)
                    manifest = json.loads((target / "manifest.json").read_text())
                    manifest["members"][name] = {
                        "path": name,
                        "size_bytes": member.stat().st_size,
                        "sha256": sha256_file(member),
                    }
                    (target / "manifest.json").write_bytes(
                        canonical_json_bytes(manifest)
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        verify_full_corpus(target, root / f"rejected-{name}.json")

    def test_publication_is_exclusive_and_reproduction_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            corpus = assemble_full_corpus(
                lanes,
                root / "corpus",
                inventory=inventory,
                split_ledger=ledger,
                scene_authority=scenes,
            )
            with self.assertRaises(FileExistsError):
                assemble_full_corpus(
                    lanes,
                    corpus,
                    inventory=inventory,
                    split_ledger=ledger,
                    scene_authority=scenes,
                )

            receipt = reproduce_full_corpus(
                corpus,
                lanes,
                root / ".reproduction",
                root / "determinism.json",
                inventory=inventory,
                split_ledger=ledger,
                scene_authority=scenes,
            )
            self.assertFalse((root / ".reproduction").exists())
            decoded = json.loads(receipt.read_text())
            self.assertEqual(decoded["status"], "accepted")
            self.assertEqual(
                decoded["reference_manifest_sha256"],
                decoded["reproduced_manifest_sha256"],
            )
            with self.assertRaises(FileExistsError):
                reproduce_full_corpus(
                    corpus,
                    lanes,
                    root / ".reproduction-again",
                    receipt,
                    inventory=inventory,
                    split_ledger=ledger,
                    scene_authority=scenes,
                )

    def test_failed_staging_validation_is_unpublished_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "corpus"
            request = "a" * 64
            first = _BuildWorkspace(output, request)
            member = first.write_bytes("payload.bin", b"durable payload")
            descriptor = {
                "path": "payload.bin",
                "size_bytes": member.stat().st_size,
                "sha256": sha256_file(member),
            }

            def reject_staging(_path: Path) -> None:
                raise ValueError("independent staging rejection")

            with self.assertRaisesRegex(ValueError, "staging rejection"):
                first.finish(
                    {"schema": "fixture/v1", "members": {"payload.bin": descriptor}},
                    validator=reject_staging,
                )
            first.close()
            self.assertFalse(output.exists())
            self.assertTrue((root / ".corpus.building/resume.json").is_file())
            self.assertFalse((root / ".corpus.building/manifest.json").exists())

            resumed = _BuildWorkspace(output, request)
            before = member.stat().st_ino
            resumed.write_bytes("payload.bin", b"durable payload")
            self.assertEqual(member.stat().st_ino, before)
            resumed.finish(
                {"schema": "fixture/v1", "members": {"payload.bin": descriptor}}
            )
            self.assertEqual((output / "payload.bin").read_bytes(), b"durable payload")

    def test_verify_rejects_coherent_name_shard_and_request_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, ledger, lanes, scenes = self._fixture(root)
            source = assemble_full_corpus(
                lanes,
                root / "corpus",
                inventory=inventory,
                split_ledger=ledger,
                scene_authority=scenes,
            )

            def mutate_names(target: Path) -> None:
                metadata_path = target / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["range_names"][0] = "coherent-but-wrong-range"
                metadata_path.write_bytes(canonical_json_bytes(metadata))

            def mutate_shards(target: Path) -> None:
                bindings_path = target / "lane-bindings.json"
                bindings = json.loads(bindings_path.read_text())
                bindings[0]["global_range_indices"] = []
                bindings_path.write_bytes(canonical_json_bytes(bindings))

            for label, relative, mutate, message in (
                ("names", "metadata.json", mutate_names, "range name|metadata"),
                (
                    "shards",
                    "lane-bindings.json",
                    mutate_shards,
                    "lane.*range|shard",
                ),
            ):
                with self.subTest(label=label):
                    target = root / f"mutated-{label}"
                    shutil.copytree(source, target)
                    mutate(target)
                    manifest = json.loads((target / "manifest.json").read_text())
                    member = target / relative
                    manifest["members"][relative] = {
                        "path": relative,
                        "size_bytes": member.stat().st_size,
                        "sha256": sha256_file(member),
                    }
                    (target / "manifest.json").write_bytes(
                        canonical_json_bytes(manifest)
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        verify_full_corpus(target, root / f"reject-{label}.json")

            target = root / "mutated-request"
            shutil.copytree(source, target)
            manifest = json.loads((target / "manifest.json").read_text())
            manifest["build_request_sha256"] = "f" * 64
            (target / "manifest.json").write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "build request|request identity"):
                verify_full_corpus(target, root / "reject-request.json")


if __name__ == "__main__":
    unittest.main()
