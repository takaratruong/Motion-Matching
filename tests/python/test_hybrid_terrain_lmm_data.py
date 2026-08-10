from __future__ import annotations

import hashlib
import inspect
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mm_sonic.hybrid_terrain_lmm_data import (
    FAMILY_NAMES,
    _assemble_hybrid_corpus,
    _authenticated_primary_manifest,
    _canonical_json_bytes,
    _load_synthetic_hybrid_cache_for_tests,
    _publish_hybrid_cache,
    _sha256,
    _validate_source_manifest,
    load_hybrid_cache,
)

from resources.g1_terrain_builder.artifacts import (
    write_support_sidecar,
)
from resources.g1_terrain_builder.database import write_holden_database
from resources.g1_terrain_builder.features import (
    _GROUPS,
    _NORMALIZATION_WEIGHTS,
    _normalize,
)
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
    ArtifactSet,
)


def _synthetic_artifacts() -> ArtifactSet:
    lengths = (30, 30, 30, 30, 30, 30)
    starts = np.cumsum((0,) + lengths[:-1], dtype=np.int32)
    stops = np.cumsum(lengths, dtype=np.int32)
    frames = int(stops[-1])
    positions = np.zeros((frames, 31, 3), np.float32)
    rotations = np.zeros((frames, 31, 4), np.float32)
    rotations[..., 0] = 1.0
    parents = np.asarray(
        (-1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1,
         14, 15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26,
         27, 28, 29),
        np.int32,
    )
    for range_index, (start, stop) in enumerate(zip(starts, stops)):
        local = np.arange(stop - start, dtype=np.float32)
        positions[start:stop, 0, 0] = 1000.0 * range_index + local
        positions[start:stop, 6, 1] = -0.5
        positions[start:stop, 12, 1] = -0.5
    artifacts = ArtifactSet(
        positions=positions,
        velocities=np.zeros_like(positions),
        rotations=rotations,
        angular_velocities=np.zeros_like(positions),
        parents=parents,
        range_starts=starts,
        range_stops=stops,
        contacts=np.zeros((frames, 2), np.uint8),
        terrain_features=np.tile(
            np.asarray([0.0, 0.1, 0.2, 0.3], np.float32), (frames, 1)),
        terrain_support=np.zeros((frames, 3), np.float32),
    )
    artifacts.validate()
    return artifacts


def _write_g1tf_v2(path: Path, active: np.ndarray) -> None:
    values = np.zeros((len(active), 12), np.float32)
    values[:, (1, 3, 5, 7)] = active
    path.write_bytes(
        struct.pack("<4sIII", b"G1TF", 2, len(values), 12)
        + np.ascontiguousarray(values, dtype="<f4").tobytes()
    )


def build_synthetic_hybrid_corpus(source_root=Path("/immutable/synthetic")):
    artifacts = _synthetic_artifacts()
    range_family_ids = np.asarray([0, 1, 1, 2, 2, 3], np.int16)
    return _assemble_hybrid_corpus(
        artifacts,
        range_family_ids=range_family_ids,
        source_root=source_root,
        manifest_receipt={"sha256": "a" * 64, "schema": "synthetic/v1"},
    )


class HybridTerrainLmmDataTests(unittest.TestCase):
    def test_cache_load_rejects_coherently_recanonicalized_metadata_tampering(self):
        mutations = {
            "family names": lambda value: value.__setitem__(
                "family_names", ["curb", "flat", "slope", "stair"]
            ),
            "counts": lambda value: value["counts"].__setitem__(
                "train_rows", value["counts"]["train_rows"] + 1
            ),
            "split": lambda value: value["split"].__setitem__(
                "takara", "untrusted-split-rule"
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                source.mkdir()
                corpus = build_synthetic_hybrid_corpus(source)
                write_holden_database(source / "database.bin", corpus.artifacts)
                _write_g1tf_v2(
                    source / "terrain_features.bin",
                    corpus.artifacts.terrain_features,
                )
                write_support_sidecar(
                    source / "terrain_support.bin",
                    corpus.artifacts.terrain_support,
                )
                output = Path(directory) / "cache"
                _publish_hybrid_cache(corpus, output)
                manifest = json.loads((output / "manifest.json").read_bytes())
                mutate(manifest)
                (output / "manifest.json").write_bytes(
                    _canonical_json_bytes(manifest)
                )

                with self.assertRaisesRegex(ValueError, "canonical metadata"):
                    _load_synthetic_hybrid_cache_for_tests(output)

    def test_primary_source_receipt_metadata_is_an_exact_contract(self):
        receipt = {
            "schema": "g1-hybrid-terrain-lmm-source-receipt/v1",
            "source_schema": "g1-terrain-artifacts/v3",
            "database_rows": 3_970_932,
            "source_ranges": 15_815,
            "output_fps": 25.0,
            "legacy_feature_dimensions": 39,
            "rebuilt_feature_dimensions": 31,
            "terrain_v2_selected_columns": [1, 3, 5, 7],
            "path": "/does/not/exist/manifest.json",
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
        mutations = {
            "receipt schema": lambda value: value.__setitem__("schema", "wrong/v1"),
            "database rows": lambda value: value.__setitem__("database_rows", 1),
            "source ranges": lambda value: value.__setitem__("source_ranges", 1),
            "rate": lambda value: value.__setitem__("output_fps", 60.0),
            "legacy dimensions": lambda value: value.__setitem__(
                "legacy_feature_dimensions", 31
            ),
            "rebuilt dimensions": lambda value: value.__setitem__(
                "rebuilt_feature_dimensions", 39
            ),
            "terrain columns": lambda value: value.__setitem__(
                "terrain_v2_selected_columns", [0, 2, 4, 6]
            ),
        }
        for label, mutate in mutations.items():
            changed = dict(receipt)
            mutate(changed)
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "source receipt contract"
            ):
                _authenticated_primary_manifest(
                    Path("/does/not/exist"), changed, {}
                )

    def test_default_normalization_preserves_legacy_float64_transform_bytes(self):
        raw = np.random.default_rng(20260810).normal(
            loc=0.25, scale=3.0, size=(127, 31)
        ).astype(np.float32)
        expected = np.empty_like(raw)
        for (start, stop), weight in zip(_GROUPS, _NORMALIZATION_WEIGHTS):
            group = raw[:, start:stop].astype(np.float64)
            offset64 = group.mean(axis=0)
            offset32 = offset64.astype(np.float32)
            group_std = float(np.mean(np.sqrt(np.mean(
                np.square(group - offset64), axis=0
            ))))
            scale32 = np.float32(group_std / weight)
            expected[:, start:stop] = (
                (group - offset32) / scale32
            ).astype(np.float32)

        actual = _normalize(raw)
        self.assertEqual(actual.values.tobytes(), expected.tobytes())

    def test_public_cache_loader_rejects_non_primary_source_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            corpus = build_synthetic_hybrid_corpus(source)
            write_holden_database(source / "database.bin", corpus.artifacts)
            _write_g1tf_v2(
                source / "terrain_features.bin", corpus.artifacts.terrain_features
            )
            write_support_sidecar(
                source / "terrain_support.bin", corpus.artifacts.terrain_support
            )
            output = Path(directory) / "cache"
            _publish_hybrid_cache(corpus, output)

            with self.assertRaisesRegex(ValueError, "exact primary source authority"):
                load_hybrid_cache(output)

    def test_primary_source_contract_is_exact(self):
        family_counts = (1, 1_769, 1_857, 12_188)
        boundaries = np.cumsum((0,) + family_counts)
        ranges = [
            {
                "global_start": 0,
                "global_stop": 17_432,
                "source_frame_count": 17_432,
                "source_name": "takara_walk_50hz",
                "source_start": 0,
                "source_stop": 17_432,
            }
        ] + [
            {"source_name": f"grail-{index}"}
            for index in range(1, 15_815)
        ]
        sources = [
            {
                "name": "takara_walk_50hz",
                "output_frames": 17_432,
                "range_start": 0,
                "range_stop": 17_432,
                "source_fps": 50.0,
            }
        ] + [
            {"name": f"grail-{index}"}
            for index in range(1, 15_815)
        ]
        manifest = {
            "schema": "g1-terrain-artifacts/v3",
            "database_frames": 3_970_932,
            "output_fps": 25.0,
            "feature_dimensions": 39,
            "terrain_dimensions": 12,
            "support_dimensions": 3,
            "terrain_feature_distances_m": [0.25, 0.5, 0.75, 1.0],
            "total_clips": 15_815,
            "grail_clips": 15_814,
            "database": {"schema": "holden-database/v1"},
            "sidecars": {
                "terrain_features": {
                    "schema": "G1TF/v2", "version": 2, "dimensions": 12,
                },
                "terrain_support": {
                    "schema": "G1SP/v1", "version": 1, "dimensions": 3,
                    "columns": [
                        "source_root_height_m",
                        "source_left_toe_height_m",
                        "source_right_toe_height_m",
                    ],
                },
            },
            "skeleton": {
                "names": list(G1_SKELETON_NAMES),
                "parents": list(G1_SKELETON_PARENTS),
                "signature": G1_SKELETON_SIGNATURE,
            },
            "motion_banks": {
                "schema": "g1-terrain-motion-banks/v1",
                "frame_count": 3_970_932,
                "ranges": ranges,
                "banks": [
                    {
                        "family": family,
                        "range_indices": list(range(
                            int(boundaries[index]), int(boundaries[index + 1])
                        )),
                    }
                    for index, family in enumerate(FAMILY_NAMES)
                ],
            },
            "sources": sources,
        }

        _validate_source_manifest(manifest)
        mutations = {
            "terrain distances": lambda value: value.__setitem__(
                "terrain_feature_distances_m", [0.25, 0.5, 0.75, 1.01]
            ),
            "terrain schema": lambda value: value["sidecars"][
                "terrain_features"
            ].__setitem__("schema", "G1TF/v1"),
            "support schema": lambda value: value["sidecars"][
                "terrain_support"
            ].__setitem__("schema", "G1SP/v2"),
            "family count": lambda value: value["motion_banks"]["banks"][1][
                "range_indices"
            ].pop(),
            "Takara identity": lambda value: value["motion_banks"]["ranges"][0].__setitem__(
                "source_name", "not-takara"
            ),
        }
        for label, mutate in mutations.items():
            clone = json.loads(json.dumps(manifest))
            mutate(clone)
            with self.subTest(label=label), self.assertRaises(ValueError):
                _validate_source_manifest(clone)

    def test_cache_loader_has_no_source_authentication_bypass(self):
        parameters = inspect.signature(load_hybrid_cache).parameters
        self.assertNotIn("verify_source", parameters)
        self.assertIn("expected_cache_manifest_sha256", parameters)

    def test_takara_holdout_is_a_virtual_range_boundary(self):
        corpus = build_synthetic_hybrid_corpus()

        np.testing.assert_array_equal(
            corpus.artifacts.range_starts[:3], [0, 27, 30]
        )
        np.testing.assert_array_equal(
            corpus.artifacts.range_stops[:3], [27, 30, 60]
        )
        np.testing.assert_array_equal(corpus.range_family_ids[:3], [0, 0, 1])
        self.assertTrue(np.all(corpus.train_mask[:27]))
        self.assertTrue(np.all(corpus.evaluation_mask[27:30]))

    def test_evaluation_perturbation_cannot_change_train_dynamics_or_features(self):
        baseline_artifacts = _synthetic_artifacts()
        perturbed_artifacts = _synthetic_artifacts()
        heldout = slice(27, 30)
        perturbed_artifacts.positions[heldout, 0, 0] += np.asarray(
            [100.0, 300.0, 900.0], np.float32
        )
        angle = np.asarray([0.25, 0.5, 0.75], np.float32)
        perturbed_artifacts.rotations[heldout, 0, 0] = np.cos(angle / 2.0)
        perturbed_artifacts.rotations[heldout, 0, 2] = np.sin(angle / 2.0)
        perturbed_artifacts.terrain_features[heldout] += np.float32(50.0)
        families = np.asarray([0, 1, 1, 2, 2, 3], np.int16)

        baseline = _assemble_hybrid_corpus(
            baseline_artifacts,
            families,
            Path("/immutable/baseline"),
            {"sha256": "a" * 64, "schema": "synthetic/v1"},
        )
        perturbed = _assemble_hybrid_corpus(
            perturbed_artifacts,
            families,
            Path("/immutable/perturbed"),
            {"sha256": "b" * 64, "schema": "synthetic/v1"},
        )

        np.testing.assert_array_equal(
            baseline.artifacts.velocities[:27],
            perturbed.artifacts.velocities[:27],
        )
        np.testing.assert_array_equal(
            baseline.artifacts.angular_velocities[:27],
            perturbed.artifacts.angular_velocities[:27],
        )
        np.testing.assert_array_equal(
            baseline.features.offset, perturbed.features.offset
        )
        np.testing.assert_array_equal(
            baseline.features.scale, perturbed.features.scale
        )
        np.testing.assert_array_equal(
            baseline.features.values[baseline.train_mask],
            perturbed.features.values[perturbed.train_mask],
        )
        self.assertFalse(np.array_equal(
            baseline.features.values[baseline.evaluation_mask],
            perturbed.features.values[perturbed.evaluation_mask],
        ))

        baseline_raw = (
            baseline.features.values * baseline.features.scale
            + baseline.features.offset
        )
        perturbed_raw = (
            perturbed.features.values * perturbed.features.scale
            + perturbed.features.offset
        )
        np.testing.assert_array_equal(
            baseline_raw[baseline.train_mask],
            perturbed_raw[perturbed.train_mask],
        )
        np.testing.assert_allclose(baseline_raw[26, 15:21], 0.0, atol=1e-6)

    def test_split_is_range_complete_and_features_do_not_cross_ranges(self):
        corpus = build_synthetic_hybrid_corpus()
        self.assertFalse(np.any(corpus.train_mask & corpus.evaluation_mask))
        self.assertTrue(np.all(corpus.train_mask | corpus.evaluation_mask))
        for start, stop in zip(
            corpus.artifacts.range_starts, corpus.artifacts.range_stops
        ):
            self.assertTrue(
                np.all(corpus.evaluation_mask[start:stop])
                or not np.any(corpus.evaluation_mask[start:stop])
            )

        raw = corpus.features.values * corpus.features.scale + corpus.features.offset
        for stop in corpus.artifacts.range_stops:
            np.testing.assert_allclose(raw[int(stop) - 1, 15:21], 0.0, atol=1e-5)

    def test_family_and_range_lookup_are_row_aligned(self):
        corpus = build_synthetic_hybrid_corpus()
        self.assertEqual(corpus.family_names, FAMILY_NAMES)
        self.assertEqual(corpus.family_ids.dtype, np.dtype(np.int16))
        self.assertEqual(corpus.range_ids.dtype, np.dtype(np.int32))
        for range_id, (start, stop) in enumerate(zip(
            corpus.artifacts.range_starts, corpus.artifacts.range_stops
        )):
            self.assertTrue(np.all(corpus.range_ids[start:stop] == range_id))
            self.assertTrue(
                np.all(corpus.family_ids[start:stop] == corpus.range_family_ids[range_id])
            )

    def test_cache_round_trip_references_source_and_mmaps_features(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            corpus = build_synthetic_hybrid_corpus(source)
            write_holden_database(source / "database.bin", corpus.artifacts)
            _write_g1tf_v2(
                source / "terrain_features.bin", corpus.artifacts.terrain_features
            )
            write_support_sidecar(
                source / "terrain_support.bin", corpus.artifacts.terrain_support
            )
            output = Path(directory) / "cache"
            _publish_hybrid_cache(corpus, output)
            manifest = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(
                manifest["schema"], "g1-hybrid-terrain-lmm-corpus/v2-strict"
            )
            self.assertEqual(
                manifest["normalization"]["fit_rows"], "train-only"
            )
            self.assertEqual(
                manifest["virtual_ranges"]["takara"]["source_range_count"], 1
            )
            self.assertEqual(
                manifest["virtual_ranges"]["takara"]["refined_range_count"], 2
            )
            loaded = _load_synthetic_hybrid_cache_for_tests(output)

            self.assertIsInstance(loaded.features.values, np.memmap)
            np.testing.assert_array_equal(loaded.features.values, corpus.features.values)
            np.testing.assert_array_equal(loaded.train_mask, corpus.train_mask)
            np.testing.assert_array_equal(loaded.evaluation_mask, corpus.evaluation_mask)
            self.assertEqual(loaded.source_root, corpus.source_root)
            self.assertEqual(loaded.manifest_receipt, corpus.manifest_receipt)
            self.assertFalse((output / "database.bin").exists())

    def test_cache_load_recomputes_and_byte_compares_the_split_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            corpus = build_synthetic_hybrid_corpus(source)
            write_holden_database(source / "database.bin", corpus.artifacts)
            _write_g1tf_v2(
                source / "terrain_features.bin", corpus.artifacts.terrain_features
            )
            write_support_sidecar(
                source / "terrain_support.bin", corpus.artifacts.terrain_support
            )
            output = Path(directory) / "cache"
            _publish_hybrid_cache(corpus, output)

            train = np.load(output / "train_mask.npy", mmap_mode="r+")
            evaluation = np.load(output / "evaluation_mask.npy", mmap_mode="r+")
            train[60:90] = False
            evaluation[60:90] = True
            train.flush()
            evaluation.flush()
            del train, evaluation
            manifest = json.loads((output / "manifest.json").read_bytes())
            for name in ("train_mask.npy", "evaluation_mask.npy"):
                manifest["members"][name]["sha256"] = _sha256(output / name)
            (output / "manifest.json").write_bytes(_canonical_json_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "canonical.*mask"):
                _load_synthetic_hybrid_cache_for_tests(output)

    def test_cache_publication_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            corpus = build_synthetic_hybrid_corpus(source)
            write_holden_database(source / "database.bin", corpus.artifacts)
            _write_g1tf_v2(
                source / "terrain_features.bin", corpus.artifacts.terrain_features
            )
            write_support_sidecar(
                source / "terrain_support.bin", corpus.artifacts.terrain_support
            )
            output = Path(directory) / "cache"
            _publish_hybrid_cache(corpus, output)
            with self.assertRaises(FileExistsError):
                _publish_hybrid_cache(corpus, output)

    def test_cache_manifest_digest_binds_feature_member_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            corpus = build_synthetic_hybrid_corpus(source)
            write_holden_database(source / "database.bin", corpus.artifacts)
            _write_g1tf_v2(
                source / "terrain_features.bin", corpus.artifacts.terrain_features
            )
            write_support_sidecar(
                source / "terrain_support.bin", corpus.artifacts.terrain_support
            )
            output = Path(directory) / "cache"
            _publish_hybrid_cache(corpus, output)
            first = _load_synthetic_hybrid_cache_for_tests(output)
            expected = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
            self.assertEqual(first.cache_manifest_sha256, expected)
            with self.assertRaisesRegex(ValueError, "expected cache manifest"):
                _load_synthetic_hybrid_cache_for_tests(
                    output, expected_cache_manifest_sha256="0" * 64
                )

            offset = np.load(output / "feature_offset.npy", mmap_mode="r+")
            offset[0] += np.float32(1.0)
            offset.flush()
            del offset
            manifest = json.loads((output / "manifest.json").read_bytes())
            member = manifest["members"]["feature_offset.npy"]
            member["sha256"] = _sha256(output / "feature_offset.npy")
            (output / "manifest.json").write_bytes(_canonical_json_bytes(manifest))

            changed = _load_synthetic_hybrid_cache_for_tests(output)
            self.assertNotEqual(
                changed.cache_manifest_sha256, first.cache_manifest_sha256
            )


if __name__ == "__main__":
    unittest.main()
