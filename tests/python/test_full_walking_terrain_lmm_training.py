from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mm_sonic.full_walking_terrain_lmm_training import (
    FULL_WALKING_DT,
    FULL_WALKING_FPS,
    HierarchicalTrainingPools,
    evaluate_frozen_test,
    full_walking_model_identity,
    load_test_receipt,
    make_training_view,
    train_all_rows,
    train_selection,
    validate_full_walking_corpus,
)
from mm_sonic.hybrid_terrain_lmm_training import (
    HybridModelConfig,
    load_hybrid_generator,
)

from resources.g1_terrain_builder.artifacts import canonical_json_bytes
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_PARENTS,
    ArtifactSet,
    FeatureSet,
)


def _sha(character: str) -> str:
    return character * 64


def _synthetic_full_corpus(*, fps: float = 60.0) -> SimpleNamespace:
    # Four terrain classes, two canonical sources per class, and all three
    # split populations.  Rows are kept deliberately small while exercising
    # every hierarchy axis and the clean/usable eligibility boundary.
    rows_per_family = 24
    frames = 4 * rows_per_family
    bones = 31
    positions = np.zeros((frames, bones, 3), dtype=np.float32)
    positions[:, 1:, 1] = np.float32(0.04)
    positions[:, 1:, 0] = np.linspace(
        0.0, 0.12, bones - 1, dtype=np.float32
    )[None]
    velocities = np.zeros_like(positions)
    angular_velocities = np.zeros_like(positions)
    rotations = np.zeros((frames, bones, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    parents = np.asarray(G1_SKELETON_PARENTS, dtype=np.int32)
    range_starts = np.arange(0, frames, 6, dtype=np.int32)
    range_stops = range_starts + 6
    contacts = np.zeros((frames, 2), dtype=np.uint8)
    contacts[1::4, 0] = 1
    contacts[2::4, 1] = 1
    contacts[3::4] = 1
    artifacts = ArtifactSet(
        positions=positions,
        velocities=velocities,
        rotations=rotations,
        angular_velocities=angular_velocities,
        parents=parents,
        range_starts=range_starts,
        range_stops=range_stops,
        contacts=contacts,
        terrain_features=np.zeros((frames, 4), dtype=np.float32),
        terrain_support=np.zeros((frames, 3), dtype=np.float32),
    )
    raw_features = np.random.default_rng(7).normal(size=(frames, 31)).astype(
        np.float32
    )
    features = FeatureSet(
        values=raw_features.copy(),
        offset=np.zeros(31, dtype=np.float32),
        scale=np.ones(31, dtype=np.float32),
    )
    family_ids = np.repeat(np.arange(4, dtype=np.uint8), rows_per_family)
    within_family = np.tile(np.arange(rows_per_family), 4)
    canonical_source_ids = (
        family_ids.astype(np.int32) * 2 + (within_family >= 12).astype(np.int32)
    )
    source_ids = canonical_source_ids.copy()
    split_ids = np.empty(frames, dtype=np.uint8)
    train_mask = np.zeros(frames, dtype=bool)
    validation_mask = np.zeros(frames, dtype=bool)
    test_mask = np.zeros(frames, dtype=bool)
    for first in range(0, frames, rows_per_family):
        train_mask[first : first + 16] = True
        validation_mask[first + 16 : first + 20] = True
        test_mask[first + 20 : first + 24] = True
    split_ids[train_mask] = 0
    split_ids[validation_mask] = 1
    split_ids[test_mask] = 2
    eligible_mask = np.ones(frames, dtype=bool)
    # One quarantined row in every split type proves that masks alone never
    # make a row trainable/evaluable.
    eligible_mask[[0, 18, 23, 24, 42, 47, 48, 66, 71, 72, 90, 95]] = False
    train_mask &= eligible_mask
    validation_mask &= eligible_mask
    test_mask &= eligible_mask
    quality_ids = np.where(eligible_mask, within_family % 2, 2).astype(np.uint8)
    root_delta_xy = np.zeros((frames, 2), dtype=np.float32)
    speed_pattern = np.asarray(
        [0.0, 0.05, 0.2, 0.5, 0.8, 0.35], dtype=np.float32
    )
    root_delta_xy[:, 0] = np.tile(speed_pattern / np.float32(fps), frames // 6)
    turn_pattern = np.asarray(
        [-0.6, -0.2, 0.0, 0.1, 0.4, 0.8], dtype=np.float32
    )
    root_delta_yaw = np.tile(turn_pattern / np.float32(fps), frames // 6)
    successor = np.arange(frames, dtype=np.int64)
    successor_valid = np.zeros(frames, dtype=bool)
    for start, stop in zip(range_starts, range_stops):
        successor[int(start) : int(stop) - 1] = np.arange(
            int(start) + 1, int(stop), dtype=np.int64
        )
        successor_valid[int(start) : int(stop) - 1] = True
    return SimpleNamespace(
        root=Path("/synthetic/full-walking-corpus"),
        artifacts=artifacts,
        features=features,
        raw_features=raw_features,
        terrain_grid=np.zeros((frames, 36), dtype=np.float32),
        family_ids=family_ids,
        source_ids=source_ids,
        source_names=tuple(f"source-{index}" for index in range(8)),
        canonical_source_ids=canonical_source_ids,
        canonical_source_names=tuple(f"canonical-{index}" for index in range(8)),
        terrain_ids=family_ids.astype(np.int32),
        terrain_names=("flat", "curb", "slope", "stair"),
        range_ids=np.repeat(np.arange(16, dtype=np.int32), 6),
        range_names=tuple(f"range-{index}" for index in range(16)),
        split_ids=split_ids,
        quality_ids=quality_ids,
        eligible_mask=eligible_mask,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        eligible_rows=np.flatnonzero(eligible_mask).astype(np.int64),
        source_left_indices=np.arange(frames, dtype=np.int32),
        source_right_indices=np.arange(frames, dtype=np.int32),
        source_alpha=np.zeros(frames, dtype=np.float32),
        successor=successor,
        successor_valid=successor_valid,
        root_delta_xy=root_delta_xy,
        root_delta_yaw=root_delta_yaw,
        manifest_sha256=_sha("a"),
        inventory_manifest_sha256=_sha("b"),
        split_ledger_manifest_sha256=_sha("c"),
        lane_manifest_sha256=(_sha("d"), _sha("e"), _sha("f")),
        fps=fps,
        horizons=(20, 40, 60),
    )


def _config(*, fit_all_rows: bool = False) -> HybridModelConfig:
    return HybridModelConfig(
        variant="latent32",
        device="cpu",
        batch_size=64,
        post_coverage_steps=0,
        normalization_chunk_size=64,
        evaluation_chunk_size=64,
        dt=FULL_WALKING_DT,
        fit_all_rows=fit_all_rows,
        loss_profile="visual-articulation",
    )


def _easy_green_corpus() -> SimpleNamespace:
    corpus = _synthetic_full_corpus()
    artifacts = corpus.artifacts
    zero_contacts = np.zeros_like(artifacts.contacts)
    easy_artifacts = ArtifactSet(
        positions=artifacts.positions,
        velocities=artifacts.velocities,
        rotations=artifacts.rotations,
        angular_velocities=artifacts.angular_velocities,
        parents=artifacts.parents,
        range_starts=artifacts.range_starts,
        range_stops=artifacts.range_stops,
        contacts=zero_contacts,
        terrain_features=artifacts.terrain_features,
        terrain_support=artifacts.terrain_support,
    )
    easy_features = FeatureSet(
        values=np.zeros_like(corpus.features.values),
        offset=corpus.features.offset,
        scale=corpus.features.scale,
    )
    return SimpleNamespace(
        **{
            **vars(corpus),
            "artifacts": easy_artifacts,
            "features": easy_features,
            "raw_features": np.zeros_like(corpus.raw_features),
        }
    )


class FullWalkingTrainingContractTests(unittest.TestCase):
    def test_exact_60_hz_abi_and_hash_authorities_are_required(self) -> None:
        corpus = _synthetic_full_corpus()
        validated = validate_full_walking_corpus(corpus)
        self.assertEqual(validated, 96)
        self.assertEqual(FULL_WALKING_FPS, 60.0)
        self.assertEqual(FULL_WALKING_DT, 1.0 / 60.0)
        self.assertEqual(_config().architecture["compressor"], [908, 512, 512, 512, 32])
        self.assertEqual(_config().architecture["decompressor"], [63, 512, 512, 458])

        with self.assertRaisesRegex(ValueError, "60 Hz"):
            validate_full_walking_corpus(_synthetic_full_corpus(fps=25.0))
        with self.assertRaisesRegex(ValueError, "horizons"):
            validate_full_walking_corpus(
                SimpleNamespace(**{**vars(corpus), "horizons": (8, 16, 24)})
            )
        with self.assertRaisesRegex(ValueError, "split ledger"):
            validate_full_walking_corpus(
                SimpleNamespace(
                    **{**vars(corpus), "split_ledger_manifest_sha256": "bad"}
                )
            )
        with self.assertRaisesRegex(ValueError, "terrain grid"):
            validate_full_walking_corpus(
                SimpleNamespace(
                    **{
                        **vars(corpus),
                        "terrain_grid": np.zeros((96, 12), dtype=np.float32),
                    }
                )
            )

    def test_selection_view_uses_only_eligible_train_and_validation_rows(self) -> None:
        corpus = _synthetic_full_corpus()
        view = make_training_view(corpus)
        np.testing.assert_array_equal(
            view.train_mask, corpus.train_mask & corpus.eligible_mask
        )
        np.testing.assert_array_equal(
            view.evaluation_mask, corpus.validation_mask & corpus.eligible_mask
        )
        np.testing.assert_array_equal(view.fit_mask, corpus.eligible_mask)
        self.assertFalse(np.any(view.train_mask & corpus.test_mask))
        self.assertFalse(np.any(view.evaluation_mask & corpus.test_mask))
        self.assertFalse(np.any((view.train_mask | view.evaluation_mask) & ~corpus.eligible_mask))

    def test_hierarchical_sampler_balances_terrain_then_canonical_source(self) -> None:
        corpus = _synthetic_full_corpus()
        view = make_training_view(corpus)
        pools = HierarchicalTrainingPools.from_corpus(view, view.train_mask)
        first = pools.sample(64, np.random.default_rng(123))
        second = pools.sample(64, np.random.default_rng(123))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(view.train_mask[first]))

        terrain_counts = np.bincount(corpus.family_ids[first], minlength=4)
        np.testing.assert_array_equal(terrain_counts, np.full(4, 16))
        for family in range(4):
            selected = first[corpus.family_ids[first] == family]
            source_counts = np.bincount(
                corpus.canonical_source_ids[selected], minlength=8
            )
            nonzero = source_counts[source_counts > 0]
            self.assertEqual(len(nonzero), 2)
            self.assertLessEqual(int(nonzero.max() - nonzero.min()), 1)

        selected_keys = {pools.key_for_row(int(row)) for row in first}
        self.assertGreaterEqual(len({key.speed_bin for key in selected_keys}), 3)
        self.assertGreaterEqual(len({key.turn_bin for key in selected_keys}), 3)
        self.assertEqual({key.contact_bin for key in selected_keys}, {0, 1, 2, 3})
        self.assertGreaterEqual(len({key.transition_bin for key in selected_keys}), 2)

    def test_model_identity_binds_corpus_split_and_exact_dimensions(self) -> None:
        corpus = _synthetic_full_corpus()
        identity = full_walking_model_identity(
            corpus, config=_config(), stage="selection"
        )
        self.assertEqual(identity["schema"], "g1-full-walking-terrain-lmm-model/v1")
        self.assertEqual(identity["stage"], "selection")
        self.assertEqual(identity["fps"], 60.0)
        self.assertEqual(identity["horizons"], [20, 40, 60])
        self.assertEqual(identity["dimensions"], {
            "matching_features": 31,
            "compressor_input": 908,
            "decoder_input": 63,
            "target": 458,
            "latent": 32,
            "width": 512,
        })
        self.assertEqual(identity["corpus_manifest_sha256"], _sha("a"))
        self.assertEqual(identity["split_ledger_manifest_sha256"], _sha("c"))
        self.assertEqual(
            identity["sampling"]["hierarchy"],
            [
                "terrain_class",
                "canonical_source",
                "speed_bin",
                "turn_bin",
                "contact_bin",
                "transition_bin",
            ],
        )
        self.assertEqual(
            identity["sampling"]["coverage"],
            "deterministic-shuffled-every-fit-row-once",
        )
        self.assertNotIn("test_metrics", identity)

    def test_test_split_is_not_consumed_when_validation_gates_are_red(self) -> None:
        corpus = _synthetic_full_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "red-selection"
            train_selection(corpus, selection, config=_config())
            with self.assertRaisesRegex(ValueError, "validation gates"):
                evaluate_frozen_test(
                    corpus,
                    selection,
                    root / "must-not-publish.json",
                    device="cpu",
                )
            self.assertFalse((root / "must-not-publish.json").exists())

    def test_selection_test_and_refit_provenance_are_immutable(self) -> None:
        corpus = _easy_green_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection"
            selection_config = replace(_config(), post_coverage_steps=50)
            receipt = train_selection(corpus, selection, config=selection_config)
            self.assertEqual(receipt["fit_scope"], "source-held-out-selection")
            selection_manifest_payload = (selection / "manifest.json").read_bytes()
            selection_manifest = json.loads(selection_manifest_payload)
            selection_sha = hashlib.sha256(selection_manifest_payload).hexdigest()
            self.assertEqual(
                selection_manifest["schema"],
                "g1-full-walking-terrain-lmm-model/v1",
            )
            self.assertEqual(selection_manifest["artifact_identity"]["stage"], "selection")
            self.assertNotIn("test", json.dumps(selection_manifest).lower())
            validation_receipt = json.loads(
                (selection / "evaluation.json").read_bytes()
            )
            self.assertEqual(
                validation_receipt["selection_population"], "validation"
            )
            self.assertEqual(
                validation_receipt["validation"],
                validation_receipt["source_held_out"],
            )
            self.assertNotIn("test", validation_receipt)

            test_output = root / "selection-test.json"
            test_receipt = evaluate_frozen_test(
                corpus, selection, test_output, device="cpu"
            )
            self.assertEqual(
                test_receipt["selection_model_manifest_sha256"], selection_sha
            )
            self.assertEqual(
                test_receipt["test_rows"],
                int(np.count_nonzero(corpus.test_mask & corpus.eligible_mask)),
            )
            self.assertEqual(load_test_receipt(test_output), test_receipt)
            with self.assertRaises(FileExistsError):
                evaluate_frozen_test(corpus, selection, test_output, device="cpu")

            refit = root / "refit"
            refit_receipt = train_all_rows(
                corpus,
                refit,
                selection_model=selection,
                test_receipt=test_output,
                config=_config(fit_all_rows=True),
            )
            self.assertEqual(
                refit_receipt["coverage_rows"],
                int(np.count_nonzero(corpus.eligible_mask)),
            )
            refit_manifest = json.loads((refit / "manifest.json").read_bytes())
            self.assertEqual(
                refit_manifest["schema"],
                "g1-full-walking-terrain-lmm-model/v1",
            )
            self.assertEqual(refit_manifest["artifact_identity"]["stage"], "refit")
            self.assertEqual(
                refit_manifest["artifact_identity"]["test_receipt_sha256"],
                hashlib.sha256(test_output.read_bytes()).hexdigest(),
            )
            loaded_refit = load_hybrid_generator(refit, corpus=corpus, device="cpu")
            self.assertTrue(loaded_refit.refit_receipt_current)
            self.assertTrue(loaded_refit.test_receipt_current)
            test_payload = test_output.read_bytes()
            test_output.unlink()
            with self.assertRaisesRegex(ValueError, "test receipt authority"):
                load_hybrid_generator(refit, corpus=corpus, device="cpu")
            test_output.write_bytes(test_payload)

            changed = dict(test_receipt)
            changed["selection_model_manifest_sha256"] = _sha("0")
            changed_output = root / "changed-test.json"
            changed_output.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "selection model"):
                train_all_rows(
                    corpus,
                    root / "changed-refit",
                    selection_model=selection,
                    test_receipt=changed_output,
                    config=_config(fit_all_rows=True),
                )

            with self.assertRaisesRegex(ValueError, "fit_all_rows"):
                train_all_rows(
                    corpus,
                    root / "wrong-config",
                    selection_model=selection,
                    test_receipt=test_output,
                    config=_config(fit_all_rows=False),
                )


if __name__ == "__main__":
    unittest.main()
