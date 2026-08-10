from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
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
from resources.g1_terrain_builder.features import _NORMALIZATION_WEIGHTS
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_PARENTS,
    ArtifactSet,
    FeatureSet,
)

_FEATURE_GROUPS = (
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (15, 21),
    (21, 27),
    (27, 31),
)
_DISABLED_SCALE = np.float32(np.finfo(np.float32).max)


def _sha(character: str) -> str:
    return character * 64


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _normalized_features(
    raw_features: np.ndarray, train_mask: np.ndarray
) -> FeatureSet:
    selected = np.flatnonzero(train_mask)
    offset = np.empty(31, dtype=np.float32)
    scale = np.empty(31, dtype=np.float32)
    for (start, stop), weight in zip(_FEATURE_GROUPS, _NORMALIZATION_WEIGHTS):
        train_values = np.asarray(raw_features[selected, start:stop], dtype=np.float64)
        mean = train_values.mean(axis=0, dtype=np.float64)
        group_std = float(np.mean(np.std(train_values, axis=0, dtype=np.float64)))
        offset[start:stop] = mean.astype(np.float32)
        if not np.isfinite(group_std) or group_std <= 0.0 or weight == 0.0:
            scale[start:stop] = _DISABLED_SCALE
        else:
            scale[start:stop] = np.float32(group_std / weight)
    values = np.empty_like(raw_features)
    for start, stop in _FEATURE_GROUPS:
        if np.all(scale[start:stop] == _DISABLED_SCALE):
            values[:, start:stop] = 0.0
        else:
            values[:, start:stop] = (
                (raw_features[:, start:stop] - offset[start:stop]) / scale[start:stop]
            ).astype(np.float32)
    return FeatureSet(values=values, offset=offset, scale=scale)


def _synthetic_full_corpus(*, fps: float = 60.0) -> SimpleNamespace:
    # Four terrain classes, two canonical sources per class, and all three
    # split populations.  Rows are kept deliberately small while exercising
    # every hierarchy axis and the clean/usable eligibility boundary.
    rows_per_family = 24
    frames = 4 * rows_per_family
    bones = 31
    positions = np.zeros((frames, bones, 3), dtype=np.float32)
    positions[:, 1:, 1] = np.float32(0.04)
    positions[:, 1:, 0] = np.linspace(0.0, 0.12, bones - 1, dtype=np.float32)[None]
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
    family_ids = np.repeat(np.arange(4, dtype=np.uint8), rows_per_family)
    within_family = np.tile(np.arange(rows_per_family), 4)
    split_stage = np.where(within_family < 16, 0, np.where(within_family < 20, 1, 2))
    canonical_source_ids = family_ids.astype(np.int32) * 4 + np.where(
        within_family < 8,
        0,
        np.where(within_family < 16, 1, np.where(within_family < 20, 2, 3)),
    ).astype(np.int32)
    source_ids = canonical_source_ids.copy()
    terrain_ids = family_ids.astype(np.int32) * 3 + split_stage.astype(np.int32)
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
    raw_features = np.random.default_rng(7).normal(size=(frames, 31)).astype(np.float32)
    features = _normalized_features(raw_features, train_mask)
    quality_ids = np.where(eligible_mask, within_family % 2, 2).astype(np.uint8)
    root_delta_xy = np.zeros((frames, 2), dtype=np.float32)
    speed_pattern = np.asarray([0.0, 0.05, 0.2, 0.5, 0.8, 0.35], dtype=np.float32)
    root_delta_xy[:, 0] = np.tile(speed_pattern / np.float32(fps), frames // 6)
    turn_pattern = np.asarray([-0.6, -0.2, 0.0, 0.1, 0.4, 0.8], dtype=np.float32)
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
        source_names=tuple(f"source-{index}" for index in range(16)),
        canonical_source_ids=canonical_source_ids,
        canonical_source_names=tuple(f"canonical-{index}" for index in range(16)),
        terrain_ids=terrain_ids,
        terrain_names=tuple(f"terrain-{index}" for index in range(12)),
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
    raw_features = np.zeros_like(corpus.raw_features)
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
    easy_features = _normalized_features(raw_features, corpus.train_mask)
    return SimpleNamespace(
        **{
            **vars(corpus),
            "artifacts": easy_artifacts,
            "features": easy_features,
            "raw_features": raw_features,
        }
    )


class FullWalkingTrainingContractTests(unittest.TestCase):
    def test_cuda_training_import_does_not_require_build_only_usd(self) -> None:
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            class BlockPXR(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "pxr" or fullname.startswith("pxr."):
                        raise ModuleNotFoundError("blocked build-only pxr dependency")
                    return None

            sys.meta_path.insert(0, BlockPXR())
            import mm_sonic.full_walking_terrain_lmm_corpus
            """
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (
                str(Path.cwd()),
                str(Path.cwd() / "resources"),
                str(Path.cwd() / "sonic/python"),
            )
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

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

    def test_canonical_source_and_terrain_identities_cannot_cross_splits(self) -> None:
        corpus = _synthetic_full_corpus()
        validate_full_walking_corpus(corpus)

        crossing_source = SimpleNamespace(
            **{
                **vars(corpus),
                "canonical_source_ids": np.asarray(
                    corpus.canonical_source_ids, dtype=np.int32
                ).copy(),
            }
        )
        crossing_source.canonical_source_ids[16] = crossing_source.canonical_source_ids[
            1
        ]
        with self.assertRaisesRegex(ValueError, "canonical source"):
            validate_full_walking_corpus(crossing_source)

        crossing_terrain = SimpleNamespace(
            **{
                **vars(corpus),
                "terrain_ids": np.asarray(corpus.terrain_ids, dtype=np.int32).copy(),
            }
        )
        crossing_terrain.terrain_ids[20] = crossing_terrain.terrain_ids[2]
        with self.assertRaisesRegex(ValueError, "terrain identity"):
            validate_full_walking_corpus(crossing_terrain)

    def test_normalization_is_recomputed_from_raw_features_and_eligible_train_rows(
        self,
    ) -> None:
        corpus = _synthetic_full_corpus()
        validate_full_walking_corpus(corpus)

        wrong_values = FeatureSet(
            values=np.zeros_like(corpus.features.values),
            offset=corpus.features.offset,
            scale=corpus.features.scale,
        )
        with self.assertRaisesRegex(ValueError, "normalized feature"):
            validate_full_walking_corpus(
                SimpleNamespace(**{**vars(corpus), "features": wrong_values})
            )

        wrong_offset = FeatureSet(
            values=corpus.features.values,
            offset=corpus.features.offset + np.float32(0.25),
            scale=corpus.features.scale,
        )
        with self.assertRaisesRegex(ValueError, "normalization offset"):
            validate_full_walking_corpus(
                SimpleNamespace(**{**vars(corpus), "features": wrong_offset})
            )

        wrong_scale = FeatureSet(
            values=corpus.features.values,
            offset=corpus.features.offset,
            scale=corpus.features.scale * np.float32(1.05),
        )
        with self.assertRaisesRegex(ValueError, "normalization scale"):
            validate_full_walking_corpus(
                SimpleNamespace(**{**vars(corpus), "features": wrong_scale})
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
        self.assertFalse(
            np.any((view.train_mask | view.evaluation_mask) & ~corpus.eligible_mask)
        )

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
                corpus.canonical_source_ids[selected], minlength=16
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
        self.assertEqual(
            identity["dimensions"],
            {
                "matching_features": 31,
                "compressor_input": 908,
                "decoder_input": 63,
                "target": 458,
                "latent": 32,
                "width": 512,
            },
        )
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
            corpus = SimpleNamespace(**{**vars(corpus), "root": root / "corpus"})
            corpus.root.mkdir()
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
            corpus = SimpleNamespace(**{**vars(corpus), "root": root / "corpus"})
            corpus.root.mkdir()
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
            self.assertEqual(
                selection_manifest["artifact_identity"]["stage"], "selection"
            )
            self.assertNotIn("test", json.dumps(selection_manifest).lower())
            validation_receipt = json.loads(
                (selection / "evaluation.json").read_bytes()
            )
            self.assertEqual(validation_receipt["selection_population"], "validation")
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
                config=replace(selection_config, fit_all_rows=True),
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
            swapped_test = dict(test_receipt)
            swapped_metrics = dict(test_receipt["metrics"])
            swapped_metrics["local_position_p95_m"] = 0.02
            swapped_test["metrics"] = swapped_metrics
            swapped_payload = canonical_json_bytes(swapped_test)
            original_read_bytes = Path.read_bytes
            swapped_once = False

            def swap_after_authority_scan(path: Path) -> bytes:
                nonlocal swapped_once
                resolved = path.expanduser().resolve()
                if resolved == test_output.resolve() and not swapped_once:
                    test_output.write_bytes(swapped_payload)
                    swapped_once = True
                    return test_payload
                return original_read_bytes(path)

            test_output.write_bytes(test_payload)
            with mock.patch.object(Path, "read_bytes", autospec=True) as patched_read:
                patched_read.side_effect = swap_after_authority_scan
                with self.assertRaisesRegex(
                    ValueError, "test receipt authority|changed during"
                ):
                    load_hybrid_generator(refit, corpus=corpus, device="cpu")
            test_output.write_bytes(test_payload)
            changed_validation = dict(test_receipt)
            changed_validation["selection_validation_receipt_sha256"] = _sha("1")
            changed_validation_output = root / "changed-validation-test.json"
            changed_validation_output.write_bytes(
                canonical_json_bytes(changed_validation)
            )
            with self.assertRaisesRegex(ValueError, "validation SHA"):
                train_all_rows(
                    corpus,
                    root / "changed-validation-refit",
                    selection_model=selection,
                    test_receipt=changed_validation_output,
                    config=replace(selection_config, fit_all_rows=True),
                )

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

            with self.assertRaisesRegex(ValueError, "selection model config"):
                train_all_rows(
                    corpus,
                    root / "wrong-schedule",
                    selection_model=selection,
                    test_receipt=test_output,
                    config=_config(fit_all_rows=True),
                )

    def test_frozen_test_is_reserved_once_globally_per_corpus_and_selection(
        self,
    ) -> None:
        corpus = _easy_green_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = SimpleNamespace(**{**vars(corpus), "root": root / "corpus"})
            corpus.root.mkdir()
            selection = root / "selection"
            train_selection(corpus, selection, config=_config())
            first_receipt = root / "first-test.json"
            second_receipt = root / "second-test.json"
            evaluate_frozen_test(corpus, selection, first_receipt, device="cpu")
            with self.assertRaisesRegex(
                (FileExistsError, ValueError), "frozen test|already"
            ):
                evaluate_frozen_test(corpus, selection, second_receipt, device="cpu")
            self.assertFalse(second_receipt.exists())

    def test_copied_selection_cannot_bypass_global_frozen_test_reservation(
        self,
    ) -> None:
        corpus = _easy_green_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = SimpleNamespace(**{**vars(corpus), "root": root / "corpus"})
            corpus.root.mkdir()
            selection = root / "selection"
            copied_parent = root / "copied-parent"
            copied_selection = copied_parent / "selection-copy"
            train_selection(corpus, selection, config=_config())
            shutil.copytree(selection, copied_selection)

            evaluate_frozen_test(
                corpus, selection, root / "first-test.json", device="cpu"
            )
            with self.assertRaisesRegex(
                (FileExistsError, ValueError), "frozen test|already"
            ):
                evaluate_frozen_test(
                    corpus,
                    copied_selection,
                    root / "copied-test.json",
                    device="cpu",
                )

    def test_frozen_test_loader_keeps_local_position_metric_diagnostic_only(
        self,
    ) -> None:
        corpus = _easy_green_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = SimpleNamespace(**{**vars(corpus), "root": root / "corpus"})
            corpus.root.mkdir()
            selection = root / "selection"
            train_selection(corpus, selection, config=_config())
            patched_metrics = {
                "finite": True,
                "rows": int(np.count_nonzero(corpus.test_mask & corpus.eligible_mask)),
                "joint_geodesic_mae_rad": 0.01,
                "joint_frame_max_p95_rad": 0.02,
                "local_position_p95_m": 0.50,
                "fk_body_position_p95_m": 0.02,
                "support_foot_position_p95_m": 0.01,
                "contact_f1": [0.99, 0.99],
                "gate_limits": {
                    "joint_geodesic_mae_rad": 0.03,
                    "joint_frame_max_p95_rad": 0.10,
                    "fk_body_position_p95_m": 0.08,
                    "support_foot_position_p95_m": 0.05,
                    "minimum_contact_f1": 0.85,
                },
                "accepted": True,
            }
            with mock.patch(
                "mm_sonic.full_walking_terrain_lmm_training.evaluate_hybrid_rows",
                return_value=patched_metrics,
            ):
                receipt = evaluate_frozen_test(
                    corpus,
                    selection,
                    root / "patched-test.json",
                    device="cpu",
                )
            self.assertTrue(receipt["accepted"])
            self.assertEqual(receipt["metrics"]["local_position_p95_m"], 0.50)
            self.assertEqual(
                load_test_receipt(root / "patched-test.json")["metrics"][
                    "local_position_p95_m"
                ],
                0.50,
            )

    def test_selection_authorization_is_validation_only_when_train_diagnostic_is_red(
        self,
    ) -> None:
        corpus = _easy_green_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection"
            train_selection(corpus, selection, config=_config())
            patched = root / "patched-selection"

            shutil.copytree(selection, patched)
            evaluation = json.loads(
                (patched / "evaluation.json").read_text(encoding="utf-8")
            )
            evaluation["train"].update(
                {
                    "accepted": False,
                    "joint_geodesic_mae_rad": 0.5,
                    "joint_frame_max_p95_rad": 0.5,
                    "fk_body_position_p95_m": 0.5,
                    "support_foot_position_p95_m": 0.5,
                    "contact_f1": [0.0, 0.0],
                }
            )
            evaluation_payload = _json_bytes(evaluation)
            (patched / "evaluation.json").write_bytes(evaluation_payload)
            manifest = json.loads(
                (patched / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["artifacts"]["evaluation.json"].update(
                {
                    "size_bytes": len(evaluation_payload),
                    "sha256": hashlib.sha256(evaluation_payload).hexdigest(),
                }
            )
            (patched / "manifest.json").write_bytes(_json_bytes(manifest))
            loaded = load_hybrid_generator(patched, corpus=corpus, device="cpu")
            self.assertTrue(loaded.canonical_selection_verified)

    def test_loader_recomputes_full_walking_identity_exactly(self) -> None:
        corpus = _easy_green_corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection"
            train_selection(corpus, selection, config=_config())
            patched = root / "patched-selection"

            shutil.copytree(selection, patched)
            checkpoint = torch.load(
                patched / "model.pt", map_location="cpu", weights_only=True
            )
            for filename in ("manifest.json", "training.json", "evaluation.json"):
                payload = json.loads((patched / filename).read_text(encoding="utf-8"))
                payload["artifact_identity"]["fps"] = 25.0
                (patched / filename).write_bytes(_json_bytes(payload))
            checkpoint["artifact_identity"]["fps"] = 25.0
            torch.save(checkpoint, patched / "model.pt")
            manifest = json.loads(
                (patched / "manifest.json").read_text(encoding="utf-8")
            )
            for filename in ("model.pt", "training.json", "evaluation.json"):
                payload = (patched / filename).read_bytes()
                manifest["artifacts"][filename].update(
                    {
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            (patched / "manifest.json").write_bytes(_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_hybrid_generator(patched, corpus=corpus, device="cpu")


if __name__ == "__main__":
    unittest.main()
