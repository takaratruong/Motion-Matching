from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import mm_sonic.hybrid_terrain_lmm_combined as combined_module
import mm_sonic.hybrid_terrain_lmm_combined_training as combined_training_module
import numpy as np
from mm_sonic.hybrid_terrain_lmm_combined import (
    build_combined_cache,
    load_combined_cache,
)
from mm_sonic.hybrid_terrain_lmm_combined_training import main as training_main
from mm_sonic.hybrid_terrain_lmm_data import (
    FAMILY_NAMES,
    _assemble_hybrid_corpus,
    _load_synthetic_hybrid_cache_for_tests,
    _publish_hybrid_cache,
)
from mm_sonic.hybrid_terrain_lmm_pfnn import (
    EXPECTED_TRAIN_STEM_FAMILIES,
    PFNNSupplement,
    publish_pfnn_supplement,
)

from resources.g1_terrain_builder.artifacts import (
    write_support_sidecar,
)
from resources.g1_terrain_builder.database import write_holden_database
from resources.g1_terrain_builder.features import _normalize
from resources.g1_terrain_builder.schema import (
    ArtifactSet,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifacts(lengths: tuple[int, ...], *, phase: float) -> ArtifactSet:
    starts = np.cumsum((0,) + lengths[:-1], dtype=np.int32)
    stops = np.cumsum(lengths, dtype=np.int32)
    frames = int(stops[-1])
    positions = np.zeros((frames, 31, 3), np.float32)
    velocities = np.zeros_like(positions)
    rotations = np.zeros((frames, 31, 4), np.float32)
    angular = np.zeros_like(positions)
    rotations[..., 0] = 1.0
    time = np.arange(frames, dtype=np.float32) + np.float32(phase)
    positions[:, 0, 0] = 0.025 * time
    positions[:, 0, 2] = 0.015 * np.square(time) / max(frames, 1)
    positions[:, 6, 0] = -0.10 + 0.01 * np.sin(time)
    positions[:, 6, 1] = -0.50
    positions[:, 12, 0] = 0.10 + 0.01 * np.cos(time)
    positions[:, 12, 1] = -0.50
    velocities[:, 0, 0] = 0.625
    velocities[:, 0, 2] = 0.75 * time / max(frames, 1)
    terrain = np.column_stack(
        tuple(0.01 * (column + 1) * time for column in range(4))
    ).astype(np.float32)
    support = np.column_stack(
        (0.005 * time, 0.006 * time, 0.007 * time)
    ).astype(np.float32)
    parents = np.asarray(
        (
            -1, 0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 16, 24, 25, 26, 27, 28, 29,
        ),
        np.int32,
    )
    result = ArtifactSet(
        positions,
        velocities,
        rotations,
        angular,
        parents,
        starts,
        stops,
        (np.arange(frames * 2).reshape(frames, 2) % 2).astype(np.uint8),
        terrain,
        support,
    )
    result.validate()
    return result


def _publish_primary(root: Path) -> tuple[Path, object]:
    source = root / "primary-source"
    source.mkdir()
    artifacts = _artifacts((12,) * 7, phase=0.0)
    corpus = _assemble_hybrid_corpus(
        artifacts,
        np.asarray([0, 1, 1, 2, 2, 3, 3], np.int16),
        source,
        {"schema": "synthetic-primary/v1", "sha256": "a" * 64},
    )
    write_holden_database(source / "database.bin", artifacts)
    terrain_v2 = np.zeros((len(artifacts.positions), 12), np.float32)
    terrain_v2[:, (1, 3, 5, 7)] = artifacts.terrain_features
    (source / "terrain_features.bin").write_bytes(
        struct.pack("<4sIII", b"G1TF", 2, len(terrain_v2), 12)
        + np.ascontiguousarray(terrain_v2, dtype="<f4").tobytes()
    )
    write_support_sidecar(source / "terrain_support.bin", artifacts.terrain_support)
    cache = root / "primary-cache"
    _publish_hybrid_cache(corpus, cache)
    return cache, corpus


def _publish_supplement(root: Path) -> tuple[Path, PFNNSupplement]:
    from tests.python.test_hybrid_terrain_lmm_pfnn import _synthetic_supplement

    supplement = _synthetic_supplement()
    output = root / "pfnn-supplement"
    publish_pfnn_supplement(supplement, output)
    return output, supplement


def _raw(features: object) -> np.ndarray:
    return (
        np.asarray(features.values, np.float64)
        * np.asarray(features.scale, np.float64)
        + np.asarray(features.offset, np.float64)
    )


class HybridTerrainLmmCombinedTests(unittest.TestCase):
    def test_cuda_training_environment_import_does_not_require_mujoco(self) -> None:
        interpreter = Path(
            os.environ.get(
                "FOUNDATION_STEREO_PYTHON",
                "/home/ubuntu/miniconda3/envs/foundation_stereo/bin/python",
            )
        )
        if not interpreter.is_file():
            self.skipTest(
                "foundation_stereo interpreter is unavailable; set "
                "FOUNDATION_STEREO_PYTHON to exercise this import boundary"
            )
        command = [
            str(interpreter),
            "-c",
            "import mm_sonic.hybrid_terrain_lmm_combined_training",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = ".:resources:sonic/python"

        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    @patch.object(
        combined_module,
        "load_hybrid_cache",
        side_effect=_load_synthetic_hybrid_cache_for_tests,
    )
    def test_combines_authorities_with_joint_normalization_and_train_only_pfnn(
        self,
        primary_loader: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path, primary = _publish_primary(root)
            pfnn_path, pfnn = _publish_supplement(root)
            primary_manifest_sha = _sha256(primary_path / "manifest.json")
            pfnn_manifest_sha = _sha256(pfnn_path / "manifest.json")
            output = root / "combined"

            manifest_path = build_combined_cache(primary_path, pfnn_path, output)
            combined = load_combined_cache(output)

            self.assertEqual(manifest_path, output / "manifest.json")
            self.assertEqual(len(combined.artifacts.positions), 2_209)
            self.assertEqual(len(combined.artifacts.range_starts), 25)
            np.testing.assert_array_equal(
                combined.train_mask[:84], primary.train_mask
            )
            np.testing.assert_array_equal(
                combined.evaluation_mask[:84], primary.evaluation_mask
            )
            self.assertTrue(np.all(combined.train_mask[84:]))
            self.assertFalse(np.any(combined.evaluation_mask[84:]))
            np.testing.assert_array_equal(
                combined.range_family_ids[-17:],
                [
                    FAMILY_NAMES.index(family)
                    for _, family in EXPECTED_TRAIN_STEM_FAMILIES
                ],
            )
            np.testing.assert_array_equal(
                combined.artifacts.range_starts[-17:],
                84 + np.arange(17, dtype=np.int32) * 125,
            )
            np.testing.assert_array_equal(
                combined.artifacts.range_stops[-17:],
                84 + np.arange(1, 18, dtype=np.int32) * 125,
            )
            expected_raw = np.concatenate((_raw(primary.features), _raw(pfnn.features)))
            np.testing.assert_allclose(
                _raw(combined.features), expected_raw, rtol=2.0e-5, atol=2.0e-5
            )
            expected_features = _normalize(
                np.ascontiguousarray(expected_raw, dtype=np.float32),
                np.asarray(combined.train_mask, dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                combined.features.values, expected_features.values
            )
            np.testing.assert_array_equal(
                combined.features.offset, expected_features.offset
            )
            np.testing.assert_array_equal(
                combined.features.scale, expected_features.scale
            )
            manifest = json.loads(manifest_path.read_bytes())
            self.assertEqual(
                manifest["schema"],
                "g1-hybrid-terrain-lmm-combined-cache/v2-strict",
            )
            self.assertEqual(
                manifest["authorities"]["primary_cache"]["sha256"],
                primary_manifest_sha,
            )
            self.assertEqual(
                manifest["authorities"]["pfnn_supplement"]["sha256"],
                pfnn_manifest_sha,
            )
            self.assertEqual(manifest["counts"]["pfnn_train_rows"], 2_125)
            self.assertEqual(manifest["counts"]["pfnn_evaluation_rows"], 0)
            self.assertTrue(manifest["split"]["pfnn_admitted_train_only"])
            self.assertTrue(manifest["split"]["primary_masks_preserved"])
            self.assertEqual(
                manifest["normalization"],
                {
                    "evaluation_rows_in_fit": 0,
                    "fit_rows": "combined-train-mask",
                    "input": "denormalized-authority-31d-features",
                    "method": "joint-group-mean-and-mean-standard-deviation",
                    "transform_rows": "all-combined-rows",
                },
            )
            self.assertEqual(_sha256(primary_path / "manifest.json"), primary_manifest_sha)
            self.assertEqual(_sha256(pfnn_path / "manifest.json"), pfnn_manifest_sha)

            second = root / "combined-second"
            second_manifest = build_combined_cache(primary_path, pfnn_path, second)
            self.assertEqual(second_manifest.read_bytes(), manifest_path.read_bytes())
            for call in primary_loader.call_args_list:
                self.assertEqual(
                    call.kwargs["expected_cache_manifest_sha256"],
                    primary_manifest_sha,
                )

    @patch.object(
        combined_module,
        "load_hybrid_cache",
        side_effect=_load_synthetic_hybrid_cache_for_tests,
    )
    def test_loader_rejects_a_changed_input_authority_before_materialization(
        self,
        _primary_loader: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path, _ = _publish_primary(root)
            pfnn_path, _ = _publish_supplement(root)
            output = root / "combined"
            build_combined_cache(primary_path, pfnn_path, output)

            with (pfnn_path / "manifest.json").open("ab") as stream:
                stream.write(b"changed")

            with self.assertRaisesRegex(ValueError, "PFNN.*authority"):
                load_combined_cache(output)

    @patch.object(
        combined_module,
        "load_hybrid_cache",
        side_effect=_load_synthetic_hybrid_cache_for_tests,
    )
    def test_loader_rejects_a_pfnn_manifest_changed_during_strict_load(
        self,
        _primary_loader: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path, _ = _publish_primary(root)
            pfnn_path, _ = _publish_supplement(root)
            output = root / "combined"
            build_combined_cache(primary_path, pfnn_path, output)
            strict_loader = combined_module.load_pfnn_supplement

            def load_then_replace_manifest(
                path: Path, *, expected_manifest_sha256: str
            ) -> PFNNSupplement:
                supplement = strict_loader(
                    path,
                    expected_manifest_sha256=expected_manifest_sha256,
                )
                manifest_path = Path(path) / "manifest.json"
                manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
                return supplement

            with patch.object(
                combined_module,
                "load_pfnn_supplement",
                side_effect=load_then_replace_manifest,
            ), self.assertRaisesRegex(
                ValueError, "PFNN authority digest changed during load"
            ):
                load_combined_cache(output)

    @patch.object(
        combined_module,
        "load_hybrid_cache",
        side_effect=_load_synthetic_hybrid_cache_for_tests,
    )
    def test_explicit_training_wrapper_accepts_the_combined_cache(
        self,
        _primary_loader: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path, primary = _publish_primary(root)
            pfnn_path, _ = _publish_supplement(root)
            corpus = root / "combined"
            model = root / "model"
            build_combined_cache(primary_path, pfnn_path, corpus)

            status = training_main(
                [
                    "train",
                    "--corpus",
                    str(corpus),
                    "--output",
                    str(model),
                    "--variant",
                    "latent32",
                    "--device",
                    "cpu",
                    "--loss-profile",
                    "visual-articulation",
                    "--post-coverage-steps",
                    "0",
                    "--batch-size",
                    "4096",
                    "--normalization-chunk-size",
                    "4096",
                    "--evaluation-chunk-size",
                    "4096",
                ]
            )

            self.assertEqual(status, 0)
            training = json.loads((model / "training.json").read_bytes())
            self.assertEqual(
                training["coverage_rows"],
                int(np.count_nonzero(primary.train_mask)) + 2_125,
            )
            self.assertEqual(training["fit_scope"], "source-held-out-selection")
            self.assertEqual(training["config"]["loss_profile"], "visual-articulation")

    def test_all_row_wrapper_forwards_an_authenticated_selection_model(self) -> None:
        corpus = object()
        with (
            patch.object(
                combined_training_module,
                "load_combined_cache",
                return_value=corpus,
            ),
            patch.object(
                combined_training_module,
                "train_hybrid_generator",
                return_value={"status": "published"},
            ) as trainer,
        ):
            status = training_main(
                [
                    "train",
                    "--corpus",
                    "/immutable/combined",
                    "--output",
                    "/immutable/refit",
                    "--variant",
                    "latent32",
                    "--device",
                    "cuda:6",
                    "--fit-all-rows",
                    "--selection-model",
                    "/immutable/selection-model",
                    "--selection-model-manifest-sha256",
                    "b" * 64,
                ]
            )

        self.assertEqual(status, 0)
        self.assertIs(trainer.call_args.args[0], corpus)
        self.assertEqual(trainer.call_args.args[1], Path("/immutable/refit"))
        self.assertTrue(trainer.call_args.kwargs["config"].fit_all_rows)
        self.assertEqual(
            trainer.call_args.kwargs["selection_model"],
            Path("/immutable/selection-model"),
        )
        self.assertEqual(
            trainer.call_args.kwargs["selection_model_manifest_sha256"],
            "b" * 64,
        )


if __name__ == "__main__":
    unittest.main()
