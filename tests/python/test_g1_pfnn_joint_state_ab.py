from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import mm_sonic.g1_pfnn_joint_state_ab as joint_state_ab
from mm_sonic.g1_pfnn_joint_state_ab import (
    EXPERIMENT_CLIPS,
    ExperimentConfig,
    ExperimentMetrics,
    apply_normalization,
    decide_experiment,
    deterministic_update_batches,
    fit_normalization,
    joint_state_oracle_diagnostics,
    phase_bank_coverage,
    run_experiment,
    select_fixed_blocks,
    treatment_inputs,
)
from mm_sonic.terrain_pfnn.layout import OUTPUT_LAYOUT


def _split_arrays(*, rows_per_clip: int = 8) -> SimpleNamespace:
    count = len(EXPERIMENT_CLIPS) * rows_per_clip
    x = np.zeros((count, 346), dtype=np.float32)
    y = np.zeros((count, OUTPUT_LAYOUT.size), dtype=np.float32)
    phase = np.empty(count, dtype=np.float32)
    clip_id = np.empty(count, dtype="<U128")
    sequence_lane = np.full(count, "motion", dtype="<U16")
    center = np.empty(count, dtype=np.int64)
    mirrored = np.zeros(count, dtype=np.bool_)
    terrain_class = np.empty(count, dtype="<U10")
    terrain_sha256 = np.empty(count, dtype="<U64")
    row = 0
    for clip_index, clip in enumerate(EXPERIMENT_CLIPS):
        for frame in range(rows_per_clip):
            clip_id[row] = clip
            center[row] = 1000 * (clip_index + 1) + 4 * frame
            phase[row] = np.float32(frame * 0.1)
            terrain_class[row] = "flat" if clip_index == 0 else "ascent"
            terrain_sha256[row] = f"{clip_index + 1:064x}"
            q = np.full(29, 0.01 * frame + 0.1 * clip_index, dtype=np.float32)
            qdot = np.full(29, 0.3 + 0.01 * frame, dtype=np.float32)
            x[row, 288:317] = q
            x[row, 317:346] = qdot
            y[row, OUTPUT_LAYOUT["joint_position"]] = q + np.float32(0.01)
            row += 1
    return SimpleNamespace(
        x=x,
        y=y,
        phase=phase,
        clip_id=clip_id,
        sequence_lane=sequence_lane,
        center_frame_120hz=center,
        mirrored=mirrored,
        terrain_class=terrain_class,
        terrain_sha256=terrain_sha256,
    )


def _dataset(*, rows_per_clip: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        splits={"train": _split_arrays(rows_per_clip=rows_per_clip)},
        dataset_sha256="a" * 64,
        selection_sha256="b" * 64,
        retarget_manifest_sha256="c" * 64,
        terrain_receipt_set_sha256="d" * 64,
        joint_state_receipt={
            "schema": "g1-pfnn-joint-state-receipt/v1",
            "maximum_joint_step_rad": 0.225,
            "state_source_by_clip": {
                clip: "direct_source" for clip in EXPERIMENT_CLIPS
            },
        },
        source_roles={clip: "train" for clip in EXPERIMENT_CLIPS},
    )


def _metrics(*, accepted: bool) -> ExperimentMetrics:
    if accepted:
        values = {
            "sample_count": 4,
            "joint_mae_rad": 0.05,
            "joint_rmse_rad": 0.08,
            "frame_max_p95_rad": 0.20,
            "maximum_joint_error_rad": 0.30,
            "worst_frame_index": 0,
            "worst_joint_index": 0,
        }
    else:
        values = {
            "sample_count": 4,
            "joint_mae_rad": 0.11,
            "joint_rmse_rad": 0.16,
            "frame_max_p95_rad": 0.51,
            "maximum_joint_error_rad": 1.01,
            "worst_frame_index": 0,
            "worst_joint_index": 0,
        }
    return ExperimentMetrics.from_mapping(values)


class G1PFNNJointStateABTests(unittest.TestCase):
    def test_default_config_keeps_the_released_pfnn_internal_width(self) -> None:
        config = ExperimentConfig()

        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.dropout_probability, 0.30)

    def test_selects_fixed_immediately_following_blocks_from_every_clip(self) -> None:
        arrays = _split_arrays()

        selection = select_fixed_blocks(
            arrays, fit_rows_per_clip=3, held_rows_per_clip=2
        )

        self.assertEqual(len(selection.fit_indices), 12)
        self.assertEqual(len(selection.held_indices), 8)
        for clip in EXPERIMENT_CLIPS:
            fit = [row for row in selection.rows if row.clip_id == clip and row.block == "fit"]
            held = [row for row in selection.rows if row.clip_id == clip and row.block == "held_out"]
            self.assertEqual(len(fit), 3)
            self.assertEqual(len(held), 2)
            self.assertEqual(held[0].center_frame_120hz, fit[-1].center_frame_120hz + 4)
            self.assertTrue(all(row.sequence_lane == "motion" for row in fit + held))
            self.assertTrue(all(not row.mirrored for row in fit + held))
            self.assertTrue(all(len(row.row_sha256) == 64 for row in fit + held))

    def test_selection_rejects_duplicate_or_unsafe_retained_rows(self) -> None:
        duplicate = _split_arrays()
        duplicate.center_frame_120hz[1] = duplicate.center_frame_120hz[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_fixed_blocks(duplicate, fit_rows_per_clip=3, held_rows_per_clip=2)

        unsafe = _split_arrays()
        joint = OUTPUT_LAYOUT["joint_position"]
        unsafe.y[2, joint.start] += np.float32(0.226)
        with self.assertRaisesRegex(ValueError, "0.225"):
            select_fixed_blocks(unsafe, fit_rows_per_clip=3, held_rows_per_clip=2)

    def test_treatments_have_exact_width_and_fit_only_normalization(self) -> None:
        arrays = _split_arrays()
        baseline = treatment_inputs(arrays.x, "baseline_288")
        raw = treatment_inputs(arrays.x, "raw_346")
        periodic = treatment_inputs(arrays.x, "periodic_375")

        self.assertEqual(baseline.shape[1], 288)
        self.assertEqual(raw.shape[1], 346)
        self.assertEqual(periodic.shape[1], 375)
        np.testing.assert_array_equal(baseline, arrays.x[:, :288])
        np.testing.assert_array_equal(raw, arrays.x)
        np.testing.assert_allclose(periodic[:, 288:317], np.sin(arrays.x[:, 288:317]))
        np.testing.assert_allclose(periodic[:, 317:346], np.cos(arrays.x[:, 288:317]))
        np.testing.assert_array_equal(periodic[:, 346:375], arrays.x[:, 317:346])

        fit_x = raw[:8]
        fit_y = arrays.y[:8]
        normalization = fit_normalization(fit_x, fit_y)
        changed_held = raw.copy()
        changed_held[8:] = np.float32(999.0)
        repeated = fit_normalization(changed_held[:8], fit_y)
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            np.testing.assert_array_equal(normalization[name], repeated[name])
        normalized_x, normalized_y = apply_normalization(fit_x, fit_y, normalization)
        self.assertEqual(normalized_x.dtype, np.dtype(np.float32))
        self.assertEqual(normalized_y.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            normalization["y_mean"][OUTPUT_LAYOUT["contact_logit"]], 0.0
        )
        np.testing.assert_array_equal(
            normalization["y_std"][OUTPUT_LAYOUT["contact_logit"]], 1.0
        )

    def test_only_raw_can_be_accepted_or_promoted(self) -> None:
        result = decide_experiment(
            baseline_fit=_metrics(accepted=True),
            baseline_held=_metrics(accepted=False),
            raw_fit=_metrics(accepted=True),
            raw_held=_metrics(accepted=True),
            periodic_fit=None,
            periodic_held=None,
            periodic_scope="skipped_runtime_deadline",
        )
        self.assertEqual(result.outcome, "raw_346_accepted")
        self.assertEqual(result.promotable_treatment, "raw_346")
        self.assertTrue(result.accepted)

        periodic_only = decide_experiment(
            baseline_fit=_metrics(accepted=True),
            baseline_held=_metrics(accepted=False),
            raw_fit=_metrics(accepted=False),
            raw_held=_metrics(accepted=False),
            periodic_fit=_metrics(accepted=True),
            periodic_held=_metrics(accepted=True),
            periodic_scope="executed",
        )
        self.assertEqual(periodic_only.outcome, "periodic_only_stop_for_revision")
        self.assertIsNone(periodic_only.promotable_treatment)
        self.assertFalse(periodic_only.accepted)

    def test_update_batches_are_exact_and_identical_across_treatments(self) -> None:
        first = deterministic_update_batches(
            7, batch_size=3, seed=23456, update_count=8
        )
        second = deterministic_update_batches(
            7, batch_size=3, seed=23456, update_count=8
        )

        self.assertEqual(len(first), 8)
        self.assertEqual(len(second), 8)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertGreaterEqual(len(left), 1)
            self.assertLessEqual(len(left), 3)
            self.assertTrue(np.all((left >= 0) & (left < 7)))
        np.testing.assert_array_equal(
            np.sort(np.concatenate(first[:3])), np.arange(7)
        )

    def test_receipt_exposes_phase_coverage_and_trivial_joint_state_oracles(self) -> None:
        arrays = _split_arrays()
        for offset in range(0, len(arrays.phase), 8):
            arrays.phase[offset : offset + 5] = np.asarray(
                (4.0, 4.4, 4.8, 0.0, 0.2), dtype=np.float32
            )
        selection = select_fixed_blocks(
            arrays, fit_rows_per_clip=3, held_rows_per_clip=2
        )

        coverage = phase_bank_coverage(arrays, selection)
        oracles = joint_state_oracle_diagnostics(arrays, selection)

        self.assertEqual(
            coverage["disjoint_primary_bank_clips"], list(EXPERIMENT_CLIPS)
        )
        for clip in EXPERIMENT_CLIPS:
            self.assertEqual(coverage["per_clip"][clip]["fit"], [0, 0, 2, 1])
            self.assertEqual(
                coverage["per_clip"][clip]["held_out"], [2, 0, 0, 0]
            )
        for name in ("q_copy", "q_plus_qdot_dt"):
            for block in ("fit", "held_out"):
                self.assertTrue(oracles[name][block]["aggregate"]["accepted"])

    def test_run_persists_sealed_rows_metrics_checkpoints_and_hashes(self) -> None:
        config = ExperimentConfig(
            seed=17,
            update_count=2,
            batch_size=4,
            hidden_size=8,
            learning_rate=1.0e-4,
            dropout_probability=0.0,
            evaluation_batch_size=16,
            fit_rows_per_clip=2,
            held_rows_per_clip=1,
            include_periodic=False,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            report = run_experiment(
                _dataset(rows_per_clip=6),
                output=output,
                config=config,
                dataset_manifest_sha256="e" * 64,
            )

            self.assertEqual(report["schema"], "g1-pfnn-joint-state-ab/v1")
            self.assertEqual(report["periodic_scope"], "skipped_runtime_deadline")
            self.assertEqual(set(report["treatments"]), {"baseline_288", "raw_346"})
            self.assertEqual(report["training"]["update_count"], 2)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertEqual(len(report["code"]["git_commit"]), 40)
            self.assertEqual(len(report["code"]["source_tree_sha256"]), 64)
            self.assertEqual(len(report["code"]["dirty_patch_sha256"]), 64)
            self.assertIn(
                "sonic/python/mm_sonic/g1_pfnn_joint_state_ab.py",
                report["code"]["source_files"],
            )
            self.assertTrue(report["code"]["source_tree_verified_unchanged"])
            self.assertEqual(report["environment"]["torch_version"], torch.__version__)
            self.assertEqual(report["environment"]["numpy_version"], np.__version__)
            self.assertEqual(
                len(
                    report["dataset"]["source_provenance"][
                        "joint_state_receipt_sha256"
                    ]
                ),
                64,
            )
            self.assertEqual(report["selection"]["fit_row_count"], 8)
            self.assertEqual(report["selection"]["held_out_row_count"], 4)
            self.assertIn("phase_bank_coverage", report["diagnostics"])
            self.assertIn("joint_state_oracles", report["diagnostics"])
            self.assertEqual(
                json.loads((output / "experiment.json").read_text()), report
            )
            selection = json.loads((output / "selection.json").read_text())
            self.assertEqual(len(selection["rows"]), 12)
            for treatment, record in report["treatments"].items():
                checkpoint = output / record["checkpoint"]["path"]
                metrics = output / record["metrics_artifact"]["path"]
                self.assertEqual(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    record["checkpoint"]["sha256"],
                )
                self.assertEqual(
                    hashlib.sha256(metrics.read_bytes()).hexdigest(),
                    record["metrics_artifact"]["sha256"],
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(payload["treatment"], treatment)
                self.assertEqual(payload["update_count"], 2)
                self.assertEqual(
                    payload["joint_state_receipt_sha256"],
                    report["dataset"]["source_provenance"][
                        "joint_state_receipt_sha256"
                    ],
                )
                self.assertFalse(payload["checkpoint_promotion_eligible"])
                self.assertEqual(
                    payload["representation_class_candidate"], treatment == "raw_346"
                )
            repeated_output = Path(directory) / "repeated"
            repeated = run_experiment(
                _dataset(rows_per_clip=6),
                output=repeated_output,
                config=config,
                dataset_manifest_sha256="e" * 64,
            )
            self.assertEqual(report, repeated)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                sorted(path.name for path in repeated_output.iterdir()),
            )
            for original in output.iterdir():
                self.assertEqual(
                    original.read_bytes(),
                    (repeated_output / original.name).read_bytes(),
                )
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                run_experiment(
                    _dataset(rows_per_clip=6),
                    output=output,
                    config=config,
                    dataset_manifest_sha256="e" * 64,
                )

    def test_run_rejects_a_source_tree_change_during_training(self) -> None:
        config = ExperimentConfig(
            update_count=1,
            batch_size=4,
            hidden_size=8,
            dropout_probability=0.0,
            fit_rows_per_clip=2,
            held_rows_per_clip=1,
            include_periodic=False,
            device="cpu",
        )
        identity = joint_state_ab._code_identity()
        changed = dict(identity)
        changed["source_tree_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            joint_state_ab, "_code_identity", side_effect=(identity, changed)
        ):
            with self.assertRaisesRegex(ValueError, "source tree changed"):
                run_experiment(
                    _dataset(rows_per_clip=6),
                    output=Path(directory) / "changed",
                    config=config,
                    dataset_manifest_sha256="e" * 64,
                )


if __name__ == "__main__":
    unittest.main()
