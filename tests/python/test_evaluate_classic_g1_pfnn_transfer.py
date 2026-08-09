from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import mm_sonic.evaluate_classic_g1_pfnn_transfer as transfer
from mm_sonic.build_g1_pfnn_vertical_dataset import (
    VerticalDataset,
    VerticalSplitArrays,
    _dataset_digest,
    save_vertical_dataset,
)
from mm_sonic.evaluate_classic_g1_pfnn_transfer import (
    joint_reconstruction_metrics,
    released_pfnn_gate_failures,
    source_joint_reconstruction_metrics,
)
from mm_sonic.terrain_pfnn.dataset import normalize_pfnn_input
from mm_sonic.terrain_pfnn.layout import INPUT_LAYOUT, OUTPUT_LAYOUT
from mm_sonic.terrain_pfnn.model import PhaseFunctionedNetwork
from mm_sonic.terrain_pfnn.training import finite_runtime_seed
from mm_sonic.train_classic_g1_pfnn import save_classic_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evaluation_fixture(
    root: Path, *, checkpoint_dataset_digest: str | None = None,
    source: str = "mixed",
) -> tuple[Path, Path, VerticalDataset]:
    x_mean = np.full(INPUT_LAYOUT.size, np.float32(0.5))
    x_std = np.full(INPUT_LAYOUT.size, np.float32(0.25))
    y_mean = np.full(OUTPUT_LAYOUT.size, np.float32(1.5))
    y_std = np.full(OUTPUT_LAYOUT.size, np.float32(2.0))
    joint = OUTPUT_LAYOUT["joint_position"]
    all_clips = np.asarray(
        (
            "terrain_slopes__slope_000__000",
            "released_a",
            "terrain_slopes__slope_000__001",
            "released_b",
        ),
        dtype="<U128",
    )
    source_indices = {
        "mixed": np.asarray((0, 1, 2, 3), dtype=np.int64),
        "released_only": np.asarray((1, 3), dtype=np.int64),
        "grail_only": np.asarray((0, 2), dtype=np.int64),
    }
    if source not in source_indices:
        raise ValueError("fixture source is invalid")
    selected = source_indices[source]
    clips = all_clips[selected]
    target = np.tile(y_mean, (len(clips), 1)).astype(np.float32)
    if "released_b" in clips:
        target[int(np.flatnonzero(clips == "released_b")[0]), joint.start + 3] = np.float32(0.0)
    all_lanes = np.asarray(
        ("motion", "motion", "idle_phase_0", "idle_phase_0"), dtype="<U16"
    )
    all_centers = np.asarray((100, 200, 300, 400), dtype=np.int64)
    all_phase = np.asarray((0.1, 0.2, 0.3, 0.4), dtype=np.float32)
    values = VerticalSplitArrays(
        x=np.full((len(clips), INPUT_LAYOUT.size), np.float32(0.75)),
        y=target,
        phase=all_phase[selected],
        clip_id=clips,
        sequence_lane=all_lanes[selected],
        center_frame_120hz=all_centers[selected],
        root_world_xy=np.zeros((len(clips), 2), dtype=np.float32),
        root_world_yaw=np.zeros(len(clips), dtype=np.float32),
        terrain_class=np.asarray(("flat",) * len(clips), dtype="<U10"),
        terrain_sha256=np.asarray(("a" * 64,) * len(clips), dtype="<U64"),
        mirrored=np.zeros(len(clips), dtype=np.bool_),
    )
    normalization = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    source_roles = {str(clip): "validation" for clip in clips}
    dataset = VerticalDataset(
        splits={"train": values, "validation": values},
        **normalization,
        selection_sha256="b" * 64,
        retarget_manifest_sha256="c" * 64,
        terrain_receipt_set_sha256="d" * 64,
        source_roles=source_roles,
        dataset_sha256=_dataset_digest(
            {"train": values, "validation": values},
            normalization,
            selection_sha256="b" * 64,
            retarget_manifest_sha256="c" * 64,
            terrain_receipt_set_sha256="d" * 64,
            source_roles=source_roles,
        ),
    )
    manifest = save_vertical_dataset(root / "dataset", dataset)
    model = PhaseFunctionedNetwork(hidden_size=4, dropout_probability=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.b2[:, joint.start + 3] = 0.25
    checkpoint = root / "checkpoint.pt"
    save_classic_checkpoint(
        checkpoint,
        model=model,
        normalization=normalization,
        runtime_seed=finite_runtime_seed(),
        dataset_digest=(
            dataset.dataset_sha256
            if checkpoint_dataset_digest is None
            else checkpoint_dataset_digest
        ),
        kinematic_signature_sha256="e" * 64,
        joint_limits=torch.tensor([[-2.0, 2.0]] * 29),
        phase_advance_q99=0.2,
        epoch=1,
        validation_loss=0.1,
        seed=7,
        source_kind="mixed",
        vertical_slice_receipt_sha256="f" * 64,
        terrain_receipt_set_sha256="0" * 64,
    )
    return manifest, checkpoint, dataset


class ClassicG1PFNNTransferEvaluationTests(unittest.TestCase):
    def test_joint_metrics_report_exact_error_and_worst_frame_joint(self) -> None:
        target = np.zeros((4, 29), dtype=np.float32)
        predicted = target.copy()
        predicted[2, 3] = np.float32(0.2)

        metrics = joint_reconstruction_metrics(predicted, target)

        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["joint_mae_rad"], 0.2 / (4 * 29))
        self.assertAlmostEqual(metrics["joint_rmse_rad"], 0.2 / np.sqrt(4 * 29))
        self.assertAlmostEqual(metrics["frame_max_p95_rad"], 0.17)
        self.assertAlmostEqual(metrics["maximum_joint_error_rad"], 0.2)
        self.assertEqual(metrics["worst_frame_index"], 2)
        self.assertEqual(metrics["worst_joint_index"], 3)

    def test_released_gate_cannot_be_hidden_by_good_grail_rows(self) -> None:
        target = np.zeros((21, 29), dtype=np.float32)
        predicted = target.copy()
        predicted[0] = np.float32(0.2)
        clips = np.asarray(
            ("released_pfnn_walk",)
            + tuple(f"terrain_slopes__slope_000__{index:03d}" for index in range(20))
        )

        metrics = source_joint_reconstruction_metrics(predicted, target, clips)
        aggregate = joint_reconstruction_metrics(predicted, target)

        self.assertLess(aggregate["joint_mae_rad"], 0.1)
        self.assertEqual(metrics["released_pfnn"]["sample_count"], 1)
        self.assertAlmostEqual(metrics["released_pfnn"]["joint_mae_rad"], 0.2)
        self.assertEqual(
            released_pfnn_gate_failures(metrics["released_pfnn"]),
            ("joint_mae_rad", "joint_rmse_rad"),
        )
        self.assertEqual(released_pfnn_gate_failures(metrics["grail"]), ())

    def test_evaluate_records_exact_inputs_one_denormalization_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, checkpoint, dataset = _write_evaluation_fixture(root)
            seen: list[tuple[torch.Tensor, torch.Tensor]] = []
            original_forward = PhaseFunctionedNetwork.forward

            def recording_forward(self, x, phase):
                seen.append((x.detach().cpu().clone(), phase.detach().cpu().clone()))
                return original_forward(self, x, phase)

            with mock.patch.object(PhaseFunctionedNetwork, "forward", recording_forward):
                report = transfer.evaluate(
                    checkpoint_path=checkpoint,
                    dataset_path=manifest,
                    device="cpu",
                )
            expected_x = normalize_pfnn_input(
                dataset.splits["validation"].x, dataset.x_mean, dataset.x_std
            )
            self.assertEqual(len(seen), 1)
            np.testing.assert_array_equal(seen[0][0].numpy(), expected_x)
            self.assertEqual(seen[0][0].dtype, torch.float32)
            self.assertEqual(seen[0][1].dtype, torch.float32)
            np.testing.assert_array_equal(
                seen[0][1].numpy(), dataset.splits["validation"].phase
            )
            self.assertAlmostEqual(
                report["released_pfnn"]["maximum_joint_error_rad"], 2.0
            )
            self.assertEqual(
                report["released_pfnn"]["worst"],
                {
                    "source_frame_index": 1,
                    "split_index": 3,
                    "clip_id": "released_b",
                    "sequence_lane": "idle_phase_0",
                    "center_frame_120hz": 400,
                    "joint_index": 3,
                    "joint_name": "left_hip_roll_joint",
                },
            )
            self.assertEqual(report["checkpoint_sha256"], _sha256(checkpoint))
            self.assertEqual(report["dataset_manifest_sha256"], _sha256(manifest))

    def test_digest_mismatch_rejects_before_model_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, checkpoint, _ = _write_evaluation_fixture(
                Path(directory), checkpoint_dataset_digest="1" * 64
            )
            with mock.patch.object(
                PhaseFunctionedNetwork, "forward", side_effect=AssertionError("inference ran")
            ):
                with self.assertRaisesRegex(ValueError, "dataset digest mismatch"):
                    transfer.evaluate(
                        checkpoint_path=checkpoint, dataset_path=manifest, device="cpu"
                    )

    def test_evaluate_rejects_replaced_checkpoint_or_manifest_after_inference(self) -> None:
        for artifact in ("checkpoint", "manifest"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                manifest, checkpoint, _ = _write_evaluation_fixture(Path(directory))
                original_forward = PhaseFunctionedNetwork.forward
                changed = False

                def replacing_forward(self, x, phase):
                    nonlocal changed
                    if not changed:
                        changed = True
                        path = checkpoint if artifact == "checkpoint" else manifest
                        with path.open("ab") as stream:
                            stream.write(b"replacement")
                    return original_forward(self, x, phase)

                with mock.patch.object(PhaseFunctionedNetwork, "forward", replacing_forward):
                    with self.assertRaisesRegex(ValueError, f"{artifact}.*changed"):
                        transfer.evaluate(
                            checkpoint_path=checkpoint, dataset_path=manifest, device="cpu"
                        )

    def test_cli_writes_receipt_refuses_overwrite_and_returns_gate_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, checkpoint, _ = _write_evaluation_fixture(root)
            output = root / "receipt.json"
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                status = transfer.main(
                    [
                        "--checkpoint", str(checkpoint), "--dataset", str(manifest),
                        "--device", "cpu", "--output", str(output),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text()))
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                transfer.main(
                    [
                        "--checkpoint", str(checkpoint), "--dataset", str(manifest),
                        "--device", "cpu", "--output", str(output),
                    ]
                )

    def test_released_only_evaluate_and_cli_emit_null_grail_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, checkpoint, _ = _write_evaluation_fixture(
                root, source="released_only"
            )

            report = transfer.evaluate(
                checkpoint_path=checkpoint, dataset_path=manifest, device="cpu"
            )

            self.assertEqual(report["released_pfnn"]["sample_count"], 2)
            self.assertIsNone(report["grail"])
            self.assertEqual(
                report["released_pfnn"]["worst"],
                {
                    "source_frame_index": 1,
                    "split_index": 1,
                    "clip_id": "released_b",
                    "sequence_lane": "idle_phase_0",
                    "center_frame_120hz": 400,
                    "joint_index": 3,
                    "joint_name": "left_hip_roll_joint",
                },
            )
            self.assertEqual(report["checkpoint_sha256"], _sha256(checkpoint))
            self.assertEqual(report["dataset_manifest_sha256"], _sha256(manifest))
            self.assertFalse(report["released_pfnn_gate"]["accepted"])
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                status = transfer.main(
                    [
                        "--checkpoint", str(checkpoint), "--dataset", str(manifest),
                        "--device", "cpu",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIsNone(json.loads(stdout.getvalue())["grail"])

    def test_grail_only_evaluation_rejects_missing_released_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, checkpoint, _ = _write_evaluation_fixture(
                Path(directory), source="grail_only"
            )

            with self.assertRaisesRegex(ValueError, "requires released PFNN rows"):
                transfer.evaluate(
                    checkpoint_path=checkpoint, dataset_path=manifest, device="cpu"
                )


if __name__ == "__main__":
    unittest.main()
