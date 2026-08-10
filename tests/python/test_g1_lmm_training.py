import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from resources.g1_lmm.dataset import (
    G1LmmDimensions,
    build_training_arrays,
    load_training_bundle,
    range_safe_deltas,
    range_safe_windows,
)
from resources.g1_lmm.models import Compressor, Decompressor, Projector, Stepper
from resources.g1_lmm import training as g1_lmm_training
from resources.g1_terrain_builder.schema import (
    G1_SKELETON_NAMES,
    G1_SKELETON_PARENTS,
    G1_SKELETON_SIGNATURE,
)
from resources.g1_lmm.training import (
    TrainingConfig,
    TrainingGateError,
    _legacy_autoencoder_normalization,
    deterministic_withheld_ranges,
    evaluate_exported_networks,
    export_network,
    load_exported_network,
    normalized_projector_targets,
    orange_duck_decompressor_losses,
    train_flat_bundle,
    train_tiny_fixture,
)
from resources import txform
from resources.train_g1_lmm import build_parser


def _array2_bytes(values, dtype):
    values = np.ascontiguousarray(values, dtype=dtype)
    return struct.pack("<II", *values.shape[:2]) + values.tobytes(order="C")


def _array1_bytes(values, dtype):
    values = np.ascontiguousarray(values, dtype=dtype)
    return struct.pack("<I", len(values)) + values.tobytes(order="C")


def _write_flat_training_bundle(
    path: Path,
    *,
    gap_after_first_range: bool = False,
    local_rotation_step_rad: float = 0.0,
    motion_class: str = "flat-walk",
    terrain_class: str = "flat",
    coherent_parent_change: tuple[int, int] | None = None,
) -> dict:
    path.mkdir()
    frames = 256
    bones = 31
    positions = np.zeros((frames, bones, 3), dtype=np.float32)
    positions[:, 1:, 1] = np.arange(1, bones, dtype=np.float32)[None] * 0.01
    positions[:, 0, 0] = np.arange(frames, dtype=np.float32) / 60.0
    velocities = np.zeros_like(positions)
    velocities[:, 0, 0] = 1.0
    rotations = np.zeros((frames, bones, 4), dtype=np.float32)
    rotations[..., 0] = 1.0
    if local_rotation_step_rad:
        rotations[5:, 1, 0] = np.cos(local_rotation_step_rad / 2.0)
        rotations[5:, 1, 1] = np.sin(local_rotation_step_rad / 2.0)
    angular_velocities = np.zeros_like(positions)
    parents = np.asarray(G1_SKELETON_PARENTS, dtype=np.int32)
    if coherent_parent_change is not None:
        child, parent = coherent_parent_change
        parents[child] = parent
    range_starts = np.array([0], dtype=np.int32)
    range_stops = np.array([frames], dtype=np.int32)
    if gap_after_first_range:
        range_starts = np.array([0, 129], dtype=np.int32)
        range_stops = np.array([128, frames], dtype=np.int32)
    contacts = np.zeros((frames, 2), dtype=np.uint8)
    for start, length in ((30, 49), (90, 40), (150, 27)):
        contacts[start : start + length, 0] = 1
    for start, length in ((0, 45), (60, 30), (120, 30), (180, 12)):
        contacts[start : start + length, 1] = 1
    database_payload = b"".join(
        (
            _array2_bytes(positions, "<f4"),
            _array2_bytes(velocities, "<f4"),
            _array2_bytes(rotations, "<f4"),
            _array2_bytes(angular_velocities, "<f4"),
            _array1_bytes(parents, "<i4"),
            _array1_bytes(range_starts, "<i4"),
            _array1_bytes(range_stops, "<i4"),
            _array2_bytes(contacts, "u1"),
        )
    )
    (path / "database.bin").write_bytes(database_payload)

    feature_values = np.zeros((frames, 31), dtype=np.float32)
    feature_values[:, 0] = np.linspace(-1.0, 1.0, frames, dtype=np.float32)
    feature_offset = np.zeros(31, dtype=np.float32)
    feature_scale = np.ones(31, dtype=np.float32)
    feature_payload = b"".join(
        (
            _array2_bytes(feature_values, "<f4"),
            _array1_bytes(feature_offset, "<f4"),
            _array1_bytes(feature_scale, "<f4"),
        )
    )
    (path / "features.bin").write_bytes(feature_payload)
    artifacts = {
        name: {
            "path": name,
            "size_bytes": (path / name).stat().st_size,
            "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest(),
        }
        for name in ("database.bin", "features.bin")
    }
    left_source_index = np.arange(7659, 8171, 2, dtype=np.int32)
    right_source_index = left_source_index.copy()
    source_alpha = np.zeros(frames, dtype=np.float32)
    source_map_digest = hashlib.sha256(
        b"".join(
            (
                left_source_index.astype("<i4", copy=False).tobytes(order="C"),
                right_source_index.astype("<i4", copy=False).tobytes(order="C"),
                source_alpha.astype("<f4", copy=False).tobytes(order="C"),
            )
        )
    ).hexdigest()
    manifest = {
        "schema": "g1-lmm-flat-data/v3",
        "status": "accepted",
        "output_fps": 60.0,
        "trajectory_horizons": [20, 40, 60],
        "feature_dimensions": 31,
        "database_frames": frames,
        "total_clips": 1,
        "source_count": 1,
        "range_count": len(range_starts),
        "dimensions": {"bones": 31, "features": 31, "contacts": 2},
        "skeleton": {
            "names": list(G1_SKELETON_NAMES),
            "parents": parents.tolist(),
            "basis": "holden-y-up-right-handed-forward-plus-z",
            "signature": (
                G1_SKELETON_SIGNATURE
                if coherent_parent_change is None
                else hashlib.sha256(
                    json.dumps(
                        {
                            "names": list(G1_SKELETON_NAMES),
                            "parents": parents.tolist(),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
            ),
        },
        "kinematics_model": {
            "asset": "g1_29dof.xml",
            "sha256": "749209c06a5c0023deb27f728420028b62b1f3092a22e24920183c1a897e4376",
            "size_bytes": 26914,
        },
        "ranges": [
            {
                "start": int(start),
                "stop": int(stop),
                "source": "LocomotionFlat01_000-walk-only-7659-8171-120hz",
                "source_first_frame": int(left_source_index[start]),
                "source_last_frame": int(right_source_index[stop - 1]),
                "motion_class": motion_class,
                "terrain_class": terrain_class,
            }
            for start, stop in zip(range_starts, range_stops)
        ],
        "continuity": {
            "schema": "g1-lmm-continuity/v1",
            "threshold_rad_per_frame": 0.25,
            "minimum_range_frames": 61,
            "source_native_rejected_edge_count": 0,
            "database_local_rejected_edge_count": 0,
            "union_rejected_edge_count": 0,
            "dropped_fragment_count": 0,
            "dropped_frame_count": 0,
            "published_range_count": len(range_starts),
            "published_frame_count": frames,
            "maximum_admitted_native_step_rad": local_rotation_step_rad,
            "maximum_admitted_local_rotation_step_rad": local_rotation_step_rad,
            "range_digest_sha256": hashlib.sha256(
                np.asarray(
                    [
                        [
                            int(start),
                            int(stop),
                            int(left_source_index[start]),
                            int(right_source_index[stop - 1]),
                        ]
                        for start, stop in zip(range_starts, range_stops)
                    ],
                    dtype="<i4",
                ).tobytes(order="C")
            ).hexdigest(),
            "source_map_digest_sha256": source_map_digest,
        },
        "sources": [
            {
                "name": "LocomotionFlat01_000-walk-only-7659-8171-120hz",
                "terrain_id": "flat",
                "path": "/synthetic/LocomotionFlat01_000-walk-only-7659-8171-120hz.npz",
                "sha256": "bbdeb79760950480582ae937e54b913c376caa49f344896a8958476b82f3317f",
                "receipt_path": "/synthetic/LocomotionFlat01_000-walk-only-7659-8171-120hz.receipt.json",
                "receipt_sha256": "2d0e93f485bab9c54773c66cf07c14d25837f5f4e50f5c007f9f8e2c5c20520f",
                "receipt_schema": "native-g1-pfnn-sample-retarget/v1",
                "receipt_status": "accepted",
                "source_fps": 120.0,
                "source_frames": 512,
                "output_frames": frames,
                "left_source_index": left_source_index.tolist(),
                "right_source_index": right_source_index.tolist(),
                "source_alpha": source_alpha.tolist(),
            }
        ],
        "time_filters": {
            "root_position_frames": 31,
            "root_position_order": 3,
            "root_direction_frames": 61,
            "root_direction_order": 3,
            "contact_median_frames": 6,
            "forward_terrain_path_rows": 121,
        },
        "contact": {
            "semantics": "bundled-orange-duck-global-toe-speed-only",
            "speed_threshold": 0.15,
            "median_filter_frames": 6,
            "median_filter_mode": "nearest",
        },
        "contact_observations": {
            "schema": "g1-lmm-bilateral-contact/v1",
            "left_contact_frames": 116,
            "right_contact_frames": 117,
            "left_run_count": 3,
            "right_run_count": 4,
            "left_max_run_frames": 49,
            "right_max_run_frames": 45,
            "alternating_run_transition_count": 6,
        },
        "artifacts": artifacts,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return manifest


def _orange_duck_loss_fixture(frames=4):
    bones = 31
    parents = np.concatenate(([-1], np.arange(bones - 1))).astype(np.int32)
    local_position = torch.zeros(frames, bones, 3)
    local_position[:, 1:, 1] = 0.01
    local_rotation_xy = torch.zeros(frames, bones, 3, 2)
    local_rotation_xy[..., 0, 0] = 1.0
    local_rotation_xy[..., 1, 1] = 1.0
    local_velocity = torch.zeros_like(local_position)
    local_angular_velocity = torch.zeros_like(local_position)
    local_transform = txform.from_xy(local_rotation_xy)
    global_transform, global_position, global_velocity, global_angular_velocity = (
        txform.fk_vel(
            local_transform,
            local_position,
            local_velocity,
            local_angular_velocity,
            parents,
        )
    )
    character_transform = txform.inv_mul(
        global_transform[:, 0:1], global_transform
    )
    character_position = txform.inv_mul_vec(
        global_transform[:, 0:1], global_position - global_position[:, 0:1]
    )
    character_velocity = txform.inv_mul_vec(
        global_transform[:, 0:1], global_velocity
    )
    character_angular_velocity = txform.inv_mul_vec(
        global_transform[:, 0:1], global_angular_velocity
    )
    root_velocity = torch.zeros(frames, 3)
    root_angular_velocity = torch.zeros(frames, 3)
    contacts = torch.zeros(frames, 2)
    target = torch.cat(
        (
            local_position[:, 1:].reshape(frames, -1),
            local_rotation_xy[:, 1:].reshape(frames, -1),
            local_velocity[:, 1:].reshape(frames, -1),
            local_angular_velocity[:, 1:].reshape(frames, -1),
            root_velocity,
            root_angular_velocity,
            contacts,
        ),
        dim=1,
    )
    ground = {
        "local_positions": local_position,
        "local_rotation_xy": local_rotation_xy,
        "local_velocities": local_velocity,
        "local_angular_velocities": local_angular_velocity,
        "character_positions": character_position,
        "character_transforms": character_transform,
        "character_velocities": character_velocity,
        "character_angular_velocities": character_angular_velocity,
        "root_velocity": root_velocity,
        "root_angular_velocity": root_angular_velocity,
        "contacts": contacts,
    }
    return parents, target, ground


class G1LmmTrainingTest(unittest.TestCase):
    def test_orange_duck_decompressor_loss_matches_every_reference_term(self):
        parents, target, rows = _orange_duck_loss_fixture()
        batch = torch.tensor([[0, 1]], dtype=torch.long)
        ground = {name: values[batch] for name, values in rows.items()}
        delta = torch.linspace(-0.01, 0.01, target.shape[1]).reshape(1, 1, -1)
        prediction = target[batch] + delta * torch.tensor([1.0, 2.0]).reshape(1, 2, 1)
        latent = torch.linspace(-0.2, 0.3, 64).reshape(1, 2, 32)

        actual = orange_duck_decompressor_losses(
            prediction,
            latent,
            parents=parents,
            dt=1.0 / 60.0,
            **ground,
        )

        non_root = 30
        position_end = 3 * non_root
        rotation_end = 9 * non_root
        velocity_end = 12 * non_root
        angular_end = 15 * non_root
        predicted_position = prediction[..., :position_end].reshape(1, 2, non_root, 3)
        predicted_xy = prediction[..., position_end:rotation_end].reshape(
            1, 2, non_root, 3, 2
        )
        predicted_velocity = prediction[..., rotation_end:velocity_end].reshape(
            1, 2, non_root, 3
        )
        predicted_angular = prediction[..., velocity_end:angular_end].reshape(
            1, 2, non_root, 3
        )
        predicted_root_velocity = prediction[..., angular_end : angular_end + 3]
        predicted_root_angular = prediction[..., angular_end + 3 : angular_end + 6]
        predicted_contacts = prediction[..., angular_end + 6 :]
        predicted_position = torch.cat(
            (ground["local_positions"][..., :1, :], predicted_position), dim=-2
        )
        predicted_xy = torch.cat(
            (ground["local_rotation_xy"][..., :1, :, :], predicted_xy), dim=-3
        )
        predicted_velocity = torch.cat(
            (ground["local_velocities"][..., :1, :], predicted_velocity), dim=-2
        )
        predicted_angular = torch.cat(
            (ground["local_angular_velocities"][..., :1, :], predicted_angular),
            dim=-2,
        )
        predicted_transform = txform.from_xy(predicted_xy)
        global_transform, global_position, global_velocity, global_angular = (
            txform.fk_vel(
                predicted_transform,
                predicted_position,
                predicted_velocity,
                predicted_angular,
                parents,
            )
        )
        character_transform = txform.inv_mul(
            global_transform[..., 0:1, :, :], global_transform
        )
        character_position = txform.inv_mul_vec(
            global_transform[..., 0:1, :, :],
            global_position - global_position[..., 0:1, :],
        )
        character_velocity = txform.inv_mul_vec(
            global_transform[..., 0:1, :, :], global_velocity
        )
        character_angular = txform.inv_mul_vec(
            global_transform[..., 0:1, :, :], global_angular
        )
        inverse_dt = 60.0
        expected = {
            "local_position": torch.mean(
                75.0 * torch.abs(ground["local_positions"] - predicted_position)
            ),
            "local_rotation_xy": torch.mean(
                10.0 * torch.abs(ground["local_rotation_xy"] - predicted_xy)
            ),
            "local_velocity": torch.mean(
                10.0 * torch.abs(ground["local_velocities"] - predicted_velocity)
            ),
            "local_angular_velocity": torch.mean(
                1.25
                * torch.abs(
                    ground["local_angular_velocities"] - predicted_angular
                )
            ),
            "root_velocity": torch.mean(
                2.0 * torch.abs(ground["root_velocity"] - predicted_root_velocity)
            ),
            "root_angular_velocity": torch.mean(
                2.0
                * torch.abs(
                    ground["root_angular_velocity"] - predicted_root_angular
                )
            ),
            "contacts": torch.mean(
                2.0 * torch.abs(ground["contacts"] - predicted_contacts)
            ),
            "character_position": torch.mean(
                15.0
                * torch.abs(ground["character_positions"] - character_position)
            ),
            "character_transform": torch.mean(
                5.0
                * torch.abs(ground["character_transforms"] - character_transform)
            ),
            "character_velocity": torch.mean(
                2.0
                * torch.abs(ground["character_velocities"] - character_velocity)
            ),
            "character_angular_velocity": torch.mean(
                0.75
                * torch.abs(
                    ground["character_angular_velocities"] - character_angular
                )
            ),
            "local_position_delta": torch.mean(
                10.0
                * torch.abs(
                    torch.diff(ground["local_positions"], dim=1) * inverse_dt
                    - torch.diff(predicted_position, dim=1) * inverse_dt
                )
            ),
            "local_rotation_xy_delta": torch.mean(
                1.75
                * torch.abs(
                    torch.diff(ground["local_rotation_xy"], dim=1) * inverse_dt
                    - torch.diff(predicted_xy, dim=1) * inverse_dt
                )
            ),
            "character_position_delta": torch.mean(
                2.0
                * torch.abs(
                    torch.diff(ground["character_positions"], dim=1) * inverse_dt
                    - torch.diff(character_position, dim=1) * inverse_dt
                )
            ),
            "character_transform_delta": torch.mean(
                0.75
                * torch.abs(
                    torch.diff(ground["character_transforms"], dim=1) * inverse_dt
                    - torch.diff(character_transform, dim=1) * inverse_dt
                )
            ),
            "latent_l1": torch.mean(0.1 * torch.abs(latent)),
            "latent_l2": torch.mean(0.1 * torch.square(latent)),
            "latent_velocity": torch.mean(
                0.01 * torch.abs(torch.diff(latent, dim=1) * inverse_dt)
            ),
        }
        expected["total"] = sum(expected.values())
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-6, atol=1e-7)

    def test_orange_duck_temporal_losses_do_not_cross_range_boundaries(self):
        parents, target, rows = _orange_duck_loss_fixture()
        windows = range_safe_windows(
            np.array([0, 2], dtype=np.int32),
            np.array([2, 4], dtype=np.int32),
            2,
        )
        np.testing.assert_array_equal(windows, np.array([[0, 1], [2, 3]]))
        batch = torch.as_tensor(windows)
        prediction = target.clone()
        prediction[2:, 0] += 0.1
        latent = torch.zeros(4, 32)
        latent[2:] = 1.0
        safe = orange_duck_decompressor_losses(
            prediction[batch],
            latent[batch],
            parents=parents,
            dt=1.0 / 60.0,
            **{name: values[batch] for name, values in rows.items()},
        )
        self.assertEqual(float(safe["local_position_delta"]), 0.0)
        self.assertEqual(float(safe["character_position_delta"]), 0.0)
        self.assertEqual(float(safe["latent_velocity"]), 0.0)

        crossing = torch.tensor([[1, 2]])
        unsafe = orange_duck_decompressor_losses(
            prediction[crossing],
            latent[crossing],
            parents=parents,
            dt=1.0 / 60.0,
            **{name: values[crossing] for name, values in rows.items()},
        )
        self.assertGreater(float(unsafe["local_position_delta"]), 0.0)
        self.assertGreater(float(unsafe["latent_velocity"]), 0.0)

    def test_orange_duck_physical_loss_corrects_low_mse_high_fk_error(self):
        parents, target, rows = _orange_duck_loss_fixture(frames=2)
        prediction = target.clone()
        theta = 0.05
        prediction[:, 90:96] = torch.tensor(
            [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta), 0.0, 0.0]
        )
        prediction.requires_grad_(True)
        batch = torch.tensor([[0, 1]])
        losses = orange_duck_decompressor_losses(
            prediction[batch],
            torch.zeros(1, 2, 32),
            parents=parents,
            dt=1.0 / 60.0,
            **{name: values[batch] for name, values in rows.items()},
        )
        normalized_mse = torch.mean(torch.square(prediction - target))
        self.assertLess(float(normalized_mse.detach()), 1.0e-4)

        predicted_xy = prediction[:, 90:270].reshape(2, 30, 3, 2)
        local_xy = torch.cat((rows["local_rotation_xy"][:, :1], predicted_xy), dim=1)
        _, predicted_global, _, _ = txform.fk_vel(
            txform.from_xy(local_xy),
            rows["local_positions"],
            rows["local_velocities"],
            rows["local_angular_velocities"],
            parents,
        )
        _, actual_global, _, _ = txform.fk_vel(
            txform.from_xy(rows["local_rotation_xy"]),
            rows["local_positions"],
            rows["local_velocities"],
            rows["local_angular_velocities"],
            parents,
        )
        fk_max = torch.linalg.vector_norm(
            predicted_global - actual_global, dim=-1
        ).max()
        self.assertGreater(float(fk_max.detach()), 0.01)
        losses["total"].backward()
        self.assertGreater(float(torch.linalg.vector_norm(prediction.grad[:, 90:96])), 0.0)

    def test_autoencoder_normalization_uses_only_explicit_fitted_admitted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory) / "data"
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)
            fitted = bundle.admitted_mask.copy()
            fitted[100:224] = False

            baseline = _legacy_autoencoder_normalization(bundle, arrays, fitted)
            compressor = arrays.compressor_input.copy()
            target = arrays.decompressor_target.copy()
            compressor[~fitted] += np.float32(10_000.0)
            target[~fitted] -= np.float32(10_000.0)
            changed = _legacy_autoencoder_normalization(
                bundle,
                replace(
                    arrays,
                    compressor_input=compressor,
                    decompressor_target=target,
                ),
                fitted,
            )

            for field in (
                "compressor_mean",
                "compressor_std",
                "decompressor_mean",
                "decompressor_std",
            ):
                np.testing.assert_array_equal(
                    getattr(baseline, field), getattr(changed, field)
                )

    def test_training_loop_exception_publishes_rejected_diagnostics_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "rejected"
            _write_flat_training_bundle(data_directory)
            accepted_overfit = {
                "accepted": True,
                "final_loss": 0.01,
                "frames": 64,
                "initial_loss": 1.0,
                "range": [0, 64],
                "steps": 1,
            }
            with (
                mock.patch(
                    "resources.g1_lmm.training._train_64_frame_overfit",
                    return_value=accepted_overfit,
                ),
                mock.patch(
                    "resources.g1_lmm.training._train_decompressor_stage",
                    side_effect=FloatingPointError("nonfinite loss"),
                ),
            ):
                with self.assertRaisesRegex(TrainingGateError, "decompressor"):
                    train_flat_bundle(
                        data_directory,
                        output_directory,
                        config=TrainingConfig(
                            device="cpu",
                            overfit_steps=1,
                            decompressor_steps=1,
                            stepper_steps=1,
                            projector_steps=1,
                        ),
                        stage="all",
                    )
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"training.json", "evaluation.json"},
            )
            receipt = json.loads(
                (output_directory / "training.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["status"], "rejected")
            self.assertEqual(receipt["stopped_after"], "decompressor")
            self.assertEqual(receipt["diagnostics"]["error_type"], "FloatingPointError")
            self.assertEqual(receipt["diagnostics"]["message"], "nonfinite loss")

    def test_nonfinite_gate_atomically_publishes_sanitized_rejected_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "rejected"
            _write_flat_training_bundle(data_directory)
            nonfinite_gate = {
                "accepted": False,
                "final_loss": float("nan"),
                "frames": 64,
                "initial_loss": float("inf"),
                "range": [0, 64],
                "steps": 1,
            }
            with mock.patch(
                "resources.g1_lmm.training._train_64_frame_overfit",
                return_value=nonfinite_gate,
            ):
                with self.assertRaisesRegex(TrainingGateError, "overfit"):
                    train_flat_bundle(
                        data_directory,
                        output_directory,
                        config=TrainingConfig(
                            device="cpu",
                            overfit_steps=1,
                            decompressor_steps=1,
                            stepper_steps=1,
                            projector_steps=1,
                        ),
                        stage="all",
                    )
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"training.json", "evaluation.json"},
            )
            raw = (output_directory / "training.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            receipt = json.loads(raw)
            self.assertEqual(receipt["status"], "rejected")
            self.assertFalse(receipt["accepted"])
            self.assertIsNone(receipt["overfit_gate"]["final_loss"])
            self.assertIsNone(receipt["overfit_gate"]["initial_loss"])

    def test_training_config_rejects_any_dt_other_than_exact_60hz(self):
        self.assertEqual(TrainingConfig().dt, 1.0 / 60.0)
        with self.assertRaisesRegex(ValueError, "exactly 1/60"):
            TrainingConfig(dt=1.0 / 30.0)
        with self.assertRaisesRegex(ValueError, "exactly 1/60"):
            TrainingConfig(dt=np.nextafter(1.0 / 60.0, 1.0))

    def test_single_clip_overfit_canary_config_and_cli_are_strictly_opt_in(self):
        self.assertIs(TrainingConfig().single_clip_overfit_canary, False)
        self.assertIs(
            build_parser()
            .parse_args(["data", "model"])
            .single_clip_overfit_canary,
            False,
        )
        self.assertIs(
            build_parser()
            .parse_args(["data", "model", "--single-clip-overfit-canary"])
            .single_clip_overfit_canary,
            True,
        )
        self.assertIs(
            TrainingConfig(single_clip_overfit_canary=True).single_clip_overfit_canary,
            True,
        )
        for value in (1, np.bool_(True), "true", None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "single_clip_overfit_canary.*bool"),
            ):
                TrainingConfig(single_clip_overfit_canary=value)

    def test_loader_requires_exact_flat_walking_labels_for_every_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_motion = root / "wrong-motion"
            _write_flat_training_bundle(
                wrong_motion, motion_class="idle"
            )
            with self.assertRaisesRegex(ValueError, "flat-walk"):
                load_training_bundle(wrong_motion)

            wrong_terrain = root / "wrong-terrain"
            _write_flat_training_bundle(
                wrong_terrain, terrain_class="slope"
            )
            with self.assertRaisesRegex(ValueError, "terrain.*flat"):
                load_training_bundle(wrong_terrain)

    def test_loader_requires_one_source_and_exact_source_bound_range_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split-source"
            manifest = _write_flat_training_bundle(split)
            source = manifest["sources"][0]
            cut = 128
            first = {
                **source,
                "name": "synthetic-flat-walk-a",
                "output_frames": cut,
                "left_source_index": source["left_source_index"][:cut],
                "right_source_index": source["right_source_index"][:cut],
                "source_alpha": source["source_alpha"][:cut],
            }
            second = {
                **source,
                "name": "synthetic-flat-walk-b",
                "output_frames": 256 - cut,
                "left_source_index": source["left_source_index"][cut:],
                "right_source_index": source["right_source_index"][cut:],
                "source_alpha": source["source_alpha"][cut:],
            }
            manifest["sources"] = [first, second]
            manifest["source_count"] = 2
            manifest["total_clips"] = 2
            for entry in manifest["ranges"]:
                entry["source"] = (
                    first["name"] if entry["start"] < cut else second["name"]
                )
            (split / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_training_bundle(split)

            missing = root / "missing-source"
            manifest = _write_flat_training_bundle(missing)
            del manifest["ranges"][0]["source"]
            (missing / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact keys"):
                load_training_bundle(missing)

            changed = root / "changed-source"
            manifest = _write_flat_training_bundle(changed)
            manifest["ranges"][0]["source"] = "another-source"
            (changed / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sole source"):
                load_training_bundle(changed)

            extra = root / "extra-range-key"
            manifest = _write_flat_training_bundle(extra)
            manifest["ranges"][0]["unexpected"] = True
            (extra / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exact keys"):
                load_training_bundle(extra)

    def test_loader_rejects_a_gap_in_the_admitted_range_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory) / "gapped"
            _write_flat_training_bundle(
                data_directory, gap_after_first_range=True
            )
            with self.assertRaisesRegex(ValueError, "contiguous partition"):
                load_training_bundle(data_directory)

    def test_loader_requires_v3_continuity_receipt_and_recomputes_local_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in ("v1", "v2"):
                legacy = root / version
                manifest = _write_flat_training_bundle(legacy)
                manifest["schema"] = f"g1-lmm-flat-data/{version}"
                (legacy / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                with (
                    self.subTest(version=version),
                    self.assertRaisesRegex(ValueError, "v3"),
                ):
                    load_training_bundle(legacy)

            unsafe = root / "unsafe"
            _write_flat_training_bundle(unsafe, local_rotation_step_rad=0.5)
            with self.assertRaisesRegex(ValueError, "0.25"):
                load_training_bundle(unsafe)

            mismatch = root / "mismatch"
            manifest = _write_flat_training_bundle(mismatch)
            manifest["continuity"]["maximum_admitted_local_rotation_step_rad"] = 0.1
            (mismatch / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "recomputed"):
                load_training_bundle(mismatch)

            source_map_mismatch = root / "source-map-mismatch"
            manifest = _write_flat_training_bundle(source_map_mismatch)
            manifest["continuity"]["source_map_digest_sha256"] = "0" * 64
            (source_map_mismatch / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source-map digest"):
                load_training_bundle(source_map_mismatch)

            rounded_quaternions = root / "rounded-quaternions"
            _write_flat_training_bundle(
                rounded_quaternions, local_rotation_step_rad=0.01
            )
            rounded = load_training_bundle(rounded_quaternions)
            self.assertAlmostEqual(
                rounded.manifest["continuity"][
                    "maximum_admitted_local_rotation_step_rad"
                ],
                0.01,
            )

    def test_loader_requires_canonical_v3_walk_source_and_orange_duck_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            stale_source = root / "stale-source"
            manifest = _write_flat_training_bundle(stale_source)
            manifest["sources"][0]["name"] = "LocomotionFlat01_000-120hz"
            manifest["ranges"][0]["source"] = "LocomotionFlat01_000-120hz"
            (stale_source / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "walk-only"):
                load_training_bundle(stale_source)

            changed_contact = root / "changed-contact"
            manifest = _write_flat_training_bundle(changed_contact)
            manifest["contact"]["semantics"] = "height-and-speed"
            (changed_contact / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Orange Duck contact"):
                load_training_bundle(changed_contact)

            changed_observations = root / "changed-observations"
            manifest = _write_flat_training_bundle(changed_observations)
            manifest["contact_observations"]["left_contact_frames"] -= 1
            (changed_observations / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "contact observations"):
                load_training_bundle(changed_observations)

            extra_source_key = root / "extra-source-key"
            manifest = _write_flat_training_bundle(extra_source_key)
            manifest["sources"][0]["unexpected"] = True
            (extra_source_key / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source receipt keys"):
                load_training_bundle(extra_source_key)

    def test_loader_requires_exact_canonical_kinematics_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                ("missing", None),
                ("asset", "g1.xml"),
                ("sha256", "0" * 64),
                ("size_bytes", 26913),
                ("extra", True),
            )
            for name, value in cases:
                data_directory = root / name
                manifest = _write_flat_training_bundle(data_directory)
                if name == "missing":
                    del manifest["kinematics_model"]
                elif name == "extra":
                    manifest["kinematics_model"]["unexpected"] = value
                else:
                    manifest["kinematics_model"][name] = value
                (data_directory / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "kinematics model"),
                ):
                    load_training_bundle(data_directory)

    def test_loader_authenticates_exact_canonical_skeleton_and_database_parents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []

            missing = root / "missing"
            manifest = _write_flat_training_bundle(missing)
            del manifest["skeleton"]
            cases.append(("missing", missing, manifest))

            reordered = root / "reordered"
            manifest = _write_flat_training_bundle(reordered)
            manifest["skeleton"]["names"][2:4] = reversed(
                manifest["skeleton"]["names"][2:4]
            )
            cases.append(("reordered", reordered, manifest))

            reparented = root / "reparented"
            manifest = _write_flat_training_bundle(reparented)
            manifest["skeleton"]["parents"][30] = 28
            cases.append(("reparented", reparented, manifest))

            changed_basis = root / "basis"
            manifest = _write_flat_training_bundle(changed_basis)
            manifest["skeleton"]["basis"] = "z-up"
            cases.append(("basis", changed_basis, manifest))

            changed_signature = root / "signature"
            manifest = _write_flat_training_bundle(changed_signature)
            manifest["skeleton"]["signature"] = "0" * 64
            cases.append(("signature", changed_signature, manifest))

            coherent = root / "coherent-db-manifest"
            manifest = _write_flat_training_bundle(
                coherent, coherent_parent_change=(30, 28)
            )
            cases.append(("coherent", coherent, manifest))

            database_only = root / "coherent-database-artifact"
            manifest = _write_flat_training_bundle(
                database_only, coherent_parent_change=(30, 28)
            )
            manifest["skeleton"] = {
                "names": list(G1_SKELETON_NAMES),
                "parents": list(G1_SKELETON_PARENTS),
                "basis": "holden-y-up-right-handed-forward-plus-z",
                "signature": G1_SKELETON_SIGNATURE,
            }
            cases.append(("database", database_only, manifest))

            for name, data_directory, manifest in cases:
                (data_directory / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "canonical.*skeleton"),
                ):
                    load_training_bundle(data_directory)

    def test_loader_validates_source_map_json_losslessly_before_numpy_coercion(self):
        def bind_source_map_digest(manifest):
            source = manifest["sources"][0]
            manifest["continuity"]["source_map_digest_sha256"] = hashlib.sha256(
                b"".join(
                    (
                        np.asarray(
                            source["left_source_index"], dtype="<i4"
                        ).tobytes(order="C"),
                        np.asarray(
                            source["right_source_index"], dtype="<i4"
                        ).tobytes(order="C"),
                        np.asarray(
                            source["source_alpha"], dtype="<f4"
                        ).tobytes(order="C"),
                    )
                )
            ).hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = (
                ("fractional-index", "left_source_index", 0, 7659.9),
                ("boolean-index", "left_source_index", 0, True),
                ("string-index", "right_source_index", 0, "7659"),
                ("boolean-alpha", "source_alpha", 0, False),
                ("integer-alpha", "source_alpha", 0, 0),
                ("string-alpha", "source_alpha", 0, "0.0"),
                ("nonfinite-alpha", "source_alpha", 0, float("nan")),
                ("negative-zero-alpha", "source_alpha", 0, -0.0),
                ("noncanonical-alpha", "source_alpha", 0, 0.9),
            )
            for name, field, index, value in malformed:
                data_directory = root / name
                manifest = _write_flat_training_bundle(data_directory)
                manifest["sources"][0][field][index] = value
                bind_source_map_digest(manifest)
                (data_directory / "manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "source map JSON"),
                ):
                    load_training_bundle(data_directory)

            coherent = root / "coherent-digest"
            manifest = _write_flat_training_bundle(coherent)
            manifest["sources"][0]["left_source_index"][0] = 7661
            manifest["sources"][0]["right_source_index"][0] = 7661
            bind_source_map_digest(manifest)
            (coherent / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "canonical v3 source interpolation map"
            ):
                load_training_bundle(coherent)

    def test_all_stage_stops_after_failed_decompressor_gate_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "rejected"
            _write_flat_training_bundle(data_directory)
            config = TrainingConfig(
                seed=1234,
                device="cpu",
                batch_size=8,
                overfit_steps=24,
                decompressor_steps=1,
                stepper_steps=1,
                projector_steps=1,
                withheld_frames=16,
                withheld_halo=2,
            )
            with self.assertRaisesRegex(TrainingGateError, "decompressor"):
                train_flat_bundle(
                    data_directory,
                    output_directory,
                    config=config,
                    stage="all",
                )
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"training.json", "evaluation.json"},
            )
            training = json.loads((output_directory / "training.json").read_text(encoding="utf-8"))
            self.assertFalse(training["accepted"])
            self.assertEqual(training["status"], "rejected")
            self.assertEqual(training["artifacts"], {})
            self.assertEqual(training["stopped_after"], "decompressor")
            decompressor_training = training["decompressor_training"]
            self.assertEqual(
                decompressor_training["objective"],
                "orange-duck-denormalized-weighted/v1",
            )
            self.assertEqual(
                set(decompressor_training["final_audit_loss_terms"]),
                {
                    "character_angular_velocity",
                    "character_position",
                    "character_position_delta",
                    "character_transform",
                    "character_transform_delta",
                    "character_velocity",
                    "contacts",
                    "latent_l1",
                    "latent_l2",
                    "latent_velocity",
                    "local_angular_velocity",
                    "local_position",
                    "local_position_delta",
                    "local_rotation_xy",
                    "local_rotation_xy_delta",
                    "local_velocity",
                    "root_angular_velocity",
                    "root_velocity",
                    "total",
                },
            )
            self.assertEqual(
                decompressor_training["final_audit_loss"],
                decompressor_training["final_audit_loss_terms"]["total"],
            )

    def test_real_bundle_overfit_stage_gates_64_adjacent_rows_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "overfit"
            _write_flat_training_bundle(data_directory)
            receipt = train_flat_bundle(
                data_directory,
                output_directory,
                config=TrainingConfig(
                    seed=1234,
                    device="cpu",
                    batch_size=32,
                    overfit_steps=24,
                    decompressor_steps=1,
                    stepper_steps=1,
                    projector_steps=1,
                ),
                stage="overfit",
            )
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt["status"], "rejected")
            self.assertTrue(receipt["overfit_gate"]["accepted"])
            self.assertEqual(receipt["overfit_gate"]["frames"], 64)
            self.assertEqual(receipt["overfit_gate"]["range"], [0, 64])
            self.assertLess(
                receipt["overfit_gate"]["final_loss"],
                0.1 * receipt["overfit_gate"]["initial_loss"],
            )
            self.assertTrue((output_directory / "training.json").is_file())
            self.assertTrue((output_directory / "evaluation.json").is_file())
            self.assertEqual(
                {path.name for path in output_directory.iterdir()},
                {"training.json", "evaluation.json"},
            )

    def test_cli_exposes_device_stage_seed_and_configurable_budgets(self):
        arguments = build_parser().parse_args(
            [
                "data",
                "model",
                "--device",
                "cuda:1",
                "--stage",
                "all",
                "--seed",
                "77",
                "--overfit-steps",
                "101",
                "--decompressor-steps",
                "202",
                "--stepper-steps",
                "303",
                "--projector-steps",
                "404",
            ]
        )
        self.assertEqual(arguments.data_directory, Path("data"))
        self.assertEqual(arguments.output_directory, Path("model"))
        self.assertEqual(arguments.device, "cuda:1")
        self.assertEqual(arguments.stage, "all")
        self.assertEqual(
            (
                arguments.seed,
                arguments.overfit_steps,
                arguments.decompressor_steps,
                arguments.stepper_steps,
                arguments.projector_steps,
            ),
            (77, 101, 202, 303, 404),
        )

    def test_deterministic_withheld_blocks_have_full_feature_and_window_halo(self):
        starts, stops = deterministic_withheld_ranges(
            np.array([0, 1000], np.int32),
            np.array([800, 1800], np.int32),
            frames=64,
            seed=1234,
        )
        self.assertEqual(starts.shape, stops.shape)
        np.testing.assert_array_equal(stops - starts, np.array([64, 64], np.int64))
        fitted = range_safe_windows(
            np.array([0, 1000], np.int32),
            np.array([800, 1800], np.int32),
            20,
            excluded_starts=starts,
            excluded_stops=stops,
            exclusion_halo=TrainingConfig().withheld_halo,
        )
        for window in fitted:
            for start, stop in zip(starts, stops):
                self.assertFalse(
                    window[0] < stop + TrainingConfig().withheld_halo
                    and start - TrainingConfig().withheld_halo <= window[-1]
                )

    def test_default_v3_withheld_block_leaves_range_safe_decompressor_windows(self):
        config = TrainingConfig()
        starts, stops = deterministic_withheld_ranges(
            np.array([0], np.int32),
            np.array([256], np.int32),
            frames=config.withheld_frames,
            seed=config.seed,
            margin=config.withheld_halo,
        )
        np.testing.assert_array_equal(starts, np.array([131], np.int64))
        np.testing.assert_array_equal(stops, np.array([195], np.int64))
        windows = range_safe_windows(
            np.array([0], np.int32),
            np.array([256], np.int32),
            2,
            excluded_starts=starts,
            excluded_stops=stops,
            exclusion_halo=config.withheld_halo,
        )
        self.assertEqual(windows.shape, (70, 2))
        np.testing.assert_array_equal(np.unique(windows), np.arange(71))

    def test_single_clip_canary_decompressor_fits_and_evaluates_all_admitted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            staging = root / "staging"
            staging.mkdir()
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)
            decoded_rows = []

            def lightweight_loss(prediction, latent, **_ground):
                total = prediction.square().mean() + latent.square().mean()
                return {"total": total}

            def accepted_metrics(_bundle, _arrays, _prediction, rows, **_kwargs):
                decoded_rows.append(np.asarray(rows, dtype=np.int64).copy())
                return {"accepted": True}

            with (
                mock.patch.object(
                    g1_lmm_training,
                    "orange_duck_decompressor_losses",
                    side_effect=lightweight_loss,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "_decode_reconstruction_metrics",
                    side_effect=accepted_metrics,
                ),
            ):
                stage = g1_lmm_training._train_decompressor_stage(
                    staging,
                    bundle,
                    arrays,
                    TrainingConfig(
                        device="cpu",
                        batch_size=2,
                        overfit_steps=1,
                        decompressor_steps=1,
                        stepper_steps=1,
                        projector_steps=1,
                        single_clip_overfit_canary=True,
                    ),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )

            self.assertEqual(stage.training_metrics["windows"], 255)
            self.assertEqual(stage.training_metrics["fitted_rows"], 256)
            self.assertTrue(stage.gate["accepted"])
            self.assertEqual(stage.gate["withheld"], [])
            self.assertEqual(
                stage.gate["evaluation_scope"], "complete-corpus-overfit"
            )
            self.assertEqual(len(decoded_rows), 1)
            np.testing.assert_array_equal(decoded_rows[0], np.arange(256))

    def test_single_clip_canary_stepper_and_projector_use_complete_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory) / "data"
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            config = TrainingConfig(single_clip_overfit_canary=True)
            empty = np.empty(0, dtype=np.int64)

            pairs = g1_lmm_training._range_safe_training_windows(
                bundle, 2, config, empty, empty
            )
            stepper = g1_lmm_training._range_safe_training_windows(
                bundle, config.stepper_window, config, empty, empty
            )
            projector = g1_lmm_training._projector_fitted_mask(
                bundle, config, empty, empty
            )

            self.assertEqual(pairs.shape, (255, 2))
            self.assertEqual(stepper.shape, (237, 20))
            np.testing.assert_array_equal(np.unique(pairs), np.arange(256))
            np.testing.assert_array_equal(np.unique(stepper), np.arange(256))
            np.testing.assert_array_equal(projector, bundle.admitted_mask)

    def test_default_decompressor_cannot_accept_an_empty_withheld_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            staging = root / "staging"
            staging.mkdir()
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)
            with self.assertRaisesRegex(ValueError, "withheld.*required"):
                g1_lmm_training._train_decompressor_stage(
                    staging,
                    bundle,
                    arrays,
                    TrainingConfig(
                        device="cpu",
                        overfit_steps=1,
                        decompressor_steps=1,
                        stepper_steps=1,
                        projector_steps=1,
                    ),
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64),
                )

    def test_single_clip_canary_partial_receipts_bind_complete_corpus_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "partial"
            _write_flat_training_bundle(data_directory)
            accepted_overfit = {
                "accepted": True,
                "final_loss": 0.01,
                "frames": 64,
                "initial_loss": 1.0,
                "range": [0, 64],
                "steps": 1,
            }
            accepted_autoencoder = mock.Mock(
                gate={
                    "accepted": True,
                    "evaluation_scope": "complete-corpus-overfit",
                    "fitted": {"accepted": True},
                    "withheld": [],
                },
                training_metrics={"fitted_rows": 256, "windows": 255},
            )
            with (
                mock.patch.object(
                    g1_lmm_training,
                    "_train_64_frame_overfit",
                    return_value=accepted_overfit,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "_train_decompressor_stage",
                    return_value=accepted_autoencoder,
                ) as train_decompressor,
            ):
                receipt = train_flat_bundle(
                    data_directory,
                    output_directory,
                    config=TrainingConfig(
                        device="cpu",
                        overfit_steps=1,
                        decompressor_steps=1,
                        stepper_steps=1,
                        projector_steps=1,
                        single_clip_overfit_canary=True,
                    ),
                    stage="decompressor",
                )

            call = train_decompressor.call_args.args
            self.assertEqual(call[-2].shape, (0,))
            self.assertEqual(call[-1].shape, (0,))
            self.assertEqual(receipt["withheld"], [])
            self.assertEqual(
                receipt["model_scope"], "single-clip-overfit-canary"
            )
            self.assertEqual(
                receipt["evaluation_scope"], "complete-corpus-overfit"
            )
            self.assertIs(receipt["config"]["single_clip_overfit_canary"], True)
            for name in ("training.json", "evaluation.json"):
                published = json.loads(
                    (output_directory / name).read_text(encoding="utf-8")
                )
                self.assertEqual(published["status"], "rejected")
                self.assertEqual(
                    published["model_scope"], "single-clip-overfit-canary"
                )
                self.assertEqual(
                    published["evaluation_scope"], "complete-corpus-overfit"
                )
                self.assertEqual(published["withheld"], [])

    def test_default_train_path_still_selects_the_canonical_withheld_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "partial"
            _write_flat_training_bundle(data_directory)
            accepted_overfit = {
                "accepted": True,
                "final_loss": 0.01,
                "frames": 64,
                "initial_loss": 1.0,
                "range": [0, 64],
                "steps": 1,
            }
            rejected_autoencoder = mock.Mock(
                gate={
                    "accepted": False,
                    "fitted": {"accepted": True},
                    "withheld": [{"accepted": False, "range": [131, 195]}],
                },
                training_metrics={"fitted_rows": 71, "windows": 70},
            )
            with (
                mock.patch.object(
                    g1_lmm_training,
                    "_train_64_frame_overfit",
                    return_value=accepted_overfit,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "_train_decompressor_stage",
                    return_value=rejected_autoencoder,
                ) as train_decompressor,
                self.assertRaisesRegex(TrainingGateError, "decompressor"),
            ):
                train_flat_bundle(
                    data_directory,
                    output_directory,
                    config=TrainingConfig(
                        device="cpu",
                        overfit_steps=1,
                        decompressor_steps=1,
                        stepper_steps=1,
                        projector_steps=1,
                    ),
                    stage="all",
                )

            call = train_decompressor.call_args.args
            np.testing.assert_array_equal(call[-2], np.array([131]))
            np.testing.assert_array_equal(call[-1], np.array([195]))
            training = json.loads(
                (output_directory / "training.json").read_text(encoding="utf-8")
            )
            self.assertEqual(training["withheld"]["ranges"], [[131, 195]])
            self.assertIs(training["config"]["single_clip_overfit_canary"], False)

    def test_accepted_single_clip_canary_manifest_binds_exact_model_and_evaluation_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            output_directory = root / "accepted"
            _write_flat_training_bundle(data_directory)
            accepted_overfit = {
                "accepted": True,
                "final_loss": 0.01,
                "frames": 64,
                "initial_loss": 1.0,
                "range": [0, 64],
                "steps": 1,
            }

            def fake_decompressor(staging, bundle, *_args):
                (staging / "latent.bin").write_bytes(b"latent")
                (staging / "decompressor.bin").write_bytes(b"decompressor")
                return mock.Mock(
                    gate={
                        "accepted": True,
                        "evaluation_scope": "complete-corpus-overfit",
                        "fitted": {"accepted": True},
                        "withheld": [],
                    },
                    latent=np.zeros(
                        (bundle.frames, bundle.dimensions.latent), dtype=np.float32
                    ),
                    training_metrics={"fitted_rows": 256, "windows": 255},
                )

            def fake_stepper(staging, *_args):
                (staging / "stepper.bin").write_bytes(b"stepper")
                return mock.Mock(
                    gate={"accepted": True}, training_metrics={"windows": 237}
                )

            def fake_projector(staging, *_args):
                (staging / "projector.bin").write_bytes(b"projector")
                return mock.Mock(
                    gate={"accepted": True}, training_metrics={"oracle_rows": 256}
                )

            with (
                mock.patch.object(
                    g1_lmm_training,
                    "_train_64_frame_overfit",
                    return_value=accepted_overfit,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "_train_decompressor_stage",
                    side_effect=fake_decompressor,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "_train_stepper_stage",
                    side_effect=fake_stepper,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "_train_projector_stage",
                    side_effect=fake_projector,
                ),
                mock.patch.object(
                    g1_lmm_training,
                    "evaluate_exported_networks",
                    return_value={"finite": True},
                ),
            ):
                receipt = train_flat_bundle(
                    data_directory,
                    output_directory,
                    config=TrainingConfig(
                        device="cpu",
                        overfit_steps=1,
                        decompressor_steps=1,
                        stepper_steps=1,
                        projector_steps=1,
                        single_clip_overfit_canary=True,
                    ),
                    stage="all",
                )

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(receipt["accepted"])
            self.assertEqual(
                manifest["model_scope"], "single-clip-overfit-canary"
            )
            self.assertIs(type(manifest["evaluation_scope"]), str)
            self.assertEqual(
                manifest["evaluation_scope"], "complete-corpus-overfit-canary"
            )
            self.assertEqual(
                receipt["evaluation_scope"], "complete-corpus-overfit"
            )
            self.assertEqual(receipt["decompressor_gate"]["withheld"], [])

    def test_stepper_gate_uses_exact_one_two_four_second_source_horizons(self):
        class ZeroStepper(torch.nn.Module):
            def forward(self, state):
                return torch.zeros_like(state)

        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory) / "data"
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)
            autoencoder = mock.Mock(
                latent=np.zeros(
                    (bundle.frames, bundle.dimensions.latent), dtype=np.float32
                )
            )
            config = TrainingConfig(device="cpu")
            state_dimensions = bundle.dimensions.state

            def exact_source_decode(_autoencoder, state_rows, _device):
                return arrays.decompressor_target[: len(state_rows)].copy()

            with mock.patch.object(
                g1_lmm_training,
                "_decode_state_rows",
                side_effect=exact_source_decode,
            ):
                gate = g1_lmm_training._stepper_rollout_gate(
                    bundle,
                    arrays,
                    autoencoder,
                    ZeroStepper(),
                    np.zeros(state_dimensions, dtype=np.float32),
                    np.ones(state_dimensions, dtype=np.float32),
                    np.zeros(state_dimensions, dtype=np.float32),
                    np.ones(state_dimensions, dtype=np.float32),
                    config,
                )

            self.assertTrue(gate["accepted"])
            self.assertEqual(gate["skipped_ranges"], [])
            self.assertEqual(len(gate["rollouts"]), 1)
            rollout = gate["rollouts"][0]
            self.assertEqual(rollout["seed"], 0)
            self.assertEqual(
                [receipt["frames"] for receipt in rollout["horizons"]],
                [60, 120, 240],
            )
            self.assertEqual(
                [receipt["seconds"] for receipt in rollout["horizons"]],
                [1.0, 2.0, 4.0],
            )

            frames = 240
            short_bundle = replace(
                bundle,
                positions=bundle.positions[:frames],
                rotations=bundle.rotations[:frames],
                velocities=bundle.velocities[:frames],
                angular_velocities=bundle.angular_velocities[:frames],
                range_starts=np.array([0], dtype=np.int32),
                range_stops=np.array([frames], dtype=np.int32),
                contacts=bundle.contacts[:frames],
                features=bundle.features[:frames],
                admitted_mask=np.ones(frames, dtype=bool),
            )
            short_autoencoder = mock.Mock(
                latent=autoencoder.latent[:frames]
            )
            short_gate = g1_lmm_training._stepper_rollout_gate(
                short_bundle,
                arrays,
                short_autoencoder,
                ZeroStepper(),
                np.zeros(state_dimensions, dtype=np.float32),
                np.ones(state_dimensions, dtype=np.float32),
                np.zeros(state_dimensions, dtype=np.float32),
                np.ones(state_dimensions, dtype=np.float32),
                config,
            )
            self.assertFalse(short_gate["accepted"])
            self.assertEqual(short_gate["rollouts"], [])
            self.assertEqual(
                short_gate["skipped_ranges"],
                [
                    {
                        "range_index": 0,
                        "reason": "range has 240 or fewer valid successors",
                    }
                ],
            )

    def test_accepted_flat_bundle_builds_exact_legacy_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            data_directory = Path(directory) / "data"
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)

            self.assertEqual(bundle.dimensions, G1LmmDimensions())
            self.assertEqual(bundle.frames, 256)
            self.assertEqual(int(bundle.admitted_mask.sum()), 256)
            self.assertFalse(bundle.admitted_mask.flags.writeable)
            self.assertEqual(arrays.compressor_input.shape, (256, 908))
            self.assertEqual(arrays.decompressor_target.shape, (256, 458))
            np.testing.assert_array_equal(arrays.compressor_input[:, :90], bundle.positions[:, 1:].reshape(256, -1))
            np.testing.assert_array_equal(arrays.decompressor_target[:, :90], bundle.positions[:, 1:].reshape(256, -1))
            np.testing.assert_array_equal(arrays.decompressor_target[:, -2:], bundle.contacts.astype(np.float32))
            self.assertTrue(np.isfinite(arrays.compressor_input).all())
            self.assertTrue(np.isfinite(arrays.decompressor_target).all())

            with (data_directory / "features.bin").open("r+b") as stream:
                stream.seek(12)
                original = stream.read(1)
                stream.seek(12)
                stream.write(bytes((original[0] ^ 1,)))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_training_bundle(data_directory)

    def test_flat_dimensions_and_exact_orange_duck_architectures(self):
        dims = G1LmmDimensions(features=31, latent=32, bones=31, contacts=2)
        self.assertEqual(dims.state, 63)
        self.assertEqual(dims.compressor_input, 908)
        self.assertEqual(dims.decompressor_output, 458)

        compressor = Compressor(dims)
        decompressor = Decompressor(dims)
        stepper = Stepper(dims)
        projector = Projector(dims)

        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in compressor.layers],
            [(908, 512), (512, 512), (512, 512), (512, 32)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in decompressor.layers],
            [(63, 512), (512, 458)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in stepper.layers],
            [(63, 512), (512, 512), (512, 63)],
        )
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in projector.layers],
            [(31, 512), (512, 512), (512, 512), (512, 512), (512, 63)],
        )

        with torch.no_grad():
            self.assertEqual(tuple(compressor(torch.zeros(2, 3, 908)).shape), (2, 3, 32))
            self.assertEqual(tuple(decompressor(torch.zeros(2, 3, 63)).shape), (2, 3, 458))
            self.assertEqual(tuple(stepper(torch.zeros(2, 3, 63)).shape), (2, 3, 63))
            self.assertEqual(tuple(projector(torch.zeros(2, 3, 31)).shape), (2, 3, 63))

    def test_windows_derivatives_and_withheld_halos_are_range_safe(self):
        starts = np.array([0, 100], dtype=np.int32)
        stops = np.array([80, 180], dtype=np.int32)
        windows = range_safe_windows(starts, stops, 20)
        self.assertEqual(windows.shape, (122, 20))
        self.assertFalse(any(window[0] < 80 <= window[-1] for window in windows))
        self.assertFalse(any(window[0] < 100 <= window[-1] for window in windows))

        fitted = range_safe_windows(
            starts,
            stops,
            20,
            excluded_starts=np.array([30], dtype=np.int32),
            excluded_stops=np.array([40], dtype=np.int32),
            exclusion_halo=2,
        )
        self.assertTrue(all(not np.any((window >= 28) & (window < 42)) for window in fitted))

        rows = np.arange(180, dtype=np.float32)[:, None]
        derivatives = range_safe_deltas(rows, starts, stops, dt=1.0 / 60.0)
        self.assertEqual(derivatives.shape, (158, 1))
        np.testing.assert_array_equal(derivatives, np.full((158, 1), 60.0, np.float32))

    def test_projector_targets_and_costs_stay_in_normalized_space(self):
        features_normalized = np.array(
            [[-1.0, 0.0], [0.25, 0.5], [2.0, -0.5]], dtype=np.float32
        )
        latent = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        query_normalized = np.array([[0.20, 0.45], [1.8, -0.6]], dtype=np.float32)

        result = normalized_projector_targets(query_normalized, features_normalized, latent)
        np.testing.assert_array_equal(result.indices, np.array([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(result.features, features_normalized[[1, 2]])
        np.testing.assert_array_equal(result.latent, latent[[1, 2]])
        expected = np.sqrt(np.sum((query_normalized - features_normalized[[1, 2]]) ** 2, axis=1))
        np.testing.assert_allclose(result.distance, expected, rtol=0.0, atol=0.0)

    def test_network_binary_is_little_endian_transposed_and_fail_closed(self):
        layer = torch.nn.Linear(2, 3)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32))
            layer.bias.copy_(torch.tensor([7, 8, 9], dtype=torch.float32))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.bin"
            export_network(
                path,
                [layer],
                input_mean=np.array([10, 11], dtype=np.float32),
                input_std=np.array([12, 13], dtype=np.float32),
                output_mean=np.array([14, 15, 16], dtype=np.float32),
                output_std=np.array([17, 18, 19], dtype=np.float32),
            )
            payload = path.read_bytes()

            cursor = 0
            vectors = []
            for expected_size in (2, 2, 3, 3):
                size = struct.unpack_from("<I", payload, cursor)[0]
                cursor += 4
                self.assertEqual(size, expected_size)
                vectors.append(np.frombuffer(payload, dtype="<f4", count=size, offset=cursor).copy())
                cursor += size * 4
            self.assertEqual(struct.unpack_from("<I", payload, cursor)[0], 1)
            cursor += 4
            self.assertEqual(struct.unpack_from("<II", payload, cursor), (2, 3))
            cursor += 8
            np.testing.assert_array_equal(
                np.frombuffer(payload, dtype="<f4", count=6, offset=cursor),
                np.array([1, 3, 5, 2, 4, 6], dtype=np.float32),
            )

            loaded = load_exported_network(path, expected_layers=[(2, 3)])
            np.testing.assert_array_equal(loaded.input_mean, vectors[0])
            np.testing.assert_array_equal(loaded.layers[0].weight, layer.weight.detach().numpy())
            np.testing.assert_array_equal(loaded.layers[0].bias, layer.bias.detach().numpy())

            path.write_bytes(payload + b"trailing")
            with self.assertRaisesRegex(ValueError, "trailing"):
                load_exported_network(path, expected_layers=[(2, 3)])

    def test_tiny_training_overfits_64_frames_and_is_bitwise_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = train_tiny_fixture(root / "first", seed=1234, device="cpu")
            second = train_tiny_fixture(root / "second", seed=1234, device="cpu")

            self.assertTrue(first["finite"])
            self.assertTrue(first["overfit_gate"]["accepted"])
            self.assertEqual(first["overfit_gate"]["frames"], 64)
            self.assertLess(first["overfit_gate"]["final_loss"], first["overfit_gate"]["initial_loss"])
            self.assertLess(
                first["overfit_gate"]["final_loss"],
                0.1 * first["overfit_gate"]["initial_loss"],
            )
            self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])
            self.assertEqual(evaluate_exported_networks(root / "first"), first["reload_metrics"])

            expected_files = {
                "latent.bin",
                "decompressor.bin",
                "stepper.bin",
                "projector.bin",
                "training.json",
                "evaluation.json",
                "manifest.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in (root / "first").iterdir()}))
            manifest = json.loads((root / "first" / "manifest.json").read_text(encoding="utf-8"))
            for name in ("latent.bin", "decompressor.bin", "stepper.bin", "projector.bin"):
                self.assertEqual(
                    manifest["artifacts"][name]["sha256"],
                    hashlib.sha256((root / "first" / name).read_bytes()).hexdigest(),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_isolated_cuda_synthetic_exports_are_bitwise_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = """
import json
import sys
from pathlib import Path
from resources.g1_lmm.training import train_tiny_fixture
result = train_tiny_fixture(Path(sys.argv[1]), seed=1234, device='cuda:0')
print(json.dumps(result, sort_keys=True))
"""
            results = []
            for name in ("first", "second"):
                environment = os.environ.copy()
                environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
                environment["CUDA_VISIBLE_DEVICES"] = "1"
                environment["PYTHONPATH"] = ".:resources"
                completed = subprocess.run(
                    [sys.executable, "-c", script, str(root / name)],
                    cwd=Path(__file__).resolve().parents[2],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
            self.assertEqual(results[0]["device"], "cuda:0")
            self.assertEqual(
                results[0]["determinism"]["cublas_workspace_config"], ":4096:8"
            )
            self.assertEqual(results[0]["artifact_sha256"], results[1]["artifact_sha256"])
            for artifact in results[0]["artifact_sha256"]:
                self.assertEqual(
                    (root / "first" / artifact).read_bytes(),
                    (root / "second" / artifact).read_bytes(),
                )

    def test_package_sets_cublas_workspace_before_first_torch_import(self):
        script = """
import builtins
import json
import os
import sys
original_import = builtins.__import__
observed = []
def probe(name, *args, **kwargs):
    if name == 'torch' and name not in sys.modules:
        observed.append(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    return original_import(name, *args, **kwargs)
builtins.__import__ = probe
import resources.g1_lmm
print(json.dumps({'observed': observed, 'final': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}))
"""
        environment = os.environ.copy()
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        environment["PYTHONPATH"] = ".:resources"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip())
        self.assertEqual(result["observed"], [":4096:8"])
        self.assertEqual(result["final"], ":4096:8")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_determinism_rejects_conflicting_cublas_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["CUBLAS_WORKSPACE_CONFIG"] = "invalid"
            environment["CUDA_VISIBLE_DEVICES"] = "1"
            environment["PYTHONPATH"] = ".:resources"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "from resources.g1_lmm.training import train_tiny_fixture; "
                        "train_tiny_fixture(Path(__import__('sys').argv[1]), device='cuda:0')"
                    ),
                    str(Path(directory) / "rejected"),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("CUBLAS_WORKSPACE_CONFIG", completed.stderr)

    def test_cpu_index_alias_reaches_eager_decompressor_kernel(self):
        dimensions = G1LmmDimensions()
        parents, target, ground_rows = _orange_duck_loss_fixture(frames=2)
        compressor = Compressor(dimensions)
        decompressor = Decompressor(dimensions)
        compressor_rows = torch.zeros(2, dimensions.compressor_input)
        feature_rows = torch.zeros(2, dimensions.features)
        kernel = g1_lmm_training._build_decompressor_training_kernel(
            compressor=compressor,
            decompressor=decompressor,
            compressor_rows=compressor_rows,
            feature_rows=feature_rows,
            output_mean=target.mean(dim=0),
            output_std=torch.full((dimensions.decompressor_output,), 0.03),
            ground_rows=ground_rows,
            parents=tuple(int(parent) for parent in parents),
            dt=1.0 / 60.0,
            device=torch.device("cpu:0"),
        )

        loss = kernel(torch.tensor([[0, 1]], dtype=torch.long))

        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(kernel.compiled)
        self.assertEqual(kernel.name, "eager/v1")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class G1LmmCompiledCudaTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0")
        g1_lmm_training._configure_determinism(8128, cls.device)
        cls.dimensions = G1LmmDimensions()
        cls.parents, target, ground = _orange_duck_loss_fixture(frames=64)
        cls.parents_tuple = tuple(int(parent) for parent in cls.parents)
        cls.ground_rows = {
            name: values.to(cls.device) for name, values in ground.items()
        }
        generator = torch.Generator(device=cls.device)
        generator.manual_seed(9917)
        cls.compressor_rows = torch.randn(
            64,
            cls.dimensions.compressor_input,
            generator=generator,
            device=cls.device,
        )
        cls.feature_rows = torch.randn(
            64,
            cls.dimensions.features,
            generator=generator,
            device=cls.device,
        )
        cls.output_mean = target.mean(dim=0).to(cls.device)
        cls.output_std = torch.full(
            (cls.dimensions.decompressor_output,),
            0.03,
            device=cls.device,
        )
        torch.manual_seed(4411)
        torch.cuda.manual_seed_all(4411)
        cls.compressor = Compressor(cls.dimensions).to(cls.device)
        cls.decompressor = Decompressor(cls.dimensions).to(cls.device)
        cls.initial_state = {
            f"compressor.{name}": value.detach().clone()
            for name, value in cls.compressor.state_dict().items()
        }
        cls.initial_state.update(
            {
                f"decompressor.{name}": value.detach().clone()
                for name, value in cls.decompressor.state_dict().items()
            }
        )
        cls.kernel = g1_lmm_training._build_decompressor_training_kernel(
            compressor=cls.compressor,
            decompressor=cls.decompressor,
            compressor_rows=cls.compressor_rows,
            feature_rows=cls.feature_rows,
            output_mean=cls.output_mean,
            output_std=cls.output_std,
            ground_rows=cls.ground_rows,
            parents=cls.parents_tuple,
            dt=1.0 / 60.0,
            device=cls.device,
        )
        starts = torch.arange(400, device=cls.device).reshape(100, 4) % 63
        cls.schedule = torch.stack((starts, starts + 1), dim=-1)

    def setUp(self):
        self._restore_initial_state()

    def _restore_initial_state(self):
        compressor_state = {
            name.removeprefix("compressor."): value
            for name, value in self.initial_state.items()
            if name.startswith("compressor.")
        }
        decompressor_state = {
            name.removeprefix("decompressor."): value
            for name, value in self.initial_state.items()
            if name.startswith("decompressor.")
        }
        self.compressor.load_state_dict(compressor_state)
        self.decompressor.load_state_dict(decompressor_state)
        self.output_std.fill_(0.03)
        self.compressor.zero_grad(set_to_none=True)
        self.decompressor.zero_grad(set_to_none=True)

    def _eager_loss(self, batch):
        encoded = self.compressor(self.compressor_rows[batch])
        decoded = (
            self.decompressor(
                torch.cat((self.feature_rows[batch], encoded), dim=-1)
            )
            * self.output_std
            + self.output_mean
        )
        return orange_duck_decompressor_losses(
            decoded,
            encoded,
            parents=self.parents,
            dt=1.0 / 60.0,
            **{name: values[batch] for name, values in self.ground_rows.items()},
        )["total"]

    def _parameter_digest_after_100_steps(self):
        self._restore_initial_state()
        optimizer = torch.optim.AdamW(
            [*self.compressor.parameters(), *self.decompressor.parameters()],
            lr=1.0e-3,
            amsgrad=True,
            weight_decay=1.0e-3,
        )
        for batch in self.schedule:
            g1_lmm_training._decompressor_training_step(
                self.kernel, batch, optimizer
            )
        digest = hashlib.sha256()
        for module_name, module in (
            ("compressor", self.compressor),
            ("decompressor", self.decompressor),
        ):
            for name, value in module.state_dict().items():
                digest.update(f"{module_name}.{name}".encode("utf-8"))
                digest.update(
                    value.detach().cpu().contiguous().numpy().tobytes(order="C")
                )
        return digest.hexdigest()

    def test_compiled_kernel_matches_eager_loss_and_parameter_gradients(self):
        batch = self.schedule[0]
        eager_loss = self._eager_loss(batch)
        parameters = tuple(
            [*self.compressor.parameters(), *self.decompressor.parameters()]
        )
        eager_gradients = torch.autograd.grad(eager_loss, parameters)

        self.kernel.begin_step()
        compiled_loss = self.kernel(batch)
        compiled_gradients = torch.autograd.grad(compiled_loss, parameters)

        torch.testing.assert_close(compiled_loss, eager_loss, rtol=1.0e-6, atol=1.0e-6)
        for compiled, eager in zip(compiled_gradients, eager_gradients):
            torch.testing.assert_close(compiled, eager, rtol=2.0e-4, atol=5.0e-6)
        self.assertTrue(self.kernel.compiled)
        self.assertEqual(
            self.kernel.name,
            "torch-compile-reduce-overhead/fullgraph/v1",
        )

    def test_unindexed_cuda_alias_reaches_compiled_decompressor_kernel(self):
        alias_kernel = g1_lmm_training._build_decompressor_training_kernel(
            compressor=self.compressor,
            decompressor=self.decompressor,
            compressor_rows=self.compressor_rows,
            feature_rows=self.feature_rows,
            output_mean=self.output_mean,
            output_std=self.output_std,
            ground_rows=self.ground_rows,
            parents=self.parents_tuple,
            dt=1.0 / 60.0,
            device=torch.device("cuda"),
        )

        alias_kernel.begin_step()
        loss = alias_kernel(self.schedule[0])

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(alias_kernel.compiled)

    def test_compiled_kernel_rejects_actual_cross_device_tensor(self):
        mismatched_ground = dict(self.ground_rows)
        mismatched_ground["contacts"] = mismatched_ground["contacts"].cpu()

        with self.assertRaisesRegex(ValueError, "one actual device"):
            g1_lmm_training._build_decompressor_training_kernel(
                compressor=self.compressor,
                decompressor=self.decompressor,
                compressor_rows=self.compressor_rows,
                feature_rows=self.feature_rows,
                output_mean=self.output_mean,
                output_std=self.output_std,
                ground_rows=mismatched_ground,
                parents=self.parents_tuple,
                dt=1.0 / 60.0,
                device=torch.device("cuda"),
            )

    def test_compiled_kernel_has_repeatable_100_step_parameter_sha(self):
        first = self._parameter_digest_after_100_steps()
        second = self._parameter_digest_after_100_steps()
        self.assertEqual(first, second)

    def test_compiled_kernel_rejects_nonfinite_loss_before_optimizer_update(self):
        optimizer = torch.optim.AdamW(
            [*self.compressor.parameters(), *self.decompressor.parameters()],
            lr=1.0e-3,
            amsgrad=True,
            weight_decay=1.0e-3,
        )
        before = [
            parameter.detach().clone() for parameter in self.compressor.parameters()
        ]
        self.output_std.fill_(float("nan"))

        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            g1_lmm_training._decompressor_training_step(
                self.kernel, self.schedule[0], optimizer
            )

        self.assertEqual(len(optimizer.state), 0)
        for expected, actual in zip(before, self.compressor.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_real_decompressor_loop_uses_kernel_and_keeps_eager_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_directory = root / "data"
            staging = root / "staging"
            staging.mkdir()
            _write_flat_training_bundle(data_directory)
            bundle = load_training_bundle(data_directory)
            arrays = build_training_arrays(bundle)
            observed = {}

            class FakeKernel:
                compiled = True
                name = "test-compiled-kernel"

                def __init__(self, parameters):
                    self.parameters = tuple(parameters)
                    self.begin_count = 0
                    self.call_count = 0

                def begin_step(self):
                    self.begin_count += 1

                def __call__(self, batch):
                    self.call_count += 1
                    return sum(
                        parameter.square().mean() for parameter in self.parameters
                    )

            def fake_builder(**kwargs):
                observed["parents"] = kwargs["parents"]
                observed["device"] = kwargs["device"]
                observed["kernel"] = FakeKernel(
                    [
                        *kwargs["compressor"].parameters(),
                        *kwargs["decompressor"].parameters(),
                    ]
                )
                return observed["kernel"]

            original_loss = g1_lmm_training.orange_duck_decompressor_losses
            with (
                mock.patch.object(
                    g1_lmm_training,
                    "_build_decompressor_training_kernel",
                    side_effect=fake_builder,
                ) as builder,
                mock.patch.object(
                    g1_lmm_training,
                    "orange_duck_decompressor_losses",
                    wraps=original_loss,
                ) as eager_loss,
                mock.patch.object(
                    g1_lmm_training,
                    "_decode_reconstruction_metrics",
                    return_value={"accepted": False},
                ),
            ):
                stage = g1_lmm_training._train_decompressor_stage(
                    staging,
                    bundle,
                    arrays,
                    TrainingConfig(
                        device="cuda:0",
                        batch_size=4,
                        decompressor_steps=1,
                        overfit_steps=1,
                        stepper_steps=1,
                        projector_steps=1,
                        withheld_frames=64,
                        withheld_halo=2,
                    ),
                    np.array([100], dtype=np.int32),
                    np.array([164], dtype=np.int32),
                )

            builder.assert_called_once()
            self.assertIsInstance(observed["parents"], tuple)
            self.assertEqual(observed["parents"], tuple(int(x) for x in bundle.parents))
            self.assertEqual(observed["device"], torch.device("cuda:0"))
            self.assertEqual(observed["kernel"].begin_count, 1)
            self.assertEqual(observed["kernel"].call_count, 1)
            self.assertEqual(eager_loss.call_count, 2)
            self.assertEqual(
                stage.training_metrics["training_kernel"], "test-compiled-kernel"
            )


if __name__ == "__main__":
    unittest.main()
