"""Test-first proofs for the shared Torch feature extractor and database.

These tests pin the frozen 27-value search feature contract, group
normalization, and the immutable provenance-safe motion database. They compare
the production Torch extractor against an independent NumPy oracle defined in
``torch_motion_test_utils`` and never call the production feature math to build
their own expectations.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import mm_sonic.torch_motion_features as features_module
from mm_sonic.torch_motion_data import MotionFolder
from mm_sonic.torch_motion_features import (
    FEATURE_GROUPS,
    FEATURE_HORIZON_FRAMES,
    CommandTrajectory,
    GeneratedFeatureState,
    SearchFeatureExtension,
    TorchMotionDatabase,
    extract_query_features,
)

from tests.python.torch_motion_test_utils import (
    ORACLE_LEFT_FOOT_BODY,
    ORACLE_RIGHT_FOOT_BODY,
    ORACLE_ROOT_BODY,
    build_takara_arrays,
    build_varying_takara_arrays,
    oracle_clip_rows,
    oracle_normalization,
    oracle_yaw_from_wxyz,
    write_takara_arrays,
)

_CPU = torch.device("cpu")


class _TwoValueExtension:
    name = "test_extension"
    dimension = 2
    weight = 2.0

    def __init__(self, *, dtype=torch.float32, wrong_rows=0, constant=False):
        self.dtype = dtype
        self.wrong_rows = wrong_rows
        self.constant = constant
        self.query_calls = 0
        self.database_calls = 0

    def database_rows(self, folder, device):
        self.database_calls += 1
        rows = []
        for clip in folder.clips:
            count = clip.valid_frame_stop + self.wrong_rows
            frame = torch.arange(count, dtype=self.dtype, device=device)
            if self.constant:
                frame = torch.zeros_like(frame)
            rows.append(torch.stack((frame, 0.25 * frame * frame + frame), dim=1))
        return tuple(rows)

    def query_row(self, state, trajectory):
        self.query_calls += 1
        x = state.root_position_world[0]
        return torch.stack((x, x * x + x)).to(dtype=self.dtype)


def _generated_state_from_arrays(arrays, frame, device=_CPU):
    body_pos = torch.tensor(arrays["body_pos_w"], dtype=torch.float32, device=device)
    body_quat = torch.tensor(arrays["body_quat_w"], dtype=torch.float32, device=device)
    body_lin = torch.tensor(arrays["body_lin_vel_w"], dtype=torch.float32, device=device)
    return GeneratedFeatureState(
        root_position_world=body_pos[frame, ORACLE_ROOT_BODY],
        root_orientation_world_wxyz=body_quat[frame, ORACLE_ROOT_BODY],
        root_linear_velocity_world=body_lin[frame, ORACLE_ROOT_BODY],
        left_foot_position_world=body_pos[frame, ORACLE_LEFT_FOOT_BODY],
        right_foot_position_world=body_pos[frame, ORACLE_RIGHT_FOOT_BODY],
        left_foot_velocity_world=body_lin[frame, ORACLE_LEFT_FOOT_BODY],
        right_foot_velocity_world=body_lin[frame, ORACLE_RIGHT_FOOT_BODY],
    )


def _command_trajectory_from_arrays(arrays, frame, device=_CPU):
    body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    pos_xy = np.stack(
        [body_pos[frame + off, ORACLE_ROOT_BODY, :2] for off in FEATURE_HORIZON_FRAMES]
    )
    face_xy = np.stack(
        [
            np.array(
                [
                    np.cos(oracle_yaw_from_wxyz(body_quat[frame + off, ORACLE_ROOT_BODY])),
                    np.sin(oracle_yaw_from_wxyz(body_quat[frame + off, ORACLE_ROOT_BODY])),
                ]
            )
            for off in FEATURE_HORIZON_FRAMES
        ]
    )
    return CommandTrajectory(
        position_world_xy=torch.tensor(pos_xy, dtype=torch.float32, device=device),
        facing_world_xy=torch.tensor(face_xy, dtype=torch.float32, device=device),
    )


class FeatureVectorContractTests(unittest.TestCase):
    def test_feature_vector_is_exactly_27_values_in_frozen_group_order(self):
        self.assertEqual(FEATURE_HORIZON_FRAMES, (15, 30, 45))
        names = [g[0] for g in FEATURE_GROUPS]
        self.assertEqual(
            names,
            [
                "left_foot_position",
                "right_foot_position",
                "left_foot_velocity",
                "right_foot_velocity",
                "pelvis_velocity",
                "trajectory_position",
                "trajectory_facing",
            ],
        )
        weights = [g[2] for g in FEATURE_GROUPS]
        self.assertEqual(weights, [0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.5])
        covered = sum(g[1].stop - g[1].start for g in FEATURE_GROUPS)
        self.assertEqual(covered, 27)

        arrays = build_takara_arrays(frames=60, yaw_rate=0.3)
        state = _generated_state_from_arrays(arrays, 5)
        traj = _command_trajectory_from_arrays(arrays, 5)
        feat = extract_query_features(state, traj)
        self.assertEqual(tuple(feat.shape), (27,))
        expected = oracle_clip_rows(arrays)[5]
        np.testing.assert_allclose(
            feat.cpu().numpy().astype(np.float64), expected, rtol=0.0, atol=1e-4
        )

    def test_clip_row_and_generated_query_use_identical_extractor(self):
        arrays = build_varying_takara_arrays(frames=70)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(folder, device="cpu")

        mean, scale = oracle_normalization(oracle_clip_rows(arrays))
        for frame in (0, 3, 10):
            state = _generated_state_from_arrays(arrays, frame)
            traj = _command_trajectory_from_arrays(arrays, frame)
            raw = extract_query_features(state, traj)
            normalized = db.normalization.normalize(raw)
            db_row = db.normalized_features_copy()[frame]
            np.testing.assert_allclose(
                normalized.cpu().numpy().astype(np.float64),
                db_row.cpu().numpy().astype(np.float64),
                rtol=0.0,
                atol=1e-4,
            )

    def test_query_and_database_both_route_through_single_feature_seam(self):
        # Identity, not just numeric parity: both the runtime query path and the
        # database-row path must call the same private ``_feature_row`` seam so
        # the feature math is never duplicated. A call-counting spy proves the
        # query path invokes the seam exactly once, and the database path invokes
        # it once for the query and once per clip as a batched extraction. This
        # also prevents an accidental full-clip host-to-device copy per frame.
        arrays = build_varying_takara_arrays(frames=60)
        real_feature_row = features_module._feature_row

        state = _generated_state_from_arrays(arrays, 4)
        traj = _command_trajectory_from_arrays(arrays, 4)
        with mock.patch.object(
            features_module, "_feature_row", side_effect=real_feature_row
        ) as spy:
            extract_query_features(state, traj)
        self.assertEqual(spy.call_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            folder = MotionFolder.load(root)
            valid = 60 - FEATURE_HORIZON_FRAMES[-1]
            with mock.patch.object(
                features_module, "_feature_row", side_effect=real_feature_row
            ) as spy:
                db = TorchMotionDatabase.from_folder(folder, device="cpu")
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(db.feature_shape[0], valid)

    def test_global_xy_translation_and_common_z_yaw_do_not_change_features(self):
        arrays = build_takara_arrays(frames=60, yaw_rate=0.25)
        base_state = _generated_state_from_arrays(arrays, 8)
        base_traj = _command_trajectory_from_arrays(arrays, 8)
        base_feat = extract_query_features(base_state, base_traj).cpu().numpy()

        # Apply a common planar translation to every world position.
        shift = torch.tensor([3.0, -2.0, 0.0], dtype=torch.float32)
        shift_xy = shift[:2]
        shifted_state = GeneratedFeatureState(
            root_position_world=base_state.root_position_world + shift,
            root_orientation_world_wxyz=base_state.root_orientation_world_wxyz,
            root_linear_velocity_world=base_state.root_linear_velocity_world,
            left_foot_position_world=base_state.left_foot_position_world + shift,
            right_foot_position_world=base_state.right_foot_position_world + shift,
            left_foot_velocity_world=base_state.left_foot_velocity_world,
            right_foot_velocity_world=base_state.right_foot_velocity_world,
        )
        shifted_traj = CommandTrajectory(
            position_world_xy=base_traj.position_world_xy + shift_xy,
            facing_world_xy=base_traj.facing_world_xy,
        )
        shifted_feat = extract_query_features(shifted_state, shifted_traj).cpu().numpy()
        np.testing.assert_allclose(shifted_feat, base_feat, rtol=0.0, atol=1e-5)

    def test_rejects_wrong_shape_dtype_device_and_nonfinite(self):
        arrays = build_takara_arrays(frames=60)
        state = _generated_state_from_arrays(arrays, 2)
        traj = _command_trajectory_from_arrays(arrays, 2)

        # Wrong root position shape.
        with self.assertRaises((ValueError, RuntimeError)):
            bad = GeneratedFeatureState(
                root_position_world=torch.zeros(4),
                root_orientation_world_wxyz=state.root_orientation_world_wxyz,
                root_linear_velocity_world=state.root_linear_velocity_world,
                left_foot_position_world=state.left_foot_position_world,
                right_foot_position_world=state.right_foot_position_world,
                left_foot_velocity_world=state.left_foot_velocity_world,
                right_foot_velocity_world=state.right_foot_velocity_world,
            )
            extract_query_features(bad, traj)

        # Non-finite value.
        with self.assertRaises((ValueError, RuntimeError)):
            nan_state = GeneratedFeatureState(
                root_position_world=torch.tensor([float("nan"), 0.0, 0.8]),
                root_orientation_world_wxyz=state.root_orientation_world_wxyz,
                root_linear_velocity_world=state.root_linear_velocity_world,
                left_foot_position_world=state.left_foot_position_world,
                right_foot_position_world=state.right_foot_position_world,
                left_foot_velocity_world=state.left_foot_velocity_world,
                right_foot_velocity_world=state.right_foot_velocity_world,
            )
            extract_query_features(nan_state, traj)

        # dtype disagreement.
        with self.assertRaises((ValueError, RuntimeError)):
            wrong_dtype = GeneratedFeatureState(
                root_position_world=state.root_position_world.double(),
                root_orientation_world_wxyz=state.root_orientation_world_wxyz,
                root_linear_velocity_world=state.root_linear_velocity_world,
                left_foot_position_world=state.left_foot_position_world,
                right_foot_position_world=state.right_foot_position_world,
                left_foot_velocity_world=state.left_foot_velocity_world,
                right_foot_velocity_world=state.right_foot_velocity_world,
            )
            extract_query_features(wrong_dtype, traj)


class NormalizationAndDatabaseTests(unittest.TestCase):
    def test_optional_extension_appends_one_normalized_group(self):
        self.assertTrue(issubclass(SearchFeatureExtension, object))
        arrays = build_varying_takara_arrays(frames=80)
        extension = _TwoValueExtension()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(
                folder, device="cpu", extension=extension
            )

        self.assertEqual(db.feature_shape, (35, 29))
        self.assertEqual(extension.database_calls, 1)
        mean, scale = db.normalization.parameters_copy()
        self.assertEqual(tuple(mean.shape), (29,))
        self.assertEqual(tuple(scale.shape), (29,))
        self.assertTrue(torch.isfinite(db.normalized_features_copy()).all())
        self.assertEqual(db.motion_feature_dim, 27)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_alias_is_canonicalized_before_extension_validation(self):
        arrays = build_varying_takara_arrays(frames=80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(
                folder,
                device="cuda",
                extension=_TwoValueExtension(),
            )
        self.assertEqual(db.device, torch.device("cuda:0"))
        self.assertEqual(db.normalized_features_copy().device, db.device)

    def test_extension_rejects_wrong_rows_dtype_nonfinite_and_zero_variance(self):
        arrays = build_varying_takara_arrays(frames=80)

        def build(extension):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_takara_arrays(root / "walk", arrays)
                folder = MotionFolder.load(root)
                return TorchMotionDatabase.from_folder(
                    folder, device="cpu", extension=extension
                )

        with self.assertRaisesRegex(Exception, "rows"):
            build(_TwoValueExtension(wrong_rows=1))
        with self.assertRaisesRegex(Exception, "float32"):
            build(_TwoValueExtension(dtype=torch.float64))

        nonfinite = _TwoValueExtension()
        original = nonfinite.database_rows

        def nan_rows(folder, device):
            rows = list(original(folder, device))
            rows[0][0, 0] = float("nan")
            return tuple(rows)

        nonfinite.database_rows = nan_rows
        with self.assertRaisesRegex(Exception, "finite"):
            build(nonfinite)
        with self.assertRaisesRegex(Exception, "scale"):
            build(_TwoValueExtension(constant=True))

    def test_group_means_scales_and_normalized_rows_match_numpy_oracle(self):
        arrays = build_varying_takara_arrays(frames=80)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(folder, device="cpu")

        rows = oracle_clip_rows(arrays)
        mean, scale = oracle_normalization(rows)
        got_mean, got_scale = db.normalization.parameters_copy()
        np.testing.assert_allclose(
            got_mean.cpu().numpy().astype(np.float64), mean, rtol=0.0, atol=1e-4
        )
        np.testing.assert_allclose(
            got_scale.cpu().numpy().astype(np.float64), scale, rtol=0.0, atol=1e-4
        )
        expected_norm = (rows - mean) / scale
        np.testing.assert_allclose(
            db.normalized_features_copy().cpu().numpy().astype(np.float64),
            expected_norm,
            rtol=0.0,
            atol=1e-3,
        )
        self.assertEqual(db.feature_shape, (rows.shape[0], 27))

    def test_zero_variation_group_fails_database_build(self):
        # A perfectly static clip yields zero-variance groups, so the group
        # standard deviation is non-positive and construction must fail.
        arrays = build_takara_arrays(frames=60, yaw_rate=0.0, root_velocity_xy=(0.0, 0.0))
        # Freeze all bodies to remove every source of feature variation.
        arrays["body_pos_w"][:] = arrays["body_pos_w"][0:1]
        arrays["body_quat_w"][:] = arrays["body_quat_w"][0:1]
        arrays["body_lin_vel_w"][:] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "static", arrays)
            folder = MotionFolder.load(root)
            with self.assertRaises(Exception):
                TorchMotionDatabase.from_folder(folder, device="cpu")

    def test_database_provenance_and_successor_lookup_do_not_cross_clips(self):
        arrays_a = build_varying_takara_arrays(frames=60)
        arrays_b = build_varying_takara_arrays(frames=70, yaw_amplitude=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "a", arrays_a)
            write_takara_arrays(root / "b", arrays_b)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(folder, device="cpu")

        valid_a = 60 - FEATURE_HORIZON_FRAMES[-1]
        valid_b = 70 - FEATURE_HORIZON_FRAMES[-1]
        self.assertEqual(db.feature_shape[0], valid_a + valid_b)

        # First clip rows map to clip 0 in order.
        self.assertEqual(db.row_for_source(0, 0), 0)
        self.assertEqual(db.row_for_source(0, valid_a - 1), valid_a - 1)
        # Frame just past the first clip's valid range has no row.
        self.assertIsNone(db.row_for_source(0, valid_a))
        # Second clip rows begin exactly after the first clip's rows.
        self.assertEqual(db.row_for_source(1, 0), valid_a)
        self.assertEqual(db.row_for_source(1, valid_b - 1), valid_a + valid_b - 1)
        # A non-existent clip index has no row.
        self.assertIsNone(db.row_for_source(2, 0))

    def test_reset_row_uses_min_joint_velocity_with_smallest_row_tiebreak(self):
        # Two clips whose smallest joint-velocity frame ties; the smallest
        # global database row must win.
        arrays_a = build_varying_takara_arrays(frames=60)
        arrays_b = build_varying_takara_arrays(frames=60)
        # Make one interior frame in each clip have zero joint velocity.
        arrays_a["joint_vel"][:] = 1.0
        arrays_a["joint_vel"][2] = 0.0
        arrays_b["joint_vel"][:] = 1.0
        arrays_b["joint_vel"][4] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "a", arrays_a)
            write_takara_arrays(root / "b", arrays_b)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(folder, device="cpu")
        # Clip a frame 2 is the earliest global row with minimal joint velocity.
        self.assertEqual(db.reset_row, db.row_for_source(0, 2))

    def test_reset_clip_path_limits_reset_selection_to_exact_clip(self):
        arrays_flat = build_varying_takara_arrays(frames=70)
        arrays_stair = build_varying_takara_arrays(frames=70)
        arrays_flat["joint_vel"][:] = 2.0
        arrays_flat["joint_vel"][7] = 0.2
        arrays_stair["joint_vel"][:] = 1.0
        arrays_stair["joint_vel"][3] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "flat", arrays_flat)
            write_takara_arrays(root / "stair", arrays_stair)
            folder = MotionFolder.load(root)
            global_db = TorchMotionDatabase.from_folder(folder, device="cpu")
            flat_db = TorchMotionDatabase.from_folder(
                folder,
                device="cpu",
                reset_clip_path="flat/motion.npz",
            )
            with self.assertRaisesRegex(Exception, "reset clip"):
                TorchMotionDatabase.from_folder(
                    folder,
                    device="cpu",
                    reset_clip_path="missing/motion.npz",
                )

        self.assertEqual(global_db.reset_row, global_db.row_for_source(1, 3))
        self.assertEqual(flat_db.reset_row, flat_db.row_for_source(0, 7))

    def test_cpu_and_cuda_database_tensors_are_immutable_by_api(self):
        arrays = build_varying_takara_arrays(frames=60)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_takara_arrays(root / "walk", arrays)
            folder = MotionFolder.load(root)
            db = TorchMotionDatabase.from_folder(folder, device="cpu")

        # Diagnostic accessors return clones: mutating them leaves the DB intact.
        copy_a = db.normalized_features_copy()
        copy_a += 999.0
        copy_b = db.normalized_features_copy()
        self.assertFalse(torch.allclose(copy_a, copy_b))

        mean_a, scale_a = db.normalization.parameters_copy()
        mean_a += 5.0
        mean_b, _ = db.normalization.parameters_copy()
        self.assertFalse(torch.allclose(mean_a, mean_b))

        # No public mutator is exposed on the database.
        self.assertFalse(hasattr(db, "set_features"))


if __name__ == "__main__":
    unittest.main()
