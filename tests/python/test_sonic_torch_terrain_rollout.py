import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mm_sonic.torch_motion_matcher import TorchMotionMatcher
from mm_sonic.torch_terrain_features import (
    TerrainFootClearanceValidator,
)
from mm_sonic.torch_terrain_rollout import (
    evaluate_dense_acceptance,
    load_experiment_config,
    matcher_config_from_resolved,
    resolve_stair_config,
    run_stair_rollout,
    save_stair_rollout,
    terrain_transition_validator_from_resolved,
)
from resources.g1_torch_stair_builder.publish import publish_stair_slice
from resources.g1_torch_stair_builder.surface import ZUpHeightGrid
from tests.python.test_torch_stair_conversion import (
    _FakeKinematics,
    write_synthetic_pinned_corpus,
)
from tests.python.torch_motion_test_utils import write_takara_clip


CONFIG_PATH = (
    Path(__file__).parents[2]
    / "sonic"
    / "configs"
    / "experiments"
    / "torch_stair_small.json"
)


def _wide_curved_grid(_source) -> ZUpHeightGrid:
    cell = 0.10
    axis = np.arange(-30.0, 30.0 + cell / 2.0, cell, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    height = 0.10 * x + 0.02 * y + 0.01 * x * x
    return ZUpHeightGrid(
        origin_xy=np.array([-30.0, -30.0], np.float32),
        cell_size_m=cell,
        height_z=height.astype(np.float32),
    )


class TerrainRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        grail = cls.root / "grail"
        write_synthetic_pinned_corpus(grail)
        flat_motion = write_takara_clip(cls.root / "flat", frames=80)
        g1_xml = cls.root / "g1.xml"
        g1_xml.write_text("<mujoco/>", encoding="utf-8")
        cls.dataset_root = cls.root / "dataset"
        publish_stair_slice(
            output=cls.dataset_root,
            grail_root=grail,
            g1_xml=g1_xml,
            flat_motion=flat_motion,
            kinematics=_FakeKinematics(),
            grid_builder=_wide_curved_grid,
        )
        raw = load_experiment_config(CONFIG_PATH)
        raw["duration_s"] = 0.20
        cls.resolved = resolve_stair_config(
            cls.dataset_root, raw, device="cpu"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_matcher_config_is_exact_validated_and_reconstructed(self):
        raw = load_experiment_config(CONFIG_PATH)
        self.assertEqual(
            set(raw["matcher"]),
            {
                "acceleration_mps2",
                "deceleration_mps2",
                "exclusion_frames",
                "inertialization_halflife_s",
                "reversal_speed_mps",
                "search_interval_steps",
                "stop_speed_mps",
                "transition_joint_position_weight",
                "transition_joint_velocity_weight",
                "transition_window_candidate_count",
                "transition_window_jerk_weight",
                "transition_penalty",
                "transition_settle_duration_s",
                "transition_settle_penalty",
                "yaw_rate_rad_s",
            },
        )
        matcher = matcher_config_from_resolved(
            self.resolved.resolved_config
        )
        self.assertEqual(matcher.search_interval_steps, 1)
        self.assertEqual(matcher.dt, 0.02)
        self.assertAlmostEqual(
            matcher.inertialization_halflife_s, 0.1
        )
        self.assertAlmostEqual(
            matcher.transition_settle_duration_s, 0.20
        )
        self.assertGreater(matcher.transition_settle_penalty, 0.0)
        self.assertEqual(matcher.transition_joint_position_weight, 0.1)
        self.assertEqual(matcher.transition_joint_velocity_weight, 0.1)
        self.assertEqual(matcher.transition_window_candidate_count, 32)
        self.assertEqual(matcher.transition_window_jerk_weight, 0.0)
        validator = terrain_transition_validator_from_resolved(
            self.resolved
        )
        self.assertIsInstance(
            validator, TerrainFootClearanceValidator
        )
        self.assertEqual(validator.preview_steps, 10)
        self.assertEqual(
            validator.minimum_clearance_m,
            self.resolved.resolved_config["acceptance"][
                "minimum_foot_clearance_m"
            ],
        )

        cases = {}
        missing = json.loads(json.dumps(raw))
        del missing["matcher"]
        cases["missing"] = missing
        extra = json.loads(json.dumps(raw))
        extra["matcher"]["unknown"] = 1
        cases["keys"] = extra
        interval = json.loads(json.dumps(raw))
        interval["matcher"]["search_interval_steps"] = 0
        cases["search_interval_steps"] = interval
        boolean_interval = json.loads(json.dumps(raw))
        boolean_interval["matcher"]["search_interval_steps"] = True
        cases["search_interval_steps boolean"] = boolean_interval
        exclusion = json.loads(json.dumps(raw))
        exclusion["matcher"]["exclusion_frames"] = -1
        cases["exclusion_frames"] = exclusion
        halflife = json.loads(json.dumps(raw))
        halflife["matcher"]["inertialization_halflife_s"] = 0.0
        cases["inertialization_halflife_s"] = halflife
        nonfinite = json.loads(json.dumps(raw))
        nonfinite["matcher"]["yaw_rate_rad_s"] = float("nan")
        cases["yaw_rate_rad_s"] = nonfinite
        negative_settle_duration = json.loads(json.dumps(raw))
        negative_settle_duration["matcher"][
            "transition_settle_duration_s"
        ] = -0.01
        cases["transition_settle_duration_s"] = negative_settle_duration
        nonfinite_settle_penalty = json.loads(json.dumps(raw))
        nonfinite_settle_penalty["matcher"][
            "transition_settle_penalty"
        ] = float("nan")
        cases["transition_settle_penalty"] = nonfinite_settle_penalty
        for label, value in (
            ("negative", -0.01),
            ("NaN", float("nan")),
            ("infinity", float("inf")),
            ("boolean", True),
        ):
            invalid_jerk_weight = json.loads(json.dumps(raw))
            invalid_jerk_weight["matcher"][
                "transition_window_jerk_weight"
            ] = value
            cases[
                f"transition_window_jerk_weight {label}"
            ] = invalid_jerk_weight
        for value in (0, -1, 1.5, True):
            invalid_candidate_count = json.loads(json.dumps(raw))
            invalid_candidate_count["matcher"][
                "transition_window_candidate_count"
            ] = value
            cases[
                f"transition_window_candidate_count {value!r}"
            ] = invalid_candidate_count
        for name in (
            "transition_joint_position_weight",
            "transition_joint_velocity_weight",
        ):
            for label, value in (
                ("negative", -0.01),
                ("NaN", float("nan")),
                ("infinity", float("inf")),
                ("boolean", True),
            ):
                invalid_weight = json.loads(json.dumps(raw))
                invalid_weight["matcher"][name] = value
                cases[f"{name} {label}"] = invalid_weight
            missing_weight = json.loads(json.dumps(raw))
            del missing_weight["matcher"][name]
            cases[f"{name} missing"] = missing_weight
        for value in (0, 47, True):
            preview = json.loads(json.dumps(raw))
            preview["terrain_transition_preview_steps"] = value
            cases[f"terrain preview {value!r}"] = preview
        for label, invalid in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(Exception):
                    resolve_stair_config(
                        self.dataset_root, invalid, device="cpu"
                    )

        disabled = json.loads(json.dumps(raw))
        disabled["matcher"]["transition_settle_duration_s"] = 0.0
        disabled["matcher"]["transition_settle_penalty"] = 0.0
        disabled["matcher"]["transition_joint_position_weight"] = 1.25
        disabled["matcher"]["transition_joint_velocity_weight"] = 2.5
        disabled_resolved = resolve_stair_config(
            self.dataset_root, disabled, device="cpu"
        )
        disabled_matcher = matcher_config_from_resolved(
            disabled_resolved.resolved_config
        )
        self.assertEqual(disabled_matcher.transition_settle_duration_s, 0.0)
        self.assertEqual(disabled_matcher.transition_settle_penalty, 0.0)
        self.assertEqual(
            disabled_matcher.transition_joint_position_weight, 1.25
        )
        self.assertEqual(
            disabled_matcher.transition_joint_velocity_weight, 2.5
        )

    def test_rollout_passes_pinned_matcher_config_to_matcher(self):
        expected = matcher_config_from_resolved(
            self.resolved.resolved_config
        )
        real_build = TorchMotionMatcher.from_folder
        calls = []

        def recording_build(*args, **kwargs):
            calls.append(dict(kwargs))
            kwargs["emitted_window_validator"] = None
            return real_build(*args, **kwargs)

        with mock.patch.object(
            TorchMotionMatcher,
            "from_folder",
            side_effect=recording_build,
        ):
            run_stair_rollout(self.resolved, "flat", device="cpu")
            run_stair_rollout(self.resolved, "dense", device="cpu")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["config"], expected)
        self.assertEqual(calls[1]["config"], expected)
        self.assertIsNone(calls[0]["emitted_window_validator"])
        self.assertIsInstance(
            calls[1]["emitted_window_validator"],
            TerrainFootClearanceValidator,
        )

    def test_deterministic_hash_and_commands_are_condition_independent(self):
        first = run_stair_rollout(self.resolved, "flat", device="cpu")
        second = run_stair_rollout(self.resolved, "flat", device="cpu")
        legacy = run_stair_rollout(self.resolved, "legacy", device="cpu")
        dense = run_stair_rollout(self.resolved, "dense", device="cpu")

        self.assertEqual(first.deterministic_sha256, second.deterministic_sha256)
        self.assertEqual(len(first.deterministic_sha256), 64)
        for candidate in (second, legacy, dense):
            np.testing.assert_array_equal(
                first.arrays["command_velocity_world_xy"],
                candidate.arrays["command_velocity_world_xy"],
            )
            np.testing.assert_array_equal(
                first.arrays["command_heading_world_yaw"],
                candidate.arrays["command_heading_world_yaw"],
            )
        self.assertNotEqual(
            first.resolved_config_sha256,
            legacy.resolved_config_sha256,
        )

    def test_every_20ms_row_has_kinematics_costs_selection_and_clearance(self):
        try:
            import mujoco
        except ImportError:
            context = mock.patch.dict({}, {})
        else:
            context = mock.patch.object(
                mujoco,
                "mj_step",
                side_effect=AssertionError("kinematic rollout called mj_step"),
            )
        with context:
            rollout = run_stair_rollout(
                self.resolved, "dense", device="cpu"
            )

        arrays = rollout.arrays
        self.assertEqual(len(arrays["time_s"]), 10)
        np.testing.assert_allclose(np.diff(arrays["time_s"]), 0.02, atol=1e-8)
        for name, trailing in (
            ("selected_frame", ()),
            ("selected_clip_index", ()),
            ("searched", ()),
            ("transitioned", ()),
            ("transition_rejected", ()),
            ("terrain_safety_override", ()),
            ("terrain_safety_override_rank", ()),
            ("motion_feature_cost", ()),
            ("terrain_feature_cost", ()),
            ("total_feature_cost", ()),
            ("selected_total_cost", ()),
            ("selected_transition_position_cost", ()),
            ("selected_transition_velocity_cost", ()),
            ("selected_transition_continuity_cost", ()),
            ("step_time_ns", ()),
            ("search_time_ns", ()),
            ("joint_position", (29,)),
            ("root_position_world", (3,)),
            ("root_orientation_world_wxyz", (4,)),
            ("feature_body_position_world", (3, 3)),
            ("terrain_patch_position_world", (91, 3)),
            ("foot_clearance_m", (2,)),
            ("progress_m", ()),
        ):
            self.assertEqual(arrays[name].shape, (10,) + trailing)
        self.assertTrue(np.isfinite(arrays["foot_clearance_m"]).all())
        self.assertTrue(np.isfinite(arrays["step_time_ns"]).all())
        for name in (
            "selected_transition_position_cost",
            "selected_transition_velocity_cost",
            "selected_transition_continuity_cost",
        ):
            self.assertEqual(arrays[name].dtype, np.dtype(np.float32))
            self.assertTrue(np.isfinite(arrays[name]).all())
            self.assertTrue(np.all(arrays[name] >= 0.0))
        self.assertEqual(
            rollout.metrics["rejected_transition_count"],
            int(arrays["transition_rejected"].sum()),
        )
        self.assertEqual(
            rollout.metrics["terrain_safety_override_count"],
            int(arrays["terrain_safety_override"].sum()),
        )
        self.assertEqual(
            rollout.metrics["maximum_terrain_safety_override_rank"],
            int(arrays["terrain_safety_override_rank"].max()),
        )
        self.assertEqual(
            rollout.metrics[
                "p95_positive_terrain_safety_override_rank"
            ],
            0.0,
        )
        np.testing.assert_allclose(
            arrays["motion_feature_cost"] + arrays["terrain_feature_cost"],
            arrays["total_feature_cost"],
            atol=2e-4,
        )
        self.assertTrue(
            np.all(
                arrays["selected_total_cost"]
                >= arrays["total_feature_cost"]
            )
        )
        np.testing.assert_allclose(
            arrays["selected_transition_position_cost"]
            + arrays["selected_transition_velocity_cost"],
            arrays["selected_transition_continuity_cost"],
            rtol=0,
            atol=2e-5,
        )

        extension = self.resolved.measurement_extension
        body = arrays["feature_body_position_world"][0]
        points = extension.alignment.matcher_to_scene_xy(
            __import__("torch").tensor(body[1:, :2], dtype=__import__("torch").float32)
        )
        height = extension.query_grid.sample_xy(points).numpy()
        np.testing.assert_allclose(
            arrays["foot_clearance_m"][0],
            body[1:, 2] - height,
            atol=1e-6,
        )
        penetration = np.maximum(0.0, -arrays["foot_clearance_m"])
        self.assertAlmostEqual(
            rollout.metrics["integrated_foot_penetration_m_s"],
            float(penetration.sum() * 0.02),
            places=7,
        )
        direction = np.asarray(
            rollout.resolved_config["reference_direction_matcher_xy"]
        )
        expected_progress = (
            arrays["root_position_world"][:, :2]
            - arrays["root_position_world"][0, :2]
        ) @ direction
        np.testing.assert_allclose(
            arrays["progress_m"], expected_progress, atol=1e-6
        )

    def test_acceptance_uses_all_five_frozen_thresholds(self):
        reference = self.resolved.resolved_config
        first_riser = reference["first_riser_progress_m"]
        horizontal = reference["reference_horizontal_progress_m"]
        height = reference["reference_root_height_gain_m"]
        flat = {
            "maximum_progress_m": horizontal * 0.7,
            "maximum_root_height_gain_m": height * 0.5,
            "minimum_foot_clearance_m": -0.2,
            "integrated_foot_penetration_m_s": 2.0,
            "reached_upper_landing": False,
        }
        dense = {
            "first_stair_selection_progress_m": first_riser - 0.21,
            "maximum_progress_m": horizontal * 0.91,
            "maximum_root_height_gain_m": height * 0.81,
            "minimum_foot_clearance_m": -0.029,
            "integrated_foot_penetration_m_s": 0.98,
            "reached_upper_landing": False,
        }
        accepted = evaluate_dense_acceptance(dense, flat, reference)
        self.assertTrue(accepted["passed"])
        self.assertEqual(len(accepted["criteria"]), 5)
        for field in (
            "first_stair_selection_progress_m",
            "maximum_progress_m",
            "maximum_root_height_gain_m",
            "minimum_foot_clearance_m",
            "integrated_foot_penetration_m_s",
        ):
            failed = dict(dense)
            if field == "first_stair_selection_progress_m":
                failed[field] = first_riser - 0.19
            elif field == "maximum_progress_m":
                failed[field] = horizontal * 0.89
            elif field == "maximum_root_height_gain_m":
                failed[field] = height * 0.79
            elif field == "minimum_foot_clearance_m":
                failed[field] = -0.031
            else:
                failed[field] = 1.02
            self.assertFalse(
                evaluate_dense_acceptance(failed, flat, reference)["passed"],
                field,
            )

    def test_saved_artifacts_are_pickle_free_and_structured(self):
        rollout = run_stair_rollout(self.resolved, "flat", device="cpu")
        output = self.root / "saved"
        save_stair_rollout(rollout, output)
        with np.load(output / "rollout.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(rollout.arrays))
        metrics = json.loads((output / "metrics.json").read_text("utf-8"))
        self.assertEqual(
            metrics["deterministic_sha256"], rollout.deterministic_sha256
        )
        events = [
            json.loads(line)
            for line in (output / "events.jsonl").read_text("utf-8").splitlines()
        ]
        self.assertEqual(events, list(rollout.events))
        for event in events:
            for name in (
                "selected_transition_position_cost",
                "selected_transition_velocity_cost",
                "selected_transition_continuity_cost",
            ):
                self.assertIn(name, event)
                self.assertIsInstance(event[name], float)
                self.assertTrue(np.isfinite(event[name]))
                self.assertGreaterEqual(event[name], 0.0)


if __name__ == "__main__":
    unittest.main()
