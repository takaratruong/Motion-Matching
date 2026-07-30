import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mm_sonic.torch_terrain_rollout import (
    evaluate_dense_acceptance,
    load_experiment_config,
    resolve_stair_config,
    run_stair_rollout,
    save_stair_rollout,
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
            ("motion_feature_cost", ()),
            ("terrain_feature_cost", ()),
            ("total_feature_cost", ()),
            ("selected_total_cost", ()),
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


if __name__ == "__main__":
    unittest.main()
