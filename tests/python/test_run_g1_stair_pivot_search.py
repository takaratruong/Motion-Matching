import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mm_sonic.joints import ContractError


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_g1_stair_pivot_search",
    ROOT / "resources" / "run_g1_stair_pivot_search.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunG1StairPivotSearchTest(unittest.TestCase):
    def test_stationary_seed_discards_unplayed_source_velocity(self):
        arrays = {
            "joint_position": np.stack(
                (np.zeros(29), np.ones(29)),
            ),
            "root_position_world": np.asarray(
                ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0)),
            ),
            "foot_position_world": np.zeros((2, 2, 3)),
            "root_orientation_world_wxyz": np.tile(
                (1.0, 0.0, 0.0, 0.0),
                (2, 1),
            ),
        }

        boundary = MODULE._boundary(arrays, 1, stationary=True)

        np.testing.assert_array_equal(boundary.joint_velocity, 0.0)
        np.testing.assert_array_equal(boundary.root_velocity_world, 0.0)

    def test_parser_exposes_two_step_offline_search_contract(self):
        args = MODULE._parser().parse_args(
            (
                "--graph",
                "graph.json",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--incoming-edge",
                "abc",
                "--g1-xml",
                "g1.xml",
                "--output",
                "output",
                "--total-yaw-delta-rad",
                "1.57079632679",
            )
        )

        self.assertEqual(args.step_count, 2)
        self.assertEqual(args.frame_count_per_step, 61)
        self.assertEqual(args.beam_width, 3)
        self.assertEqual(args.minimum_stance_width_m, 0.18)
        self.assertEqual(args.maximum_stance_width_m, 0.40)
        self.assertIsNone(args.target_root_displacement_u_m)
        self.assertIsNone(args.target_root_displacement_v_m)
        self.assertEqual(args.maximum_terminal_displacement_error_m, 0.03)
        self.assertIsNone(args.root_height_offset_m)
        self.assertEqual(
            args.maximum_joint_acceleration_rad_s2,
            100.0,
        )
        self.assertEqual(
            args.maximum_terminal_sole_contact_error_m,
            0.025,
        )

    def test_parser_collects_a_per_step_root_height_lattice(self):
        args = MODULE._parser().parse_args(
            (
                "--graph",
                "graph.json",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--incoming-edge",
                "abc",
                "--g1-xml",
                "g1.xml",
                "--output",
                "output",
                "--total-yaw-delta-rad",
                "0",
                "--root-height-offset-m",
                "0",
                "--root-height-offset-m",
                "-0.04",
            )
        )

        self.assertEqual(args.root_height_offset_m, [0.0, -0.04])

    def test_target_displacement_has_exactly_stair_u_and_v(self):
        args = MODULE._parser().parse_args(
            (
                "--graph",
                "graph.json",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--incoming-edge",
                "abc",
                "--g1-xml",
                "g1.xml",
                "--output",
                "output",
                "--total-yaw-delta-rad",
                "0",
                "--target-root-displacement-u-m",
                "-0.2",
                "--target-root-displacement-v-m",
                "0.1",
            )
        )

        np.testing.assert_allclose(
            MODULE._target_displacement(args),
            (-0.2, 0.1),
        )

    def test_explicit_step_schedule_preserves_direction_changes(self):
        args = MODULE._parser().parse_args(
            (
                "--graph",
                "graph.json",
                "--dataset",
                "dataset",
                "--config",
                "config.json",
                "--incoming-edge",
                "abc",
                "--g1-xml",
                "g1.xml",
                "--output",
                "output",
                "--step-yaw-delta-rad",
                "0.4",
                "--step-yaw-delta-rad",
                "-0.4",
                "--step-target-root-u-m",
                "0.1",
                "--step-target-root-u-m",
                "0.1",
                "--step-target-root-v-m",
                "0",
                "--step-target-root-v-m",
                "-0.1",
            )
        )

        yaw, targets = MODULE._step_schedule(args)

        np.testing.assert_allclose(yaw, (0.4, -0.4))
        np.testing.assert_allclose(targets, ((0.1, 0.0), (0.1, -0.1)))

    def test_stance_width_gate_rejects_collapsed_footholds(self):
        self.assertFalse(
            MODULE._valid_stance_width(
                ((0.0, 0.0), (0.10, 0.0)),
                minimum_m=0.18,
                maximum_m=0.40,
            )
        )
        self.assertTrue(
            MODULE._valid_stance_width(
                ((0.0, 0.0), (0.25, 0.0)),
                minimum_m=0.18,
                maximum_m=0.40,
            )
        )

    def test_offset_lattice_rotates_from_stair_frame_to_world(self):
        offsets = MODULE._stair_offsets_world(
            ((1.0, 0.0), (0.0, 1.0)),
            ascent_world_xy=(0.0, 2.0),
        )

        np.testing.assert_allclose(offsets, ((0.0, 1.0), (-1.0, 0.0)))

    def test_segment_ranges_share_exact_landing_boundaries(self):
        self.assertEqual(
            MODULE._landing_segment_ranges(
                total_frame_count=271,
                frame_count_per_step=91,
                step_count=3,
            ),
            ((0, 91), (90, 181), (180, 271)),
        )

        with self.assertRaisesRegex(ContractError, "frame count"):
            MODULE._landing_segment_ranges(
                total_frame_count=270,
                frame_count_per_step=91,
                step_count=3,
            )

    def test_world_displacement_projects_into_stair_frame(self):
        result = MODULE._world_displacement_to_stair(
            (-2.0, 3.0),
            ascent_world_xy=(0.0, 1.0),
        )

        np.testing.assert_allclose(result, (3.0, 2.0))

    def test_graph_ascent_requires_matching_dataset_manifest(self):
        source = {
            "dataset_manifest_sha256": "a" * 64,
            "stair_frame": {
                "origin_world_xy": [1.0, 2.0],
                "ascent_world_yaw": math.pi / 2.0,
                "width_m": 1.0,
                "tread_depth_m": 0.3,
                "riser_height_m": 0.2,
                "tread_count": 4,
                "lateral_cell_m": 0.1,
            },
        }

        ascent = MODULE._authenticated_graph_ascent_world_xy(
            source,
            dataset_manifest_sha256="a" * 64,
        )

        np.testing.assert_allclose(ascent, (0.0, 1.0), atol=1e-12)
        with self.assertRaisesRegex(ContractError, "manifest"):
            MODULE._authenticated_graph_ascent_world_xy(
                source,
                dataset_manifest_sha256="b" * 64,
            )

    def test_joint_acceleration_uses_emitted_position_second_difference(self):
        joints = np.zeros((4, 29), dtype=np.float64)
        joints[:, 3] = (0.0, 0.0, 0.04, 0.12)

        acceleration = MODULE._maximum_joint_acceleration_rad_s2(
            joints,
            dt_s=0.02,
        )

        self.assertAlmostEqual(acceleration, 100.0)

    def test_terminal_sole_contact_metrics_include_float_and_tilt(self):
        maximum_error, maximum_spread = MODULE._sole_contact_metrics(
            np.asarray(
                (
                    (-0.01, 0.00, 0.02),
                    (0.04, 0.05, 0.06),
                )
            )
        )

        self.assertAlmostEqual(maximum_error, 0.06)
        self.assertAlmostEqual(maximum_spread, 0.03)

    def test_join_recomputes_velocity_from_complete_emitted_path(self):
        previous_position = np.zeros((3, 29), dtype=np.float64)
        previous_position[:, 0] = (0.0, 0.01, 0.04)
        current_position = np.zeros((3, 29), dtype=np.float64)
        current_position[:, 0] = (0.04, 0.09, 0.16)

        joined = MODULE._join(
            {
                "joint_position": previous_position,
                "joint_velocity": np.zeros_like(previous_position),
            },
            {
                "joint_position": current_position,
                "joint_velocity": np.zeros_like(current_position),
            },
        )

        np.testing.assert_allclose(
            joined["joint_velocity"],
            np.gradient(
                joined["joint_position"],
                0.02,
                axis=0,
                edge_order=2,
            ),
        )

    def test_optimized_archive_is_bound_to_claimed_hash_and_shapes(self):
        arrays = {
            "joint_position": np.zeros((3, 29), dtype=np.float64),
            "joint_velocity": np.zeros((3, 29), dtype=np.float64),
            "root_position_world": np.zeros((3, 3), dtype=np.float64),
            "root_orientation_world_wxyz": np.tile(
                (1.0, 0.0, 0.0, 0.0),
                (3, 1),
            ),
            "foot_position_world": np.zeros((3, 2, 3), dtype=np.float64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connector.npz"
            np.savez_compressed(path, **arrays)

            loaded = MODULE._load_optimized_arrays(
                path,
                expected_sha256=MODULE._sha256(arrays),
            )
            self.assertEqual(loaded["joint_position"].shape, (3, 29))

            with self.assertRaisesRegex(ContractError, "identity"):
                MODULE._load_optimized_arrays(
                    path,
                    expected_sha256="0" * 64,
                )

            np.savez_compressed(
                path,
                joint_position=np.zeros((3, 29)),
            )
            with self.assertRaisesRegex(ContractError, "arrays"):
                MODULE._load_optimized_arrays(
                    path,
                    expected_sha256="0" * 64,
                )

    def test_boundary_hash_verification_rejects_a_claimed_seam(self):
        state = {
            "joint_position": np.zeros(29),
            "joint_velocity": np.zeros(29),
            "root_position_world": np.zeros(3),
            "root_orientation_world_wxyz": np.asarray(
                (1.0, 0.0, 0.0, 0.0)
            ),
            "foot_position_world": np.zeros((2, 3)),
            "foot_surface_height_m": np.zeros(2),
        }

        actual = MODULE._verified_optimized_boundary_sha256(
            expected_sha256=None,
            **state,
        )
        self.assertEqual(len(actual), 64)
        with self.assertRaisesRegex(ContractError, "boundary"):
            MODULE._verified_optimized_boundary_sha256(
                expected_sha256="0" * 64,
                **state,
            )

    def test_connector_start_is_pinned_to_exact_incoming_boundary(self):
        boundary = MODULE.StairConnectorBoundary(
            joint_position=np.arange(29, dtype=np.float64),
            joint_velocity=np.full(29, 0.25),
            root_position_world=np.asarray((1.0, 2.0, 3.0)),
            root_velocity_world=np.zeros(3),
            root_orientation_world_wxyz=np.asarray(
                (1.0, 0.0, 0.0, 0.0)
            ),
            foot_position_world=np.ones((2, 3)),
        )
        arrays = {
            "joint_position": np.zeros((2, 29)),
            "joint_velocity": np.zeros((2, 29)),
            "root_position_world": np.zeros((2, 3)),
            "root_orientation_world_wxyz": np.zeros((2, 4)),
            "foot_position_world": np.zeros((2, 2, 3)),
        }

        MODULE._pin_connector_start(arrays, boundary)

        np.testing.assert_array_equal(
            arrays["joint_position"][0],
            boundary.joint_position,
        )
        np.testing.assert_array_equal(
            arrays["joint_velocity"][0],
            boundary.joint_velocity,
        )
        np.testing.assert_array_equal(
            arrays["foot_position_world"][0],
            boundary.foot_position_world,
        )


if __name__ == "__main__":
    unittest.main()
