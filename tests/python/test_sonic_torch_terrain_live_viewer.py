import inspect
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import mm_sonic.torch_terrain_live_viewer as live_module
from mm_sonic.joints import ContractError, PINNED_TARGET_TO_SOURCE_PERMUTATION
from mm_sonic.torch_motion_features import CommandTrajectory
from mm_sonic.torch_terrain_live_viewer import (
    ControlEdgeLatch,
    apply_kinematic_state,
    build_kinematic_scene,
    build_live_viewer_argument_parser,
    command_from_keys,
    dense_patch_positions,
    matcher_result_qpos,
)
from mm_sonic.torch_terrain_features import (
    TerrainSceneAlignment,
    _TorchHeightGrid,
)
from mm_sonic.torch_terrain_rollout import (
    ResolvedStairConfig,
)


def _minimal_g1_xml(path: Path) -> None:
    opening = [
        '<mujoco model="live-viewer-test">',
        '<compiler angle="radian"/>',
        '<option gravity="0 0 0"/>',
        "<asset/>",
        "<worldbody>",
        '<geom name="floor" type="plane" size="0 0 0.05"/>',
        '<body name="pelvis" pos="0 0 0.8">',
        '<joint name="floating_base_joint" type="free"/>',
        '<geom type="sphere" size="0.03" mass="1"/>',
    ]
    closing = []
    for index in range(29):
        opening.extend(
            [
                f'<body name="link_{index}" pos="0.03 0 0">',
                f'<joint name="joint_{index}" type="hinge" axis="0 0 1"/>',
                '<geom type="capsule" fromto="0 0 0 0.03 0 0" '
                'size="0.005" mass="0.01"/>',
            ]
        )
        closing.append("</body>")
    path.write_text(
        "\n".join(
            opening
            + list(reversed(closing))
            + ["</body>", "</worldbody>", "</mujoco>"]
        ),
        encoding="utf-8",
    )


class LiveCommandTests(unittest.TestCase):
    def test_release_finishes_only_the_active_chunk(self):
        latch_type = getattr(live_module, "LiveActionChunkLatch", None)
        self.assertIsNotNone(latch_type)
        latch = latch_type()
        moving = command_from_keys(
            frozenset(("W",)),
            reference_direction_xy=np.array([1.0, 0.0]),
            speed_mps=0.4,
            previous_heading_rad=0.0,
        )
        stopped = command_from_keys(
            frozenset(),
            reference_direction_xy=np.array([1.0, 0.0]),
            speed_mps=0.4,
            previous_heading_rad=0.0,
        )

        self.assertIs(latch.resolve(moving, selected_frame=396, endpoint=466), moving)
        self.assertIs(latch.resolve(stopped, selected_frame=418, endpoint=466), moving)
        self.assertIsNone(latch.resolve(stopped, selected_frame=465, endpoint=466))
        self.assertIsNone(latch.resolve(stopped, selected_frame=465, endpoint=466))

        latch.resolve(moving, selected_frame=396, endpoint=466)
        latch.reset()
        self.assertIsNone(latch.resolve(stopped, selected_frame=418, endpoint=466))

    def test_wasd_uses_reference_frame_and_normalizes_diagonals(self):
        reference = np.array([0.0, 1.0])
        speed = 0.4
        cases = {
            frozenset(): np.array([0.0, 0.0]),
            frozenset(("W",)): np.array([0.0, 0.4]),
            frozenset(("S",)): np.array([0.0, -0.4]),
            frozenset(("A",)): np.array([-0.4, 0.0]),
            frozenset(("D",)): np.array([0.4, 0.0]),
            frozenset(("W", "D")): np.array([1.0, 1.0])
            * (speed / math.sqrt(2.0)),
        }
        for pressed, expected in cases.items():
            with self.subTest(pressed=pressed):
                command = command_from_keys(
                    pressed,
                    reference_direction_xy=reference,
                    speed_mps=speed,
                    previous_heading_rad=0.37,
                )
                np.testing.assert_allclose(
                    command.velocity_world_xy, expected, atol=1e-7
                )
                expected_heading = (
                    0.37
                    if not np.linalg.norm(expected)
                    else math.atan2(expected[1], expected[0])
                )
                self.assertAlmostEqual(
                    command.heading_world_yaw, expected_heading
                )
                self.assertEqual(
                    command.advance_matcher,
                    bool(np.linalg.norm(expected)),
                )

    def test_space_has_stop_precedence(self):
        command = command_from_keys(
            frozenset(("SPACE", "W", "D")),
            reference_direction_xy=np.array([1.0, 0.0]),
            speed_mps=0.5,
            previous_heading_rad=-0.2,
        )
        np.testing.assert_array_equal(
            command.velocity_world_xy, np.zeros(2)
        )
        self.assertAlmostEqual(command.heading_world_yaw, -0.2)
        self.assertFalse(command.advance_matcher)

    def test_reset_and_exit_are_rising_edges(self):
        latch = ControlEdgeLatch()
        first = latch.update(frozenset(("BACKSPACE",)))
        held = latch.update(frozenset(("BACKSPACE",)))
        released = latch.update(frozenset())
        second = latch.update(frozenset(("BACKSPACE", "X")))
        self.assertTrue(first.reset_requested)
        self.assertFalse(first.exit_requested)
        self.assertFalse(held.reset_requested)
        self.assertFalse(released.reset_requested)
        self.assertTrue(second.reset_requested)
        self.assertTrue(second.exit_requested)

    def test_matcher_result_maps_target_joints_into_source_qpos(self):
        target = np.arange(29, dtype=np.float32) + 0.25
        result = SimpleNamespace(
            root_position_world=torch.tensor([1.0, 2.0, 3.0]),
            root_orientation_world_wxyz=torch.tensor([0.5, 0.5, 0.5, 0.5]),
            joint_position=torch.tensor(target),
        )
        qpos = matcher_result_qpos(result)
        self.assertEqual(qpos.shape, (36,))
        self.assertEqual(qpos.dtype, np.float64)
        np.testing.assert_array_equal(qpos[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(qpos[3:7], [0.5, 0.5, 0.5, 0.5])
        expected_source = np.empty(29, np.float64)
        expected_source[
            np.asarray(PINNED_TARGET_TO_SOURCE_PERMUTATION)
        ] = target
        np.testing.assert_array_equal(qpos[7:], expected_source)


class LiveMujocoSceneTests(unittest.TestCase):
    def test_multi_horizon_factory_uses_plain_27_value_base_database(self):
        resolved = SimpleNamespace(
            dataset=SimpleNamespace(root=Path("motions")),
            device=torch.device("cpu"),
            resolved_config={"reset_clip": "flat/motion.npz"},
            measurement_extension=object(),
        )
        base = SimpleNamespace(database=object())
        qualified = object()
        with (
            mock.patch.object(
                live_module.TorchMotionMatcher,
                "from_folder",
                return_value=base,
            ) as from_folder,
            mock.patch.object(
                live_module, "build_terrain_skill_inventory", return_value="skills"
            ) as build_skills,
            mock.patch.object(
                live_module, "build_horizon_inventory", return_value="horizons"
            ),
            mock.patch.object(
                live_module, "MujocoG1FootKinematics", return_value="feet"
            ),
            mock.patch(
                "mm_sonic.torch_g1_sole_kinematics."
                "MujocoG1SoleKinematics",
                return_value="soles",
            ),
            mock.patch.object(
                live_module,
                "horizon_search_config_from_experiment",
                return_value="search",
            ),
            mock.patch.object(
                live_module,
                "TerrainSkillHorizonMatcher",
                return_value=qualified,
            ) as adapter,
            mock.patch.object(
                live_module, "matcher_config_from_resolved", return_value="matcher"
            ),
            mock.patch.object(
                live_module.MotionFolder, "load", return_value="source-folder"
            ),
            mock.patch.object(
                live_module.TorchMotionDatabase,
                "from_folder",
                return_value=SimpleNamespace(normalization="normalization"),
            ) as source_database,
        ):
            result = live_module._build_live_matcher(
                resolved,
                "g1.xml",
                multi_horizon=True,
                contact_segment_policy=None,
                foothold_action_policy=None,
                maximum_source_contact_p95_m=0.033,
                normalization_source="baseline",
                maximum_translation_warp_m=0.025,
                maximum_yaw_warp_rad=0.10,
                minimum_endpoint_warp_yaw_rad=0.0,
                maximum_endpoint_warp_yaw_rad=3.141592653589793,
                maximum_endpoint_warp_terrain_delta_m=0.0,
                emitted_contact_preview=True,
                maximum_emitted_contact_candidates=64,
                maximum_emitted_contact_rescue_candidates=256,
            )

        self.assertIs(result, qualified)
        kwargs = from_folder.call_args.kwargs
        self.assertNotIn("extension", kwargs)
        self.assertNotIn("emitted_window_validator", kwargs)
        self.assertEqual(kwargs["config"], "matcher")
        self.assertEqual(kwargs["normalization_override"], "normalization")
        source_database.assert_called_once_with(
            "source-folder",
            device=resolved.device,
            reset_clip_path="flat/motion.npz",
        )
        build_skills.assert_called_once_with(
            resolved.dataset,
            base.database,
            maximum_source_contact_p95_m=0.033,
        )
        adapter.assert_called_once_with(
            base_matcher=base,
            skill_inventory="skills",
            horizon_inventory="horizons",
            dataset=resolved.dataset,
            query_terrain=resolved.measurement_extension,
            foot_kinematics="feet",
            sole_kinematics="soles",
            config="matcher",
            search_config="search",
            maximum_translation_warp_m=0.025,
            maximum_yaw_warp_rad=0.10,
            minimum_endpoint_warp_yaw_rad=0.0,
            maximum_endpoint_warp_yaw_rad=3.141592653589793,
            maximum_endpoint_warp_terrain_delta_m=0.0,
            emitted_contact_preview_enabled=True,
            maximum_emitted_contact_candidates=64,
            maximum_emitted_contact_rescue_candidates=256,
        )

    def test_multi_horizon_mode_is_explicit_and_exclusive(self):
        arguments = build_live_viewer_argument_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "horizon.json",
                "--g1-xml", "g1.xml",
                "--multi-horizon",
                "--continuous-skill",
                "--emitted-contact-preview",
                "--maximum-emitted-contact-candidates", "64",
                "--maximum-emitted-contact-rescue-candidates", "256",
                "--foot-lock",
                "--contact-phase-gate",
                "--swing-clearance-margin-m", "0.01",
                "--swing-plan-sigma-frames", "2.5",
                "--foot-correction-halflife-s", "0.02",
                "--maximum-source-contact-p95-m", "0.033",
                "--normalization-source", "baseline",
                "--maximum-translation-warp-m", "0.025",
                "--maximum-yaw-warp-rad", "0.10",
                "--minimum-endpoint-warp-yaw-rad", "0.0",
                "--maximum-endpoint-warp-yaw-rad", "3.141592653589793",
                "--maximum-endpoint-warp-terrain-delta-m", "0.0",
            ]
        )
        self.assertTrue(arguments.multi_horizon)
        self.assertTrue(arguments.continuous_skill)
        self.assertTrue(arguments.emitted_contact_preview)
        self.assertEqual(arguments.maximum_emitted_contact_candidates, 64)
        self.assertEqual(
            arguments.maximum_emitted_contact_rescue_candidates, 256
        )
        self.assertTrue(arguments.foot_lock)
        self.assertTrue(arguments.contact_phase_gate)
        self.assertEqual(arguments.swing_clearance_margin_m, 0.01)
        self.assertEqual(arguments.swing_plan_sigma_frames, 2.5)
        self.assertEqual(arguments.foot_correction_halflife_s, 0.02)
        self.assertEqual(arguments.maximum_source_contact_p95_m, 0.033)
        self.assertEqual(arguments.normalization_source, "baseline")
        self.assertEqual(arguments.maximum_translation_warp_m, 0.025)
        self.assertEqual(arguments.maximum_yaw_warp_rad, 0.10)
        self.assertEqual(arguments.minimum_endpoint_warp_yaw_rad, 0.0)
        self.assertAlmostEqual(arguments.maximum_endpoint_warp_yaw_rad, math.pi)
        self.assertEqual(arguments.maximum_endpoint_warp_terrain_delta_m, 0.0)
        live_module._validate_live_mode(
            multi_horizon=True,
            contact_segments=False,
            foothold_arm=None,
            foot_lock=True,
            contact_phase_gate=True,
            swing_clearance_margin_m=0.01,
            swing_plan_sigma_frames=2.5,
            foot_correction_halflife_s=0.02,
            emitted_contact_preview=True,
        )
        with self.assertRaisesRegex(ContractError, "mutually exclusive"):
            live_module._validate_live_mode(
                multi_horizon=True,
                contact_segments=True,
                foothold_arm=None,
            )
        with self.assertRaisesRegex(ContractError, "requires multi-horizon"):
            live_module._validate_live_mode(
                multi_horizon=False,
                contact_segments=False,
                foothold_arm=None,
                foot_lock=True,
            )
        with self.assertRaisesRegex(ContractError, "requires multi-horizon"):
            live_module._validate_live_mode(
                multi_horizon=False,
                contact_segments=False,
                foothold_arm=None,
                contact_phase_gate=True,
            )
        with self.assertRaisesRegex(ContractError, "requires multi-horizon"):
            live_module._validate_live_mode(
                multi_horizon=False,
                contact_segments=False,
                foothold_arm=None,
                continuous_skill=True,
            )
        with self.assertRaisesRegex(ContractError, "requires multi-horizon"):
            live_module._validate_live_mode(
                multi_horizon=False,
                contact_segments=False,
                foothold_arm=None,
                emitted_contact_preview=True,
            )
        with self.assertRaisesRegex(ContractError, "requires terrain foot lock"):
            live_module._validate_live_mode(
                multi_horizon=True,
                contact_segments=False,
                foothold_arm=None,
                swing_clearance_margin_m=0.01,
            )
        with self.assertRaisesRegex(ContractError, "requires swing clearance"):
            live_module._validate_live_mode(
                multi_horizon=True,
                contact_segments=False,
                foothold_arm=None,
                foot_lock=True,
                swing_plan_sigma_frames=2.5,
            )

    def test_horizon_overlay_does_not_require_legacy_cost_fields(self):
        result = SimpleNamespace(
            diagnostics=SimpleNamespace(
                selected_clip_path="terrain/motion.npz",
                selected_frame=31,
                step_time_ns=12_000_000,
            )
        )
        event = SimpleNamespace(
            target_frames=50,
            endpoint_frame_exclusive=62,
            cost=SimpleNamespace(entry=0.4, outcome=0.7, total=1.1),
        )
        command = live_module.LiveControlCommand(np.array([0.4, 0.0]), 0.0)

        overlay = live_module._diagnostic_overlay(
            result,
            command,
            focused=True,
            horizon_event=event,
        )

        self.assertIn("MULTI-HORIZON", overlay[2])
        self.assertIn("horizon=50", overlay[3])
        self.assertIn("endpoint=62", overlay[3])
        self.assertIn("entry=0.40", overlay[3])

    def test_layered_graph_arm_is_available_in_viewer(self):
        arguments = build_live_viewer_argument_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "contact.json",
                "--g1-xml", "g1.xml",
                "--contact-segments",
                "--foothold-arm", "layered-graph-hybrid",
            ]
        )

        self.assertEqual(arguments.foothold_arm, "layered-graph-hybrid")

    def test_turn_path_cost_weights_are_explicit_in_viewer(self):
        arguments = build_live_viewer_argument_parser().parse_args(
            [
                "--dataset", "motions",
                "--config", "contact.json",
                "--g1-xml", "g1.xml",
                "--turn-root-lateral-cost-weight", "1200",
                "--turn-root-progress-cost-weight", "75",
                "--turn-sequence-candidate-count", "32",
                "--heading-maintenance-yaw-cost-weight", "12",
                "--turn-lateral-root-warp-gain", "0.6",
                "--small-turn-lateral-root-warp-gain", "0.2",
                "--reversal-lateral-root-warp-gain", "0.1",
                "--strafe-action-gate",
            ]
        )

        self.assertEqual(arguments.turn_root_lateral_cost_weight, 1200.0)
        self.assertEqual(arguments.turn_root_progress_cost_weight, 75.0)
        self.assertEqual(arguments.turn_sequence_candidate_count, 32)
        self.assertEqual(arguments.heading_maintenance_yaw_cost_weight, 12.0)
        self.assertEqual(arguments.turn_lateral_root_warp_gain, 0.6)
        self.assertEqual(arguments.small_turn_lateral_root_warp_gain, 0.2)
        self.assertEqual(arguments.reversal_lateral_root_warp_gain, 0.1)
        self.assertTrue(arguments.strafe_action_gate)

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        grid = SimpleNamespace(
            origin_xy=torch.tensor([-1.0, -1.0], dtype=torch.float32),
            cell_size_m=0.1,
            height_z=torch.tensor(
                [
                    [0.0, 0.0, 0.1],
                    [0.0, 0.1, 0.2],
                    [0.1, 0.2, 0.3],
                ],
                dtype=torch.float32,
            ),
        )
        alignment = SimpleNamespace(
            translation_scene_xy=torch.zeros(2, dtype=torch.float32),
            yaw_scene_from_matcher=torch.zeros((), dtype=torch.float32),
        )
        cls.resolved = ResolvedStairConfig(
            dataset=SimpleNamespace(),
            measurement_extension=SimpleNamespace(
                query_grid=grid,
                alignment=alignment,
            ),
            resolved_config={},
            base_config_sha256="0" * 64,
            device=torch.device("cpu"),
        )
        cls.g1_xml = cls.root / "g1.xml"
        _minimal_g1_xml(cls.g1_xml)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_scene_contains_authenticated_heightfield_and_forwarded_qpos(self):
        import mujoco

        model, data = build_kinematic_scene(
            self.g1_xml, self.resolved
        )
        self.assertEqual(model.nq, 36)
        self.assertEqual(model.nhfield, 1)
        self.assertTrue(np.isfinite(model.hfield_data).all())
        self.assertGreater(float(model.hfield_data.max()), 0.0)
        qpos = np.zeros(36, np.float64)
        qpos[:3] = [0.3, -0.2, 1.0]
        qpos[3] = 1.0
        before = data.xpos.copy()
        apply_kinematic_state(mujoco, model, data, qpos)
        np.testing.assert_array_equal(data.qpos, qpos)
        self.assertFalse(np.array_equal(data.xpos, before))

    def test_module_has_no_physics_or_sonic_dependency(self):
        source = inspect.getsource(live_module)
        self.assertNotIn("mj_step(", source)
        self.assertNotIn("manual_demo", source)
        self.assertNotIn("SonicReferenceAdapter", source)
        self.assertIn(
            "matcher_config_from_resolved",
            inspect.getsource(live_module.run_live_viewer),
        )
        self.assertIn(
            "terrain_transition_validator_from_resolved",
            inspect.getsource(live_module._build_live_matcher),
        )
        self.assertIn(
            "emitted_window_validator",
            inspect.getsource(live_module._build_live_matcher),
        )
        help_text = build_live_viewer_argument_parser().format_help()
        for option in (
            "--dataset", "--config", "--g1-xml", "--device",
            "--contact-segments", "--foothold-arm", "--multi-horizon",
            "--foot-lock",
            "--contact-phase-gate",
            "--swing-clearance-margin-m",
            "--foot-correction-halflife-s",
            "--maximum-source-contact-p95-m",
            "--normalization-source",
        ):
            self.assertIn(option, help_text)
        self.assertIn(
            "contact_segment_policy",
            inspect.getsource(live_module.run_live_viewer),
        )
        self.assertIn(
            "foothold_action_policy",
            inspect.getsource(live_module.run_live_viewer),
        )
        self.assertIn(
            "predict_command_trajectory(",
            inspect.getsource(live_module.run_live_viewer),
        )
        self.assertIn(
            "marker_trajectory",
            inspect.getsource(live_module.run_live_viewer),
        )

    def test_dense_patch_requires_the_trajectory_used_by_search(self):
        self.assertEqual(
            tuple(inspect.signature(dense_patch_positions).parameters),
            ("result", "measurement", "trajectory"),
        )

    def test_dense_patch_markers_follow_commanded_path(self):
        class Grid:
            @staticmethod
            def sample_xy(points):
                return 0.1 * points[..., 0] + 0.2 * points[..., 1]

        measurement = SimpleNamespace(
            alignment=TerrainSceneAlignment(
                translation_scene_xy=torch.zeros(2),
                yaw_scene_from_matcher=torch.zeros(()),
            ),
            query_grid=Grid(),
        )
        result = SimpleNamespace(
            root_position_world=torch.tensor([0.0, 0.0, 0.8]),
            root_orientation_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        straight = CommandTrajectory(
            position_world_xy=torch.tensor(
                [[0.3, 0.0], [0.6, 0.0], [0.9, 0.0]]
            ),
            facing_world_xy=torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
            ),
        )
        lateral = CommandTrajectory(
            position_world_xy=torch.tensor(
                [[0.0, 0.3], [0.0, 0.6], [0.0, 0.9]]
            ),
            facing_world_xy=torch.tensor(
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
            ),
        )

        straight_points = dense_patch_positions(
            result, measurement, straight
        )
        lateral_points = dense_patch_positions(result, measurement, lateral)

        np.testing.assert_allclose(
            straight_points[:91].reshape(13, 7, 3)[:, 3, :2],
            np.stack((np.asarray(live_module.DENSE_FORWARD_M), np.zeros(13)), axis=1),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            lateral_points[:91].reshape(13, 7, 3)[:, 3, :2],
            np.stack((np.asarray(live_module.DENSE_FORWARD_M), np.zeros(13)), axis=1),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            straight_points[91:, :2],
            [[0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            lateral_points[91:, :2],
            [[0.0, 0.25], [0.0, 0.5], [0.0, 0.75], [0.0, 1.0]],
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            straight_points[:91, :2], lateral_points[:91, :2]
        )
        self.assertFalse(
            np.array_equal(straight_points[91:], lateral_points[91:])
        )

    def test_dense_patch_display_clamps_outside_grid_without_weakening_search(self):
        measurement = SimpleNamespace(
            alignment=TerrainSceneAlignment(
                translation_scene_xy=torch.zeros(2),
                yaw_scene_from_matcher=torch.zeros(()),
            ),
            query_grid=_TorchHeightGrid(
                origin_xy=torch.tensor((-0.1, -0.1)),
                cell_size_m=0.1,
                height_z=torch.zeros((3, 3)),
            ),
        )
        result = SimpleNamespace(
            root_position_world=torch.tensor([0.0, 0.0, 0.8]),
            root_orientation_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        )
        trajectory = CommandTrajectory(
            position_world_xy=torch.tensor(
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
            ),
            facing_world_xy=torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
            ),
        )

        points = dense_patch_positions(result, measurement, trajectory)

        self.assertEqual(points.shape, (95, 3))
        self.assertTrue(np.isfinite(points).all())
        with self.assertRaisesRegex(ContractError, "outside"):
            measurement.query_grid.sample_xy(torch.tensor([[1.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
