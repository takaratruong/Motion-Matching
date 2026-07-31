import inspect
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import mm_sonic.torch_terrain_live_viewer as live_module
from mm_sonic.joints import PINNED_TARGET_TO_SOURCE_PERMUTATION
from mm_sonic.torch_terrain_live_viewer import (
    ControlEdgeLatch,
    apply_kinematic_state,
    build_kinematic_scene,
    build_live_viewer_argument_parser,
    command_from_keys,
    matcher_result_qpos,
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
            inspect.getsource(live_module.run_live_viewer),
        )
        self.assertIn(
            "emitted_window_validator",
            inspect.getsource(live_module.run_live_viewer),
        )
        help_text = build_live_viewer_argument_parser().format_help()
        for option in (
            "--dataset", "--config", "--g1-xml", "--device",
            "--contact-segments",
        ):
            self.assertIn(option, help_text)
        self.assertIn(
            "contact_segment_policy",
            inspect.getsource(live_module.run_live_viewer),
        )


if __name__ == "__main__":
    unittest.main()
