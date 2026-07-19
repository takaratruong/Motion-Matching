from __future__ import annotations

import math
import unittest

from mm_sonic.commands import CommandSample
from mm_sonic.joints import ContractError
from mm_sonic.holden_control import (
    CameraState,
    HoldenControlMapper,
    MappedControlState,
    NormalizedControlState,
)


class NormalizedControlStateTests(unittest.TestCase):
    def test_defaults_are_neutral(self) -> None:
        state = NormalizedControlState()
        self.assertEqual(state.left_x, 0.0)
        self.assertEqual(state.left_z, 0.0)
        self.assertEqual(state.right_x, 0.0)
        self.assertEqual(state.right_z, 0.0)
        self.assertFalse(state.strafe)
        self.assertFalse(state.walk)
        self.assertEqual(state.zoom, 0.0)
        self.assertFalse(state.stand)
        self.assertFalse(state.terminate)

    def test_nonfinite_axis_rejected(self) -> None:
        with self.assertRaises(ContractError):
            NormalizedControlState(left_x=float("nan"))

    def test_axis_out_of_range_rejected(self) -> None:
        with self.assertRaises(ContractError):
            NormalizedControlState(left_z=1.5)

    def test_nonbool_flag_rejected(self) -> None:
        with self.assertRaises(ContractError):
            NormalizedControlState(strafe=1)


class HoldenControlMapperTests(unittest.TestCase):
    def test_forward_and_left_match_mujoco_axes(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        forward = mapper.update(NormalizedControlState(left_z=-1.0), 0.02)
        self.assertAlmostEqual(forward.velocity_mujoco[0], 0.9, places=6)
        self.assertAlmostEqual(forward.velocity_mujoco[1], 0.0, places=6)
        left = mapper.update(NormalizedControlState(left_x=-1.0), 0.02)
        self.assertAlmostEqual(left.velocity_mujoco[0], 0.0, places=6)
        self.assertAlmostEqual(left.velocity_mujoco[1], 0.6, places=6)

    def test_camera_relative_forward_and_strafe_heading_are_independent(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=math.pi / 2.0)
        mapped = mapper.update(
            NormalizedControlState(
                left_z=-1.0,
                right_x=-1.0,
                strafe=True,
            ),
            0.02,
        )
        self.assertAlmostEqual(mapped.velocity_mujoco[0], 0.0, places=6)
        self.assertAlmostEqual(mapped.velocity_mujoco[1], 0.9, places=6)
        self.assertNotEqual(
            mapped.desired_heading_mujoco_wxyz, (1.0, 0.0, 0.0, 0.0)
        )

    def test_shift_converges_to_registered_walk_speeds(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        mapped = None
        for _ in range(100):
            mapped = mapper.update(
                NormalizedControlState(left_z=-1.0, walk=True), 0.02
            )
        assert mapped is not None
        self.assertAlmostEqual(mapped.velocity_mujoco[0], 0.5, places=4)

    def test_stand_zeroes_velocity_but_retains_heading(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.3)
        moving = mapper.update(NormalizedControlState(left_z=-1.0), 0.02)
        standing = mapper.update(
            NormalizedControlState(left_z=-1.0, stand=True), 0.02
        )
        self.assertEqual(standing.velocity_mujoco, (0.0, 0.0, 0.0))
        self.assertEqual(
            standing.desired_heading_mujoco_wxyz,
            moving.desired_heading_mujoco_wxyz,
        )

    def test_deadzone_ignores_small_stick(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        mapped = mapper.update(NormalizedControlState(left_z=-0.1), 0.02)
        self.assertEqual(mapped.velocity_mujoco, (0.0, 0.0, 0.0))

    def test_idle_frame_retains_heading(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        moving = mapper.update(NormalizedControlState(left_x=-1.0), 0.02)
        idle = mapper.update(NormalizedControlState(), 0.02)
        self.assertEqual(
            idle.desired_heading_mujoco_wxyz,
            moving.desired_heading_mujoco_wxyz,
        )

    def test_zoom_clamps_distance(self) -> None:
        mapper = HoldenControlMapper(
            initial_heading_yaw_rad=0.0, initial_distance_m=0.2
        )
        mapped = mapper.update(NormalizedControlState(zoom=-1.0), 0.02)
        self.assertGreaterEqual(mapped.camera.distance_m, 0.1)

    def test_nonpositive_dt_rejected(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        with self.assertRaises(ContractError):
            mapper.update(NormalizedControlState(), 0.0)

    def test_command_returns_sample_and_terminates(self) -> None:
        mapper = HoldenControlMapper(initial_heading_yaw_rad=0.0)
        mapped = mapper.update(NormalizedControlState(left_z=-1.0), 0.02)
        sample = mapper.command(0, mapped)
        self.assertIsInstance(sample, CommandSample)
        self.assertEqual(sample.chunk_index, 0)
        terminated = mapper.update(
            NormalizedControlState(terminate=True), 0.02
        )
        self.assertIsNone(mapper.command(1, terminated))


if __name__ == "__main__":
    unittest.main()
