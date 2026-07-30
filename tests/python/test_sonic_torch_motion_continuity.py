"""Tests for row-aligned full-pose transition continuity costs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mm_sonic.schema import ContractError
from mm_sonic.torch_motion_continuity import TransitionContinuityDatabase
from mm_sonic.torch_motion_data import MotionFolder

from tests.python.torch_motion_test_utils import (
    build_takara_arrays,
    write_takara_arrays,
)


def _continuity_folder(
    root: Path,
    *,
    position_constant: bool = False,
    velocity_constant: bool = False,
) -> MotionFolder:
    """Write two distinguishable clips in reverse lexical creation order."""
    for relative_path, clip_offset in (("zeta", 10.0), ("alpha", -4.0)):
        arrays = build_takara_arrays(frames=49)
        frame = np.arange(49, dtype=np.float32)[:, None]
        joint = np.arange(29, dtype=np.float32)[None, :]

        if position_constant:
            arrays["joint_pos"] = np.full((49, 29), 2.5, dtype=np.float32)
        else:
            arrays["joint_pos"] = (
                clip_offset + 0.2 * frame + 0.01 * joint
            ).astype(np.float32)
            # This component is constant across every searchable row and clip.
            arrays["joint_pos"][:, 7] = 3.25

        if velocity_constant:
            arrays["joint_vel"] = np.full((49, 29), -1.5, dtype=np.float32)
        else:
            arrays["joint_vel"] = (
                -0.5 * clip_offset + 0.05 * frame - 0.02 * joint
            ).astype(np.float32)

        # Make excluded tail frames conspicuous so slicing mistakes are obvious.
        arrays["joint_pos"][-45:] += 1000.0
        arrays["joint_vel"][-45:] -= 1000.0
        if not position_constant:
            arrays["joint_pos"][:, 7] = 3.25
        if position_constant:
            arrays["joint_pos"][:] = 2.5
        if velocity_constant:
            arrays["joint_vel"][:] = -1.5
        write_takara_arrays(root / relative_path, arrays)
    return MotionFolder.load(root)


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", torch.cuda.current_device()))
    return devices


class TransitionContinuityDatabaseTests(unittest.TestCase):
    def test_source_states_follow_searchable_relative_path_row_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = _continuity_folder(Path(tmp))
            database = TransitionContinuityDatabase.from_folder(folder, "cpu")

        self.assertEqual(
            [clip.relative_path for clip in folder.clips],
            ["alpha/motion.npz", "zeta/motion.npz"],
        )
        expected_position = np.concatenate(
            [
                clip.joint_position[: clip.valid_frame_stop]
                for clip in folder.clips
            ],
            axis=0,
        )
        expected_velocity = np.concatenate(
            [
                clip.joint_velocity[: clip.valid_frame_stop]
                for clip in folder.clips
            ],
            axis=0,
        )
        position, velocity = database.source_states_copy()
        torch.testing.assert_close(
            position, torch.tensor(expected_position, dtype=torch.float32)
        )
        torch.testing.assert_close(
            velocity, torch.tensor(expected_velocity, dtype=torch.float32)
        )

        position.zero_()
        velocity.zero_()
        position_again, velocity_again = database.source_states_copy()
        self.assertFalse(torch.equal(position_again, position))
        self.assertFalse(torch.equal(velocity_again, velocity))

    def test_cost_components_match_independent_tensor_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = _continuity_folder(Path(tmp))
            expected_source_position = torch.tensor(
                np.concatenate(
                    [
                        clip.joint_position[: clip.valid_frame_stop]
                        for clip in folder.clips
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
            )
            expected_source_velocity = torch.tensor(
                np.concatenate(
                    [
                        clip.joint_velocity[: clip.valid_frame_stop]
                        for clip in folder.clips
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
            )

        for device in _devices():
            with self.subTest(device=device):
                database = TransitionContinuityDatabase.from_folder(folder, device)
                source_position = expected_source_position.to(device)
                source_velocity = expected_source_velocity.to(device)
                current_position = torch.linspace(
                    -0.4, 0.8, 29, dtype=torch.float32, device=device
                )
                current_velocity = torch.linspace(
                    0.7, -0.2, 29, dtype=torch.float32, device=device
                )
                position_scale = torch.std(
                    source_position, dim=0, correction=0
                )
                velocity_scale = torch.std(
                    source_velocity, dim=0, correction=0
                )
                active_position = torch.isfinite(position_scale) & (
                    position_scale > 0
                )
                active_velocity = torch.isfinite(velocity_scale) & (
                    velocity_scale > 0
                )

                self.assertFalse(bool(active_position[7].item()))
                self.assertTrue(bool(torch.any(active_position).item()))
                self.assertTrue(bool(torch.all(active_velocity).item()))

                expected_position = 0.25 * torch.mean(
                    torch.square(
                        (
                            source_position[:, active_position]
                            - current_position[active_position]
                        )
                        / position_scale[active_position]
                    ),
                    dim=1,
                )
                expected_velocity = 0.50 * torch.mean(
                    torch.square(
                        (
                            source_velocity[:, active_velocity]
                            - current_velocity[active_velocity]
                        )
                        / velocity_scale[active_velocity]
                    ),
                    dim=1,
                )
                costs = database.costs(
                    current_position,
                    current_velocity,
                    position_weight=0.25,
                    velocity_weight=0.50,
                )
                torch.testing.assert_close(costs.position, expected_position)
                torch.testing.assert_close(costs.velocity, expected_velocity)
                torch.testing.assert_close(
                    costs.total, expected_position + expected_velocity
                )
                for component in (costs.position, costs.velocity, costs.total):
                    self.assertEqual(component.dtype, torch.float32)
                    self.assertEqual(component.device, device)
                    self.assertEqual(
                        tuple(component.shape), (source_position.shape[0],)
                    )

                scales = database.scales_copy()
                torch.testing.assert_close(scales[0], position_scale)
                torch.testing.assert_close(scales[1], velocity_scale)
                torch.testing.assert_close(scales[2], active_position)
                torch.testing.assert_close(scales[3], active_velocity)

    def test_costs_reject_invalid_state_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = TransitionContinuityDatabase.from_folder(
                _continuity_folder(Path(tmp)), "cpu"
            )

        invalid_calls = (
            lambda: database.costs(
                torch.zeros(28),
                torch.zeros(29),
                position_weight=1,
                velocity_weight=1,
            ),
            lambda: database.costs(
                torch.zeros(29, dtype=torch.float64),
                torch.zeros(29),
                position_weight=1,
                velocity_weight=1,
            ),
            lambda: database.costs(
                torch.full((29,), float("nan")),
                torch.zeros(29),
                position_weight=1,
                velocity_weight=1,
            ),
            lambda: database.costs(
                torch.zeros(29),
                torch.zeros(29),
                position_weight=-1,
                velocity_weight=1,
            ),
            lambda: database.costs(
                torch.zeros(29),
                torch.zeros(29),
                position_weight=1,
                velocity_weight=float("inf"),
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(ContractError):
                invalid_call()

    def test_costs_reject_finite_inputs_that_overflow_cost_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = _continuity_folder(Path(tmp))

        for device in _devices():
            with self.subTest(device=device):
                database = TransitionContinuityDatabase.from_folder(folder, device)
                zero = torch.zeros(29, dtype=torch.float32, device=device)
                extreme_state = torch.full(
                    (29,),
                    torch.finfo(torch.float32).max,
                    dtype=torch.float32,
                    device=device,
                )
                invalid_calls = (
                    lambda: database.costs(
                        extreme_state,
                        zero,
                        position_weight=1,
                        velocity_weight=0,
                    ),
                    lambda: database.costs(
                        zero,
                        zero,
                        position_weight=1e300,
                        velocity_weight=0,
                    ),
                )
                for invalid_call in invalid_calls:
                    with self.subTest(
                        call=invalid_call
                    ), self.assertRaisesRegex(
                        ContractError,
                        "continuity costs must be finite and non-negative",
                    ):
                        invalid_call()

    def test_zero_weights_return_exact_float32_zero_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = TransitionContinuityDatabase.from_folder(
                _continuity_folder(Path(tmp)), "cpu"
            )

        costs = database.costs(
            torch.zeros(29),
            torch.zeros(29),
            position_weight=0,
            velocity_weight=0,
        )
        expected = torch.zeros(8, dtype=torch.float32)
        for component in (costs.position, costs.velocity, costs.total):
            self.assertEqual(component.dtype, torch.float32)
            self.assertTrue(torch.equal(component, expected))

    def test_constant_position_group_only_fails_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = TransitionContinuityDatabase.from_folder(
                _continuity_folder(Path(tmp), position_constant=True), "cpu"
            )

        current = torch.zeros(29)
        with self.assertRaises(ContractError):
            database.costs(
                current,
                current,
                position_weight=1,
                velocity_weight=0,
            )
        costs = database.costs(
            current,
            current,
            position_weight=0,
            velocity_weight=1,
        )
        self.assertTrue(torch.equal(costs.position, torch.zeros(8)))
        self.assertTrue(bool(torch.all(torch.isfinite(costs.total)).item()))

    def test_constant_velocity_group_only_fails_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = TransitionContinuityDatabase.from_folder(
                _continuity_folder(Path(tmp), velocity_constant=True), "cpu"
            )

        current = torch.zeros(29)
        with self.assertRaises(ContractError):
            database.costs(
                current,
                current,
                position_weight=0,
                velocity_weight=1,
            )
        costs = database.costs(
            current,
            current,
            position_weight=1,
            velocity_weight=0,
        )
        self.assertTrue(torch.equal(costs.velocity, torch.zeros(8)))
        self.assertTrue(bool(torch.all(torch.isfinite(costs.total)).item()))


if __name__ == "__main__":
    unittest.main()
