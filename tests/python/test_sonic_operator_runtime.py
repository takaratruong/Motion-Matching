from __future__ import annotations

import unittest

from mm_sonic.commands import CommandSample
from mm_sonic.joints import ContractError
from mm_sonic.operator import OperatorSampler, OperatorState
from mm_sonic.operator_runtime import run_operator_boundary_loop


class RunOperatorBoundaryLoopTests(unittest.TestCase):
    def test_latches_each_key_state_once_at_ascending_chunk_indices(self) -> None:
        states = iter(
            [
                OperatorState(forward=True),
                OperatorState(heading_left=True),
                OperatorState(stand=True),
            ]
        )
        dispatched: list[CommandSample] = []

        chunks = run_operator_boundary_loop(
            key_source=lambda: next(states),
            sampler=OperatorSampler(),
            run_chunk=dispatched.append,
            max_chunks=3,
        )

        self.assertEqual(chunks, 3)
        self.assertEqual([command.chunk_index for command in dispatched], [0, 1, 2])
        self.assertGreater(dispatched[0].requested_velocity_mujoco[0], 0.0)
        self.assertEqual(dispatched[2].requested_velocity_mujoco, (0.0, 0.0, 0.0))

    def test_terminate_key_stops_before_dispatching_that_chunk(self) -> None:
        states = iter(
            [
                OperatorState(forward=True),
                OperatorState(terminate=True),
                OperatorState(forward=True),
            ]
        )
        dispatched: list[CommandSample] = []

        chunks = run_operator_boundary_loop(
            key_source=lambda: next(states),
            sampler=OperatorSampler(),
            run_chunk=dispatched.append,
        )

        self.assertEqual(chunks, 1)
        self.assertEqual([command.chunk_index for command in dispatched], [0])

    def test_finite_chunk_limit_stops_without_terminate(self) -> None:
        dispatched: list[CommandSample] = []

        chunks = run_operator_boundary_loop(
            key_source=lambda: OperatorState(forward=True),
            sampler=OperatorSampler(),
            run_chunk=dispatched.append,
            max_chunks=2,
        )

        self.assertEqual(chunks, 2)
        self.assertEqual([command.chunk_index for command in dispatched], [0, 1])

    def test_rejects_non_positive_chunk_limit(self) -> None:
        with self.assertRaises(ContractError):
            run_operator_boundary_loop(
                key_source=lambda: OperatorState(),
                sampler=OperatorSampler(),
                run_chunk=lambda command: None,
                max_chunks=0,
            )


if __name__ == "__main__":
    unittest.main()
