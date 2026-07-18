"""Pure, dependency-injected boundary loop for the SONIC operator."""

from __future__ import annotations

from typing import Callable

from .commands import CommandSample
from .joints import ContractError
from .operator import OperatorSampler, OperatorState


def run_operator_boundary_loop(
    *,
    key_source: Callable[[], OperatorState],
    sampler: OperatorSampler,
    run_chunk: Callable[[CommandSample], None],
    max_chunks: int | None = None,
) -> int:
    """Latch one key state and dispatch one command per chunk boundary."""
    if not isinstance(sampler, OperatorSampler):
        raise ContractError("boundary loop requires an OperatorSampler")
    if not callable(key_source):
        raise ContractError("boundary loop key_source must be callable")
    if not callable(run_chunk):
        raise ContractError("boundary loop run_chunk must be callable")
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks <= 0):
        raise ContractError("boundary loop max_chunks must be a positive integer")

    chunk_index = 0
    while max_chunks is None or chunk_index < max_chunks:
        state = key_source()
        if type(state) is not OperatorState:
            raise ContractError("boundary loop key_source must yield OperatorState")
        sampler.update(state)
        command = sampler.sample_boundary(chunk_index)
        if command is None:
            break
        run_chunk(command)
        chunk_index += 1
    return chunk_index
