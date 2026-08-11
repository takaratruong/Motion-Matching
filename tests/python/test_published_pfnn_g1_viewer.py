#!/usr/bin/env python3

import queue
import threading
import time
import unittest

import numpy as np

from mm_sonic import published_pfnn_g1_viewer as viewer


class _ObservedQueue(queue.Queue):
    def __init__(self, *, target: object) -> None:
        super().__init__(maxsize=1)
        self.target = target
        self.target_inserted = threading.Event()

    def put(self, item, block=True, timeout=None):
        result = super().put(item, block=block, timeout=timeout)
        if item is self.target:
            self.target_inserted.set()
        return result


def _qpos_frame(value: float) -> viewer.Frame:
    return viewer.Frame(
        qpos_wxyz=np.full(36, value, dtype=np.float64),
        source_root_pos_zup_m=np.full(3, value, dtype=np.float64),
        input_space="gmr",
    )


def _raw_frame(value: float) -> viewer.Frame:
    return viewer.Frame(
        positions=np.full((31, 3), value, dtype=np.float64),
        rotations_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (31, 1)),
        input_space="source",
    )


class ProducerQueuePolicyTest(unittest.TestCase):
    def _finish(self, thread: threading.Thread, output: queue.Queue) -> None:
        deadline = time.monotonic() + 1.0
        while thread.is_alive() and time.monotonic() < deadline:
            try:
                output.get(timeout=0.02)
            except queue.Empty:
                pass
        thread.join(timeout=0.1)
        self.assertFalse(thread.is_alive(), "producer did not terminate during test cleanup")

    def test_pre_retargeted_frames_replace_stale_display_frame(self) -> None:
        first = _qpos_frame(1.0)
        latest = _qpos_frame(2.0)
        output = _ObservedQueue(target=latest)
        thread = threading.Thread(
            target=viewer._producer, args=(iter((first, latest)), output), daemon=True
        )
        thread.start()
        try:
            self.assertTrue(
                output.target_inserted.wait(timeout=0.5),
                "latest qpos frame backlogged behind stale display frame",
            )
            self.assertIs(output.get_nowait(), latest)
        finally:
            self._finish(thread, output)

    def test_raw_source_frames_remain_sequential(self) -> None:
        first = _raw_frame(1.0)
        second = _raw_frame(2.0)
        output = _ObservedQueue(target=second)
        thread = threading.Thread(
            target=viewer._producer, args=(iter((first, second)), output), daemon=True
        )
        thread.start()
        try:
            self.assertFalse(output.target_inserted.wait(timeout=0.05))
            self.assertIs(output.get_nowait(), first)
            self.assertTrue(output.target_inserted.wait(timeout=0.5))
            self.assertIs(output.get_nowait(), second)
        finally:
            self._finish(thread, output)


class ExpectedWorldTests(unittest.TestCase):
    def test_rejects_frame_from_a_different_pfnn_world(self) -> None:
        frame = viewer.Frame(metadata={"world": 4})

        with self.assertRaisesRegex(
            ValueError, "PFNN world mismatch: expected 5, received 4"
        ):
            viewer.require_expected_world(frame, 5)


if __name__ == "__main__":
    unittest.main()
