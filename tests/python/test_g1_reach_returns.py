import unittest

import numpy as np

from resources.g1_reach_builder.returns import find_return_stop


def synthetic_reach(
    *,
    outbound: int,
    contact_pause: int,
    inbound: int,
    nominal_pause: int,
) -> np.ndarray:
    outbound_trace = np.linspace(
        [0.0, 0.0, 0.0], [0.48, 0.0, 0.0], outbound
    )
    contact_trace = np.repeat(outbound_trace[-1:], contact_pause, axis=0)
    inbound_trace = np.linspace(
        outbound_trace[-1], [0.0, 0.0, 0.0], inbound, endpoint=False
    )
    nominal_trace = np.zeros((nominal_pause, 3), np.float64)
    return np.concatenate(
        (outbound_trace, contact_trace, inbound_trace, nominal_trace), axis=0
    )


class PairedReturnSegmentationTest(unittest.TestCase):
    def test_finds_stable_return_after_contact_pause(self):
        trace = synthetic_reach(
            outbound=30, contact_pause=3, inbound=24, nominal_pause=6
        )

        stop = find_return_stop(trace, 0, 29)

        self.assertEqual(stop, len(trace))

    def test_rejects_motion_without_retraction(self):
        trace = np.zeros((80, 3), np.float64)
        trace[20:, 0] = 0.45

        with self.assertRaisesRegex(ValueError, "paired return"):
            find_return_stop(trace, 0, 20)


if __name__ == "__main__":
    unittest.main()
