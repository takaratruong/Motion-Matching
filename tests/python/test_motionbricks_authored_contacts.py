from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from mm_sonic.motionbricks_authored_contacts import (
    AuthoredFootContacts,
    collapse_authored_contacts,
    sample_authored_contacts,
)


class AuthoredContactContractTest(unittest.TestCase):
    def test_collapses_heel_and_toe_by_foot(self) -> None:
        self.assertEqual(
            collapse_authored_contacts((False, True, False, False)),
            (True, False),
        )
        self.assertEqual(
            collapse_authored_contacts((False, False, True, False)),
            (False, True),
        )

    def test_rejects_malformed_channels(self) -> None:
        for value in ((True, False), (True, False, True, np.nan)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collapse_authored_contacts(value)

    def test_samples_exact_unnormalized_feature_frame(self) -> None:
        features = np.arange(1 * 3 * 7, dtype=np.float32).reshape(1, 3, 7)

        class MotionRep:
            def __init__(self) -> None:
                self.calls: list[tuple[np.ndarray, bool, float]] = []

            def extract_foot_contacts(
                self,
                values: object,
                *,
                is_normalized: bool,
                contact_thresh: float,
            ) -> np.ndarray:
                array = np.asarray(values)
                self.calls.append(
                    (array.copy(), is_normalized, contact_thresh)
                )
                return np.asarray([[[False, True, False, False]]])

        motion_rep = MotionRep()
        agent = SimpleNamespace(
            frames={"model_features": features},
            _motion_rep=motion_rep,
        )

        result = sample_authored_contacts(agent, 1)

        self.assertEqual(
            result,
            AuthoredFootContacts(
                channels=(False, True, False, False),
                stance=(True, False),
                valid=True,
                reason="ok",
            ),
        )
        np.testing.assert_array_equal(
            motion_rep.calls[0][0], features[:, 1:2]
        )
        self.assertFalse(motion_rep.calls[0][1])
        self.assertEqual(motion_rep.calls[0][2], 0.5)

    def test_unavailable_contacts_fail_open_to_swing(self) -> None:
        agent = SimpleNamespace(frames={}, _motion_rep=None)

        result = sample_authored_contacts(agent, 0)

        self.assertFalse(result.valid)
        self.assertEqual(result.channels, (False, False, False, False))
        self.assertEqual(result.stance, (False, False))
        self.assertIn("model_features", result.reason)


if __name__ == "__main__":
    unittest.main()
