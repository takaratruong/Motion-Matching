"""Probe P3: the immutable input-transition-to-prefix boundary trace."""

from __future__ import annotations

from dataclasses import fields, replace
import unittest

from mm_sonic.boundary_trace import BoundaryTrace
from mm_sonic.joints import ContractError


# The eight monotone timestamps in their design-mandated non-decreasing order.
_TIMESTAMP_FIELDS = (
    "input_observed_ns",
    "sampled_ns",
    "mm_started_ns",
    "mm_completed_ns",
    "committed_ns",
    "published_ack_ns",
    "physics_released_ns",
    "first_simulated_frame_ns",
)


def make_trace(**overrides) -> BoundaryTrace:
    base = dict(
        input_transition_id="transition-1",
        presented_prefix_id="prefix-1",
        input_observed_ns=1_000,
        sampled_ns=2_000,
        mm_started_ns=3_000,
        mm_completed_ns=4_000,
        committed_ns=5_000,
        published_ack_ns=6_000,
        physics_released_ns=7_000,
        first_simulated_frame_ns=8_000,
        requested_velocity_mujoco=(0.5, 0.0, 0.0),
        requested_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
        generated_virtual_root_displacement_mujoco=(0.2, 0.0, 0.0),
        published_physical_root_displacement_mujoco=(0.18, 0.0, 0.0),
        observed_mujoco_root_displacement=(0.17, 0.0, 0.0),
    )
    base.update(overrides)
    return BoundaryTrace(**base)


class BoundaryTraceBindingTests(unittest.TestCase):
    def test_trace_binds_one_transition_to_one_prefix_with_monotonic_stamps(self):
        trace = make_trace()
        self.assertEqual(trace.input_transition_id, "transition-1")
        self.assertEqual(trace.presented_prefix_id, "prefix-1")
        stamps = [getattr(trace, name) for name in _TIMESTAMP_FIELDS]
        self.assertEqual(stamps, sorted(stamps))
        self.assertTrue(all(isinstance(value, int) for value in stamps))

        # Frozen: the record is immutable.
        with self.assertRaises(Exception):
            trace.input_transition_id = "other"

        # Equal adjacent timestamps are allowed (non-decreasing, not strict).
        equal = make_trace(sampled_ns=1_000)
        self.assertEqual(equal.sampled_ns, equal.input_observed_ns)

        # Exactly the fifteen registered public constructor fields.
        self.assertEqual(
            tuple(f.name for f in fields(BoundaryTrace)),
            (
                "input_transition_id",
                "presented_prefix_id",
                "input_observed_ns",
                "sampled_ns",
                "mm_started_ns",
                "mm_completed_ns",
                "committed_ns",
                "published_ack_ns",
                "physics_released_ns",
                "first_simulated_frame_ns",
                "requested_velocity_mujoco",
                "requested_heading_mujoco_wxyz",
                "generated_virtual_root_displacement_mujoco",
                "published_physical_root_displacement_mujoco",
                "observed_mujoco_root_displacement",
            ),
        )


class BoundaryTraceRejectionTests(unittest.TestCase):
    def test_trace_rejects_nonmonotonic_or_unbound_records(self):
        # Empty identities.
        for identity in ("input_transition_id", "presented_prefix_id"):
            with self.assertRaises(ContractError):
                make_trace(**{identity: ""})

        # Non-monotonic timestamp sequence (physics before commit).
        with self.assertRaises(ContractError):
            make_trace(physics_released_ns=4_500)

        # Negative and non-integer timestamps.
        with self.assertRaises(ContractError):
            make_trace(input_observed_ns=-1)
        with self.assertRaises(ContractError):
            make_trace(sampled_ns=2_000.0)
        with self.assertRaises(ContractError):
            make_trace(mm_started_ns=True)

        # Wrong vector widths.
        with self.assertRaises(ContractError):
            make_trace(requested_velocity_mujoco=(0.5, 0.0))
        with self.assertRaises(ContractError):
            make_trace(requested_heading_mujoco_wxyz=(1.0, 0.0, 0.0))

        # Non-finite numeric vectors.
        with self.assertRaises(ContractError):
            make_trace(
                generated_virtual_root_displacement_mujoco=(float("nan"), 0.0, 0.0)
            )
        with self.assertRaises(ContractError):
            make_trace(
                observed_mujoco_root_displacement=(float("inf"), 0.0, 0.0)
            )

    def test_last_two_timestamps_must_not_regress(self):
        with self.assertRaises(ContractError):
            make_trace(first_simulated_frame_ns=6_500)


if __name__ == "__main__":
    unittest.main()
