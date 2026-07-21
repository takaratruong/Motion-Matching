"""Probe P3: the immutable input-transition-to-prefix boundary trace.

Corrected Stage-R1 evidence semantics: the eight timestamps reflect the real
adapter ordering (publication is *sent* before both commits, so
``publication_sent_ns`` precedes ``committed_ns``), and the last two stamps are
the honest ``physics_release_requested_ns`` (when ``gate.release_steps`` was
requested) and ``simulation_advance_completed_ns`` (when the advance returned).
The two prior acknowledgement/first-frame timestamp claims are gone because the
responsive publish is ``wait=False`` (no GEAR acknowledgement after CONTROL
activation) and no first-simulated-frame timestamp is exposed.

The fabricated published-physical-root-displacement vector is removed because
the SONIC wire carries no root translation.  ``observed_mujoco_root_displacement``
is either a real state-log measurement (width-3 finite vector) or explicitly
``None`` when unavailable -- never a made-up zero.
"""

from __future__ import annotations

from dataclasses import fields
import unittest

from mm_sonic.boundary_trace import BoundaryTrace
from mm_sonic.joints import ContractError


# The eight monotone timestamps in their corrected non-decreasing order.
_TIMESTAMP_FIELDS = (
    "input_observed_ns",
    "sampled_ns",
    "mm_started_ns",
    "mm_completed_ns",
    "publication_sent_ns",
    "committed_ns",
    "physics_release_requested_ns",
    "simulation_advance_completed_ns",
)


def make_trace(**overrides) -> BoundaryTrace:
    base = dict(
        input_transition_id="transition-1",
        presented_prefix_id="prefix-1",
        input_observed_ns=1_000,
        sampled_ns=2_000,
        mm_started_ns=3_000,
        mm_completed_ns=4_000,
        publication_sent_ns=5_000,
        committed_ns=6_000,
        physics_release_requested_ns=7_000,
        simulation_advance_completed_ns=8_000,
        requested_velocity_mujoco=(0.5, 0.0, 0.0),
        requested_heading_mujoco_wxyz=(1.0, 0.0, 0.0, 0.0),
        generated_virtual_root_displacement_mujoco=(0.2, 0.0, 0.0),
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

        # Publication is sent before both commits: honest adapter ordering.
        self.assertLess(trace.publication_sent_ns, trace.committed_ns)

        # Frozen: the record is immutable.
        with self.assertRaises(Exception):
            trace.input_transition_id = "other"

        # Equal adjacent timestamps are allowed (non-decreasing, not strict).
        equal = make_trace(sampled_ns=1_000)
        self.assertEqual(equal.sampled_ns, equal.input_observed_ns)

        # Exactly the fourteen registered public constructor fields; the false
        # acknowledgement, first-simulated-frame, and published-physical-root
        # displacement fields are gone.
        self.assertEqual(
            tuple(f.name for f in fields(BoundaryTrace)),
            (
                "input_transition_id",
                "presented_prefix_id",
                "input_observed_ns",
                "sampled_ns",
                "mm_started_ns",
                "mm_completed_ns",
                "publication_sent_ns",
                "committed_ns",
                "physics_release_requested_ns",
                "simulation_advance_completed_ns",
                "requested_velocity_mujoco",
                "requested_heading_mujoco_wxyz",
                "generated_virtual_root_displacement_mujoco",
                "observed_mujoco_root_displacement",
            ),
        )

    def test_observed_root_may_be_unavailable_none(self):
        # When the state log does not bound the released chunk, the observed
        # displacement is explicitly None -- never a fabricated zero.
        trace = make_trace(observed_mujoco_root_displacement=None)
        self.assertIsNone(trace.observed_mujoco_root_displacement)

    def test_observed_root_zero_is_a_real_measurement_when_supplied(self):
        # A genuine (0,0,0) measurement is still a real width-3 vector; the
        # contract forbids *made-up* zeros, not a truthfully measured one.
        trace = make_trace(observed_mujoco_root_displacement=(0.0, 0.0, 0.0))
        self.assertEqual(trace.observed_mujoco_root_displacement, (0.0, 0.0, 0.0))


class BoundaryTraceRejectionTests(unittest.TestCase):
    def test_trace_rejects_nonmonotonic_or_unbound_records(self):
        # Empty identities.
        for identity in ("input_transition_id", "presented_prefix_id"):
            with self.assertRaises(ContractError):
                make_trace(**{identity: ""})

        # Non-monotonic timestamp sequence (commit before publication).
        with self.assertRaises(ContractError):
            make_trace(committed_ns=4_500)

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
        # Wrong observed width is still rejected (None is the only non-vector).
        with self.assertRaises(ContractError):
            make_trace(observed_mujoco_root_displacement=(0.1, 0.2))

    def test_last_two_timestamps_must_not_regress(self):
        with self.assertRaises(ContractError):
            make_trace(simulation_advance_completed_ns=6_500)


if __name__ == "__main__":
    unittest.main()
