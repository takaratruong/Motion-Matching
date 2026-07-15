import copy
import json
import struct
import unittest

import numpy as np

from resources.g1_interaction_builder.phases import (
    derive_interaction_labels,
    hand_in_object,
    quaternion_angle,
)
from resources.g1_interaction_builder.schema import InteractionPhase
from tests.python.interaction_fixture import canonical_pickup_fixture


_EXPECTED_KEYS = (
    "actual_carry_staging",
    "actual_fit",
    "attachment_transitions",
    "destination_selected_directly",
    "far_preview_accepted",
    "far_preview_ready",
    "final_attached",
    "final_object_state",
    "final_reason",
    "final_result",
    "ik_config_bound",
    "ik_config_fingerprint",
    "mode",
    "release_frame",
    "repick_support_is_destination",
    "reverse_start_frame",
    "runtime_preview_only",
    "staged_preview_ready",
    "staged_selection_id_changed",
    "state_sequence",
)

_EXPECTED_STATES = (
    "Carry",
    "PlacePreflight",
    "PlaceAlign",
    "PlaceReplay",
    "PlaceRelease",
    "Locomotion",
)

_STABLE_HOLD_SAMPLES = 5
_MAX_RELATIVE_POSITION_M = 0.020
_MAX_RELATIVE_ANGLE_RADIANS = np.deg2rad(10.0)
_DEFAULT_IK_CONFIG = (
    0.12,
    0.436332313,
    0.04,
    0.261799388,
    0.05,
    0.001,
    0.25,
    0.10,
    8,
)


def _canonical_ik_fingerprint(config=_DEFAULT_IK_CONFIG):
    """Mirror the public canonical hash contract for one exact IK snapshot."""
    payload = bytearray()
    for tag in (0x494B4346, 0x5008):
        payload.append(0xD3)
        payload.extend(struct.pack("<I", tag))
    for value in config[:-1]:
        payload.extend(struct.pack("<f", value))
    payload.extend(struct.pack("<i", config[-1]))

    state = 14695981039346656037
    for value in payload:
        state ^= value
        state = (state * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return state or 0x9E3779B97F4A7C15


def _certified_reverse_frames(labeled):
    """Derive the reverse release/start frames from one labeled clip."""
    motion = labeled.motion
    hand = int(labeled.active_hand)
    relative_positions, relative_rotations = hand_in_object(
        motion.hand_positions[:, hand],
        motion.hand_rotations[:, hand],
        motion.object_positions,
        motion.object_rotations,
    )
    contact = int(
        np.flatnonzero(labeled.phases == InteractionPhase.CONTACT)[0]
    )
    last_start = len(labeled.phases) - _STABLE_HOLD_SAMPLES
    for start in range(labeled.hold_frame, last_start + 1):
        stop = start + _STABLE_HOLD_SAMPLES
        sample = slice(start, stop)
        if not np.all(labeled.phases[sample] == InteractionPhase.HOLD):
            continue
        if not np.all(motion.hand_contacts[sample, hand] == 1):
            continue
        if not np.array_equal(
            np.diff(motion.source_frames[sample]),
            np.ones(_STABLE_HOLD_SAMPLES - 1, dtype=np.int32),
        ):
            continue

        positions = relative_positions[sample]
        pairwise_position = np.linalg.norm(
            positions[:, None] - positions[None, :], axis=-1
        )
        if float(np.max(pairwise_position)) > _MAX_RELATIVE_POSITION_M:
            continue

        rotations = relative_rotations[sample]
        pairwise_angle = max(
            quaternion_angle(rotations[left], rotations[right])
            for left in range(len(rotations))
            for right in range(left + 1, len(rotations))
        )
        if pairwise_angle > _MAX_RELATIVE_ANGLE_RADIANS:
            continue
        return contact, stop - 1
    raise AssertionError("fixture has no certified reverse Hold window")


def _fixture_record(labeled):
    release, reverse_start = _certified_reverse_frames(labeled)
    return {
        "actual_carry_staging": True,
        "actual_fit": True,
        "attachment_transitions": 1,
        "destination_selected_directly": True,
        "far_preview_accepted": True,
        "far_preview_ready": False,
        "final_attached": False,
        "final_object_state": "Free",
        "final_reason": "None",
        "final_result": "Succeeded",
        "ik_config_bound": True,
        "ik_config_fingerprint": _canonical_ik_fingerprint(),
        "mode": "reversed_pickup",
        "release_frame": release,
        "repick_support_is_destination": True,
        "reverse_start_frame": reverse_start,
        "runtime_preview_only": True,
        "staged_preview_ready": True,
        "staged_selection_id_changed": True,
        "state_sequence": list(_EXPECTED_STATES),
    }


def validate_probe_record(encoded, labeled):
    """Validate one deterministic probe JSON line against its source clip."""
    try:
        record = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise AssertionError("probe output must be one JSON record") from error
    if not isinstance(record, dict):
        raise AssertionError("probe output must be a JSON object")
    if tuple(sorted(record)) != _EXPECTED_KEYS:
        raise AssertionError("probe output keys do not match the contract")
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    if encoded != canonical:
        raise AssertionError("probe output must be compact and key-sorted")

    expected = _fixture_record(labeled)
    for key, expected_value in expected.items():
        if record[key] != expected_value:
            raise AssertionError(
                f"unexpected {key}: {record[key]!r} != {expected_value!r}"
            )
    if isinstance(record["ik_config_fingerprint"], bool) or not isinstance(
        record["ik_config_fingerprint"], int
    ):
        raise AssertionError("IK fingerprint must be an integer")
    if record["ik_config_fingerprint"] <= 0:
        raise AssertionError("IK fingerprint must be canonical and nonzero")
    return record


class PlaceProbeFixtureValidatorTests(unittest.TestCase):
    def setUp(self):
        self.labeled = derive_interaction_labels(canonical_pickup_fixture())
        hand = int(self.labeled.active_hand)
        first_hold = self.labeled.hold_frame
        self.labeled.motion.hand_positions[first_hold, hand, 0] += 0.030

    def test_derives_earliest_certified_window_and_ignores_last_contact(self):
        release, reverse_start = _certified_reverse_frames(self.labeled)
        self.assertEqual(release, self.labeled.contact_frame)
        self.assertEqual(
            reverse_start,
            self.labeled.hold_frame + _STABLE_HOLD_SAMPLES,
        )

        mutated = copy.deepcopy(self.labeled)
        hand = int(mutated.active_hand)
        contact_samples = np.flatnonzero(
            mutated.motion.hand_contacts[:, hand] == 1
        )
        last_contact = int(contact_samples[-1])
        original_contacts = mutated.motion.hand_contacts.copy()
        mutated.motion.hand_contacts[last_contact, hand] = 0
        self.assertEqual(
            int(np.count_nonzero(
                original_contacts != mutated.motion.hand_contacts
            )),
            1,
        )
        self.assertEqual(
            _certified_reverse_frames(mutated),
            (release, reverse_start),
        )

        record = _fixture_record(self.labeled)
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        self.assertEqual(validate_probe_record(encoded, mutated), record)
        self.assertEqual(
            record["ik_config_fingerprint"],
            _canonical_ik_fingerprint(_DEFAULT_IK_CONFIG),
        )
        changed_config = list(_DEFAULT_IK_CONFIG)
        changed_config[4] = float(np.nextafter(
            np.float32(changed_config[4]), np.float32(np.inf)
        ))
        self.assertNotEqual(
            record["ik_config_fingerprint"],
            _canonical_ik_fingerprint(tuple(changed_config)),
        )

    def test_rejects_illustrative_or_hard_coded_frame_numbers(self):
        record = _fixture_record(self.labeled)
        record["release_frame"] += 1
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(AssertionError, "release_frame"):
            validate_probe_record(encoded, self.labeled)

    def test_rejects_missing_runtime_owned_placement_evidence(self):
        for key, value in (
            ("actual_carry_staging", False),
            ("destination_selected_directly", False),
            ("far_preview_accepted", False),
            ("far_preview_ready", True),
            ("ik_config_bound", False),
            ("ik_config_fingerprint", 0),
            ("runtime_preview_only", False),
            ("staged_preview_ready", False),
            ("staged_selection_id_changed", False),
        ):
            with self.subTest(key=key):
                record = _fixture_record(self.labeled)
                record[key] = value
                encoded = json.dumps(
                    record, sort_keys=True, separators=(",", ":")
                )
                with self.assertRaises(AssertionError):
                    validate_probe_record(encoded, self.labeled)

    def test_rejects_noncompact_or_unsorted_output(self):
        record = _fixture_record(self.labeled)
        with self.assertRaisesRegex(AssertionError, "compact and key-sorted"):
            validate_probe_record(json.dumps(record), self.labeled)


if __name__ == "__main__":
    unittest.main()
