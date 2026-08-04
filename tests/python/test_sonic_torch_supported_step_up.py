import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_supported_step_up import (
    find_complete_step_up_sequence,
    validate_placed_step_up,
)


def _uneven_sequence():
    support = np.zeros((70, 2), dtype=bool)
    support[:24, 1] = True
    support[20:24, 0] = True
    support[24:, 0] = True
    support[50:, 1] = True
    surface = np.zeros((70, 2), dtype=np.float64)
    surface[:, 0] = 0.35
    surface[50:, 1] = 0.53
    return support, surface


class CompleteStepUpSequenceTests(unittest.TestCase):
    def test_finds_trailing_contact_on_a_different_level(self):
        support, surface = _uneven_sequence()

        sequence = find_complete_step_up_sequence(
            support_mask=support,
            surface_height_m=surface,
            start_frame=0,
            first_contact_frame=20,
            landing_foot=0,
            stable_contact_frames=3,
        )

        self.assertEqual(sequence.trailing_contact_frame, 50)
        self.assertEqual(sequence.end_exclusive, 53)
        self.assertAlmostEqual(
            sequence.source_final_height_delta_m, 0.18
        )

    def test_rejects_a_clip_cropped_after_only_the_first_contact(self):
        support, surface = _uneven_sequence()

        with self.assertRaisesRegex(ContractError, "trailing contact"):
            find_complete_step_up_sequence(
                support_mask=support[:45],
                surface_height_m=surface[:45],
                start_frame=0,
                first_contact_frame=20,
                landing_foot=0,
                stable_contact_frames=3,
            )

    def test_rejects_an_unsupported_interval(self):
        support, surface = _uneven_sequence()
        support[35] = False

        with self.assertRaisesRegex(ContractError, "unsupported"):
            find_complete_step_up_sequence(
                support_mask=support,
                surface_height_m=surface,
                start_frame=0,
                first_contact_frame=20,
                landing_foot=0,
                stable_contact_frames=3,
            )


class PlacedStepUpValidationTests(unittest.TestCase):
    def setUp(self):
        self.support, self.surface = _uneven_sequence()
        self.sequence = find_complete_step_up_sequence(
            support_mask=self.support,
            surface_height_m=self.surface,
            start_frame=0,
            first_contact_frame=20,
            landing_foot=0,
            stable_contact_frames=3,
        )
        self.ankle_clearance = np.full((70, 2), 0.50)
        self.ankle_clearance[self.support] = 0.0
        self.sole_clearance = np.full((70, 2, 7), 0.10)
        self.sole_clearance[20, 0] = -0.015
        self.sole_clearance[50:53] = -0.015

    def test_accepts_zero_flight_split_height_support(self):
        result = validate_placed_step_up(
            sequence=self.sequence,
            source_support_mask=self.support,
            ankle_clearance_m=self.ankle_clearance,
            sole_clearance_m=self.sole_clearance,
            target_final_height_delta_m=0.18,
        )

        self.assertEqual(result.unsupported_frame_count, 0)
        self.assertTrue(result.final_split_height_double_support)
        self.assertAlmostEqual(result.contact_height_pattern_error_m, 0.0)
        self.assertAlmostEqual(result.minimum_sole_clearance_m, -0.015)

    def test_rejects_level_target_for_uneven_source_contacts(self):
        with self.assertRaisesRegex(ContractError, "height pattern"):
            validate_placed_step_up(
                sequence=self.sequence,
                source_support_mask=self.support,
                ankle_clearance_m=self.ankle_clearance,
                sole_clearance_m=self.sole_clearance,
                target_final_height_delta_m=0.0,
            )

    def test_rejects_lost_stance_contact(self):
        self.ankle_clearance[35, 0] = 0.10

        with self.assertRaisesRegex(ContractError, "unsupported"):
            validate_placed_step_up(
                sequence=self.sequence,
                source_support_mask=self.support,
                ankle_clearance_m=self.ankle_clearance,
                sole_clearance_m=self.sole_clearance,
                target_final_height_delta_m=0.18,
            )

    def test_rejects_excessive_sole_penetration(self):
        self.sole_clearance[51, 1, 3] = -0.030

        with self.assertRaisesRegex(ContractError, "penetration"):
            validate_placed_step_up(
                sequence=self.sequence,
                source_support_mask=self.support,
                ankle_clearance_m=self.ankle_clearance,
                sole_clearance_m=self.sole_clearance,
                target_final_height_delta_m=0.18,
            )


if __name__ == "__main__":
    unittest.main()
