from dataclasses import replace
import unittest

import numpy as np

from mm_sonic.joints import ContractError
from mm_sonic.torch_supported_step_up import (
    find_complete_step_up_sequence,
    swing_clearance_targets,
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
    def test_allows_leading_foot_to_lift_during_trailing_transfer(self):
        support, surface = _uneven_sequence()
        support[51:, 0] = False

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
        self.assertTrue(result.final_split_height_contact_transfer)
        self.assertAlmostEqual(result.contact_height_pattern_error_m, 0.0)
        self.assertAlmostEqual(result.minimum_sole_clearance_m, -0.015)

    def test_accepts_dynamic_split_height_contact_transfer(self):
        support = self.support.copy()
        support[51:, 0] = False
        ankle = self.ankle_clearance.copy()
        ankle[51:, 0] = 0.50
        sequence = find_complete_step_up_sequence(
            support_mask=support,
            surface_height_m=self.surface,
            start_frame=0,
            first_contact_frame=20,
            landing_foot=0,
            stable_contact_frames=3,
        )

        result = validate_placed_step_up(
            sequence=sequence,
            source_support_mask=support,
            ankle_clearance_m=ankle,
            sole_clearance_m=self.sole_clearance,
            target_final_height_delta_m=0.18,
        )

        self.assertTrue(result.final_split_height_contact_transfer)

    def test_rejects_level_target_for_uneven_source_contacts(self):
        with self.assertRaisesRegex(ContractError, "split height"):
            validate_placed_step_up(
                sequence=self.sequence,
                source_support_mask=self.support,
                ankle_clearance_m=self.ankle_clearance,
                sole_clearance_m=self.sole_clearance,
                target_final_height_delta_m=0.0,
            )

    def test_rejects_level_source_and_target_as_not_uneven(self):
        level_sequence = replace(
            self.sequence, source_final_height_delta_m=0.0
        )

        with self.assertRaisesRegex(ContractError, "split height"):
            validate_placed_step_up(
                sequence=level_sequence,
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


class SwingClearanceTargetTests(unittest.TestCase):
    def test_lifts_only_a_colliding_unsupported_foot(self):
        feet = np.array(((0.0, 0.0, 0.20), (0.2, 0.0, 0.30)))

        mask, targets = swing_clearance_targets(
            foot_position_world=feet,
            support_mask=np.array((True, False)),
            minimum_sole_clearance_m=np.array((-0.015, -0.046)),
            clearance_margin_m=0.005,
        )

        np.testing.assert_array_equal(mask, (False, True))
        np.testing.assert_allclose(targets[0], feet[0])
        self.assertAlmostEqual(targets[1, 2], 0.351)

    def test_does_not_move_a_swing_foot_within_penetration_tolerance(self):
        mask, targets = swing_clearance_targets(
            foot_position_world=np.array(
                ((0.0, 0.0, 0.20), (0.2, 0.0, 0.30))
            ),
            support_mask=np.array((True, False)),
            minimum_sole_clearance_m=np.array((-0.015, -0.015)),
            clearance_margin_m=0.005,
        )

        np.testing.assert_array_equal(mask, (False, False))
        self.assertEqual(targets.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
