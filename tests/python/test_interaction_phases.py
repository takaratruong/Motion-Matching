import copy
import unittest

import numpy as np

from resources import quat as holden_quat
from resources.g1_interaction_builder.conversion import (
    finite_difference_quaternions,
    finite_difference_vectors,
)
from resources.g1_interaction_builder.phases import (
    derive_interaction_labels,
    quaternion_angle,
)
from resources.g1_interaction_builder.schema import (
    InteractionHand,
    InteractionPhase,
    InteractionValidationError,
)
from tests.python.interaction_fixture import (
    canonical_pickup_fixture,
    transform_fixture_world,
)


GRASP_POSITION_OBJECT = np.array([0.02, 0.00, -0.03])


def follow_object_with_active_hand(clip, start: int = 38) -> None:
    hand = int(InteractionHand.RIGHT)
    clip.hand_positions[start:, hand] = (
        clip.object_positions[start:]
        + holden_quat.mul_vec(
            clip.object_rotations[start:], GRASP_POSITION_OBJECT
        )
    )
    clip.hand_rotations[start:, hand] = clip.object_rotations[start:]


class InteractionPhaseTests(unittest.TestCase):
    def assert_validation_code(self, expected: str, clip) -> None:
        with self.assertRaises(InteractionValidationError) as caught:
            derive_interaction_labels(clip)
        self.assertEqual(caught.exception.code, expected)
        self.assertIn(clip.sequence_id, str(caught.exception))

    def test_fixture_is_canonical_25_hz_with_containing_frame_derivatives(
        self,
    ):
        clip = canonical_pickup_fixture()

        self.assertEqual(clip.fps, 25.0)
        self.assertEqual(len(clip.positions), 75)
        np.testing.assert_array_equal(clip.source_frames, np.arange(75))
        np.testing.assert_allclose(clip.object_positions[:38, 1], 0.75)
        self.assertAlmostEqual(float(clip.object_positions[49, 1]), 0.90)
        np.testing.assert_allclose(clip.object_positions[49:, 1], 0.90)
        np.testing.assert_array_equal(clip.hand_contacts[:38], 0)
        np.testing.assert_array_equal(clip.hand_contacts[38:, 0], 0)
        np.testing.assert_array_equal(clip.hand_contacts[38:, 1], 1)
        np.testing.assert_array_equal(clip.hand_dof_velocities, 0.0)

        expected = finite_difference_quaternions(
            clip.object_rotations, clip.fps
        )
        np.testing.assert_allclose(
            clip.object_angular_velocities, expected, atol=1e-6
        )
        self.assertGreater(
            float(np.max(np.abs(clip.object_angular_velocities))), 0.0
        )

    def test_derives_one_active_hand_and_monotonic_phases(self):
        labeled = derive_interaction_labels(canonical_pickup_fixture())

        self.assertEqual(labeled.active_hand, InteractionHand.RIGHT)
        self.assertEqual(labeled.contact_frame, 38)
        self.assertEqual(labeled.phases.dtype, np.dtype(np.uint8))
        self.assertEqual(labeled.phases[12], InteractionPhase.APPROACH)
        self.assertEqual(labeled.phases[13], InteractionPhase.REACH)
        self.assertEqual(labeled.phases[38], InteractionPhase.CONTACT)
        self.assertGreaterEqual(labeled.lift_frame, 38)
        self.assertGreater(labeled.hold_frame, labeled.lift_frame)
        self.assertTrue(
            np.all(np.diff(labeled.phases.astype(np.int16)) >= 0)
        )
        self.assertAlmostEqual(labeled.time_to_contact[13], 1.0, places=5)
        self.assertAlmostEqual(labeled.time_to_contact[38], 0.0, places=5)

        self.assertTrue(
            np.all(labeled.phases[:13] == InteractionPhase.APPROACH)
        )
        self.assertTrue(
            np.all(
                labeled.phases[13:38] == InteractionPhase.REACH
            )
        )
        self.assertTrue(
            np.all(
                labeled.phases[38:labeled.lift_frame]
                == InteractionPhase.CONTACT
            )
        )
        self.assertTrue(
            np.all(
                labeled.phases[
                    labeled.lift_frame:labeled.hold_frame
                ]
                == InteractionPhase.LIFT
            )
        )
        self.assertTrue(
            np.all(
                labeled.phases[labeled.hold_frame:]
                == InteractionPhase.HOLD
            )
        )

        motion = labeled.motion
        self.assertGreaterEqual(
            motion.object_positions[labeled.lift_frame, 1],
            labeled.support_height + 0.05,
        )
        self.assertLess(
            motion.object_positions[labeled.lift_frame - 1, 1],
            labeled.support_height + 0.05,
        )
        speed = np.abs(motion.object_velocities[:, 1])
        self.assertTrue(
            np.all(speed[labeled.hold_frame:labeled.hold_frame + 5] < 0.1)
        )
        self.assertFalse(
            np.all(
                speed[labeled.hold_frame - 1:labeled.hold_frame + 4] < 0.1
            )
        )

    def test_grasp_is_object_local_and_world_transform_invariant(self):
        a = derive_interaction_labels(canonical_pickup_fixture())
        b = derive_interaction_labels(
            transform_fixture_world(
                canonical_pickup_fixture(),
                translation=[3.0, 0.0, -2.0],
                yaw_degrees=73.0,
            )
        )

        np.testing.assert_allclose(
            a.grasp_position_object, GRASP_POSITION_OBJECT, atol=1e-5
        )
        np.testing.assert_allclose(
            a.grasp_position_object, b.grasp_position_object, atol=1e-5
        )
        self.assertLess(
            quaternion_angle(
                a.grasp_rotation_object, b.grasp_rotation_object
            ),
            1e-5,
        )

    def test_approach_is_horizontal_object_local_and_world_invariant(self):
        a = derive_interaction_labels(canonical_pickup_fixture())
        b = derive_interaction_labels(
            transform_fixture_world(
                canonical_pickup_fixture(),
                translation=[-4.0, 1.0, 2.0],
                yaw_degrees=121.0,
            )
        )

        self.assertAlmostEqual(a.approach_direction_object[1], 0.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(a.approach_direction_object)),
            1.0,
            places=6,
        )
        np.testing.assert_allclose(
            a.approach_direction_object,
            b.approach_direction_object,
            atol=1e-5,
        )

    def test_grasp_uses_first_point_two_seconds_and_aligns_quaternion_signs(
        self,
    ):
        clip = canonical_pickup_fixture()
        clip.hand_rotations[39, 1] *= -1.0
        later_grasp = np.array([0.20, 0.10, -0.15])
        clip.hand_positions[43:, 1] = (
            clip.object_positions[43:]
            + holden_quat.mul_vec(
                clip.object_rotations[43:], later_grasp
            )
        )

        labeled = derive_interaction_labels(clip)

        np.testing.assert_allclose(
            labeled.grasp_position_object,
            GRASP_POSITION_OBJECT,
            atol=1e-5,
        )
        self.assertLess(
            quaternion_angle(
                labeled.grasp_rotation_object,
                np.array([1.0, 0.0, 0.0, 0.0]),
            ),
            1e-5,
        )

    def test_rejects_two_hands_and_unstable_contact(self):
        both = canonical_pickup_fixture()
        both.hand_contacts[38:, :] = 1
        both.hand_positions[:, 0] = both.hand_positions[:, 1]
        both.hand_rotations[:, 0] = both.hand_rotations[:, 1]
        self.assert_validation_code("ambiguous_active_hand", both)

        unstable = canonical_pickup_fixture()
        unstable.hand_positions[38::2, 1, 0] += 0.05
        self.assert_validation_code("no_stable_contact", unstable)

    def test_rejects_rotationally_unstable_contact(self):
        clip = canonical_pickup_fixture()
        angle = np.deg2rad(11.0)
        turn = np.array(
            [np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0]
        )
        clip.hand_rotations[38::2, 1] = holden_quat.normalize(
            holden_quat.mul(clip.object_rotations[38::2], turn)
        )

        self.assert_validation_code("no_stable_contact", clip)

    def test_three_duplicate_frames_do_not_count_as_source_samples(self):
        clip = canonical_pickup_fixture()
        clip.hand_contacts[:, 1] = 0
        clip.hand_contacts[38:41, 1] = 1
        clip.source_frames[39:41] = clip.source_frames[38]

        self.assert_validation_code("no_stable_contact", clip)

    def test_rejects_contact_loss_before_hold(self):
        clip = canonical_pickup_fixture()
        clip.hand_contacts[45, 1] = 0

        self.assert_validation_code("contact_lost_before_hold", clip)

    def test_rejects_pickup_without_five_centimeter_lift(self):
        clip = canonical_pickup_fixture()
        clip.object_positions[:, 1] = 0.75
        clip.object_velocities = finite_difference_vectors(
            clip.object_positions, clip.fps
        )
        follow_object_with_active_hand(clip)

        self.assert_validation_code("no_five_centimeter_lift", clip)

    def test_rejects_lift_without_continuous_stable_hold(self):
        clip = canonical_pickup_fixture()
        clip.object_positions[37:, 1] = 0.75 + 0.005 * np.arange(
            len(clip.object_positions) - 37
        )
        clip.object_velocities = finite_difference_vectors(
            clip.object_positions, clip.fps
        )
        follow_object_with_active_hand(clip)

        self.assert_validation_code("no_stable_hold", clip)

    def test_rejects_near_zero_horizontal_approach(self):
        clip = canonical_pickup_fixture()
        clip.hand_positions[13, 1] = clip.hand_positions[38, 1]

        self.assert_validation_code("invalid_approach", clip)

    def test_input_is_not_mutated(self):
        clip = canonical_pickup_fixture()
        original = copy.deepcopy(clip)

        derive_interaction_labels(clip)

        for name in (
            "hand_positions",
            "hand_rotations",
            "object_positions",
            "object_rotations",
            "hand_contacts",
            "source_frames",
        ):
            np.testing.assert_array_equal(
                getattr(clip, name), getattr(original, name)
            )


if __name__ == "__main__":
    unittest.main()
