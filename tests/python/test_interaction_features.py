import copy
import dataclasses
import unittest
import warnings

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from resources import quat as holden_quat
from resources.g1_interaction_builder import features as feature_module
from resources.g1_interaction_builder.conversion import (
    finite_difference_vectors,
)
from resources.g1_interaction_builder.features import (
    FEATURE_GROUPS,
    build_features,
    normalize_feature_groups,
)
from resources.g1_interaction_builder.phases import derive_interaction_labels
from resources.g1_interaction_builder.schema import (
    EvaluationSplit,
    FeatureGroup,
    G1_SKELETON,
    InteractionValidationError,
)
from resources.g1_interaction_builder.splits import (
    partition_clips,
    split_objects,
)
from resources.g1_terrain_builder.schema import SkeletonSpec
from tests.python.interaction_fixture import (
    canonical_pickup_fixture,
    labeled_clips_for_objects,
    transform_fixture_world,
)


EXPECTED_GROUPS = (
    ("pose", 0, 33),
    ("trajectory", 33, 45),
    ("grasp", 45, 57),
    ("root_target", 57, 65),
    ("context", 65, 71),
)


def denormalized(feature_set) -> np.ndarray:
    return (
        feature_set.values.astype(np.float64)
        * feature_set.scales.astype(np.float64)
        + feature_set.offsets.astype(np.float64)
    )


def wxyz(rotation: ScipyRotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def controlled_labeled_clip():
    labeled = derive_interaction_labels(canonical_pickup_fixture())
    clip = copy.deepcopy(labeled.motion)
    frames = len(clip.positions)
    frame = np.arange(frames, dtype=np.float32)

    clip.positions.fill(0.0)
    clip.positions[:, 0, 0] = frame
    clip.positions[:, 0, 1] = 1.0
    clip.positions[:, 0, 2] = 2.0 * frame
    clip.velocities = finite_difference_vectors(clip.positions, clip.fps)
    clip.rotations = holden_quat.eye((frames, 31), dtype=np.float32)
    clip.angular_velocities.fill(0.0)
    clip.hand_positions[:] = clip.positions[:, None, 0]
    clip.hand_rotations = holden_quat.eye((frames, 2), dtype=np.float32)
    clip.object_positions[:] = [10.0, 3.0, -5.0]
    clip.object_rotations = holden_quat.eye((frames,), dtype=np.float32)
    clip.object_velocities.fill(0.0)
    clip.object_angular_velocities.fill(0.0)
    clip.object_dimensions[:] = [0.1, 0.2, 0.3]

    return dataclasses.replace(
        labeled,
        motion=clip,
        grasp_position_object=np.array([1.0, 2.0, 3.0], np.float32),
        grasp_rotation_object=np.array(
            [1.0, 0.0, 0.0, 0.0], np.float32
        ),
        approach_direction_object=np.array(
            [0.6, 0.0, 0.8], np.float32
        ),
    )


def truncate_with_root_path(labeled, frames: int, start: float):
    out = copy.deepcopy(labeled)
    source_frames = len(out.motion.positions)
    for field in dataclasses.fields(out.motion):
        value = getattr(out.motion, field.name)
        if (
            isinstance(value, np.ndarray)
            and value.ndim
            and value.shape[0] == source_frames
        ):
            setattr(out.motion, field.name, value[:frames].copy())
    out.phases = out.phases[:frames].copy()
    out.time_to_contact = out.time_to_contact[:frames].copy()
    frame = np.arange(frames, dtype=np.float32)
    out.motion.positions[:, 0, 0] = start + frame
    out.motion.positions[:, 0, 2] = 2.0 * frame
    out.motion.velocities = finite_difference_vectors(
        out.motion.positions, out.motion.fps
    )
    return out


class InteractionFeatureTests(unittest.TestCase):
    def test_feature_layout_is_exactly_seventy_one(self):
        feature_set = build_features(
            [derive_interaction_labels(canonical_pickup_fixture())],
            G1_SKELETON,
        )

        self.assertEqual(feature_set.values.shape, (75, 71))
        self.assertEqual(
            tuple((group.name, group.start, group.stop)
                  for group in feature_set.groups),
            EXPECTED_GROUPS,
        )
        self.assertEqual(
            tuple((group.name, group.start, group.stop)
                  for group in FEATURE_GROUPS),
            EXPECTED_GROUPS,
        )
        self.assertEqual(feature_set.values.dtype, np.float32)
        self.assertEqual(feature_set.offsets.dtype, np.float32)
        self.assertEqual(feature_set.scales.dtype, np.float32)
        self.assertTrue(np.isfinite(feature_set.values).all())
        self.assertTrue(np.isfinite(feature_set.offsets).all())
        self.assertTrue(np.all(feature_set.scales > 0.0))

    def test_every_feature_slot_matches_an_independent_identity_frame_oracle(
        self,
    ):
        labeled = controlled_labeled_clip()
        raw = denormalized(build_features([labeled], G1_SKELETON))

        root_velocity = [25.0, 0.0, 50.0]
        expected = np.array(
            [
                *([0.0, 0.0, 0.0] * 5),
                *(root_velocity * 5),
                25.0,
                50.0,
                0.0,
                8.0,
                16.0,
                17.0,
                34.0,
                25.0,
                50.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                -11.0,
                -4.0,
                2.0,
                0.0,
                0.0,
                0.0,
                *root_velocity,
                0.0,
                0.0,
                0.0,
                -11.0,
                -4.0,
                2.0,
                0.0,
                1.0,
                *root_velocity,
                4.275,
                0.6,
                0.8,
                0.1,
                0.2,
                0.3,
            ],
            np.float64,
        )

        self.assertEqual(len(expected), 71)
        np.testing.assert_allclose(raw[0], expected, atol=2e-5, rtol=0.0)
        np.testing.assert_allclose(raw[0, :33], expected[:33], atol=2e-5)
        np.testing.assert_allclose(raw[0, 33:45], expected[33:45], atol=2e-5)
        np.testing.assert_allclose(raw[0, 45:57], expected[45:57], atol=2e-5)
        np.testing.assert_allclose(raw[0, 57:65], expected[57:65], atol=2e-5)
        np.testing.assert_allclose(raw[0, 65:71], expected[65:71], atol=2e-5)

    def test_world_grasp_composes_nonidentity_object_and_hand_in_object(self):
        labeled = controlled_labeled_clip()
        object_rotation = ScipyRotation.from_euler(
            "xyz", [25.0, 40.0, -15.0], degrees=True
        )
        grasp_object_rotation = ScipyRotation.from_euler(
            "xyz", [20.0, -30.0, 10.0], degrees=True
        )
        labeled.motion.object_rotations[:] = wxyz(object_rotation)
        labeled.grasp_rotation_object = wxyz(
            grasp_object_rotation
        ).astype(np.float32)
        labeled.grasp_position_object = np.array(
            [0.5, -0.25, 0.75], np.float32
        )
        object_velocity = np.array([0.7, -0.2, 0.4], np.float32)
        object_angular_velocity = np.array(
            [0.3, -0.4, 0.2], np.float32
        )
        labeled.motion.object_velocities[:] = object_velocity
        labeled.motion.object_angular_velocities[:] = (
            object_angular_velocity
        )

        raw = denormalized(build_features([labeled], G1_SKELETON))

        object_position = np.array([10.0, 3.0, -5.0])
        hand_position = np.array([0.0, 1.0, 0.0])
        hand_velocity = np.array([25.0, 0.0, 50.0])
        world_grasp_position = object_position + object_rotation.apply(
            labeled.grasp_position_object
        )
        world_grasp_rotation = object_rotation * grasp_object_rotation
        inverse_grasp = world_grasp_rotation.inv()
        expected_hand_position = inverse_grasp.apply(
            hand_position - world_grasp_position
        )
        expected_orientation = (
            inverse_grasp * ScipyRotation.identity()
        ).as_rotvec()
        grasp_offset_world = object_rotation.apply(
            labeled.grasp_position_object
        )
        grasp_velocity = object_velocity + np.cross(
            object_angular_velocity, grasp_offset_world
        )
        expected_velocity = inverse_grasp.apply(
            hand_velocity - grasp_velocity
        )
        expected_angular_velocity = inverse_grasp.apply(
            -object_angular_velocity
        )
        expected_facing = inverse_grasp.apply([0.0, 0.0, 1.0])

        np.testing.assert_allclose(
            raw[0, 45:48], expected_hand_position, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[0, 48:51], expected_orientation, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[0, 51:54], expected_velocity, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[0, 54:57],
            expected_angular_velocity,
            atol=2e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            raw[0, 57:60], expected_hand_position, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[0, 60:62], expected_facing[[0, 2]], atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[0, 62:65], expected_velocity, atol=2e-5, rtol=0.0
        )
        expected_height = world_grasp_position[1] - (
            labeled.motion.table_position[1]
            + 0.5 * labeled.motion.table_size[1]
        )
        self.assertAlmostEqual(raw[0, 65], expected_height, places=5)

    def test_hand_angular_velocity_comes_from_canonical_world_fk(self):
        labeled = controlled_labeled_clip()
        yaw = ScipyRotation.from_euler("y", 90.0, degrees=True)
        labeled.motion.rotations[:, 0] = wxyz(yaw)
        root_angular = np.array([0.1, 0.2, 0.3], np.float32)
        wrist_local_angular = np.array([0.4, -0.1, 0.2], np.float32)
        labeled.motion.angular_velocities[:, 0] = root_angular
        labeled.motion.angular_velocities[:, 30] = wrist_local_angular

        # These world hand rotations are deliberately unrelated. Recomputing a
        # derivative from them would not recover the canonical FK velocity.
        alternating = ScipyRotation.from_euler(
            "z",
            (np.arange(len(labeled.motion.positions)) * 37.0)[:, None],
            degrees=True,
        ).as_quat()
        labeled.motion.hand_rotations[:, 1] = alternating[:, [3, 0, 1, 2]]

        raw = denormalized(build_features([labeled], G1_SKELETON))
        expected_world = root_angular + yaw.apply(wrist_local_angular)

        np.testing.assert_allclose(
            raw[:, 54:57],
            np.broadcast_to(expected_world, (len(raw), 3)),
            atol=2e-5,
            rtol=0.0,
        )

    def test_common_world_translation_and_yaw_leave_features_unchanged(self):
        clip = canonical_pickup_fixture()
        a = build_features(
            [derive_interaction_labels(clip)], G1_SKELETON
        )
        moved = transform_fixture_world(clip, [4.0, 0.0, -3.0], 121.0)
        b = build_features(
            [derive_interaction_labels(moved)], G1_SKELETON
        )

        np.testing.assert_allclose(a.values, b.values, atol=2e-4, rtol=0.0)
        np.testing.assert_allclose(a.offsets, b.offsets, atol=2e-4, rtol=0.0)
        np.testing.assert_allclose(a.scales, b.scales, atol=2e-4, rtol=0.0)
        np.testing.assert_allclose(
            denormalized(a), denormalized(b), atol=2e-4, rtol=0.0
        )

    def test_future_horizons_clamp_within_each_clip(self):
        base = controlled_labeled_clip()
        first = truncate_with_root_path(base, frames=10, start=0.0)
        second = truncate_with_root_path(base, frames=30, start=100.0)
        turn = ScipyRotation.from_euler("y", 90.0, degrees=True)
        second.motion.rotations[:, 0] = wxyz(turn)

        raw = denormalized(
            build_features([first, second], G1_SKELETON)
        )

        np.testing.assert_allclose(
            raw[5, 33:39], [4.0, 8.0] * 3, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[5, 39:45], [0.0, 1.0] * 3, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(raw[9, 33:39], 0.0, atol=2e-5)
        np.testing.assert_allclose(
            raw[9, 39:45], [0.0, 1.0] * 3, atol=2e-5, rtol=0.0
        )
        np.testing.assert_allclose(
            raw[10, 39:45], [0.0, 1.0] * 3, atol=2e-5, rtol=0.0
        )

    def test_quaternion_sign_equivalent_clips_have_identical_features(self):
        positive_motion = canonical_pickup_fixture()
        frame = 10
        hand = 1
        wrist = 30
        perturbation = wxyz(
            ScipyRotation.from_rotvec([0.001, -0.002, 0.0015])
        )
        hand_rotation = holden_quat.normalize(
            holden_quat.mul(
                positive_motion.object_rotations[frame], perturbation
            )
        )
        positive_motion.rotations[frame, wrist] = hand_rotation
        positive_motion.hand_rotations[frame, hand] = hand_rotation
        positive_motion.validate()

        negative_motion = copy.deepcopy(positive_motion)
        negative_motion.rotations[frame, wrist] *= -1.0
        negative_motion.hand_rotations[frame, hand] *= -1.0
        negative_motion.validate()

        positive = derive_interaction_labels(positive_motion)
        negative = derive_interaction_labels(negative_motion)
        negative_world_rotations = holden_quat.fk(
            negative.motion.rotations,
            negative.motion.positions,
            G1_SKELETON.parents,
        )[0]
        negative_grasp = holden_quat.mul(
            negative.motion.object_rotations[frame],
            negative.grasp_rotation_object,
        )
        negative_relative = holden_quat.mul(
            holden_quat.inv(negative_grasp),
            negative_world_rotations[frame, wrist],
        )
        self.assertLess(negative_relative[0], -0.999)

        positive_raw = feature_module._raw_clip_features(positive)
        negative_raw = feature_module._raw_clip_features(negative)
        positive_features = build_features([positive], G1_SKELETON)
        negative_features = build_features([negative], G1_SKELETON)

        np.testing.assert_allclose(
            positive_raw[:, 48:51],
            negative_raw[:, 48:51],
            atol=2e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            positive_features.offsets,
            negative_features.offsets,
            atol=2e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            positive_features.scales,
            negative_features.scales,
            atol=2e-5,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            positive_features.values,
            negative_features.values,
            atol=2e-5,
            rtol=0.0,
        )

    def test_discrete_labels_and_source_identity_are_not_continuous_features(
        self,
    ):
        original = controlled_labeled_clip()
        changed = copy.deepcopy(original)
        changed.motion.sequence_id = "entirely_different_source"
        changed.motion.object_id = "entirely_different_object"
        changed.motion.foot_contacts[:] = 1 - changed.motion.foot_contacts
        changed.motion.hand_contacts[:] = 1 - changed.motion.hand_contacts
        changed.motion.hand_dof[:] = 42.0
        changed.motion.hand_dof_velocities[:] = -17.0
        changed.motion.source_frames += 500
        changed.phases[:] = changed.phases[::-1]
        changed.time_to_contact[:] = 123.0
        changed.contact_frame = 1
        changed.lift_frame = 2
        changed.hold_frame = 3

        a = build_features([original], G1_SKELETON)
        b = build_features([changed], G1_SKELETON)

        np.testing.assert_array_equal(a.values, b.values)
        np.testing.assert_array_equal(a.offsets, b.offsets)
        np.testing.assert_array_equal(a.scales, b.scales)

    def test_group_normalization_uses_mean_component_std_and_floor(self):
        raw = np.array(
            [[1.0, 2.0, 5.0, 5.0], [3.0, 6.0, 5.0, 5.0]],
            np.float64,
        )
        groups = (
            FeatureGroup("varying", 0, 2),
            FeatureGroup("constant", 2, 4),
        )

        normalized = normalize_feature_groups(raw, groups)

        np.testing.assert_allclose(normalized.offsets, [2.0, 4.0, 5.0, 5.0])
        np.testing.assert_allclose(
            normalized.scales, [1.5, 1.5, 1e-5, 1e-5]
        )
        np.testing.assert_allclose(
            normalized.values,
            [[-2.0 / 3.0, -4.0 / 3.0, 0.0, 0.0],
             [2.0 / 3.0, 4.0 / 3.0, 0.0, 0.0]],
        )
        self.assertEqual(normalized.values.dtype, np.float32)
        self.assertEqual(normalized.offsets.dtype, np.float32)
        self.assertEqual(normalized.scales.dtype, np.float32)
        self.assertEqual(normalized.groups, groups)

    def test_group_normalization_rejects_non_finite_input(self):
        raw = np.zeros((2, 3), np.float32)
        raw[1, 1] = np.nan

        with self.assertRaisesRegex(
            InteractionValidationError,
            "non_finite_features: group pose contains non-finite values",
        ):
            normalize_feature_groups(
                raw, (FeatureGroup("pose", 0, 3),)
            )

    def test_group_normalization_requires_an_exact_named_column_partition(
        self,
    ):
        raw = np.zeros((2, 3), np.float32)
        cases = (
            ((), "at least one feature group"),
            (
                (FeatureGroup("empty", 0, 0),
                 FeatureGroup("rest", 0, 3)),
                "empty range.*empty.*\\[0, 0\\)",
            ),
            (
                (FeatureGroup("negative", -1, 1),
                 FeatureGroup("rest", 1, 3)),
                "negative range.*negative.*\\[-1, 1\\)",
            ),
            (
                (FeatureGroup("reversed", 2, 1),
                 FeatureGroup("rest", 1, 3)),
                "reversed range.*reversed.*\\[2, 1\\)",
            ),
            (
                (FeatureGroup("outside", 0, 4),),
                "out-of-range.*outside.*\\[0, 4\\).*3 columns",
            ),
            (
                (FeatureGroup("first", 0, 2),
                 FeatureGroup("overlap", 1, 3)),
                "overlap.*overlap.*starts at 1.*expected 2",
            ),
            (
                (FeatureGroup("first", 0, 1),
                 FeatureGroup("gap", 2, 3)),
                "gap.*gap.*starts at 2.*expected 1",
            ),
            (
                (FeatureGroup("leading_gap", 1, 3),),
                "gap.*leading_gap.*starts at 1.*expected 0",
            ),
            (
                (FeatureGroup("trailing_gap", 0, 2),),
                "trailing gap.*column 2.*3 columns",
            ),
            (
                (FeatureGroup("same", 0, 1),
                 FeatureGroup("same", 1, 3)),
                "duplicate feature group name.*same",
            ),
        )

        for groups, message in cases:
            with self.subTest(groups=groups):
                with self.assertRaisesRegex(
                    InteractionValidationError, message
                ):
                    normalize_feature_groups(raw, groups)

    def test_identity_orientation_error_emits_no_runtime_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_features([controlled_labeled_clip()], G1_SKELETON)

        self.assertEqual(caught, [])

    def test_build_rejects_empty_database_and_wrong_skeleton(self):
        with self.assertRaises(InteractionValidationError) as empty:
            build_features([], G1_SKELETON)
        self.assertEqual(empty.exception.code, "empty_database")

        parents = G1_SKELETON.parents.copy()
        parents[1] = -1
        wrong = SkeletonSpec(G1_SKELETON.names, parents)
        with self.assertRaises(InteractionValidationError) as mismatch:
            build_features([controlled_labeled_clip()], wrong)
        self.assertEqual(mismatch.exception.code, "skeleton_mismatch")

    def test_database_feature_boundary_preserves_exact_clip_and_row_order(self):
        clips = labeled_clips_for_objects(
            ["heldout", "db_a", "db_a", "db_b", "heldout"]
        )
        for index, clip in enumerate(clips):
            clip.motion.object_dimensions[:] = [
                0.1 + index,
                0.2 + index,
                0.3 + index,
            ]
        split = EvaluationSplit(
            seed=7,
            database_objects=("db_a", "db_b"),
            heldout_objects=("heldout",),
        )

        database, heldout = partition_clips(clips, split)
        features = feature_module.build_database_features(
            clips, split, G1_SKELETON
        )
        expected = build_features(
            [clips[1], clips[2], clips[3]], G1_SKELETON
        )

        self.assertEqual(
            [clip.motion.sequence_id for clip in database],
            [clips[1].motion.sequence_id,
             clips[2].motion.sequence_id,
             clips[3].motion.sequence_id],
        )
        self.assertEqual(
            [clip.motion.sequence_id for clip in heldout],
            [clips[0].motion.sequence_id, clips[4].motion.sequence_id],
        )
        self.assertEqual(features.values.shape[0], 3 * 75)
        np.testing.assert_array_equal(features.values, expected.values)
        np.testing.assert_array_equal(features.offsets, expected.offsets)
        np.testing.assert_array_equal(features.scales, expected.scales)

    def test_heldout_object_cannot_change_database_normalization(self):
        clips = labeled_clips_for_objects(["a", "b", "c", "d"])
        split = split_objects(clips, heldout_count=1, seed=7)
        before = feature_module.build_database_features(
            clips, split, G1_SKELETON
        )

        heldout = next(
            clip for clip in clips
            if clip.motion.object_id in split.heldout_objects
        )
        heldout.motion.object_dimensions[:] = [900.0, 800.0, 700.0]
        after = feature_module.build_database_features(
            clips, split, G1_SKELETON
        )
        contaminated = build_features(clips, G1_SKELETON)

        np.testing.assert_array_equal(before.offsets, after.offsets)
        np.testing.assert_array_equal(before.scales, after.scales)
        np.testing.assert_array_equal(before.values, after.values)
        self.assertFalse(
            np.allclose(before.offsets[68:71], contaminated.offsets[68:71])
        )

    def test_database_feature_boundary_rejects_heldout_only_input(self):
        clips = labeled_clips_for_objects(["heldout", "database"])
        split = EvaluationSplit(
            seed=7,
            database_objects=("database",),
            heldout_objects=("heldout",),
        )

        with self.assertRaisesRegex(
            ValueError,
            "unknown split object IDs not present in clips.*database",
        ):
            feature_module.build_database_features(
                [clips[0]], split, G1_SKELETON
            )

    def test_database_feature_boundary_rejects_malformed_split(self):
        clips = labeled_clips_for_objects(["database", "heldout"])
        malformed = EvaluationSplit(
            seed=7,
            database_objects=("database", "database"),
            heldout_objects=("heldout",),
        )

        with self.assertRaisesRegex(
            ValueError, "duplicate database object IDs.*database"
        ):
            feature_module.build_database_features(
                clips, malformed, G1_SKELETON
            )


if __name__ == "__main__":
    unittest.main()
