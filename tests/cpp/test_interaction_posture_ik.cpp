#include "interaction_posture_ik.h"

#include "g1_arm_joint_metadata.h"
#include "g1_posture_ik_fixture.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace {

using namespace g1_posture_fixture;

void test_public_contract_and_angle_round_trip() {
    interaction::Pose pose = make_pose();
    const interaction::LeftUpperBodyAngles expected{
        0.22F, -0.17F, 0.11F,
        0.10F, 0.35F, -0.20F, 0.80F, 0.10F, -0.15F, 0.05F};
    interaction::apply_left_upper_body(pose, expected);
    const auto actual = interaction::decompose_left_upper_body(pose);
    for (size_t joint = 0U; joint < expected.size(); ++joint) {
        assert(near(actual[joint], expected[joint]));
    }
    static_assert(interaction::kLeftUpperBodyJointCount == 10U);
    assert(interaction::kWaist[0].bone == g1_skeleton::Spine);
    assert(interaction::kWaist[1].bone == g1_skeleton::Spine1);
    assert(interaction::kWaist[2].bone == g1_skeleton::Spine2);

    const interaction::PostureIKConfig config{};
    assert(config.accepted_position_m == 0.001F);
    assert(config.accepted_orientation_radians == 0.261799388F);
    assert(config.maximum_iterations == 30);
}

void test_zero_target_returns_source_byte_exact() {
    const interaction::Pose source = make_pose();
    interaction::Pose solved = source;
    const auto result = interaction::solve_left_hand_posture_ik(
        solved,
        left_hand_transform(source),
        source,
        interaction::decompose_left_upper_body(source));

    assert(result.accepted);
    assert(result.reason == interaction::Reason::None);
    assert(result.position_error_m <= kTolerance);
    assert(result.orientation_error_radians <= kTolerance);
    assert(same_pose(source, solved));
}

void test_right_zero_target_returns_source_byte_exact() {
    const interaction::Pose source = make_pose();
    const interaction::WorldPose world = interaction::world_pose(source);
    const interaction::Transform target{
        world.positions[g1_skeleton::RightWrist],
        world.rotations[g1_skeleton::RightWrist],
    };
    interaction::Pose solved = source;
    const interaction::UpperBodyAngles seed =
        interaction::decompose_upper_body(
            source, interaction::Hand::Right);
    const auto result = interaction::solve_hand_posture_ik(
        solved,
        interaction::Hand::Right,
        target,
        source,
        seed);

    assert(result.accepted);
    assert(result.reason == interaction::Reason::None);
    assert(result.position_error_m <= kTolerance);
    assert(result.orientation_error_radians <= kTolerance);
    assert(same_pose(source, solved));
}

void test_reachable_ten_joint_target_converges() {
    const interaction::Pose source = make_pose();
    interaction::Pose goal = source;
    auto goal_angles = interaction::decompose_left_upper_body(goal);
    const interaction::LeftUpperBodyAngles delta{
        0.06F, -0.04F, 0.03F, -0.10F, 0.08F,
        0.05F, 0.12F, -0.06F, 0.05F, -0.04F};
    for (size_t joint = 0U; joint < goal_angles.size(); ++joint) {
        goal_angles[joint] += delta[joint];
    }
    interaction::apply_left_upper_body(goal, goal_angles);

    interaction::Pose solved = source;
    interaction::PostureIKConfig config{};
    config.source_scale_m_per_radian.fill(0.0F);
    config.temporal_scale_m_per_radian.fill(0.0F);
    config.elbow_pole_scale_m = 0.0F;
    const auto result = interaction::solve_left_hand_posture_ik(
        solved,
        left_hand_transform(goal),
        source,
        interaction::decompose_left_upper_body(source),
        config);

    assert(result.accepted);
    assert(result.position_error_m <= config.accepted_position_m);
    assert(result.orientation_error_radians <=
           config.accepted_orientation_radians);
    assert_finite_bounded(result.joint_angles);
    assert_non_owned_local_channels_equal(source, solved);
}

void test_right_reachable_ten_joint_target_converges() {
    const interaction::Pose source = make_pose();
    interaction::Pose goal = source;
    auto goal_angles = interaction::decompose_upper_body(
        goal, interaction::Hand::Right);
    const interaction::UpperBodyAngles delta{
        0.05F, -0.03F, 0.02F, -0.08F, -0.06F,
        0.04F, 0.10F, 0.05F, -0.04F, 0.03F};
    for (size_t joint = 0U; joint < goal_angles.size(); ++joint) {
        goal_angles[joint] += delta[joint];
    }
    interaction::apply_upper_body(
        goal, interaction::Hand::Right, goal_angles);

    interaction::Pose solved = source;
    interaction::PostureIKConfig config{};
    config.source_scale_m_per_radian.fill(0.0F);
    config.temporal_scale_m_per_radian.fill(0.0F);
    config.elbow_pole_scale_m = 0.0F;
    const auto result = interaction::solve_hand_posture_ik(
        solved,
        interaction::Hand::Right,
        hand_transform(goal, interaction::Hand::Right),
        source,
        interaction::decompose_upper_body(
            source, interaction::Hand::Right),
        config);

    assert(result.accepted);
    assert(result.position_error_m <= config.accepted_position_m);
    assert(result.orientation_error_radians <=
           config.accepted_orientation_radians);
    assert_finite_bounded(result.joint_angles, interaction::Hand::Right);
    assert_non_owned_local_channels_equal(
        source, solved, interaction::Hand::Right);
}

void assert_task_priority_reaches(
    interaction::Hand hand,
    const interaction::UpperBodyAngles& delta) {
    const interaction::Pose source = make_pose();
    interaction::Pose goal = source;
    auto goal_angles = interaction::decompose_upper_body(goal, hand);
    for (size_t joint = 0U; joint < goal_angles.size(); ++joint) {
        goal_angles[joint] += delta[joint];
    }
    interaction::apply_upper_body(goal, hand, goal_angles);

    interaction::PostureIKConfig config{};
    config.source_scale_m_per_radian.fill(0.30F);
    config.temporal_scale_m_per_radian.fill(0.30F);
    interaction::Pose quality = source;
    const interaction::UpperBodyAngles seed =
        interaction::decompose_upper_body(source, hand);
    const auto quality_result = interaction::solve_hand_posture_ik(
        quality,
        hand,
        hand_transform(goal, hand),
        source,
        seed,
        config);
    assert(!quality_result.accepted);

    interaction::Pose solved = source;
    const auto result = interaction::solve_hand_posture_ik_task_priority(
        solved,
        hand,
        hand_transform(goal, hand),
        source,
        seed,
        config);
    assert(result.accepted);
    assert(result.position_error_m <= config.accepted_position_m);
    assert(result.orientation_error_radians <=
           config.accepted_orientation_radians);
    assert_finite_bounded(result.joint_angles, hand);
    assert_non_owned_local_channels_equal(source, solved, hand);
    assert(exact(
        source.positions[g1_skeleton::Simulation],
        solved.positions[g1_skeleton::Simulation]));
}

void test_task_priority_reaches_bilaterally_despite_posture_penalties() {
    assert_task_priority_reaches(
        interaction::Hand::Left,
        {0.06F, -0.04F, 0.03F, -0.10F, 0.08F,
         0.05F, 0.12F, -0.06F, 0.05F, -0.04F});
    assert_task_priority_reaches(
        interaction::Hand::Right,
        {0.05F, -0.03F, 0.02F, -0.08F, -0.06F,
         0.04F, 0.10F, 0.05F, -0.04F, 0.03F});
}

void test_task_priority_unreachable_target_is_finite_and_keeps_root() {
    const interaction::Pose source = make_pose();
    interaction::Pose solved = source;
    interaction::Transform target = left_hand_transform(source);
    target.position = target.position + vec3(2.0F, 1.0F, -1.0F);

    const auto result = interaction::solve_hand_posture_ik_task_priority(
        solved,
        interaction::Hand::Left,
        target,
        source,
        interaction::decompose_left_upper_body(source));

    assert(!result.accepted);
    assert_finite_pose(solved);
    assert_finite_bounded(result.joint_angles);
    assert_non_owned_local_channels_equal(source, solved);
    assert(exact(
        source.positions[g1_skeleton::Simulation],
        solved.positions[g1_skeleton::Simulation]));
}

void test_unreachable_target_returns_finite_bounded_best_pose() {
    const interaction::Pose source = make_pose();
    interaction::Pose solved = source;
    interaction::Transform target = left_hand_transform(source);
    target.position = target.position + vec3(2.0F, 1.0F, -1.0F);

    const auto result = interaction::solve_left_hand_posture_ik(
        solved,
        target,
        source,
        interaction::decompose_left_upper_body(source));

    assert(!result.accepted);
    assert(result.reason == interaction::Reason::JointLimit ||
           result.reason == interaction::Reason::CorrectionLimit);
    assert_finite_bounded(result.joint_angles);
    assert_finite_pose(solved);
    assert_non_owned_local_channels_equal(source, solved);
}

void test_best_seen_pose_includes_the_byte_exact_source_candidate() {
    const interaction::Pose source = make_pose();
    auto transported_seed =
        interaction::decompose_left_upper_body(source);
    transported_seed[3] += 1.0F;
    transported_seed[4] += 0.8F;

    interaction::PostureIKConfig config{};
    config.temporal_scale_m_per_radian.fill(0.0F);
    config.maximum_iterations = 0;
    interaction::Pose solved = source;
    const auto result = interaction::solve_left_hand_posture_ik(
        solved,
        left_hand_transform(source),
        source,
        transported_seed,
        config);

    assert(result.accepted);
    assert(result.objective <= 1.0e-10);
    assert(same_pose(source, solved));
    const auto source_angles =
        interaction::decompose_left_upper_body(source);
    for (size_t joint = 0U; joint < source_angles.size(); ++joint) {
        assert(near(result.joint_angles[joint], source_angles[joint]));
    }
}

void test_elbow_pole_geometry_and_degeneracy() {
    const interaction::Pose source = make_pose();
    const interaction::ElbowPole reference =
        interaction::left_elbow_pole(source);
    assert(reference.valid);

    interaction::Pose perturbed = source;
    auto angles = interaction::decompose_left_upper_body(perturbed);
    angles[3] -= 0.08F;
    angles[4] += 0.06F;
    interaction::apply_left_upper_body(perturbed, angles);
    const float error = interaction::transported_elbow_pole_error(
        reference,
        interaction::left_elbow_pole(perturbed));
    assert(std::isfinite(error));
    assert(error < 1.570796327F);

    interaction::Pose degenerate = source;
    for (const size_t bone : {
             static_cast<size_t>(g1_skeleton::LeftShoulderRoll),
             static_cast<size_t>(g1_skeleton::LeftShoulderYaw),
             static_cast<size_t>(g1_skeleton::LeftElbow),
             static_cast<size_t>(g1_skeleton::LeftWristRoll),
             static_cast<size_t>(g1_skeleton::LeftWristPitch),
             static_cast<size_t>(g1_skeleton::LeftWrist)}) {
        degenerate.positions[bone] = vec3();
    }
    const interaction::ElbowPole straight =
        interaction::left_elbow_pole(degenerate);
    assert(!straight.valid);
    assert(interaction::transported_elbow_pole_error(
               reference, straight) == 0.0F);
}

void test_solver_does_not_flip_elbow_across_small_target_sweep() {
    const interaction::Pose source = make_pose();
    const interaction::ElbowPole reference =
        interaction::left_elbow_pole(source);
    const std::array<vec3, 6U> offsets = {
        vec3(-0.03F, 0.0F, 0.0F), vec3(0.03F, 0.0F, 0.0F),
        vec3(0.0F, -0.03F, 0.0F), vec3(0.0F, 0.03F, 0.0F),
        vec3(0.0F, 0.0F, -0.03F), vec3(0.0F, 0.0F, 0.03F),
    };
    for (const vec3 offset : offsets) {
        interaction::Pose solved = source;
        interaction::Transform target = left_hand_transform(source);
        target.position = target.position + offset;
        const auto result = interaction::solve_left_hand_posture_ik(
            solved,
            target,
            source,
            interaction::decompose_left_upper_body(source));
        assert_finite_pose(solved);
        assert_finite_bounded(result.joint_angles);
        const interaction::ElbowPole current =
            interaction::left_elbow_pole(solved);
        assert(!current.valid ||
               interaction::transported_elbow_pole_error(
                   reference, current) < 1.570796327F);
        assert(!result.accepted || result.position_error_m <= 0.001F);
    }
}

void test_right_solver_does_not_flip_elbow_across_small_target_sweep() {
    const interaction::Pose source = make_pose();
    const interaction::ElbowPole reference =
        interaction::hand_elbow_pole(source, interaction::Hand::Right);
    const std::array<vec3, 3U> offsets = {
        vec3(-0.03F, 0.0F, 0.0F),
        vec3(0.0F, 0.03F, 0.0F),
        vec3(0.0F, 0.0F, 0.03F),
    };
    for (const vec3 offset : offsets) {
        interaction::Pose solved = source;
        interaction::Transform target =
            hand_transform(source, interaction::Hand::Right);
        target.position = target.position + offset;
        const auto result = interaction::solve_hand_posture_ik(
            solved,
            interaction::Hand::Right,
            target,
            source,
            interaction::decompose_upper_body(
                source, interaction::Hand::Right));
        assert_finite_pose(solved);
        assert_finite_bounded(
            result.joint_angles, interaction::Hand::Right);
        const interaction::ElbowPole current =
            interaction::hand_elbow_pole(
                solved, interaction::Hand::Right);
        assert(!current.valid ||
               interaction::transported_elbow_pole_error(
                   reference, current) < 1.570796327F);
    }
}

}  // namespace

int main() {
    test_public_contract_and_angle_round_trip();
    test_zero_target_returns_source_byte_exact();
    test_right_zero_target_returns_source_byte_exact();
    test_reachable_ten_joint_target_converges();
    test_right_reachable_ten_joint_target_converges();
    test_task_priority_reaches_bilaterally_despite_posture_penalties();
    test_task_priority_unreachable_target_is_finite_and_keeps_root();
    test_unreachable_target_returns_finite_bounded_best_pose();
    test_best_seen_pose_includes_the_byte_exact_source_candidate();
    test_elbow_pole_geometry_and_degeneracy();
    test_solver_does_not_flip_elbow_across_small_target_sweep();
    test_right_solver_does_not_flip_elbow_across_small_target_sweep();
}
