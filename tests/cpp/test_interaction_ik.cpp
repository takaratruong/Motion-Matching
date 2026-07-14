#include "interaction_ik.h"

#include "g1_arm_joint_metadata.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

constexpr float kTolerance = 2.0e-5F;
constexpr vec3 kLiteralVec3(1.0F, 2.0F, 3.0F);
constexpr quat kLiteralQuat(1.0F, 0.0F, 0.0F, 0.0F);
static_assert(kLiteralVec3.x == 1.0F && kLiteralVec3.z == 3.0F);
static_assert(kLiteralQuat.w == 1.0F && kLiteralQuat.z == 0.0F);

bool near(float left, float right, float tolerance = kTolerance) {
    return std::abs(left - right) <= tolerance;
}

bool exact(vec3 left, vec3 right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool exact(quat left, quat right) {
    return left.w == right.w && left.x == right.x &&
           left.y == right.y && left.z == right.z;
}

float orientation_error(quat current, quat target) {
    const quat delta = quat_abs(quat_mul(
        quat_normalize(target), quat_inv(quat_normalize(current))));
    return length(quat_to_scaled_angle_axis(delta));
}

const std::array<interaction::HingeJoint, 7>& arm_metadata(
    interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? interaction::kLeftArm
        : interaction::kRightArm;
}

size_t hand_bone(interaction::Hand hand) {
    return hand == interaction::Hand::Left
        ? static_cast<size_t>(g1_skeleton::LeftWrist)
        : static_cast<size_t>(g1_skeleton::RightWrist);
}

quat joint_rotation(
    const interaction::HingeJoint& joint,
    float angle) {
    return quat_normalize(quat_mul(
        joint.rest_rotation,
        quat_from_angle_axis(angle, joint.axis)));
}

float joint_angle(
    const interaction::HingeJoint& joint,
    quat local_rotation) {
    const quat delta = quat_abs(quat_normalize(quat_inv_mul(
        joint.rest_rotation, local_rotation)));
    const vec3 vector(delta.x, delta.y, delta.z);
    const float vector_length = length(vector);
    const vec3 scaled_angle_axis = vector_length < 1.0e-8F
        ? 2.0F * vector
        : (2.0F * std::atan2(vector_length, delta.w) / vector_length) *
            vector;
    return dot(scaled_angle_axis, joint.axis);
}

void set_arm_angles(
    interaction::Pose& pose,
    interaction::Hand hand,
    const std::array<float, 7>& angles) {
    const auto& metadata = arm_metadata(hand);
    for (size_t joint = 0; joint < metadata.size(); ++joint) {
        pose.rotations[static_cast<size_t>(metadata[joint].bone)] =
            joint_rotation(metadata[joint], angles[joint]);
    }
}

std::array<float, 7> initial_angles(interaction::Hand hand) {
    if (hand == interaction::Hand::Left) {
        return {0.10F, 0.35F, -0.20F, 0.80F, 0.10F, -0.15F, 0.05F};
    }
    return {0.10F, -0.35F, 0.20F, 0.80F, -0.10F, -0.15F, -0.05F};
}

interaction::Pose make_pose() {
    using namespace interaction;

    Pose pose{};
    pose.positions[g1_skeleton::Simulation] = vec3(0.20F, 0.0F, -0.10F);
    pose.rotations[g1_skeleton::Simulation] =
        quat_from_angle_axis(0.17F, vec3(0.0F, 1.0F, 0.0F));
    pose.positions[g1_skeleton::Hips] = vec3(0.0F, 0.82F, 0.0F);
    pose.positions[g1_skeleton::Spine] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::Spine1] = vec3(0.0F, 0.10F, 0.0F);
    pose.positions[g1_skeleton::Spine2] = vec3(0.0F, 0.10F, 0.0F);

    pose.positions[g1_skeleton::LeftShoulderPitch] =
        vec3(0.0039563F, 0.23778F, -0.10022F);
    pose.positions[g1_skeleton::LeftShoulderRoll] =
        vec3(0.0F, -0.013831F, -0.038F);
    pose.positions[g1_skeleton::LeftShoulderYaw] =
        vec3(0.0F, -0.1032F, -0.00624F);
    pose.positions[g1_skeleton::LeftElbow] =
        vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::LeftWristRoll] =
        vec3(0.10F, -0.01F, -0.00188791F);
    pose.positions[g1_skeleton::LeftWristPitch] =
        vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::LeftWrist] =
        vec3(0.046F, 0.0F, 0.0F);

    pose.positions[g1_skeleton::RightShoulderPitch] =
        vec3(0.0039563F, 0.23778F, 0.10021F);
    pose.positions[g1_skeleton::RightShoulderRoll] =
        vec3(0.0F, -0.013831F, 0.038F);
    pose.positions[g1_skeleton::RightShoulderYaw] =
        vec3(0.0F, -0.1032F, 0.00624F);
    pose.positions[g1_skeleton::RightElbow] =
        vec3(0.015783F, -0.080518F, 0.0F);
    pose.positions[g1_skeleton::RightWristRoll] =
        vec3(0.10F, -0.01F, 0.00188791F);
    pose.positions[g1_skeleton::RightWristPitch] =
        vec3(0.038F, 0.0F, 0.0F);
    pose.positions[g1_skeleton::RightWrist] =
        vec3(0.046F, 0.0F, 0.0F);

    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        const float value = static_cast<float>(bone);
        pose.velocities[bone] =
            vec3(value + 0.1F, value + 0.2F, value + 0.3F);
        pose.angular_velocities[bone] =
            vec3(value + 0.4F, value + 0.5F, value + 0.6F);
    }
    for (size_t dof = 0; dof < pose.hand_dof.size(); ++dof) {
        pose.hand_dof[dof] = static_cast<float>(dof) + 0.25F;
        pose.hand_dof_velocities[dof] = static_cast<float>(dof) + 0.75F;
    }
    pose.foot_contacts = {1U, 0U};
    set_arm_angles(pose, Hand::Left, initial_angles(Hand::Left));
    set_arm_angles(pose, Hand::Right, initial_angles(Hand::Right));
    return pose;
}

interaction::Transform hand_transform(
    const interaction::Pose& pose,
    interaction::Hand hand) {
    const interaction::WorldPose world = interaction::world_pose(pose);
    const size_t bone = hand_bone(hand);
    return {world.positions[bone], world.rotations[bone]};
}

bool same_pose(
    const interaction::Pose& left,
    const interaction::Pose& right) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        if (!exact(left.positions[bone], right.positions[bone]) ||
            !exact(left.velocities[bone], right.velocities[bone]) ||
            !exact(left.rotations[bone], right.rotations[bone]) ||
            !exact(left.angular_velocities[bone],
                   right.angular_velocities[bone])) {
            return false;
        }
    }
    return left.hand_dof == right.hand_dof &&
           left.hand_dof_velocities == right.hand_dof_velocities &&
           left.foot_contacts == right.foot_contacts;
}

bool is_active_arm_bone(size_t bone, interaction::Hand hand) {
    for (const interaction::HingeJoint& joint : arm_metadata(hand)) {
        if (static_cast<size_t>(joint.bone) == bone) {
            return true;
        }
    }
    return false;
}

void assert_only_active_arm_rotations_changed(
    const interaction::Pose& before,
    const interaction::Pose& after,
    interaction::Hand hand) {
    for (size_t bone = 0; bone < g1_skeleton::BoneCount; ++bone) {
        assert(exact(before.positions[bone], after.positions[bone]));
        assert(exact(before.velocities[bone], after.velocities[bone]));
        assert(exact(
            before.angular_velocities[bone],
            after.angular_velocities[bone]));
        if (!is_active_arm_bone(bone, hand)) {
            assert(exact(before.rotations[bone], after.rotations[bone]));
        }
    }
    assert(before.hand_dof == after.hand_dof);
    assert(before.hand_dof_velocities == after.hand_dof_velocities);
    assert(before.foot_contacts == after.foot_contacts);
}

bool active_arm_changed(
    const interaction::Pose& before,
    const interaction::Pose& after,
    interaction::Hand hand) {
    for (const interaction::HingeJoint& joint : arm_metadata(hand)) {
        const size_t bone = static_cast<size_t>(joint.bone);
        if (!exact(before.rotations[bone], after.rotations[bone])) {
            return true;
        }
    }
    return false;
}

void assert_result_matches_pose(
    const interaction::Pose& pose,
    interaction::Hand hand,
    const interaction::Transform& target,
    const interaction::IKResult& result) {
    const auto& metadata = arm_metadata(hand);
    for (size_t joint = 0; joint < metadata.size(); ++joint) {
        const float angle = result.joint_angles[joint];
        assert(std::isfinite(angle));
        assert(angle >= metadata[joint].lower - kTolerance);
        assert(angle <= metadata[joint].upper + kTolerance);
        assert(orientation_error(
            pose.rotations[static_cast<size_t>(metadata[joint].bone)],
            joint_rotation(metadata[joint], angle)) <= kTolerance);
        assert(near(
            joint_angle(
                metadata[joint],
                pose.rotations[static_cast<size_t>(metadata[joint].bone)]),
            angle));
    }
    const interaction::Transform actual = hand_transform(pose, hand);
    assert(near(
        result.position_error_m,
        length(target.position - actual.position),
        5.0e-5F));
    assert(near(
        result.orientation_error_radians,
        orientation_error(actual.rotation, target.rotation),
        5.0e-5F));
}

void test_public_contract_and_frozen_defaults() {
    using namespace interaction;

    using SolveSignature = IKResult (*)(
        Pose&, Hand, Transform, const IKConfig&);
    SolveSignature solve = &solve_hand_ik;
    (void)solve;
    static_assert(std::is_same_v<decltype(IKResult{}.accepted), bool>);
    static_assert(std::is_same_v<decltype(IKResult{}.reason), Reason>);
    static_assert(std::is_same_v<
        decltype(IKResult{}.joint_angles), std::array<float, 7>>);

    const IKConfig config{};
    assert(config.maximum_request_position_m == 0.12F);
    assert(config.maximum_request_orientation_radians == 0.436332313F);
    assert(config.accepted_position_m == 0.04F);
    assert(config.accepted_orientation_radians == 0.261799388F);
    assert(config.damping == 0.05F);
    assert(config.finite_difference_radians == 0.001F);
    assert(config.orientation_scale_m_per_radian == 0.25F);
    assert(config.maximum_step_radians == 0.10F);
    assert(config.maximum_iterations == 8);
    static_assert(kLeftArm.size() == 7U && kRightArm.size() == 7U);
}

void test_local_rotation_decomposition_round_trips_both_arms() {
    using namespace interaction;

    for (const Hand hand : {Hand::Left, Hand::Right}) {
        Pose pose = make_pose();
        std::array<float, 7> angles = initial_angles(hand);
        angles[0] -= 0.71F;
        angles[1] += 0.24F;
        angles[2] -= 0.31F;
        angles[3] += 0.17F;
        angles[4] -= 0.22F;
        angles[5] += 0.19F;
        angles[6] -= 0.28F;
        set_arm_angles(pose, hand, angles);
        const Pose before = pose;
        const Transform target = hand_transform(pose, hand);

        const IKResult result = solve_hand_ik(pose, hand, target, IKConfig{});

        assert(result.accepted);
        assert(result.reason == Reason::None);
        assert(result.position_error_m <= kTolerance);
        assert(result.orientation_error_radians <= kTolerance);
        for (size_t joint = 0; joint < angles.size(); ++joint) {
            assert(near(result.joint_angles[joint], angles[joint]));
        }
        assert(same_pose(pose, before));
        assert_result_matches_pose(pose, hand, target, result);
    }
}

void test_reachable_targets_converge_for_both_hands() {
    using namespace interaction;

    const std::array<std::array<float, 7>, 2> deltas = {{
        {{-0.22F, -0.12F, 0.10F, 0.20F, -0.08F, 0.06F, -0.10F}},
        {{0.12F, 0.08F, -0.08F, 0.16F, 0.06F, -0.04F, 0.08F}},
    }};
    const std::array<Hand, 2> hands = {Hand::Left, Hand::Right};
    const IKConfig config{};

    for (size_t index = 0; index < hands.size(); ++index) {
        const Hand hand = hands[index];
        Pose pose = make_pose();
        Pose goal = pose;
        std::array<float, 7> goal_angles = initial_angles(hand);
        for (size_t joint = 0; joint < goal_angles.size(); ++joint) {
            goal_angles[joint] += deltas[index][joint];
        }
        set_arm_angles(goal, hand, goal_angles);
        const Transform target = hand_transform(goal, hand);
        const Transform initial = hand_transform(pose, hand);
        const float initial_position =
            length(target.position - initial.position);
        const float initial_orientation =
            orientation_error(initial.rotation, target.rotation);
        assert(initial_position <= config.maximum_request_position_m);
        assert(initial_orientation <=
               config.maximum_request_orientation_radians);
        assert(initial_position > config.accepted_position_m ||
               initial_orientation > config.accepted_orientation_radians);
        const Pose before = pose;

        const IKResult result = solve_hand_ik(pose, hand, target, config);

        assert(result.accepted);
        assert(result.reason == Reason::None);
        assert(result.position_error_m <= config.accepted_position_m);
        assert(result.orientation_error_radians <=
               config.accepted_orientation_radians);
        assert(result.position_error_m < initial_position ||
               result.orientation_error_radians < initial_orientation);
        assert_only_active_arm_rotations_changed(before, pose, hand);
        assert(active_arm_changed(before, pose, hand));
        assert_result_matches_pose(pose, hand, target, result);
    }
}

void test_orientation_correction_and_shortest_quaternion_residual() {
    using namespace interaction;

    Pose pose = make_pose();
    Pose goal = pose;
    std::array<float, 7> angles = initial_angles(Hand::Right);
    angles[6] += 0.30F;
    set_arm_angles(goal, Hand::Right, angles);
    const Transform target = hand_transform(goal, Hand::Right);
    const Transform initial = hand_transform(pose, Hand::Right);
    assert(length(target.position - initial.position) <= kTolerance);
    assert(near(orientation_error(initial.rotation, target.rotation), 0.30F));
    const Pose before = pose;

    const IKResult result = solve_hand_ik(
        pose, Hand::Right, target, IKConfig{});

    assert(result.accepted);
    assert(result.reason == Reason::None);
    assert(result.orientation_error_radians < 0.30F);
    assert_only_active_arm_rotations_changed(before, pose, Hand::Right);
    assert_result_matches_pose(pose, Hand::Right, target, result);

    Pose sign_equivalent = make_pose();
    const Pose sign_before = sign_equivalent;
    Transform same_target = hand_transform(sign_equivalent, Hand::Left);
    same_target.rotation = -same_target.rotation;
    const IKResult same = solve_hand_ik(
        sign_equivalent, Hand::Left, same_target, IKConfig{});
    assert(same.accepted);
    assert(same.reason == Reason::None);
    assert(same.orientation_error_radians <= kTolerance);
    assert(same_pose(sign_equivalent, sign_before));
}

void test_requests_above_the_envelope_reject_without_mutation() {
    using namespace interaction;

    const IKConfig config{};
    Pose position_pose = make_pose();
    const Pose position_before = position_pose;
    Transform position_target = hand_transform(position_pose, Hand::Right);
    position_target.position.x += config.maximum_request_position_m + 0.001F;

    const IKResult position_result = solve_hand_ik(
        position_pose, Hand::Right, position_target, config);

    assert(!position_result.accepted);
    assert(position_result.reason == Reason::CorrectionLimit);
    assert(position_result.position_error_m >
           config.maximum_request_position_m);
    assert(same_pose(position_pose, position_before));

    Pose orientation_pose = make_pose();
    const Pose orientation_before = orientation_pose;
    Transform orientation_target = hand_transform(
        orientation_pose, Hand::Left);
    const quat excess = quat_from_angle_axis(
        config.maximum_request_orientation_radians + 0.001F,
        vec3(0.0F, 1.0F, 0.0F));
    orientation_target.rotation = quat_mul(
        excess, orientation_target.rotation);

    const IKResult orientation_result = solve_hand_ik(
        orientation_pose, Hand::Left, orientation_target, config);

    assert(!orientation_result.accepted);
    assert(orientation_result.reason == Reason::CorrectionLimit);
    assert(orientation_result.orientation_error_radians >
           config.maximum_request_orientation_radians);
    assert(same_pose(orientation_pose, orientation_before));
}

void test_bounded_failure_writes_the_best_pose_and_is_deterministic() {
    using namespace interaction;

    Pose initial_pose = make_pose();
    Pose goal = initial_pose;
    std::array<float, 7> angles = initial_angles(Hand::Right);
    angles[6] += 0.20F;
    set_arm_angles(goal, Hand::Right, angles);
    const Transform target = hand_transform(goal, Hand::Right);
    const Transform initial_hand = hand_transform(initial_pose, Hand::Right);
    const float initial_score =
        length(target.position - initial_hand.position) +
        0.25F * orientation_error(initial_hand.rotation, target.rotation);
    IKConfig config{};
    config.maximum_iterations = 1;
    config.accepted_position_m = 1.0e-7F;
    config.accepted_orientation_radians = 1.0e-7F;

    Pose first_pose = initial_pose;
    const IKResult first = solve_hand_ik(
        first_pose, Hand::Right, target, config);

    assert(!first.accepted);
    assert(first.reason == Reason::CorrectionLimit);
    assert(active_arm_changed(initial_pose, first_pose, Hand::Right));
    assert_only_active_arm_rotations_changed(
        initial_pose, first_pose, Hand::Right);
    assert(first.position_error_m +
               config.orientation_scale_m_per_radian *
                   first.orientation_error_radians <
           initial_score);
    assert_result_matches_pose(first_pose, Hand::Right, target, first);

    Pose second_pose = initial_pose;
    const IKResult second = solve_hand_ik(
        second_pose, Hand::Right, target, config);
    assert(first.accepted == second.accepted);
    assert(first.reason == second.reason);
    assert(first.position_error_m == second.position_error_m);
    assert(first.orientation_error_radians ==
           second.orientation_error_radians);
    assert(first.joint_angles == second.joint_angles);
    assert(same_pose(first_pose, second_pose));
}

void test_joint_limited_failure_is_bounded_and_reports_joint_limit() {
    using namespace interaction;

    Pose pose = make_pose();
    std::array<float, 7> angles = initial_angles(Hand::Right);
    angles[6] = kRightArm[6].upper;
    set_arm_angles(pose, Hand::Right, angles);
    Pose goal = pose;
    angles[6] += 0.20F;
    set_arm_angles(goal, Hand::Right, angles);
    const Transform target = hand_transform(goal, Hand::Right);
    const Transform initial = hand_transform(pose, Hand::Right);
    assert(length(target.position - initial.position) <= kTolerance);
    assert(near(orientation_error(initial.rotation, target.rotation), 0.20F));
    const Pose before = pose;
    IKConfig config{};
    config.maximum_iterations = 1;
    config.accepted_position_m = 1.0e-7F;
    config.accepted_orientation_radians = 1.0e-7F;

    const IKResult result = solve_hand_ik(
        pose, Hand::Right, target, config);

    assert(!result.accepted);
    assert(result.reason == Reason::JointLimit);
    assert_only_active_arm_rotations_changed(before, pose, Hand::Right);
    assert_result_matches_pose(pose, Hand::Right, target, result);
    assert(result.joint_angles[6] <= kRightArm[6].upper);
}

}  // namespace

int main() {
    test_public_contract_and_frozen_defaults();
    test_local_rotation_decomposition_round_trips_both_arms();
    test_reachable_targets_converge_for_both_hands();
    test_orientation_correction_and_shortest_quaternion_residual();
    test_requests_above_the_envelope_reject_without_mutation();
    test_bounded_failure_writes_the_best_pose_and_is_deterministic();
    test_joint_limited_failure_is_bounded_and_reports_joint_limit();
}
